#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTT-lite inference: per-test-sample inner update then predict then reset."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ggmv_core import GGMVParams, forward_ggmv
from ggmv_graph import load_ggmv_graph
from ggmv_llm import LLMMechanismScorer
from ggmv_support import build_query_from_record, build_support_pool_from_labels, select_support_set_for_query
from ttt_common import format_prediction_block, load_label_map, load_prompts, render_chat_prompt

# kernprof injects `profile` into builtins; provide a no-op fallback for normal runs.
try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):  # type: ignore[no-redef]
        return func


def _trainable_named_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _snapshot_params(named_params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in named_params}


def _restore_params(named_params: List[Tuple[str, torch.nn.Parameter]], snap: Dict[str, torch.Tensor]) -> None:
    for n, p in named_params:
        p.data.copy_(snap[n].data)


def _build_model_and_tokenizer(args):
    print(f"[TTT-lite] Loading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[TTT-lite] Loading base model from: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    if args.init_adapter:
        print(f"[TTT-lite] Loading initial adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        print("[TTT-lite] Initializing fresh LoRA adapter")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        )
        model = get_peft_model(model, lora_cfg)

    model.train()
    return model, tokenizer


def _is_fatal_cuda_error(msg: str) -> bool:
    s = msg.lower()
    fatal_patterns = (
        "unspecified launch failure",
        "device-side assert",
        "cuda error",
        "cublas",
        "cudnn",
        "nccl",
    )
    return any(p in s for p in fatal_patterns)


def _score_label_candidates(
    model,
    device: torch.device,
    prompt_ids: List[int],
    prefix_ids: List[int],
    candidate_id_lists: List[List[int]],
) -> List[float]:
    """Score each candidate by summed log-probability conditioned on prompt+prefix."""
    scores: List[float] = []
    base_ids = prompt_ids + prefix_ids
    for cand_ids in candidate_id_lists:
        if not cand_ids:
            scores.append(float("-inf"))
            continue
        full_ids = base_ids + cand_ids
        input_t = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_t, attention_mask=torch.ones_like(input_t)).logits
            log_probs = F.log_softmax(logits, dim=-1)
        # Token at position t is predicted from logits at t-1.
        start = len(base_ids)
        s = 0.0
        for j, tok in enumerate(cand_ids):
            pos = start + j - 1
            s += float(log_probs[0, pos, tok].detach().cpu())
        scores.append(s)
    return scores


def _load_ggmv_params(args) -> GGMVParams:
    params = GGMVParams(
        top_k_modules=args.ggmv_top_k_modules,
        posterior_mode=args.ggmv_posterior_mode,
        w0=args.ggmv_w0,
        w1=args.ggmv_w1,
        w2=args.ggmv_w2,
        beta=args.ggmv_beta,
        gamma_mech=args.ggmv_gamma_mech,
        lambda_v=args.ggmv_lambda_v,
        gamma_a=args.ggmv_gamma_a,
        eta1=args.ggmv_eta1,
        eta2=args.ggmv_eta2,
        eta3=args.ggmv_eta3,
        eta4=args.ggmv_eta4,
        b0=args.ggmv_b0,
        b1=args.ggmv_b1,
        no_graph_prior=args.ggmv_no_graph_prior,
        no_support_utility=args.ggmv_no_support_utility,
        no_llm_mech=args.ggmv_no_llm_mech,
        no_verification=args.ggmv_no_verification,
        no_sufficiency=args.ggmv_no_sufficiency,
    )
    if not args.ggmv_params_json:
        return params

    with open(args.ggmv_params_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    for key, value in payload.items():
        if hasattr(params, key):
            setattr(params, key, value)
    return params


def _extract_base_score_map(label_candidates: List[str], scores: List[float]) -> Dict[str, float]:
    out = {k.strip().lower(): float(v) for k, v in zip(label_candidates, scores)}
    out.setdefault("yes", float("-inf"))
    out.setdefault("no", float("-inf"))
    out.setdefault("insufficient", float("-inf"))
    return out


def _format_debug_modules(module_pairs: List[Tuple[int, float]], graph, top_n: int = 6) -> List[dict]:
    out = []
    for module_id, score in module_pairs[:top_n]:
        if module_id >= 0 and module_id < len(graph.idx_to_module):
            name = graph.idx_to_module[module_id]
        else:
            name = f"module_{module_id}"
        out.append({"module_id": int(module_id), "module_name": name, "score": float(score)})
    return out


@profile
def run(args) -> None:
    print("[TTT-lite] ===== Run Config =====")
    print(
        f"[TTT-lite] model={args.model_name} | prompts={args.prompts_file} | labels={args.labels_csv} | out={args.out}"
    )
    print(
        f"[TTT-lite] inner_steps={args.inner_steps} inner_lr={args.inner_lr} inner_max_tokens={args.inner_max_tokens} "
        f"max_new_tokens={args.max_new_tokens} temperature={args.temperature} top_p={args.top_p} "
        f"prediction_mode={args.prediction_mode}"
    )
    model, tokenizer = _build_model_and_tokenizer(args)
    print("[TTT-lite] Loading prompts and labels")
    labels = load_label_map(args.labels_csv)
    records = load_prompts(args.prompts_file)
    print(f"[TTT-lite] Loaded prompt records: {len(records)} | label rows: {len(labels)}")

    test_items = []
    for rec in records:
        if not rec.pert or not rec.gene or not rec.system_prompt or not rec.user_input:
            continue
        row = labels.get((rec.pert, rec.gene))
        if row is None:
            continue
        if row.get("split", "").strip().lower() == "test":
            test_items.append((rec, row))

    if not test_items:
        raise RuntimeError("No test records found from prompts + labels split.")
    print(f"[TTT-lite] Test records to run: {len(test_items)}")

    named_params = _trainable_named_params(model)
    print(f"[TTT-lite] Trainable params for inner update: {len(named_params)} tensors")
    optimizer = torch.optim.AdamW([p for _, p in named_params], lr=args.inner_lr)

    out_blocks: List[str] = []
    failed_rows: List[str] = []
    fatal_cuda_encountered = False
    device = next(model.parameters()).device
    print(f"[TTT-lite] Running on device: {device}")
    label_candidates = [x.strip() for x in args.label_candidates.split(",") if x.strip()]
    label_prefix_ids = tokenizer(args.label_prefix, add_special_tokens=False).input_ids
    candidate_id_lists = [
        tokenizer(" " + cand if not cand.startswith(" ") else cand, add_special_tokens=False).input_ids
        for cand in label_candidates
    ]
    if args.prediction_mode in {"label-ranking", "graph-mechanism-verification"}:
        print(f"[TTT-lite] label-ranking candidates: {label_candidates}")

    ggmv_graph = None
    ggmv_support_pool = None
    ggmv_params = None
    ggmv_mech_scorer = None
    if args.prediction_mode == "graph-mechanism-verification":
        if not args.kg_dir:
            raise RuntimeError("--kg-dir is required when --prediction-mode graph-mechanism-verification")
        required_labels = {"yes", "no", "insufficient"}
        if not required_labels.issubset({x.strip().lower() for x in label_candidates}):
            raise RuntimeError(
                "graph-mechanism-verification requires --label-candidates containing yes,no,insufficient"
            )

        print(f"[TTT-lite][GGMV] Loading KG from {args.kg_dir}")
        ggmv_graph = load_ggmv_graph(
            kg_dir=args.kg_dir,
            alpha=args.ggmv_alpha,
            nodes_file=args.kg_nodes_file,
            edges_file=args.kg_edges_file,
            graph_file=args.kg_graph_file,
            strict=False,
        )
        ggmv_params = _load_ggmv_params(args)
        print(f"[TTT-lite][GGMV] Params: {ggmv_params}")

        ggmv_support_pool = build_support_pool_from_labels(
            labels_csv=args.labels_csv,
            graph=ggmv_graph,
            split=args.ggmv_support_split,
            default_cell_id=args.ggmv_default_cell_id,
            max_genes_per_perturbation=args.ggmv_max_genes_per_support,
        )
        print(f"[TTT-lite][GGMV] Support pool perturbations: {len(ggmv_support_pool)}")

        if args.ggmv_use_llm_mech and not args.ggmv_no_llm_mech:
            ggmv_mech_scorer = LLMMechanismScorer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_tokens=args.ggmv_llm_max_tokens,
                module_top_genes=args.ggmv_module_top_genes,
            )
            print("[TTT-lite][GGMV] LLM mechanism scorer: enabled")
        else:
            print("[TTT-lite][GGMV] LLM mechanism scorer: disabled")

    for i, (rec, label_row) in enumerate(test_items, 1):
        header = f"Prompt {rec.prompt_id}" if rec.prompt_id is not None else f"Prompt_{rec.idx}"
        prompt_text = render_chat_prompt(tokenizer, rec.system_prompt, rec.user_input)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        if args.inner_max_tokens > 0:
            prompt_ids = prompt_ids[-args.inner_max_tokens:]

        # Inner adaptation + generation on current test prompt.
        snap = _snapshot_params(named_params)
        inner_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        loss = None
        try:
            for _ in range(args.inner_steps):
                out = model(input_ids=inner_input, attention_mask=torch.ones_like(inner_input), labels=inner_input)
                loss = out.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            model.eval()
            if args.prediction_mode in {"label-ranking", "graph-mechanism-verification"}:
                scores = _score_label_candidates(
                    model=model,
                    device=device,
                    prompt_ids=prompt_ids,
                    prefix_ids=label_prefix_ids,
                    candidate_id_lists=candidate_id_lists,
                )
                if args.prediction_mode == "label-ranking":
                    best_idx = max(range(len(scores)), key=lambda k: scores[k])
                    best_label = label_candidates[best_idx]
                    response = (
                        "Final Deterministic Prediction:\n"
                        f"{best_label}\n"
                        f"[label_ranking_scores] {dict(zip(label_candidates, [round(s, 4) for s in scores]))}"
                    )
                else:
                    if ggmv_graph is None or ggmv_support_pool is None or ggmv_params is None:
                        raise RuntimeError("GGMV mode not initialized correctly.")

                    base_score_map = _extract_base_score_map(label_candidates, scores)
                    query = build_query_from_record(
                        rec=rec,
                        label_row=label_row,
                        graph=ggmv_graph,
                        base_scores=base_score_map,
                        default_cell_id=args.ggmv_default_cell_id,
                    )
                    support_set = select_support_set_for_query(
                        query=query,
                        support_pool=ggmv_support_pool,
                        graph=ggmv_graph,
                        max_support_perturbations=args.ggmv_max_support_perturbations,
                        include_same_drug=args.ggmv_include_same_drug,
                    )

                    if ggmv_mech_scorer is not None:
                        llm_mech_fn = (
                            lambda module_ids: ggmv_mech_scorer.score_candidate_modules(
                                prompt_ids=prompt_ids,
                                candidate_module_ids=module_ids,
                                graph=ggmv_graph,
                            )
                        )
                        llm_suff = float(ggmv_mech_scorer.score_sufficiency(prompt_ids=prompt_ids))
                    else:
                        llm_mech_fn = None
                        llm_suff = 0.0

                    out_ggmv = forward_ggmv(
                        query=query,
                        support_set=support_set,
                        graph=ggmv_graph,
                        params=ggmv_params,
                        llm_mech_score_fn=llm_mech_fn,
                        llm_sufficiency_score=llm_suff,
                    )
                    calibrated = out_ggmv["scores"]
                    final_label = max(["yes", "no", "insufficient"], key=lambda x: float(calibrated[x]))
                    response_lines = [
                        "Final Deterministic Prediction:",
                        final_label,
                        f"[label_ranking_scores] {{'yes': {round(float(calibrated['yes']), 4)}, 'no': {round(float(calibrated['no']), 4)}, 'insufficient': {round(float(calibrated['insufficient']), 4)}}}",  # noqa: E501
                    ]
                    if args.ggmv_debug:
                        debug_payload = {
                            "query": {
                                "query_id": query.get("query_id"),
                                "drug_name": query.get("drug_name"),
                                "gene_name": query.get("gene_name"),
                                "drug_id": query.get("drug_id"),
                                "gene_id": query.get("gene_id"),
                                "support_size": len(support_set),
                            },
                            "candidate_modules_top": _format_debug_modules(
                                out_ggmv["debug"]["top_prior_modules"], ggmv_graph, top_n=args.ggmv_debug_top_n
                            ),
                            "graph_prior_top": _format_debug_modules(
                                out_ggmv["debug"]["top_prior_modules"], ggmv_graph, top_n=args.ggmv_debug_top_n
                            ),
                            "support_evidence_top": _format_debug_modules(
                                out_ggmv["debug"]["top_support_evidence_modules"], ggmv_graph, top_n=args.ggmv_debug_top_n
                            ),
                            "llm_mech_top": _format_debug_modules(
                                out_ggmv["debug"]["top_llm_mech_modules"], ggmv_graph, top_n=args.ggmv_debug_top_n
                            ),
                            "posterior_top": _format_debug_modules(
                                out_ggmv["debug"]["top_posterior_modules"], ggmv_graph, top_n=args.ggmv_debug_top_n
                            ),
                            "verification_score": round(float(out_ggmv["verification_score"]), 6),
                            "sufficiency_score": round(float(out_ggmv["sufficiency_score"]), 6),
                            "base_scores": {
                                "yes": round(float(base_score_map.get("yes", 0.0)), 6),
                                "no": round(float(base_score_map.get("no", 0.0)), 6),
                                "insufficient": round(float(base_score_map.get("insufficient", 0.0)), 6),
                            },
                            "calibrated_scores": {
                                "yes": round(float(calibrated["yes"]), 6),
                                "no": round(float(calibrated["no"]), 6),
                                "insufficient": round(float(calibrated["insufficient"]), 6),
                            },
                            "final_prediction": final_label,
                        }
                        response_lines.append(f"[ggmv_debug] {json.dumps(debug_payload, ensure_ascii=True)}")
                    response = "\n".join(response_lines)
            else:
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=inner_input,
                        attention_mask=torch.ones_like(inner_input),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.temperature > 0,
                        temperature=max(args.temperature, 1e-5),
                        top_p=args.top_p,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = gen[:, inner_input.shape[1]:]
                response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            out_blocks.append(format_prediction_block(header, response))
        except Exception as e:  # Keep long runs alive even if one sample fails on CUDA/runtime.
            msg = str(e).replace("\t", " ").replace("\n", " ")
            failed_rows.append(f"{i}\t{header}\t{rec.idx}\t{rec.prompt_id}\t{msg}\n")
            print(f"[TTT-lite][WARN] failed {i}/{len(test_items)} ({header}): {msg}")
            fatal_cuda_encountered = _is_fatal_cuda_error(msg)
            if torch.cuda.is_available() and not fatal_cuda_encountered:
                try:
                    torch.cuda.empty_cache()
                except Exception as cache_e:
                    cache_msg = str(cache_e).replace("\t", " ").replace("\n", " ")
                    print(f"[TTT-lite][WARN] empty_cache failed: {cache_msg}")
                    fatal_cuda_encountered = True
        finally:
            model.train()
            # Reset to base params for next sample.
            if not fatal_cuda_encountered:
                try:
                    _restore_params(named_params, snap)
                    optimizer.zero_grad(set_to_none=True)
                except Exception as restore_e:
                    restore_msg = str(restore_e).replace("\t", " ").replace("\n", " ")
                    failed_rows.append(
                        f"{i}\t{header}\t{rec.idx}\t{rec.prompt_id}\tRESTORE_FAILED: {restore_msg}\n"
                    )
                    print(f"[TTT-lite][WARN] restore failed at {i}/{len(test_items)} ({header}): {restore_msg}")
                    fatal_cuda_encountered = True

        if fatal_cuda_encountered:
            print("[TTT-lite][ERROR] Fatal CUDA state detected; stopping this run early. Please relaunch to continue.")
            break

        if i % 10 == 0 or i == len(test_items):
            loss_val = float(loss.detach().cpu()) if loss is not None else float("nan")
            print(f"[TTT-lite] done {i}/{len(test_items)} | last_inner_loss={loss_val:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(out_blocks)
    print(f"[TTT-lite] Saved predictions: {args.out}")
    if failed_rows:
        failed_out = args.failed_out or f"{os.path.splitext(args.out)[0]}.failed.tsv"
        with open(failed_out, "w", encoding="utf-8") as f:
            f.write("sample_idx\theader\trecord_idx\tprompt_id\terror\n")
            f.writelines(failed_rows)
        print(f"[TTT-lite] Failed samples: {len(failed_rows)} | details: {failed_out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run TTT-lite inference on test split prompts.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--failed-out", default=None, help="Optional TSV for failed sample errors.")
    p.add_argument("--init-adapter", default=None, help="Optional LoRA adapter checkpoint as TTT init")
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--inner-lr", type=float, default=5e-5)
    p.add_argument("--inner-max-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--prediction-mode",
        choices=["generate", "label-ranking", "graph-mechanism-verification"],
        default="generate",
        help=(
            "`generate` uses HF generation; `label-ranking` scores labels directly; "
            "`graph-mechanism-verification` applies GGMV calibration over base label ranking."
        ),
    )
    p.add_argument(
        "--label-candidates",
        default="yes,no,insufficient",
        help="Comma-separated candidate labels used in label-ranking mode.",
    )
    p.add_argument(
        "--label-prefix",
        default="\nFinal Deterministic Prediction:\n",
        help="Prefix text before label candidates in label-ranking mode.",
    )
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--line-profile", action="store_true", help="Enable line_profiler for run().")
    p.add_argument("--line-profile-out", default=None, help="Optional output file for line_profiler stats.")

    # GGMV options (active only when --prediction-mode graph-mechanism-verification).
    p.add_argument("--kg-dir", default=None, help="KG directory containing nodes/edges/graph JSON files.")
    p.add_argument("--kg-nodes-file", default="nodes.json")
    p.add_argument("--kg-edges-file", default="edges.json")
    p.add_argument("--kg-graph-file", default="graph.json")
    p.add_argument("--ggmv-params-json", default=None, help="Optional JSON file to override GGMV scalar params.")
    p.add_argument("--ggmv-alpha", type=float, default=0.1, help="Alpha for R_tilde graph diffusion.")
    p.add_argument("--ggmv-top-k-modules", type=int, default=64)
    p.add_argument("--ggmv-posterior-mode", choices=["sparsemax", "softmax"], default="sparsemax")
    p.add_argument("--ggmv-support-split", default="train", help="Which split to use as support pool from labels CSV.")
    p.add_argument("--ggmv-default-cell-id", type=int, default=0)
    p.add_argument("--ggmv-max-support-perturbations", type=int, default=32)
    p.add_argument("--ggmv-max-genes-per-support", type=int, default=256)
    p.add_argument("--ggmv-include-same-drug", action="store_true", help="Allow same-drug supports in support set.")
    p.add_argument("--ggmv-use-llm-mech", action="store_true", help="Enable hidden-state LLM mechanism scorer.")
    p.add_argument("--ggmv-llm-max-tokens", type=int, default=1024)
    p.add_argument("--ggmv-module-top-genes", type=int, default=8)
    p.add_argument("--ggmv-debug", action="store_true", help="Append GGMV debug JSON payload into each output block.")
    p.add_argument("--ggmv-debug-top-n", type=int, default=6)

    # GGMV scalar defaults (kept explicit for ablation friendliness).
    p.add_argument("--ggmv-w0", type=float, default=0.0)
    p.add_argument("--ggmv-w1", type=float, default=1.0)
    p.add_argument("--ggmv-w2", type=float, default=1.0)
    p.add_argument("--ggmv-beta", type=float, default=1.0)
    p.add_argument("--ggmv-gamma-mech", type=float, default=1.0)
    p.add_argument("--ggmv-lambda-v", type=float, default=1.0)
    p.add_argument("--ggmv-gamma-a", type=float, default=0.1)
    p.add_argument("--ggmv-eta1", type=float, default=1.0)
    p.add_argument("--ggmv-eta2", type=float, default=1.0)
    p.add_argument("--ggmv-eta3", type=float, default=1.0)
    p.add_argument("--ggmv-eta4", type=float, default=0.0)
    p.add_argument("--ggmv-b0", type=float, default=0.0)
    p.add_argument("--ggmv-b1", type=float, default=1.0)

    # Ablation switches.
    p.add_argument("--ggmv-no-graph-prior", action="store_true")
    p.add_argument("--ggmv-no-support-utility", action="store_true")
    p.add_argument("--ggmv-no-llm-mech", action="store_true")
    p.add_argument("--ggmv-no-verification", action="store_true")
    p.add_argument("--ggmv-no-sufficiency", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.line_profile:
        run(args)
        return 0

    try:
        from line_profiler import LineProfiler
    except ImportError as e:
        raise RuntimeError(
            "line_profiler is not installed. Install with `pip install line_profiler` "
            "or run without --line-profile."
        ) from e

    print("[TTT-lite] line_profiler enabled")
    lp = LineProfiler()
    profiled_run = lp(run)
    profiled_run(args)
    lp.print_stats()
    if args.line_profile_out:
        with open(args.line_profile_out, "w", encoding="utf-8") as f:
            lp.print_stats(stream=f)
        print(f"[TTT-lite] Saved line_profiler stats to: {args.line_profile_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
