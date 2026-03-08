#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query-only inference for support-supervised latent mechanism distillation (MSLD)."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from msld_model import MSLDHeadConfig, MSLDHeads, compute_sufficiency_score, distribution_entropy
from ttt_common import format_prediction_block, load_label_map, load_prompts, render_chat_prompt


def _build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def _score_label_candidates(
    model,
    device: torch.device,
    prompt_ids: List[int],
    prefix_ids: List[int],
    candidate_id_lists: List[List[int]],
) -> List[float]:
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
        start = len(base_ids)
        s = 0.0
        for j, tok in enumerate(cand_ids):
            pos = start + j - 1
            s += float(log_probs[0, pos, tok].detach().cpu())
        scores.append(s)
    return scores


def _extract_base_scores(label_candidates: List[str], scores: List[float]) -> Dict[str, float]:
    out = {k.strip().lower(): float(v) for k, v in zip(label_candidates, scores)}
    out.setdefault("yes", float("-inf"))
    out.setdefault("no", float("-inf"))
    out.setdefault("insufficient", float("-inf"))
    return out


def _encode_hidden(model, device: torch.device, prompt_ids: List[int]) -> np.ndarray:
    input_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(
            input_ids=input_t,
            attention_mask=torch.ones_like(input_t),
            output_hidden_states=True,
            return_dict=True,
        )
    if out.hidden_states is None or len(out.hidden_states) == 0:
        raise RuntimeError("Model did not return hidden states")
    return out.hidden_states[-1][0].mean(dim=0).float().detach().cpu().numpy().astype(np.float32)


def _build_test_records(prompts_file: str, labels_csv: str, split: str) -> List[tuple]:
    split_norm = split.strip().lower()
    labels = load_label_map(labels_csv)
    records = load_prompts(prompts_file)
    out = []
    for rec in records:
        if not rec.pert or not rec.gene or not rec.system_prompt or not rec.user_input:
            continue
        row = labels.get((rec.pert, rec.gene))
        if row is None:
            continue
        if str(row.get("split", "")).strip().lower() != split_norm:
            continue
        out.append((rec, row))
    return out


def run(args) -> None:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if ckpt.get("version") not in {"msld_v1", "msld_v2_consistency_refine"}:
        raise RuntimeError("Unsupported checkpoint version")

    model, tokenizer = _build_model_and_tokenizer(args)
    model_device = next(model.parameters()).device
    head_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_raw = ckpt["head_config"]
    cfg = MSLDHeadConfig(
        hidden_dim=int(cfg_raw["hidden_dim"]),
        num_modules=int(cfg_raw["num_modules"]),
        module_emb_dim=int(cfg_raw["module_emb_dim"]),
        label_hidden_dim=int(cfg_raw["label_hidden_dim"]),
        dropout=float(cfg_raw["dropout"]),
        refine_hidden_dim=int(cfg_raw.get("refine_hidden_dim", 64)),
        type_emb_dim=int(cfg_raw.get("type_emb_dim", 16)),
        max_cell_types=int(cfg_raw.get("max_cell_types", 256)),
        max_drug_types=int(cfg_raw.get("max_drug_types", 8192)),
    )
    heads = MSLDHeads(cfg).to(head_device)
    load_info = heads.load_state_dict(ckpt["heads_state_dict"], strict=False)
    if load_info.missing_keys or load_info.unexpected_keys:
        print(
            f"[MSLD] load_state_dict non-strict: "
            f"missing={len(load_info.missing_keys)} unexpected={len(load_info.unexpected_keys)}"
        )
    heads.eval()

    label_candidates = (
        [x.strip() for x in args.label_candidates.split(",") if x.strip()]
        if args.label_candidates
        else list(ckpt.get("label_candidates", ["yes", "no", "insufficient"]))
    )
    if not {"yes", "no", "insufficient"}.issubset({x.lower() for x in label_candidates}):
        raise RuntimeError("label candidates must include yes,no,insufficient")
    label_prefix = args.label_prefix if args.label_prefix else ckpt.get("label_prefix", "\nFinal Deterministic Prediction:\n")
    prefix_ids = tokenizer(label_prefix, add_special_tokens=False).input_ids
    candidate_id_lists = [
        tokenizer(" " + cand if not cand.startswith(" ") else cand, add_special_tokens=False).input_ids
        for cand in label_candidates
    ]

    decision = ckpt.get("decision", {})
    tau = float(args.tau if args.tau is not None else decision.get("tau", 0.0))
    eta1 = float(args.eta1 if args.eta1 is not None else decision.get("eta1", 1.0))
    eta2 = float(args.eta2 if args.eta2 is not None else decision.get("eta2", 1.0))
    eta3 = float(args.eta3 if args.eta3 is not None else decision.get("eta3", 1.0))

    module_names = list(ckpt.get("module_names", []))
    records = _build_test_records(args.prompts_file, args.labels_csv, split=args.split)
    if not records:
        raise RuntimeError("No query records found")
    print(f"[MSLD] Query-only inference records: {len(records)}")

    out_blocks: List[str] = []
    for i, (rec, _) in enumerate(records, 1):
        header = f"Prompt {rec.prompt_id}" if rec.prompt_id is not None else f"Prompt_{rec.idx}"
        prompt_text = render_chat_prompt(tokenizer, rec.system_prompt, rec.user_input)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        if args.max_input_tokens > 0:
            prompt_ids = prompt_ids[-args.max_input_tokens:]

        h_np = _encode_hidden(model=model, device=model_device, prompt_ids=prompt_ids)
        h_t = torch.tensor(h_np, dtype=torch.float32, device=head_device).unsqueeze(0)

        base_scores = _score_label_candidates(
            model=model,
            device=model_device,
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            candidate_id_lists=candidate_id_lists,
        )
        base_map = _extract_base_scores(label_candidates, base_scores)
        base_margin = float(base_map["yes"] - base_map["no"])
        bm_t = torch.tensor([base_margin], dtype=torch.float32, device=head_device)

        with torch.no_grad():
            mech_logits = heads.mechanism_logits(h_t)
            pred = heads.predict_margin(h_t, mech_logits, bm_t)
            margin = pred["margin"].squeeze(0)
            z_hat = pred["z_hat"].squeeze(0)
            expected_deg_prob = pred["expected_deg_prob"].squeeze(0)
            a = compute_sufficiency_score(
                mechanism_probs=z_hat.unsqueeze(0),
                margin=margin.unsqueeze(0),
                eta1=eta1,
                eta2=eta2,
                eta3=eta3,
            ).squeeze(0)

        margin_f = float(margin.detach().cpu())
        a_f = float(a.detach().cpu())
        expected_deg_f = float(expected_deg_prob.detach().cpu())
        if a_f < tau:
            final_label = "insufficient"
        else:
            final_label = "yes" if margin_f >= 0 else "no"

        yes_score = margin_f
        no_score = -margin_f
        ins_score = float(tau - a_f)
        response_lines = [
            "Final Deterministic Prediction:",
            final_label,
            f"[label_ranking_scores] {{'yes': {round(yes_score, 6)}, 'no': {round(no_score, 6)}, 'insufficient': {round(ins_score, 6)}}}",
        ]
        if args.debug:
            z_np = z_hat.detach().cpu().numpy()
            topk = np.argsort(z_np)[::-1][: args.debug_top_k]
            top_modules = []
            for mid in topk.tolist():
                name = module_names[mid] if mid < len(module_names) else f"module_{mid}"
                top_modules.append({"module_id": int(mid), "module_name": str(name), "prob": float(z_np[mid])})
            ent = float(distribution_entropy(z_hat.unsqueeze(0)).squeeze(0).detach().cpu())
            payload = {
                "query": {"pert": rec.pert, "gene": rec.gene, "prompt_id": rec.prompt_id},
                "base_scores": base_map,
                "base_margin": base_margin,
                "pred_margin": margin_f,
                "expected_deg_prob": expected_deg_f,
                "sufficiency": a_f,
                "tau": tau,
                "mechanism_entropy": ent,
                "top_mechanisms": top_modules,
                "final_prediction": final_label,
            }
            response_lines.append(f"[msld_debug] {json.dumps(payload, ensure_ascii=True)}")
        out_blocks.append(format_prediction_block(header, "\n".join(response_lines)))

        if i % 50 == 0 or i == len(records):
            print(f"[MSLD] done {i}/{len(records)}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(out_blocks)
    print(f"[MSLD] Saved predictions: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query-only inference for MSLD.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--label-candidates", default="", help="Override label candidates. Empty means use ckpt.")
    p.add_argument("--label-prefix", default="", help="Override label prefix. Empty means use ckpt.")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--eta1", type=float, default=None)
    p.add_argument("--eta2", type=float, default=None)
    p.add_argument("--eta3", type=float, default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-top-k", type=int, default=8)
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
