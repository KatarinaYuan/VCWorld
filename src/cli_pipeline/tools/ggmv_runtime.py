#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime helpers for graph-mechanism-verification mode.

This module intentionally isolates GGMV-specific logic from run_ttt_lite.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ggmv_core import GGMVParams, forward_ggmv
from ggmv_graph import GGMVGraph, load_ggmv_graph
from ggmv_llm import LLMMechanismScorer, build_query_support_summary
from ggmv_support import build_query_from_record, build_support_pool_from_labels


def load_ggmv_params(args) -> GGMVParams:
    """Build GGMV scalar params from CLI args plus optional JSON overrides."""
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


def extract_base_score_map(label_candidates: List[str], scores: List[float]) -> Dict[str, float]:
    out = {k.strip().lower(): float(v) for k, v in zip(label_candidates, scores)}
    out.setdefault("yes", float("-inf"))
    out.setdefault("no", float("-inf"))
    out.setdefault("insufficient", float("-inf"))
    return out


def format_debug_modules(module_pairs: List[Tuple[int, float]], graph: GGMVGraph, top_n: int = 6) -> List[dict]:
    out = []
    for module_id, score in module_pairs[:top_n]:
        if module_id >= 0 and module_id < len(graph.idx_to_module):
            name = graph.idx_to_module[module_id]
        else:
            name = f"module_{module_id}"
        out.append({"module_id": int(module_id), "module_name": name, "score": float(score)})
    return out


def _norm_pair_key(pert: str, gene: str) -> Tuple[str, str]:
    return (str(pert).strip().lower(), str(gene).strip().upper())


def _render_chat_prompt_with_optional_output(
    tokenizer,
    system_prompt: str,
    user_input: str,
    output_text: Optional[str],
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    if output_text:
        messages.append({"role": "assistant", "content": output_text})
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    else:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if isinstance(rendered, list):
        return rendered[0]
    return rendered


def build_prompt_record_lookup(records) -> Dict[Tuple[str, str], object]:
    """Map (pert,gene) -> prompt record for support-based adaptation."""
    out: Dict[Tuple[str, str], object] = {}
    for rec in records:
        if not rec.pert or not rec.gene or not rec.system_prompt or not rec.user_input:
            continue
        key = _norm_pair_key(rec.pert, rec.gene)
        if key not in out:
            out[key] = rec
    return out


def select_global_support_set(
    support_pool: List[dict],
    *,
    max_support_perturbations: int,
    seed: int,
) -> List[dict]:
    """Select one fixed support set for the entire run (query-independent)."""
    if not support_pool:
        return []
    n = len(support_pool)
    k = max(1, min(int(max_support_perturbations), n))
    rng = np.random.default_rng(int(seed))
    perm = list(rng.permutation(n))
    return [support_pool[i] for i in perm[:k]]


def build_support_adaptation_token_ids(
    *,
    tokenizer,
    support_set: List[dict],
    graph: GGMVGraph,
    prompt_record_lookup: Dict[Tuple[str, str], object],
    max_examples: int,
    max_tokens: int,
) -> List[List[int]]:
    """Build tokenized adaptation examples from a fixed support set."""
    out: List[List[int]] = []
    seen: set = set()

    for support in support_set:
        pert = str(support.get("perturbation_id", "")).strip()
        if not pert:
            continue
        gene_ids = list(support.get("labeled_gene_ids", []))
        for gid in gene_ids:
            try:
                gid_i = int(gid)
            except (TypeError, ValueError):
                continue
            if gid_i < 0 or gid_i >= len(graph.idx_to_gene):
                continue
            gene_name = graph.idx_to_gene[gid_i]
            key = _norm_pair_key(pert, gene_name)
            if key in seen:
                continue
            rec = prompt_record_lookup.get(key)
            if rec is None or not rec.system_prompt or not rec.user_input:
                continue

            text = _render_chat_prompt_with_optional_output(
                tokenizer=tokenizer,
                system_prompt=rec.system_prompt,
                user_input=rec.user_input,
                output_text=rec.output_text,
            )
            tok_ids = tokenizer(text, add_special_tokens=False).input_ids
            if max_tokens > 0:
                tok_ids = tok_ids[-max_tokens:]
            if not tok_ids:
                continue
            out.append(tok_ids)
            seen.add(key)
            if max_examples > 0 and len(out) >= max_examples:
                return out
    return out


@dataclass
class GGMVRuntime:
    graph: GGMVGraph
    params: GGMVParams
    support_pool: List[dict]
    fixed_support_set: List[dict]
    support_adapt_token_ids: List[List[int]]
    prompt_record_lookup: Dict[Tuple[str, str], object]
    mech_scorer: Optional[LLMMechanismScorer]


def initialize_ggmv_runtime(
    *,
    args,
    model,
    tokenizer,
    device: torch.device,
    label_candidates: List[str],
    records,
    labels_csv: str,
) -> GGMVRuntime:
    """Initialize all GGMV components once for the run."""
    required_labels = {"yes", "no", "insufficient"}
    if not required_labels.issubset({x.strip().lower() for x in label_candidates}):
        raise RuntimeError("graph-mechanism-verification requires label candidates containing yes,no,insufficient")
    if not args.kg_dir:
        raise RuntimeError("--kg-dir is required when --prediction-mode graph-mechanism-verification")

    print(f"[TTT-lite][GGMV] Loading KG from {args.kg_dir}")
    graph = load_ggmv_graph(
        kg_dir=args.kg_dir,
        alpha=args.ggmv_alpha,
        nodes_file=args.kg_nodes_file,
        edges_file=args.kg_edges_file,
        graph_file=args.kg_graph_file,
        strict=False,
    )
    params = load_ggmv_params(args)
    print(f"[TTT-lite][GGMV] Params: {params}")

    support_pool = build_support_pool_from_labels(
        labels_csv=labels_csv,
        graph=graph,
        split=args.ggmv_support_split,
        default_cell_id=args.ggmv_default_cell_id,
        max_genes_per_perturbation=args.ggmv_max_genes_per_support,
    )
    print(f"[TTT-lite][GGMV] Support pool perturbations: {len(support_pool)}")

    fixed_support_set = select_global_support_set(
        support_pool,
        max_support_perturbations=args.ggmv_max_support_perturbations,
        seed=args.ggmv_global_support_seed,
    )
    print(f"[TTT-lite][GGMV] Fixed global support perturbations: {len(fixed_support_set)}")

    prompt_record_lookup = build_prompt_record_lookup(records)
    support_adapt_token_ids = build_support_adaptation_token_ids(
        tokenizer=tokenizer,
        support_set=fixed_support_set,
        graph=graph,
        prompt_record_lookup=prompt_record_lookup,
        max_examples=args.inner_support_max_examples,
        max_tokens=args.inner_max_tokens,
    )
    print(f"[TTT-lite][GGMV] Support adaptation examples: {len(support_adapt_token_ids)}")

    if args.ggmv_use_llm_mech and not args.ggmv_no_llm_mech:
        mech_scorer = LLMMechanismScorer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_tokens=args.ggmv_llm_max_tokens,
            module_top_genes=args.ggmv_module_top_genes,
            use_prompt_context=args.ggmv_use_prompt_context,
            prompt_context_tokens=args.ggmv_prompt_context_tokens,
            max_support_in_summary=args.ggmv_summary_max_support,
            max_genes_per_support=args.ggmv_summary_max_genes,
            overlap_bonus_weight=args.ggmv_overlap_bonus_weight,
        )
        print("[TTT-lite][GGMV] LLM mechanism scorer: enabled")
    else:
        mech_scorer = None
        print("[TTT-lite][GGMV] LLM mechanism scorer: disabled")

    return GGMVRuntime(
        graph=graph,
        params=params,
        support_pool=support_pool,
        fixed_support_set=fixed_support_set,
        support_adapt_token_ids=support_adapt_token_ids,
        prompt_record_lookup=prompt_record_lookup,
        mech_scorer=mech_scorer,
    )


def run_global_support_adaptation(
    *,
    model,
    device: torch.device,
    optimizer,
    support_adapt_token_ids: List[List[int]],
    inner_steps: int,
) -> Optional[float]:
    """Apply one global adaptation phase on fixed support examples."""
    if inner_steps <= 0:
        return None
    if not support_adapt_token_ids:
        return None

    model.train()
    loss: Optional[float] = None
    for _ in range(inner_steps):
        for tok_ids in support_adapt_token_ids:
            adapt_input = torch.tensor([tok_ids], dtype=torch.long, device=device)
            out = model(
                input_ids=adapt_input,
                attention_mask=torch.ones_like(adapt_input),
                labels=adapt_input,
            )
            l = out.loss
            l.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss = float(l.detach().cpu())
    return loss


def build_ggmv_prediction_response(
    *,
    rec,
    label_row: dict,
    prompt_ids: List[int],
    base_score_map: Dict[str, float],
    runtime: GGMVRuntime,
    args,
) -> str:
    """Run one GGMV forward pass for a test query using fixed global support set."""
    query = build_query_from_record(
        rec=rec,
        label_row=label_row,
        graph=runtime.graph,
        base_scores=base_score_map,
        default_cell_id=args.ggmv_default_cell_id,
    )
    support_set = runtime.fixed_support_set

    support_summary_text = build_query_support_summary(
        query=query,
        support_set=support_set,
        graph=runtime.graph,
        max_support=args.ggmv_summary_max_support,
        max_genes_per_support=args.ggmv_summary_max_genes,
    )

    if runtime.mech_scorer is not None:
        context_vec, support_summary_text = runtime.mech_scorer.encode_query_support_context(
            query=query,
            support_set=support_set,
            graph=runtime.graph,
            prompt_ids=prompt_ids,
            summary_text=support_summary_text,
        )
        llm_mech_fn = (
            lambda module_ids, _context_vec=context_vec: runtime.mech_scorer.score_candidate_modules_from_context(
                context_vec=_context_vec,
                query=query,
                candidate_module_ids=module_ids,
                graph=runtime.graph,
            )
        )
        llm_suff = float(runtime.mech_scorer.score_sufficiency(context_vec=context_vec, support_set=support_set))
    else:
        llm_mech_fn = None
        llm_suff = 0.0

    out_ggmv = forward_ggmv(
        query=query,
        support_set=support_set,
        graph=runtime.graph,
        params=runtime.params,
        llm_mech_score_fn=llm_mech_fn,
        llm_sufficiency_score=llm_suff,
    )
    calibrated = out_ggmv["scores"]
    final_label = max(["yes", "no", "insufficient"], key=lambda x: float(calibrated[x]))

    response_lines = [
        "Final Deterministic Prediction:",
        final_label,
        f"[label_ranking_scores] {{'yes': {round(float(calibrated['yes']), 4)}, 'no': {round(float(calibrated['no']), 4)}, 'insufficient': {round(float(calibrated['insufficient']), 4)}}}",
    ]
    if args.ggmv_debug:
        support_ids = [str(s.get("perturbation_id", "")) for s in support_set]
        debug_payload = {
            "query": {
                "query_id": query.get("query_id"),
                "drug_name": query.get("drug_name"),
                "gene_name": query.get("gene_name"),
                "drug_id": query.get("drug_id"),
                "gene_id": query.get("gene_id"),
                "support_size": len(support_set),
                "support_adapt_examples": len(runtime.support_adapt_token_ids),
            },
            "global_support_set": support_ids,
            "support_summary": support_summary_text,
            "candidate_modules_top": format_debug_modules(
                out_ggmv["debug"]["top_prior_modules"], runtime.graph, top_n=args.ggmv_debug_top_n
            ),
            "graph_prior_top": format_debug_modules(
                out_ggmv["debug"]["top_prior_modules"], runtime.graph, top_n=args.ggmv_debug_top_n
            ),
            "support_evidence_top": format_debug_modules(
                out_ggmv["debug"]["top_support_evidence_modules"], runtime.graph, top_n=args.ggmv_debug_top_n
            ),
            "llm_mech_top": format_debug_modules(
                out_ggmv["debug"]["top_llm_mech_modules"], runtime.graph, top_n=args.ggmv_debug_top_n
            ),
            "posterior_top": format_debug_modules(
                out_ggmv["debug"]["top_posterior_modules"], runtime.graph, top_n=args.ggmv_debug_top_n
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
    return "\n".join(response_lines)
