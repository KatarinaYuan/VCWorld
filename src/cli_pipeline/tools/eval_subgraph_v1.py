#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate V1 support-conditioned subgraph pipeline on unseen-drug split."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch

from graph_only_baseline_utils import (
    build_module_feature_matrix,
    build_perturbations,
    compute_binary_metrics,
    load_graph_for_baseline,
    load_graph_only_examples,
    sample_episode_for_perturbation,
)
from subgraph_v1 import (
    FastScorer,
    FrozenLLMVerifier,
    HeuristicPathSeededPCSTProposer,
    build_candidate_feature_vector,
    build_gene_feature_matrix_for_candidate,
    summarize_candidate_score_components,
)


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _module_preview(module_names: List[str], module_ids: List[int], max_items: int) -> List[str]:
    out: List[str] = []
    for mid in module_ids[: max(1, int(max_items))]:
        if 0 <= int(mid) < len(module_names):
            out.append(str(module_names[int(mid)]))
        else:
            out.append(f"module_{int(mid)}")
    return out


def _append_jsonl(path: str, payload: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _should_log_perturbation(args, perturbation_idx: int) -> bool:
    if perturbation_idx <= int(args.debug_first_perturbations):
        return True
    every = max(1, int(args.log_every_perturbations))
    return (perturbation_idx % every) == 0


def _resolve_config(args, ckpt: dict, key: str, default=None):
    val = getattr(args, key, None)
    if val is not None:
        return val
    if key in ckpt.get("train_config", {}):
        return ckpt["train_config"][key]
    if key in ckpt.get("graph_config", {}):
        return ckpt["graph_config"][key]
    return default


def run(args) -> None:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if ckpt.get("version") != "subgraph_v1":
        raise RuntimeError(f"Unsupported checkpoint version: {ckpt.get('version')}")

    graph = load_graph_for_baseline(
        kg_dir=args.kg_dir,
        graph_loader=str(_resolve_config(args, ckpt, "graph_loader", "ggmv")),
        graph_alpha=float(_resolve_config(args, ckpt, "graph_alpha", 0.1)),
        graph_diffusion_hops=int(_resolve_config(args, ckpt, "graph_diffusion_hops", 1)),
        graph_diffusion_decay=float(_resolve_config(args, ckpt, "graph_diffusion_decay", 1.0)),
        max_modules=int(_resolve_config(args, ckpt, "max_modules", 2048)),
        kg_nodes_file=str(_resolve_config(args, ckpt, "kg_nodes_file", "nodes.json")),
        kg_edges_file=str(_resolve_config(args, ckpt, "kg_edges_file", "edges.json")),
        kg_graph_file=str(_resolve_config(args, ckpt, "kg_graph_file", "graph.json")),
    )

    eval_examples = load_graph_only_examples(
        labels_csv=args.labels_csv,
        graph=graph,
        split=args.split,
        default_cell_id=int(_resolve_config(args, ckpt, "default_cell_id", 0)),
        allow_missing_drug=True,
    )
    perturbations = build_perturbations(
        eval_examples,
        min_genes_per_perturbation=int(_resolve_config(args, ckpt, "min_genes_per_perturbation", 16)),
    )
    if not perturbations:
        raise RuntimeError("No eval perturbations after filtering.")

    model_cfg = ckpt["model_config"]
    scorer = FastScorer(
        candidate_feature_dim=int(model_cfg["candidate_feature_dim"]),
        gene_feature_dim=int(model_cfg["gene_feature_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    scorer.load_state_dict(ckpt["scorer_state_dict"], strict=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    scorer = scorer.to(device)
    scorer.eval()

    proposer = HeuristicPathSeededPCSTProposer(
        graph=graph,
        max_neighbors=int(_resolve_config(args, ckpt, "max_neighbors", 24)),
        rng_seed=int(_resolve_config(args, ckpt, "seed", 42)),
        a1_near_drug=float(_resolve_config(args, ckpt, "a1_near_drug", 1.0)),
        a2_cover_pos=float(_resolve_config(args, ckpt, "a2_cover_pos", 1.0)),
        a3_cover_neg=float(_resolve_config(args, ckpt, "a3_cover_neg", 1.0)),
        a4_type_prior=float(_resolve_config(args, ckpt, "a4_type_prior", 0.25)),
        b1_confidence=float(_resolve_config(args, ckpt, "b1_confidence", 1.0)),
        b2_type_penalty=float(_resolve_config(args, ckpt, "b2_type_penalty", 0.25)),
    )
    verifier = FrozenLLMVerifier(replay_weight=0.5, grounding_weight=0.5)

    support_fraction = float(_resolve_config(args, ckpt, "support_fraction", 0.4))
    min_support_size = int(_resolve_config(args, ckpt, "min_support_size", 8))
    min_query_size = int(_resolve_config(args, ckpt, "min_query_size", 8))
    max_support_size = int(_resolve_config(args, ckpt, "max_support_size", 256))
    num_candidates = int(_resolve_config(args, ckpt, "num_candidates", 12))
    top_k_verifier = int(_resolve_config(args, ckpt, "top_k_verifier", 3))
    min_candidate_nodes = int(_resolve_config(args, ckpt, "min_candidate_nodes", 4))
    max_candidate_nodes = int(_resolve_config(args, ckpt, "max_candidate_nodes", 24))
    beta_replay = float(_resolve_config(args, ckpt, "beta_replay", 0.5))
    gamma_llm = float(_resolve_config(args, ckpt, "gamma_llm", 0.5))
    lambda_size = float(_resolve_config(args, ckpt, "lambda_size", 0.01))
    decision_threshold = float(args.decision_threshold)
    seed = int(args.seed if args.seed is not None else _resolve_config(args, ckpt, "seed", 42))

    if args.debug_log_jsonl:
        dbg_dir = os.path.dirname(args.debug_log_jsonl)
        if dbg_dir:
            os.makedirs(dbg_dir, exist_ok=True)
        with open(args.debug_log_jsonl, "w", encoding="utf-8") as f:
            f.write("")
        print(f"[SubgraphV1][Eval] debug_log_jsonl={args.debug_log_jsonl}")

    y_true: List[int] = []
    y_prob: List[float] = []
    rows: List[Dict[str, object]] = []
    perturbation_rows: List[Dict[str, object]] = []
    skipped = 0

    with torch.no_grad():
        for i, pert in enumerate(perturbations, 1):
            ep = sample_episode_for_perturbation(
                pert,
                seed=seed,
                epoch=0,
                support_fraction=support_fraction,
                min_support_size=min_support_size,
                min_query_size=min_query_size,
                max_support_size=max_support_size,
            )
            if ep is None:
                skipped += 1
                continue

            _, _, mod_aux = build_module_feature_matrix(
                graph=graph,
                drug_id=int(pert.drug_id),
                support_gene_ids=ep["support_gene_ids"].tolist(),
                support_labels=ep["support_labels"].tolist(),
                proposer_feature_set="full",
            )
            candidates = proposer.generate(
                mod_aux=mod_aux,
                num_candidates=num_candidates,
                min_nodes=min_candidate_nodes,
                max_nodes=max_candidate_nodes,
            )
            if not candidates:
                skipped += 1
                continue
            proposer_components = proposer.score_components(mod_aux)

            quality_preds: List[float] = []
            support_probs_list: List[np.ndarray] = []
            query_probs_list: List[np.ndarray] = []
            replay_list: List[float] = []

            for cand in candidates:
                c_feat = build_candidate_feature_vector(
                    candidate=cand,
                    mod_aux=mod_aux,
                    module_types=graph.module_types,
                    total_modules=int(graph.num_modules),
                )
                q_feat = build_gene_feature_matrix_for_candidate(
                    graph=graph,
                    gene_ids=ep["query_gene_ids"],
                    candidate=cand,
                    mod_aux=mod_aux,
                )
                s_feat = build_gene_feature_matrix_for_candidate(
                    graph=graph,
                    gene_ids=ep["support_gene_ids"],
                    candidate=cand,
                    mod_aux=mod_aux,
                )
                q_logits, q_pred = scorer(candidate_features=_to_torch(c_feat, device), gene_features=_to_torch(q_feat, device))
                s_logits, _ = scorer(candidate_features=_to_torch(c_feat, device), gene_features=_to_torch(s_feat, device))

                q_prob = torch.sigmoid(q_logits).detach().cpu().numpy().astype(np.float32)
                s_prob = torch.sigmoid(s_logits).detach().cpu().numpy().astype(np.float32)
                replay = float(np.mean((s_prob >= 0.5).astype(np.float32) == ep["support_labels"])) if s_prob.size == ep["support_labels"].size and s_prob.size > 0 else 0.0
                quality_preds.append(float(q_pred.detach().cpu()))
                query_probs_list.append(q_prob)
                support_probs_list.append(s_prob)
                replay_list.append(replay)

            order = np.argsort(np.asarray(quality_preds, dtype=np.float32))[::-1]
            topk = order[: max(1, min(top_k_verifier, len(candidates)))].tolist()

            llm_score = np.zeros((len(candidates),), dtype=np.float32)
            llm_summary: Dict[int, str] = {}
            for ci in topk:
                out = verifier.verify_candidate(
                    candidate=candidates[ci],
                    support_labels=ep["support_labels"],
                    support_probs=support_probs_list[ci],
                    module_names=graph.idx_to_module,
                )
                llm_score[ci] = float(out["confidence"])
                llm_summary[ci] = str(out["mechanism_summary"])

            final_score = np.asarray(quality_preds, dtype=np.float32)
            final_score = final_score + beta_replay * np.asarray(replay_list, dtype=np.float32)
            final_score = final_score + gamma_llm * llm_score
            final_score = final_score - lambda_size * np.asarray([c.size for c in candidates], dtype=np.float32)
            best_idx = int(np.argmax(final_score))
            llm_bonus = gamma_llm * llm_score

            do_sparse_log = _should_log_perturbation(args, perturbation_idx=i)
            do_print = bool(args.print_candidate_details and do_sparse_log)
            do_file = bool(args.debug_log_jsonl and (args.debug_log_all_perturbations or do_sparse_log))
            if do_print or do_file:
                candidate_details: List[Dict[str, object]] = []
                for ci, cand in enumerate(candidates):
                    comp = summarize_candidate_score_components(
                        candidate=cand,
                        score_components=proposer_components,
                    )
                    candidate_details.append(
                        {
                            "candidate_index": int(ci),
                            "quality_pred": float(quality_preds[ci]),
                            "replay_score": float(replay_list[ci]),
                            "llm_score": float(llm_score[ci]),
                            "llm_bonus": float(llm_bonus[ci]),
                            "size_penalty": float(lambda_size) * float(cand.size),
                            "final_score": float(final_score[ci]),
                            "candidate_size": int(cand.size),
                            "candidate_edges": int(cand.estimated_edges),
                            "modules": _module_preview(
                                graph.idx_to_module,
                                cand.module_ids,
                                max_items=args.log_modules_per_candidate,
                            ),
                            **comp,
                        }
                    )
                by_final = sorted(candidate_details, key=lambda x: float(x["final_score"]), reverse=True)
                by_quality = sorted(candidate_details, key=lambda x: float(x["quality_pred"]), reverse=True)
                top_prize_mid = np.argsort(np.asarray(proposer_components["prize"], dtype=np.float32))[::-1][
                    : max(1, int(args.debug_top_modules))
                ].tolist()
                top_prize_names = _module_preview(
                    graph.idx_to_module,
                    top_prize_mid,
                    max_items=args.debug_top_modules,
                )

                if do_print:
                    best = by_final[0] if by_final else {}
                    print(
                        f"[SubgraphV1][Eval][Pert {i}] pert={pert.pert} support={int(ep['support_gene_ids'].size)} "
                        f"query={int(ep['query_gene_ids'].size)} cand={len(candidates)} "
                        f"best_idx={best.get('candidate_index')} best_final={best.get('final_score')} "
                        f"best_q={best.get('quality_pred')} best_llm_bonus={best.get('llm_bonus')}"
                    )
                    print(f"[SubgraphV1][Eval][Pert {i}] proposer_top_prize_modules={top_prize_names}")
                    for item in by_final[: max(1, int(args.debug_top_candidates))]:
                        print(
                            f"[SubgraphV1][Eval][Pert {i}][Cand {item['candidate_index']}] "
                            f"final={item['final_score']:.4f} q={item['quality_pred']:.4f} replay={item['replay_score']:.4f} "
                            f"llm={item['llm_score']:.4f} llm_bonus={item['llm_bonus']:.4f} "
                            f"mean_prize={item['mean_prize']:.4f} size={item['candidate_size']} modules={item['modules']}"
                        )
                    print(
                        f"[SubgraphV1][Eval][Pert {i}] "
                        f"top_by_quality={[int(x['candidate_index']) for x in by_quality[: max(1, int(args.debug_top_candidates))]]} "
                        f"top_by_final={[int(x['candidate_index']) for x in by_final[: max(1, int(args.debug_top_candidates))]]}"
                    )

                if do_file:
                    payload: Dict[str, object] = {
                        "stage": "eval",
                        "perturbation_index": int(i),
                        "perturbation_key": str(pert.perturbation_key),
                        "pert": str(pert.pert),
                        "drug_id": int(pert.drug_id),
                        "num_support": int(ep["support_gene_ids"].size),
                        "num_query": int(ep["query_gene_ids"].size),
                        "num_candidates": int(len(candidates)),
                        "best_candidate_index": int(best_idx),
                        "top_by_quality": [int(x["candidate_index"]) for x in by_quality],
                        "top_by_final": [int(x["candidate_index"]) for x in by_final],
                        "proposer_top_prize_modules": top_prize_names,
                        "candidates": candidate_details,
                    }
                    _append_jsonl(args.debug_log_jsonl, payload)

            best_q_prob = query_probs_list[best_idx]
            best_cand = candidates[best_idx]
            pred_bin = (best_q_prob >= decision_threshold).astype(np.int64)
            for gid, yt, yp, pd in zip(
                ep["query_gene_ids"].tolist(),
                ep["query_labels"].tolist(),
                best_q_prob.tolist(),
                pred_bin.tolist(),
            ):
                gene_name = graph.idx_to_gene[int(gid)] if 0 <= int(gid) < len(graph.idx_to_gene) else str(gid)
                rows.append(
                    {
                        "perturbation_key": pert.perturbation_key,
                        "pert": pert.pert,
                        "cell_id": int(pert.cell_id),
                        "drug_id": int(pert.drug_id),
                        "gene_id": int(gid),
                        "gene": str(gene_name),
                        "true_label": int(yt),
                        "pred_prob": float(yp),
                        "pred_label": int(pd),
                    }
                )
                y_true.append(int(yt))
                y_prob.append(float(yp))

            perturbation_rows.append(
                {
                    "perturbation_key": pert.perturbation_key,
                    "pert": pert.pert,
                    "num_support": int(ep["support_gene_ids"].size),
                    "num_query": int(ep["query_gene_ids"].size),
                    "num_candidates": int(len(candidates)),
                    "selected_candidate_size": int(best_cand.size),
                    "selected_candidate_edges": int(best_cand.estimated_edges),
                    "selected_candidate_modules": [
                        graph.idx_to_module[mid] if 0 <= int(mid) < len(graph.idx_to_module) else f"module_{mid}"
                        for mid in best_cand.module_ids[: args.top_modules_report]
                    ],
                    "selected_quality_score": float(quality_preds[best_idx]),
                    "selected_llm_score": float(llm_score[best_idx]),
                    "selected_replay": float(replay_list[best_idx]),
                    "selected_mechanism_summary": llm_summary.get(best_idx, ""),
                }
            )

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, decision_threshold=decision_threshold)
    summary = {
        "method": "subgraph_v1",
        "split": args.split,
        "num_eval_examples": len(eval_examples),
        "num_eval_perturbations": len(perturbations),
        "num_skipped_perturbations": int(skipped),
        "num_query_predictions": len(rows),
        "decision_threshold": float(decision_threshold),
        "metrics": metrics,
        "args": vars(args),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "evaluation_summary.json")
    rows_path = os.path.join(args.out_dir, "evaluation_rows.csv")
    pert_path = os.path.join(args.out_dir, "selected_candidates.jsonl")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    with open(rows_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "perturbation_key",
                "pert",
                "cell_id",
                "drug_id",
                "gene_id",
                "gene",
                "true_label",
                "pred_prob",
                "pred_label",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(pert_path, "w", encoding="utf-8") as f:
        for item in perturbation_rows:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"[SubgraphV1][Eval] summary={summary_path}")
    print(f"[SubgraphV1][Eval] rows={rows_path}")
    print(f"[SubgraphV1][Eval] candidates={pert_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate V1 support-conditioned subgraph pipeline.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--kg-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="test")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--top-modules-report", type=int, default=8)
    p.add_argument("--log-every-perturbations", type=int, default=20)
    p.add_argument("--debug-first-perturbations", type=int, default=3)
    p.add_argument("--debug-top-candidates", type=int, default=3)
    p.add_argument("--debug-top-modules", type=int, default=8)
    p.add_argument("--log-modules-per-candidate", type=int, default=6)
    p.add_argument("--print-candidate-details", action="store_true")
    p.add_argument("--debug-log-jsonl", default="")
    p.add_argument("--debug-log-all-perturbations", action="store_true")

    # Optional overrides, otherwise from checkpoint
    p.add_argument("--graph-loader", default=None, choices=["ggmv", "msld"])
    p.add_argument("--graph-alpha", type=float, default=None)
    p.add_argument("--graph-diffusion-hops", type=int, default=None)
    p.add_argument("--graph-diffusion-decay", type=float, default=None)
    p.add_argument("--max-modules", type=int, default=None)
    p.add_argument("--kg-nodes-file", default=None)
    p.add_argument("--kg-edges-file", default=None)
    p.add_argument("--kg-graph-file", default=None)

    p.add_argument("--cpu", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
