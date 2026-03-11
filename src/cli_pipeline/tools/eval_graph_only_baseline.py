#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Baseline-2: graph-only module proposer + shallow gene scorer."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from graph_only_baseline_model import (
    GraphOnlyGeneScorer,
    GraphOnlyModuleProposer,
    GraphOnlyPerturbationPredictor,
)
from graph_only_baseline_utils import (
    build_gene_feature_matrix,
    build_module_feature_matrix,
    build_perturbations,
    compute_binary_metrics,
    distribution_entropy,
    load_graph_for_baseline,
    load_graph_only_examples,
    sample_episode_for_perturbation,
    top_modules_from_posterior,
)


def _resolve_config(args, ckpt: dict, key: str):
    defaults = {"episode_mode": "full_perturbation"}
    value = getattr(args, key)
    if value is not None:
        return value
    if key in ckpt.get("graph_config", {}):
        return ckpt["graph_config"][key]
    if key in ckpt.get("episode_config", {}):
        return ckpt["episode_config"][key]
    if key in ckpt.get("args", {}):
        return ckpt["args"][key]
    if key in defaults:
        return defaults[key]
    raise KeyError(f"Cannot resolve config for key={key}")


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def run(args) -> None:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if ckpt.get("version") != "graph_only_baseline_v1":
        raise RuntimeError(f"Unsupported checkpoint version: {ckpt.get('version')}")

    graph_loader = str(_resolve_config(args, ckpt, "graph_loader"))
    graph = load_graph_for_baseline(
        kg_dir=args.kg_dir,
        graph_loader=graph_loader,
        graph_alpha=float(_resolve_config(args, ckpt, "graph_alpha")),
        graph_diffusion_hops=int(_resolve_config(args, ckpt, "graph_diffusion_hops")),
        graph_diffusion_decay=float(_resolve_config(args, ckpt, "graph_diffusion_decay")),
        max_modules=int(_resolve_config(args, ckpt, "max_modules")),
        kg_nodes_file=str(_resolve_config(args, ckpt, "kg_nodes_file")),
        kg_edges_file=str(_resolve_config(args, ckpt, "kg_edges_file")),
        kg_graph_file=str(_resolve_config(args, ckpt, "kg_graph_file")),
    )

    model_cfg = ckpt["model_config"]
    if int(model_cfg["num_modules"]) != int(graph.num_modules):
        raise RuntimeError(
            f"Module count mismatch: ckpt={int(model_cfg['num_modules'])} graph={int(graph.num_modules)}. "
            "Use consistent KG/graph settings."
        )

    proposer = GraphOnlyModuleProposer(
        feature_dim=int(model_cfg["module_feature_dim"]),
        arch=str(model_cfg["proposer_arch"]),
        hidden_dim=int(model_cfg["proposer_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    gene_scorer = GraphOnlyGeneScorer(
        feature_dim=int(model_cfg["gene_feature_dim"]),
        arch=str(model_cfg["gene_scorer_arch"]),
        hidden_dim=int(model_cfg["gene_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    model = GraphOnlyPerturbationPredictor(proposer=proposer, gene_scorer=gene_scorer)
    load_info = model.load_state_dict(ckpt["predictor_state_dict"], strict=True)
    if load_info.missing_keys or load_info.unexpected_keys:
        raise RuntimeError(
            f"Unexpected state_dict mismatch: missing={load_info.missing_keys} unexpected={load_info.unexpected_keys}"
        )

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    eval_examples = load_graph_only_examples(
        labels_csv=args.labels_csv,
        graph=graph,
        split=args.split,
        default_cell_id=args.default_cell_id,
        allow_missing_drug=True,
    )
    if not eval_examples:
        raise RuntimeError("No evaluation examples loaded.")

    min_genes = int(_resolve_config(args, ckpt, "min_genes_per_perturbation"))
    perturbations = build_perturbations(
        eval_examples,
        min_genes_per_perturbation=min_genes,
    )
    if not perturbations:
        raise RuntimeError("No perturbations survive min-gene filtering on eval split.")

    episode_mode = str(_resolve_config(args, ckpt, "episode_mode"))
    support_fraction = float(_resolve_config(args, ckpt, "support_fraction"))
    min_support_size = int(_resolve_config(args, ckpt, "min_support_size"))
    min_query_size = int(_resolve_config(args, ckpt, "min_query_size"))
    max_support_size = int(_resolve_config(args, ckpt, "max_support_size"))

    proposer_feature_set = str(model_cfg["proposer_feature_set"])
    gene_feature_set = str(model_cfg["gene_feature_set"])
    include_top_cov = bool(model_cfg.get("include_top_module_coverage", True))
    top_k_modules = int(model_cfg.get("gene_top_k_modules", 8))
    seed = int(args.seed if args.seed is not None else ckpt.get("args", {}).get("seed", 42))
    top_modules_report = int(args.top_modules_report)
    decision_threshold = float(args.decision_threshold)

    rows: List[Dict[str, object]] = []
    pert_summaries: List[Dict[str, object]] = []
    y_true: List[int] = []
    y_prob: List[float] = []
    skipped_perturbations = 0

    with torch.no_grad():
        for i, pert in enumerate(perturbations, 1):
            if episode_mode == "full_perturbation":
                episode = {
                    "support_gene_ids": pert.gene_ids.astype(np.int64),
                    "support_labels": pert.labels.astype(np.float32),
                    "query_gene_ids": pert.gene_ids.astype(np.int64),
                    "query_labels": pert.labels.astype(np.float32),
                }
            else:
                episode = sample_episode_for_perturbation(
                    pert,
                    seed=seed,
                    epoch=0,
                    support_fraction=support_fraction,
                    min_support_size=min_support_size,
                    min_query_size=min_query_size,
                    max_support_size=max_support_size,
                )
                if episode is None:
                    skipped_perturbations += 1
                    continue

            mod_feat_np, _, mod_aux = build_module_feature_matrix(
                graph=graph,
                drug_id=int(pert.drug_id),
                support_gene_ids=episode["support_gene_ids"].tolist(),
                support_labels=episode["support_labels"].tolist(),
                proposer_feature_set=proposer_feature_set,
            )
            if mod_feat_np.size == 0:
                skipped_perturbations += 1
                continue

            mod_feat_t = _to_torch(mod_feat_np, device=device)
            q_t = model.infer_module_posterior(mod_feat_t)["q"]
            q_np = q_t.detach().cpu().numpy().astype(np.float32)

            gene_feat_np, _, _ = build_gene_feature_matrix(
                graph=graph,
                query_gene_ids=episode["query_gene_ids"].tolist(),
                q_probs=q_np,
                target_module_distribution=mod_aux["target_module_distribution"],
                gene_feature_set=gene_feature_set,
                include_top_module_coverage=include_top_cov,
                top_k_modules=top_k_modules,
            )
            if gene_feat_np.shape[0] <= 0:
                skipped_perturbations += 1
                continue

            gene_feat_t = _to_torch(gene_feat_np, device=device)
            logits_t = model.score_genes(gene_feat_t)
            prob_np = torch.sigmoid(logits_t).detach().cpu().numpy().astype(np.float32)
            pred_np = (prob_np >= float(decision_threshold)).astype(np.int64)
            q_entropy = float(distribution_entropy(q_np))

            top_modules = top_modules_from_posterior(
                q_probs=q_np,
                module_names=graph.idx_to_module,
                module_types=graph.module_types,
                top_k=top_modules_report,
            )
            support_pos_rate = float(np.mean(episode["support_labels"])) if episode["support_labels"].size > 0 else 0.0
            target_count = len(graph.get_targets(int(pert.drug_id))) if int(pert.drug_id) >= 0 else 0

            pert_summaries.append(
                {
                    "perturbation_key": pert.perturbation_key,
                    "pert": pert.pert,
                    "cell_id": int(pert.cell_id),
                    "drug_id": int(pert.drug_id),
                    "num_support": int(episode["support_gene_ids"].size),
                    "num_query": int(episode["query_gene_ids"].size),
                    "support_positive_rate": support_pos_rate,
                    "posterior_entropy": q_entropy,
                    "num_targets": int(target_count),
                    "top_modules": top_modules,
                }
            )

            query_gene_ids = episode["query_gene_ids"].astype(np.int64)
            query_labels = episode["query_labels"].astype(np.float32)
            for gid, yt, yp, pd in zip(query_gene_ids.tolist(), query_labels.tolist(), prob_np.tolist(), pred_np.tolist()):
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

            if i % args.log_every_perturbations == 0:
                print(f"[GraphOnly-Eval] done {i}/{len(perturbations)} perturbations")

    if not rows:
        raise RuntimeError("No query predictions generated. Relax split constraints.")

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        decision_threshold=decision_threshold,
    )
    mean_entropy = float(np.mean([x["posterior_entropy"] for x in pert_summaries])) if pert_summaries else None
    mean_support_pos_rate = float(np.mean([x["support_positive_rate"] for x in pert_summaries])) if pert_summaries else None

    summary = {
        "method": "graph_only_baseline_v1",
        "split": args.split,
        "num_eval_examples": len(eval_examples),
        "num_eval_perturbations": len(perturbations),
        "num_evaluated_perturbations": len(pert_summaries),
        "num_skipped_perturbations": int(skipped_perturbations),
        "num_query_predictions": len(rows),
        "decision_threshold": decision_threshold,
        "metrics": metrics,
        "mean_posterior_entropy": mean_entropy,
        "mean_support_positive_rate": mean_support_pos_rate,
        "episode_config": {
            "episode_mode": episode_mode,
            "support_fraction": support_fraction,
            "min_support_size": min_support_size,
            "min_query_size": min_query_size,
            "max_support_size": max_support_size,
            "min_genes_per_perturbation": min_genes,
            "seed": seed,
        },
        "args": vars(args),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "evaluation_summary.json")
    rows_path = os.path.join(args.out_dir, "evaluation_rows.csv")
    pert_path = os.path.join(args.out_dir, "perturbation_module_posteriors.jsonl")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    with open(rows_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
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
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(pert_path, "w", encoding="utf-8") as f:
        for item in pert_summaries:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"[GraphOnly-Eval] Saved summary: {summary_path}")
    print(f"[GraphOnly-Eval] Saved rows: {rows_path}")
    print(f"[GraphOnly-Eval] Saved perturbation posteriors: {pert_path}")
    print(
        "[GraphOnly-Eval] "
        f"acc={metrics['accuracy']} prec={metrics['precision']} rec={metrics['recall']} "
        f"f1={metrics['f1']} auroc={metrics['auroc']} auprc={metrics['auprc']} "
        f"yes_rate={metrics['yes_prediction_rate']}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate graph-only baseline checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--kg-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--default-cell-id", type=int, default=0)

    p.add_argument("--graph-loader", "--graph_loader", dest="graph_loader", default=None, choices=["ggmv", "msld"])
    p.add_argument("--graph-alpha", "--graph_alpha", dest="graph_alpha", type=float, default=None)
    p.add_argument("--graph-diffusion-hops", "--graph_diffusion_hops", dest="graph_diffusion_hops", type=int, default=None)
    p.add_argument(
        "--graph-diffusion-decay",
        "--graph_diffusion_decay",
        dest="graph_diffusion_decay",
        type=float,
        default=None,
    )
    p.add_argument("--max-modules", "--max_modules", dest="max_modules", type=int, default=None)
    p.add_argument("--kg-nodes-file", "--kg_nodes_file", dest="kg_nodes_file", default=None)
    p.add_argument("--kg-edges-file", "--kg_edges_file", dest="kg_edges_file", default=None)
    p.add_argument("--kg-graph-file", "--kg_graph_file", dest="kg_graph_file", default=None)

    p.add_argument("--support-fraction", "--support_fraction", dest="support_fraction", type=float, default=None)
    p.add_argument(
        "--episode-mode",
        "--episode_mode",
        dest="episode_mode",
        default=None,
        choices=["full_perturbation", "intra_perturbation"],
    )
    p.add_argument("--min-support-size", "--min_support_size", dest="min_support_size", type=int, default=None)
    p.add_argument("--min-query-size", "--min_query_size", dest="min_query_size", type=int, default=None)
    p.add_argument("--max-support-size", "--max_support_size", dest="max_support_size", type=int, default=None)
    p.add_argument(
        "--min-genes-per-perturbation",
        "--min_genes_per_perturbation",
        dest="min_genes_per_perturbation",
        type=int,
        default=None,
    )

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--top-modules-report", type=int, default=8)
    p.add_argument("--log-every-perturbations", type=int, default=50)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
