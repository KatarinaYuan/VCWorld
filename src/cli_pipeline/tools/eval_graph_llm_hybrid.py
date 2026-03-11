#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate graph + LLM-hidden hybrid baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch

from graph_llm_hybrid_model import GraphLLMGeneScorer, GraphLLMHybridPredictor, GraphOnlyModuleProposer
from graph_llm_hybrid_utils import ensure_hidden_cache, normalize_key
from graph_only_baseline_utils import (
    build_module_feature_matrix,
    build_perturbations,
    compute_binary_metrics,
    distribution_entropy,
    load_graph_for_baseline,
    top_modules_from_posterior,
)
from msld_data import load_msld_examples


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _resolve_config(args, ckpt: dict, key: str):
    defaults = {
        "beta_prior_logit": 1.0,
        "gamma_support_logit": 1.0,
        "posterior_temperature": 1.0,
        "posterior_topk": 0,
    }
    value = getattr(args, key, None)
    if value is not None:
        return value
    for section in ("model_config", "loss_config", "episode_config", "graph_config", "llm_config", "args"):
        if key in ckpt.get(section, {}):
            return ckpt[section][key]
    if key in defaults:
        return defaults[key]
    raise KeyError(f"Cannot resolve config key={key}")


def _build_graph_gene_features_torch(
    *,
    graph,
    query_gene_ids: np.ndarray,
    q_t: torch.Tensor,
    target_module_distribution: np.ndarray,
    gene_feature_set: str,
    include_top_cov: bool,
    top_k_modules: int,
    device: torch.device,
) -> torch.Tensor:
    gids = np.asarray(query_gene_ids, dtype=np.int64)
    prox = _to_torch(graph.r_tilde[gids].toarray().astype(np.float32), device=device)
    feats: List[torch.Tensor] = [prox @ q_t]
    if gene_feature_set == "full":
        target_dist_t = _to_torch(np.asarray(target_module_distribution, dtype=np.float32), device=device)
        feats.append(prox @ target_dist_t)
        if include_top_cov:
            with torch.no_grad():
                k = max(1, min(int(top_k_modules), int(q_t.shape[0])))
                top_idx = torch.topk(q_t.detach(), k=k).indices
            feats.append((prox[:, top_idx] > 0.0).float().sum(dim=-1))
    return torch.stack(feats, dim=-1)


def _posterior_from_logits(
    *,
    module_logits: torch.Tensor,
    prior_probs: torch.Tensor,
    support_contrast: torch.Tensor,
    beta_prior_logit: float,
    gamma_support_logit: float,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    logits = module_logits.reshape(-1)
    prior = prior_probs.reshape(-1).clamp_min(1e-8)
    contrast = support_contrast.reshape(-1)
    if contrast.numel() > 0:
        c_mean = contrast.mean()
        c_std = contrast.std(unbiased=False).clamp_min(1e-6)
        contrast = (contrast - c_mean) / c_std
    logits = logits + float(beta_prior_logit) * torch.log(prior) + float(gamma_support_logit) * contrast
    temp = float(max(1e-4, temperature))
    logits = logits / temp
    m = int(logits.numel())
    k = int(topk)
    if k > 0 and k < m:
        top_idx = torch.topk(logits, k=k).indices
        masked = torch.full_like(logits, fill_value=-1e9)
        masked[top_idx] = logits[top_idx]
        logits = masked
    return torch.softmax(logits, dim=-1)


def run(args) -> None:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if ckpt.get("version") != "graph_llm_hybrid_v1":
        raise RuntimeError(f"Unsupported checkpoint version: {ckpt.get('version')}")

    graph = load_graph_for_baseline(
        kg_dir=args.kg_dir,
        graph_loader=str(_resolve_config(args, ckpt, "graph_loader")),
        graph_alpha=float(_resolve_config(args, ckpt, "graph_alpha")),
        graph_diffusion_hops=int(_resolve_config(args, ckpt, "graph_diffusion_hops")),
        graph_diffusion_decay=float(_resolve_config(args, ckpt, "graph_diffusion_decay")),
        max_modules=int(_resolve_config(args, ckpt, "max_modules")),
        kg_nodes_file=str(_resolve_config(args, ckpt, "kg_nodes_file")),
        kg_edges_file=str(_resolve_config(args, ckpt, "kg_edges_file")),
        kg_graph_file=str(_resolve_config(args, ckpt, "kg_graph_file")),
    )

    eval_examples = load_msld_examples(
        prompts_file=args.prompts_file,
        labels_csv=args.labels_csv,
        graph=graph,
        split=args.split,
        default_cell_id=args.default_cell_id,
        require_label=True,
    )
    if not eval_examples:
        raise RuntimeError("No eval examples.")

    hidden_cache_path = str(_resolve_config(args, ckpt, "hidden_cache"))
    hidden_cache = ensure_hidden_cache(
        examples=eval_examples,
        model_name=str(_resolve_config(args, ckpt, "model_name")),
        max_input_tokens=int(_resolve_config(args, ckpt, "max_input_tokens")),
        cache_path=hidden_cache_path,
        bf16=bool(args.bf16),
        trust_remote_code=bool(args.trust_remote_code),
        log_every=args.log_every_hidden,
    )

    model_cfg = ckpt["model_config"]
    proposer = GraphOnlyModuleProposer(
        feature_dim=int(model_cfg["module_feature_dim"]),
        arch=str(model_cfg["proposer_arch"]),
        hidden_dim=int(model_cfg["proposer_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    gene_scorer = GraphLLMGeneScorer(
        graph_feature_dim=int(model_cfg["graph_gene_feature_dim"]),
        hidden_input_dim=int(model_cfg["hidden_input_dim"]),
        hidden_proj_dim=int(model_cfg["hidden_proj_dim"]),
        arch=str(model_cfg["gene_scorer_arch"]),
        scorer_hidden_dim=int(model_cfg["gene_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    model = GraphLLMHybridPredictor(proposer=proposer, gene_scorer=gene_scorer)
    model.load_state_dict(ckpt["predictor_state_dict"], strict=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = model.to(device)
    model.eval()

    perturbations = build_perturbations(
        eval_examples,
        min_genes_per_perturbation=int(_resolve_config(args, ckpt, "min_genes_per_perturbation")),
    )
    support_fraction = float(_resolve_config(args, ckpt, "support_fraction"))
    min_support_size = int(_resolve_config(args, ckpt, "min_support_size"))
    min_query_size = int(_resolve_config(args, ckpt, "min_query_size"))
    include_top_cov = bool(model_cfg.get("include_top_module_coverage", True))
    top_k_modules = int(model_cfg.get("gene_top_k_modules", 8))
    posterior_temperature = float(_resolve_config(args, ckpt, "posterior_temperature"))
    posterior_topk = int(_resolve_config(args, ckpt, "posterior_topk"))
    beta_prior_logit = float(_resolve_config(args, ckpt, "beta_prior_logit"))
    gamma_support_logit = float(_resolve_config(args, ckpt, "gamma_support_logit"))
    gene_feature_set = str(model_cfg["gene_feature_set"])
    proposer_feature_set = str(model_cfg["proposer_feature_set"])
    seed = int(args.seed if args.seed is not None else ckpt.get("args", {}).get("seed", 42))

    # lookup hidden by perturbation/gene_id
    by_pert_gene_hidden: Dict[str, Dict[int, np.ndarray]] = {}
    for ex in eval_examples:
        k = normalize_key(ex.pert, ex.gene)
        h = hidden_cache.get(k)
        if h is None:
            continue
        pkey = f"{ex.cell_id}::{ex.pert.strip().lower()}"
        by_pert_gene_hidden.setdefault(pkey, {})[int(ex.gene_id)] = np.asarray(h, dtype=np.float32)

    rows: List[Dict[str, object]] = []
    pert_summaries: List[Dict[str, object]] = []
    y_true: List[int] = []
    y_prob: List[float] = []
    skipped = 0
    threshold = float(args.decision_threshold)

    with torch.no_grad():
        for i, pert in enumerate(perturbations, 1):
            n = int(pert.gene_ids.shape[0])
            if n < (min_support_size + min_query_size):
                skipped += 1
                continue
            perm = np.random.default_rng(seed + i).permutation(n)
            s_n = max(min_support_size, int(round(support_fraction * n)))
            s_n = min(s_n, n - min_query_size)
            if s_n <= 0:
                skipped += 1
                continue
            s_idx = np.sort(perm[:s_n])
            q_idx = np.sort(perm[s_n:])
            s_gene_ids = pert.gene_ids[s_idx].astype(np.int64)
            s_labels = pert.labels[s_idx].astype(np.float32)
            q_gene_ids = pert.gene_ids[q_idx].astype(np.int64)
            q_labels = pert.labels[q_idx].astype(np.float32)

            mod_feat_np, _, mod_aux = build_module_feature_matrix(
                graph=graph,
                drug_id=int(pert.drug_id),
                support_gene_ids=s_gene_ids.tolist(),
                support_labels=s_labels.tolist(),
                proposer_feature_set=proposer_feature_set,
            )
            mod_out = model.infer_module_posterior(_to_torch(mod_feat_np, device=device))
            prior_t = _to_torch(mod_aux["module_prior"].astype(np.float32), device=device)
            contrast_t = _to_torch(mod_aux["contrast_score"].astype(np.float32), device=device)
            q_t = _posterior_from_logits(
                module_logits=mod_out["module_logits"],
                prior_probs=prior_t,
                support_contrast=contrast_t,
                beta_prior_logit=beta_prior_logit,
                gamma_support_logit=gamma_support_logit,
                temperature=posterior_temperature,
                topk=posterior_topk,
            )
            q_np = q_t.detach().cpu().numpy().astype(np.float32)

            graph_feat_t = _build_graph_gene_features_torch(
                graph=graph,
                query_gene_ids=q_gene_ids,
                q_t=q_t,
                target_module_distribution=mod_aux["target_module_distribution"],
                gene_feature_set=gene_feature_set,
                include_top_cov=include_top_cov,
                top_k_modules=top_k_modules,
                device=device,
            )
            hdict = by_pert_gene_hidden.get(pert.perturbation_key, {})
            hidden_list = []
            hidden_dim = int(model_cfg["hidden_input_dim"])
            for gid in q_gene_ids.tolist():
                h = hdict.get(int(gid))
                if h is None:
                    hidden_list.append(np.zeros((hidden_dim,), dtype=np.float32))
                else:
                    hidden_list.append(h)
            hidden_t = _to_torch(np.asarray(hidden_list, dtype=np.float32), device=device)

            logits = model.score_genes(graph_feat_t, hidden_t)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            preds = (probs >= threshold).astype(np.int64)

            for gid, yt, yp, pd in zip(q_gene_ids.tolist(), q_labels.tolist(), probs.tolist(), preds.tolist()):
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

            pert_summaries.append(
                {
                    "perturbation_key": pert.perturbation_key,
                    "pert": pert.pert,
                    "num_support": int(s_gene_ids.size),
                    "num_query": int(q_gene_ids.size),
                    "support_positive_rate": float(np.mean(s_labels)) if s_labels.size > 0 else 0.0,
                    "posterior_entropy": float(distribution_entropy(q_np)),
                    "top_modules": top_modules_from_posterior(
                        q_probs=q_np,
                        module_names=graph.idx_to_module,
                        module_types=graph.module_types,
                        top_k=args.top_modules_report,
                    ),
                }
            )

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, decision_threshold=threshold)
    summary = {
        "method": "graph_llm_hybrid_v1",
        "split": args.split,
        "num_eval_examples": len(eval_examples),
        "num_eval_perturbations": len(perturbations),
        "num_skipped_perturbations": int(skipped),
        "num_query_predictions": len(rows),
        "decision_threshold": threshold,
        "metrics": metrics,
        "args": vars(args),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "evaluation_summary.json")
    rows_path = os.path.join(args.out_dir, "evaluation_rows.csv")
    pert_path = os.path.join(args.out_dir, "perturbation_module_posteriors.jsonl")
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
        for item in pert_summaries:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"[GraphLLM-Eval] Saved summary: {summary_path}")
    print(f"[GraphLLM-Eval] Saved rows: {rows_path}")
    print(f"[GraphLLM-Eval] Saved perturbation posteriors: {pert_path}")
    print(
        f"[GraphLLM-Eval] acc={metrics['accuracy']} prec={metrics['precision']} rec={metrics['recall']} "
        f"f1={metrics['f1']} auroc={metrics['auroc']} auprc={metrics['auprc']} yes_rate={metrics['yes_prediction_rate']}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate graph+LLM-hidden hybrid baseline.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--kg-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--default-cell-id", type=int, default=0)

    p.add_argument("--graph-loader", default=None, choices=["ggmv", "msld"])
    p.add_argument("--graph-alpha", type=float, default=None)
    p.add_argument("--graph-diffusion-hops", type=int, default=None)
    p.add_argument("--graph-diffusion-decay", type=float, default=None)
    p.add_argument("--max-modules", type=int, default=None)
    p.add_argument("--kg-nodes-file", default=None)
    p.add_argument("--kg-edges-file", default=None)
    p.add_argument("--kg-graph-file", default=None)

    p.add_argument("--model-name", default=None)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--hidden-cache", default=None)
    p.add_argument("--posterior-temperature", type=float, default=None)
    p.add_argument("--posterior-topk", type=int, default=None)
    p.add_argument("--beta-prior-logit", type=float, default=None)
    p.add_argument("--gamma-support-logit", type=float, default=None)
    p.add_argument("--support-fraction", type=float, default=None)
    p.add_argument("--min-support-size", type=int, default=None)
    p.add_argument("--min-query-size", type=int, default=None)
    p.add_argument("--min-genes-per-perturbation", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--top-modules-report", type=int, default=8)
    p.add_argument("--log-every-hidden", type=int, default=200)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
