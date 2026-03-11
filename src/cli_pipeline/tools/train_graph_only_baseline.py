#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Baseline-2: graph-only module proposer + shallow gene scorer."""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from graph_only_baseline_model import (
    GraphOnlyGeneScorer,
    GraphOnlyModuleProposer,
    GraphOnlyPerturbationPredictor,
)
from graph_only_baseline_utils import (
    build_module_feature_matrix,
    build_perturbations,
    distribution_entropy,
    get_gene_feature_names,
    get_module_feature_names,
    load_graph_for_baseline,
    load_graph_only_examples,
    sample_episode_for_perturbation,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _build_gene_features_torch(
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
    if gids.size <= 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)

    # Constant graph proximity matrix for query genes: [N, M].
    prox_np = graph.r_tilde[gids].toarray().astype(np.float32)
    prox_t = _to_torch(prox_np, device=device)

    feats: List[torch.Tensor] = []
    mode = gene_feature_set.strip().lower()
    if mode not in {"alignment_only", "full"}:
        raise ValueError(f"Unsupported gene_feature_set: {gene_feature_set}")

    # Differentiable path: label loss -> gene scorer -> q_t -> proposer.
    module_alignment = prox_t @ q_t
    feats.append(module_alignment)

    if mode == "full":
        target_dist_t = _to_torch(np.asarray(target_module_distribution, dtype=np.float32), device=device)
        target_gene_prox = prox_t @ target_dist_t
        feats.append(target_gene_prox)

        if include_top_cov:
            with torch.no_grad():
                k = max(1, min(int(top_k_modules), int(q_t.shape[0])))
                top_idx = torch.topk(q_t.detach(), k=k).indices
            top_cov = (prox_t[:, top_idx] > 0.0).float().sum(dim=-1)
            feats.append(top_cov)

    return torch.stack(feats, dim=-1)


def run(args) -> None:
    _set_seed(args.seed)
    graph = load_graph_for_baseline(
        kg_dir=args.kg_dir,
        graph_loader=args.graph_loader,
        graph_alpha=args.graph_alpha,
        graph_diffusion_hops=args.graph_diffusion_hops,
        graph_diffusion_decay=args.graph_diffusion_decay,
        max_modules=args.max_modules,
        kg_nodes_file=args.kg_nodes_file,
        kg_edges_file=args.kg_edges_file,
        kg_graph_file=args.kg_graph_file,
    )
    if graph.num_modules <= 0:
        raise RuntimeError("No modules parsed from KG.")

    train_examples = load_graph_only_examples(
        labels_csv=args.labels_csv,
        graph=graph,
        split=args.train_split,
        default_cell_id=args.default_cell_id,
        allow_missing_drug=True,
    )
    if not train_examples:
        raise RuntimeError("No training examples loaded from labels CSV.")

    train_perturbations = build_perturbations(
        train_examples,
        min_genes_per_perturbation=args.min_genes_per_perturbation,
    )
    if not train_perturbations:
        raise RuntimeError("No perturbations survive min-gene filtering.")

    module_feature_names = get_module_feature_names(args.proposer_feature_set)
    include_top_cov = not args.disable_top_module_coverage
    gene_feature_names = get_gene_feature_names(
        gene_feature_set=args.gene_feature_set,
        include_top_module_coverage=include_top_cov,
    )

    proposer = GraphOnlyModuleProposer(
        feature_dim=len(module_feature_names),
        arch=args.proposer_arch,
        hidden_dim=args.proposer_hidden_dim,
        dropout=args.dropout,
    )
    gene_scorer = GraphOnlyGeneScorer(
        feature_dim=len(gene_feature_names),
        arch=args.gene_scorer_arch,
        hidden_dim=args.gene_hidden_dim,
        dropout=args.dropout,
    )
    model = GraphOnlyPerturbationPredictor(proposer=proposer, gene_scorer=gene_scorer)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state: Dict[str, torch.Tensor] = {}
    best_loss = float("inf")

    print(
        f"[GraphOnly-Train] device={device} modules={graph.num_modules} "
        f"train_examples={len(train_examples)} perturbations={len(train_perturbations)}"
    )
    print(
        f"[GraphOnly-Train] proposer={args.proposer_arch}/{args.proposer_feature_set} "
        f"gene_scorer={args.gene_scorer_arch}/{args.gene_feature_set} "
        f"episode_mode={args.episode_mode} support_fraction={args.support_fraction:.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_perturbations))

        total_loss = 0.0
        total_label = 0.0
        total_prior = 0.0
        total_sp = 0.0
        total_acc = 0.0
        total_q_entropy = 0.0
        trained_episodes = 0
        skipped_episodes = 0

        for step_i, order_idx in enumerate(order.tolist(), 1):
            pert = train_perturbations[int(order_idx)]
            if args.episode_mode == "full_perturbation":
                episode = {
                    "support_gene_ids": pert.gene_ids.astype(np.int64),
                    "support_labels": pert.labels.astype(np.float32),
                    "query_gene_ids": pert.gene_ids.astype(np.int64),
                    "query_labels": pert.labels.astype(np.float32),
                }
            else:
                episode = sample_episode_for_perturbation(
                    pert,
                    seed=args.seed,
                    epoch=epoch,
                    support_fraction=args.support_fraction,
                    min_support_size=args.min_support_size,
                    min_query_size=args.min_query_size,
                    max_support_size=args.max_support_size,
                )
                if episode is None:
                    skipped_episodes += 1
                    continue

            mod_feat_np, _, mod_aux = build_module_feature_matrix(
                graph=graph,
                drug_id=int(pert.drug_id),
                support_gene_ids=episode["support_gene_ids"].tolist(),
                support_labels=episode["support_labels"].tolist(),
                proposer_feature_set=args.proposer_feature_set,
            )
            if mod_feat_np.size == 0:
                skipped_episodes += 1
                continue

            mod_feat_t = _to_torch(mod_feat_np, device=device)
            out = model.infer_module_posterior(mod_feat_t)
            q_t = out["q"]  # [M]

            prior_np = mod_aux["module_prior"].astype(np.float32)
            prior_t = _to_torch(prior_np, device=device)

            gene_feat_t = _build_gene_features_torch(
                graph=graph,
                query_gene_ids=episode["query_gene_ids"],
                q_t=q_t,
                target_module_distribution=mod_aux["target_module_distribution"],
                gene_feature_set=args.gene_feature_set,
                include_top_cov=include_top_cov,
                top_k_modules=args.gene_top_k_modules,
                device=device,
            )
            if gene_feat_t.shape[0] <= 0:
                skipped_episodes += 1
                continue

            y_query_t = _to_torch(episode["query_labels"].astype(np.float32), device=device)
            gene_logits_t = model.score_genes(gene_feat_t)

            l_label = F.binary_cross_entropy_with_logits(gene_logits_t, y_query_t)
            l_prior = torch.sum(q_t * (torch.log(q_t.clamp_min(1e-8)) - torch.log(prior_t.clamp_min(1e-8))))
            l_sp = -torch.sum(q_t * torch.log(q_t.clamp_min(1e-8)))
            loss = l_label + args.lambda_prior * l_prior + args.lambda_sp * l_sp

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(gene_logits_t)
                pred = (probs >= 0.5).float()
                acc = float((pred == y_query_t).float().mean().detach().cpu())
                q_entropy = float(distribution_entropy(q_t.detach().cpu().numpy()))

            total_loss += float(loss.detach().cpu())
            total_label += float(l_label.detach().cpu())
            total_prior += float(l_prior.detach().cpu())
            total_sp += float(l_sp.detach().cpu())
            total_acc += acc
            total_q_entropy += q_entropy
            trained_episodes += 1

            if step_i % args.log_every_episodes == 0:
                print(
                    f"[GraphOnly-Train][Epoch {epoch}/{args.epochs}] "
                    f"step={step_i} episodes={trained_episodes} skipped={skipped_episodes} "
                    f"loss={total_loss/max(1,trained_episodes):.6f}"
                )

        if trained_episodes <= 0:
            raise RuntimeError("No trainable episodes in this epoch; relax split constraints.")

        mean_loss = total_loss / trained_episodes
        mean_label = total_label / trained_episodes
        mean_prior = total_prior / trained_episodes
        mean_sp = total_sp / trained_episodes
        mean_acc = total_acc / trained_episodes
        mean_q_ent = total_q_entropy / trained_episodes
        print(
            f"[GraphOnly-Train][Epoch {epoch}/{args.epochs}] "
            f"loss={mean_loss:.6f} label={mean_label:.6f} prior={mean_prior:.6f} "
            f"entropy={mean_sp:.6f} query_acc={mean_acc:.6f} q_entropy={mean_q_ent:.6f} "
            f"episodes={trained_episodes} skipped={skipped_episodes}"
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state, strict=True)
    model.eval()

    ckpt = {
        "version": "graph_only_baseline_v1",
        "predictor_state_dict": model.state_dict(),
        "model_config": {
            "num_modules": int(graph.num_modules),
            "module_feature_dim": len(module_feature_names),
            "gene_feature_dim": len(gene_feature_names),
            "proposer_arch": args.proposer_arch,
            "proposer_hidden_dim": int(args.proposer_hidden_dim),
            "gene_scorer_arch": args.gene_scorer_arch,
            "gene_hidden_dim": int(args.gene_hidden_dim),
            "dropout": float(args.dropout),
            "proposer_feature_set": args.proposer_feature_set,
            "gene_feature_set": args.gene_feature_set,
            "include_top_module_coverage": bool(include_top_cov),
            "gene_top_k_modules": int(args.gene_top_k_modules),
        },
        "loss_config": {
            "lambda_prior": float(args.lambda_prior),
            "lambda_sp": float(args.lambda_sp),
        },
        "episode_config": {
            "episode_mode": args.episode_mode,
            "support_fraction": float(args.support_fraction),
            "min_support_size": int(args.min_support_size),
            "min_query_size": int(args.min_query_size),
            "max_support_size": int(args.max_support_size),
            "min_genes_per_perturbation": int(args.min_genes_per_perturbation),
        },
        "graph_config": {
            "graph_loader": args.graph_loader,
            "graph_alpha": float(args.graph_alpha),
            "graph_diffusion_hops": int(args.graph_diffusion_hops),
            "graph_diffusion_decay": float(args.graph_diffusion_decay),
            "max_modules": int(args.max_modules),
            "kg_nodes_file": args.kg_nodes_file,
            "kg_edges_file": args.kg_edges_file,
            "kg_graph_file": args.kg_graph_file,
        },
        "module_feature_names": module_feature_names,
        "gene_feature_names": gene_feature_names,
        "module_names": list(graph.idx_to_module),
        "module_types": list(graph.module_types),
        "train_summary": {
            "train_split": args.train_split,
            "num_train_examples": len(train_examples),
            "num_train_perturbations": len(train_perturbations),
            "num_modules": int(graph.num_modules),
            "best_loss": float(best_loss),
        },
        "args": vars(args),
    }
    out_dir = os.path.dirname(args.out_ckpt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, args.out_ckpt)
    print(f"[GraphOnly-Train] Saved checkpoint: {args.out_ckpt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train graph-only baseline (module proposer + shallow gene scorer).")
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--kg-dir", required=True)
    p.add_argument("--out-ckpt", required=True)

    p.add_argument("--train-split", default="train")
    p.add_argument("--default-cell-id", type=int, default=0)

    p.add_argument("--graph-loader", default="ggmv", choices=["ggmv", "msld"])
    p.add_argument("--kg-nodes-file", default="nodes.json")
    p.add_argument("--kg-edges-file", default="edges.json")
    p.add_argument("--kg-graph-file", default="graph.json")
    p.add_argument("--graph-alpha", type=float, default=0.1)
    p.add_argument("--graph-diffusion-hops", type=int, default=1)
    p.add_argument("--graph-diffusion-decay", type=float, default=1.0)
    p.add_argument("--max-modules", type=int, default=2048)

    p.add_argument("--min-genes-per-perturbation", type=int, default=8)
    p.add_argument(
        "--episode-mode",
        default="full_perturbation",
        choices=["full_perturbation", "intra_perturbation"],
        help="full_perturbation: use all genes as support/query; intra_perturbation: split support/query within pert.",
    )
    p.add_argument("--support-fraction", type=float, default=0.2)
    p.add_argument("--min-support-size", type=int, default=4)
    p.add_argument("--min-query-size", type=int, default=4)
    p.add_argument("--max-support-size", type=int, default=64)

    p.add_argument("--proposer-feature-set", default="full", choices=["target_only", "support_only", "full"])
    p.add_argument("--proposer-arch", default="linear", choices=["linear", "mlp"])
    p.add_argument("--proposer-hidden-dim", type=int, default=64)

    p.add_argument("--gene-feature-set", default="full", choices=["alignment_only", "full"])
    p.add_argument("--gene-scorer-arch", default="linear", choices=["linear", "mlp"])
    p.add_argument("--gene-hidden-dim", type=int, default=32)
    p.add_argument("--gene-top-k-modules", type=int, default=8)
    p.add_argument("--disable-top-module-coverage", action="store_true")

    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-prior", type=float, default=0.1)
    p.add_argument("--lambda-sp", type=float, default=0.01)
    p.add_argument("--log-every-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
