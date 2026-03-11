#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train graph + LLM-hidden hybrid baseline."""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from graph_llm_hybrid_model import GraphLLMGeneScorer, GraphLLMHybridPredictor, GraphOnlyModuleProposer
from graph_llm_hybrid_utils import ensure_hidden_cache, normalize_key
from graph_only_baseline_utils import (
    build_module_feature_matrix,
    build_perturbations,
    distribution_entropy,
    get_gene_feature_names,
    get_module_feature_names,
    load_graph_for_baseline,
)
from msld_data import load_msld_examples


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


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
    prox = _to_torch(graph.r_tilde[gids].toarray().astype(np.float32), device=device)  # [N, M]
    feats: List[torch.Tensor] = []
    mode = gene_feature_set.strip().lower()
    if mode not in {"alignment_only", "full"}:
        raise ValueError(f"Unsupported gene_feature_set: {gene_feature_set}")
    feats.append(prox @ q_t)
    if mode == "full":
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
    temperature: float,
    topk: int,
) -> torch.Tensor:
    logits = module_logits.reshape(-1)
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
    train_examples = load_msld_examples(
        prompts_file=args.prompts_file,
        labels_csv=args.labels_csv,
        graph=graph,
        split=args.train_split,
        default_cell_id=args.default_cell_id,
        require_label=True,
    )
    if not train_examples:
        raise RuntimeError("No train examples after prompts+labels+graph filtering.")

    hidden_cache = ensure_hidden_cache(
        examples=train_examples,
        model_name=args.model_name,
        max_input_tokens=args.max_input_tokens,
        cache_path=args.hidden_cache,
        bf16=args.bf16,
        trust_remote_code=args.trust_remote_code,
        log_every=args.log_every_hidden,
    )
    hidden_dim = int(next(iter(hidden_cache.values())).shape[0])

    perturbations = build_perturbations(train_examples, min_genes_per_perturbation=args.min_genes_per_perturbation)
    if not perturbations:
        raise RuntimeError("No perturbations survive min-gene filtering.")

    ex_key_to_hidden: Dict[str, np.ndarray] = {}
    for ex in train_examples:
        key = normalize_key(ex.pert, ex.gene)
        if key in hidden_cache:
            ex_key_to_hidden[key] = hidden_cache[key]

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
    gene_scorer = GraphLLMGeneScorer(
        graph_feature_dim=len(gene_feature_names),
        hidden_input_dim=hidden_dim,
        hidden_proj_dim=args.hidden_proj_dim,
        arch=args.gene_scorer_arch,
        scorer_hidden_dim=args.gene_hidden_dim,
        dropout=args.dropout,
    )
    model = GraphLLMHybridPredictor(proposer=proposer, gene_scorer=gene_scorer)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_t = torch.tensor([float(args.pos_weight)], dtype=torch.float32, device=device)

    print(
        f"[GraphLLM-Train] device={device} modules={graph.num_modules} "
        f"train_examples={len(train_examples)} perturbations={len(perturbations)} hidden_dim={hidden_dim}"
    )
    print(
        f"[GraphLLM-Train] posterior_temperature={args.posterior_temperature:.4f} "
        f"posterior_topk={int(args.posterior_topk)} lambda_sp={args.lambda_sp:.4f} "
        f"lambda_entropy_hinge={args.lambda_entropy_hinge:.4f} target_entropy={args.target_entropy:.4f}"
    )

    # Build quick lookup: perturbation -> (gene_id -> (label, hidden))
    by_pert: Dict[str, Dict[int, Dict[str, object]]] = {}
    for ex in train_examples:
        pkey = f"{ex.cell_id}::{ex.pert.strip().lower()}"
        key = normalize_key(ex.pert, ex.gene)
        h = ex_key_to_hidden.get(key)
        if h is None:
            continue
        by_pert.setdefault(pkey, {})[int(ex.gene_id)] = {"label": int(ex.label), "hidden": h}

    best_state: Dict[str, torch.Tensor] = {}
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(perturbations))

        total_loss = 0.0
        total_label = 0.0
        total_prior = 0.0
        total_sp = 0.0
        total_sp_hinge = 0.0
        total_acc = 0.0
        total_yes = 0.0
        total_q_ent = 0.0
        n_ep = 0
        skipped = 0

        for idx in order.tolist():
            pert = perturbations[int(idx)]
            pdict = by_pert.get(pert.perturbation_key, {})
            if not pdict:
                skipped += 1
                continue

            # strict support->query split within perturbation for training objective
            n = int(pert.gene_ids.shape[0])
            if n < (args.min_support_size + args.min_query_size):
                skipped += 1
                continue
            perm = np.random.default_rng(args.seed + epoch + idx).permutation(n)
            s_n = max(args.min_support_size, int(round(args.support_fraction * n)))
            s_n = min(s_n, n - args.min_query_size)
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
                proposer_feature_set=args.proposer_feature_set,
            )
            mod_feat_t = _to_torch(mod_feat_np, device=device)
            mod_out = model.infer_module_posterior(mod_feat_t)
            q_t = _posterior_from_logits(
                module_logits=mod_out["module_logits"],
                temperature=args.posterior_temperature,
                topk=args.posterior_topk,
            )

            graph_feat_t = _build_graph_gene_features_torch(
                graph=graph,
                query_gene_ids=q_gene_ids,
                q_t=q_t,
                target_module_distribution=mod_aux["target_module_distribution"],
                gene_feature_set=args.gene_feature_set,
                include_top_cov=include_top_cov,
                top_k_modules=args.gene_top_k_modules,
                device=device,
            )
            hidden_np_list = []
            for gid in q_gene_ids.tolist():
                item = pdict.get(int(gid))
                if item is None:
                    hidden_np_list.append(np.zeros((hidden_dim,), dtype=np.float32))
                else:
                    hidden_np_list.append(np.asarray(item["hidden"], dtype=np.float32))
            hidden_t = _to_torch(np.asarray(hidden_np_list, dtype=np.float32), device=device)
            y_t = _to_torch(q_labels.astype(np.float32), device=device)

            logits_t = model.score_genes(graph_feat_t, hidden_t)
            prior_t = _to_torch(mod_aux["module_prior"].astype(np.float32), device=device)

            l_label = F.binary_cross_entropy_with_logits(logits_t, y_t, pos_weight=pos_weight_t)
            l_prior = torch.sum(q_t * (torch.log(q_t.clamp_min(1e-8)) - torch.log(prior_t.clamp_min(1e-8))))
            l_sp = -torch.sum(q_t * torch.log(q_t.clamp_min(1e-8)))
            l_sp_hinge = torch.relu(l_sp - float(args.target_entropy))
            loss = (
                l_label
                + args.lambda_prior * l_prior
                + args.lambda_sp * l_sp
                + args.lambda_entropy_hinge * l_sp_hinge
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                p = torch.sigmoid(logits_t)
                pred = (p >= 0.5).float()
                total_acc += float((pred == y_t).float().mean().detach().cpu())
                total_yes += float(pred.mean().detach().cpu())
                total_q_ent += float(distribution_entropy(q_t.detach().cpu().numpy()))

            total_loss += float(loss.detach().cpu())
            total_label += float(l_label.detach().cpu())
            total_prior += float(l_prior.detach().cpu())
            total_sp += float(l_sp.detach().cpu())
            total_sp_hinge += float(l_sp_hinge.detach().cpu())
            n_ep += 1

        if n_ep <= 0:
            raise RuntimeError("No episodes trained. Check data/filter settings.")
        mean_loss = total_loss / n_ep
        print(
            f"[GraphLLM-Train][Epoch {epoch}/{args.epochs}] loss={mean_loss:.6f} "
            f"label={total_label/n_ep:.6f} prior={total_prior/n_ep:.6f} "
            f"entropy={total_sp/n_ep:.6f} entropy_hinge={total_sp_hinge/n_ep:.6f} "
            f"query_acc={total_acc/n_ep:.6f} yes_rate={total_yes/n_ep:.6f} q_entropy={total_q_ent/n_ep:.6f} "
            f"episodes={n_ep} skipped={skipped}"
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state, strict=True)
    ckpt = {
        "version": "graph_llm_hybrid_v1",
        "predictor_state_dict": model.state_dict(),
        "model_config": {
            "num_modules": int(graph.num_modules),
            "module_feature_dim": len(module_feature_names),
            "graph_gene_feature_dim": len(gene_feature_names),
            "hidden_input_dim": int(hidden_dim),
            "hidden_proj_dim": int(args.hidden_proj_dim),
            "proposer_arch": args.proposer_arch,
            "proposer_hidden_dim": int(args.proposer_hidden_dim),
            "gene_scorer_arch": args.gene_scorer_arch,
            "gene_hidden_dim": int(args.gene_hidden_dim),
            "dropout": float(args.dropout),
            "proposer_feature_set": args.proposer_feature_set,
            "gene_feature_set": args.gene_feature_set,
            "include_top_module_coverage": bool(include_top_cov),
            "gene_top_k_modules": int(args.gene_top_k_modules),
            "posterior_temperature": float(args.posterior_temperature),
            "posterior_topk": int(args.posterior_topk),
        },
        "loss_config": {
            "lambda_prior": float(args.lambda_prior),
            "lambda_sp": float(args.lambda_sp),
            "lambda_entropy_hinge": float(args.lambda_entropy_hinge),
            "target_entropy": float(args.target_entropy),
            "pos_weight": float(args.pos_weight),
        },
        "episode_config": {
            "support_fraction": float(args.support_fraction),
            "min_support_size": int(args.min_support_size),
            "min_query_size": int(args.min_query_size),
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
        "llm_config": {
            "model_name": args.model_name,
            "max_input_tokens": int(args.max_input_tokens),
            "hidden_cache": args.hidden_cache,
        },
        "module_feature_names": module_feature_names,
        "gene_feature_names": gene_feature_names,
        "module_names": list(graph.idx_to_module),
        "module_types": list(graph.module_types),
        "args": vars(args),
    }
    out_dir = os.path.dirname(args.out_ckpt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, args.out_ckpt)
    print(f"[GraphLLM-Train] Saved checkpoint: {args.out_ckpt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train graph+LLM-hidden hybrid baseline.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--kg-dir", required=True)
    p.add_argument("--out-ckpt", required=True)
    p.add_argument("--hidden-cache", required=True)

    p.add_argument("--train-split", default="train")
    p.add_argument("--default-cell-id", type=int, default=0)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--log-every-hidden", type=int, default=200)

    p.add_argument("--graph-loader", default="ggmv", choices=["ggmv", "msld"])
    p.add_argument("--kg-nodes-file", default="nodes.json")
    p.add_argument("--kg-edges-file", default="edges.json")
    p.add_argument("--kg-graph-file", default="graph.json")
    p.add_argument("--graph-alpha", type=float, default=0.1)
    p.add_argument("--graph-diffusion-hops", type=int, default=1)
    p.add_argument("--graph-diffusion-decay", type=float, default=1.0)
    p.add_argument("--max-modules", type=int, default=2048)
    p.add_argument("--min-genes-per-perturbation", type=int, default=8)

    p.add_argument("--support-fraction", type=float, default=0.4)
    p.add_argument("--min-support-size", type=int, default=8)
    p.add_argument("--min-query-size", type=int, default=8)

    p.add_argument("--proposer-feature-set", default="full", choices=["target_only", "support_only", "full"])
    p.add_argument("--proposer-arch", default="mlp", choices=["linear", "mlp"])
    p.add_argument("--proposer-hidden-dim", type=int, default=128)
    p.add_argument("--gene-feature-set", default="full", choices=["alignment_only", "full"])
    p.add_argument("--gene-scorer-arch", default="linear", choices=["linear", "mlp"])
    p.add_argument("--gene-hidden-dim", type=int, default=64)
    p.add_argument("--hidden-proj-dim", type=int, default=64)
    p.add_argument("--gene-top-k-modules", type=int, default=8)
    p.add_argument("--disable-top-module-coverage", action="store_true")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-prior", type=float, default=0.1)
    p.add_argument("--lambda-sp", type=float, default=0.01)
    p.add_argument(
        "--lambda-entropy-hinge",
        type=float,
        default=0.0,
        help="Extra sparsity term: max(0, H(q)-target_entropy).",
    )
    p.add_argument(
        "--target-entropy",
        type=float,
        default=4.0,
        help="Target entropy used by entropy-hinge regularization.",
    )
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--posterior-temperature", type=float, default=1.0)
    p.add_argument("--posterior-topk", type=int, default=0, help="0 means full softmax; >0 keeps only top-k modules.")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
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
