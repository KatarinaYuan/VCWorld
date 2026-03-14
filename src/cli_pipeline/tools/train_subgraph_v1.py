#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train V1: heuristic proposer + fast scorer + frozen verifier rerank."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

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
    CandidateBatchLoss,
    CandidateSubgraph,
    FastScorer,
    FrozenLLMVerifier,
    HeuristicPathSeededPCSTProposer,
    build_candidate_feature_vector,
    build_gene_feature_matrix_for_candidate,
    pairwise_ranking_loss,
    summarize_candidate_score_components,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _module_preview(module_names: Sequence[str], module_ids: Sequence[int], max_items: int) -> List[str]:
    out: List[str] = []
    for mid in list(module_ids)[: max(1, int(max_items))]:
        if 0 <= int(mid) < len(module_names):
            out.append(str(module_names[int(mid)]))
        else:
            out.append(f"module_{int(mid)}")
    return out


def _should_log_episode(args, episode_idx: int) -> bool:
    if episode_idx <= int(args.debug_episodes_per_epoch):
        return True
    every = max(1, int(args.log_every_episodes))
    return (episode_idx % every) == 0


def _append_jsonl(path: str, payload: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _episode_candidates(
    *,
    graph,
    proposer: HeuristicPathSeededPCSTProposer,
    drug_id: int,
    support_gene_ids: np.ndarray,
    support_labels: np.ndarray,
    num_candidates: int,
    min_nodes: int,
    max_nodes: int,
) -> Tuple[Dict[str, np.ndarray], List[CandidateSubgraph]]:
    _, _, mod_aux = build_module_feature_matrix(
        graph=graph,
        drug_id=int(drug_id),
        support_gene_ids=support_gene_ids.tolist(),
        support_labels=support_labels.tolist(),
        proposer_feature_set="full",
    )
    candidates = proposer.generate(
        mod_aux=mod_aux,
        num_candidates=int(num_candidates),
        min_nodes=int(min_nodes),
        max_nodes=int(max_nodes),
    )
    return mod_aux, candidates


def _eval_split(
    *,
    graph,
    scorer: FastScorer,
    proposer: HeuristicPathSeededPCSTProposer,
    verifier: FrozenLLMVerifier,
    perturbations,
    seed: int,
    support_fraction: float,
    min_support_size: int,
    min_query_size: int,
    max_support_size: int,
    num_candidates: int,
    top_k_verifier: int,
    min_candidate_nodes: int,
    max_candidate_nodes: int,
    beta_replay: float,
    gamma_llm: float,
    lambda_size: float,
    device: torch.device,
) -> Dict[str, object]:
    scorer.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    skipped = 0

    with torch.no_grad():
        for i, pert in enumerate(perturbations, 1):
            ep = sample_episode_for_perturbation(
                pert,
                seed=int(seed),
                epoch=0,
                support_fraction=float(support_fraction),
                min_support_size=int(min_support_size),
                min_query_size=int(min_query_size),
                max_support_size=int(max_support_size),
            )
            if ep is None:
                skipped += 1
                continue

            mod_aux, candidates = _episode_candidates(
                graph=graph,
                proposer=proposer,
                drug_id=int(pert.drug_id),
                support_gene_ids=ep["support_gene_ids"],
                support_labels=ep["support_labels"],
                num_candidates=int(num_candidates),
                min_nodes=int(min_candidate_nodes),
                max_nodes=int(max_candidate_nodes),
            )
            if not candidates:
                skipped += 1
                continue

            quality_pred_np: List[float] = []
            query_probs_list: List[np.ndarray] = []
            support_probs_list: List[np.ndarray] = []
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

                q_probs = torch.sigmoid(q_logits).detach().cpu().numpy().astype(np.float32)
                s_probs = torch.sigmoid(s_logits).detach().cpu().numpy().astype(np.float32)
                if s_probs.size == ep["support_labels"].size and s_probs.size > 0:
                    replay = float(np.mean((s_probs >= 0.5).astype(np.float32) == ep["support_labels"]))
                else:
                    replay = 0.0

                quality_pred_np.append(float(q_pred.detach().cpu()))
                query_probs_list.append(q_probs)
                support_probs_list.append(s_probs)
                replay_list.append(replay)

            order = np.argsort(np.asarray(quality_pred_np, dtype=np.float32))[::-1]
            topk_idx = order[: max(1, min(int(top_k_verifier), len(candidates)))].tolist()

            llm_score = np.zeros((len(candidates),), dtype=np.float32)
            for idx in topk_idx:
                out = verifier.verify_candidate(
                    candidate=candidates[idx],
                    support_labels=ep["support_labels"],
                    support_probs=support_probs_list[idx],
                    module_names=graph.idx_to_module,
                )
                llm_score[idx] = float(out["confidence"])

            final_score = np.asarray(quality_pred_np, dtype=np.float32)
            final_score = final_score + float(beta_replay) * np.asarray(replay_list, dtype=np.float32)
            final_score = final_score + float(gamma_llm) * llm_score
            final_score = final_score - float(lambda_size) * np.asarray([c.size for c in candidates], dtype=np.float32)
            best_idx = int(np.argmax(final_score))

            q_probs = query_probs_list[best_idx]
            for yt, yp in zip(ep["query_labels"].tolist(), q_probs.tolist()):
                y_true.append(int(yt))
                y_prob.append(float(yp))

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, decision_threshold=0.5)
    return {
        "metrics": metrics,
        "num_perturbations": len(perturbations),
        "num_skipped": int(skipped),
        "num_query_predictions": len(y_true),
    }


def run(args) -> None:
    _set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

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

    train_examples = load_graph_only_examples(
        labels_csv=args.labels_csv,
        graph=graph,
        split=args.train_split,
        default_cell_id=args.default_cell_id,
        allow_missing_drug=True,
    )
    val_examples = load_graph_only_examples(
        labels_csv=args.labels_csv,
        graph=graph,
        split=args.val_split,
        default_cell_id=args.default_cell_id,
        allow_missing_drug=True,
    )
    train_perturbations = build_perturbations(
        train_examples,
        min_genes_per_perturbation=args.min_genes_per_perturbation,
    )
    val_perturbations = build_perturbations(
        val_examples,
        min_genes_per_perturbation=args.min_genes_per_perturbation,
    )
    if not train_perturbations:
        raise RuntimeError("No train perturbations after filtering.")
    if not val_perturbations:
        raise RuntimeError("No val perturbations after filtering.")

    proposer = HeuristicPathSeededPCSTProposer(
        graph=graph,
        max_neighbors=args.max_neighbors,
        rng_seed=args.seed,
        a1_near_drug=args.a1_near_drug,
        a2_cover_pos=args.a2_cover_pos,
        a3_cover_neg=args.a3_cover_neg,
        a4_type_prior=args.a4_type_prior,
        b1_confidence=args.b1_confidence,
        b2_type_penalty=args.b2_type_penalty,
    )
    scorer = FastScorer(
        candidate_feature_dim=12,
        gene_feature_dim=4,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    verifier = FrozenLLMVerifier(replay_weight=0.5, grounding_weight=0.5)

    loss_fn = CandidateBatchLoss(pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_auprc = float("-inf")
    best_epoch = 0

    print(
        f"[SubgraphV1][Train] device={device} train_perts={len(train_perturbations)} "
        f"val_perts={len(val_perturbations)} modules={graph.num_modules}"
    )
    if args.print_candidate_details:
        print(
            f"[SubgraphV1][Train] candidate_debug=on log_every={args.log_every_episodes} "
            f"first_n={args.debug_episodes_per_epoch} top_candidates={args.debug_top_candidates}"
        )
    if args.debug_log_jsonl:
        dbg_dir = os.path.dirname(args.debug_log_jsonl)
        if dbg_dir:
            os.makedirs(dbg_dir, exist_ok=True)
        with open(args.debug_log_jsonl, "w", encoding="utf-8") as f:
            f.write("")
        print(f"[SubgraphV1][Train] debug_log_jsonl={args.debug_log_jsonl}")

    for epoch in range(1, args.epochs + 1):
        scorer.train()
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(train_perturbations))

        total_loss = 0.0
        total_gene = 0.0
        total_qual = 0.0
        total_rank = 0.0
        n_ep = 0
        skipped = 0

        for idx in order.tolist():
            pert = train_perturbations[int(idx)]
            ep = sample_episode_for_perturbation(
                pert,
                seed=args.seed,
                epoch=epoch,
                support_fraction=args.support_fraction,
                min_support_size=args.min_support_size,
                min_query_size=args.min_query_size,
                max_support_size=args.max_support_size,
            )
            if ep is None:
                skipped += 1
                continue

            mod_aux, candidates = _episode_candidates(
                graph=graph,
                proposer=proposer,
                drug_id=int(pert.drug_id),
                support_gene_ids=ep["support_gene_ids"],
                support_labels=ep["support_labels"],
                num_candidates=args.num_candidates,
                min_nodes=args.min_candidate_nodes,
                max_nodes=args.max_candidate_nodes,
            )
            if not candidates:
                skipped += 1
                continue

            bce_terms: List[torch.Tensor] = []
            quality_preds: List[torch.Tensor] = []
            reward_terms: List[float] = []
            support_probs_list: List[np.ndarray] = []

            q_label_t = _to_torch(ep["query_labels"], device)
            s_label_np = ep["support_labels"].astype(np.float32)
            proposer_components = proposer.score_components(mod_aux)
            bce_values: List[float] = []

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

                bce_q = loss_fn.bce_logits(q_logits, q_label_t)
                bce_terms.append(bce_q)
                bce_values.append(float(bce_q.detach().cpu()))
                quality_preds.append(q_pred)

                s_prob = torch.sigmoid(s_logits).detach().cpu().numpy().astype(np.float32)
                support_probs_list.append(s_prob)
                replay = float(np.mean((s_prob >= 0.5).astype(np.float32) == s_label_np)) if s_prob.size == s_label_np.size and s_prob.size > 0 else 0.0

                reward = -float(bce_q.detach().cpu()) + float(args.beta_replay) * replay - float(args.lambda_size) * float(cand.size)
                reward_terms.append(reward)

            qual_np = np.asarray([float(x.detach().cpu()) for x in quality_preds], dtype=np.float32)
            topk = np.argsort(qual_np)[::-1][: max(1, min(args.top_k_verifier, len(candidates)))].tolist()
            llm_bonus = np.zeros((len(candidates),), dtype=np.float32)
            for ci in topk:
                out = verifier.verify_candidate(
                    candidate=candidates[ci],
                    support_labels=s_label_np,
                    support_probs=support_probs_list[ci],
                    module_names=graph.idx_to_module,
                )
                bonus = float(args.gamma_llm) * float(out["confidence"])
                llm_bonus[ci] = bonus
                reward_terms[ci] += bonus

            quality_t = torch.stack(quality_preds, dim=0)
            reward_t = _to_torch(np.asarray(reward_terms, dtype=np.float32), device)

            l_gene = torch.stack(bce_terms).mean()
            l_qual = loss_fn.quality_mse(quality_t, reward_t)
            l_rank = pairwise_ranking_loss(quality_t, reward_t, margin=args.rank_margin)
            loss = l_gene + float(args.eta_qual) * l_qual + float(args.eta_rank) * l_rank

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_gene += float(l_gene.detach().cpu())
            total_qual += float(l_qual.detach().cpu())
            total_rank += float(l_rank.detach().cpu())
            n_ep += 1

            if args.print_candidate_details or args.debug_log_jsonl:
                episode_idx = int(n_ep)
                do_sparse_log = _should_log_episode(args, episode_idx=episode_idx)
                do_print = bool(args.print_candidate_details and do_sparse_log)
                do_file = bool(args.debug_log_jsonl and (args.debug_log_all_episodes or do_sparse_log))
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
                                "quality_pred": float(qual_np[ci]),
                                "query_bce": float(bce_values[ci]),
                                "replay_score": float(
                                    np.mean((support_probs_list[ci] >= 0.5).astype(np.float32) == s_label_np)
                                )
                                if support_probs_list[ci].size == s_label_np.size and s_label_np.size > 0
                                else 0.0,
                                "llm_bonus": float(llm_bonus[ci]),
                                "size_penalty": float(args.lambda_size) * float(cand.size),
                                "final_reward": float(reward_terms[ci]),
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

                    by_reward = sorted(candidate_details, key=lambda x: float(x["final_reward"]), reverse=True)
                    by_quality = sorted(candidate_details, key=lambda x: float(x["quality_pred"]), reverse=True)
                    best = by_reward[0] if by_reward else {}
                    top_prize_mid = np.argsort(np.asarray(proposer_components["prize"], dtype=np.float32))[::-1][
                        : max(1, int(args.debug_top_modules))
                    ].tolist()
                    top_prize_names = _module_preview(
                        graph.idx_to_module,
                        top_prize_mid,
                        max_items=args.debug_top_modules,
                    )

                    if do_print:
                        print(
                            f"[SubgraphV1][Train][Epoch {epoch}][Ep {episode_idx}] "
                            f"pert={pert.pert} support={int(ep['support_gene_ids'].size)} query={int(ep['query_gene_ids'].size)} "
                            f"cand={len(candidates)} best_idx={best.get('candidate_index')} "
                            f"best_reward={best.get('final_reward')} best_q={best.get('quality_pred')} "
                            f"best_replay={best.get('replay_score')} best_llm_bonus={best.get('llm_bonus')}"
                        )
                        print(
                            f"[SubgraphV1][Train][Epoch {epoch}][Ep {episode_idx}] "
                            f"proposer_top_prize_modules={top_prize_names}"
                        )
                        for item in by_reward[: max(1, int(args.debug_top_candidates))]:
                            print(
                                f"[SubgraphV1][Train][Epoch {epoch}][Ep {episode_idx}][Cand {item['candidate_index']}] "
                                f"reward={item['final_reward']:.4f} q={item['quality_pred']:.4f} bce={item['query_bce']:.4f} "
                                f"replay={item['replay_score']:.4f} llm_bonus={item['llm_bonus']:.4f} "
                                f"mean_prize={item['mean_prize']:.4f} mean_target={item['mean_target']:.4f} "
                                f"size={item['candidate_size']} edges={item['candidate_edges']} modules={item['modules']}"
                            )
                        print(
                            f"[SubgraphV1][Train][Epoch {epoch}][Ep {episode_idx}] "
                            f"top_by_quality={[int(x['candidate_index']) for x in by_quality[: max(1, int(args.debug_top_candidates))]]} "
                            f"top_by_reward={[int(x['candidate_index']) for x in by_reward[: max(1, int(args.debug_top_candidates))]]}"
                        )

                    if do_file:
                        payload: Dict[str, object] = {
                            "stage": "train",
                            "epoch": int(epoch),
                            "episode_index": int(episode_idx),
                            "perturbation_key": str(pert.perturbation_key),
                            "pert": str(pert.pert),
                            "drug_id": int(pert.drug_id),
                            "num_support": int(ep["support_gene_ids"].size),
                            "num_query": int(ep["query_gene_ids"].size),
                            "num_candidates": int(len(candidates)),
                            "top_by_quality": [int(x["candidate_index"]) for x in by_quality],
                            "top_by_reward": [int(x["candidate_index"]) for x in by_reward],
                            "proposer_top_prize_modules": top_prize_names,
                            "candidates": candidate_details,
                        }
                        _append_jsonl(args.debug_log_jsonl, payload)

        if n_ep <= 0:
            raise RuntimeError("No train episodes produced; relax episode constraints.")

        val_out = _eval_split(
            graph=graph,
            scorer=scorer,
            proposer=proposer,
            verifier=verifier,
            perturbations=val_perturbations,
            seed=args.seed,
            support_fraction=args.support_fraction,
            min_support_size=args.min_support_size,
            min_query_size=args.min_query_size,
            max_support_size=args.max_support_size,
            num_candidates=args.num_candidates,
            top_k_verifier=args.top_k_verifier,
            min_candidate_nodes=args.min_candidate_nodes,
            max_candidate_nodes=args.max_candidate_nodes,
            beta_replay=args.beta_replay,
            gamma_llm=args.gamma_llm,
            lambda_size=args.lambda_size,
            device=device,
        )
        val_auprc = val_out["metrics"].get("auprc")
        val_auprc_float = float(val_auprc) if val_auprc is not None else float("-inf")

        print(
            f"[SubgraphV1][Epoch {epoch}/{args.epochs}] "
            f"loss={total_loss/n_ep:.6f} gene={total_gene/n_ep:.6f} qual={total_qual/n_ep:.6f} rank={total_rank/n_ep:.6f} "
            f"train_ep={n_ep} train_skipped={skipped} "
            f"val_auprc={val_auprc_float:.6f} val_auroc={val_out['metrics'].get('auroc')}"
        )

        if val_auprc_float > best_val_auprc + float(args.save_eps):
            best_val_auprc = val_auprc_float
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}

    scorer.load_state_dict(best_state, strict=True)
    ckpt = {
        "version": "subgraph_v1",
        "scorer_state_dict": scorer.state_dict(),
        "model_config": {
            "candidate_feature_dim": 12,
            "gene_feature_dim": 4,
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
        },
        "train_config": vars(args),
        "best_val_auprc": float(best_val_auprc),
        "best_epoch": int(best_epoch),
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
    }
    out_dir = os.path.dirname(args.out_ckpt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, args.out_ckpt)

    report = {
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_val_auprc),
        "out_ckpt": args.out_ckpt,
    }
    if args.out_report:
        rep_dir = os.path.dirname(args.out_report)
        if rep_dir:
            os.makedirs(rep_dir, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)
    print(f"[SubgraphV1][Train] Saved checkpoint: {args.out_ckpt}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train V1 support-conditioned subgraph pipeline.")
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--kg-dir", required=True)
    p.add_argument("--out-ckpt", required=True)
    p.add_argument("--out-report", default="")

    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--default-cell-id", type=int, default=0)

    p.add_argument("--graph-loader", default="ggmv", choices=["ggmv", "msld"])
    p.add_argument("--kg-nodes-file", default="nodes.json")
    p.add_argument("--kg-edges-file", default="edges.json")
    p.add_argument("--kg-graph-file", default="graph.json")
    p.add_argument("--graph-alpha", type=float, default=0.1)
    p.add_argument("--graph-diffusion-hops", type=int, default=1)
    p.add_argument("--graph-diffusion-decay", type=float, default=1.0)
    p.add_argument("--max-modules", type=int, default=2048)

    p.add_argument("--min-genes-per-perturbation", type=int, default=16)
    p.add_argument("--support-fraction", type=float, default=0.4)
    p.add_argument("--min-support-size", type=int, default=8)
    p.add_argument("--min-query-size", type=int, default=8)
    p.add_argument("--max-support-size", type=int, default=256)

    p.add_argument("--num-candidates", type=int, default=12)
    p.add_argument("--top-k-verifier", type=int, default=3)
    p.add_argument("--min-candidate-nodes", type=int, default=4)
    p.add_argument("--max-candidate-nodes", type=int, default=24)
    p.add_argument("--max-neighbors", type=int, default=24)

    p.add_argument("--a1-near-drug", type=float, default=1.0)
    p.add_argument("--a2-cover-pos", type=float, default=1.0)
    p.add_argument("--a3-cover-neg", type=float, default=1.0)
    p.add_argument("--a4-type-prior", type=float, default=0.25)
    p.add_argument("--b1-confidence", type=float, default=1.0)
    p.add_argument("--b2-type-penalty", type=float, default=0.25)

    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=1.0)

    p.add_argument("--beta-replay", type=float, default=0.5)
    p.add_argument("--gamma-llm", type=float, default=0.5)
    p.add_argument("--lambda-size", type=float, default=0.01)
    p.add_argument("--eta-qual", type=float, default=1.0)
    p.add_argument("--eta-rank", type=float, default=0.5)
    p.add_argument("--rank-margin", type=float, default=1.0)
    p.add_argument("--save-eps", type=float, default=1e-5)
    p.add_argument("--log-every-episodes", type=int, default=50)
    p.add_argument("--debug-episodes-per-epoch", type=int, default=3)
    p.add_argument("--debug-top-candidates", type=int, default=3)
    p.add_argument("--debug-top-modules", type=int, default=8)
    p.add_argument("--log-modules-per-candidate", type=int, default=6)
    p.add_argument("--print-candidate-details", action="store_true")
    p.add_argument("--debug-log-jsonl", default="")
    p.add_argument("--debug-log-all-episodes", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
