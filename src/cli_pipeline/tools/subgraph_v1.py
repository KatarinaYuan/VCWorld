#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1 support-conditioned subgraph pipeline components.

This module provides:
- Heuristic path-seeded PCST-like proposer (module-level approximation)
- Lightweight fast scorer (gene head + quality head)
- Frozen verifier (cheap stand-in for top-K mechanism verification)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CandidateSubgraph:
    module_ids: List[int]
    estimated_edges: int

    @property
    def size(self) -> int:
        return len(self.module_ids)


def pairwise_ranking_loss(preds: torch.Tensor, targets: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Pairwise hinge ranking loss on 1D vectors."""
    if preds.numel() <= 1:
        return preds.new_tensor(0.0)
    p = preds.reshape(-1)
    t = targets.reshape(-1)
    diff_t = t[:, None] - t[None, :]
    mask = diff_t > 0
    if not bool(mask.any()):
        return preds.new_tensor(0.0)
    diff_p = p[:, None] - p[None, :]
    hinge = torch.relu(float(margin) - diff_p)
    return hinge[mask].mean()


class HeuristicPathSeededPCSTProposer:
    """Generate module candidates by prize-maximizing connected growth.

    This is a module-level approximation of path-seeded PCST to keep V1 cheap.
    """

    def __init__(
        self,
        *,
        graph,
        max_neighbors: int = 24,
        rng_seed: int = 42,
        a1_near_drug: float = 1.0,
        a2_cover_pos: float = 1.0,
        a3_cover_neg: float = 1.0,
        a4_type_prior: float = 0.25,
        b1_confidence: float = 1.0,
        b2_type_penalty: float = 0.25,
    ) -> None:
        self.graph = graph
        self.max_neighbors = int(max_neighbors)
        self.rng = np.random.default_rng(int(rng_seed))
        self.a1 = float(a1_near_drug)
        self.a2 = float(a2_cover_pos)
        self.a3 = float(a3_cover_neg)
        self.a4 = float(a4_type_prior)
        self.b1 = float(b1_confidence)
        self.b2 = float(b2_type_penalty)

        self._neighbors = self._build_module_neighbors(max_neighbors=self.max_neighbors)

    def _module_type_prior(self, module_type: str) -> float:
        t = str(module_type).strip().lower()
        if t == "reactome":
            return 1.0
        if t == "complex":
            return 0.9
        if t == "go_bp":
            return 0.8
        if t == "go_mf":
            return 0.6
        return 0.5

    def _build_module_neighbors(self, *, max_neighbors: int) -> Dict[int, List[Tuple[int, float]]]:
        """Build sparse module adjacency from shared genes."""
        inv: Dict[int, List[int]] = {}
        module_sizes: Dict[int, int] = {}
        for mid, genes in enumerate(self.graph.module_gene_lists):
            uniq = sorted(set(int(g) for g in genes))
            module_sizes[mid] = max(1, len(uniq))
            for gid in uniq:
                inv.setdefault(int(gid), []).append(int(mid))

        shared: Dict[Tuple[int, int], int] = {}
        for mids in inv.values():
            if len(mids) < 2:
                continue
            for i in range(len(mids)):
                a = mids[i]
                for j in range(i + 1, len(mids)):
                    b = mids[j]
                    if a == b:
                        continue
                    key = (a, b) if a < b else (b, a)
                    shared[key] = shared.get(key, 0) + 1

        nbrs: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(int(self.graph.num_modules))}
        for (a, b), cnt in shared.items():
            den = float(max(1, min(module_sizes[a], module_sizes[b])))
            sim = float(cnt) / den
            if sim <= 0.0:
                continue
            nbrs[a].append((b, sim))
            nbrs[b].append((a, sim))

        out: Dict[int, List[Tuple[int, float]]] = {}
        for mid, pairs in nbrs.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
            if max_neighbors > 0:
                pairs_sorted = pairs_sorted[:max_neighbors]
            out[mid] = pairs_sorted
        return out

    def _compute_node_prize(self, mod_aux: Dict[str, np.ndarray]) -> np.ndarray:
        target = np.asarray(mod_aux["target_to_module_score"], dtype=np.float32)
        pos = np.asarray(mod_aux["positive_support_score"], dtype=np.float32)
        neg = np.asarray(mod_aux["negative_support_score"], dtype=np.float32)

        priors = np.asarray(
            [self._module_type_prior(self.graph.module_types[i] if i < len(self.graph.module_types) else "") for i in range(len(target))],
            dtype=np.float32,
        )
        prize = self.a1 * target + self.a2 * pos - self.a3 * neg + self.a4 * priors
        return prize.astype(np.float32)

    def score_components(self, mod_aux: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Expose proposer score components for monitoring/logging."""
        target = np.asarray(mod_aux["target_to_module_score"], dtype=np.float32).reshape(-1)
        pos = np.asarray(mod_aux["positive_support_score"], dtype=np.float32).reshape(-1)
        neg = np.asarray(mod_aux["negative_support_score"], dtype=np.float32).reshape(-1)
        contrast = np.asarray(mod_aux["contrast_score"], dtype=np.float32).reshape(-1)
        prize = self._compute_node_prize(mod_aux).reshape(-1)
        return {
            "target": target.astype(np.float32),
            "pos": pos.astype(np.float32),
            "neg": neg.astype(np.float32),
            "contrast": contrast.astype(np.float32),
            "prize": prize.astype(np.float32),
        }

    def _best_edge_gain(self, node: int, current: set[int], prize: np.ndarray) -> float:
        max_sim = 0.0
        same_type_bonus = 0.0
        t_node = self.graph.module_types[node] if node < len(self.graph.module_types) else ""
        for nbr, sim in self._neighbors.get(node, []):
            if nbr not in current:
                continue
            if sim > max_sim:
                max_sim = float(sim)
            t_nbr = self.graph.module_types[nbr] if nbr < len(self.graph.module_types) else ""
            if t_node == t_nbr:
                same_type_bonus = 1.0
        edge_cost = self.b1 * (1.0 - max_sim) + self.b2 * (1.0 - same_type_bonus)
        return float(prize[node] - edge_cost)

    def _grow_candidate(
        self,
        *,
        seed: int,
        prize: np.ndarray,
        min_nodes: int,
        max_nodes: int,
    ) -> CandidateSubgraph:
        selected: set[int] = {int(seed)}
        frontier: set[int] = {int(n) for n, _ in self._neighbors.get(int(seed), [])}

        target_size = int(self.rng.integers(max(1, min_nodes), max(min_nodes, max_nodes) + 1))
        while frontier and len(selected) < target_size:
            scored = []
            for nid in frontier:
                if nid in selected:
                    continue
                gain = self._best_edge_gain(nid, selected, prize)
                scored.append((gain, nid))
            if not scored:
                break
            scored.sort(key=lambda x: x[0], reverse=True)
            best_gain, best_nid = scored[0]
            # Keep growing until minimum size; then require non-trivial gain.
            if len(selected) >= int(min_nodes) and best_gain < -0.10:
                break
            selected.add(int(best_nid))
            frontier.discard(int(best_nid))
            for nbr, _ in self._neighbors.get(int(best_nid), []):
                if nbr not in selected:
                    frontier.add(int(nbr))

        mids = sorted(selected)
        edges = 0
        mids_set = set(mids)
        for mid in mids:
            for nbr, sim in self._neighbors.get(mid, []):
                if nbr in mids_set and nbr > mid and sim > 0.0:
                    edges += 1
        if edges <= 0 and len(mids) > 1:
            edges = len(mids) - 1
        return CandidateSubgraph(module_ids=mids, estimated_edges=int(edges))

    @staticmethod
    def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
        sa = set(int(x) for x in a)
        sb = set(int(x) for x in b)
        if not sa and not sb:
            return 1.0
        return float(len(sa & sb) / max(1, len(sa | sb)))

    def generate(
        self,
        *,
        mod_aux: Dict[str, np.ndarray],
        num_candidates: int,
        min_nodes: int,
        max_nodes: int,
    ) -> List[CandidateSubgraph]:
        m = int(self.graph.num_modules)
        if m <= 0:
            return []

        prize = self._compute_node_prize(mod_aux)
        target = np.asarray(mod_aux["target_to_module_score"], dtype=np.float32)
        pos = np.asarray(mod_aux["positive_support_score"], dtype=np.float32)
        contrast = np.asarray(mod_aux["contrast_score"], dtype=np.float32)

        k_seed = max(1, min(int(num_candidates * 3), m))
        seed_pool = set(np.argsort(prize)[::-1][:k_seed].tolist())
        seed_pool.update(np.argsort(target)[::-1][:k_seed].tolist())
        seed_pool.update(np.argsort(pos)[::-1][:k_seed].tolist())
        seed_pool.update(np.argsort(contrast)[::-1][:k_seed].tolist())
        seed_list = list(seed_pool)
        self.rng.shuffle(seed_list)

        out: List[CandidateSubgraph] = []
        for seed in seed_list:
            cand = self._grow_candidate(seed=int(seed), prize=prize, min_nodes=min_nodes, max_nodes=max_nodes)
            is_dup = any(self._jaccard(cand.module_ids, x.module_ids) >= 0.95 for x in out)
            if not is_dup:
                out.append(cand)
            if len(out) >= int(num_candidates):
                break

        # Backfill with top-1 singletons if connectivity search is sparse.
        if len(out) < int(num_candidates):
            top = np.argsort(prize)[::-1]
            for mid in top.tolist():
                cand = CandidateSubgraph(module_ids=[int(mid)], estimated_edges=0)
                is_dup = any(self._jaccard(cand.module_ids, x.module_ids) >= 0.95 for x in out)
                if not is_dup:
                    out.append(cand)
                if len(out) >= int(num_candidates):
                    break
        return out


class FastScorer(nn.Module):
    """Lightweight scorer with per-gene and per-candidate heads."""

    def __init__(
        self,
        *,
        candidate_feature_dim: int,
        gene_feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gene_head = nn.Sequential(
            nn.Linear(gene_feature_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        candidate_features: torch.Tensor,
        gene_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (gene_logits, quality_pred).

        candidate_features: [C]
        gene_features: [N, G]
        """
        cand_h = self.candidate_encoder(candidate_features.reshape(1, -1)).reshape(-1)
        n = int(gene_features.shape[0])
        cand_expand = cand_h.reshape(1, -1).expand(n, -1)
        gene_in = torch.cat([gene_features, cand_expand], dim=-1)
        gene_logits = self.gene_head(gene_in).reshape(-1)
        quality_pred = self.quality_head(cand_h.reshape(1, -1)).reshape(-1)[0]
        return gene_logits, quality_pred


def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    den = float(np.abs(x).sum())
    if den <= eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / den).astype(np.float32)


def build_candidate_feature_vector(
    *,
    candidate: CandidateSubgraph,
    mod_aux: Dict[str, np.ndarray],
    module_types: Sequence[str],
    total_modules: int,
) -> np.ndarray:
    mids = np.asarray(candidate.module_ids, dtype=np.int64)
    if mids.size <= 0:
        return np.zeros((12,), dtype=np.float32)

    target = np.asarray(mod_aux["target_to_module_score"], dtype=np.float32)
    pos = np.asarray(mod_aux["positive_support_score"], dtype=np.float32)
    neg = np.asarray(mod_aux["negative_support_score"], dtype=np.float32)
    contrast = np.asarray(mod_aux["contrast_score"], dtype=np.float32)

    t = target[mids]
    p = pos[mids]
    n = neg[mids]
    c = contrast[mids]

    type_hist = np.zeros((4,), dtype=np.float32)  # reactome, go_bp, go_mf, complex
    for mid in mids.tolist():
        tname = str(module_types[mid] if mid < len(module_types) else "").strip().lower()
        if tname == "reactome":
            type_hist[0] += 1.0
        elif tname == "go_bp":
            type_hist[1] += 1.0
        elif tname == "go_mf":
            type_hist[2] += 1.0
        elif tname == "complex":
            type_hist[3] += 1.0
    type_hist = type_hist / max(1.0, float(mids.size))

    vec = np.asarray(
        [
            float(np.mean(t)),
            float(np.mean(p)),
            float(np.mean(n)),
            float(np.mean(c)),
            float(np.max(t)),
            float(np.max(c)),
            float(mids.size / max(1, int(total_modules))),
            float(candidate.estimated_edges / max(1, mids.size - 1 if mids.size > 1 else 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([vec, type_hist], axis=0).astype(np.float32)


def summarize_candidate_score_components(
    *,
    candidate: CandidateSubgraph,
    score_components: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Aggregate proposer score components over one candidate."""
    mids = np.asarray(candidate.module_ids, dtype=np.int64)
    if mids.size <= 0:
        return {
            "mean_prize": 0.0,
            "max_prize": 0.0,
            "mean_target": 0.0,
            "mean_pos": 0.0,
            "mean_neg": 0.0,
            "mean_contrast": 0.0,
        }

    prize = np.asarray(score_components.get("prize", []), dtype=np.float32)
    target = np.asarray(score_components.get("target", []), dtype=np.float32)
    pos = np.asarray(score_components.get("pos", []), dtype=np.float32)
    neg = np.asarray(score_components.get("neg", []), dtype=np.float32)
    contrast = np.asarray(score_components.get("contrast", []), dtype=np.float32)

    def _take_mean(x: np.ndarray) -> float:
        if x.size <= 0:
            return 0.0
        return float(np.mean(x[mids]))

    def _take_max(x: np.ndarray) -> float:
        if x.size <= 0:
            return 0.0
        return float(np.max(x[mids]))

    return {
        "mean_prize": _take_mean(prize),
        "max_prize": _take_max(prize),
        "mean_target": _take_mean(target),
        "mean_pos": _take_mean(pos),
        "mean_neg": _take_mean(neg),
        "mean_contrast": _take_mean(contrast),
    }


def build_gene_feature_matrix_for_candidate(
    *,
    graph,
    gene_ids: np.ndarray,
    candidate: CandidateSubgraph,
    mod_aux: Dict[str, np.ndarray],
) -> np.ndarray:
    gids = np.asarray(gene_ids, dtype=np.int64)
    if gids.size <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    mids = np.asarray(candidate.module_ids, dtype=np.int64)
    if mids.size <= 0:
        return np.zeros((gids.size, 4), dtype=np.float32)

    prox = graph.r_tilde[gids][:, mids].toarray().astype(np.float32)
    align_mean = np.mean(prox, axis=1).astype(np.float32)
    align_max = np.max(prox, axis=1).astype(np.float32)

    target_dist = _safe_norm(np.asarray(mod_aux.get("target_module_distribution", []), dtype=np.float32))
    if target_dist.size == int(graph.num_modules):
        target_gene = (graph.r_tilde[gids].toarray().astype(np.float32) @ target_dist).astype(np.float32)
    else:
        target_gene = np.zeros((gids.size,), dtype=np.float32)

    contrast = _safe_norm(np.asarray(mod_aux.get("contrast_score", []), dtype=np.float32))
    if contrast.size == int(graph.num_modules):
        contrast_gene = (graph.r_tilde[gids].toarray().astype(np.float32) @ contrast).astype(np.float32)
    else:
        contrast_gene = np.zeros((gids.size,), dtype=np.float32)

    return np.stack([align_mean, align_max, target_gene, contrast_gene], axis=1).astype(np.float32)


class FrozenLLMVerifier:
    """Cheap verifier API-compatible with future LLM verifier.

    The score combines support replay consistency and contrast grounding.
    """

    def __init__(self, replay_weight: float = 0.5, grounding_weight: float = 0.5) -> None:
        self.replay_weight = float(replay_weight)
        self.grounding_weight = float(grounding_weight)

    def verify_candidate(
        self,
        *,
        candidate: CandidateSubgraph,
        support_labels: np.ndarray,
        support_probs: np.ndarray,
        module_names: Sequence[str],
    ) -> Dict[str, object]:
        y = np.asarray(support_labels, dtype=np.float32).reshape(-1)
        p = np.asarray(support_probs, dtype=np.float32).reshape(-1)
        if y.size > 0 and p.size == y.size:
            pred = (p >= 0.5).astype(np.float32)
            consistency = float(np.mean(pred == y))
            pos = p[y >= 0.5]
            neg = p[y < 0.5]
            if pos.size > 0 and neg.size > 0:
                grounding_raw = float(np.mean(pos) - np.mean(neg))
            else:
                grounding_raw = 0.0
            grounding = float(np.clip((grounding_raw + 1.0) * 0.5, 0.0, 1.0))
        else:
            consistency = 0.0
            grounding = 0.0

        score = float(np.clip(self.replay_weight * consistency + self.grounding_weight * grounding, 0.0, 1.0))
        top_modules = [module_names[m] if 0 <= int(m) < len(module_names) else f"module_{m}" for m in candidate.module_ids[:4]]
        summary = (
            "Candidate subgraph links support pattern through modules: "
            + ", ".join(top_modules)
            + "."
        )
        return {
            "mechanism_summary": summary,
            "support_replay_predictions": [int(x) for x in (p >= 0.5).astype(np.int64).tolist()],
            "consistency_score": consistency,
            "grounding_score": grounding,
            "confidence": score,
        }


class CandidateBatchLoss(nn.Module):
    """Bundle of V1 losses for one episode."""

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)

    def bce_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pw = logits.new_tensor([self.pos_weight])
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)

    def quality_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred.reshape(-1), target.reshape(-1))
