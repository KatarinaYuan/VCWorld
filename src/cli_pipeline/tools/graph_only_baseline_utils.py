#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for graph-only module proposer + shallow gene scorer baseline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


class GraphLike(Protocol):
    idx_to_gene: List[str]
    idx_to_module: List[str]
    module_types: List[str]
    global_module_prior: np.ndarray
    r_tilde: sp.csr_matrix

    @property
    def num_genes(self) -> int:
        ...

    @property
    def num_modules(self) -> int:
        ...

    def lookup_gene_id(self, gene_name: str) -> Optional[int]:
        ...

    def lookup_drug_id(self, drug_name: str) -> Optional[int]:
        ...

    def get_targets(self, drug_id: int) -> List[int]:
        ...


@dataclass
class GraphOnlyExample:
    pert: str
    gene: str
    split: str
    label: int
    cell_id: int
    drug_id: int
    gene_id: int

    @property
    def perturbation_key(self) -> str:
        return f"{self.cell_id}::{self.pert.strip().lower()}"


@dataclass
class GraphOnlyPerturbation:
    perturbation_key: str
    pert: str
    cell_id: int
    drug_id: int
    gene_ids: np.ndarray
    labels: np.ndarray


def load_graph_for_baseline(
    *,
    kg_dir: str,
    graph_loader: str,
    graph_alpha: float,
    graph_diffusion_hops: int,
    graph_diffusion_decay: float,
    max_modules: int,
    kg_nodes_file: str,
    kg_edges_file: str,
    kg_graph_file: str,
) -> GraphLike:
    loader = graph_loader.strip().lower()
    if loader == "msld":
        from msld_graph import load_msld_graph

        return load_msld_graph(
            kg_dir=kg_dir,
            alpha=graph_alpha,
            diffusion_hops=graph_diffusion_hops,
            diffusion_decay=graph_diffusion_decay,
            max_modules=max_modules,
            nodes_file=kg_nodes_file,
            edges_file=kg_edges_file,
            graph_file=kg_graph_file,
        )
    if loader == "ggmv":
        from ggmv_graph import load_ggmv_graph

        return load_ggmv_graph(
            kg_dir=kg_dir,
            alpha=graph_alpha,
            nodes_file=kg_nodes_file,
            edges_file=kg_edges_file,
            graph_file=kg_graph_file,
            strict=False,
        )
    raise ValueError(f"Unsupported --graph-loader: {graph_loader}")


def load_graph_only_examples(
    *,
    labels_csv: str,
    graph: GraphLike,
    split: str,
    default_cell_id: int,
    allow_missing_drug: bool = True,
) -> List[GraphOnlyExample]:
    split_norm = split.strip().lower()
    out: List[GraphOnlyExample] = []
    with open(labels_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = str(row.get("split", "")).strip().lower()
            if split_norm and row_split != split_norm:
                continue
            pert = str(row.get("pert", "")).strip()
            gene = str(row.get("gene", "")).strip()
            if not pert or not gene:
                continue
            try:
                label = int(row.get("label", ""))
            except (TypeError, ValueError):
                continue
            if label not in (0, 1):
                continue

            gid = graph.lookup_gene_id(gene)
            if gid is None:
                continue
            did = graph.lookup_drug_id(pert)
            if did is None and not allow_missing_drug:
                continue

            out.append(
                GraphOnlyExample(
                    pert=pert,
                    gene=gene,
                    split=row_split,
                    label=int(label),
                    cell_id=int(default_cell_id),
                    drug_id=int(did) if did is not None else -1,
                    gene_id=int(gid),
                )
            )
    return out


def build_perturbations(
    examples: Sequence[GraphOnlyExample],
    *,
    min_genes_per_perturbation: int,
) -> List[GraphOnlyPerturbation]:
    grouped: Dict[str, Dict[str, object]] = {}
    for ex in examples:
        item = grouped.setdefault(
            ex.perturbation_key,
            {
                "pert": ex.pert,
                "cell_id": int(ex.cell_id),
                "drug_id": int(ex.drug_id),
                "gene_to_label": {},
            },
        )
        item["gene_to_label"][int(ex.gene_id)] = int(ex.label)  # type: ignore[index]

    out: List[GraphOnlyPerturbation] = []
    for key in sorted(grouped.keys()):
        item = grouped[key]
        pairs = sorted(item["gene_to_label"].items(), key=lambda x: x[0])  # type: ignore[union-attr]
        if len(pairs) < int(min_genes_per_perturbation):
            continue
        gene_ids = np.asarray([int(g) for g, _ in pairs], dtype=np.int64)
        labels = np.asarray([float(y) for _, y in pairs], dtype=np.float32)
        out.append(
            GraphOnlyPerturbation(
                perturbation_key=key,
                pert=str(item["pert"]),
                cell_id=int(item["cell_id"]),
                drug_id=int(item["drug_id"]),
                gene_ids=gene_ids,
                labels=labels,
            )
        )
    return out


def _episode_seed(seed: int, epoch: int, perturbation_key: str) -> int:
    token = f"{int(seed)}::{int(epoch)}::{perturbation_key}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def sample_support_query_indices(
    *,
    num_items: int,
    rng: np.random.Generator,
    support_fraction: float,
    min_support_size: int,
    min_query_size: int,
    max_support_size: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    n = int(num_items)
    min_support = max(1, int(min_support_size))
    min_query = max(1, int(min_query_size))
    if n < (min_support + min_query):
        return None
    support_n = int(round(float(support_fraction) * float(n)))
    support_n = max(min_support, support_n)
    if max_support_size > 0:
        support_n = min(support_n, int(max_support_size))
    support_n = min(support_n, n - min_query)
    if support_n < min_support:
        return None
    if (n - support_n) < min_query:
        return None
    perm = rng.permutation(n)
    support_idx = np.sort(perm[:support_n]).astype(np.int64)
    query_idx = np.sort(perm[support_n:]).astype(np.int64)
    if query_idx.size == 0:
        return None
    return support_idx, query_idx


def sample_episode_for_perturbation(
    perturbation: GraphOnlyPerturbation,
    *,
    seed: int,
    epoch: int,
    support_fraction: float,
    min_support_size: int,
    min_query_size: int,
    max_support_size: int,
) -> Optional[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(_episode_seed(seed, epoch, perturbation.perturbation_key))
    split = sample_support_query_indices(
        num_items=int(perturbation.gene_ids.shape[0]),
        rng=rng,
        support_fraction=support_fraction,
        min_support_size=min_support_size,
        min_query_size=min_query_size,
        max_support_size=max_support_size,
    )
    if split is None:
        return None
    support_idx, query_idx = split
    return {
        "support_gene_ids": perturbation.gene_ids[support_idx].astype(np.int64),
        "support_labels": perturbation.labels[support_idx].astype(np.float32),
        "query_gene_ids": perturbation.gene_ids[query_idx].astype(np.int64),
        "query_labels": perturbation.labels[query_idx].astype(np.float32),
    }


def get_module_feature_names(proposer_feature_set: str) -> List[str]:
    mode = proposer_feature_set.strip().lower()
    if mode == "target_only":
        return ["target_to_module_score"]
    if mode == "support_only":
        return ["positive_support_score", "negative_support_score", "contrast_score"]
    if mode == "full":
        return [
            "target_to_module_score",
            "positive_support_score",
            "negative_support_score",
            "contrast_score",
        ]
    raise ValueError(f"Unsupported proposer feature set: {proposer_feature_set}")


def get_gene_feature_names(
    *,
    gene_feature_set: str,
    include_top_module_coverage: bool,
) -> List[str]:
    mode = gene_feature_set.strip().lower()
    if mode == "alignment_only":
        return ["module_alignment"]
    if mode == "full":
        names = ["module_alignment", "target_gene_proximity"]
        if include_top_module_coverage:
            names.append("top_module_coverage_count")
        return names
    raise ValueError(f"Unsupported gene feature set: {gene_feature_set}")


def _safe_mean_rows(matrix: sp.csr_matrix, row_ids: Sequence[int]) -> np.ndarray:
    m = int(matrix.shape[1])
    if m == 0:
        return np.zeros((0,), dtype=np.float32)
    if not row_ids:
        return np.zeros((m,), dtype=np.float32)
    row_np = np.asarray(row_ids, dtype=np.int64)
    if row_np.size == 0:
        return np.zeros((m,), dtype=np.float32)
    arr = matrix[row_np].toarray().astype(np.float32)
    if arr.size == 0:
        return np.zeros((m,), dtype=np.float32)
    return arr.mean(axis=0).astype(np.float32)


def _normalize_nonnegative(values: np.ndarray, fallback: Optional[np.ndarray] = None, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return v
    v = np.maximum(v, 0.0)
    den = float(v.sum())
    if den > eps:
        return (v / den).astype(np.float32)
    if fallback is not None:
        fb = np.asarray(fallback, dtype=np.float32).reshape(-1)
        if fb.size != v.size:
            raise ValueError(f"fallback size mismatch: expected={v.size} got={fb.size}")
        fb = np.maximum(fb, 0.0)
        den_fb = float(fb.sum())
        if den_fb > eps:
            return (fb / den_fb).astype(np.float32)
    return np.zeros_like(v, dtype=np.float32)


def compute_target_to_module_score(graph: GraphLike, drug_id: int) -> np.ndarray:
    if graph.num_modules <= 0:
        return np.zeros((0,), dtype=np.float32)
    if int(drug_id) < 0:
        return np.zeros((graph.num_modules,), dtype=np.float32)
    targets = [int(g) for g in graph.get_targets(int(drug_id)) if 0 <= int(g) < graph.num_genes]
    return _safe_mean_rows(graph.r_tilde, targets)


def compute_support_module_scores(
    *,
    graph: GraphLike,
    support_gene_ids: Sequence[int],
    support_labels: Sequence[float],
) -> Dict[str, np.ndarray]:
    if graph.num_modules <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return {
            "positive_support_score": z,
            "negative_support_score": z,
            "contrast_score": z,
        }
    n = min(len(support_gene_ids), len(support_labels))
    if n <= 0:
        z = np.zeros((graph.num_modules,), dtype=np.float32)
        return {
            "positive_support_score": z,
            "negative_support_score": z,
            "contrast_score": z,
        }

    gids = np.asarray(support_gene_ids[:n], dtype=np.int64)
    labels = np.asarray(support_labels[:n], dtype=np.float32)
    valid = (gids >= 0) & (gids < graph.num_genes)
    gids = gids[valid]
    labels = labels[valid]
    if gids.size == 0:
        z = np.zeros((graph.num_modules,), dtype=np.float32)
        return {
            "positive_support_score": z,
            "negative_support_score": z,
            "contrast_score": z,
        }

    pos_ids = gids[labels >= 0.5].tolist()
    neg_ids = gids[labels < 0.5].tolist()
    pos = _safe_mean_rows(graph.r_tilde, pos_ids)
    neg = _safe_mean_rows(graph.r_tilde, neg_ids)
    return {
        "positive_support_score": pos.astype(np.float32),
        "negative_support_score": neg.astype(np.float32),
        "contrast_score": (pos - neg).astype(np.float32),
    }


def build_module_feature_matrix(
    *,
    graph: GraphLike,
    drug_id: int,
    support_gene_ids: Sequence[int],
    support_labels: Sequence[float],
    proposer_feature_set: str,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    m = int(graph.num_modules)
    if m <= 0:
        return np.zeros((0, 0), dtype=np.float32), [], {}

    target = compute_target_to_module_score(graph, int(drug_id))
    support_scores = compute_support_module_scores(
        graph=graph,
        support_gene_ids=support_gene_ids,
        support_labels=support_labels,
    )
    pos = support_scores["positive_support_score"]
    neg = support_scores["negative_support_score"]
    contrast = support_scores["contrast_score"]

    names = get_module_feature_names(proposer_feature_set)
    feature_map = {
        "target_to_module_score": target,
        "positive_support_score": pos,
        "negative_support_score": neg,
        "contrast_score": contrast,
    }
    mat = np.stack([feature_map[name] for name in names], axis=1).astype(np.float32)

    module_prior = _normalize_nonnegative(target, fallback=graph.global_module_prior)
    if module_prior.size > 0 and float(module_prior.sum()) <= 0.0:
        module_prior = np.ones_like(module_prior, dtype=np.float32) / float(module_prior.size)
    target_module_distribution = _normalize_nonnegative(target, fallback=None)
    return (
        mat,
        names,
        {
            **feature_map,
            "module_prior": module_prior.astype(np.float32),
            "target_module_distribution": target_module_distribution.astype(np.float32),
        },
    )


def build_gene_feature_matrix(
    *,
    graph: GraphLike,
    query_gene_ids: Sequence[int],
    q_probs: np.ndarray,
    target_module_distribution: np.ndarray,
    gene_feature_set: str,
    include_top_module_coverage: bool,
    top_k_modules: int,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    names = get_gene_feature_names(
        gene_feature_set=gene_feature_set,
        include_top_module_coverage=include_top_module_coverage,
    )
    n = len(query_gene_ids)
    if n <= 0:
        return np.zeros((0, len(names)), dtype=np.float32), names, {}
    if graph.num_modules <= 0:
        return np.zeros((n, len(names)), dtype=np.float32), names, {}

    gids = np.asarray(query_gene_ids, dtype=np.int64)
    mat = graph.r_tilde[gids].toarray().astype(np.float32)  # [N, M]
    q = np.asarray(q_probs, dtype=np.float32).reshape(-1)
    if q.size != graph.num_modules:
        raise ValueError(f"q_probs size mismatch: expected={graph.num_modules} got={q.size}")
    tdist = np.asarray(target_module_distribution, dtype=np.float32).reshape(-1)
    if tdist.size != graph.num_modules:
        raise ValueError(f"target_module_distribution size mismatch: expected={graph.num_modules} got={tdist.size}")

    module_alignment = (mat @ q).astype(np.float32)
    target_gene_proximity = (mat @ tdist).astype(np.float32)

    top_module_coverage_count = np.zeros((n,), dtype=np.float32)
    if include_top_module_coverage and q.size > 0:
        k = max(1, min(int(top_k_modules), int(q.size)))
        top_ids = np.argsort(q)[::-1][:k]
        top_module_coverage_count = (mat[:, top_ids] > 0.0).sum(axis=1).astype(np.float32)

    feat_map = {
        "module_alignment": module_alignment,
        "target_gene_proximity": target_gene_proximity,
        "top_module_coverage_count": top_module_coverage_count,
    }
    feat = np.stack([feat_map[name] for name in names], axis=1).astype(np.float32)
    return feat, names, feat_map


def distribution_entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return 0.0
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def top_modules_from_posterior(
    *,
    q_probs: np.ndarray,
    module_names: Sequence[str],
    module_types: Sequence[str],
    top_k: int,
) -> List[Dict[str, object]]:
    q = np.asarray(q_probs, dtype=np.float32).reshape(-1)
    if q.size == 0:
        return []
    k = max(1, min(int(top_k), int(q.size)))
    order = np.argsort(q)[::-1][:k]
    out: List[Dict[str, object]] = []
    for mid in order.tolist():
        name = module_names[mid] if mid < len(module_names) else f"module_{mid}"
        mtype = module_types[mid] if mid < len(module_types) else "module"
        out.append(
            {
                "module_id": int(mid),
                "module_name": str(name),
                "module_type": str(mtype),
                "prob": float(q[mid]),
            }
        )
    return out


def _safe_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    if len(set(int(x) for x in y_true)) < 2:
        return None
    pos_scores = [float(s) for y, s in zip(y_true, y_score) if int(y) == 1]
    neg_scores = [float(s) for y, s in zip(y_true, y_score) if int(y) == 0]
    if not pos_scores or not neg_scores:
        return None
    better = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                better += 1.0
            elif ps == ns:
                better += 0.5
    return float(better / float(len(pos_scores) * len(neg_scores)))


def _safe_auprc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    if len(set(int(x) for x in y_true)) < 2:
        return None
    pairs = sorted(zip(y_score, y_true), key=lambda x: float(x[0]), reverse=True)
    total_pos = sum(1 for y in y_true if int(y) == 1)
    if total_pos <= 0:
        return None
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for score, y in pairs:
        _ = score
        if int(y) == 1:
            tp += 1
        else:
            fp += 1
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / total_pos)
        if int(y) == 1:
            ap += precision * (recall - prev_recall)
            prev_recall = recall
    return float(ap)


def compute_binary_metrics(
    *,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    decision_threshold: float,
) -> Dict[str, Optional[float]]:
    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "auprc": None,
            "yes_prediction_rate": None,
        }

    y_bin = [int(y) for y in y_true]
    y_pred = [1 if float(p) >= float(decision_threshold) else 0 for p in y_prob]

    tp = sum(1 for yt, yp in zip(y_bin, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_bin, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_bin, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_bin, y_pred) if yt == 1 and yp == 0)

    accuracy = float((tp + tn) / n)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    yes_rate = float(sum(y_pred) / n)
    return {
        "n": int(n),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _safe_auroc(y_bin, y_prob),
        "auprc": _safe_auprc(y_bin, y_prob),
        "yes_prediction_rate": yes_rate,
    }
