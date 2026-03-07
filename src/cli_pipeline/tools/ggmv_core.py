#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core GGMV scoring functions (graph prior + support evidence + calibration)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ggmv_graph import GGMVGraph


@dataclass
class GGMVParams:
    """Small calibrator parameters for GGMV inference/training."""

    top_k_modules: int = 64
    posterior_mode: str = "sparsemax"  # sparsemax | softmax
    eps: float = 1e-8

    # support utility
    w0: float = 0.0
    w1: float = 1.0
    w2: float = 1.0

    # posterior mixture
    beta: float = 1.0  # support evidence weight
    gamma_mech: float = 1.0  # llm mechanism score weight

    # verification/sufficiency
    lambda_v: float = 1.0
    gamma_a: float = 0.1
    eta1: float = 1.0
    eta2: float = 1.0
    eta3: float = 1.0
    eta4: float = 0.0
    b0: float = 0.0
    b1: float = 1.0

    # ablation flags
    no_graph_prior: bool = False
    no_support_utility: bool = False
    no_llm_mech: bool = False
    no_verification: bool = False
    no_sufficiency: bool = False


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    if probs.size == 0:
        return 0.0
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    return float(-(p * np.log(p)).sum())


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits.astype(np.float32)
    z = logits - np.max(logits)
    exp = np.exp(z)
    den = float(exp.sum())
    if den <= 0:
        return np.ones_like(logits, dtype=np.float32) / float(logits.size)
    return (exp / den).astype(np.float32)


def sparsemax(logits: np.ndarray) -> np.ndarray:
    """Sparsemax for 1D logits."""
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if z.size == 0:
        return z.astype(np.float32)

    z_sorted = np.sort(z)[::-1]
    z_cumsum = np.cumsum(z_sorted)
    ks = np.arange(1, z.size + 1, dtype=np.float64)

    support = 1.0 + ks * z_sorted > z_cumsum
    if not np.any(support):
        return np.ones_like(z, dtype=np.float32) / float(z.size)

    k = int(np.max(np.where(support)[0]) + 1)
    tau = (z_cumsum[k - 1] - 1.0) / float(k)
    p = np.maximum(z - tau, 0.0)

    s = float(p.sum())
    if s <= 0:
        return np.ones_like(z, dtype=np.float32) / float(z.size)
    return (p / s).astype(np.float32)


def compute_query_module_prior(drug_id: int, graph: GGMVGraph) -> np.ndarray:
    """Compute query drug -> module prior over all modules."""
    if graph.num_modules == 0:
        return np.zeros((0,), dtype=np.float32)

    did = int(drug_id)
    if did in graph.module_prior_cache:
        return graph.module_prior_cache[did]

    if did < 0:
        graph.module_prior_cache[did] = graph.global_module_prior.copy()
        return graph.module_prior_cache[did]

    targets = graph.get_targets(did)
    if not targets:
        graph.module_prior_cache[did] = graph.global_module_prior.copy()
        return graph.module_prior_cache[did]

    prior_raw = np.asarray(graph.r_tilde[targets].sum(axis=0)).reshape(-1).astype(np.float32)
    prior_raw = np.maximum(prior_raw, 0.0)
    if float(prior_raw.sum()) <= 0:
        prior = graph.global_module_prior.copy()
    else:
        prior = prior_raw / float(prior_raw.sum())

    graph.module_prior_cache[did] = prior
    return prior


def get_candidate_modules(module_prior: np.ndarray, top_k: int = 64) -> Tuple[List[int], np.ndarray]:
    prior = np.asarray(module_prior, dtype=np.float32).reshape(-1)
    if prior.size == 0:
        return [], np.zeros((0,), dtype=np.float32)

    k = max(1, min(int(top_k), int(prior.size)))
    if k == prior.size:
        idx = np.argsort(prior)[::-1]
    else:
        part = np.argpartition(prior, -k)[-k:]
        idx = part[np.argsort(prior[part])[::-1]]

    ids = [int(i) for i in idx.tolist()]
    vals = prior[idx].astype(np.float32)
    return ids, vals


def compute_support_evidence(support: dict, candidate_module_ids: List[int], graph: GGMVGraph) -> np.ndarray:
    """Compute one support perturbation -> module evidence vector."""
    k = len(candidate_module_ids)
    if k == 0:
        return np.zeros((0,), dtype=np.float32)

    pert_id = str(support.get("perturbation_id", ""))
    cache_key = (pert_id, tuple(candidate_module_ids))
    cached = graph.support_evidence_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    gene_ids = support.get("labeled_gene_ids", [])
    labels = support.get("labels", [])
    n = min(len(gene_ids), len(labels))
    if n <= 0:
        out = np.zeros((k,), dtype=np.float32)
        graph.support_evidence_cache[cache_key] = out
        return out.copy()

    gene_ids_np = np.asarray(gene_ids[:n], dtype=np.int64)
    labels_np = np.asarray(labels[:n], dtype=np.float32)
    signs = 2.0 * labels_np - 1.0

    valid = (gene_ids_np >= 0) & (gene_ids_np < graph.num_genes)
    if not np.any(valid):
        out = np.zeros((k,), dtype=np.float32)
        graph.support_evidence_cache[cache_key] = out
        return out.copy()

    gene_ids_np = gene_ids_np[valid]
    signs = signs[valid]
    mat = graph.r_tilde[gene_ids_np][:, np.asarray(candidate_module_ids, dtype=np.int64)].toarray().astype(np.float32)
    evidence = (signs[:, None] * mat).mean(axis=0).astype(np.float32)
    graph.support_evidence_cache[cache_key] = evidence
    return evidence.copy()


def _target_jaccard(graph: GGMVGraph, did_a: int, did_b: int) -> float:
    if did_a < 0 or did_b < 0:
        return 0.0
    a = set(graph.get_targets(did_a))
    b = set(graph.get_targets(did_b))
    if not a and not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def compute_support_utility(query: dict, support: dict, graph: GGMVGraph, params: GGMVParams) -> float:
    if params.no_support_utility:
        return 1.0

    q_cell = int(query.get("cell_id", -1))
    s_cell = int(support.get("cell_id", -2))
    same_cell = 1.0 if q_cell == s_cell else 0.0

    q_drug = int(query.get("drug_id", -1))
    s_drug = int(support.get("drug_id", -1))
    target_overlap = _target_jaccard(graph, q_drug, s_drug)

    x = params.w0 + params.w1 * same_cell + params.w2 * target_overlap
    return float(_sigmoid(x))


def aggregate_support_evidence(
    query: dict,
    support_set: List[dict],
    candidate_module_ids: List[int],
    graph: GGMVGraph,
    params: GGMVParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = len(candidate_module_ids)
    if k == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    if not support_set:
        return (
            np.zeros((k,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((k,), dtype=np.float32),
        )

    agg = np.zeros((k,), dtype=np.float32)
    util_vals: List[float] = []
    per_support_evidence = np.zeros((k,), dtype=np.float32)

    for support in support_set:
        e_j = compute_support_evidence(support, candidate_module_ids, graph)
        u_j = compute_support_utility(query, support, graph, params)
        agg += float(u_j) * e_j
        per_support_evidence += e_j
        util_vals.append(float(u_j))

    if util_vals:
        per_support_evidence /= float(len(util_vals))

    return agg, np.asarray(util_vals, dtype=np.float32), per_support_evidence


def compute_mechanism_posterior(
    *,
    candidate_prior: np.ndarray,
    aggregated_evidence: np.ndarray,
    llm_mech_scores: np.ndarray,
    params: GGMVParams,
) -> np.ndarray:
    if candidate_prior.size == 0:
        return np.zeros((0,), dtype=np.float32)

    prior = np.asarray(candidate_prior, dtype=np.float32).reshape(-1)
    prior = np.maximum(prior, 0.0)
    if float(prior.sum()) <= 0:
        prior = np.ones_like(prior, dtype=np.float32) / float(prior.size)
    else:
        prior = prior / float(prior.sum())

    e = np.asarray(aggregated_evidence, dtype=np.float32).reshape(-1)
    if e.size != prior.size:
        e = np.zeros_like(prior)

    m = np.asarray(llm_mech_scores, dtype=np.float32).reshape(-1)
    if m.size != prior.size:
        m = np.zeros_like(prior)
    if params.no_llm_mech:
        m = np.zeros_like(prior)

    logits = np.log(prior + params.eps) + params.beta * e + params.gamma_mech * m

    if params.posterior_mode == "softmax":
        return _softmax(logits)
    return sparsemax(logits)


def compute_verification_score(query_gene_id: int, candidate_module_ids: List[int], posterior: np.ndarray, graph: GGMVGraph) -> float:
    if query_gene_id < 0 or query_gene_id >= graph.num_genes:
        return 0.0
    if len(candidate_module_ids) == 0:
        return 0.0

    row = graph.r_tilde[int(query_gene_id), np.asarray(candidate_module_ids, dtype=np.int64)].toarray().reshape(-1)
    score = float(np.dot(posterior.astype(np.float64), row.astype(np.float64)))
    return score


def compute_sufficiency_score(
    posterior: np.ndarray,
    utility_values: np.ndarray,
    params: GGMVParams,
    llm_sufficiency_score: float = 0.0,
) -> float:
    if params.no_sufficiency:
        return 0.0

    if posterior.size == 0:
        max_z = 0.0
        entropy = 0.0
    else:
        max_z = float(np.max(posterior))
        entropy = _safe_entropy(posterior, eps=params.eps)

    mean_u = float(np.mean(utility_values)) if utility_values.size > 0 else 0.0
    return (
        params.eta1 * max_z
        - params.eta2 * entropy
        + params.eta3 * mean_u
        + params.eta4 * float(llm_sufficiency_score)
    )


def calibrate_scores(base_scores: Dict[str, float], verification_score: float, sufficiency_score: float, params: GGMVParams) -> Dict[str, float]:
    v = 0.0 if params.no_verification else float(verification_score)
    a = 0.0 if params.no_sufficiency else float(sufficiency_score)

    s_yes_base = float(base_scores.get("yes", 0.0))
    s_no_base = float(base_scores.get("no", 0.0))
    s_ins_base = float(base_scores.get("insufficient", 0.0))

    s_yes = s_yes_base + params.lambda_v * v + params.gamma_a * a
    s_no = s_no_base - params.lambda_v * v + params.gamma_a * a
    s_ins = s_ins_base + params.b0 - params.b1 * a

    return {
        "yes": float(s_yes),
        "no": float(s_no),
        "insufficient": float(s_ins),
    }


def _top_pairs(ids: List[int], values: np.ndarray, top_n: int = 8) -> List[Tuple[int, float]]:
    if len(ids) == 0 or values.size == 0:
        return []
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.argsort(vals)[::-1]
    out: List[Tuple[int, float]] = []
    for i in order[: min(top_n, len(order))]:
        out.append((int(ids[int(i)]), float(vals[int(i)])))
    return out


def forward_ggmv(
    *,
    query: dict,
    support_set: List[dict],
    graph: GGMVGraph,
    params: GGMVParams,
    llm_mech_score_fn: Optional[Callable[[List[int]], np.ndarray]] = None,
    llm_sufficiency_score: float = 0.0,
) -> dict:
    """One deterministic GGMV forward pass for a query episode."""
    if graph.num_modules == 0:
        empty_scores = dict(query.get("base_scores", {}))
        empty_scores.setdefault("yes", 0.0)
        empty_scores.setdefault("no", 0.0)
        empty_scores.setdefault("insufficient", 0.0)
        return {
            "candidate_module_ids": [],
            "candidate_prior": np.zeros((0,), dtype=np.float32),
            "aggregated_evidence": np.zeros((0,), dtype=np.float32),
            "llm_mech_scores": np.zeros((0,), dtype=np.float32),
            "posterior": np.zeros((0,), dtype=np.float32),
            "verification_score": 0.0,
            "sufficiency_score": 0.0,
            "scores": empty_scores,
            "binary_margin": float(empty_scores["yes"] - empty_scores["no"]),
            "utility_values": np.zeros((0,), dtype=np.float32),
            "debug": {
                "top_prior_modules": [],
                "top_support_evidence_modules": [],
                "top_llm_mech_modules": [],
                "top_posterior_modules": [],
            },
        }

    if params.no_graph_prior:
        module_prior = graph.global_module_prior.copy()
    else:
        module_prior = compute_query_module_prior(int(query.get("drug_id", -1)), graph)

    candidate_ids, candidate_prior = get_candidate_modules(module_prior, top_k=params.top_k_modules)

    aggregated_evidence, utility_values, support_mean_evidence = aggregate_support_evidence(
        query=query,
        support_set=support_set,
        candidate_module_ids=candidate_ids,
        graph=graph,
        params=params,
    )

    if llm_mech_score_fn is not None and not params.no_llm_mech:
        llm_scores = np.asarray(llm_mech_score_fn(candidate_ids), dtype=np.float32).reshape(-1)
        if llm_scores.size != len(candidate_ids):
            llm_scores = np.zeros((len(candidate_ids),), dtype=np.float32)
    else:
        llm_scores = np.zeros((len(candidate_ids),), dtype=np.float32)

    posterior = compute_mechanism_posterior(
        candidate_prior=candidate_prior,
        aggregated_evidence=aggregated_evidence,
        llm_mech_scores=llm_scores,
        params=params,
    )

    v = compute_verification_score(
        query_gene_id=int(query.get("gene_id", -1)),
        candidate_module_ids=candidate_ids,
        posterior=posterior,
        graph=graph,
    )
    if params.no_verification:
        v = 0.0

    a = compute_sufficiency_score(
        posterior=posterior,
        utility_values=utility_values,
        params=params,
        llm_sufficiency_score=llm_sufficiency_score,
    )

    calibrated = calibrate_scores(
        base_scores=query.get("base_scores", {}),
        verification_score=v,
        sufficiency_score=a,
        params=params,
    )

    binary_margin = float(calibrated["yes"] - calibrated["no"])

    return {
        "candidate_module_ids": candidate_ids,
        "candidate_prior": candidate_prior,
        "aggregated_evidence": aggregated_evidence,
        "llm_mech_scores": llm_scores,
        "posterior": posterior,
        "verification_score": float(v),
        "sufficiency_score": float(a),
        "scores": calibrated,
        "binary_margin": binary_margin,
        "utility_values": utility_values,
        "debug": {
            "top_prior_modules": _top_pairs(candidate_ids, candidate_prior),
            "top_support_evidence_modules": _top_pairs(candidate_ids, support_mean_evidence),
            "top_llm_mech_modules": _top_pairs(candidate_ids, llm_scores),
            "top_posterior_modules": _top_pairs(candidate_ids, posterior),
        },
    }
