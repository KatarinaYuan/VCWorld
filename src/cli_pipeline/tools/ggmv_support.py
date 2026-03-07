#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Support-set construction and counterfactual shuffling for GGMV."""

from __future__ import annotations

import csv
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from ggmv_graph import GGMVGraph


def load_label_rows(labels_csv: str) -> List[dict]:
    """Load raw label rows from CSV with expected columns pert,gene,label,split."""
    rows: List[dict] = []
    with open(labels_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_support_pool_from_labels(
    *,
    labels_csv: str,
    graph: GGMVGraph,
    split: str = "train",
    default_cell_id: int = 0,
    max_genes_per_perturbation: int = 256,
) -> List[dict]:
    """Build perturbation-organized support pool from labels CSV."""
    rows = load_label_rows(labels_csv)
    grouped: Dict[str, Dict[int, int]] = {}

    split_norm = split.strip().lower()
    for row in rows:
        row_split = str(row.get("split", "")).strip().lower()
        if split_norm and row_split != split_norm:
            continue

        pert = str(row.get("pert", "")).strip()
        gene = str(row.get("gene", "")).strip()
        if not pert or not gene:
            continue

        try:
            label = int(row.get("label", 0))
        except (TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue

        gid = graph.lookup_gene_id(gene)
        if gid is None:
            continue

        by_gene = grouped.setdefault(pert, {})
        by_gene[int(gid)] = int(label)

    supports: List[dict] = []
    for pert in sorted(grouped.keys()):
        pairs = sorted(grouped[pert].items(), key=lambda x: x[0])
        if max_genes_per_perturbation > 0:
            pairs = pairs[:max_genes_per_perturbation]
        if not pairs:
            continue

        drug_id = graph.lookup_drug_id(pert)
        supports.append(
            {
                "perturbation_id": pert,
                "cell_id": int(default_cell_id),
                "drug_id": int(drug_id) if drug_id is not None else -1,
                "labeled_gene_ids": [int(g) for g, _ in pairs],
                "labels": [int(y) for _, y in pairs],
            }
        )

    return supports


def build_query_from_record(
    *,
    rec,
    label_row: dict,
    graph: GGMVGraph,
    base_scores: Dict[str, float],
    default_cell_id: int = 0,
) -> dict:
    """Build GGMV query dict from prompt record + labels row + base scores."""
    pert = str(rec.pert).strip() if rec.pert else ""
    gene = str(rec.gene).strip() if rec.gene else ""

    drug_id = graph.lookup_drug_id(pert)
    gene_id = graph.lookup_gene_id(gene)

    query: dict = {
        "query_id": str(rec.prompt_id) if rec.prompt_id is not None else f"idx_{rec.idx}",
        "cell_id": int(default_cell_id),
        "drug_id": int(drug_id) if drug_id is not None else -1,
        "gene_id": int(gene_id) if gene_id is not None else -1,
        "drug_name": pert,
        "gene_name": gene,
        "base_scores": {
            "yes": float(base_scores.get("yes", 0.0)),
            "no": float(base_scores.get("no", 0.0)),
            "insufficient": float(base_scores.get("insufficient", 0.0)),
        },
    }

    if label_row is not None and "label" in label_row:
        try:
            query["y"] = int(label_row["label"])
        except (TypeError, ValueError):
            pass

    return query


def _target_jaccard(graph: GGMVGraph, did_a: int, did_b: int) -> float:
    if did_a < 0 or did_b < 0:
        return 0.0
    a = set(graph.get_targets(did_a))
    b = set(graph.get_targets(did_b))
    if not a and not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def select_support_set_for_query(
    *,
    query: dict,
    support_pool: List[dict],
    graph: GGMVGraph,
    max_support_perturbations: int = 32,
    include_same_drug: bool = False,
) -> List[dict]:
    """Pick a support subset for one query using deterministic utility-style ranking."""
    if not support_pool:
        return []

    q_drug = int(query.get("drug_id", -1))
    q_cell = int(query.get("cell_id", -1))
    q_name = str(query.get("drug_name", "")).strip().lower()

    scored = []
    for support in support_pool:
        s_drug = int(support.get("drug_id", -1))
        s_name = str(support.get("perturbation_id", "")).strip().lower()

        if not include_same_drug:
            if q_drug >= 0 and s_drug == q_drug:
                continue
            if q_name and s_name and q_name == s_name:
                continue

        same_cell = 1.0 if int(support.get("cell_id", -2)) == q_cell else 0.0
        jacc = _target_jaccard(graph, q_drug, s_drug)
        n_labels = len(support.get("labels", []))
        score = 2.0 * same_cell + jacc + 1e-3 * float(n_labels)
        scored.append((score, support))

    if not scored:
        fallback = support_pool[: max(1, max_support_perturbations)]
        return [deepcopy(x) for x in fallback]

    scored.sort(key=lambda x: x[0], reverse=True)
    keep = [deepcopy(s) for _, s in scored[: max(1, max_support_perturbations)]]
    return keep


def make_shuffled_support(support_set: List[dict], mode: str, rng: np.random.Generator) -> List[dict]:
    """Counterfactual support generator for ablations / training loss."""
    supports = [deepcopy(s) for s in support_set]
    n = len(supports)
    if n <= 1:
        return supports

    if mode == "shuffle_perturbations":
        perm = list(rng.permutation(n))
        out = [deepcopy(x) for x in supports]
        for i in range(n):
            src = supports[perm[i]]
            out[i]["perturbation_id"] = src.get("perturbation_id")
            out[i]["drug_id"] = src.get("drug_id", -1)
            out[i]["cell_id"] = src.get("cell_id", -1)
        return out

    if mode == "shuffle_labels":
        for s in supports:
            labels = list(s.get("labels", []))
            if len(labels) <= 1:
                continue
            perm = list(rng.permutation(len(labels)))
            s["labels"] = [labels[i] for i in perm]
        return supports

    raise ValueError(f"Unknown shuffle mode: {mode}")
