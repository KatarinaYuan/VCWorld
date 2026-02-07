#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build retrieval results JSON for drug-gene pairs."""

import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_drug_data(csv_file: str) -> Dict[str, List[dict]]:
    df = pd.read_csv(csv_file)
    data = {"train": [], "test": []}
    for _, row in df.iterrows():
        item = {
            "pert": row["pert"],
            "gene": row["gene"],
            "label": int(row["label"]),
            "split": row["split"],
        }
        data[item["split"]].append(item)
    return data


def load_similarity_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # expected formats:
    # - {item: [ {"Drug": "..."}, ...] } OR {item: ["x", ...]}
    # - {gene: {"direct_neighbors": [...], ...}} (use direct_neighbors if present)
    out = {}
    for key, vals in raw.items():
        if not vals:
            out[key] = []
            continue
        if isinstance(vals, dict):
            # gene similarity dict from graph results
            for field in ("direct_neighbors", "neighbors", "close_genes", "similar_genes", "top_genes", "two_hop_neighbors"):
                candidates = vals.get(field)
                if isinstance(candidates, list):
                    out[key] = [str(v) for v in candidates]
                    break
            else:
                out[key] = [str(v) for v in vals.values()]
            continue
        if isinstance(vals, list):
            if not vals:
                out[key] = []
            elif isinstance(vals[0], dict):
                if "Drug" in vals[0]:
                    out[key] = [v.get("Drug") for v in vals if v.get("Drug")]
                elif "Gene" in vals[0]:
                    out[key] = [v.get("Gene") for v in vals if v.get("Gene")]
                else:
                    out[key] = [str(v) for v in vals]
            else:
                out[key] = [str(v) for v in vals]
            continue
        out[key] = [str(vals)]
    return out


def build_seen_structures(train_data: List[dict]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    seen: Dict[str, List[str]] = {}
    seen_gene: Dict[str, List[str]] = {}
    for item in train_data:
        drug = item["pert"]
        gene = item["gene"]
        if drug not in seen:
            seen[drug] = []
        if gene not in seen[drug]:
            seen[drug].append(gene)
        if gene not in seen_gene:
            seen_gene[gene] = []
        if drug not in seen_gene[gene]:
            seen_gene[gene].append(drug)
    return seen, seen_gene


def get_drug_gene_pairs(*, drug: str, gene: str, close_drugs: List[str], close_genes: List[str],
                        seen: Dict[str, List[str]], seen_gene: Dict[str, List[str]],
                        budget: int, seed: int = 0) -> List[List[str]]:
    np.random.seed(seed)
    drug_pairs = []
    gene_pairs = []

    if drug in seen:
        for gene2 in close_genes:
            if gene2 in seen[drug]:
                drug_pairs.append([drug, gene2])
    elif gene in seen_gene:
        for drug2 in close_drugs:
            if drug2 in seen_gene[gene]:
                gene_pairs.append([drug2, gene])

    if len(drug_pairs) > budget:
        drug_pairs = [drug_pairs[i] for i in np.random.choice(len(drug_pairs), budget, replace=False)]
    if len(gene_pairs) > budget:
        gene_pairs = [gene_pairs[i] for i in np.random.choice(len(gene_pairs), budget, replace=False)]

    both_pairs = []
    for drug2 in close_drugs:
        for gene2 in close_genes:
            if drug2 in seen and gene2 in seen[drug2]:
                both_pairs.append([drug2, gene2])
    if len(both_pairs) > budget:
        both_pairs = [both_pairs[i] for i in np.random.choice(len(both_pairs), budget, replace=False)]

    cur_drug_pairs = []
    cur_gene_pairs = []
    drug_budget = budget - len(drug_pairs)
    gene_budget = budget - len(gene_pairs)

    if drug in seen and drug_budget > 0:
        for gene2 in seen[drug]:
            cur_drug_pairs.append([drug, gene2])
        if len(cur_drug_pairs) > drug_budget:
            cur_drug_pairs = [cur_drug_pairs[i] for i in np.random.choice(len(cur_drug_pairs), drug_budget, replace=False)]
    elif gene in seen_gene and gene_budget > 0:
        for drug2 in seen_gene[gene]:
            cur_gene_pairs.append([drug2, gene])
        if len(cur_gene_pairs) > gene_budget:
            cur_gene_pairs = [cur_gene_pairs[i] for i in np.random.choice(len(cur_gene_pairs), gene_budget, replace=False)]

    drug_pairs.extend(cur_drug_pairs)
    gene_pairs.extend(cur_gene_pairs)

    cur_drug_pairs = []
    cur_gene_pairs = []
    drug_budget = budget - len(drug_pairs)
    gene_budget = budget - len(gene_pairs)

    if drug_budget > 0:
        for drug2 in close_drugs:
            if drug2 not in seen:
                continue
            for gene2 in seen[drug2]:
                cur_drug_pairs.append([drug2, gene2])
        if len(cur_drug_pairs) > drug_budget:
            cur_drug_pairs = [cur_drug_pairs[i] for i in np.random.choice(len(cur_drug_pairs), drug_budget, replace=False)]
    elif gene_budget > 0:
        for gene2 in close_genes:
            if gene2 not in seen_gene:
                continue
            for drug2 in seen_gene[gene2]:
                cur_gene_pairs.append([drug2, gene2])
        if len(cur_gene_pairs) > gene_budget:
            cur_gene_pairs = [cur_gene_pairs[i] for i in np.random.choice(len(cur_gene_pairs), gene_budget, replace=False)]

    drug_pairs.extend(cur_drug_pairs)
    gene_pairs.extend(cur_gene_pairs)

    return drug_pairs + gene_pairs + both_pairs


def build_retrieval_results(*, data_csv: str, drug_sim_json: str, out_json: str,
                           gene_sim_json: str, budget: int = 10,
                           seed: int = 0, max_cases: Optional[int] = None,
                           case_split: str = "test") -> None:
    data = load_drug_data(data_csv)
    train_data = data["train"]
    eval_data = data[case_split]

    drug_similarity = load_similarity_json(drug_sim_json)
    gene_similarity = load_similarity_json(gene_sim_json)

    seen, seen_gene = build_seen_structures(train_data)

    unique_cases = []
    seen_pairs = set()
    for item in eval_data:
        key = (item["pert"], item["gene"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_cases.append(item)

    if max_cases and max_cases < len(unique_cases):
        random.seed(seed)
        unique_cases = random.sample(unique_cases, max_cases)

    results = []
    for item in unique_cases:
        drug = item["pert"]
        gene = item["gene"]
        close_drugs = drug_similarity.get(drug, [])[:budget]
        close_genes = gene_similarity.get(gene, [])[:budget]
        retrieved = get_drug_gene_pairs(
            drug=drug,
            gene=gene,
            close_drugs=close_drugs,
            close_genes=close_genes,
            seen=seen,
            seen_gene=seen_gene,
            budget=budget,
            seed=seed,
        )
        results.append({
            "test_case": {"drug": drug, "gene": gene},
            "retrieved_pairs": retrieved,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved retrieval results: {out_json} (cases: {len(results)})")
