#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare DE/DIR CSV datasets from h5ad."""

import os
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm


def process_cell_line(
    *,
    adata_path: str,
    output_dir: str,
    cell_line_name: str,
    perturbation_col: str = "drug",
    control_value: str = "DMSO_TF",
    train_fraction: float = 0.3,
    seed: int = 42,
    fdr: float = 0.05,
    lfc: float = 0.25,
    pval_neg: float = 0.1,
    n_neg: int = 200,
) -> None:
    """
    Process a single cell line:
    - differential expression (DE) dataset
    - direction (DIR) dataset
    """
    print(f"\n{'='*20} Processing cell line: {cell_line_name} {'='*20}")

    try:
        adata = sc.read_h5ad(adata_path)
        print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    except FileNotFoundError:
        print(f"ERROR: file not found: {adata_path}")
        return

    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Running DE...")
    all_perturbations = adata.obs[perturbation_col].unique()
    drug_perturbations = [p for p in all_perturbations if p != control_value]

    if not drug_perturbations:
        print("WARNING: no perturbations found; skip")
        return

    sc.tl.rank_genes_groups(
        adata,
        groupby=perturbation_col,
        groups=drug_perturbations,
        reference=control_value,
        method="wilcoxon",
        corr_method="benjamini-hochberg",
    )

    print("Extracting DE results...")

    results_list = []
    for drug in tqdm(drug_perturbations):
        try:
            result_df = pd.DataFrame({
                "gene": adata.uns["rank_genes_groups"]["names"][drug],
                "logfoldchanges": adata.uns["rank_genes_groups"]["logfoldchanges"][drug],
                "pvals": adata.uns["rank_genes_groups"]["pvals"][drug],
                "pvals_adj": adata.uns["rank_genes_groups"]["pvals_adj"][drug],
            })
            result_df["pert"] = drug
            results_list.append(result_df)
        except KeyError:
            print(f"WARNING: missing results for pert {drug}")

    if not results_list:
        print("WARNING: no DE results extracted; skip")
        return

    results_df = pd.concat(results_list, ignore_index=True)

    np.random.seed(seed)
    perturbations_shuffled = drug_perturbations.copy()
    np.random.shuffle(perturbations_shuffled)
    split_index = int(len(perturbations_shuffled) * train_fraction)
    train_set = set(perturbations_shuffled[:split_index])
    test_set = set(perturbations_shuffled[split_index:])
    results_df["split"] = results_df["pert"].map(lambda p: "train" if p in train_set else "test")

    print(f"Total perturbations: {len(drug_perturbations)}")
    print(f"Train perturbations: {len(train_set)} | Test perturbations: {len(test_set)}")

    # Build DE dataset
    degs_mask = (results_df["pvals_adj"] < fdr) & (results_df["logfoldchanges"].abs() > lfc)
    degs_df = results_df[degs_mask].copy()
    degs_df["label"] = 1

    non_degs_candidates = results_df[results_df["pvals"] > pval_neg]

    def sample_group(group, n_samples):
        n = min(n_samples, len(group))
        return group.sample(n=n, random_state=seed) if n > 0 else None

    non_degs_df = non_degs_candidates.groupby("pert", group_keys=False).apply(sample_group, n_samples=n_neg)

    if non_degs_df is not None and not non_degs_df.empty:
        non_degs_df["label"] = 0
        final_labels_df = pd.concat([degs_df, non_degs_df], ignore_index=True)
    else:
        final_labels_df = degs_df

    final_labels_df = final_labels_df[["pert", "gene", "label", "split"]]

    os.makedirs(output_dir, exist_ok=True)
    de_output = os.path.join(output_dir, f"{cell_line_name}_DE.csv")
    final_labels_df.to_csv(de_output, index=False)
    print(f"Saved DE CSV: {de_output}")

    # Build DIR dataset
    degs_df["direction_label"] = np.where(degs_df["logfoldchanges"] > 0, 1, 0)
    dir_labels_df = degs_df[["pert", "gene", "direction_label", "split"]].copy()
    dir_labels_df.rename(columns={"direction_label": "label"}, inplace=True)

    dir_output = os.path.join(output_dir, f"{cell_line_name}_DIR.csv")
    dir_labels_df.to_csv(dir_output, index=False)
    print(f"Saved DIR CSV: {dir_output}")

    print("Done.")
