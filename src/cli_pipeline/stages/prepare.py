#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare DE/DIR CSV datasets from h5ad."""

import os
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm


def _print_split_stats(df: pd.DataFrame, name: str) -> None:
    total_rows = len(df)
    train_rows = int((df["split"] == "train").sum())
    test_rows = int((df["split"] == "test").sum())
    train_perts = int(df.loc[df["split"] == "train", "pert"].nunique())
    test_perts = int(df.loc[df["split"] == "test", "pert"].nunique())
    train_genes = int(df.loc[df["split"] == "train", "gene"].nunique())
    test_genes = int(df.loc[df["split"] == "test", "gene"].nunique())

    print(f"{name} stats:")
    print(f"  rows total/train/test: {total_rows}/{train_rows}/{test_rows}")
    print(f"  unique perts train/test: {train_perts}/{test_perts}")
    print(f"  unique genes train/test: {train_genes}/{test_genes}")


def _sample_n_per_pert(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rng = np.random.default_rng(seed)
    tmp = df.copy()
    tmp["_rand"] = rng.random(len(tmp))
    sampled = (
        tmp.sort_values(["pert", "_rand"])
        .groupby("pert", group_keys=False)
        .head(n)
        .drop(columns="_rand")
        .copy()
    )
    return sampled


def process_cell_line(
    *,
    adata_path: str,
    output_dir: str,
    cell_line_name: str,
    perturbation_col: str = "drug",
    control_value: str = "DMSO_TF",
    train_fraction: float = 0.3,
    split_mode: str = "random_perturbation",
    k_support_perturbations: Optional[int] = None,
    m_genes_per_perturbation: Optional[int] = None,
    fixed_test_fraction: Optional[float] = None,
    fixed_test_seed: Optional[int] = None,
    output_tag: Optional[str] = None,
    overwrite: bool = False,
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

    if perturbation_col not in adata.obs:
        raise ValueError(f"Column '{perturbation_col}' not found in adata.obs")

    print("Raw data stats:")
    print(f"  obs: {adata.n_obs}, vars: {adata.n_vars}")
    print(f"  perturbation column: {perturbation_col}")
    print(f"  unique perturbations (incl control): {adata.obs[perturbation_col].nunique()}")
    pert_counts = adata.obs[perturbation_col].value_counts()
    control_cells = int((adata.obs[perturbation_col] == control_value).sum())
    print(f"  control value: {control_value}, control cells: {control_cells}")
    print(f"  perturbation cells min/median/max: {int(pert_counts.min())}/{float(pert_counts.median()):.1f}/{int(pert_counts.max())}")

    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Running DE...")
    all_perturbations = adata.obs[perturbation_col].unique()
    drug_perturbations = [p for p in all_perturbations if p != control_value]
    drug_cell_counts = adata.obs[adata.obs[perturbation_col] != control_value][perturbation_col].value_counts()
    if len(drug_cell_counts) > 0:
        print(
            "Drug perturbation cell stats min/median/max: "
            f"{int(drug_cell_counts.min())}/{float(drug_cell_counts.median()):.1f}/{int(drug_cell_counts.max())}"
        )

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

    rng = np.random.default_rng(seed)
    perturbations_shuffled = drug_perturbations.copy()
    rng.shuffle(perturbations_shuffled)

    unused_set = set()
    if split_mode == "random_perturbation":
        split_index = int(len(perturbations_shuffled) * train_fraction)
        train_set = set(perturbations_shuffled[:split_index])
        test_set = set(perturbations_shuffled[split_index:])
    elif split_mode == "k_perturbation_fixed_genes":
        if k_support_perturbations is None or m_genes_per_perturbation is None:
            raise ValueError(
                "split_mode=k_perturbation_fixed_genes requires "
                "--k-support-perturbations and --m-genes-per-perturbation."
            )
        if k_support_perturbations < 0 or m_genes_per_perturbation <= 0:
            raise ValueError(
                "k_support_perturbations must be >= 0 and m_genes_per_perturbation must be > 0."
            )
        if fixed_test_fraction is not None:
            if not 0.0 < fixed_test_fraction < 1.0:
                raise ValueError("fixed_test_fraction must be in (0, 1).")
            if len(perturbations_shuffled) < 2:
                raise ValueError("Need at least 2 perturbations to build disjoint train/test.")

            test_rng_seed = seed if fixed_test_seed is None else fixed_test_seed
            test_rng = np.random.default_rng(test_rng_seed)
            perturbations_for_test = drug_perturbations.copy()
            test_rng.shuffle(perturbations_for_test)

            n_total = len(perturbations_for_test)
            n_test = int(round(n_total * fixed_test_fraction))
            n_test = max(1, min(n_total - 1, n_test))
            print("Split fixed test data: n_total: ", n_total, "n_test: ", n_test)

            test_set = set(perturbations_for_test[:n_test])
            support_pool = [p for p in perturbations_for_test if p not in test_set]
            if k_support_perturbations > len(support_pool):
                raise ValueError(
                    f"k_support_perturbations={k_support_perturbations} exceeds support pool "
                    f"size={len(support_pool)} after fixed test split."
                )
            train_set = set(support_pool[:k_support_perturbations])
            unused_set = set(support_pool[k_support_perturbations:])
            print(
                "Fixed test enabled: "
                f"fraction={fixed_test_fraction}, test_seed={test_rng_seed}, "
                f"test_perts={len(test_set)}, support_pool={len(support_pool)}, unused_perts={len(unused_set)}"
            )
            if k_support_perturbations == 0:
                print("Zero-shot mode enabled: no support perturbations selected for training.")
        else:
            k = min(k_support_perturbations, len(perturbations_shuffled))
            train_set = set(perturbations_shuffled[:k])
            test_set = set(perturbations_shuffled[k:])
            if k == 0:
                print("Zero-shot mode enabled: no support perturbations selected for training.")
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    def _assign_split(pert: str) -> str:
        if pert in train_set:
            return "train"
        if pert in test_set:
            return "test"
        return "unused"

    results_df["split"] = results_df["pert"].map(_assign_split)

    print(f"Split mode: {split_mode}")
    print(f"Total perturbations: {len(drug_perturbations)}")
    print(f"Train perturbations: {len(train_set)} | Test perturbations: {len(test_set)}")
    if unused_set:
        print(f"Unused perturbations: {len(unused_set)}")

    # Build DE dataset
    degs_mask = (results_df["pvals_adj"] < fdr) & (results_df["logfoldchanges"].abs() > lfc)
    degs_df = results_df[degs_mask].copy()
    degs_df["label"] = 1
    print(f"DE positives: {len(degs_df)}")

    non_degs_candidates = results_df[results_df["pvals"] > pval_neg]
    print(f"DE negatives candidate pool: {len(non_degs_candidates)}")

    non_degs_df = _sample_n_per_pert(non_degs_candidates, n=n_neg, seed=seed)

    if non_degs_df is not None and not non_degs_df.empty:
        non_degs_df["label"] = 0
        final_labels_df = pd.concat([degs_df, non_degs_df], ignore_index=True)
    else:
        final_labels_df = degs_df
    label_counts = final_labels_df["label"].value_counts(dropna=False).to_dict()
    print(f"DE label counts before A2 resampling: {label_counts}")

    if split_mode == "k_perturbation_fixed_genes":
        m = int(m_genes_per_perturbation)
        train_rows = final_labels_df[final_labels_df["pert"].isin(train_set)]
        sampled_train_rows = _sample_n_per_pert(train_rows, n=m, seed=seed)

        sampled_train_rows["split"] = "train"
        test_rows = final_labels_df[final_labels_df["pert"].isin(test_set)].copy()
        test_rows["split"] = "test"
        final_labels_df = pd.concat([sampled_train_rows, test_rows], ignore_index=True)

        realized_train_n = int((final_labels_df["split"] == "train").sum())
        print(
            f"A2 sampling: target train tuples = {len(train_set)} * {m} = {len(train_set) * m}, "
            f"realized train tuples = {realized_train_n}"
        )

    final_labels_df = final_labels_df[final_labels_df["split"].isin(["train", "test"])].copy()
    final_labels_df = final_labels_df[["pert", "gene", "label", "split"]]
    _print_split_stats(final_labels_df, "DE")

    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{output_tag}" if output_tag else ""
    de_output = os.path.join(output_dir, f"{cell_line_name}{tag}_DE.csv")
    if os.path.exists(de_output) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {de_output}. "
            "Use --output-tag to create a unique file name or pass --overwrite to replace."
        )
    final_labels_df.to_csv(de_output, index=False)
    print(f"Saved DE CSV: {de_output}")

    # Build DIR dataset
    degs_df["direction_label"] = np.where(degs_df["logfoldchanges"] > 0, 1, 0)
    dir_labels_df = degs_df[["pert", "gene", "direction_label", "split"]].copy()
    dir_labels_df.rename(columns={"direction_label": "label"}, inplace=True)

    if split_mode == "k_perturbation_fixed_genes":
        m = int(m_genes_per_perturbation)
        dir_train_rows = dir_labels_df[dir_labels_df["pert"].isin(train_set)]
        sampled_dir_train_rows = _sample_n_per_pert(dir_train_rows, n=m, seed=seed)
        sampled_dir_train_rows["split"] = "train"
        dir_test_rows = dir_labels_df[dir_labels_df["pert"].isin(test_set)].copy()
        dir_test_rows["split"] = "test"
        dir_labels_df = pd.concat([sampled_dir_train_rows, dir_test_rows], ignore_index=True)

        realized_dir_train_n = int((dir_labels_df["split"] == "train").sum())
        print(
            f"A2 sampling (DIR): target train tuples = {len(train_set)} * {m} = {len(train_set) * m}, "
            f"realized train tuples = {realized_dir_train_n}"
        )

    dir_labels_df = dir_labels_df[dir_labels_df["split"].isin(["train", "test"])].copy()
    _print_split_stats(dir_labels_df, "DIR")

    dir_output = os.path.join(output_dir, f"{cell_line_name}{tag}_DIR.csv")
    if os.path.exists(dir_output) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {dir_output}. "
            "Use --output-tag to create a unique file name or pass --overwrite to replace."
        )
    dir_labels_df.to_csv(dir_output, index=False)
    print(f"Saved DIR CSV: {dir_output}")

    print("Done.")
