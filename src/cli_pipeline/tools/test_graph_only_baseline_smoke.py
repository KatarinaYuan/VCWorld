#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for graph-only baseline components."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from graph_only_baseline_model import (
    GraphOnlyGeneScorer,
    GraphOnlyModuleProposer,
    GraphOnlyPerturbationPredictor,
)
from graph_only_baseline_utils import (
    GraphOnlyPerturbation,
    build_gene_feature_matrix,
    build_module_feature_matrix,
    sample_episode_for_perturbation,
)
from msld_graph import MSLDGraph


def _toy_graph() -> MSLDGraph:
    gene_to_idx = {"G1": 0, "G2": 1, "G3": 2}
    idx_to_gene = ["G1", "G2", "G3"]
    drug_to_idx = {"d1": 0}
    idx_to_drug = ["d1"]
    module_to_idx = {"reactome::m1": 0, "go_term::m2": 1}
    idx_to_module = ["M1", "M2"]
    module_types = ["reactome", "go_term"]

    r = sp.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    p = sp.csr_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    r_tilde = ((1.0 - 0.1) * r + 0.1 * p.dot(r)).tocsr().astype(np.float32)
    global_prior = np.asarray([0.6, 0.4], dtype=np.float32)
    module_gene_lists = [[0, 1], [1, 2]]
    drug_to_targets = {0: [0, 1]}
    return MSLDGraph(
        gene_to_idx=gene_to_idx,
        idx_to_gene=idx_to_gene,
        drug_to_idx=drug_to_idx,
        idx_to_drug=idx_to_drug,
        module_to_idx=module_to_idx,
        idx_to_module=idx_to_module,
        module_types=module_types,
        drug_to_targets=drug_to_targets,
        gene_module_matrix=r,
        gene_gene_adj=p,
        r_tilde=r_tilde,
        module_gene_lists=module_gene_lists,
        global_module_prior=global_prior,
    )


class TestGraphOnlyBaselineSmoke(unittest.TestCase):
    def test_feature_builders(self):
        graph = _toy_graph()
        mod_feat, mod_names, aux = build_module_feature_matrix(
            graph=graph,
            drug_id=0,
            support_gene_ids=[0, 1, 2],
            support_labels=[1, 0, 1],
            proposer_feature_set="full",
        )
        self.assertEqual(mod_feat.shape, (2, 4))
        self.assertEqual(mod_names, ["target_to_module_score", "positive_support_score", "negative_support_score", "contrast_score"])
        self.assertEqual(aux["module_prior"].shape[0], 2)

        q = np.asarray([0.7, 0.3], dtype=np.float32)
        gene_feat, gene_names, _ = build_gene_feature_matrix(
            graph=graph,
            query_gene_ids=[0, 2],
            q_probs=q,
            target_module_distribution=aux["target_module_distribution"],
            gene_feature_set="full",
            include_top_module_coverage=True,
            top_k_modules=2,
        )
        self.assertEqual(gene_feat.shape, (2, 3))
        self.assertEqual(gene_names, ["module_alignment", "target_gene_proximity", "top_module_coverage_count"])

    def test_predictor_forward(self):
        graph = _toy_graph()
        mod_feat, _, aux = build_module_feature_matrix(
            graph=graph,
            drug_id=0,
            support_gene_ids=[0, 1, 2],
            support_labels=[1, 0, 1],
            proposer_feature_set="full",
        )
        proposer = GraphOnlyModuleProposer(feature_dim=4, arch="linear")
        gene_scorer = GraphOnlyGeneScorer(feature_dim=3, arch="mlp", hidden_dim=8)
        model = GraphOnlyPerturbationPredictor(proposer=proposer, gene_scorer=gene_scorer)

        with torch.no_grad():
            mod_out = model.infer_module_posterior(torch.tensor(mod_feat, dtype=torch.float32))
            q = mod_out["q"].detach().cpu().numpy()
            gene_feat, _, _ = build_gene_feature_matrix(
                graph=graph,
                query_gene_ids=[0, 2],
                q_probs=q,
                target_module_distribution=aux["target_module_distribution"],
                gene_feature_set="full",
                include_top_module_coverage=True,
                top_k_modules=2,
            )
            logits = model.score_genes(torch.tensor(gene_feat, dtype=torch.float32))

        self.assertEqual(mod_out["module_logits"].shape, (2,))
        self.assertAlmostEqual(float(mod_out["q"].sum()), 1.0, places=5)
        self.assertEqual(logits.shape, (2,))

    def test_episode_split_reproducibility(self):
        pert = GraphOnlyPerturbation(
            perturbation_key="0::d1",
            pert="d1",
            cell_id=0,
            drug_id=0,
            gene_ids=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
            labels=np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float32),
        )
        ep_a = sample_episode_for_perturbation(
            pert,
            seed=123,
            epoch=2,
            support_fraction=0.5,
            min_support_size=2,
            min_query_size=2,
            max_support_size=4,
        )
        ep_b = sample_episode_for_perturbation(
            pert,
            seed=123,
            epoch=2,
            support_fraction=0.5,
            min_support_size=2,
            min_query_size=2,
            max_support_size=4,
        )
        assert ep_a is not None and ep_b is not None
        self.assertTrue(np.array_equal(ep_a["support_gene_ids"], ep_b["support_gene_ids"]))
        self.assertTrue(np.array_equal(ep_a["query_gene_ids"], ep_b["query_gene_ids"]))


if __name__ == "__main__":
    unittest.main()

