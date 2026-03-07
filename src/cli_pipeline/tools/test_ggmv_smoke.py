#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal smoke tests for GGMV core logic."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from ggmv_core import GGMVParams, forward_ggmv, sparsemax
from ggmv_graph import GGMVGraph


def _toy_graph() -> GGMVGraph:
    gene_to_idx = {"G1": 0, "G2": 1, "G3": 2}
    idx_to_gene = ["G1", "G2", "G3"]

    drug_to_idx = {"d1": 0, "d2": 1}
    idx_to_drug = ["d1", "d2"]

    module_to_idx = {"reactome::m1": 0, "complex::m2": 1}
    idx_to_module = ["M1", "M2"]
    module_types = ["reactome", "complex"]

    drug_to_targets = {0: [0], 1: [2]}

    R = sp.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0],
                [0.5, 0.2],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    P = sp.csr_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    R_tilde = ((1.0 - 0.1) * R + 0.1 * P.dot(R)).tocsr().astype(np.float32)

    module_gene_lists = [[0, 1], [1, 2]]
    global_prior = np.asarray([0.5, 0.5], dtype=np.float32)

    return GGMVGraph(
        gene_to_idx=gene_to_idx,
        idx_to_gene=idx_to_gene,
        drug_to_idx=drug_to_idx,
        idx_to_drug=idx_to_drug,
        module_to_idx=module_to_idx,
        idx_to_module=idx_to_module,
        module_types=module_types,
        drug_to_targets=drug_to_targets,
        gene_module_matrix=R,
        gene_gene_adj=P,
        r_tilde=R_tilde,
        module_gene_lists=module_gene_lists,
        global_module_prior=global_prior,
    )


class TestGGMVSmoke(unittest.TestCase):
    def test_sparsemax_simplex(self):
        out = sparsemax(np.asarray([2.0, 1.0, -0.5], dtype=np.float32))
        self.assertTrue(np.all(out >= -1e-8))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)

    def test_forward_episode(self):
        graph = _toy_graph()
        params = GGMVParams(top_k_modules=2, posterior_mode="sparsemax")

        query = {
            "query_id": "q0",
            "cell_id": 0,
            "drug_id": 0,
            "gene_id": 0,
            "base_scores": {"yes": 0.2, "no": 0.1, "insufficient": 0.0},
        }
        support_set = [
            {
                "perturbation_id": "s1",
                "cell_id": 0,
                "drug_id": 1,
                "labeled_gene_ids": [0, 1, 2],
                "labels": [1, 0, 0],
            }
        ]

        out = forward_ggmv(query=query, support_set=support_set, graph=graph, params=params)
        self.assertIn("scores", out)
        self.assertIn("posterior", out)
        self.assertEqual(len(out["candidate_module_ids"]), 2)
        self.assertAlmostEqual(float(out["posterior"].sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
