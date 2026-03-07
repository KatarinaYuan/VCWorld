#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for MSLD core components."""

from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp
import torch

from msld_graph import MSLDGraph, compute_drug_prior, compute_support_evidence, verify_mechanism_posterior
from msld_model import MSLDHeadConfig, MSLDHeads, compute_sufficiency_score


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
                [0.6, 0.2],
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
    global_prior = np.asarray([0.55, 0.45], dtype=np.float32)
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


class TestMSLDSmoke(unittest.TestCase):
    def test_teacher_posterior(self):
        graph = _toy_graph()
        prior = compute_drug_prior(graph, 0)
        evidence = compute_support_evidence(graph, gene_ids=[0, 1, 2], labels=[1, 0, 1])
        proposer = np.asarray([0.2, -0.1], dtype=np.float32)
        z = verify_mechanism_posterior(
            proposer_logits=proposer,
            graph_prior=prior,
            support_evidence=evidence,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            top_k=2,
        )
        self.assertEqual(z.shape[0], 2)
        self.assertTrue(np.all(z >= -1e-8))
        self.assertAlmostEqual(float(z.sum()), 1.0, places=5)

    def test_heads_forward(self):
        cfg = MSLDHeadConfig(hidden_dim=16, num_modules=8, module_emb_dim=4, label_hidden_dim=12, dropout=0.0)
        heads = MSLDHeads(cfg)
        h = torch.randn(5, 16)
        mech_logits = heads.mechanism_logits(h)
        out = heads.predict_margin(h, mech_logits, base_margin=torch.randn(5))
        self.assertEqual(mech_logits.shape, (5, 8))
        self.assertEqual(out["z_hat"].shape, (5, 8))
        self.assertEqual(out["m_emb"].shape, (5, 4))
        self.assertEqual(out["margin"].shape, (5,))

        a = compute_sufficiency_score(out["z_hat"], out["margin"], eta1=1.0, eta2=1.0, eta3=1.0)
        self.assertEqual(a.shape, (5,))


if __name__ == "__main__":
    unittest.main()
