#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for GGMV runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp

from ggmv_graph import GGMVGraph
from ggmv_runtime import build_prompt_record_lookup, build_support_adaptation_token_ids, select_global_support_set


class _DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[(ord(ch) % 127) + 1 for ch in text][:256])

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool = False):
        del tokenize
        body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            body += "\nassistant:"
        return body


def _toy_graph() -> GGMVGraph:
    gene_to_idx = {"G1": 0, "G2": 1, "G3": 2}
    idx_to_gene = ["G1", "G2", "G3"]
    drug_to_idx = {"d1": 0, "d2": 1}
    idx_to_drug = ["d1", "d2"]
    module_to_idx = {"reactome::m1": 0}
    idx_to_module = ["M1"]
    module_types = ["reactome"]

    drug_to_targets = {0: [0], 1: [2]}
    r = sp.csr_matrix(np.asarray([[1.0], [0.3], [0.7]], dtype=np.float32))
    p = sp.eye(3, dtype=np.float32, format="csr")
    r_tilde = r.copy()
    module_gene_lists = [[0, 1, 2]]
    global_prior = np.asarray([1.0], dtype=np.float32)

    return GGMVGraph(
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


class TestGGMVRuntimeSmoke(unittest.TestCase):
    def test_global_support_is_deterministic(self):
        support_pool = [
            {"perturbation_id": "p1", "labeled_gene_ids": [0], "labels": [1]},
            {"perturbation_id": "p2", "labeled_gene_ids": [1], "labels": [0]},
            {"perturbation_id": "p3", "labeled_gene_ids": [2], "labels": [1]},
        ]
        s1 = select_global_support_set(support_pool, max_support_perturbations=2, seed=42)
        s2 = select_global_support_set(support_pool, max_support_perturbations=2, seed=42)
        self.assertEqual([x["perturbation_id"] for x in s1], [x["perturbation_id"] for x in s2])

    def test_build_support_adaptation_examples(self):
        graph = _toy_graph()
        records = [
            SimpleNamespace(
                pert="d1",
                gene="G1",
                system_prompt="sys",
                user_input="u1",
                output_text="yes",
            ),
            SimpleNamespace(
                pert="d2",
                gene="G3",
                system_prompt="sys",
                user_input="u2",
                output_text="no",
            ),
        ]
        lookup = build_prompt_record_lookup(records)
        support_set = [
            {"perturbation_id": "d1", "labeled_gene_ids": [0], "labels": [1]},
            {"perturbation_id": "d2", "labeled_gene_ids": [2], "labels": [0]},
        ]
        token_ids = build_support_adaptation_token_ids(
            tokenizer=_DummyTokenizer(),
            support_set=support_set,
            graph=graph,
            prompt_record_lookup=lookup,
            max_examples=8,
            max_tokens=128,
        )
        self.assertEqual(len(token_ids), 2)
        self.assertTrue(all(len(x) > 0 for x in token_ids))


if __name__ == "__main__":
    unittest.main()
