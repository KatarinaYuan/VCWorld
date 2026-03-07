#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for Phase-2 GGMV LLM mechanism scorer."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from ggmv_graph import GGMVGraph
from ggmv_llm import LLMMechanismScorer, build_query_support_summary


class _DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        # Simple deterministic byte-level tokenization.
        del add_special_tokens
        ids = [(ord(ch) % 251) + 1 for ch in text]
        return SimpleNamespace(input_ids=ids)


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 512, hidden_size: int = 16) -> None:
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(13)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(vocab_size, hidden_size, generator=g))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        del attention_mask, output_hidden_states
        h = self.embedding(input_ids)
        logits = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.embedding.num_embeddings, dtype=h.dtype, device=h.device
        )
        if return_dict:
            return SimpleNamespace(hidden_states=[h], logits=logits)
        return (logits,)


def _toy_graph() -> GGMVGraph:
    gene_to_idx = {"G1": 0, "G2": 1, "G3": 2}
    idx_to_gene = ["G1", "G2", "G3"]

    drug_to_idx = {"d1": 0, "d2": 1}
    idx_to_drug = ["d1", "d2"]

    module_to_idx = {"reactome::m1": 0, "complex::m2": 1}
    idx_to_module = ["M1", "M2"]
    module_types = ["reactome", "complex"]

    drug_to_targets = {0: [0], 1: [2]}

    r = sp.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0],
                [0.5, 0.2],
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
        gene_module_matrix=r,
        gene_gene_adj=p,
        r_tilde=r_tilde,
        module_gene_lists=module_gene_lists,
        global_module_prior=global_prior,
    )


class TestGGMVLLMSmoke(unittest.TestCase):
    def test_phase2_context_and_module_scores(self):
        graph = _toy_graph()
        query = {"query_id": "q0", "cell_id": 0, "drug_id": 0, "gene_id": 1, "drug_name": "d1", "gene_name": "G2"}
        support_set = [
            {
                "perturbation_id": "s1",
                "cell_id": 0,
                "drug_id": 1,
                "labeled_gene_ids": [0, 1, 2],
                "labels": [1, 0, 1],
            }
        ]

        summary = build_query_support_summary(
            query=query, support_set=support_set, graph=graph, max_support=4, max_genes_per_support=3
        )
        self.assertIn("Query drug:", summary)
        self.assertIn("Support#1", summary)

        scorer = LLMMechanismScorer(
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            device=torch.device("cpu"),
            max_tokens=256,
            module_top_genes=4,
            use_prompt_context=True,
            prompt_context_tokens=16,
            max_support_in_summary=4,
            max_genes_per_support=3,
            overlap_bonus_weight=0.2,
        )

        context_vec, _ = scorer.encode_query_support_context(
            query=query,
            support_set=support_set,
            graph=graph,
            prompt_ids=[1, 2, 3, 4, 5],
            summary_text=summary,
        )
        self.assertGreater(int(context_vec.size), 0)

        scores = scorer.score_candidate_modules_from_context(
            context_vec=context_vec,
            query=query,
            candidate_module_ids=[0, 1],
            graph=graph,
        )
        self.assertEqual(scores.shape[0], 2)
        self.assertTrue(np.isfinite(scores).all())

        suff = scorer.score_sufficiency(context_vec=context_vec, support_set=support_set)
        self.assertGreaterEqual(float(suff), 0.0)
        self.assertLessEqual(float(suff), 1.0)


if __name__ == "__main__":
    unittest.main()
