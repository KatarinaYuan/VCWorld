#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for GGMV KG parser."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ggmv_graph import load_ggmv_graph


class TestGGMVGraphParser(unittest.TestCase):
    def test_parse_nodes_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            nodes = [
                {"node_type": "gene", "entity_id": "G1", "node_index": 0},
                {"node_type": "gene", "entity_id": "G2", "node_index": 1},
                {"node_type": "drug", "entity_id": "D1", "node_index": 2},
                {"node_type": "reactome_pathway", "entity_id": "M1", "node_index": 3},
            ]
            edges = [
                {
                    "src_type": "drug",
                    "dst_type": "gene",
                    "src_id": "D1",
                    "dst_id": "G1",
                    "relation_type": "drug_target",
                    "source": "DrugBank",
                },
                {
                    "src_type": "gene",
                    "dst_type": "reactome_pathway",
                    "src_id": "G1",
                    "dst_id": "M1",
                    "relation_type": "reactome_membership",
                    "source": "Reactome",
                },
                {
                    "src_type": "gene",
                    "dst_type": "gene",
                    "src_id": "G1",
                    "dst_id": "G2",
                    "relation_type": "bioplex_ppi",
                    "source": "BioPlex",
                },
            ]
            (d / "nodes.json").write_text(json.dumps(nodes), encoding="utf-8")
            (d / "edges.json").write_text(json.dumps(edges), encoding="utf-8")

            graph = load_ggmv_graph(kg_dir=str(d), alpha=0.1)
            self.assertEqual(graph.num_genes, 2)
            self.assertEqual(graph.num_modules, 1)
            d1 = graph.lookup_drug_id("D1")
            self.assertIsNotNone(d1)
            self.assertEqual(graph.get_targets(int(d1))[0], graph.lookup_gene_id("G1"))


if __name__ == "__main__":
    unittest.main()
