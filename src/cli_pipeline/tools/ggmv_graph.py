#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph parsing and sparse feature construction for GGMV."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _canon_gene(name: object) -> str:
    if name is None:
        return ""
    return str(name).strip().upper()


def _canon_drug(name: object) -> str:
    if name is None:
        return ""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _is_nonempty(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in {"nan", "none", "null"}


def _safe_float(value: object, default: float = 1.0) -> float:
    try:
        v = float(value)
        if math.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Python's json parser accepts NaN/Infinity; this is useful for current KG dumps.
    return json.loads(content)


def _classify_module_type(*, node_type: str, relation_type: str, source: str, node_name: str) -> Optional[str]:
    blob = " ".join([node_type, relation_type, source, node_name]).lower()
    if any(x in blob for x in ["cellular_component", "go_cc", "location"]):
        return None
    if "reactome" in blob:
        return "reactome"
    if any(x in blob for x in ["go_bp", "biological_process", "biological process"]):
        return "go_bp"
    if any(x in blob for x in ["go_mf", "molecular_function", "molecular function"]):
        return "go_mf"
    if "complex" in blob:
        return "complex"
    # Weak fallback for generic pathway nodes.
    if "pathway" in blob and "reactome" in source:
        return "reactome"
    return None


def _is_ppi_edge(relation_type: str, source: str) -> bool:
    rel = relation_type.lower()
    src = source.lower()
    if "bioplex" in rel or "string" in rel:
        return True
    if "ppi" in rel and any(x in src for x in ["bioplex", "string"]):
        return True
    if "ppi" in rel and any(x in rel for x in ["bioplex", "string"]):
        return True
    if any(x in src for x in ["bioplex", "string"]) and "gene" in rel:
        return True
    return False


def _is_drug_target_relation(relation_type: str, source: str) -> bool:
    rel = relation_type.lower()
    src = source.lower()
    keys = ["target", "drugbank", "dgidb", "chembl", "binding", "moa", "mechanism"]
    return any(k in rel for k in keys) or any(k in src for k in keys)


@dataclass
class GGMVGraph:
    """Container for graph objects used by GGMV."""

    gene_to_idx: Dict[str, int]
    idx_to_gene: List[str]
    drug_to_idx: Dict[str, int]
    idx_to_drug: List[str]
    module_to_idx: Dict[str, int]
    idx_to_module: List[str]
    module_types: List[str]

    drug_to_targets: Dict[int, List[int]]
    gene_module_matrix: sp.csr_matrix
    gene_gene_adj: sp.csr_matrix
    r_tilde: sp.csr_matrix

    module_gene_lists: List[List[int]]
    global_module_prior: np.ndarray

    module_prior_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    support_evidence_cache: Dict[Tuple[str, Tuple[int, ...]], np.ndarray] = field(default_factory=dict)

    def lookup_gene_id(self, gene_name: str) -> Optional[int]:
        key = _canon_gene(gene_name)
        if not key:
            return None
        return self.gene_to_idx.get(key)

    def lookup_drug_id(self, drug_name: str) -> Optional[int]:
        key = _canon_drug(drug_name)
        if not key:
            return None
        return self.drug_to_idx.get(key)

    def get_targets(self, drug_id: int) -> List[int]:
        return self.drug_to_targets.get(int(drug_id), [])

    @property
    def num_genes(self) -> int:
        return self.gene_module_matrix.shape[0]

    @property
    def num_modules(self) -> int:
        return self.gene_module_matrix.shape[1]

    def module_text(self, module_id: int, top_n_genes: int = 8) -> str:
        mid = int(module_id)
        if mid < 0 or mid >= len(self.idx_to_module):
            return f"module_{mid}"
        module_name = self.idx_to_module[mid]
        module_type = self.module_types[mid] if mid < len(self.module_types) else "module"
        gene_ids = self.module_gene_lists[mid] if mid < len(self.module_gene_lists) else []
        genes = [self.idx_to_gene[g] for g in gene_ids[:top_n_genes] if 0 <= g < len(self.idx_to_gene)]
        if genes:
            return f"{module_type}: {module_name}. Top genes: {', '.join(genes)}"
        return f"{module_type}: {module_name}."


class _NodeStore:
    def __init__(self) -> None:
        self.gene_to_idx: Dict[str, int] = {}
        self.idx_to_gene: List[str] = []

        self.drug_to_idx: Dict[str, int] = {}
        self.idx_to_drug: List[str] = []

        self.module_to_idx: Dict[str, int] = {}
        self.idx_to_module: List[str] = []
        self.module_types: List[str] = []

    def ensure_gene(self, gene_name: str) -> Optional[int]:
        key = _canon_gene(gene_name)
        if not key:
            return None
        idx = self.gene_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(self.idx_to_gene)
        self.gene_to_idx[key] = idx
        self.idx_to_gene.append(str(gene_name).strip())
        return idx

    def ensure_drug(self, drug_name: str) -> Optional[int]:
        key = _canon_drug(drug_name)
        if not key:
            return None
        idx = self.drug_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(self.idx_to_drug)
        self.drug_to_idx[key] = idx
        self.idx_to_drug.append(str(drug_name).strip())
        return idx

    def ensure_module(self, module_name: str, module_type: str) -> Optional[int]:
        name = str(module_name).strip()
        if not name:
            return None
        key = f"{module_type}::{name.lower()}"
        idx = self.module_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(self.idx_to_module)
        self.module_to_idx[key] = idx
        self.idx_to_module.append(name)
        self.module_types.append(module_type)
        return idx


def _extract_node_name(node: dict) -> str:
    for key in ["entity_id", "drug_name_raw", "go_name", "protein_name"]:
        val = node.get(key)
        if _is_nonempty(val):
            return str(val)
    return ""


def _load_nodes_edges(
    *,
    kg_dir: str,
    nodes_file: str = "nodes.json",
    edges_file: str = "edges.json",
    graph_file: str = "graph.json",
    strict: bool = False,
) -> Tuple[List[dict], List[dict]]:
    nodes_path = os.path.join(kg_dir, nodes_file)
    edges_path = os.path.join(kg_dir, edges_file)
    graph_path = os.path.join(kg_dir, graph_file)

    nodes: List[dict] = []
    edges: List[dict] = []

    if os.path.exists(nodes_path) and os.path.exists(edges_path):
        try:
            nodes = _read_json(nodes_path)  # type: ignore[assignment]
            edges = _read_json(edges_path)  # type: ignore[assignment]
        except Exception as exc:
            if strict:
                raise
            print(f"[GGMV][WARN] Failed to load nodes/edges JSON: {exc}. Falling back to graph.json")

    if (not nodes or not edges) and os.path.exists(graph_path):
        try:
            payload = _read_json(graph_path)
            if isinstance(payload, dict):
                nodes = payload.get("nodes", []) if not nodes else nodes
                edges = payload.get("edges", []) if not edges else edges
        except Exception as exc:
            if strict:
                raise
            print(f"[GGMV][WARN] Failed to parse graph.json: {exc}. Continue with available parts.")

    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    return nodes, edges


def _row_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    nz = row_sum > 0
    inv[nz] = 1.0 / row_sum[nz]
    return sp.diags(inv, offsets=0, format="csr").dot(adj).tocsr()


def build_r_tilde(gene_module_matrix: sp.csr_matrix, gene_gene_adj: sp.csr_matrix, alpha: float = 0.1) -> sp.csr_matrix:
    """Build R_tilde = (1-alpha)R + alpha * (P @ R)."""
    if gene_module_matrix.shape[0] != gene_gene_adj.shape[0]:
        raise ValueError(
            f"R/P shape mismatch: R has {gene_module_matrix.shape[0]} rows, P has {gene_gene_adj.shape[0]} rows"
        )
    pr = gene_gene_adj.dot(gene_module_matrix)
    return ((1.0 - alpha) * gene_module_matrix + alpha * pr).tocsr().astype(np.float32)


def load_ggmv_graph(
    *,
    kg_dir: str,
    alpha: float = 0.1,
    nodes_file: str = "nodes.json",
    edges_file: str = "edges.json",
    graph_file: str = "graph.json",
    strict: bool = False,
) -> GGMVGraph:
    """Parse KG and build sparse objects required by GGMV.

    The parser is schema-tolerant:
    - prefers explicit `nodes.json` + `edges.json`
    - falls back to `graph.json` with keys `nodes` / `edges`
    - tolerates missing pieces and keeps a functional (possibly sparse) graph object
    """
    nodes, edges = _load_nodes_edges(
        kg_dir=kg_dir,
        nodes_file=nodes_file,
        edges_file=edges_file,
        graph_file=graph_file,
        strict=strict,
    )
    print(f"[GGMV] Loaded KG nodes={len(nodes)} edges={len(edges)} from {kg_dir}")

    store = _NodeStore()
    node_index_lookup: Dict[int, Tuple[str, str]] = {}

    # Seed indices from nodes table.
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = _norm_text(node.get("node_type"))
        name = _extract_node_name(node)
        node_index = node.get("node_index")

        if node_type == "gene":
            gid = store.ensure_gene(name)
            if gid is not None and isinstance(node_index, int):
                node_index_lookup[int(node_index)] = ("gene", name)
        elif node_type == "drug":
            did = store.ensure_drug(name)
            if did is not None and isinstance(node_index, int):
                node_index_lookup[int(node_index)] = ("drug", name)
        else:
            mtype = _classify_module_type(
                node_type=node_type,
                relation_type="",
                source="",
                node_name=name,
            )
            if mtype is not None and _is_nonempty(name):
                store.ensure_module(name, mtype)
                if isinstance(node_index, int):
                    node_index_lookup[int(node_index)] = ("module", name)

    # Edge collections.
    ppi_rows: List[int] = []
    ppi_cols: List[int] = []
    ppi_vals: List[float] = []

    gm_rows: List[int] = []
    gm_cols: List[int] = []
    gm_vals: List[float] = []

    target_edges_primary: Dict[int, Set[int]] = {}
    target_edges_fallback: Dict[int, Set[int]] = {}

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        src_type = _norm_text(edge.get("src_type"))
        dst_type = _norm_text(edge.get("dst_type"))
        src_id = edge.get("src_id")
        dst_id = edge.get("dst_id")

        if not _is_nonempty(src_id) and isinstance(edge.get("src_index"), int):
            src_id = node_index_lookup.get(int(edge["src_index"]), ("", ""))[1]
        if not _is_nonempty(dst_id) and isinstance(edge.get("dst_index"), int):
            dst_id = node_index_lookup.get(int(edge["dst_index"]), ("", ""))[1]

        relation_type = _norm_text(edge.get("relation_type"))
        source = _norm_text(edge.get("source"))
        weight = _safe_float(edge.get("weight", 1.0), default=1.0)

        # gene-gene PPI adjacency
        if src_type == "gene" and dst_type == "gene" and _is_ppi_edge(relation_type, source):
            u = store.ensure_gene(str(src_id))
            v = store.ensure_gene(str(dst_id))
            if u is not None and v is not None and u != v:
                ppi_rows.extend([u, v])
                ppi_cols.extend([v, u])
                ppi_vals.extend([weight, weight])

        # drug-gene targets
        if (src_type == "drug" and dst_type == "gene") or (src_type == "gene" and dst_type == "drug"):
            if src_type == "drug":
                drug_name = str(src_id)
                gene_name = str(dst_id)
            else:
                drug_name = str(dst_id)
                gene_name = str(src_id)

            did = store.ensure_drug(drug_name)
            gid = store.ensure_gene(gene_name)
            if did is not None and gid is not None:
                bucket = target_edges_primary if _is_drug_target_relation(relation_type, source) else target_edges_fallback
                bucket.setdefault(did, set()).add(gid)

        # gene-module membership
        is_gene_module = (src_type == "gene" and dst_type != "gene") or (dst_type == "gene" and src_type != "gene")
        if is_gene_module:
            if src_type == "gene":
                gene_name = str(src_id)
                module_name = str(dst_id)
                module_node_type = dst_type
            else:
                gene_name = str(dst_id)
                module_name = str(src_id)
                module_node_type = src_type

            module_type = _classify_module_type(
                node_type=module_node_type,
                relation_type=relation_type,
                source=source,
                node_name=module_name,
            )
            if module_type is None:
                continue
            gid = store.ensure_gene(gene_name)
            mid = store.ensure_module(module_name, module_type)
            if gid is not None and mid is not None:
                gm_rows.append(gid)
                gm_cols.append(mid)
                gm_vals.append(1.0)

    # Build target dict (primary else fallback)
    drug_to_targets: Dict[int, List[int]] = {}
    for did in range(len(store.idx_to_drug)):
        targets = target_edges_primary.get(did)
        if not targets:
            targets = target_edges_fallback.get(did, set())
        drug_to_targets[did] = sorted(int(x) for x in targets)

    num_genes = len(store.idx_to_gene)
    num_modules = len(store.idx_to_module)

    if num_genes == 0:
        raise RuntimeError("[GGMV] No gene nodes parsed from KG. Cannot continue.")

    gene_module_matrix = sp.csr_matrix((gm_vals, (gm_rows, gm_cols)), shape=(num_genes, num_modules), dtype=np.float32)

    gene_gene_adj = sp.csr_matrix((ppi_vals, (ppi_rows, ppi_cols)), shape=(num_genes, num_genes), dtype=np.float32)
    gene_gene_adj = _row_normalize(gene_gene_adj)

    if num_modules > 0:
        r_tilde = build_r_tilde(gene_module_matrix=gene_module_matrix, gene_gene_adj=gene_gene_adj, alpha=alpha)
        prior_raw = np.asarray(r_tilde.sum(axis=0)).reshape(-1).astype(np.float32)
        prior_raw = np.maximum(prior_raw, 0.0)
        if float(prior_raw.sum()) <= 0.0:
            global_module_prior = np.ones((num_modules,), dtype=np.float32) / float(num_modules)
        else:
            global_module_prior = prior_raw / float(prior_raw.sum())
    else:
        r_tilde = sp.csr_matrix((num_genes, 0), dtype=np.float32)
        global_module_prior = np.zeros((0,), dtype=np.float32)

    module_gene_lists: List[List[int]] = []
    if num_modules > 0:
        gm_csc = gene_module_matrix.tocsc()
        for mid in range(num_modules):
            module_gene_lists.append(gm_csc[:, mid].indices.tolist())

    print(
        f"[GGMV] Built graph objects: genes={num_genes} drugs={len(store.idx_to_drug)} "
        f"modules={num_modules} ppi_edges={len(ppi_vals)//2} gene_module_edges={len(gm_vals)}"
    )

    return GGMVGraph(
        gene_to_idx=store.gene_to_idx,
        idx_to_gene=store.idx_to_gene,
        drug_to_idx=store.drug_to_idx,
        idx_to_drug=store.idx_to_drug,
        module_to_idx=store.module_to_idx,
        idx_to_module=store.idx_to_module,
        module_types=store.module_types,
        drug_to_targets=drug_to_targets,
        gene_module_matrix=gene_module_matrix,
        gene_gene_adj=gene_gene_adj,
        r_tilde=r_tilde,
        module_gene_lists=module_gene_lists,
        global_module_prior=global_module_prior,
    )
