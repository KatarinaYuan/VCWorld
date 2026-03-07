#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph parser and support-time mechanism teacher for MSLD.

MSLD v1 constraints:
- module vocabulary uses ONLY reactome_term and go_term nodes
- graph verification is support-time only
- test-time inference must not call any support-conditioned graph logic
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
        return json.loads(f.read())


def _extract_node_name(node: dict) -> str:
    for key in ("entity_id", "go_name", "drug_name_raw", "protein_name"):
        val = node.get(key)
        if _is_nonempty(val):
            return str(val).strip()
    return ""


def _is_ppi_edge(relation_type: str, source: str) -> bool:
    rel = relation_type.lower()
    src = source.lower()
    if "bioplex" in rel or "string" in rel:
        return True
    if "ppi" in rel:
        return True
    if "bioplex" in src or "string" in src:
        return True
    return False


def _is_drug_target_relation(relation_type: str, source: str) -> bool:
    rel = relation_type.lower()
    src = source.lower()
    keys = ("target", "drugbank", "dgidb", "chembl", "binding", "moa", "mechanism")
    return any(k in rel for k in keys) or any(k in src for k in keys)


def _row_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    nz = row_sum > 0
    inv[nz] = 1.0 / row_sum[nz]
    return sp.diags(inv, offsets=0, format="csr").dot(adj).tocsr()


def _build_r_tilde(r: sp.csr_matrix, p: sp.csr_matrix, alpha: float) -> sp.csr_matrix:
    if r.shape[0] != p.shape[0]:
        raise ValueError(f"R/P shape mismatch: R rows={r.shape[0]} P rows={p.shape[0]}")
    return ((1.0 - alpha) * r + alpha * (p.dot(r))).tocsr().astype(np.float32)


def sparsemax(logits: np.ndarray) -> np.ndarray:
    """Sparsemax for 1D logits."""
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if z.size == 0:
        return np.zeros((0,), dtype=np.float32)
    z_sorted = np.sort(z)[::-1]
    z_cumsum = np.cumsum(z_sorted)
    ks = np.arange(1, z.size + 1, dtype=np.float64)
    support = 1.0 + ks * z_sorted > z_cumsum
    if not np.any(support):
        return np.ones_like(z, dtype=np.float32) / float(z.size)
    k = int(np.max(np.where(support)[0]) + 1)
    tau = (z_cumsum[k - 1] - 1.0) / float(k)
    p = np.maximum(z - tau, 0.0)
    den = float(p.sum())
    if den <= 0.0:
        return np.ones_like(z, dtype=np.float32) / float(z.size)
    return (p / den).astype(np.float32)


@dataclass
class MSLDGraph:
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

    @property
    def num_genes(self) -> int:
        return self.gene_module_matrix.shape[0]

    @property
    def num_modules(self) -> int:
        return self.gene_module_matrix.shape[1]

    def get_targets(self, drug_id: int) -> List[int]:
        return self.drug_to_targets.get(int(drug_id), [])


def _select_top_modules(
    r: sp.csr_matrix,
    idx_to_module: List[str],
    module_types: List[str],
    max_modules: int,
) -> Tuple[sp.csr_matrix, List[str], List[str]]:
    if max_modules <= 0 or r.shape[1] <= max_modules:
        return r, idx_to_module, module_types
    col_nnz = np.asarray((r > 0).sum(axis=0)).reshape(-1)
    keep = np.argpartition(col_nnz, -max_modules)[-max_modules:]
    keep = keep[np.argsort(col_nnz[keep])[::-1]]
    keep = keep.astype(np.int64)
    r2 = r[:, keep].tocsr()
    names = [idx_to_module[int(i)] for i in keep.tolist()]
    types = [module_types[int(i)] for i in keep.tolist()]
    return r2, names, types


def load_msld_graph(
    *,
    kg_dir: str,
    alpha: float = 0.1,
    max_modules: int = 2048,
    nodes_file: str = "nodes.json",
    edges_file: str = "edges.json",
    graph_file: str = "graph.json",
) -> MSLDGraph:
    """Load KG and build graph objects for MSLD.

    Module nodes are STRICTLY limited to:
    - reactome_term
    - go_term
    """
    nodes_path = os.path.join(kg_dir, nodes_file)
    edges_path = os.path.join(kg_dir, edges_file)
    graph_path = os.path.join(kg_dir, graph_file)

    nodes: List[dict] = []
    edges: List[dict] = []
    if os.path.exists(nodes_path) and os.path.exists(edges_path):
        nodes = _read_json(nodes_path)  # type: ignore[assignment]
        edges = _read_json(edges_path)  # type: ignore[assignment]
    elif os.path.exists(graph_path):
        payload = _read_json(graph_path)
        if isinstance(payload, dict):
            nodes = payload.get("nodes", [])
            edges = payload.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise RuntimeError("Invalid KG format: nodes/edges list not found")

    gene_to_idx: Dict[str, int] = {}
    idx_to_gene: List[str] = []
    drug_to_idx: Dict[str, int] = {}
    idx_to_drug: List[str] = []
    module_to_idx: Dict[str, int] = {}
    idx_to_module: List[str] = []
    module_types: List[str] = []
    node_index_lookup: Dict[int, Tuple[str, str]] = {}

    def ensure_gene(name: str) -> Optional[int]:
        key = _canon_gene(name)
        if not key:
            return None
        idx = gene_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(idx_to_gene)
        gene_to_idx[key] = idx
        idx_to_gene.append(str(name).strip())
        return idx

    def ensure_drug(name: str) -> Optional[int]:
        key = _canon_drug(name)
        if not key:
            return None
        idx = drug_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(idx_to_drug)
        drug_to_idx[key] = idx
        idx_to_drug.append(str(name).strip())
        return idx

    def ensure_module(name: str, module_type: str) -> Optional[int]:
        n = str(name).strip()
        if not n:
            return None
        key = f"{module_type}::{n.lower()}"
        idx = module_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(idx_to_module)
        module_to_idx[key] = idx
        idx_to_module.append(n)
        module_types.append(module_type)
        return idx

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = _norm_text(node.get("node_type"))
        name = _extract_node_name(node)
        node_index = node.get("node_index")
        if node_type == "gene":
            gid = ensure_gene(name)
            if gid is not None and isinstance(node_index, int):
                node_index_lookup[int(node_index)] = ("gene", name)
        elif node_type == "drug":
            did = ensure_drug(name)
            if did is not None and isinstance(node_index, int):
                node_index_lookup[int(node_index)] = ("drug", name)
        elif node_type in {"reactome_term", "go_term"}:
            mtype = "reactome" if node_type == "reactome_term" else "go_term"
            mid = ensure_module(name, mtype)
            if mid is not None and isinstance(node_index, int):
                node_index_lookup[int(node_index)] = ("module", name)

    gm_rows: List[int] = []
    gm_cols: List[int] = []
    gm_vals: List[float] = []
    ppi_rows: List[int] = []
    ppi_cols: List[int] = []
    ppi_vals: List[float] = []

    direct_targets: Dict[int, Set[int]] = {}
    fallback_targets: Dict[int, Set[int]] = {}
    role_to_drugs: Dict[str, Set[int]] = {}
    role_to_genes: Dict[str, Set[int]] = {}

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
        if not _is_nonempty(src_id) or not _is_nonempty(dst_id):
            continue

        relation_type = _norm_text(edge.get("relation_type"))
        source = _norm_text(edge.get("source"))
        weight = _safe_float(edge.get("weight", 1.0), default=1.0)

        # gene-gene PPI
        if src_type == "gene" and dst_type == "gene" and _is_ppi_edge(relation_type, source):
            u = ensure_gene(str(src_id))
            v = ensure_gene(str(dst_id))
            if u is not None and v is not None and u != v:
                ppi_rows.extend([u, v])
                ppi_cols.extend([v, u])
                ppi_vals.extend([weight, weight])

        # direct drug-gene target links
        if (src_type == "drug" and dst_type == "gene") or (src_type == "gene" and dst_type == "drug"):
            if src_type == "drug":
                did = ensure_drug(str(src_id))
                gid = ensure_gene(str(dst_id))
            else:
                did = ensure_drug(str(dst_id))
                gid = ensure_gene(str(src_id))
            if did is not None and gid is not None:
                bucket = direct_targets if _is_drug_target_relation(relation_type, source) else fallback_targets
                bucket.setdefault(did, set()).add(gid)

        # drug-target_role / target_role-gene bridging
        if (src_type == "drug" and dst_type == "target_role") or (src_type == "target_role" and dst_type == "drug"):
            if src_type == "drug":
                did = ensure_drug(str(src_id))
                rid = str(dst_id).strip()
            else:
                did = ensure_drug(str(dst_id))
                rid = str(src_id).strip()
            if did is not None and rid:
                role_to_drugs.setdefault(rid, set()).add(did)

        if (src_type == "target_role" and dst_type == "gene") or (src_type == "gene" and dst_type == "target_role"):
            if src_type == "target_role":
                rid = str(src_id).strip()
                gid = ensure_gene(str(dst_id))
            else:
                rid = str(dst_id).strip()
                gid = ensure_gene(str(src_id))
            if gid is not None and rid:
                role_to_genes.setdefault(rid, set()).add(gid)

        # gene-module membership ONLY for reactome_term/go_term
        left_is_gene = src_type == "gene" and dst_type in {"reactome_term", "go_term"}
        right_is_gene = dst_type == "gene" and src_type in {"reactome_term", "go_term"}
        if left_is_gene or right_is_gene:
            if left_is_gene:
                gid = ensure_gene(str(src_id))
                module_name = str(dst_id)
                mtype = "reactome" if dst_type == "reactome_term" else "go_term"
            else:
                gid = ensure_gene(str(dst_id))
                module_name = str(src_id)
                mtype = "reactome" if src_type == "reactome_term" else "go_term"
            mid = ensure_module(module_name, mtype)
            if gid is not None and mid is not None:
                gm_rows.append(gid)
                gm_cols.append(mid)
                gm_vals.append(1.0)

    # merge role bridges into target map
    for role_id, dids in role_to_drugs.items():
        gids = role_to_genes.get(role_id, set())
        if not gids:
            continue
        for did in dids:
            fallback_targets.setdefault(did, set()).update(gids)

    num_genes = len(idx_to_gene)
    if num_genes == 0:
        raise RuntimeError("No genes parsed from KG")

    num_modules_raw = len(idx_to_module)
    r_raw = sp.csr_matrix((gm_vals, (gm_rows, gm_cols)), shape=(num_genes, num_modules_raw), dtype=np.float32)
    r, idx_to_module, module_types = _select_top_modules(r_raw, idx_to_module, module_types, max_modules=max_modules)

    # remap module_to_idx after pruning
    module_to_idx = {}
    for i, (mname, mtype) in enumerate(zip(idx_to_module, module_types)):
        module_to_idx[f"{mtype}::{mname.lower()}"] = int(i)

    p = sp.csr_matrix((ppi_vals, (ppi_rows, ppi_cols)), shape=(num_genes, num_genes), dtype=np.float32)
    p = _row_normalize(p)
    r_tilde = _build_r_tilde(r=r, p=p, alpha=alpha) if r.shape[1] > 0 else sp.csr_matrix((num_genes, 0), dtype=np.float32)

    # targets: prefer direct target edges, fallback to bridged/weak links
    drug_to_targets: Dict[int, List[int]] = {}
    for did in range(len(idx_to_drug)):
        targets = direct_targets.get(did)
        if not targets:
            targets = fallback_targets.get(did, set())
        drug_to_targets[did] = sorted(int(g) for g in targets)

    if r_tilde.shape[1] > 0:
        prior_raw = np.asarray(r_tilde.sum(axis=0)).reshape(-1).astype(np.float32)
        prior_raw = np.maximum(prior_raw, 0.0)
        if float(prior_raw.sum()) <= 0.0:
            global_prior = np.ones((r_tilde.shape[1],), dtype=np.float32) / float(r_tilde.shape[1])
        else:
            global_prior = prior_raw / float(prior_raw.sum())
    else:
        global_prior = np.zeros((0,), dtype=np.float32)

    module_gene_lists: List[List[int]] = []
    if r.shape[1] > 0:
        r_csc = r.tocsc()
        for mid in range(r.shape[1]):
            module_gene_lists.append(r_csc[:, mid].indices.tolist())

    print(
        f"[MSLD] Graph loaded: genes={num_genes} drugs={len(idx_to_drug)} modules={r.shape[1]} "
        f"ppi_edges={len(ppi_vals)//2} gene_module_edges={len(gm_vals)}"
    )

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


def compute_drug_prior(graph: MSLDGraph, drug_id: int) -> np.ndarray:
    if graph.num_modules == 0:
        return np.zeros((0,), dtype=np.float32)
    did = int(drug_id)
    targets = graph.get_targets(did)
    if not targets:
        return graph.global_module_prior.copy()
    prior_raw = np.asarray(graph.r_tilde[targets].sum(axis=0)).reshape(-1).astype(np.float32)
    prior_raw = np.maximum(prior_raw, 0.0)
    if float(prior_raw.sum()) <= 0:
        return graph.global_module_prior.copy()
    return prior_raw / float(prior_raw.sum())


def compute_support_evidence(
    graph: MSLDGraph,
    gene_ids: Sequence[int],
    labels: Sequence[int],
) -> np.ndarray:
    """Compute perturbation-level support evidence over full module vocabulary."""
    if graph.num_modules == 0:
        return np.zeros((0,), dtype=np.float32)
    n = min(len(gene_ids), len(labels))
    if n <= 0:
        return np.zeros((graph.num_modules,), dtype=np.float32)
    g = np.asarray(gene_ids[:n], dtype=np.int64)
    y = np.asarray(labels[:n], dtype=np.float32)
    valid = (g >= 0) & (g < graph.num_genes)
    if not np.any(valid):
        return np.zeros((graph.num_modules,), dtype=np.float32)
    g = g[valid]
    y = y[valid]
    signs = 2.0 * y - 1.0
    mat = graph.r_tilde[g].toarray().astype(np.float32)
    return (signs[:, None] * mat).mean(axis=0).astype(np.float32)


def verify_mechanism_posterior(
    *,
    proposer_logits: np.ndarray,
    graph_prior: np.ndarray,
    support_evidence: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int = 256,
    eps: float = 1e-8,
) -> np.ndarray:
    """Build graph-verified perturbation mechanism posterior over global vocabulary."""
    u = np.asarray(proposer_logits, dtype=np.float32).reshape(-1)
    pi = np.asarray(graph_prior, dtype=np.float32).reshape(-1)
    ev = np.asarray(support_evidence, dtype=np.float32).reshape(-1)
    if not (u.size == pi.size == ev.size):
        raise ValueError(f"Shape mismatch: proposer={u.size} prior={pi.size} evidence={ev.size}")
    m = u.size
    if m == 0:
        return np.zeros((0,), dtype=np.float32)

    pi = np.maximum(pi, 0.0)
    if float(pi.sum()) <= 0:
        pi = np.ones_like(pi, dtype=np.float32) / float(m)
    else:
        pi = pi / float(pi.sum())

    if top_k > 0 and top_k < m:
        part = np.argpartition(pi, -top_k)[-top_k:]
        sel = part[np.argsort(pi[part])[::-1]]
        logits_sub = alpha * u[sel] + beta * np.log(pi[sel] + eps) + gamma * ev[sel]
        z_sub = sparsemax(logits_sub)
        z = np.zeros((m,), dtype=np.float32)
        z[sel] = z_sub
        den = float(z.sum())
        if den > 0:
            z /= den
        return z.astype(np.float32)

    logits = alpha * u + beta * np.log(pi + eps) + gamma * ev
    return sparsemax(logits)
