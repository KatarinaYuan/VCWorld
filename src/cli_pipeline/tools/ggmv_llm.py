#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-based mechanism scoring utilities for GGMV (support-aware, no finetuning)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ggmv_graph import GGMVGraph


def _target_jaccard(graph: GGMVGraph, did_a: int, did_b: int) -> float:
    if did_a < 0 or did_b < 0:
        return 0.0
    a = set(graph.get_targets(did_a))
    b = set(graph.get_targets(did_b))
    if not a and not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _gene_name(graph: GGMVGraph, gid: int) -> str:
    if gid < 0 or gid >= len(graph.idx_to_gene):
        return f"gene_{gid}"
    return graph.idx_to_gene[gid]


def _drug_name(graph: GGMVGraph, did: int) -> str:
    if did < 0 or did >= len(graph.idx_to_drug):
        return f"drug_{did}"
    return graph.idx_to_drug[did]


def build_query_support_summary(
    *,
    query: dict,
    support_set: List[dict],
    graph: GGMVGraph,
    max_support: int = 8,
    max_genes_per_support: int = 6,
) -> str:
    """Build compact structured summary text for (query, support) pair."""
    q_drug_id = int(query.get("drug_id", -1))
    q_gene_id = int(query.get("gene_id", -1))
    q_drug_name = str(query.get("drug_name") or _drug_name(graph, q_drug_id))
    q_gene_name = str(query.get("gene_name") or _gene_name(graph, q_gene_id))

    q_targets = graph.get_targets(q_drug_id)
    q_target_names = [_gene_name(graph, g) for g in q_targets[:8]]

    lines = [
        "[GGMV Query-Support Summary]",
        f"Query drug: {q_drug_name}",
        f"Query gene: {q_gene_name}",
        f"Query targets({len(q_targets)}): {', '.join(q_target_names) if q_target_names else 'NA'}",
        f"Supports: {len(support_set)} perturbations",
    ]

    for idx, support in enumerate(support_set[: max(1, max_support)], 1):
        s_drug_id = int(support.get("drug_id", -1))
        s_drug_name = str(support.get("perturbation_id") or _drug_name(graph, s_drug_id))
        same_cell = int(support.get("cell_id", -2) == query.get("cell_id", -1))
        jacc = _target_jaccard(graph, q_drug_id, s_drug_id)

        gids = list(support.get("labeled_gene_ids", []))
        ys = list(support.get("labels", []))
        n = min(len(gids), len(ys))
        pos = [_gene_name(graph, int(gids[i])) for i in range(n) if int(ys[i]) == 1][:max_genes_per_support]
        neg = [_gene_name(graph, int(gids[i])) for i in range(n) if int(ys[i]) == 0][:max_genes_per_support]

        lines.append(
            f"Support#{idx}: drug={s_drug_name}; same_cell={same_cell}; target_jaccard={jacc:.3f}; "
            f"pos_genes={', '.join(pos) if pos else 'NA'}; neg_genes={', '.join(neg) if neg else 'NA'}"
        )

    return "\n".join(lines)


@dataclass
class LLMMechanismScorer:
    """Support-aware mechanism scorer from hidden states.

    Phase 2 design:
    - One query-support context encoding per episode.
    - Candidate modules scored against that context vector.
    - Optional prompt-tail concat for richer context when needed.
    """

    model: object
    tokenizer: object
    device: torch.device
    max_tokens: int = 1024
    module_top_genes: int = 8

    use_prompt_context: bool = False
    prompt_context_tokens: int = 256
    max_support_in_summary: int = 8
    max_genes_per_support: int = 6

    overlap_bonus_weight: float = 0.2

    module_embed_cache: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    context_embed_cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def _encode_token_ids(self, token_ids: List[int]) -> np.ndarray:
        if not token_ids:
            return np.zeros((1,), dtype=np.float32)

        if self.max_tokens > 0 and len(token_ids) > self.max_tokens:
            token_ids = token_ids[-self.max_tokens :]

        input_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(input_t)

        with torch.no_grad():
            out = self.model(
                input_ids=input_t,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True,
            )

        if out.hidden_states is None or len(out.hidden_states) == 0:
            return np.zeros((1,), dtype=np.float32)

        h = out.hidden_states[-1][0]  # [seq, hidden]
        return h.mean(dim=0).float().detach().cpu().numpy().astype(np.float32)

    def _encode_text(self, text: str) -> np.ndarray:
        tok = self.tokenizer(text, add_special_tokens=False).input_ids
        return self._encode_token_ids(tok)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        num = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        den = float(np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(b.astype(np.float64)))
        if den <= eps:
            return 0.0
        return num / den

    def _module_text_for_query(self, *, module_id: int, query: dict, graph: GGMVGraph) -> str:
        base = graph.module_text(module_id, top_n_genes=self.module_top_genes)
        q_drug = int(query.get("drug_id", -1))
        q_targets = graph.get_targets(q_drug)
        m_genes = set(graph.module_gene_lists[module_id]) if module_id < len(graph.module_gene_lists) else set()
        overlap = [g for g in q_targets if g in m_genes]
        overlap_names = [_gene_name(graph, g) for g in overlap[:6]]
        return (
            f"{base} "
            f"Query-target-overlap: {len(overlap)}/{len(q_targets)}. "
            f"Overlap genes: {', '.join(overlap_names) if overlap_names else 'NA'}."
        )

    def encode_query_support_context(
        self,
        *,
        query: dict,
        support_set: List[dict],
        graph: GGMVGraph,
        prompt_ids: Optional[List[int]] = None,
        summary_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """Encode one support-aware context vector per episode."""
        if summary_text is None:
            summary_text = build_query_support_summary(
                query=query,
                support_set=support_set,
                graph=graph,
                max_support=self.max_support_in_summary,
                max_genes_per_support=self.max_genes_per_support,
            )

        summary_ids = self.tokenizer(summary_text, add_special_tokens=False).input_ids
        if self.use_prompt_context and prompt_ids is not None and len(prompt_ids) > 0:
            prompt_tail = prompt_ids[-self.prompt_context_tokens :] if self.prompt_context_tokens > 0 else list(prompt_ids)
            token_ids = prompt_tail + summary_ids
        else:
            token_ids = summary_ids

        key_payload = ",".join(str(x) for x in token_ids[:4096])
        key = hashlib.md5(key_payload.encode("utf-8")).hexdigest()
        cached = self.context_embed_cache.get(key)
        if cached is not None:
            return cached, summary_text

        vec = self._encode_token_ids(token_ids)
        self.context_embed_cache[key] = vec
        return vec, summary_text

    def score_candidate_modules_from_context(
        self,
        *,
        context_vec: np.ndarray,
        query: dict,
        candidate_module_ids: List[int],
        graph: GGMVGraph,
    ) -> np.ndarray:
        if not candidate_module_ids:
            return np.zeros((0,), dtype=np.float32)

        q_drug = int(query.get("drug_id", -1))
        q_targets = graph.get_targets(q_drug)
        q_target_set = set(q_targets)

        scores: List[float] = []
        for mid in candidate_module_ids:
            cache_key = (int(mid), int(q_drug))
            module_vec = self.module_embed_cache.get(cache_key)
            if module_vec is None:
                m_text = self._module_text_for_query(module_id=int(mid), query=query, graph=graph)
                module_vec = self._encode_text(m_text)
                self.module_embed_cache[cache_key] = module_vec

            score = float(self._cosine(context_vec, module_vec))

            # Lightweight overlap bonus to stabilize query-specific signal.
            module_genes = set(graph.module_gene_lists[mid]) if mid < len(graph.module_gene_lists) else set()
            if q_target_set:
                overlap = len(q_target_set & module_genes) / float(max(1, len(q_target_set)))
                score += self.overlap_bonus_weight * float(overlap)
            scores.append(score)

        return np.asarray(scores, dtype=np.float32)

    def score_sufficiency(self, *, context_vec: np.ndarray, support_set: List[dict]) -> float:
        """Optional LLM-based sufficiency score in [0, 1]."""
        if context_vec.size == 0:
            return 0.0
        emb_strength = float(np.tanh(np.linalg.norm(context_vec.astype(np.float64)) / 120.0))
        support_count = min(1.0, float(len(support_set)) / float(max(1, self.max_support_in_summary)))
        if not support_set:
            label_density = 0.0
        else:
            n_labels = [len(s.get("labels", [])) for s in support_set]
            label_density = min(1.0, float(np.mean(n_labels)) / 32.0)
        return float(np.clip(0.5 * emb_strength + 0.3 * support_count + 0.2 * label_density, 0.0, 1.0))
