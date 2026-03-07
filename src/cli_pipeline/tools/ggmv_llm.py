#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-based mechanism scoring utilities for GGMV (support-aware, no finetuning)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from ggmv_graph import GGMVGraph


@dataclass
class LLMMechanismScorer:
    """Encode query/support context + module text and score via cosine similarity.

    This is a lightweight, no-finetuning scorer:
    - query/support representation: hidden states from current prompt ids
    - module representation: hidden states from module text
    """

    model: object
    tokenizer: object
    device: torch.device
    max_tokens: int = 1024
    module_top_genes: int = 8
    module_embed_cache: Dict[int, np.ndarray] = field(default_factory=dict)

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
        vec = h.mean(dim=0).float().detach().cpu().numpy().astype(np.float32)
        return vec

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

    def score_candidate_modules(self, *, prompt_ids: List[int], candidate_module_ids: List[int], graph: GGMVGraph) -> np.ndarray:
        if not candidate_module_ids:
            return np.zeros((0,), dtype=np.float32)

        query_vec = self._encode_token_ids(prompt_ids)
        scores: List[float] = []

        for mid in candidate_module_ids:
            if mid not in self.module_embed_cache:
                module_text = graph.module_text(mid, top_n_genes=self.module_top_genes)
                self.module_embed_cache[mid] = self._encode_text(module_text)
            mvec = self.module_embed_cache[mid]
            scores.append(float(self._cosine(query_vec, mvec)))

        return np.asarray(scores, dtype=np.float32)

    def score_sufficiency(self, *, prompt_ids: List[int]) -> float:
        """Optional LLM-based sufficiency judge (v1 lightweight stub).

        Current heuristic keeps a bounded scalar from prompt embedding norm.
        """
        vec = self._encode_token_ids(prompt_ids)
        if vec.size == 0:
            return 0.0
        norm = float(np.linalg.norm(vec.astype(np.float64)))
        return float(np.tanh(norm / 100.0))
