#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph + LLM-hidden hybrid model components."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class GraphOnlyModuleProposer(nn.Module):
    """Shared scorer over module-level graph features."""

    def __init__(
        self,
        *,
        feature_dim: int,
        arch: str = "linear",
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        arch_norm = arch.strip().lower()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got {feature_dim}")
        if arch_norm == "linear":
            self.scorer = nn.Linear(feature_dim, 1)
        elif arch_norm == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported proposer arch: {arch}")

    def forward(self, module_features: torch.Tensor) -> torch.Tensor:
        return self.scorer(module_features).squeeze(-1)


class GraphLLMGeneScorer(nn.Module):
    """Shallow scorer from graph feature + projected LLM hidden."""

    def __init__(
        self,
        *,
        graph_feature_dim: int,
        hidden_input_dim: int,
        hidden_proj_dim: int = 64,
        arch: str = "linear",
        scorer_hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if graph_feature_dim <= 0 or hidden_input_dim <= 0:
            raise ValueError("graph_feature_dim and hidden_input_dim must be > 0")
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_input_dim, hidden_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        in_dim = int(graph_feature_dim + hidden_proj_dim)
        arch_norm = arch.strip().lower()
        if arch_norm == "linear":
            self.scorer = nn.Linear(in_dim, 1)
        elif arch_norm == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(in_dim, scorer_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(scorer_hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported gene scorer arch: {arch}")

    def forward(self, graph_features: torch.Tensor, hidden_features: torch.Tensor) -> torch.Tensor:
        if graph_features.ndim != 2 or hidden_features.ndim != 2:
            raise ValueError("graph_features and hidden_features must be 2D")
        if graph_features.shape[0] != hidden_features.shape[0]:
            raise ValueError("Batch size mismatch between graph and hidden features")
        h_proj = self.hidden_proj(hidden_features)
        fused = torch.cat([graph_features, h_proj], dim=-1)
        return self.scorer(fused).squeeze(-1)


class GraphLLMHybridPredictor(nn.Module):
    """Infer q_p from graph support and score genes with graph+LLM hidden."""

    def __init__(self, proposer: GraphOnlyModuleProposer, gene_scorer: GraphLLMGeneScorer) -> None:
        super().__init__()
        self.proposer = proposer
        self.gene_scorer = gene_scorer

    def infer_module_posterior(self, module_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        module_logits = self.proposer(module_features)
        q = torch.softmax(module_logits, dim=-1)
        return {"module_logits": module_logits, "q": q}

    def score_genes(self, graph_features: torch.Tensor, hidden_features: torch.Tensor) -> torch.Tensor:
        return self.gene_scorer(graph_features, hidden_features)

