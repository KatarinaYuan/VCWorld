#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph-only baseline model components (module proposer + shallow gene scorer)."""

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
        """
        Args:
            module_features: [num_modules, feature_dim]
        Returns:
            module_logits: [num_modules]
        """
        if module_features.ndim != 2:
            raise ValueError(f"module_features must be 2D, got shape={tuple(module_features.shape)}")
        return self.scorer(module_features).squeeze(-1)


class GraphOnlyGeneScorer(nn.Module):
    """Shallow scorer from graph-derived gene features to DEG logit."""

    def __init__(
        self,
        *,
        feature_dim: int,
        arch: str = "linear",
        hidden_dim: int = 32,
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
            raise ValueError(f"Unsupported gene scorer arch: {arch}")

    def forward(self, gene_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_features: [num_genes, feature_dim]
        Returns:
            gene_logits: [num_genes]
        """
        if gene_features.ndim != 2:
            raise ValueError(f"gene_features must be 2D, got shape={tuple(gene_features.shape)}")
        return self.scorer(gene_features).squeeze(-1)


class GraphOnlyPerturbationPredictor(nn.Module):
    """Wrapper that infers q_p from support and scores held-out genes."""

    def __init__(self, proposer: GraphOnlyModuleProposer, gene_scorer: GraphOnlyGeneScorer) -> None:
        super().__init__()
        self.proposer = proposer
        self.gene_scorer = gene_scorer

    def infer_module_posterior(self, module_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        module_logits = self.proposer(module_features)
        q = torch.softmax(module_logits, dim=-1)
        return {"module_logits": module_logits, "q": q}

    def score_genes(self, gene_features: torch.Tensor) -> torch.Tensor:
        return self.gene_scorer(gene_features)

    def forward(self, module_features: torch.Tensor, gene_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        mod = self.infer_module_posterior(module_features)
        gene_logits = self.score_genes(gene_features)
        return {
            "module_logits": mod["module_logits"],
            "q": mod["q"],
            "gene_logits": gene_logits,
            "gene_probs": torch.sigmoid(gene_logits),
        }

