#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small trainable heads for MSLD (support-supervised latent mechanism distillation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cross entropy between soft target distribution and logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    target = torch.clamp(target_probs, min=0.0)
    denom = target.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target / denom
    return -(target * log_probs).sum(dim=-1).mean()


def distribution_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def compute_sufficiency_score(
    mechanism_probs: torch.Tensor,
    margin: torch.Tensor,
    eta1: float,
    eta2: float,
    eta3: float,
) -> torch.Tensor:
    max_z = mechanism_probs.max(dim=-1).values
    ent = distribution_entropy(mechanism_probs)
    return eta1 * max_z - eta2 * ent + eta3 * margin.abs()


@dataclass
class MSLDHeadConfig:
    hidden_dim: int
    num_modules: int
    module_emb_dim: int = 128
    label_hidden_dim: int = 256
    dropout: float = 0.1


class MSLDHeads(nn.Module):
    """Small parameter set to adapt on support and infer on query-only inputs."""

    def __init__(self, cfg: MSLDHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proposer_head = nn.Linear(cfg.hidden_dim, cfg.num_modules)
        self.mechanism_head = nn.Linear(cfg.hidden_dim, cfg.num_modules)
        self.module_embedding = nn.Parameter(torch.randn(cfg.num_modules, cfg.module_emb_dim) * 0.02)
        self.label_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.module_emb_dim + 1, cfg.label_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.label_hidden_dim, 1),
        )

    def propose_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.proposer_head(h)

    def mechanism_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.mechanism_head(h)

    def predict_margin(
        self,
        h: torch.Tensor,
        mechanism_logits: torch.Tensor,
        base_margin: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_hat = F.softmax(mechanism_logits, dim=-1)
        m_emb = z_hat @ self.module_embedding
        feat = torch.cat([h, m_emb, base_margin.unsqueeze(-1)], dim=-1)
        margin = self.label_head(feat).squeeze(-1)
        return {"margin": margin, "z_hat": z_hat, "m_emb": m_emb}
