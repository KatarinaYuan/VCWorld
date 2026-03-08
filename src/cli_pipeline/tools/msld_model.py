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
    refine_hidden_dim: int = 64
    type_emb_dim: int = 16
    max_cell_types: int = 256
    max_drug_types: int = 8192


class _ScalarFeatureBranch(nn.Module):
    """Encode one scalar source feature per module with a dedicated branch."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1))


class PerturbationAwareRefineMLP(nn.Module):
    """Refiner with per-source branches + perturbation-type embeddings."""

    def __init__(self, hidden_dim: int, type_emb_dim: int, max_cell_types: int, max_drug_types: int) -> None:
        super().__init__()
        self.max_cell_types = int(max(1, max_cell_types))
        self.max_drug_types = int(max(1, max_drug_types))
        self.cell_embed = nn.Embedding(self.max_cell_types, type_emb_dim)
        self.drug_embed = nn.Embedding(self.max_drug_types, type_emb_dim)
        self.type_proj = nn.Sequential(
            nn.Linear(type_emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Separate feature branches by source.
        self.u_branch = _ScalarFeatureBranch(hidden_dim)
        self.pi_branch = _ScalarFeatureBranch(hidden_dim)
        self.ev_branch = _ScalarFeatureBranch(hidden_dim)
        self.cons_branch = _ScalarFeatureBranch(hidden_dim)
        self.z_branch = _ScalarFeatureBranch(hidden_dim)

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _index_cell(self, cell_id: int, device: torch.device) -> torch.Tensor:
        idx = int(cell_id) % self.max_cell_types
        return torch.tensor([idx], dtype=torch.long, device=device)

    def _index_drug(self, drug_id: int, device: torch.device) -> torch.Tensor:
        idx = int(drug_id) % self.max_drug_types
        return torch.tensor([idx], dtype=torch.long, device=device)

    def forward(
        self,
        *,
        u: torch.Tensor,
        log_pi: torch.Tensor,
        evidence: torch.Tensor,
        consistency: torch.Tensor,
        z_prev: torch.Tensor,
        cell_id: int,
        drug_id: int,
    ) -> torch.Tensor:
        """
        Args are module vectors with shape [M]. Return per-module correction delta [M].
        """
        if not (u.ndim == log_pi.ndim == evidence.ndim == consistency.ndim == z_prev.ndim == 1):
            raise ValueError("RefineMLP inputs must be 1D vectors over modules")
        if not (u.numel() == log_pi.numel() == evidence.numel() == consistency.numel() == z_prev.numel()):
            raise ValueError("RefineMLP input size mismatch")
        m = int(u.numel())
        if m == 0:
            return u.new_zeros((0,))

        cell_idx = self._index_cell(cell_id, u.device)
        drug_idx = self._index_drug(drug_id, u.device)
        type_h = self.type_proj(torch.cat([self.cell_embed(cell_idx), self.drug_embed(drug_idx)], dim=-1))
        type_h = type_h.expand(m, -1)

        # Source-wise branches ensure heterogenous features are treated separately.
        h = (
            self.u_branch(u)
            + self.pi_branch(log_pi)
            + self.ev_branch(evidence)
            + self.cons_branch(consistency)
            + self.z_branch(z_prev)
            + type_h
        )
        return self.out(h).squeeze(-1)


class MSLDHeads(nn.Module):
    """Small parameter set to adapt on support and infer on query-only inputs.

    Design:
    - proposer_head: perturbation-level module proposal (from pooled support hidden states)
    - mechanism_head: example-level distilled mechanism predictor (used at test-time query-only)
    - module_consistency_logits: example-level verifier over all modules
    - label_head: predicts yes/no margin from hidden + mechanism embedding + confidence features
    """

    def __init__(self, cfg: MSLDHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proposer_head = nn.Linear(cfg.hidden_dim, cfg.num_modules)
        self.mechanism_head = nn.Linear(cfg.hidden_dim, cfg.num_modules)
        self.module_embedding = nn.Parameter(torch.randn(cfg.num_modules, cfg.module_emb_dim) * 0.02)
        self.verifier_proj = nn.Linear(cfg.hidden_dim, cfg.module_emb_dim, bias=False)
        self.verifier_bias = nn.Parameter(torch.zeros(cfg.num_modules))
        self.refiner = PerturbationAwareRefineMLP(
            hidden_dim=cfg.refine_hidden_dim,
            type_emb_dim=cfg.type_emb_dim,
            max_cell_types=cfg.max_cell_types,
            max_drug_types=cfg.max_drug_types,
        )
        self.label_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.module_emb_dim + 2, cfg.label_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.label_hidden_dim, 1),
        )

    def propose_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.proposer_head(h)

    def mechanism_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.mechanism_head(h)

    def module_consistency_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Return module-conditioned verifier logits c_{jtm} for each example h_{jt}."""
        h_proj = self.verifier_proj(h)  # [B, D]
        return h_proj @ self.module_embedding.t() + self.verifier_bias.unsqueeze(0)  # [B, M]

    def refine_posterior(
        self,
        *,
        proposer_logits: torch.Tensor,
        graph_prior: torch.Tensor,
        support_evidence: torch.Tensor,
        support_consistency: torch.Tensor,
        cell_id: int,
        drug_id: int,
        alpha: float,
        beta: float,
        gamma: float,
        rho: float,
        rounds: int,
        damping: float,
        lambda_delta: float,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """
        Iterative perturbation-level posterior refinement.
        """
        u = proposer_logits.reshape(-1)
        pi = graph_prior.reshape(-1).clamp_min(0.0)
        ev = support_evidence.reshape(-1)
        cons = support_consistency.reshape(-1)
        if not (u.numel() == pi.numel() == ev.numel() == cons.numel()):
            raise ValueError("refine_posterior: module vector size mismatch")
        m = int(u.numel())
        if m == 0:
            z0 = u.new_zeros((0,))
            return {"z_refined": z0, "z_init": z0, "base_logits": z0}

        pi = pi / pi.sum().clamp_min(eps)
        log_pi = torch.log(pi + eps)
        base_logits = alpha * u + beta * log_pi + gamma * ev + rho * cons
        z = F.softmax(base_logits, dim=-1)
        rounds_n = max(0, int(rounds))
        damping_f = float(min(max(damping, 0.0), 1.0))

        for _ in range(rounds_n):
            delta = self.refiner(
                u=u,
                log_pi=log_pi,
                evidence=ev,
                consistency=cons,
                z_prev=z,
                cell_id=cell_id,
                drug_id=drug_id,
            )
            z_new = F.softmax(base_logits + float(lambda_delta) * delta, dim=-1)
            z = damping_f * z + (1.0 - damping_f) * z_new
            z = z / z.sum().clamp_min(eps)
        return {"z_refined": z, "z_init": F.softmax(base_logits, dim=-1), "base_logits": base_logits}

    def predict_margin(
        self,
        h: torch.Tensor,
        mechanism_logits: torch.Tensor,
        base_margin: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_hat = F.softmax(mechanism_logits, dim=-1)
        consistency_logits = self.module_consistency_logits(h)
        consistency_probs = torch.sigmoid(consistency_logits)
        expected_deg_prob = (z_hat * consistency_probs).sum(dim=-1).clamp(1e-6, 1.0 - 1e-6)
        expected_deg_logit = torch.logit(expected_deg_prob)
        m_emb = z_hat @ self.module_embedding
        feat = torch.cat([h, m_emb, base_margin.unsqueeze(-1), expected_deg_logit.unsqueeze(-1)], dim=-1)
        margin = self.label_head(feat).squeeze(-1)
        return {
            "margin": margin,
            "z_hat": z_hat,
            "m_emb": m_emb,
            "consistency_logits": consistency_logits,
            "consistency_probs": consistency_probs,
            "expected_deg_prob": expected_deg_prob,
            "expected_deg_logit": expected_deg_logit,
        }
