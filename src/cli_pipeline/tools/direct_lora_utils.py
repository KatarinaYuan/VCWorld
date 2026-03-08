#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for direct supervised LoRA baseline (query-only inference)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ttt_common import PromptRecord, load_label_map, load_prompts, render_chat_prompt


@dataclass
class DirectExample:
    idx: int
    prompt_id: Optional[int]
    pert: str
    gene: str
    split: str
    label: Optional[int]
    system_prompt: str
    user_input: str
    record: PromptRecord
    perturbation_key: str


def load_direct_examples(
    *,
    prompts_file: str,
    labels_csv: str,
    split: str,
    default_cell_id: int = 0,
    require_label: bool = True,
) -> List[DirectExample]:
    """Load examples from prompt blocks + label CSV without graph filtering."""
    split_norm = split.strip().lower()
    label_map = load_label_map(labels_csv)
    records = load_prompts(prompts_file)
    out: List[DirectExample] = []

    for i, rec in enumerate(records):
        if not rec.pert or not rec.gene or not rec.system_prompt or not rec.user_input:
            continue
        row = label_map.get((rec.pert, rec.gene))
        if row is None:
            continue
        row_split = str(row.get("split", "")).strip().lower()
        if row_split != split_norm:
            continue

        label_val: Optional[int]
        if "label" in row and str(row.get("label", "")).strip() != "":
            try:
                label_val = int(row["label"])
            except (TypeError, ValueError):
                label_val = None
        else:
            label_val = None
        if require_label and label_val is None:
            continue

        out.append(
            DirectExample(
                idx=i,
                prompt_id=rec.prompt_id,
                pert=str(rec.pert),
                gene=str(rec.gene),
                split=row_split,
                label=label_val,
                system_prompt=str(rec.system_prompt),
                user_input=str(rec.user_input),
                record=rec,
                perturbation_key=f"{int(default_cell_id)}::{str(rec.pert).strip().lower()}",
            )
        )
    return out


def split_examples_by_perturbation(
    examples: Sequence[DirectExample],
    val_fraction: float,
    seed: int,
) -> Tuple[List[DirectExample], List[DirectExample]]:
    """Hold out full perturbation groups for validation to avoid leakage."""
    if not examples:
        return [], []
    if val_fraction <= 0.0:
        return list(examples), []
    groups: Dict[str, List[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex.perturbation_key, []).append(i)
    keys = sorted(groups.keys())
    if len(keys) <= 1:
        return list(examples), []
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_fraction)))
    n_val = min(n_val, len(keys) - 1)
    val_keys = set(keys[:n_val])
    train_out: List[DirectExample] = []
    val_out: List[DirectExample] = []
    for ex in examples:
        if ex.perturbation_key in val_keys:
            val_out.append(ex)
        else:
            train_out.append(ex)
    return train_out, val_out


def build_query_prompt_ids(
    *,
    tokenizer,
    example: DirectExample,
    max_input_tokens: int,
) -> List[int]:
    prompt = render_chat_prompt(tokenizer, example.system_prompt, example.user_input)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if max_input_tokens > 0:
        ids = ids[-max_input_tokens:]
    return ids


def build_label_candidate_id_lists(tokenizer, label_candidates: Sequence[str]) -> List[List[int]]:
    return [
        tokenizer(" " + cand if not cand.startswith(" ") else cand, add_special_tokens=False).input_ids
        for cand in label_candidates
    ]


def score_label_candidates(
    *,
    model,
    device: torch.device,
    prompt_ids: List[int],
    prefix_ids: List[int],
    candidate_id_lists: List[List[int]],
) -> List[float]:
    scores: List[float] = []
    base_ids = prompt_ids + prefix_ids
    for cand_ids in candidate_id_lists:
        if not cand_ids:
            scores.append(float("-inf"))
            continue
        full_ids = base_ids + cand_ids
        input_t = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_t, attention_mask=torch.ones_like(input_t)).logits
            log_probs = F.log_softmax(logits, dim=-1)
        start = len(base_ids)
        s = 0.0
        for j, tok in enumerate(cand_ids):
            pos = start + j - 1
            s += float(log_probs[0, pos, tok].detach().cpu())
        scores.append(s)
    return scores


def extract_base_scores(label_candidates: Sequence[str], scores: Sequence[float]) -> Dict[str, float]:
    out = {k.strip().lower(): float(v) for k, v in zip(label_candidates, scores)}
    out.setdefault("yes", float("-inf"))
    out.setdefault("no", float("-inf"))
    out.setdefault("insufficient", float("-inf"))
    return out


def infer_label_with_abstention(
    *,
    score_map: Dict[str, float],
    abstain_threshold: Optional[float],
    disable_abstain: bool,
) -> Tuple[str, int, bool]:
    s_yes = float(score_map.get("yes", float("-inf")))
    s_no = float(score_map.get("no", float("-inf")))
    s_ins = float(score_map.get("insufficient", float("-inf")))
    if disable_abstain:
        pred_bin = 1 if s_yes >= s_no else 0
        pred_label = "yes" if pred_bin == 1 else "no"
        return pred_label, pred_bin, True

    tau = float(abstain_threshold if abstain_threshold is not None else 0.0)
    s_best = max(s_yes, s_no)
    if (s_best - s_ins) < tau:
        return "insufficient", -1, False
    pred_bin = 1 if s_yes >= s_no else 0
    pred_label = "yes" if pred_bin == 1 else "no"
    return pred_label, pred_bin, True


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_answered_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_answered": 0}
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    acc = _safe_div(float(tp + tn), float(n))
    prec = _safe_div(float(tp), float(tp + fp))
    rec = _safe_div(float(tp), float(tp + fn))
    f1 = _safe_div(2.0 * prec * rec, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "n_answered": n}


def evaluate_label_ranking_predictions(
    *,
    y_true: Sequence[int],
    score_maps: Sequence[Dict[str, float]],
    abstain_threshold: Optional[float],
    disable_abstain: bool,
) -> Dict[str, float]:
    total = len(y_true)
    answered_true: List[int] = []
    answered_pred: List[int] = []
    n_yes_pred = 0
    n_ins = 0
    for yt, sm in zip(y_true, score_maps):
        _, pred_bin, answered = infer_label_with_abstention(
            score_map=sm,
            abstain_threshold=abstain_threshold,
            disable_abstain=disable_abstain,
        )
        if not answered:
            n_ins += 1
            continue
        if pred_bin == 1:
            n_yes_pred += 1
        answered_true.append(int(yt))
        answered_pred.append(int(pred_bin))

    answered = len(answered_true)
    answered_rate = _safe_div(float(answered), float(total)) if total > 0 else 0.0
    metrics = compute_answered_metrics(answered_true, answered_pred)
    effective_f1 = float(metrics["f1"]) * answered_rate
    return {
        **metrics,
        "answered_rate": answered_rate,
        "abandonment_rate": 1.0 - answered_rate,
        "effective_f1": effective_f1,
        "yes_prediction_rate": _safe_div(float(n_yes_pred), float(answered)) if answered > 0 else 0.0,
        "insufficient_prediction_rate": _safe_div(float(n_ins), float(total)) if total > 0 else 0.0,
    }


def tune_abstain_threshold(
    *,
    y_true: Sequence[int],
    score_maps: Sequence[Dict[str, float]],
    tau_values: Iterable[float],
    disable_abstain: bool,
) -> Tuple[float, Dict[str, float]]:
    if disable_abstain:
        metrics = evaluate_label_ranking_predictions(
            y_true=y_true,
            score_maps=score_maps,
            abstain_threshold=0.0,
            disable_abstain=True,
        )
        return 0.0, metrics
    best_tau = 0.0
    best_metrics: Optional[Dict[str, float]] = None
    best_score = float("-inf")
    for tau in tau_values:
        m = evaluate_label_ranking_predictions(
            y_true=y_true,
            score_maps=score_maps,
            abstain_threshold=float(tau),
            disable_abstain=False,
        )
        score = float(m["effective_f1"])
        if score > best_score:
            best_score = score
            best_tau = float(tau)
            best_metrics = m
    assert best_metrics is not None
    return best_tau, best_metrics


def _collect_available_linear_module_suffixes(model) -> List[str]:
    names: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, torch.nn.Linear) or ("linear" in cls_name)
        if is_linear_like:
            suffix = name.split(".")[-1]
            names.append(suffix)
    uniq = sorted(set(names))
    return uniq


def select_lora_target_modules(
    *,
    model,
    requested_csv: str,
    include_mlp_projections: bool,
) -> List[str]:
    available_suffixes = _collect_available_linear_module_suffixes(model)
    available_set = set(available_suffixes)
    if requested_csv.strip():
        requested = [x.strip() for x in requested_csv.split(",") if x.strip()]
        selected = [x for x in requested if x in available_set]
        if not selected:
            raise RuntimeError(
                "None of requested LoRA target modules exist in model. "
                f"requested={requested} available_linear_suffixes={available_suffixes[:80]}"
            )
        return selected

    core = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp = ["gate_proj", "up_proj", "down_proj"]
    preferred = core + (mlp if include_mlp_projections else [])
    selected = [x for x in preferred if x in available_set]
    if selected:
        return selected

    fallback_candidates = [
        "query_key_value",
        "dense",
        "fc1",
        "fc2",
        "c_attn",
        "c_proj",
        "Wqkv",
        "out_proj",
    ]
    selected = [x for x in fallback_candidates if x in available_set]
    if selected:
        return selected

    raise RuntimeError(f"Failed to infer LoRA target modules. available_linear_suffixes={available_suffixes[:120]}")


def make_tau_grid(min_tau: float, max_tau: float, num_steps: int) -> List[float]:
    steps = max(2, int(num_steps))
    if steps <= 1 or math.isclose(min_tau, max_tau):
        return [float(min_tau)]
    return [float(min_tau + (max_tau - min_tau) * i / float(steps - 1)) for i in range(steps)]
