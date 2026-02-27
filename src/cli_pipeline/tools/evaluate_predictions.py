#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate VCWorld predictions with classification metrics and Q-score."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROMPT_SEPARATOR = "================================================================================"
PROMPT_HEADER_RE = re.compile(r"===\s*Prompt\s*(\d+)\s*\((.*?)\)\s*===", re.DOTALL)
QUERY_HEADER_RE = re.compile(r"---\s*Query for Prompt\s*(\d+)\s*---")


def _split_blocks(text: str) -> List[str]:
    return [b.strip() for b in text.split(PROMPT_SEPARATOR) if b.strip()]


def _parse_prompt_header(block: str) -> Tuple[Optional[int], str, str]:
    m = PROMPT_HEADER_RE.search(block)
    if not m:
        return None, "", ""
    prompt_id = int(m.group(1))
    pair = m.group(2).strip()
    if "|" in pair:
        pert, gene = [x.strip() for x in pair.split("|", 1)]
    else:
        pert, gene = pair, ""
    return prompt_id, pert, gene


def _parse_prompts(prompts_file: Path) -> List[dict]:
    blocks = _split_blocks(prompts_file.read_text(encoding="utf-8"))
    rows = []
    for idx, block in enumerate(blocks):
        prompt_id, pert, gene = _parse_prompt_header(block)
        rows.append(
            {
                "idx_0based": idx,
                "prompt_id": prompt_id,
                "pert": pert,
                "gene": gene,
                "prompt_block": block,
            }
        )
    return rows


def _extract_pred_label(task: str, query_text: str) -> str:
    """
    Returns one of:
      de: yes / no / insufficient
      dir: increase / decrease / insufficient
    Incomplete outputs are treated as insufficient by design.
    """
    text = query_text.strip()
    lower = text.lower()
    final_anchor = lower.rfind("final deterministic prediction")
    if final_anchor >= 0:
        focus = text[final_anchor:].lower()
    else:
        focus = "\n".join(text.splitlines()[-50:]).lower()

    if task == "de":
        # Explicit "no" patterns first to avoid false positive from generic "impact".
        if (
            re.search(r"^\s*[\*\-\s]*no[\.\:\s]", focus, re.MULTILINE)
            or re.search(r"\bdoes\s+not\s+(impact|affect|result\s+in\s+differential\s+expression)\b", focus)
            or re.search(r"\b(no|without)\s+(impact|effect)\b", focus)
        ):
            return "no"
        if (
            re.search(r"^\s*[\*\-\s]*yes[\.\:\s]", focus, re.MULTILINE)
            or re.search(r"\bresults?\s+in\s+differential\s+expression\b", focus)
            or re.search(r"\b(predict|prediction).*?\b(differential\s+expression|impact|affect)\b", focus)
            or re.search(r"\bperturbation\b.*\b(impacts?|affects?)\b", focus)
        ):
            return "yes"
        if "insufficient evidence" in focus:
            return "insufficient"
        return "insufficient"

    # dir
    if (
        re.search(r"^\s*[\*\-\s]*decrease[\.\:\s]", focus, re.MULTILINE)
        or re.search(r"\bresults?\s+in\s+(a\s+)?decrease\b", focus)
        or re.search(r"\bpredict.*\bdecrease\b", focus)
        or re.search(r"\bdecreases?\s+expression\b", focus)
    ):
        return "decrease"
    if (
        re.search(r"^\s*[\*\-\s]*increase[\.\:\s]", focus, re.MULTILINE)
        or re.search(r"\bresults?\s+in\s+(an?\s+)?increase\b", focus)
        or re.search(r"\bpredict.*\bincrease\b", focus)
        or re.search(r"\bincreases?\s+expression\b", focus)
    ):
        return "increase"
    if "insufficient evidence" in focus:
        return "insufficient"
    return "insufficient"


def _parse_predictions(task: str, predictions_file: Path) -> Dict[int, dict]:
    blocks = _split_blocks(predictions_file.read_text(encoding="utf-8"))
    by_prompt_id: Dict[int, dict] = {}
    for block in blocks:
        m = QUERY_HEADER_RE.search(block)
        if not m:
            continue
        prompt_id = int(m.group(1))
        pred_label = _extract_pred_label(task, block)
        by_prompt_id[prompt_id] = {
            "prompt_id": prompt_id,
            "pred_label": pred_label,
            "query_block": block,
        }
    return by_prompt_id


def _load_truth_map(truth_csv: Path, split: Optional[str]) -> Dict[Tuple[str, str], int]:
    truth: Dict[Tuple[str, str], int] = {}
    collisions = 0
    with truth_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split and row.get("split", "") != split:
                continue
            pert = row["pert"].strip()
            gene = row["gene"].strip()
            label = int(row["label"])
            key = (pert, gene)
            if key in truth and truth[key] != label:
                collisions += 1
            truth[key] = label
    if collisions > 0:
        print(f"WARNING: found {collisions} duplicated (pert,gene) with conflicting labels; last one kept.")
    return truth


def _pred_to_binary(task: str, pred_label: str) -> Optional[int]:
    if task == "de":
        if pred_label == "yes":
            return 1
        if pred_label == "no":
            return 0
        return None
    if pred_label == "increase":
        return 1
    if pred_label == "decrease":
        return 0
    return None


def _safe_auroc(y_true: List[int], y_score: List[int]) -> Optional[float]:
    if len(set(y_true)) < 2:
        return None
    pos_scores = [s for y, s in zip(y_true, y_score) if y == 1]
    neg_scores = [s for y, s in zip(y_true, y_score) if y == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None

    better = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                better += 1.0
            elif ps == ns:
                better += 0.5
    return better / (n_pos * n_neg)


def _safe_auprc(y_true: List[int], y_score: List[int]) -> Optional[float]:
    if len(set(y_true)) < 2:
        return None
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    total_pos = sum(y_true)
    if total_pos == 0:
        return None

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for score, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_pos
        if y == 1:
            ap += precision * (recall - prev_recall)
            prev_recall = recall
    return ap


def _binary_metrics(y_true: List[int], y_pred: List[int], y_score: List[int]) -> dict:
    n = len(y_true)
    if n == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "auprc": None,
            "n_answered": 0,
        }
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
        "n_answered": n,
    }


def _effective_metrics(metrics_answered: dict, answered_rate: Optional[float]) -> dict:
    if answered_rate is None:
        return {
            "effective_accuracy": None,
            "effective_precision": None,
            "effective_recall": None,
            "effective_f1": None,
            "effective_auroc": None,
            "effective_auprc": None,
        }

    def _eff(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        return float(v) * answered_rate

    return {
        "effective_accuracy": _eff(metrics_answered.get("accuracy")),
        "effective_precision": _eff(metrics_answered.get("precision")),
        "effective_recall": _eff(metrics_answered.get("recall")),
        "effective_f1": _eff(metrics_answered.get("f1")),
        "effective_auroc": _eff(metrics_answered.get("auroc")),
        "effective_auprc": _eff(metrics_answered.get("auprc")),
    }


def evaluate(
    *,
    task: str,
    prompts_file: Path,
    predictions_file: Path,
    truth_csv: Path,
    out_dir: Path,
    split: Optional[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _parse_prompts(prompts_file)
    pred_by_id = _parse_predictions(task, predictions_file)
    truth_map = _load_truth_map(truth_csv, split=split)
    

    rows = []
    missing_pred = 0
    missing_truth = 0

    for item in prompts:
        prompt_id = item["prompt_id"]
        pert = item["pert"]
        gene = item["gene"]
        pred = pred_by_id.get(prompt_id) if prompt_id is not None else None
        pred_label = pred["pred_label"] if pred else "insufficient"
        if pred is None:
            missing_pred += 1

        true_label = truth_map.get((pert, gene))
        if true_label is None:
            missing_truth += 1

        pred_binary = _pred_to_binary(task, pred_label)
        rows.append(
            {
                "idx_0based": item["idx_0based"],
                "prompt_id": prompt_id,
                "pert": pert,
                "gene": gene,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_binary": pred_binary,
                "answered": pred_binary is not None,
                "match": (pred_binary == true_label) if (pred_binary is not None and true_label is not None) else None,
            }
        )

    eval_rows = [r for r in rows if r["true_label"] is not None]
    answered_rows = [r for r in eval_rows if r["answered"]]
    unanswered_rows = [r for r in eval_rows if not r["answered"]]

    total_eval = len(eval_rows)
    answered = len(answered_rows)
    unanswered = len(unanswered_rows)

    q_score = (unanswered / total_eval) if total_eval > 0 else None
    answered_rate = (answered / total_eval) if total_eval > 0 else None

    metrics = {}
    if answered > 0:
        y_true = [int(r["true_label"]) for r in answered_rows]
        y_pred = [int(r["pred_binary"]) for r in answered_rows]
        y_score = y_pred  # hard-label score for AUROC/AUPRC
        metrics = _binary_metrics(y_true, y_pred, y_score)
    else:
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "auprc": None,
            "n_answered": 0,
        }
    effective = _effective_metrics(metrics, answered_rate)

    summary = {
        "task": task,
        "split": split if split is not None else "all",
        "total_prompts": len(prompts),
        "total_evaluable": total_eval,
        "missing_prediction_blocks": missing_pred,
        "missing_truth_rows": missing_truth,
        "answered_prompts_AP": answered,
        "abandoned_prompts_QP": unanswered,
        "answered_rate_AP_over_total": answered_rate,
        "q_score_abandonment_rate": q_score,
        "robustness_1_minus_q_score": (1 - q_score) if q_score is not None else None,
        "metrics_answered_only": metrics,
        "metrics_effective": effective,
        "note": (
            "Incomplete outputs are treated as insufficient. Classification metrics are computed on answered-only "
            "subset; Q-score captures abstention behavior. Effective metrics are answered-only metrics multiplied "
            "by answered_rate (1 - q_score)."
        ),
    }

    summary_path = out_dir / "evaluation_summary.json"
    rows_path = out_dir / "evaluation_rows.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "idx_0based",
        "prompt_id",
        "pert",
        "gene",
        "true_label",
        "pred_label",
        "pred_binary",
        "answered",
        "match",
    ]
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved summary: {summary_path}")
    print(f"Saved rows: {rows_path}")
    print(
        f"Evaluable={total_eval} | Answered={answered} | Abandoned={unanswered} | "
        f"Q-score={q_score if q_score is not None else 'NA'}"
    )
    if metrics["accuracy"] is not None:
        print(
            "Answered-only metrics: "
            f"acc={metrics['accuracy']:.4f}, prec={metrics['precision']:.4f}, "
            f"rec={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"auroc={metrics['auroc'] if metrics['auroc'] is not None else 'NA'}, "
            f"auprc={metrics['auprc'] if metrics['auprc'] is not None else 'NA'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VCWorld DE/DIR predictions.")
    parser.add_argument("--task", required=True, choices=["de", "dir"])
    parser.add_argument("--prompts", required=True, help="Prompts txt file")
    parser.add_argument("--predictions", required=True, help="Predictions txt file")
    parser.add_argument("--truth-csv", required=True, help="Ground-truth CSV (pert,gene,label,split)")
    parser.add_argument("--out-dir", required=True, help="Output directory for evaluation artifacts")
    parser.add_argument("--split", default="test", help="Split filter from truth CSV; use 'all' for no filter")
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    evaluate(
        task=args.task,
        prompts_file=Path(args.prompts),
        predictions_file=Path(args.predictions),
        truth_csv=Path(args.truth_csv),
        out_dir=Path(args.out_dir),
        split=split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
