#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate direct supervised LoRA baseline via label-ranking (query-only)."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from direct_lora_utils import (
    DirectExample,
    build_label_candidate_id_lists,
    build_query_prompt_ids,
    evaluate_label_ranking_predictions,
    extract_base_scores,
    infer_label_with_abstention,
    load_direct_examples,
    make_tau_grid,
    score_label_candidates,
    tune_abstain_threshold,
)


def _lazy_import_peft():
    try:
        from peft import PeftModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("peft is required for adapter evaluation. Please install `peft`.") from exc
    return PeftModel


def _build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
        model_kwargs["device_map"] = "auto" if args.device_map_auto else None
    base = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if (not args.load_in_4bit) and (not args.device_map_auto) and torch.cuda.is_available():
        base = base.to("cuda")

    if args.adapter_dir:
        PeftModel = _lazy_import_peft()
        model = PeftModel.from_pretrained(base, args.adapter_dir)
    else:
        model = base
    model.eval()
    return model, tokenizer


def _score_split(
    *,
    model,
    tokenizer,
    model_device: torch.device,
    examples: List[DirectExample],
    max_input_tokens: int,
    label_candidates: List[str],
    label_prefix: str,
    log_every_examples: int,
) -> Dict[str, List]:
    prefix_ids = tokenizer(label_prefix, add_special_tokens=False).input_ids
    candidate_ids = build_label_candidate_id_lists(tokenizer, label_candidates)
    y_true: List[int] = []
    score_maps: List[Dict[str, float]] = []
    meta: List[Dict[str, str]] = []
    total_examples = len(examples)
    log_every = max(1, int(log_every_examples))
    with torch.no_grad():
        for i, ex in enumerate(examples, 1):
            if ex.label is None:
                continue
            prompt_ids = build_query_prompt_ids(
                tokenizer=tokenizer,
                example=ex,
                max_input_tokens=max_input_tokens,
            )
            scores = score_label_candidates(
                model=model,
                device=model_device,
                prompt_ids=prompt_ids,
                prefix_ids=prefix_ids,
                candidate_id_lists=candidate_ids,
            )
            score_map = extract_base_scores(label_candidates, scores)
            y_true.append(int(ex.label))
            score_maps.append(score_map)
            meta.append(
                {
                    "prompt_id": "" if ex.prompt_id is None else str(ex.prompt_id),
                    "pert": ex.pert,
                    "gene": ex.gene,
                    "split": ex.split,
                }
            )
            if (i % log_every == 0) or (i == total_examples):
                print(f"[Direct-LoRA-eval] scored {i}/{total_examples} examples", flush=True)
    return {"y_true": y_true, "score_maps": score_maps, "meta": meta}


def run(args) -> None:
    if args.use_qlora and not args.load_in_4bit:
        args.load_in_4bit = True
        print("[Direct-LoRA-eval] --use-qlora enabled -> forcing --load-in-4bit")
    os.makedirs(args.out_dir, exist_ok=True)
    model, tokenizer = _build_model_and_tokenizer(args)
    model_device = next(model.parameters()).device

    examples = load_direct_examples(
        prompts_file=args.prompts_file,
        labels_csv=args.labels_csv,
        split=args.split,
        default_cell_id=args.default_cell_id,
        require_label=True,
    )
    if not examples:
        raise RuntimeError("No evaluable examples for requested split.")
    print(f"[Direct-LoRA-eval] split={args.split} examples={len(examples)}")

    label_candidates = [x.strip() for x in args.label_candidates.split(",") if x.strip()]
    if not {"yes", "no", "insufficient"}.issubset({x.lower() for x in label_candidates}):
        raise RuntimeError("--label-candidates must include yes,no,insufficient")

    scored = _score_split(
        model=model,
        tokenizer=tokenizer,
        model_device=model_device,
        examples=examples,
        max_input_tokens=args.max_input_tokens,
        label_candidates=label_candidates,
        label_prefix=args.label_prefix,
        log_every_examples=args.log_every_examples,
    )
    y_true: List[int] = scored["y_true"]
    score_maps: List[Dict[str, float]] = scored["score_maps"]
    meta: List[Dict[str, str]] = scored["meta"]

    if args.tune_abstain_threshold and not args.disable_abstain:
        tau_grid = make_tau_grid(args.tau_min, args.tau_max, args.tau_steps)
        tau, metrics = tune_abstain_threshold(
            y_true=y_true,
            score_maps=score_maps,
            tau_values=tau_grid,
            disable_abstain=False,
        )
    else:
        tau = float(args.abstain_threshold)
        metrics = evaluate_label_ranking_predictions(
            y_true=y_true,
            score_maps=score_maps,
            abstain_threshold=tau,
            disable_abstain=args.disable_abstain,
        )

    summary = {
        "method": "direct_supervised_lora",
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "split": args.split,
        "num_examples": len(y_true),
        "disable_abstain": bool(args.disable_abstain),
        "abstain_threshold": tau,
        "metrics": metrics,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    rows_path = os.path.join(args.out_dir, "evaluation_rows.csv")
    with open(rows_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "pert",
                "gene",
                "split",
                "true_label",
                "pred_label",
                "pred_binary",
                "yes_score",
                "no_score",
                "insufficient_score",
                "answered",
            ],
        )
        writer.writeheader()
        for i, (yt, sm, m) in enumerate(zip(y_true, score_maps, meta)):
            pred_label, pred_binary, answered = infer_label_with_abstention(
                score_map=sm,
                abstain_threshold=tau,
                disable_abstain=args.disable_abstain,
            )
            writer.writerow(
                {
                    "prompt_id": m["prompt_id"],
                    "pert": m["pert"],
                    "gene": m["gene"],
                    "split": m["split"],
                    "true_label": int(yt),
                    "pred_label": pred_label,
                    "pred_binary": pred_binary,
                    "yes_score": float(sm.get("yes", float("-inf"))),
                    "no_score": float(sm.get("no", float("-inf"))),
                    "insufficient_score": float(sm.get("insufficient", float("-inf"))),
                    "answered": int(bool(answered)),
                }
            )
    print(f"[Direct-LoRA-eval] saved summary -> {os.path.join(args.out_dir, 'evaluation_summary.json')}")
    print(f"[Direct-LoRA-eval] saved rows -> {rows_path}")
    print(
        "[Direct-LoRA-eval] "
        f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
        f"effective_f1={metrics['effective_f1']:.4f} answered_rate={metrics['answered_rate']:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate direct supervised LoRA baseline.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--adapter-dir", default="", help="LoRA adapter directory. Empty => evaluate base model.")
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--default-cell-id", type=int, default=0)

    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--log-every-examples", type=int, default=50)
    p.add_argument("--label-candidates", default="yes,no,insufficient")
    p.add_argument("--label-prefix", default="\nFinal Deterministic Prediction:\n")

    p.add_argument("--disable-abstain", action="store_true")
    p.add_argument("--abstain-threshold", type=float, default=0.0)
    p.add_argument("--tune-abstain-threshold", action="store_true")
    p.add_argument("--tau-min", type=float, default=-2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--tau-steps", type=int, default=81)

    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--use-qlora", action="store_true", help="Alias of --load-in-4bit for QLoRA-style loading.")
    p.add_argument("--device-map-auto", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
