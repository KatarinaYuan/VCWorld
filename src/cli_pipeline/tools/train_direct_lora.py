#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct supervised LoRA baseline (answer-only supervision, query-only inference)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from direct_lora_utils import (
    DirectExample,
    build_label_candidate_id_lists,
    build_query_prompt_ids,
    evaluate_label_ranking_predictions,
    extract_base_scores,
    load_direct_examples,
    make_tau_grid,
    score_label_candidates,
    select_lora_target_modules,
    split_examples_by_perturbation,
    tune_abstain_threshold,
)


def _lazy_import_peft():
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("peft is required for direct LoRA baseline. Please install `peft`.") from exc
    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DirectTrainSample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    perturbation_key: str


class DirectLoRADataset(Dataset):
    def __init__(
        self,
        *,
        examples: List[DirectExample],
        tokenizer,
        max_input_tokens: int,
        max_train_seq_len: int,
        label_prefix: str,
        add_eos: bool,
    ) -> None:
        self.samples: List[DirectTrainSample] = []
        prefix_ids = tokenizer(label_prefix, add_special_tokens=False).input_ids
        eos_id = tokenizer.eos_token_id
        for ex in examples:
            if ex.label is None:
                continue
            prompt_ids = build_query_prompt_ids(
                tokenizer=tokenizer,
                example=ex,
                max_input_tokens=max_input_tokens,
            )
            ans_text = " yes" if int(ex.label) == 1 else " no"
            ans_ids = tokenizer(ans_text, add_special_tokens=False).input_ids
            if add_eos and eos_id is not None:
                ans_ids = ans_ids + [int(eos_id)]
            suffix_len = len(prefix_ids) + len(ans_ids)
            if suffix_len <= 0:
                continue
            allowed_prompt = max(0, max_train_seq_len - suffix_len)
            if allowed_prompt > 0:
                prompt_ids = prompt_ids[-allowed_prompt:]
            else:
                prompt_ids = []
            input_ids = prompt_ids + prefix_ids + ans_ids
            labels = ([-100] * (len(prompt_ids) + len(prefix_ids))) + ans_ids
            attn = [1] * len(input_ids)
            self.samples.append(
                DirectTrainSample(
                    input_ids=input_ids,
                    attention_mask=attn,
                    labels=labels,
                    perturbation_key=ex.perturbation_key,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        s = self.samples[idx]
        return {
            "input_ids": s.input_ids,
            "attention_mask": s.attention_mask,
            "labels": s.labels,
            "perturbation_key": s.perturbation_key,
        }


@dataclass
class DirectDataCollator:
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids = []
        attention_mask = []
        labels = []
        for x in features:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + ([-100] * pad_len))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if (not args.load_in_4bit) and (not args.device_map_auto) and torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def _attach_lora_adapter(model, args):
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = _lazy_import_peft()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    target_modules = select_lora_target_modules(
        model=model,
        requested_csv=args.lora_target_modules,
        include_mlp_projections=args.include_mlp_lora,
    )
    print(f"[Direct-LoRA] target_modules={target_modules}")
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, target_modules


def _evaluate_split(
    *,
    model,
    tokenizer,
    model_device: torch.device,
    examples: List[DirectExample],
    max_input_tokens: int,
    label_candidates: List[str],
    label_prefix: str,
    tau: float,
    disable_abstain: bool,
) -> Dict[str, float]:
    prefix_ids = tokenizer(label_prefix, add_special_tokens=False).input_ids
    candidate_ids = build_label_candidate_id_lists(tokenizer, label_candidates)
    score_maps: List[Dict[str, float]] = []
    y_true: List[int] = []
    model.eval()
    with torch.no_grad():
        for ex in examples:
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
            score_maps.append(extract_base_scores(label_candidates, scores))
            y_true.append(int(ex.label))
    return evaluate_label_ranking_predictions(
        y_true=y_true,
        score_maps=score_maps,
        abstain_threshold=tau,
        disable_abstain=disable_abstain,
    )


def run(args) -> None:
    if args.use_qlora and not args.load_in_4bit:
        args.load_in_4bit = True
        print("[Direct-LoRA] --use-qlora enabled -> forcing --load-in-4bit")
    _set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    support_examples = load_direct_examples(
        prompts_file=args.prompts_file,
        labels_csv=args.labels_csv,
        split=args.support_split,
        default_cell_id=args.default_cell_id,
        require_label=True,
    )
    if not support_examples:
        raise RuntimeError("No training/support examples found after filtering.")
    train_examples, val_examples = split_examples_by_perturbation(
        support_examples,
        val_fraction=args.val_perturb_fraction,
        seed=args.seed,
    )
    if not val_examples and not args.allow_no_validation:
        raise RuntimeError("Validation set is empty. Increase --val-perturb-fraction or use --allow-no-validation.")
    print(
        f"[Direct-LoRA] support_total={len(support_examples)} "
        f"train={len(train_examples)} val={len(val_examples)}"
    )

    model, tokenizer = _build_model_and_tokenizer(args)
    model, target_modules = _attach_lora_adapter(model, args)
    model_device = next(model.parameters()).device
    print(f"[Direct-LoRA] model_device={model_device}")

    label_candidates = [x.strip() for x in args.label_candidates.split(",") if x.strip()]
    labels_lower = {x.lower() for x in label_candidates}
    if not {"yes", "no", "insufficient"}.issubset(labels_lower):
        raise RuntimeError("--label-candidates must include yes,no,insufficient")

    train_ds = DirectLoRADataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_input_tokens=args.max_input_tokens,
        max_train_seq_len=args.max_train_seq_len,
        label_prefix=args.label_prefix,
        add_eos=args.add_eos,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No train samples created for answer-only supervision.")
    print(f"[Direct-LoRA] train_samples={len(train_ds)}")
    collator = DirectDataCollator(pad_token_id=int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not args.bf16)

    tau_grid = make_tau_grid(args.tau_min, args.tau_max, args.tau_steps)
    best_val_f1 = float("-inf")
    best_epoch = -1
    best_tau = float(args.abstain_threshold)
    best_metrics: Optional[Dict[str, float]] = None
    best_dir = os.path.join(args.output_dir, "best_adapter")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(model_device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if args.bf16 else torch.float16,
                enabled=use_amp,
            ):
                out = model(**batch)
                loss = out.loss / float(args.gradient_accumulation_steps)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.detach().cpu()) * float(args.gradient_accumulation_steps)
            if (step % args.gradient_accumulation_steps) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every_steps == 0:
                    print(
                        f"[Direct-LoRA][Epoch {epoch}/{args.epochs}] "
                        f"step={global_step} avg_loss={running_loss/max(1, step):.6f}"
                    )

        train_loss_epoch = running_loss / max(1, len(train_loader))
        print(f"[Direct-LoRA] Epoch {epoch} train_loss={train_loss_epoch:.6f}")

        if val_examples:
            if args.tune_abstain_threshold and not args.disable_abstain:
                prefix_ids = tokenizer(args.label_prefix, add_special_tokens=False).input_ids
                candidate_ids = build_label_candidate_id_lists(tokenizer, label_candidates)
                score_maps: List[Dict[str, float]] = []
                y_true: List[int] = []
                model.eval()
                with torch.no_grad():
                    for ex in val_examples:
                        if ex.label is None:
                            continue
                        prompt_ids = build_query_prompt_ids(
                            tokenizer=tokenizer,
                            example=ex,
                            max_input_tokens=args.max_input_tokens,
                        )
                        scores = score_label_candidates(
                            model=model,
                            device=model_device,
                            prompt_ids=prompt_ids,
                            prefix_ids=prefix_ids,
                            candidate_id_lists=candidate_ids,
                        )
                        score_maps.append(extract_base_scores(label_candidates, scores))
                        y_true.append(int(ex.label))
                tau_epoch, metrics = tune_abstain_threshold(
                    y_true=y_true,
                    score_maps=score_maps,
                    tau_values=tau_grid,
                    disable_abstain=args.disable_abstain,
                )
            else:
                tau_epoch = float(args.abstain_threshold)
                metrics = _evaluate_split(
                    model=model,
                    tokenizer=tokenizer,
                    model_device=model_device,
                    examples=val_examples,
                    max_input_tokens=args.max_input_tokens,
                    label_candidates=label_candidates,
                    label_prefix=args.label_prefix,
                    tau=tau_epoch,
                    disable_abstain=args.disable_abstain,
                )

            val_f1 = float(metrics["f1"])
            print(
                f"[Direct-LoRA] val: tau={tau_epoch:.4f} "
                f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
                f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
                f"effective_f1={metrics['effective_f1']:.4f} yes_rate={metrics['yes_prediction_rate']:.4f}"
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_tau = float(tau_epoch)
                best_metrics = metrics
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"[Direct-LoRA] saved best adapter -> {best_dir}")

    final_dir = os.path.join(args.output_dir, "last_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    summary = {
        "method": "direct_supervised_lora",
        "model_name": args.model_name,
        "support_split": args.support_split,
        "num_support_examples": len(support_examples),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "num_train_samples": len(train_ds),
        "lora_target_modules": target_modules,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_abstain_threshold": best_tau,
        "best_val_metrics": best_metrics,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    print(f"[Direct-LoRA] saved summary -> {os.path.join(args.output_dir, 'train_summary.json')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train direct supervised LoRA baseline.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--support-split", default="train")
    p.add_argument("--default-cell-id", type=int, default=0)
    p.add_argument("--val-perturb-fraction", type=float, default=0.2)
    p.add_argument("--allow-no-validation", action="store_true")

    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--max-train-seq-len", type=int, default=2304)
    p.add_argument("--label-candidates", default="yes,no,insufficient")
    p.add_argument("--label-prefix", default="\nFinal Deterministic Prediction:\n")
    p.add_argument("--add-eos", action="store_true")

    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--use-qlora", action="store_true", help="Alias of --load-in-4bit for QLoRA-style training.")
    p.add_argument("--device-map-auto", action="store_true")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="")
    p.add_argument("--include-mlp-lora", action="store_true")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every-steps", type=int, default=20)
    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument("--disable-abstain", action="store_true")
    p.add_argument("--abstain-threshold", type=float, default=0.0)
    p.add_argument("--tune-abstain-threshold", action="store_true")
    p.add_argument("--tau-min", type=float, default=-2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--tau-steps", type=int, default=81)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
