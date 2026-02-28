#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mixed-domain LoRA training (not strict TTT / not inner-outer meta-learning)."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

PROMPT_SEPARATOR = "================================================================================"


def _parse_prompt_block(block_text: str) -> Tuple[Optional[str], Optional[str], str, Optional[str], Optional[str], Optional[str]]:
    header_match = re.search(r"===\s*Prompt\s*(\d+)\s*\((.*?)\)\s*===", block_text)
    prompt_id = header_match.group(1).strip() if header_match else None
    pair = header_match.group(2).strip() if header_match else ""

    if "|" in pair:
        pert, gene = [x.strip() for x in pair.split("|", 1)]
    else:
        pert, gene = None, None

    system_match = re.search(r"\[Start of Prompt\](.*?)\[End of Prompt\]", block_text, re.DOTALL)
    if not system_match:
        return None, None, prompt_id or "", "System prompt markers not found.", pert, gene
    system_prompt = system_match.group(1).strip()

    user_match = re.search(r"\[Start of Input\](.*?)\[End of Input\]", block_text, re.DOTALL)
    if not user_match:
        return None, None, prompt_id or "", "User input markers not found.", pert, gene
    user_input = user_match.group(1).strip()

    return system_prompt, user_input, prompt_id or "", None, pert, gene


def _answer_text(task: str, pert: str, gene: str, label: int) -> str:
    if task == "de":
        if int(label) == 1:
            return f"Yes. Perturbation of {pert} results in differential expression of {gene}."
        return f"No. Perturbation of {pert} does not impact {gene}."
    # dir
    if int(label) == 1:
        return f"Increase. Perturbation of {pert} results in an increase in expression of {gene}."
    return f"Decrease. Perturbation of {pert} results in a decrease in expression of {gene}."


def _load_labels_map(csv_path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["pert"].strip(), row["gene"].strip())
            out[key] = row
    return out


def _render_prompt(tokenizer, system_prompt: str, user_input: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if isinstance(rendered, list):
        return rendered[0]
    return rendered


class TTTDataset(Dataset):
    def __init__(self, items: List[Dict[str, List[int]]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.items[idx]


@dataclass
class CausalDataCollator:
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
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_training_items(
    *,
    task: str,
    prompts_file: str,
    labels_csv: str,
    tokenizer,
    max_seq_len: int,
    use_train_supervised: bool,
    use_test_unsupervised: bool,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, int]]:
    labels_map = _load_labels_map(labels_csv)
    with open(prompts_file, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = [b.strip() for b in content.split(PROMPT_SEPARATOR) if b.strip()]

    items: List[Dict[str, List[int]]] = []
    n_train_sup = 0
    n_test_unsup = 0
    n_skipped = 0
    for block in blocks:
        system_prompt, user_input, _, error, pert, gene = _parse_prompt_block(block)
        if error or system_prompt is None or user_input is None or pert is None or gene is None:
            n_skipped += 1
            continue

        label_row = labels_map.get((pert, gene))
        if label_row is None:
            n_skipped += 1
            continue
        split = label_row.get("split", "").strip().lower()
        label = int(label_row["label"])

        prompt_text = _render_prompt(tokenizer, system_prompt, user_input)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

        if split == "train" and use_train_supervised:
            answer = _answer_text(task, pert, gene, label)
            answer_ids = tokenizer(answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            input_ids = (prompt_ids + answer_ids)[:max_seq_len]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_len]
            attention_mask = [1] * len(input_ids)
            items.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
            n_train_sup += 1
        elif split == "test" and use_test_unsupervised:
            input_ids = prompt_ids[:max_seq_len]
            labels = input_ids.copy()
            attention_mask = [1] * len(input_ids)
            items.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
            n_test_unsup += 1
        else:
            n_skipped += 1

    stats = {
        "num_items_total": len(items),
        "num_train_supervised": n_train_sup,
        "num_test_unsupervised": n_test_unsup,
        "num_skipped": n_skipped,
    }
    return items, stats


def train_ttt(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if args.chat_template:
        with open(args.chat_template, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    model_kwargs = {}
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    items, stats = build_training_items(
        task=args.task,
        prompts_file=args.prompts_file,
        labels_csv=args.labels_csv,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        use_train_supervised=not args.disable_train_supervised,
        use_test_unsupervised=not args.disable_test_unsupervised,
    )
    if stats["num_items_total"] == 0:
        raise RuntimeError("No training samples were constructed. Check prompt/label alignment.")

    print(f"Dataset stats: {stats}")
    train_dataset = TTTDataset(items)
    collator = CausalDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + tokenizer to: {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTT-style LoRA training for VCWorld prompts.")
    p.add_argument("--task", choices=["de", "dir"], required=True)
    p.add_argument("--model-name", required=True, help="Base model path/name, e.g. Llama-3-8B-Instruct")
    p.add_argument("--prompts-file", required=True, help="Prompt txt with both train/test prompts")
    p.add_argument("--labels-csv", required=True, help="CSV with pert,gene,label,split")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chat-template", default=None)

    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--disable-train-supervised", action="store_true")
    p.add_argument("--disable-test-unsupervised", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    train_ttt(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
