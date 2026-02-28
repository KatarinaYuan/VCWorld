#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTT-lite inference: per-test-sample inner update then predict then reset."""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttt_common import format_prediction_block, load_label_map, load_prompts, render_chat_prompt


def _trainable_named_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _snapshot_params(named_params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in named_params}


def _restore_params(named_params: List[Tuple[str, torch.nn.Parameter]], snap: Dict[str, torch.Tensor]) -> None:
    for n, p in named_params:
        p.data.copy_(snap[n].data)


def _build_model_and_tokenizer(args):
    print(f"[TTT-lite] Loading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[TTT-lite] Loading base model from: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    if args.init_adapter:
        print(f"[TTT-lite] Loading initial adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        print("[TTT-lite] Initializing fresh LoRA adapter")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        )
        model = get_peft_model(model, lora_cfg)

    model.train()
    return model, tokenizer


def run(args) -> None:
    print("[TTT-lite] ===== Run Config =====")
    print(
        f"[TTT-lite] model={args.model_name} | prompts={args.prompts_file} | labels={args.labels_csv} | out={args.out}"
    )
    print(
        f"[TTT-lite] inner_steps={args.inner_steps} inner_lr={args.inner_lr} inner_max_tokens={args.inner_max_tokens} "
        f"max_new_tokens={args.max_new_tokens} temperature={args.temperature} top_p={args.top_p}"
    )
    model, tokenizer = _build_model_and_tokenizer(args)
    print("[TTT-lite] Loading prompts and labels")
    labels = load_label_map(args.labels_csv)
    records = load_prompts(args.prompts_file)
    print(f"[TTT-lite] Loaded prompt records: {len(records)} | label rows: {len(labels)}")

    test_records = []
    for rec in records:
        if not rec.pert or not rec.gene or not rec.system_prompt or not rec.user_input:
            continue
        row = labels.get((rec.pert, rec.gene))
        if row is None:
            continue
        if row.get("split", "").strip().lower() == "test":
            test_records.append(rec)

    if not test_records:
        raise RuntimeError("No test records found from prompts + labels split.")
    print(f"[TTT-lite] Test records to run: {len(test_records)}")

    named_params = _trainable_named_params(model)
    print(f"[TTT-lite] Trainable params for inner update: {len(named_params)} tensors")
    optimizer = torch.optim.AdamW([p for _, p in named_params], lr=args.inner_lr)

    out_blocks: List[str] = []
    device = next(model.parameters()).device
    print(f"[TTT-lite] Running on device: {device}")
    for i, rec in enumerate(test_records, 1):
        header = f"Prompt {rec.prompt_id}" if rec.prompt_id is not None else f"Prompt_{rec.idx}"
        prompt_text = render_chat_prompt(tokenizer, rec.system_prompt, rec.user_input)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        if args.inner_max_tokens > 0:
            prompt_ids = prompt_ids[-args.inner_max_tokens:]

        # Inner adaptation on current test prompt.
        snap = _snapshot_params(named_params)
        inner_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        for _ in range(args.inner_steps):
            out = model(input_ids=inner_input, attention_mask=torch.ones_like(inner_input), labels=inner_input)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            gen = model.generate(
                input_ids=inner_input,
                attention_mask=torch.ones_like(inner_input),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, inner_input.shape[1]:]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        out_blocks.append(format_prediction_block(header, response))
        model.train()

        # Reset to base params for next sample.
        _restore_params(named_params, snap)
        optimizer.zero_grad(set_to_none=True)

        if i % 10 == 0 or i == len(test_records):
            loss_val = float(loss.detach().cpu()) if "loss" in locals() else float("nan")
            print(f"[TTT-lite] done {i}/{len(test_records)} | last_inner_loss={loss_val:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(out_blocks)
    print(f"[TTT-lite] Saved predictions: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run TTT-lite inference on test split prompts.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init-adapter", default=None, help="Optional LoRA adapter checkpoint as TTT init")
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--inner-lr", type=float, default=5e-5)
    p.add_argument("--inner-max-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--line-profile", action="store_true", help="Enable line_profiler for run().")
    p.add_argument("--line-profile-out", default=None, help="Optional output file for line_profiler stats.")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.line_profile:
        run(args)
        return 0

    try:
        from line_profiler import LineProfiler
    except ImportError as e:
        raise RuntimeError(
            "line_profiler is not installed. Install with `pip install line_profiler` "
            "or run without --line-profile."
        ) from e

    print("[TTT-lite] line_profiler enabled")
    lp = LineProfiler()
    profiled_run = lp(run)
    profiled_run(args)
    lp.print_stats()
    if args.line_profile_out:
        with open(args.line_profile_out, "w", encoding="utf-8") as f:
            lp.print_stats(stream=f)
        print(f"[TTT-lite] Saved line_profiler stats to: {args.line_profile_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
