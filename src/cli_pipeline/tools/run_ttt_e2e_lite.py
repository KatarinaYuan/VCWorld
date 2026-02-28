#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTT-E2E-lite training with chunked inner scan + first-order outer optimization."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttt_common import load_label_map, load_prompts, render_chat_prompt


def _trainable_named_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _snapshot_params(named_params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in named_params}


def _restore_params(named_params: List[Tuple[str, torch.nn.Parameter]], snap: Dict[str, torch.Tensor]) -> None:
    for n, p in named_params:
        p.data.copy_(snap[n].data)


def _build_model_and_tokenizer(args):
    print(f"[TTT-E2E-lite] Loading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[TTT-E2E-lite] Loading base model from: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    if args.init_adapter:
        print(f"[TTT-E2E-lite] Loading initial adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        print("[TTT-E2E-lite] Initializing fresh LoRA adapter")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        )
        model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def _build_train_sequences(args, tokenizer) -> List[List[int]]:
    labels = load_label_map(args.labels_csv)
    records = load_prompts(args.prompts_file)
    seqs: List[List[int]] = []
    for rec in records:
        if not rec.pert or not rec.gene or not rec.system_prompt or not rec.user_input:
            continue
        row = labels.get((rec.pert, rec.gene))
        if row is None or row.get("split", "").strip().lower() != "train":
            continue
        txt = render_chat_prompt(tokenizer, rec.system_prompt, rec.user_input)
        ids = tokenizer(txt, add_special_tokens=False).input_ids
        if len(ids) < 32:
            continue
        if args.max_seq_len > 0:
            ids = ids[:args.max_seq_len]
        seqs.append(ids)
    return seqs


def run(args) -> None:
    print("[TTT-E2E-lite] ===== Run Config =====")
    print(
        f"[TTT-E2E-lite] model={args.model_name} | prompts={args.prompts_file} | labels={args.labels_csv} | out_dir={args.output_dir}"
    )
    print(
        f"[TTT-E2E-lite] epochs={args.num_epochs} inner_steps={args.inner_steps} inner_lr={args.inner_lr} "
        f"outer_lr={args.outer_lr} chunk_tokens={args.chunk_tokens} max_seq_len={args.max_seq_len}"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = _build_model_and_tokenizer(args)
    model.train()
    print("[TTT-E2E-lite] Loading train sequences")
    seqs = _build_train_sequences(args, tokenizer)
    if not seqs:
        raise RuntimeError("No train sequences available for TTT-E2E-lite.")
    print(f"[TTT-E2E-lite] Train sequences: {len(seqs)}")

    named_params = _trainable_named_params(model)
    print(f"[TTT-E2E-lite] Trainable params: {len(named_params)} tensors")
    inner_opt = torch.optim.SGD([p for _, p in named_params], lr=args.inner_lr)
    outer_opt = torch.optim.AdamW([p for _, p in named_params], lr=args.outer_lr)
    device = next(model.parameters()).device
    print(f"[TTT-E2E-lite] Running on device: {device}")

    for epoch in range(1, args.num_epochs + 1):
        print(f"[TTT-E2E-lite] ---- Epoch {epoch}/{args.num_epochs} start ----")
        total_outer = 0.0
        n_effective = 0
        for i, ids in enumerate(seqs, 1):
            # Chunked scan over a single sequence (no support/query split).
            chunk_tokens = max(2, args.chunk_tokens)
            chunks = [ids[j:j + chunk_tokens] for j in range(0, len(ids), chunk_tokens)]
            chunks = [c for c in chunks if len(c) >= 2]
            if not chunks:
                continue

            chunk_tensors = [torch.tensor([c], dtype=torch.long, device=device) for c in chunks]

            # Save initial trainable params before inner updates.
            init_snap = _snapshot_params(named_params)

            # Inner loop: scan over chunks and update on each chunk.
            for chunk_t in chunk_tensors:
                if args.inner_max_tokens > 0 and chunk_t.shape[1] > args.inner_max_tokens:
                    chunk_t = chunk_t[:, -args.inner_max_tokens:]
                for _ in range(args.inner_steps):
                    inner_out = model(
                        input_ids=chunk_t,
                        attention_mask=torch.ones_like(chunk_t),
                        labels=chunk_t,
                    )
                    inner_loss = inner_out.loss
                    inner_loss.backward()
                    inner_opt.step()
                    inner_opt.zero_grad(set_to_none=True)

            # Outer loss: mean over all chunks on adapted model.
            outer_losses = []
            for chunk_t in chunk_tensors:
                out = model(
                    input_ids=chunk_t,
                    attention_mask=torch.ones_like(chunk_t),
                    labels=chunk_t,
                )
                outer_losses.append(out.loss)
            outer_loss = torch.stack(outer_losses).mean()

            # First-order meta approximation: use grads at adapted params,
            # then apply them on initial params.
            grads = torch.autograd.grad(
                outer_loss,
                [p for _, p in named_params],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            _restore_params(named_params, init_snap)
            outer_opt.zero_grad(set_to_none=True)
            for (_, p), g in zip(named_params, grads):
                if g is not None:
                    p.grad = g.detach()
            outer_opt.step()

            total_outer += float(outer_loss.detach().cpu())
            n_effective += 1
            if i % args.log_every == 0:
                avg_outer = total_outer / max(1, i)
                print(
                    f"[TTT-E2E-lite] epoch={epoch} step={i}/{len(seqs)} chunks={len(chunks)} "
                    f"outer_loss={float(outer_loss.detach().cpu()):.4f} avg_outer_loss={avg_outer:.4f}"
                )

        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        avg_epoch_outer = total_outer / max(1, n_effective)
        print(
            f"[TTT-E2E-lite] Epoch {epoch} done | effective_steps={n_effective} "
            f"| avg_outer_loss={avg_epoch_outer:.4f}"
        )
        print(f"[TTT-E2E-lite] Saved epoch checkpoint: {ckpt_dir}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[TTT-E2E-lite] Saved final adapter: {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TTT-E2E-lite (first-order) adapter.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-adapter", default=None, help="Optional LoRA adapter checkpoint as init")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=5e-5)
    p.add_argument("--outer-lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--inner-max-tokens", type=int, default=2048)
    p.add_argument("--chunk-tokens", type=int, default=1024, help="Chunk size for inner scan / outer averaging.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
