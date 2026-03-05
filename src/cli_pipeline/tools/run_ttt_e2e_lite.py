#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTT-E2E-lite training with chunked inner scan + first-order outer optimization."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ttt_common import format_prediction_block, load_label_map, load_prompts, render_chat_prompt

# kernprof injects `profile` into builtins; provide a no-op fallback for normal runs.
try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):  # type: ignore[no-redef]
        return func


def _trainable_named_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _snapshot_params(named_params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in named_params}


def _restore_params(named_params: List[Tuple[str, torch.nn.Parameter]], snap: Dict[str, torch.Tensor]) -> None:
    for n, p in named_params:
        p.data.copy_(snap[n].data)


def _build_model_and_tokenizer(args, init_adapter_override: str | None = None):
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

    init_adapter = init_adapter_override if init_adapter_override is not None else args.init_adapter
    if init_adapter:
        print(f"[TTT-E2E-lite] Loading initial adapter: {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
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


def _init_wandb(args):
    if not args.wandb:
        return None, None
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            "wandb logging requested but wandb is not installed. Install with `pip install wandb`."
        ) from e

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        group=args.wandb_group or None,
        dir=args.wandb_dir or None,
        mode="offline" if args.wandb_offline else "online",
        config=vars(args),
    )
    return run, wandb


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


def _build_test_records(args):
    labels = load_label_map(args.labels_csv)
    records = load_prompts(args.prompts_file)
    test_records = []
    for rec in records:
        if not rec.pert or not rec.gene or not rec.system_prompt or not rec.user_input:
            continue
        row = labels.get((rec.pert, rec.gene))
        if row is None or row.get("split", "").strip().lower() != "test":
            continue
        test_records.append(rec)
    return test_records


def _score_label_candidates(
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


def _run_prediction(args, model, tokenizer, wandb_run=None, wandb_mod=None) -> None:
    if not args.predictions_out:
        raise RuntimeError("--predictions-out is required when --run-predict is enabled.")
    test_records = _build_test_records(args)
    if not test_records:
        raise RuntimeError("No test records found for prediction.")

    print(
        f"[TTT-E2E-lite] Prediction phase | records={len(test_records)} "
        f"mode={args.prediction_mode} out={args.predictions_out}"
    )
    model.eval()
    device = next(model.parameters()).device
    label_candidates = [x.strip() for x in args.label_candidates.split(",") if x.strip()]
    label_prefix_ids = tokenizer(args.label_prefix, add_special_tokens=False).input_ids
    candidate_id_lists = [
        tokenizer(" " + cand if not cand.startswith(" ") else cand, add_special_tokens=False).input_ids
        for cand in label_candidates
    ]
    if args.prediction_mode == "label-ranking":
        print(f"[TTT-E2E-lite] label-ranking candidates: {label_candidates}")

    out_blocks: List[str] = []
    for i, rec in enumerate(test_records, 1):
        header = f"Prompt {rec.prompt_id}" if rec.prompt_id is not None else f"Prompt_{rec.idx}"
        prompt_text = render_chat_prompt(tokenizer, rec.system_prompt, rec.user_input)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        if args.predict_max_input_tokens > 0:
            prompt_ids = prompt_ids[-args.predict_max_input_tokens:]
        prompt_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        if args.prediction_mode == "label-ranking":
            scores = _score_label_candidates(
                model=model,
                device=device,
                prompt_ids=prompt_ids,
                prefix_ids=label_prefix_ids,
                candidate_id_lists=candidate_id_lists,
            )
            best_idx = max(range(len(scores)), key=lambda k: scores[k])
            best_label = label_candidates[best_idx]
            response = (
                "Final Deterministic Prediction:\n"
                f"{best_label}\n"
                f"[label_ranking_scores] {dict(zip(label_candidates, [round(s, 4) for s in scores]))}"
            )
        else:
            with torch.no_grad():
                gen = model.generate(
                    input_ids=prompt_input,
                    attention_mask=torch.ones_like(prompt_input),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = gen[:, prompt_input.shape[1]:]
            response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        out_blocks.append(format_prediction_block(header, response))
        if i % args.predict_log_every == 0 or i == len(test_records):
            print(f"[TTT-E2E-lite] prediction done {i}/{len(test_records)}")
            if wandb_run is not None and wandb_mod is not None:
                wandb_mod.log(
                    {
                        "predict/processed": i,
                        "predict/total": len(test_records),
                        "predict/progress": i / max(1, len(test_records)),
                    }
                )

    with open(args.predictions_out, "w", encoding="utf-8") as f:
        f.writelines(out_blocks)
    print(f"[TTT-E2E-lite] Saved predictions: {args.predictions_out}")
    if wandb_run is not None and wandb_mod is not None:
        wandb_mod.log({"predict/num_outputs": len(out_blocks)})
        if args.wandb_log_artifacts:
            artifact = wandb_mod.Artifact("ttt-e2e-predictions", type="predictions")
            artifact.add_file(args.predictions_out)
            wandb_run.log_artifact(artifact)


@profile
def run(args) -> None:
    print("[TTT-E2E-lite] ===== Run Config =====")
    print(
        f"[TTT-E2E-lite] model={args.model_name} | prompts={args.prompts_file} | labels={args.labels_csv} | out_dir={args.output_dir}"
    )
    print(
        f"[TTT-E2E-lite] epochs={args.num_epochs} inner_steps={args.inner_steps} inner_lr={args.inner_lr} "
        f"outer_lr={args.outer_lr} chunk_tokens={args.chunk_tokens} max_seq_len={args.max_seq_len}"
    )
    wandb_run, wandb_mod = _init_wandb(args)
    os.makedirs(args.output_dir, exist_ok=True)
    init_adapter = args.predict_ckpt if args.predict_ckpt else args.init_adapter
    model, tokenizer = _build_model_and_tokenizer(args, init_adapter_override=init_adapter)
    model.train()
    if args.predict_only:
        if not args.run_predict:
            raise RuntimeError("--predict-only requires --run-predict.")
        print("[TTT-E2E-lite] predict-only mode: skip training and run prediction only.")
        _run_prediction(args, model, tokenizer, wandb_run=wandb_run, wandb_mod=wandb_mod)
        if wandb_run is not None:
            wandb_run.finish()
        return

    print("[TTT-E2E-lite] Loading train sequences")
    seqs = _build_train_sequences(args, tokenizer)
    if not seqs:
        raise RuntimeError("No train sequences available for TTT-E2E-lite.")
    print(f"[TTT-E2E-lite] Train sequences: {len(seqs)}")
    if wandb_run is not None and wandb_mod is not None:
        wandb_mod.log({"meta/train_sequences": len(seqs)})

    named_params = _trainable_named_params(model)
    print(f"[TTT-E2E-lite] Trainable params: {len(named_params)} tensors")
    inner_opt = torch.optim.SGD([p for _, p in named_params], lr=args.inner_lr)
    outer_opt = torch.optim.AdamW([p for _, p in named_params], lr=args.outer_lr)
    device = next(model.parameters()).device
    print(f"[TTT-E2E-lite] Running on device: {device}")
    global_step = 0

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
            global_step += 1
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                step_ckpt_dir = os.path.join(args.output_dir, f"step_{global_step:07d}")
                model.save_pretrained(step_ckpt_dir)
                tokenizer.save_pretrained(step_ckpt_dir)
                print(f"[TTT-E2E-lite] Saved step checkpoint: {step_ckpt_dir}")
                if wandb_run is not None and wandb_mod is not None and args.wandb_log_artifacts:
                    artifact = wandb_mod.Artifact(f"ttt-e2e-step-{global_step}", type="model")
                    artifact.add_dir(step_ckpt_dir)
                    wandb_run.log_artifact(artifact)
            if i % args.log_every == 0:
                avg_outer = total_outer / max(1, i)
                cur_outer = float(outer_loss.detach().cpu())
                cur_inner = float(inner_loss.detach().cpu())
                print(
                    f"[TTT-E2E-lite] epoch={epoch} step={i}/{len(seqs)} chunks={len(chunks)} "
                    f"outer_loss={cur_outer:.4f} avg_outer_loss={avg_outer:.4f} inner_loss={cur_inner:.4f}"
                )
                if wandb_run is not None and wandb_mod is not None:
                    wandb_mod.log(
                        {
                            "train/epoch": epoch,
                            "train/step_in_epoch": i,
                            "train/global_step": global_step,
                            "train/chunks_per_seq": len(chunks),
                            "train/inner_loss": cur_inner,
                            "train/outer_loss": cur_outer,
                            "train/avg_outer_loss": avg_outer,
                        },
                        step=global_step,
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
        if wandb_run is not None and wandb_mod is not None:
            wandb_mod.log(
                {
                    "epoch/index": epoch,
                    "epoch/effective_steps": n_effective,
                    "epoch/avg_outer_loss": avg_epoch_outer,
                },
                step=global_step,
            )
            if args.wandb_log_artifacts:
                artifact = wandb_mod.Artifact(f"ttt-e2e-epoch-{epoch}", type="model")
                artifact.add_dir(ckpt_dir)
                wandb_run.log_artifact(artifact)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[TTT-E2E-lite] Saved final adapter: {args.output_dir}")
    if wandb_run is not None and wandb_mod is not None and args.wandb_log_artifacts:
        artifact = wandb_mod.Artifact("ttt-e2e-final", type="model")
        artifact.add_dir(args.output_dir)
        wandb_run.log_artifact(artifact)
    if args.run_predict:
        _run_prediction(args, model, tokenizer, wandb_run=wandb_run, wandb_mod=wandb_mod)
    if wandb_run is not None:
        wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TTT-E2E-lite (first-order) adapter.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-adapter", default=None, help="Optional LoRA adapter checkpoint as init")
    p.add_argument("--predict-only", action="store_true", help="Skip training and run prediction only.")
    p.add_argument("--predict-ckpt", default=None, help="Adapter checkpoint directory to load for predict-only mode.")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=5e-5)
    p.add_argument("--outer-lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--inner-max-tokens", type=int, default=2048)
    p.add_argument("--chunk-tokens", type=int, default=1024, help="Chunk size for inner scan / outer averaging.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every-steps", type=int, default=0, help="Save checkpoint every N training steps (0 disables).")
    p.add_argument("--run-predict", action="store_true", help="Run prediction on test split after training.")
    p.add_argument("--predictions-out", default=None, help="Output file for post-train predictions.")
    p.add_argument("--predict-log-every", type=int, default=20)
    p.add_argument("--predict-max-input-tokens", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--prediction-mode",
        choices=["generate", "label-ranking"],
        default="label-ranking",
        help="Post-train prediction mode: text generation or fast label ranking.",
    )
    p.add_argument("--label-candidates", default="yes,no,insufficient")
    p.add_argument("--label-prefix", default="\nFinal Deterministic Prediction:\n")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--line-profile", action="store_true", help="Enable line_profiler for run().")
    p.add_argument("--line-profile-out", default=None, help="Optional output file for line_profiler stats.")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", default="vcworld-ttt-e2e-lite")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-dir", default=None)
    p.add_argument("--wandb-offline", action="store_true", help="Use offline W&B logging.")
    p.add_argument("--wandb-log-artifacts", action="store_true", help="Upload checkpoints/predictions as artifacts.")
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

    print("[TTT-E2E-lite] line_profiler enabled")
    lp = LineProfiler()
    profiled_run = lp(run)
    profiled_run(args)
    lp.print_stats()
    if args.line_profile_out:
        with open(args.line_profile_out, "w", encoding="utf-8") as f:
            lp.print_stats(stream=f)
        print(f"[TTT-E2E-lite] Saved line_profiler stats to: {args.line_profile_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
