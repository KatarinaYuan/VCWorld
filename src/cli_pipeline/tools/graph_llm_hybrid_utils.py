#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for graph+LLM-hidden hybrid baseline."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
import torch

from ttt_common import render_chat_prompt


def normalize_key(pert: str, gene: str) -> str:
    return f"{str(pert).strip().lower()}\t{str(gene).strip().upper()}"


def _encode_hidden(model, device: torch.device, prompt_ids: List[int]) -> np.ndarray:
    input_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(
            input_ids=input_t,
            attention_mask=torch.ones_like(input_t),
            output_hidden_states=True,
            return_dict=True,
        )
    if out.hidden_states is None or len(out.hidden_states) == 0:
        raise RuntimeError("Model did not return hidden states")
    h = out.hidden_states[-1][0].mean(dim=0).float().detach().cpu().numpy().astype(np.float32)
    return h


def _load_cache(cache_path: str) -> Dict[str, np.ndarray]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    obj = np.load(cache_path, allow_pickle=True)
    keys = obj["keys"].tolist()
    hidden = obj["hidden"]
    out: Dict[str, np.ndarray] = {}
    for k, h in zip(keys, hidden):
        out[str(k)] = np.asarray(h, dtype=np.float32)
    return out


def _save_cache(cache_path: str, cache: Dict[str, np.ndarray]) -> None:
    if not cache_path:
        return
    out_dir = os.path.dirname(cache_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    keys = sorted(cache.keys())
    hidden = np.asarray([cache[k] for k in keys], dtype=np.float32)
    np.savez_compressed(cache_path, keys=np.asarray(keys, dtype=object), hidden=hidden)


def ensure_hidden_cache(
    *,
    examples: Sequence,
    model_name: str,
    max_input_tokens: int,
    cache_path: str,
    bf16: bool,
    trust_remote_code: bool,
    log_every: int = 200,
) -> Dict[str, np.ndarray]:
    """Return map key=(pert,gene)->hidden. Compute missing and update cache."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for LLM hidden precompute. Please install `transformers` in this env."
        ) from exc

    cache = _load_cache(cache_path=cache_path)
    need_keys = {normalize_key(ex.pert, ex.gene) for ex in examples}
    missing_keys = [k for k in sorted(need_keys) if k not in cache]
    if not missing_keys:
        print(f"[GraphLLM] Hidden cache hit: {len(need_keys)} examples from {cache_path}")
        return cache

    print(f"[GraphLLM] Hidden cache missing {len(missing_keys)} examples. Computing...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device

    missing_set = set(missing_keys)
    count = 0
    for ex in examples:
        key = normalize_key(ex.pert, ex.gene)
        if key not in missing_set:
            continue
        prompt = render_chat_prompt(tokenizer, ex.system_prompt, ex.user_input)
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if max_input_tokens > 0:
            ids = ids[-max_input_tokens:]
        cache[key] = _encode_hidden(model=model, device=device, prompt_ids=ids)
        count += 1
        if count % max(1, int(log_every)) == 0 or count == len(missing_keys):
            print(f"[GraphLLM] Hidden precompute {count}/{len(missing_keys)}")

    _save_cache(cache_path=cache_path, cache=cache)
    print(f"[GraphLLM] Hidden cache saved: {cache_path}")
    return cache


def gather_hidden_matrix(keys: Sequence[str], cache: Dict[str, np.ndarray], hidden_dim: int) -> np.ndarray:
    mats: List[np.ndarray] = []
    for k in keys:
        h = cache.get(k)
        if h is None:
            mats.append(np.zeros((hidden_dim,), dtype=np.float32))
            continue
        mats.append(np.asarray(h, dtype=np.float32))
    return np.asarray(mats, dtype=np.float32)
