#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run local HF model inference on prompts file."""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_SEPARATOR = "================================================================================"


def _parse_prompt_block(block_text: str):
    header_match = re.search(r"===\s*(Prompt\s*\d+).*?===", block_text)
    header = header_match.group(1).strip() if header_match else "Unknown Prompt"

    system_match = re.search(r"\[Start of Prompt\](.*?)\[End of Prompt\]", block_text, re.DOTALL)
    if not system_match:
        return None, None, header, "System prompt markers not found."
    system_prompt = system_match.group(1).strip()

    user_match = re.search(r"\[Start of Input\](.*?)\[End of Input\]", block_text, re.DOTALL)
    if not user_match:
        return None, None, header, "User input markers not found."
    user_input = user_match.group(1).strip()

    return system_prompt, user_input, header, None


def _dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def _generate_texts_with_retry(
    *,
    model,
    tokenizer,
    texts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_input_tokens: Optional[int],
) -> List[str]:
    """Generate a batch and automatically shrink micro-batch size on CUDA OOM."""
    if not texts:
        return []

    micro_batch_size = len(texts)
    while True:
        try:
            results: List[str] = []
            for start in range(0, len(texts), micro_batch_size):
                chunk = texts[start:start + micro_batch_size]
                inputs = tokenizer(
                    list(chunk),
                    padding=True,
                    truncation=max_input_tokens is not None,
                    max_length=max_input_tokens,
                    return_tensors="pt",
                ).to(model.device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
                responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                results.extend(responses)
            return results
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if micro_batch_size == 1:
                raise
            micro_batch_size = max(1, micro_batch_size // 2)
            print(f"CUDA OOM detected. Retrying with micro-batch-size={micro_batch_size}")


def run_inference(*, model_name: str, prompts_file: str, output_file: str, batch_size: int = 4,
                  max_new_tokens: int = 512, temperature: float = 0.6, top_p: float = 0.9,
                  dtype: str = "bfloat16", device_map: str = "auto",
                  chat_template_path: Optional[str] = None,
                  max_input_tokens: Optional[int] = None,
                  sort_by_input_length: bool = True) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if chat_template_path:
        with open(chat_template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_dtype_from_name(dtype),
        device_map=device_map,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with open(prompts_file, "r", encoding="utf-8") as f:
        full_content = f.read()
    prompt_blocks = [b.strip() for b in full_content.split(PROMPT_SEPARATOR) if b.strip()]

    all_messages: List[list] = []
    prompt_metadata = []
    for block in prompt_blocks:
        system_prompt, user_input, header, error = _parse_prompt_block(block)
        if error:
            prompt_metadata.append({"header": header, "is_error": True, "error_message": error})
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        all_messages.append(messages)
        prompt_metadata.append({"header": header, "is_error": False})

    if not all_messages:
        print("No valid prompts to run")
        return

    rendered_prompts = tokenizer.apply_chat_template(
        all_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(rendered_prompts, str):
        rendered_prompts = [rendered_prompts]

    indexed_prompts = list(enumerate(rendered_prompts))
    if sort_by_input_length:
        indexed_prompts.sort(key=lambda x: len(x[1]))

    total_batches = math.ceil(len(indexed_prompts) / batch_size)
    generated_by_index: List[Optional[str]] = [None] * len(all_messages)

    for i in range(0, len(indexed_prompts), batch_size):
        batch_indexed = indexed_prompts[i:i + batch_size]
        batch_indices = [idx for idx, _ in batch_indexed]
        batch_texts = [txt for _, txt in batch_indexed]
        responses = _generate_texts_with_retry(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_input_tokens=max_input_tokens,
        )
        for idx, resp in zip(batch_indices, responses):
            generated_by_index[idx] = resp
        print(f"Batch {i//batch_size + 1}/{total_batches} done")

    all_results = []
    output_idx = 0
    for meta in prompt_metadata:
        header = meta["header"]
        if meta["is_error"]:
            formatted = (
                f"--- Query for {header} ---\n"
                f"ERROR during parsing: {meta['error_message']}\n"
                f"--- End of Query for {header} ---\n\n"
                f"{PROMPT_SEPARATOR}\n\n"
            )
        else:
            response = generated_by_index[output_idx] if output_idx < len(generated_by_index) else None
            if response is not None:
                formatted = (
                    f"--- Query for {header} ---\n"
                    f"{response.strip()}\n"
                    f"--- End of Query for {header} ---\n\n"
                    f"{PROMPT_SEPARATOR}\n\n"
                )
                output_idx += 1
            else:
                formatted = (
                    f"--- Query for {header} ---\n"
                    "ERROR: No output generated for this prompt.\n"
                    f"--- End of Query for {header} ---\n\n"
                    f"{PROMPT_SEPARATOR}\n\n"
                )
        all_results.append(formatted)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(all_results)

    print(f"Saved outputs: {output_file}")
