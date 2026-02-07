#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run local HF model inference on prompts file."""

from __future__ import annotations

import math
import re
from typing import List, Optional

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

    user_match = re.search(r"\[Start of Input\](.*?)\[End of Output\]", block_text, re.DOTALL)
    if not user_match:
        return None, None, header, "User input markers not found."
    user_input = user_match.group(0).strip()

    return system_prompt, user_input, header, None


def _dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def run_inference(*, model_name: str, prompts_file: str, output_file: str, batch_size: int = 4,
                  max_new_tokens: int = 512, temperature: float = 0.6, top_p: float = 0.9,
                  dtype: str = "bfloat16", device_map: str = "auto",
                  chat_template_path: Optional[str] = None) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if chat_template_path:
        with open(chat_template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_dtype_from_name(dtype),
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

    total_batches = math.ceil(len(all_messages) / batch_size)
    all_generated = []

    eos_token_id = tokenizer.eos_token_id

    for i in range(0, len(all_messages), batch_size):
        batch = all_messages[i:i + batch_size]
        inputs = tokenizer.apply_chat_template(
            batch,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_generated.extend(responses)
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
            if output_idx < len(all_generated):
                response = all_generated[output_idx]
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
