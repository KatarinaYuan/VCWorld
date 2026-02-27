#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run local vLLM inference on prompts file."""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

PROMPT_SEPARATOR = "================================================================================"


def _parse_prompt_block(block_text: str) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
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


def _generate_with_vllm(
    *,
    llm,
    sampling_params,
    prompt_inputs: Sequence[Any],
    batch_size: int,
) -> List[str]:
    if not prompt_inputs:
        return []

    total_batches = math.ceil(len(prompt_inputs) / batch_size)
    generated: List[str] = []
    for i in range(0, len(prompt_inputs), batch_size):
        batch = list(prompt_inputs[i:i + batch_size])
        outputs = llm.generate(batch, sampling_params=sampling_params, use_tqdm=False)
        # vLLM returns outputs aligned to input order.
        for out in outputs:
            if out.outputs:
                generated.append(out.outputs[0].text)
            else:
                generated.append("")
        print(f"vLLM batch {i//batch_size + 1}/{total_batches} done")
    return generated


def run_inference_vllm(
    *,
    model_name: str,
    prompts_file: str,
    output_file: str,
    batch_size: int = 64,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    chat_template_path: Optional[str] = None,
    sort_by_input_length: bool = True,
    long_context_first: bool = False,
    max_input_tokens: Optional[int] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = False,
) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError(
            "vLLM is not installed in current environment. Install it first to use `infer-vllm`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if chat_template_path:
        with open(chat_template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()

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
        indexed_prompts.sort(key=lambda x: len(x[1]), reverse=long_context_first)

    # Keep a safety margin so generation always has room in the context window.
    # If max_model_len is set, we cap input length to <= max_model_len - max_new_tokens - 1.
    safe_max_input_tokens = max_input_tokens
    if max_model_len is not None:
        safe_cap = max(1, int(max_model_len) - int(max_new_tokens) - 1)
        if safe_max_input_tokens is None:
            safe_max_input_tokens = safe_cap
        else:
            safe_max_input_tokens = min(int(safe_max_input_tokens), safe_cap)

    # Tokenize once and pass token IDs directly to vLLM to avoid re-tokenization drift.
    indexed_prompt_inputs = []
    for idx, prompt in indexed_prompts:
        tok = tokenizer(
            prompt,
            truncation=safe_max_input_tokens is not None,
            max_length=safe_max_input_tokens,
            add_special_tokens=False,
        )["input_ids"]
        indexed_prompt_inputs.append((idx, {"prompt_token_ids": tok}))

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    sorted_prompt_inputs = [inp for _, inp in indexed_prompt_inputs]
    sorted_outputs = _generate_with_vllm(
        llm=llm,
        sampling_params=sampling_params,
        prompt_inputs=sorted_prompt_inputs,
        batch_size=batch_size,
    )

    generated_by_index: List[Optional[str]] = [None] * len(all_messages)
    for (idx, _), text in zip(indexed_prompt_inputs, sorted_outputs):
        generated_by_index[idx] = text

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
