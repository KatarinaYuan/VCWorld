#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for TTT scripts."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

PROMPT_SEPARATOR = "================================================================================"


@dataclass
class PromptRecord:
    idx: int
    prompt_id: Optional[int]
    pert: Optional[str]
    gene: Optional[str]
    system_prompt: Optional[str]
    user_input: Optional[str]


def split_prompt_blocks(text: str) -> List[str]:
    return [b.strip() for b in text.split(PROMPT_SEPARATOR) if b.strip()]


def parse_prompt_block(block_text: str) -> PromptRecord:
    header_match = re.search(r"===\s*Prompt\s*(\d+)\s*\((.*?)\)\s*===", block_text)
    prompt_id = int(header_match.group(1)) if header_match else None
    pair = header_match.group(2).strip() if header_match else ""
    pert, gene = None, None
    if "|" in pair:
        pert, gene = [x.strip() for x in pair.split("|", 1)]

    system_match = re.search(r"\[Start of Prompt\](.*?)\[End of Prompt\]", block_text, re.DOTALL)
    user_match = re.search(r"\[Start of Input\](.*?)\[End of Input\]", block_text, re.DOTALL)
    system_prompt = system_match.group(1).strip() if system_match else None
    user_input = user_match.group(1).strip() if user_match else None

    return PromptRecord(
        idx=-1,
        prompt_id=prompt_id,
        pert=pert,
        gene=gene,
        system_prompt=system_prompt,
        user_input=user_input,
    )


def load_prompts(prompts_file: str) -> List[PromptRecord]:
    with open(prompts_file, "r", encoding="utf-8") as f:
        blocks = split_prompt_blocks(f.read())
    records: List[PromptRecord] = []
    for i, block in enumerate(blocks):
        rec = parse_prompt_block(block)
        rec.idx = i
        records.append(rec)
    return records


def load_label_map(labels_csv: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(labels_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["pert"].strip(), row["gene"].strip())
            out[key] = row
    return out


def render_chat_prompt(tokenizer, system_prompt: str, user_input: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if isinstance(rendered, list):
        return rendered[0]
    return rendered


def format_prediction_block(header: str, response: str) -> str:
    return (
        f"--- Query for {header} ---\n"
        f"{response.strip()}\n"
        f"--- End of Query for {header} ---\n\n"
        f"{PROMPT_SEPARATOR}\n\n"
    )

