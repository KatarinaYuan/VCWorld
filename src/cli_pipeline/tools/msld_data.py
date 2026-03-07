#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data helpers for MSLD support-only adaptation and query-only inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from msld_graph import MSLDGraph
from ttt_common import PromptRecord, load_label_map, load_prompts, render_chat_prompt


@dataclass
class MSLDExample:
    idx: int
    prompt_id: Optional[int]
    pert: str
    gene: str
    split: str
    label: Optional[int]
    cell_id: int
    drug_id: int
    gene_id: int
    system_prompt: str
    user_input: str
    record: PromptRecord

    @property
    def perturbation_key(self) -> str:
        return f"{self.cell_id}::{self.pert.lower()}"


def load_msld_examples(
    *,
    prompts_file: str,
    labels_csv: str,
    graph: MSLDGraph,
    split: str,
    default_cell_id: int = 0,
    require_label: bool = True,
) -> List[MSLDExample]:
    """Load examples from prompts+labels for a specific split."""
    split_norm = split.strip().lower()
    label_map = load_label_map(labels_csv)
    records = load_prompts(prompts_file)
    out: List[MSLDExample] = []

    for i, rec in enumerate(records):
        if not rec.pert or not rec.gene or not rec.system_prompt or not rec.user_input:
            continue
        row = label_map.get((rec.pert, rec.gene))
        if row is None:
            continue
        row_split = str(row.get("split", "")).strip().lower()
        if row_split != split_norm:
            continue

        drug_id = graph.lookup_drug_id(rec.pert)
        gene_id = graph.lookup_gene_id(rec.gene)
        if drug_id is None or gene_id is None:
            continue

        label_val: Optional[int]
        if "label" in row and str(row.get("label", "")).strip() != "":
            try:
                label_val = int(row["label"])
            except (TypeError, ValueError):
                label_val = None
        else:
            label_val = None
        if require_label and label_val is None:
            continue

        out.append(
            MSLDExample(
                idx=i,
                prompt_id=rec.prompt_id,
                pert=str(rec.pert),
                gene=str(rec.gene),
                split=row_split,
                label=label_val,
                cell_id=int(default_cell_id),
                drug_id=int(drug_id),
                gene_id=int(gene_id),
                system_prompt=str(rec.system_prompt),
                user_input=str(rec.user_input),
                record=rec,
            )
        )
    return out


def group_examples_by_perturbation(examples: List[MSLDExample]) -> Dict[str, List[int]]:
    """Group example indices by perturbation key (cell,drug)."""
    groups: Dict[str, List[int]] = {}
    for idx, ex in enumerate(examples):
        groups.setdefault(ex.perturbation_key, []).append(idx)
    return groups


def build_query_prompt_ids(
    *,
    tokenizer,
    example: MSLDExample,
    max_input_tokens: int,
) -> List[int]:
    """Render query prompt and tokenize."""
    prompt = render_chat_prompt(tokenizer, example.system_prompt, example.user_input)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if max_input_tokens > 0:
        ids = ids[-max_input_tokens:]
    return ids


def build_example_lookup(examples: List[MSLDExample]) -> Dict[Tuple[str, str], MSLDExample]:
    """Map (pert,gene) to example object."""
    out: Dict[Tuple[str, str], MSLDExample] = {}
    for ex in examples:
        key = (ex.pert.strip().lower(), ex.gene.strip().upper())
        if key not in out:
            out[key] = ex
    return out
