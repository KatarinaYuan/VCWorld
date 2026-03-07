#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate prompts from retrieval results and template."""

import csv
import json
import os
import random
import re
from typing import Dict, List, Tuple, Any, Optional


def load_template_vars(template_file: str) -> Dict[str, Any]:
    with open(template_file, "r", encoding="utf-8") as f:
        content = f.read()
    exec_globals: Dict[str, Any] = {
        # Provide defaults so templates using f-strings without defining these won't error.
        "desc_pert": "description of drug that is to perturb the cell",
        "desc_gene": "description of gene, the impact on which you wish to infer",
        "desc_context": "description of cell line in which the genes are expressed",
        "desc_obs": (
            "set of experimental observations that describe the impact of small molecule perturbations "
            "on related genes, to contextualize your answer"
        ),
    }
    exec_locals: Dict[str, Any] = {}
    exec(content, exec_globals, exec_locals)
    return exec_locals


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_lookup(labels_csv: str) -> Dict[Tuple[str, str], int]:
    label_map: Dict[Tuple[str, str], int] = {}
    with open(labels_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["pert"].strip(), row["gene"].strip())
            label_map[key] = int(row["label"])
    return label_map


def get_description(name: str, desc_map: Dict[str, str], label: str) -> str:
    if name in desc_map:
        return desc_map[name]
    clean = name.strip().lower()
    for key, val in desc_map.items():
        if clean == key.strip().lower():
            return val
    return f"{label} '{name}' description not found"


def format_observations(pairs: List[List[str]], drug_desc: Dict[str, str], gene_desc: Dict[str, str],
                        choices: Optional[List[str]], max_examples: int = 10) -> str:
    if max_examples <= 0:
        return "No similar experimental observations available for context."
    if not pairs:
        return "No similar experimental observations available for context."

    observations = []
    for i, (drug, gene) in enumerate(pairs[:max_examples]):
        ddesc = get_description(drug, drug_desc, "Drug")
        gdesc = get_description(gene, gene_desc, "Gene")
        obs_text = (
            f"Example {i+1}:\n"
            f"- Drug: {drug}\n"
            f"- Gene: {gene}\n"
            f"- Drug Description: {ddesc}\n"
            f"- Gene Description: {gdesc}"
        )
        if choices:
            answer = random.choice(choices)
            obs_text += f"\n- Result: {answer}"
        observations.append(obs_text)
    return "\n\n".join(observations)


def _default_template_path(task: str) -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if task == "de":
        return os.path.join(base_dir, "support", "DE_template.py")
    return os.path.join(base_dir, "support", "DIR_template.py")


def _label_answer_text(task: str, label: int, drug: str, gene: str) -> str:
    if task == "de":
        if label == 1:
            return f"- Yes. Perturbation of {drug} results in differential expression of {gene}."
        return f"- No. Perturbation of {drug} does not impact {gene}."
    if label == 1:
        return f"- Increase. Perturbation of {drug} increases expression of {gene}."
    return f"- Decrease. Perturbation of {drug} decreases expression of {gene}."


def _inject_output_answer(prompt_text: str, answer_line: str) -> str:
    pattern = r"(\[Start of Output\]\s*)(.*?)(\s*\[End of Output\])"
    repl = r"\1" + answer_line + r"\n\3"
    updated, n = re.subn(pattern, repl, prompt_text, flags=re.DOTALL)
    if n > 0:
        return updated
    return f"{prompt_text}\n\n[Start of Output]\n{answer_line}\n[End of Output]"


def generate_prompts(*, task: str, retrieval_json: str, drug_desc_json: str, gene_desc_json: str,
                     template_file: Optional[str], output_file: str,
                     cell_line_idx: Optional[int] = None, max_cases: Optional[int] = None,
                     seed: int = 42, include_gold_label: bool = False,
                     labels_csv: Optional[str] = None,
                     max_observation_examples: int = 10,
                     disable_observation_results: bool = False) -> None:
    random.seed(seed)

    retrieval = load_json(retrieval_json)
    drug_desc = load_json(drug_desc_json)
    gene_desc = load_json(gene_desc_json)
    if template_file is None:
        template_file = _default_template_path(task)
    tmpl_vars = load_template_vars(template_file)

    cell_lines: List[Tuple[str, str]] = tmpl_vars.get("cell_lines", [])
    if not cell_lines:
        raise RuntimeError("cell_lines not found in template file")

    choices_de = tmpl_vars.get("choices_de", [])
    choices_dir = tmpl_vars.get("choices_dir", [])
    prompt_de = tmpl_vars.get("prompt_vcworld_DE", "") or tmpl_vars.get("prompt_test_de", "")
    prompt_dir = tmpl_vars.get("prompt_vcworld_DIR", "") or tmpl_vars.get("prompt_test_dir", "")

    if task == "de":
        prompt_template = prompt_de
        choices = choices_de
    else:
        prompt_template = prompt_dir
        choices = choices_dir

    if not prompt_template:
        raise RuntimeError(f"prompt template for {task} not found in template file")

    label_lookup: Optional[Dict[Tuple[str, str], int]] = None
    if include_gold_label and labels_csv:
        label_lookup = load_label_lookup(labels_csv)

    cases = retrieval
    if max_cases is not None and max_cases < len(cases):
        cases = cases[:max_cases]

    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(cases):
            drug = item["test_case"]["drug"].strip()
            gene = item["test_case"]["gene"].strip()
            retrieved_pairs = item.get("retrieved_pairs", [])

            if cell_line_idx is None:
                idx = random.randint(0, len(cell_lines) - 1)
            else:
                idx = cell_line_idx
            cell_short, cell_desc = cell_lines[idx]

            obs_choices = None if disable_observation_results else choices
            obs = format_observations(
                retrieved_pairs,
                drug_desc,
                gene_desc,
                obs_choices,
                max_examples=max_observation_examples,
            )

            filled = prompt_template.format(
                pert=drug,
                gene=gene,
                pert_desc=get_description(drug, drug_desc, "Drug"),
                gene_desc=get_description(gene, gene_desc, "Gene"),
                cell_short=cell_short,
                cell_desc=cell_desc,
                obs=obs,
            )
            if include_gold_label:
                label = item.get("test_case", {}).get("label")
                if label is None and label_lookup is not None:
                    label = label_lookup.get((drug, gene))
                if label is None:
                    raise RuntimeError(
                        f"include_gold_label=True but label not found for ({drug}, {gene}). "
                        "Pass labels_csv or regenerate retrieval with label field."
                    )
                answer_line = _label_answer_text(task, int(label), drug, gene)
                filled = _inject_output_answer(filled, answer_line)

            f.write(f"=== Prompt {i+1} ({drug} | {gene}) ===\n")
            f.write(filled)
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"Saved prompts: {output_file} (count: {len(cases)})")
