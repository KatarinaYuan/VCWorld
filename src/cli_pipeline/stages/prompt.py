#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate prompts from retrieval results and template."""

import csv
import json
import os
import random
import re
from typing import Dict, List, Tuple, Any, Optional, Set


_DESC_INDEX_CACHE: Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]] = {}
_GENE_ALIAS_INDEX_CACHE: Dict[Tuple[int, int], Dict[str, List[str]]] = {}

_DRUG_SALT_TOKENS = {
    "hydrochloride",
    "dihydrochloride",
    "hydrobromide",
    "maleate",
    "mesylate",
    "tosylate",
    "citrate",
    "phosphate",
    "sulfate",
    "acetate",
    "chloride",
    "bromide",
    "fumarate",
    "tartrate",
    "sodium",
    "potassium",
    "calcium",
    "hydrate",
    "monohydrate",
    "dihydrate",
    "trihydrate",
    "tetrahydrate",
    "hexahydrate",
    "octahydrate",
    "hemihydrate",
    "solvate",
    "salt",
    "erbumine",
    "disodium",
    "olamine",
}


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


def load_optional_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
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


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _generic_norm_key(text: str) -> str:
    s = str(text).strip().lower().replace("–", "-").replace("—", "-")
    return _normalize_spaces(s)


def _drug_norm_key(text: str) -> str:
    # Remove parenthetical formulation/salt information, then normalize.
    s = str(text).strip().replace("–", "-").replace("—", "-")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = _normalize_spaces(s)
    s = re.sub(r"^[^a-zA-Z0-9]+", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+$", "", s)
    return _generic_norm_key(s)


def _strip_drug_formulation_tokens(norm_drug_name: str) -> str:
    if not norm_drug_name:
        return norm_drug_name
    tokens = norm_drug_name.split()
    # Strip trailing formulation tokens such as hydrochloride/mesylate/hydrate.
    while tokens and tokens[-1] in _DRUG_SALT_TOKENS:
        tokens.pop()
    # Strip leading ions in names like "sodium salicylate".
    while len(tokens) > 1 and tokens[0] in {"sodium", "potassium", "calcium"}:
        tokens.pop(0)
    return " ".join(tokens)


def _gene_norm_key(text: str) -> str:
    s = str(text).strip()
    if s.upper().startswith("ENSG"):
        # Drop optional version suffix ENSGxxxx.y -> ENSGxxxx
        s = s.split(".", 1)[0].upper()
    return _generic_norm_key(s)


def _build_desc_index(desc_map: Dict[str, Any], label: str) -> Dict[str, Dict[str, str]]:
    cache_key = (id(desc_map), len(desc_map), label.lower())
    cached = _DESC_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    exact: Dict[str, str] = {}
    casefold: Dict[str, str] = {}
    generic_norm: Dict[str, str] = {}
    drug_norm: Dict[str, str] = {}
    drug_stripped: Dict[str, str] = {}
    gene_norm: Dict[str, str] = {}

    for k, v in desc_map.items():
        key = str(k)
        val = v if isinstance(v, str) else str(v)
        exact[key] = val
        casefold.setdefault(_generic_norm_key(key), val)
        generic_norm.setdefault(_generic_norm_key(key), val)
        if label.lower() == "drug":
            dn = _drug_norm_key(key)
            if dn:
                drug_norm.setdefault(dn, val)
                ds = _strip_drug_formulation_tokens(dn)
                if ds:
                    drug_stripped.setdefault(ds, val)
        if label.lower() == "gene":
            gn = _gene_norm_key(key)
            if gn:
                gene_norm.setdefault(gn, val)

    out = {
        "exact": exact,
        "casefold": casefold,
        "generic_norm": generic_norm,
        "drug_norm": drug_norm,
        "drug_stripped": drug_stripped,
        "gene_norm": gene_norm,
    }
    _DESC_INDEX_CACHE[cache_key] = out
    return out


def _extract_gene_alias_index(gene_alias_map: Dict[str, Any]) -> Dict[str, List[str]]:
    cache_key = (id(gene_alias_map), len(gene_alias_map))
    cached = _GENE_ALIAS_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    idx: Dict[str, Set[str]] = {}

    def add(alias_text: str, canonical_text: str) -> None:
        alias_norm = _gene_norm_key(alias_text)
        canonical_clean = str(canonical_text).strip()
        if not alias_norm or not canonical_clean:
            return
        idx.setdefault(alias_norm, set()).add(canonical_clean)

    for k, v in gene_alias_map.items():
        key = str(k).strip()
        if not key:
            continue

        if isinstance(v, str):
            canonical = v.strip()
            if canonical:
                add(key, canonical)
                add(canonical, canonical)
            continue

        if isinstance(v, list):
            canonical = key
            add(canonical, canonical)
            for alias in v:
                if isinstance(alias, str):
                    add(alias, canonical)
            continue

        if isinstance(v, dict):
            canonical_candidates: List[str] = []
            for field in ("canonical", "symbol", "gene", "name", "entity_id", "ensembl_id", "target"):
                val = v.get(field)
                if isinstance(val, str) and val.strip():
                    canonical_candidates.append(val.strip())
            if not canonical_candidates:
                canonical_candidates.append(key)

            alias_candidates: List[str] = [key]
            for field in ("aliases", "synonyms", "names", "alias"):
                vals = v.get(field)
                if isinstance(vals, str) and vals.strip():
                    alias_candidates.append(vals.strip())
                elif isinstance(vals, list):
                    for a in vals:
                        if isinstance(a, str) and a.strip():
                            alias_candidates.append(a.strip())

            for canonical in canonical_candidates:
                add(canonical, canonical)
                for alias in alias_candidates:
                    add(alias, canonical)
            continue

        # Unknown value type: fallback as identity mapping of key.
        add(key, key)

    out = {k: sorted(v) for k, v in idx.items()}
    _GENE_ALIAS_INDEX_CACHE[cache_key] = out
    return out


def _resolve_description_direct(name: str, desc_map: Dict[str, Any], label: str) -> Tuple[str, bool]:
    index = _build_desc_index(desc_map, label)
    if name in index["exact"]:
        return index["exact"][name], True

    name_case = _generic_norm_key(name)
    if name_case in index["casefold"]:
        return index["casefold"][name_case], True

    if label.lower() == "drug":
        dn = _drug_norm_key(name)
        if dn in index["drug_norm"]:
            return index["drug_norm"][dn], True
        ds = _strip_drug_formulation_tokens(dn)
        if ds in index["drug_stripped"]:
            return index["drug_stripped"][ds], True
    elif label.lower() == "gene":
        gn = _gene_norm_key(name)
        if gn in index["gene_norm"]:
            return index["gene_norm"][gn], True

    if name_case in index["generic_norm"]:
        return index["generic_norm"][name_case], True
    return "", False


def _resolve_description(
    name: str,
    desc_map: Dict[str, Any],
    label: str,
    gene_alias_index: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, bool]:
    desc, hit = _resolve_description_direct(name, desc_map, label)
    if hit:
        return desc, True

    if label.lower() == "gene" and gene_alias_index:
        alias_norm = _gene_norm_key(name)
        for canonical in gene_alias_index.get(alias_norm, []):
            d2, h2 = _resolve_description_direct(canonical, desc_map, label)
            if h2:
                return d2, True

    return f"{label} '{name}' description not found", False


def get_description(
    name: str,
    desc_map: Dict[str, str],
    label: str,
    gene_alias_index: Optional[Dict[str, List[str]]] = None,
) -> str:
    desc, _ = _resolve_description(name, desc_map, label, gene_alias_index=gene_alias_index)
    return desc


def format_observations(
    pairs: List[List[str]],
    drug_desc: Dict[str, str],
    gene_desc: Dict[str, str],
    choices: Optional[List[str]],
    max_examples: int = 10,
    gene_alias_index: Optional[Dict[str, List[str]]] = None,
) -> str:
    if max_examples <= 0:
        return "No similar experimental observations available for context."
    if not pairs:
        return "No similar experimental observations available for context."

    observations = []
    for i, (drug, gene) in enumerate(pairs[:max_examples]):
        ddesc = get_description(drug, drug_desc, "Drug", gene_alias_index=gene_alias_index)
        gdesc = get_description(gene, gene_desc, "Gene", gene_alias_index=gene_alias_index)
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
                     gene_alias_json: Optional[str] = None,
                     max_observation_examples: int = 10,
                     disable_observation_results: bool = False) -> None:
    random.seed(seed)

    retrieval = load_json(retrieval_json)
    drug_desc = load_json(drug_desc_json)
    gene_desc = load_json(gene_desc_json)
    gene_alias_map = load_optional_json(gene_alias_json)
    gene_alias_index = _extract_gene_alias_index(gene_alias_map) if gene_alias_map else None
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

    query_drugs = sorted({str(item.get("test_case", {}).get("drug", "")).strip() for item in cases if item.get("test_case")})
    query_genes = sorted({str(item.get("test_case", {}).get("gene", "")).strip() for item in cases if item.get("test_case")})
    query_drug_hits = sum(1 for d in query_drugs if _resolve_description(d, drug_desc, "Drug", gene_alias_index=gene_alias_index)[1])
    query_gene_hits = sum(1 for g in query_genes if _resolve_description(g, gene_desc, "Gene", gene_alias_index=gene_alias_index)[1])

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
                gene_alias_index=gene_alias_index,
            )

            pert_desc, _ = _resolve_description(drug, drug_desc, "Drug", gene_alias_index=gene_alias_index)
            gene_desc_text, _ = _resolve_description(gene, gene_desc, "Gene", gene_alias_index=gene_alias_index)
            filled = prompt_template.format(
                pert=drug,
                gene=gene,
                pert_desc=pert_desc,
                gene_desc=gene_desc_text,
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
    if query_drugs:
        print(
            f"[Prompt] Query drug description coverage: {query_drug_hits}/{len(query_drugs)} "
            f"({query_drug_hits/max(1,len(query_drugs)):.1%})"
        )
    if query_genes:
        print(
            f"[Prompt] Query gene description coverage: {query_gene_hits}/{len(query_genes)} "
            f"({query_gene_hits/max(1,len(query_genes)):.1%})"
        )
    if gene_alias_index is not None:
        print(f"[Prompt] Gene alias index loaded: {len(gene_alias_index)} normalized aliases")
