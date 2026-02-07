#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a single-case prompt for out-of-dataset samples."""

from __future__ import annotations

import json
import os
import random
import sys
import difflib
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from ..prompt import (
    load_template_vars,
    get_description,
    _default_template_path,
)


def _load_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_similarity_json(path: str) -> Dict[str, List[str]]:
    """Use the same parsing logic as retrieve stage (lightweight copy)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, List[str]] = {}
    for key, vals in raw.items():
        if not vals:
            out[key] = []
            continue
        if isinstance(vals, dict):
            for field in (
                "direct_neighbors",
                "neighbors",
                "close_genes",
                "similar_genes",
                "top_genes",
                "two_hop_neighbors",
            ):
                candidates = vals.get(field)
                if isinstance(candidates, list):
                    out[key] = [str(v) for v in candidates]
                    break
            else:
                out[key] = [str(v) for v in vals.values()]
            continue
        if isinstance(vals, list):
            if not vals:
                out[key] = []
            elif isinstance(vals[0], dict):
                if "Drug" in vals[0]:
                    out[key] = [v.get("Drug") for v in vals if v.get("Drug")]
                elif "Gene" in vals[0]:
                    out[key] = [v.get("Gene") for v in vals if v.get("Gene")]
                else:
                    out[key] = [str(v) for v in vals]
            else:
                out[key] = [str(v) for v in vals]
            continue
        out[key] = [str(vals)]
    return out


def _load_csv_pairs(path: str) -> List[Tuple[str, str, int, str]]:
    pairs: List[Tuple[str, str, int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx_pert = header.index("pert")
        idx_gene = header.index("gene")
        idx_label = header.index("label")
        idx_split = header.index("split")
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split(",")
            if len(cols) <= max(idx_pert, idx_gene, idx_label, idx_split):
                continue
            pairs.append((cols[idx_pert], cols[idx_gene], int(cols[idx_label]), cols[idx_split]))
    return pairs


def _casefold_map(keys: List[str]) -> Dict[str, str]:
    return {k.strip().lower(): k for k in keys}


def _resolve_name(name: str, canon_map: Dict[str, str]) -> Optional[str]:
    key = name.strip().lower()
    return canon_map.get(key)

def _normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _normalize_map(keys: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in keys:
        norm = _normalize_key(key)
        if norm and norm not in out:
            out[norm] = key
    return out


def _name_similarity(a: str, b: str) -> float:
    na = _normalize_key(a)
    nb = _normalize_key(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _score_similarity(query_desc: str, cand_desc: str) -> float:
    """Simple token-overlap score (no external deps)."""
    q = set(query_desc.lower().split())
    c = set(cand_desc.lower().split())
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def _pick_similar_by_heuristic(name: str, desc_map: Dict[str, str], top_k: int) -> List[str]:
    """Heuristic fallback when similarity JSON lacks the query."""
    # If we can get a description for the query from desc_map, use it; else
    # use the name itself for token overlap.
    query_desc = desc_map.get(name, "")
    scored = []
    for cand, desc in desc_map.items():
        name_sim = _name_similarity(name, cand)
        desc_sim = _score_similarity(query_desc, desc) if query_desc else 0.0
        score = 0.7 * name_sim + 0.3 * desc_sim
        scored.append((cand, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cand for cand, _ in scored[:top_k]]


def _resolve_api_key(cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key
    env_key = os.getenv("LLM_DRUG_API_KEY") or os.getenv("API_KEY")
    if env_key:
        return env_key
    return ""


def _post_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"API HTTPError: {e.code} {detail}") from e
    except URLError as e:
        raise RuntimeError(f"API URLError: {e.reason}") from e


def _llm_rank_candidates(
    *,
    query_name: str,
    query_desc: str,
    candidates: List[Tuple[str, str]],
    top_k: int,
    api_url: str,
    api_model: str,
    api_key: str,
    timeout: int,
    task: str,
) -> List[str]:
    if not candidates:
        return []
    cand_lines = []
    for name, desc in candidates:
        desc_short = desc.replace("\n", " ").strip()
        if len(desc_short) > 240:
            desc_short = desc_short[:240] + "..."
        cand_lines.append(f"- {name}: {desc_short}")

    if task == "drug":
        goal = "mechanism of action and targets"
    else:
        goal = "function and pathway relevance"

    system_prompt = (
        "You are a biomedical expert. Return only a JSON array of names. "
        "Do not include explanations."
    )
    user_prompt = (
        f"Query: {query_name}\n"
        f"Description: {query_desc}\n\n"
        f"Select the {top_k} most similar candidates by {goal}. "
        "Return a JSON array of candidate names, ordered most similar first.\n\n"
        "Candidates:\n" + "\n".join(cand_lines)
    )

    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
    }
    try:
        resp = _post_json(api_url, payload, api_key, timeout)
    except RuntimeError as exc:
        # Some APIs/models require max_completion_tokens instead of max_tokens.
        err = str(exc)
        if "max_tokens" in err and "max_completion_tokens" in err:
            payload.pop("max_tokens", None)
            payload["max_completion_tokens"] = 512
            resp = _post_json(api_url, payload, api_key, timeout)
        else:
            raise
    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        content = json.dumps(resp, ensure_ascii=False)

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(x) for x in parsed][:top_k]
    except Exception:
        pass
    return []


def _find_similar_drugs(
    pert: str,
    drug_desc: Dict[str, str],
    drug_sim_json: Dict[str, List[str]],
    max_candidates: int,
    llm_api_url: Optional[str],
    llm_api_model: Optional[str],
    llm_api_key: Optional[str],
    llm_candidate_pool: int,
    llm_timeout: int,
) -> List[str]:
    # Prefer similarity JSON if possible (case-insensitive)
    query_name = pert
    canon = _casefold_map(list(drug_sim_json.keys()))
    key = _resolve_name(query_name, canon)
    if not key:
        norm_map = _normalize_map(list(drug_sim_json.keys()))
        key = norm_map.get(_normalize_key(query_name))
    if key:
        sims = [s for s in drug_sim_json.get(key, []) if s][:max_candidates]
        if pert not in sims:
            sims.insert(0, pert)
        return sims[:max_candidates]

    if llm_api_url and llm_api_model:
        api_key = _resolve_api_key(llm_api_key)
        if not api_key:
            api_key = ""
        query_desc = drug_desc.get(query_name, query_name)
        scored = []
        for cand, desc in drug_desc.items():
            scored.append((cand, _score_similarity(query_desc, desc)))
        scored.sort(key=lambda x: x[1], reverse=True)
        pool = scored[:llm_candidate_pool]
        candidates = [(name, drug_desc.get(name, "")) for name, _ in pool]
        print(
            f"[single-case] LLM fallback for drug '{pert}' (pool={len(candidates)})",
            file=sys.stderr,
        )
        ranked = _llm_rank_candidates(
            query_name=pert,
            query_desc=query_desc,
            candidates=candidates,
            top_k=max_candidates,
            api_url=llm_api_url,
            api_model=llm_api_model,
            api_key=api_key,
            timeout=llm_timeout,
            task="drug",
        )
        if ranked:
            if pert not in ranked:
                ranked.insert(0, pert)
            return ranked[:max_candidates]

    sims = _pick_similar_by_heuristic(pert, drug_desc, max_candidates)
    if pert not in sims:
        sims.insert(0, pert)
    return sims[:max_candidates]


def _find_similar_genes(
    gene: str,
    gene_desc: Dict[str, str],
    gene_sim_json: Dict[str, List[str]],
    max_candidates: int,
    llm_api_url: Optional[str],
    llm_api_model: Optional[str],
    llm_api_key: Optional[str],
    llm_candidate_pool: int,
    llm_timeout: int,
) -> List[str]:
    gene_aliases = {
        "vegfr-1": "FLT1",
        "vegfr1": "FLT1",
        "vegfr-2": "KDR",
        "vegfr2": "KDR",
        "alk3": "ACVR1",
        "bmp-2": "BMP2",
        "bmp2": "BMP2",
    }
    alias_key = gene.strip().lower()
    if alias_key in gene_aliases:
        gene_query = gene_aliases[alias_key]
    else:
        gene_query = gene
    canon = _casefold_map(list(gene_sim_json.keys()))
    key = _resolve_name(gene_query, canon)
    if not key:
        norm_map = _normalize_map(list(gene_sim_json.keys()))
        key = norm_map.get(_normalize_key(gene_query))
    if key:
        sims = [s for s in gene_sim_json.get(key, []) if s][:max_candidates]
        if gene not in sims:
            sims.insert(0, gene)
        return sims[:max_candidates]

    if llm_api_url and llm_api_model:
        api_key = _resolve_api_key(llm_api_key)
        if not api_key:
            api_key = ""
        query_desc = gene_desc.get(gene_query, gene_query)
        scored = []
        for cand in gene_sim_json.keys():
            scored.append((cand, _score_similarity(query_desc, gene_desc.get(cand, cand))))
        scored.sort(key=lambda x: x[1], reverse=True)
        pool = scored[:llm_candidate_pool]
        candidates = [(name, gene_desc.get(name, "")) for name, _ in pool]
        print(
            f"[single-case] LLM fallback for gene '{gene}' (pool={len(candidates)})",
            file=sys.stderr,
        )
        ranked = _llm_rank_candidates(
            query_name=gene,
            query_desc=query_desc,
            candidates=candidates,
            top_k=max_candidates,
            api_url=llm_api_url,
            api_model=llm_api_model,
            api_key=api_key,
            timeout=llm_timeout,
            task="gene",
        )
        if ranked:
            if gene not in ranked:
                ranked.insert(0, gene)
            return ranked[:max_candidates]

    sims = _pick_similar_by_heuristic(gene, gene_desc, max_candidates)
    if gene not in sims:
        sims.insert(0, gene)
    return sims[:max_candidates]


def _collect_retrieved_pairs(
    data_csv: str,
    close_drugs: List[str],
    close_genes: List[str],
    budget: int,
    case_split: str,
    seed: int,
) -> List[Tuple[str, str, int]]:
    rows = _load_csv_pairs(data_csv)
    pairs: List[Tuple[str, str, int]] = []
    close_drug_set = set([d.strip().lower() for d in close_drugs])
    close_gene_set = set([g.strip().lower() for g in close_genes])
    for pert, gene, _label, split in rows:
        if case_split and split != case_split:
            continue
        if pert.strip().lower() in close_drug_set or gene.strip().lower() in close_gene_set:
            pairs.append((pert, gene, _label))
    if len(pairs) > budget:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, budget)
    return pairs


def generate_single_case_prompt(
    *,
    task: str,
    pert: str,
    gene: str,
    cell_line: str,
    data_csv: str,
    drug_desc_json: str,
    gene_desc_json: str,
    drug_sim_json: str,
    gene_sim_json: str,
    output_file: str,
    template_file: Optional[str] = None,
    max_candidates: int = 10,
    budget: int = 10,
    case_split: str = "",
    seed: int = 42,
    llm_api_url: Optional[str] = None,
    llm_api_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_candidate_pool: int = 80,
    llm_timeout: int = 60,
) -> None:
    random.seed(seed)

    drug_desc = _load_json(drug_desc_json)
    gene_desc = _load_json(gene_desc_json)
    drug_sim = _load_similarity_json(drug_sim_json)
    gene_sim = _load_similarity_json(gene_sim_json)

    close_drugs = _find_similar_drugs(
        pert,
        drug_desc,
        drug_sim,
        max_candidates,
        llm_api_url,
        llm_api_model,
        llm_api_key,
        llm_candidate_pool,
        llm_timeout,
    )
    close_genes = _find_similar_genes(
        gene,
        gene_desc,
        gene_sim,
        max_candidates,
        llm_api_url,
        llm_api_model,
        llm_api_key,
        llm_candidate_pool,
        llm_timeout,
    )

    retrieved_pairs = _collect_retrieved_pairs(
        data_csv=data_csv,
        close_drugs=close_drugs,
        close_genes=close_genes,
        budget=budget,
        case_split=case_split,
        seed=seed,
    )

    if template_file is None:
        template_file = _default_template_path(task)
    tmpl_vars = load_template_vars(template_file)

    cell_lines: List[Tuple[str, str]] = tmpl_vars.get("cell_lines", [])
    if not cell_lines:
        raise RuntimeError("cell_lines not found in template file")

    # find cell line description
    cell_desc = None
    for name, desc in cell_lines:
        if name.strip().lower() == cell_line.strip().lower():
            cell_desc = desc
            cell_short = name
            break
    else:
        # default to first if not found
        cell_short, cell_desc = cell_lines[0]

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

    if not retrieved_pairs:
        obs = "No similar experimental observations available for context."
    else:
        observations = []
        for i, (drug, gene_name, label) in enumerate(retrieved_pairs[:budget]):
            ddesc = get_description(drug, drug_desc, "Drug")
            gdesc = get_description(gene_name, gene_desc, "Gene")
            obs_text = (
                f"Example {i+1}:\n"
                f"- Drug: {drug}\n"
                f"- Gene: {gene_name}\n"
                f"- Drug Description: {ddesc}\n"
                f"- Gene Description: {gdesc}"
            )
            if choices:
                idx = 0 if int(label) <= 0 else 1
                if idx < len(choices):
                    obs_text += f"\n- Result: {choices[idx]}"
            observations.append(obs_text)
        obs = "\n\n".join(observations)

    filled = prompt_template.format(
        pert=pert,
        gene=gene,
        pert_desc=get_description(pert, drug_desc, "Drug"),
        gene_desc=get_description(gene, gene_desc, "Gene"),
        cell_short=cell_short,
        cell_desc=cell_desc,
        obs=obs,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== Prompt 1 ({pert} | {gene}) ===\n")
        f.write(filled)
        f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"Saved single-case prompt: {output_file}")
    if not retrieved_pairs:
        print("Warning: no retrieved pairs found; prompt will have empty context.")
