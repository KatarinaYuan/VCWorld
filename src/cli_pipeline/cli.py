#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry for DE/DIR pipeline stages."""

import argparse
import sys

from stages.prepare import process_cell_line
from stages.retrieve import build_retrieval_results
from stages.prompt import generate_prompts
from stages.infer import run_inference
from stages.infer_api import run_inference_api
from stages.single_case.prompt import generate_single_case_prompt


def _add_prepare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--h5ad", required=True, help="Input .h5ad file")
    p.add_argument("--out-dir", required=True, help="Output directory for CSVs")
    p.add_argument("--cell-line", required=True, help="Cell line name for output file prefix")
    p.add_argument("--perturbation-col", default="drug", help="AnnData obs column for perturbation")
    p.add_argument("--control-value", default="DMSO_TF", help="Control group name")
    p.add_argument("--train-fraction", type=float, default=0.3, help="Train split fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--lfc", type=float, default=0.25)
    p.add_argument("--pval-neg", type=float, default=0.1)
    p.add_argument("--n-neg", type=int, default=200)


def _add_retrieve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-csv", required=True, help="Input CSV with pert/gene/label/split")
    p.add_argument("--drug-sim", required=True, help="Drug similarity JSON")
    p.add_argument("--gene-sim", required=True, help="Gene similarity JSON (required)")
    p.add_argument("--out", required=True, help="Output retrieval JSON")
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--case-split", default="test", choices=["train", "test"], help="Which split to generate retrieval for")


def _add_prompt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--retrieval", required=True, help="Retrieval JSON from previous stage")
    p.add_argument("--drug-desc", required=True, help="Drug description JSON")
    p.add_argument("--gene-desc", required=True, help="Gene description JSON")
    p.add_argument("--template", default=None, help="Prompt template file (optional)")
    p.add_argument("--out", required=True, help="Output prompts text file")
    p.add_argument("--cell-line-idx", type=int, default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)


def _add_infer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF model path or name")
    p.add_argument("--prompts", required=True, help="Prompt text file")
    p.add_argument("--out", required=True, help="Output text file")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--chat-template", default=None, help="Optional chat template file to override tokenizer.chat_template")
    p.add_argument("--max-input-tokens", type=int, default=None,
                   help="Optional max input token length. Long prompts will be truncated.")
    p.add_argument("--disable-length-sort", action="store_true",
                   help="Disable sorting prompts by input length before batching.")


def _add_infer_api_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-url", required=True, help="API endpoint URL")
    p.add_argument("--api-model", required=True, help="API model name")
    p.add_argument("--api-key", default=None, help="API key (or set LLM_DRUG_API_KEY)")
    p.add_argument("--prompts", required=True, help="Prompt text file")
    p.add_argument("--out", required=True, help="Output text file")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--sleep-secs", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM Drug pipeline (DE/DIR) CLI")
    task_sub = parser.add_subparsers(dest="task", required=True)

    for task_name in ("de", "dir"):
        task_parser = task_sub.add_parser(task_name)
        stage_sub = task_parser.add_subparsers(dest="stage", required=True)

        prepare_p = stage_sub.add_parser("prepare")
        _add_prepare_args(prepare_p)

        retrieve_p = stage_sub.add_parser("retrieve")
        _add_retrieve_args(retrieve_p)

        prompt_p = stage_sub.add_parser("prompt")
        _add_prompt_args(prompt_p)

        infer_p = stage_sub.add_parser("infer")
        _add_infer_args(infer_p)

        infer_api_p = stage_sub.add_parser("infer-api")
        _add_infer_api_args(infer_api_p)

    single_p = task_sub.add_parser("single")
    single_stage = single_p.add_subparsers(dest="stage", required=True)
    single_prompt = single_stage.add_parser("prompt")
    single_prompt.add_argument("--pert", required=True, help="Drug/perturbation name")
    single_prompt.add_argument("--gene", required=True, help="Gene name")
    single_prompt.add_argument("--cell-line", required=True, help="Cell line name (must exist in template)")
    single_prompt.add_argument("--mode", default="de", choices=["de", "dir"], help="Prompt type")
    single_prompt.add_argument("--data-csv", required=True, help="CSV with pert/gene/label/split for retrieval")
    single_prompt.add_argument("--drug-desc", required=True, help="Drug description JSON")
    single_prompt.add_argument("--gene-desc", required=True, help="Gene description JSON")
    single_prompt.add_argument("--drug-sim", required=True, help="Drug similarity JSON")
    single_prompt.add_argument("--gene-sim", required=True, help="Gene similarity JSON")
    single_prompt.add_argument("--template", default=None, help="Prompt template file (optional)")
    single_prompt.add_argument("--out", required=True, help="Output prompt text file")
    single_prompt.add_argument("--max-candidates", type=int, default=10)
    single_prompt.add_argument("--budget", type=int, default=10)
    single_prompt.add_argument("--case-split", default="train", choices=["train", "test", "all"])
    single_prompt.add_argument("--seed", type=int, default=42)
    single_prompt.add_argument("--llm-api-url", default=None, help="LLM API endpoint for fallback similarity")
    single_prompt.add_argument("--llm-api-model", default=None, help="LLM model name for fallback similarity")
    single_prompt.add_argument("--llm-api-key", default=None, help="API key (or set LLM_DRUG_API_KEY)")
    single_prompt.add_argument("--llm-candidate-pool", type=int, default=80)
    single_prompt.add_argument("--llm-timeout", type=int, default=60)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.task in ("de", "dir") and args.stage == "prepare":
        process_cell_line(
            adata_path=args.h5ad,
            output_dir=args.out_dir,
            cell_line_name=args.cell_line,
            perturbation_col=args.perturbation_col,
            control_value=args.control_value,
            train_fraction=args.train_fraction,
            seed=args.seed,
            fdr=args.fdr,
            lfc=args.lfc,
            pval_neg=args.pval_neg,
            n_neg=args.n_neg,
        )
        return 0

    if args.task in ("de", "dir") and args.stage == "retrieve":
        build_retrieval_results(
            data_csv=args.data_csv,
            drug_sim_json=args.drug_sim,
            out_json=args.out,
            gene_sim_json=args.gene_sim,
            budget=args.budget,
            seed=args.seed,
            max_cases=args.max_cases,
            case_split=args.case_split,
        )
        return 0

    if args.task in ("de", "dir") and args.stage == "prompt":
        generate_prompts(
            task=args.task,
            retrieval_json=args.retrieval,
            drug_desc_json=args.drug_desc,
            gene_desc_json=args.gene_desc,
            template_file=args.template,
            output_file=args.out,
            cell_line_idx=args.cell_line_idx,
            max_cases=args.max_cases,
            seed=args.seed,
        )
        return 0

    if args.task in ("de", "dir") and args.stage == "infer":
        run_inference(
            model_name=args.model,
            prompts_file=args.prompts,
            output_file=args.out,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            dtype=args.dtype,
            device_map=args.device_map,
            chat_template_path=args.chat_template,
            max_input_tokens=args.max_input_tokens,
            sort_by_input_length=not args.disable_length_sort,
        )
        return 0

    if args.task in ("de", "dir") and args.stage == "infer-api":
        run_inference_api(
            api_url=args.api_url,
            api_model=args.api_model,
            api_key=args.api_key,
            prompts_file=args.prompts,
            output_file=args.out,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            sleep_secs=args.sleep_secs,
        )
        return 0

    if args.task == "single" and args.stage == "prompt":
        case_split = "" if args.case_split == "all" else args.case_split
        generate_single_case_prompt(
            task=args.mode,
            pert=args.pert,
            gene=args.gene,
            cell_line=args.cell_line,
            data_csv=args.data_csv,
            drug_desc_json=args.drug_desc,
            gene_desc_json=args.gene_desc,
            drug_sim_json=args.drug_sim,
            gene_sim_json=args.gene_sim,
            template_file=args.template,
            output_file=args.out,
            max_candidates=args.max_candidates,
            budget=args.budget,
            case_split=case_split,
            seed=args.seed,
            llm_api_url=args.llm_api_url,
            llm_api_model=args.llm_api_model,
            llm_api_key=args.llm_api_key,
            llm_candidate_pool=args.llm_candidate_pool,
            llm_timeout=args.llm_timeout,
        )
        return 0

    parser.error("Unknown stage")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
