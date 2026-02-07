# VCWorld: A Biological World Model for Virtual Cell Simulation

VCWorld is a cell-level white-box simulator that integrates structured biological knowledge with LLM-based reasoning to predict cellular responses to perturbations in an interpretable, data-efficient way.

This repository provides the official implementation for VCWorld, including:
- the CLI pipeline for DE/DIR prediction,
- prompt construction templates and single-case analysis utilities,
- inference runners for local HF models or API-backed LLMs.
![VCWorld pipeline](assets/VCWorld%20pipeline.png)
## Overview
VCWorld introduces a biological world model that explicitly reasons through mechanisms rather than relying on black-box prediction. It is designed for data-efficient, interpretable prediction of perturbation effects.

Key features:
- White-box reasoning grounded in pathways, protein interactions, and gene regulation.
- LLM-integrated inference with structured chain-of-thought style prompts.
- GeneTAK benchmark for DE and DIR prediction.
- Interpretable outputs with explicit reasoning traces.

## Paper
VCWorld: A Biological World Model for Virtual Cell Simulation
Poster at ICLR 2026

Paper (PDF): https://openreview.net/pdf/ca648eec9dadd2dff792120e2f222ad1bb9a14ff.pdf

## Model Architecture (high level)
The VCWorld pipeline runs in three stages:
1) Knowledge integration: builds an open-world biological knowledge graph from public sources.
2) Evidence retrieval: finds supporting cases using semantic and graph-aware similarity.
3) Chain-of-thought reasoning: synthesizes evidence to predict DE or DIR with mechanistic explanation.

## Dataset: GeneTAK
GeneTAK is derived from the Tahoe-100M single-cell atlas and focuses on gene-level perturbation responses.

- Cell lines: 5 (C32, HOP62, HepG2/C3A, Hs 766T, PANC-1)
- Perturbations: 348 drug compounds
- Tasks: Differential Expression (DE) and Directional Change (DIR)
- Format: triplets (cell line, perturbation, gene) with binary labels
- Splits: train/test by perturbation (30/70) to simulate few-shot conditions

## Repository Structure
- `cli.py`: CLI entrypoint
- `stages/prepare.py`: build DE/DIR CSV labels from h5ad
- `stages/retrieve.py`: build retrieval JSON from CSV + similarity files
- `stages/prompt.py`: build prompts from retrieval JSON + templates
- `stages/single_case/prompt.py`: build a single prompt for out-of-dataset triples
- `stages/infer.py`: local HF inference on prompts
- `stages/infer_api.py`: API inference on prompts

## Quick Start
### Environment Setup
```bash
git clone https://github.com/yourusername/VCWorld.git
cd VCWorld
conda create -n vcworld python=3.10
conda activate vcworld
pip install -r requirements.txt
```

## CLI Pipeline (DE/DIR)
Run from `pipeline/cli_pipeline`:

```bash
python cli.py de prepare \
  --h5ad /C32_cells.h5ad \
  --out-dir  \
  --cell-line C32

python cli.py de retrieve \
  --data-csv /C32_2000GENE.csv \
  --drug-sim /combined_similarity_sorted.json \
  --gene-sim /results_close_gene.json \
  --out /C32_DE.json \
  --budget 10 --seed 42

python cli.py de prompt \
  --retrieval /C32_DE.json \
  --template /DE_template.py \
  --drug-desc /drug_simp.json \
  --gene-desc /gene_output.json \
  --out /C32_DE_prompts.txt

python cli.py de infer \
  --model /Llama3.1-8B \
  --prompts /C32_prompts.txt \
  --out /C32.txt \
  --batch-size 4 --max-new-tokens 1024

python cli.py de infer-api \
  --api-url https://api.example.com/v1/chat/completions \
  --api-model your-model-name \
  --prompts /C32_prompts.txt \
  --out /C32.txt \
  --max-new-tokens 1024
```

For DIR, replace `de` with `dir` and use DIR CSV/output paths.

## Single-case (small sample) analysis
Use this when the (Pert, Gene, Cell line) triple is out of dataset. The flow is:
1) Search drug/gene similarity JSONs.
2) If missing, optionally use an LLM to pick the most similar drug/gene from description lists.
3) Pull similar (pert, gene) pairs from the CSV as evidence examples.

Example:

```bash
python cli.py single prompt \
  --pert BMP-2 \
  --gene ALK3 \
  --cell-line "C32 cells" \
  --data-csv /C32_2000GENE.csv \
  --drug-desc /drug_simp.json \
  --gene-desc /gene_output.json \
  --drug-sim /combined_similarity_sorted.json \
  --gene-sim /results_close_gene.json \
  --out /BMP-2_ALK3_C32_single_prompt.txt \
  --mode de \
  --case-split train
```

LLM fallback (optional):

```bash
python cli.py single prompt \
  --pert BMP-2 \
  --gene ALK3 \
  --cell-line "C32 cells" \
  --data-csv /C32_2000GENE.csv \
  --drug-desc /drug_simp.json \
  --gene-desc /gene_output.json \
  --drug-sim /combined_similarity_sorted.json \
  --gene-sim /results_close_gene.json \
  --out /BMP-2_ALK3_C32_single_prompt_llm.txt \
  --mode de \
  --llm-api-url https://api.openai-proxy.org/v1/chat/completions \
  --llm-api-model your-model-name \
  --llm-api-key $LLM_DRUG_API_KEY
```

Notes:
- `--cell-line` must match a name in the prompt template `cell_lines` list; otherwise the first entry is used.
- `--case-split` defaults to `train`; use `all` to search across splits.
- LLM fallback runs only when the query drug/gene is missing from the similarity JSON.
- `--mode` selects DE or DIR prompt format for the single-case prompt.

## Required Inputs and Formats
### Stage 1: prepare
Input: `.h5ad` (AnnData) with at least:
- `adata.obs[perturbation_col]` (default: `drug`)
- a control group value (default: `DMSO_TF`)

Output:
- `{cell_line}_DE.csv` with columns: `pert, gene, label, split`
- `{cell_line}_DIR.csv` with columns: `pert, gene, label, split`

### Stage 2: retrieve
Inputs:
- `data-csv`: CSV from stage 1
- `drug-sim`: JSON mapping `drug -> list` (your file uses `{\"Drug\": name}` entries)
- `gene-sim` (required): JSON mapping `gene -> list` (or `direct_neighbors` entries)

Output:
- retrieval JSON: list of objects
  - `test_case`: `{drug, gene}`
  - `retrieved_pairs`: `[[drug, gene], ...]`

### Stage 3: prompt
Inputs:
- `retrieval`: JSON from stage 2
- `drug-desc`: JSON mapping `drug -> description`
- `gene-desc`: JSON mapping `gene -> description`
- `template`: optional. If not provided, defaults to:
  - DE: `pipeline/support/DE_template.py` (expects `prompt_vcworld_DE`)
  - DIR: `pipeline/support/DIR_template.py` (expects `prompt_vcworld_DIR`)
  - Backward-compatible with `prompt_test_de` / `prompt_test_dir` if present

Output:
- text file with blocks:
  - `=== Prompt N (drug | gene) ===`
  - `[Start of Prompt] ... [End of Prompt]`
  - `[Start of Input] ... [End of Output]`

### Stage 4: infer
Inputs:
- `model`: HF model path or name
- `prompts`: prompt text file (stage 3 output)

Output:
- text file with blocks:
  - `--- Query for Prompt N ---`
  - model answer
  - separator line

### Stage 4b: infer-api
Inputs:
- `api-url`: API endpoint URL
- `api-model`: API model name
- `api-key`: optional; can also set `LLM_DRUG_API_KEY`
- `prompts`: prompt text file (stage 3 output)

## Citation
If you use VCWorld, please cite:

```bibtex
@inproceedings{vcworld2026,
  title={VCWorld: A Biological World Model for Virtual Cell Simulation},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Resources
- Paper (PDF): https://openreview.net/pdf/ca648eec9dadd2dff792120e2f222ad1bb9a14ff.pdf
- Tahoe-100M Dataset: https://huggingface.co/datasets/tahoebio/Tahoe-100M
