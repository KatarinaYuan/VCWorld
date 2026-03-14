# VCWorld Collaboration README

This README is for collaborators working on the current `VCWorld` codebase in this repository.  
It focuses on:

- what training/inference paradigms are implemented now,
- which script corresponds to each paradigm,
- copy-paste command templates you can run directly.

---

## 1. Repo layout and run directory

Most runnable scripts are under:

- `VCWorld/src/cli_pipeline/cli.py` (main DE/DIR pipeline)
- `VCWorld/src/cli_pipeline/tools/*.py` (training/eval/adaptation pipelines)

Run all commands below from:

```bash
cd VCWorld/src/cli_pipeline
```

---

## 2. Shared variables (recommended)

Set these once, then reuse in all commands:

```bash
export TASK=de                      # de or dir
export CELL=C32
export RUN_DIR=/path/to/exp/${CELL}_${TASK}
export MODEL_NAME=/path/to/your/base-llm
export KG_DIR=../../KG

export H5AD=/path/to/${CELL}.h5ad
export DRUG_SIM=/path/to/combined_similarity_sorted.json
export GENE_SIM=/path/to/results_close_gene.json
export DRUG_DESC=${KG_DIR}/drug_simp.json
export GENE_DESC=${KG_DIR}/gene_output.json
```

After `prepare` step, these are commonly used:

```bash
export LABELS_CSV=${RUN_DIR}/${CELL}_DE.csv
export RETR_TRAIN=${RUN_DIR}/${CELL}_DE_retrieval_train.json
export RETR_TEST=${RUN_DIR}/${CELL}_DE_retrieval_test.json
export PROMPTS_TRAIN=${RUN_DIR}/${CELL}_DE_prompts_train.txt
export PROMPTS_TEST=${RUN_DIR}/${CELL}_DE_prompts_test.txt
export PROMPTS_TRAIN_GOLD=${RUN_DIR}/${CELL}_DE_prompts_train_gold.txt
```

If `TASK=dir`, replace `DE` with `DIR` in file names.

---

## 3. Paradigm A: Prompt-only pipeline (no model training)

This is the default end-to-end VCWorld pipeline:

`prepare -> retrieve -> prompt -> infer -> evaluate`

### 3.1 Prepare labels/splits

```bash
python cli.py ${TASK} prepare \
  --h5ad "${H5AD}" \
  --out-dir "${RUN_DIR}" \
  --cell-line "${CELL}" \
  --seed 42
```

Supported split modes in `prepare`:

- `random_perturbation` (default)
- `k_perturbation_fixed_genes` (for fixed-support settings with `--k-support-perturbations` / `--m-genes-per-perturbation`)

### 3.2 Retrieval for train/test prompts

```bash
python cli.py ${TASK} retrieve \
  --data-csv "${LABELS_CSV}" \
  --drug-sim "${DRUG_SIM}" \
  --gene-sim "${GENE_SIM}" \
  --out "${RETR_TRAIN}" \
  --case-split train \
  --budget 10

python cli.py ${TASK} retrieve \
  --data-csv "${LABELS_CSV}" \
  --drug-sim "${DRUG_SIM}" \
  --gene-sim "${GENE_SIM}" \
  --out "${RETR_TEST}" \
  --case-split test \
  --budget 10
```

### 3.3 Prompt generation

```bash
python cli.py ${TASK} prompt \
  --retrieval "${RETR_TRAIN}" \
  --drug-desc "${DRUG_DESC}" \
  --gene-desc "${GENE_DESC}" \
  --out "${PROMPTS_TRAIN}"

python cli.py ${TASK} prompt \
  --retrieval "${RETR_TEST}" \
  --drug-desc "${DRUG_DESC}" \
  --gene-desc "${GENE_DESC}" \
  --out "${PROMPTS_TEST}"
```

### 3.4 Inference (choose one backend)

HF Transformers:

```bash
python cli.py ${TASK} infer \
  --model "${MODEL_NAME}" \
  --prompts "${PROMPTS_TEST}" \
  --out "${RUN_DIR}/pred_${TASK}_hf.txt" \
  --batch-size 4 \
  --max-new-tokens 1024
```

vLLM:

```bash
python cli.py ${TASK} infer-vllm \
  --model "${MODEL_NAME}" \
  --prompts "${PROMPTS_TEST}" \
  --out "${RUN_DIR}/pred_${TASK}_vllm.txt" \
  --batch-size 16 \
  --max-new-tokens 1024 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.75
```

API endpoint:

```bash
python cli.py ${TASK} infer-api \
  --api-url "https://your-endpoint/v1/chat/completions" \
  --api-model "your-model-name" \
  --prompts "${PROMPTS_TEST}" \
  --out "${RUN_DIR}/pred_${TASK}_api.txt"
```

### 3.5 Evaluation

```bash
python tools/evaluate_predictions.py \
  --task "${TASK}" \
  --prompts "${PROMPTS_TEST}" \
  --predictions "${RUN_DIR}/pred_${TASK}_hf.txt" \
  --truth-csv "${LABELS_CSV}" \
  --out-dir "${RUN_DIR}/eval_prompt_only" \
  --split test
```

---

## 4. Training paradigms currently implemented

### 4.1 Graph-only baseline

Scripts:

- train: `tools/train_graph_only_baseline.py`
- inference/eval: `tools/eval_graph_only_baseline.py`

Train:

```bash
python tools/train_graph_only_baseline.py \
  --labels-csv "${LABELS_CSV}" \
  --kg-dir "${KG_DIR}" \
  --out-ckpt "${RUN_DIR}/ckpt/graph_only.ckpt"
```

Inference/Eval:

```bash
python tools/eval_graph_only_baseline.py \
  --ckpt "${RUN_DIR}/ckpt/graph_only.ckpt" \
  --labels-csv "${LABELS_CSV}" \
  --kg-dir "${KG_DIR}" \
  --out-dir "${RUN_DIR}/eval_graph_only" \
  --split test
```

---

### 4.2 Graph + LLM-hidden hybrid baseline

Scripts:

- train: `tools/train_graph_llm_hybrid.py`
- inference/eval: `tools/eval_graph_llm_hybrid.py`

Train:

```bash
python tools/train_graph_llm_hybrid.py \
  --model-name "${MODEL_NAME}" \
  --prompts-file "${PROMPTS_TRAIN}" \
  --labels-csv "${LABELS_CSV}" \
  --kg-dir "${KG_DIR}" \
  --hidden-cache "${RUN_DIR}/cache/graph_llm_hidden.npz" \
  --out-ckpt "${RUN_DIR}/ckpt/graph_llm_hybrid.ckpt" \
  --bf16 \
  --trust-remote-code
```

Inference/Eval:

```bash
python tools/eval_graph_llm_hybrid.py \
  --ckpt "${RUN_DIR}/ckpt/graph_llm_hybrid.ckpt" \
  --prompts-file "${PROMPTS_TEST}" \
  --labels-csv "${LABELS_CSV}" \
  --kg-dir "${KG_DIR}" \
  --out-dir "${RUN_DIR}/eval_graph_llm_hybrid" \
  --split test \
  --bf16 \
  --trust-remote-code
```

---

### 4.3 Subgraph V1 (support-conditioned candidate subgraph selection)

Scripts:

- train: `tools/train_subgraph_v1.py`
- inference/eval: `tools/eval_subgraph_v1.py`

Train:

```bash
python tools/train_subgraph_v1.py \
  --labels-csv "${LABELS_CSV}" \
  --kg-dir "${KG_DIR}" \
  --out-ckpt "${RUN_DIR}/ckpt/subgraph_v1.ckpt" \
  --val-split test
```

Inference/Eval:

```bash
python tools/eval_subgraph_v1.py \
  --ckpt "${RUN_DIR}/ckpt/subgraph_v1.ckpt" \
  --labels-csv "${LABELS_CSV}" \
  --kg-dir "${KG_DIR}" \
  --out-dir "${RUN_DIR}/eval_subgraph_v1" \
  --split test
```

Note: default `train_subgraph_v1.py --val-split` is `val`.  
If your CSV has only `train/test`, set `--val-split test` (as above).

---

### 4.4 Direct supervised LoRA baseline

Scripts:

- train: `tools/train_direct_lora.py`
- inference/eval: `tools/eval_direct_lora.py`

Train:

```bash
python tools/train_direct_lora.py \
  --model-name "${MODEL_NAME}" \
  --prompts-file "${PROMPTS_TRAIN}" \
  --labels-csv "${LABELS_CSV}" \
  --output-dir "${RUN_DIR}/direct_lora" \
  --bf16 \
  --trust-remote-code
```

Inference/Eval:

```bash
python tools/eval_direct_lora.py \
  --model-name "${MODEL_NAME}" \
  --adapter-dir "${RUN_DIR}/direct_lora/best_adapter" \
  --prompts-file "${PROMPTS_TEST}" \
  --labels-csv "${LABELS_CSV}" \
  --split test \
  --out-dir "${RUN_DIR}/eval_direct_lora" \
  --bf16 \
  --trust-remote-code
```

---

### 4.5 MSLD (support-supervised latent mechanism distillation)

Scripts:

- train: `tools/train_msld_support.py`
- query inference: `tools/infer_msld_query.py`
- optional metric eval: `tools/evaluate_predictions.py`

Train:

```bash
python tools/train_msld_support.py \
  --model-name "${MODEL_NAME}" \
  --prompts-file "${PROMPTS_TRAIN}" \
  --labels-csv "${LABELS_CSV}" \
  --kg-dir "${KG_DIR}" \
  --out-ckpt "${RUN_DIR}/ckpt/msld.ckpt" \
  --bf16 \
  --trust-remote-code
```

Query inference:

```bash
python tools/infer_msld_query.py \
  --model-name "${MODEL_NAME}" \
  --prompts-file "${PROMPTS_TEST}" \
  --labels-csv "${LABELS_CSV}" \
  --ckpt "${RUN_DIR}/ckpt/msld.ckpt" \
  --out "${RUN_DIR}/pred_msld.txt" \
  --split test \
  --bf16 \
  --trust-remote-code
```

Metric evaluation:

```bash
python tools/evaluate_predictions.py \
  --task "${TASK}" \
  --prompts "${PROMPTS_TEST}" \
  --predictions "${RUN_DIR}/pred_msld.txt" \
  --truth-csv "${LABELS_CSV}" \
  --out-dir "${RUN_DIR}/eval_msld" \
  --split test
```

---

### 4.6 TTT family

#### TTT-LoRA (mixed-domain LoRA training)

Script:

- train: `tools/run_ttt_lora.py`

Train:

```bash
python tools/run_ttt_lora.py \
  --task "${TASK}" \
  --model-name "${MODEL_NAME}" \
  --prompts-file "${PROMPTS_TRAIN}" \
  --labels-csv "${LABELS_CSV}" \
  --output-dir "${RUN_DIR}/ttt_lora_adapter" \
  --bf16 \
  --trust-remote-code
```

Inference: reuse `tools/eval_direct_lora.py` with `--adapter-dir "${RUN_DIR}/ttt_lora_adapter"`.

#### TTT-lite (test-time adaptation inference)

Script:

- inference: `tools/run_ttt_lite.py`

```bash
python tools/run_ttt_lite.py \
  --model-name "${MODEL_NAME}" \
  --prompts-file "${PROMPTS_TEST}" \
  --labels-csv "${LABELS_CSV}" \
  --out "${RUN_DIR}/pred_ttt_lite.txt" \
  --prediction-mode label-ranking \
  --inner-adapt-source query \
  --inner-steps 3 \
  --bf16 \
  --trust-remote-code
```

For graph-mechanism-verification mode, add:

```bash
--prediction-mode graph-mechanism-verification --kg-dir "${KG_DIR}"
```

#### TTT-E2E-lite (first-order train + optional prediction)

Script:

- train/predict: `tools/run_ttt_e2e_lite.py`

If you need supervised output targets in prompts, generate prompts with gold labels:

```bash
python cli.py ${TASK} prompt \
  --retrieval "${RETR_TRAIN}" \
  --drug-desc "${DRUG_DESC}" \
  --gene-desc "${GENE_DESC}" \
  --labels-csv "${LABELS_CSV}" \
  --include-gold-label \
  --out "${PROMPTS_TRAIN_GOLD}"
```

Train + predict:

```bash
python tools/run_ttt_e2e_lite.py \
  --model-name "${MODEL_NAME}" \
  --prompts-file "${PROMPTS_TRAIN_GOLD}" \
  --labels-csv "${LABELS_CSV}" \
  --output-dir "${RUN_DIR}/ttt_e2e_adapter" \
  --num-epochs 1 \
  --run-predict \
  --predictions-out "${RUN_DIR}/pred_ttt_e2e.txt" \
  --prediction-mode label-ranking \
  --bf16 \
  --trust-remote-code
```

Predict-only from an existing adapter:

```bash
python tools/run_ttt_e2e_lite.py \
  --model-name "${MODEL_NAME}" \
  --prompts-file "${PROMPTS_TEST}" \
  --labels-csv "${LABELS_CSV}" \
  --output-dir "${RUN_DIR}/ttt_e2e_predict_only" \
  --predict-only \
  --predict-ckpt "${RUN_DIR}/ttt_e2e_adapter" \
  --run-predict \
  --predictions-out "${RUN_DIR}/pred_ttt_e2e_predict_only.txt" \
  --prediction-mode label-ranking \
  --bf16 \
  --trust-remote-code
```

---

## 5. Sharded parallel inference utilities

Split prompts:

```bash
python tools/shard_prompts.py \
  --prompts "${PROMPTS_TEST}" \
  --out-dir "${RUN_DIR}/shards_32" \
  --num-shards 32 \
  --shard-prefix prompts
```

Run inference per shard (example for shard 000):

```bash
python cli.py ${TASK} infer-vllm \
  --model "${MODEL_NAME}" \
  --prompts "${RUN_DIR}/shards_32/prompts_000.txt" \
  --out "${RUN_DIR}/shards_32/predictions_000.txt" \
  --batch-size 16 \
  --max-new-tokens 1024
```

Merge shard predictions:

```bash
python tools/merge_predictions.py \
  --manifest "${RUN_DIR}/shards_32/shard_manifest.json" \
  --pred-dir "${RUN_DIR}/shards_32" \
  --out "${RUN_DIR}/pred_sharded_merged.txt"
```

Optional shard verification:

```bash
python tools/verify_prompt_shards.py \
  --original-prompts "${PROMPTS_TEST}" \
  --manifest "${RUN_DIR}/shards_32/shard_manifest.json" \
  --reconstructed-out "${RUN_DIR}/shards_32/reconstructed_prompts.txt"
```

---

## 6. Quick mapping table

| Paradigm | Train script | Inference / eval script |
|---|---|---|
| Prompt-only (no training) | N/A | `cli.py {de/dir} infer` / `infer-vllm` / `infer-api` |
| Graph-only baseline | `tools/train_graph_only_baseline.py` | `tools/eval_graph_only_baseline.py` |
| Graph + LLM-hidden hybrid | `tools/train_graph_llm_hybrid.py` | `tools/eval_graph_llm_hybrid.py` |
| Subgraph V1 | `tools/train_subgraph_v1.py` | `tools/eval_subgraph_v1.py` |
| Direct LoRA | `tools/train_direct_lora.py` | `tools/eval_direct_lora.py` |
| MSLD | `tools/train_msld_support.py` | `tools/infer_msld_query.py` |
| TTT-LoRA | `tools/run_ttt_lora.py` | `tools/eval_direct_lora.py` |
| TTT-lite | N/A (adapt at test time) | `tools/run_ttt_lite.py` |
| TTT-E2E-lite | `tools/run_ttt_e2e_lite.py` | `tools/run_ttt_e2e_lite.py --predict-only` |

---

## 7. Common gotchas

- Run directory: use `VCWorld/src/cli_pipeline` as working directory.
- Split consistency: make sure `prompts_file` and `labels_csv` refer to the same `(pert, gene)` universe.
- Task switch:
  - use `de` vs `dir` consistently in CLI task name and templates;
  - for `dir`, labels correspond to `increase/decrease/insufficient` semantics.
- Subgraph training: if CSV has no `val` split, pass `--val-split test`.
- TTT-E2E supervised training: use prompts that contain meaningful `[Start of Output]` targets (typically via `--include-gold-label`).

---

## 8. Citation

If you use VCWorld, please cite:

```bibtex
@inproceedings{vcworld2026,
  title={VCWorld: A Biological World Model for Virtual Cell Simulation},
  author={Wei, Zhijian and Ma, Runze and Wang, Zichen and Li, Zhongmin and Song, Shuotong and Zheng, Shuangjia},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
