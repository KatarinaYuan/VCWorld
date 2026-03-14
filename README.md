# VCWorld Collaboration README

This README is for collaborators working on the current `VCWorld` codebase in this repository.  
It focuses on:

- what training/inference paradigms are implemented now,
- which script corresponds to each paradigm,
- copy-paste command templates you can run directly.

---

## File structure

```
|- PerturbReason
    |- VCWorld (submodule from `https://github.com/KatarinaYuan/VCWorld/`) # current coding is built in VCWorld dir
    |- src/
```

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
export CELL=C32_k10_m50
export RUN_DIR=/projects/AI4D/core-132/PerturbReason_data
export MODEL_NAME=/projects/AI4D/core-132/PerturbReason_ckpt/Meta-Llama-3.2-3B-Instruct
export KG_DIR=/projects/AI4D/core-132/PerturbReason_data/VCWorld/KG
export VCWORLD_DATA_DIR=/projects/AI4D/core-132/PerturbReason_data/VCWorld

export H5AD=/projects/AI4D/core-132/PerturbReason_data/C32_all_plates_hvg.h5ad
export DRUG_SIM=${VCWORLD_DATA_DIR}/combined_similarity_sorted.json
export GENE_SIM=${VCWORLD_DATA_DIR}/results_close_gene.json
export DRUG_DESC=${VCWORLD_DATA_DIR}/drug_simp.json
export GENE_DESC=${VCWORLD_DATA_DIR}/gene_output.json
```

After `prepare` step, these are commonly used:

```bash
export LABELS_CSV=${RUN_DIR}/${CELL}_DE.csv
export RETR_TRAIN=${RUN_DIR}/${CELL}_DE_training_retrieval.json
export RETR_TEST=${RUN_DIR}/${CELL}_DE_retrieval.json
export PROMPTS_TRAIN=${RUN_DIR}/${CELL}_DE_training_prompts.txt
export PROMPTS_TEST=${RUN_DIR}/${CELL}_DE_prompts.txt
export PROMPTS_TRAIN_GOLD=${RUN_DIR}/${CELL}_DE_prompts_train_gold.txt
```

If `TASK=dir`, replace `DE` with `DIR` in file names.

---

## 3. Paradigm A: Prompt-only pipeline (no model training)

This is the default end-to-end VCWorld pipeline:

`prepare -> retrieve -> prompt -> infer -> evaluate`



<detail>
<summary>3.1 Prepare labels/splits</summary>
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
</detail>

<details>
<summary>3.2 Retrieval for train/test prompts</summary>
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
</details>

<details>
<summary>3.3 Prompt generation</summary>
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
</detail>



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

### 4.1 Direct supervised LoRA baseline

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

## 5. Sharded parallel inference utilities

> Split prompts:
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
| Direct LoRA | `tools/train_direct_lora.py` | `tools/eval_direct_lora.py` |

---

## 7. Common gotchas

- Run directory: use `VCWorld/src/cli_pipeline` as working directory.

---

