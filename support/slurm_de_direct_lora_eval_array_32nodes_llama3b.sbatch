#!/bin/bash
#SBATCH --job-name=vcworld-de-direct-lora-eval
#SBATCH --partition=TrixieMain
#SBATCH --account=ai4d-core-132
#SBATCH --exclude=trixie-cn109
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=0-11:59:00
#SBATCH --array=0-31
#SBATCH --output=slurm-direct-lora-eval-%A_%a.out
#SBATCH --error=slurm-direct-lora-eval-%A_%a.err

set -euo pipefail

echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID}"
/usr/bin/nvidia-smi

module purge
module load conda/3-24.9.0
conda activate /home/ext.xinyu.yuan/work/py10torch26

cd /gpfs/work/ext.xinyu.yuan/projects/PerturbReason/VCWorld/src/cli_pipeline
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

flag="${FLAG:-k10_m50}"
tag=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")

# Prefer direct-LoRA sharded prompts. Fall back to generic DE shards if needed.
SHARD_ROOT="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE_direct_lora_shards_32"
if [ ! -d "${SHARD_ROOT}" ]; then
  SHARD_ROOT="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE_shards_32"
fi
PROMPTS_SHARD="${SHARD_ROOT}/prompts_${tag}.txt"
TRUTH_CSV="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE.csv"

ADAPTER_DIR="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE_llama3b_direct_lora_predictions/best_adapter"
OUT_ROOT="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE_llama3b_direct_lora_predictions"
OUT_DIR="${OUT_ROOT}/shard_${tag}"

if [ ! -f "${PROMPTS_SHARD}" ]; then
  echo "ERROR: prompts shard not found: ${PROMPTS_SHARD}"
  exit 1
fi
if [ ! -f "${TRUTH_CSV}" ]; then
  echo "ERROR: labels csv not found: ${TRUTH_CSV}"
  exit 1
fi
if [ ! -d "${ADAPTER_DIR}" ]; then
  echo "ERROR: adapter dir not found: ${ADAPTER_DIR}"
  exit 1
fi

for _ in $(seq 1 30); do
  MIN_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{m=$1} $1<m{m=$1} END{print m}')
  if [ "${MIN_FREE_MIB}" -ge 24500 ]; then
    echo "GPU free memory looks ready: min_free=${MIN_FREE_MIB} MiB"
    break
  fi
  echo "Waiting for GPU memory release... min_free=${MIN_FREE_MIB} MiB"
  sleep 10
done

if [ "${MIN_FREE_MIB}" -lt 24500 ]; then
  echo "ERROR: GPU memory not ready after 5 minutes (min_free=${MIN_FREE_MIB} MiB). Abort this shard."
  exit 1
fi

echo "Running Direct-LoRA eval | flag=${flag} shard=${tag}"
echo "prompts=${PROMPTS_SHARD}"
echo "labels=${TRUTH_CSV}"
echo "adapter=${ADAPTER_DIR}"
echo "out_dir=${OUT_DIR}"

PYTHONUNBUFFERED=1 python -u tools/eval_direct_lora.py \
  --model-name /projects/AI4D/core-132/PerturbReason_ckpt/Meta-Llama-3.2-3B-Instruct/ \
  --adapter-dir "${ADAPTER_DIR}" \
  --prompts-file "${PROMPTS_SHARD}" \
  --labels-csv "${TRUTH_CSV}" \
  --split test \
  --out-dir "${OUT_DIR}" \
  --device-map-auto \
  --bf16 \
  --log-every-examples 50
