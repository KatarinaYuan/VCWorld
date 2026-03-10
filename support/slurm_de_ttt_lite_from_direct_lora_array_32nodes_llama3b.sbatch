#!/bin/bash
#SBATCH --job-name=vcworld-de-ttt-lite-directlora
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
#SBATCH --output=slurm-tttlite-directlora-%A_%a.out
#SBATCH --error=slurm-tttlite-directlora-%A_%a.err

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
inner_steps="${INNER_STEPS:-1}"
inner_lr="${INNER_LR:-5e-5}"
inner_max_tokens="${INNER_MAX_TOKENS:-5120}"
tag=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")

SHARD_ROOT="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE_direct_lora_shards_32"
PROMPTS_SHARD="${SHARD_ROOT}/prompts_${tag}.txt"
TRUTH_CSV="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE.csv"
INIT_ADAPTER="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE_llama3b_direct_lora_predictions/best_adapter"
OUT_FILE="/projects/AI4D/core-132/PerturbReason_data/C32_${flag}_DE_ttt_lite_from_direct_lora_fast_labelpred_predictions_${tag}.txt"

if [ ! -f "${PROMPTS_SHARD}" ]; then
  echo "ERROR: prompts shard not found: ${PROMPTS_SHARD}"
  exit 1
fi
if [ ! -f "${TRUTH_CSV}" ]; then
  echo "ERROR: labels csv not found: ${TRUTH_CSV}"
  exit 1
fi
if [ ! -d "${INIT_ADAPTER}" ]; then
  echo "ERROR: init adapter not found: ${INIT_ADAPTER}"
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

echo "flag=${flag} tag=${tag} inner_steps=${inner_steps} inner_lr=${inner_lr} inner_max_tokens=${inner_max_tokens}"
echo "prompts=${PROMPTS_SHARD}"
echo "labels=${TRUTH_CSV}"
echo "init_adapter=${INIT_ADAPTER}"
echo "out=${OUT_FILE}"

PYTHONUNBUFFERED=1 python -u tools/run_ttt_lite.py \
  --model-name /projects/AI4D/core-132/PerturbReason_ckpt/Meta-Llama-3.2-3B-Instruct/ \
  --init-adapter "${INIT_ADAPTER}" \
  --prompts-file "${PROMPTS_SHARD}" \
  --labels-csv "${TRUTH_CSV}" \
  --out "${OUT_FILE}" \
  --inner-steps "${inner_steps}" \
  --inner-lr "${inner_lr}" \
  --inner-max-tokens "${inner_max_tokens}" \
  --prediction-mode label-ranking \
  --label-candidates yes,no,insufficient \
  --temperature 0.0 \
  --top-p 1.0 \
  --bf16
