#!/bin/bash
#SBATCH --job-name=vcworld-de-msld-infer-k0_m50
#SBATCH --partition=TrixieMain
#SBATCH --account=ai4d-core-132
#SBATCH --exclude=trixie-cn109,trixie-cn130
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=0-11:59:00
#SBATCH --array=0-31
#SBATCH --output=slurm-msld-infer-k0_m50-%A_%a.out
#SBATCH --error=slurm-msld-infer-k0_m50-%A_%a.err

set -euo pipefail

echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID}"
/usr/bin/nvidia-smi

module purge
module load conda/3-24.9.0
conda activate /home/ext.xinyu.yuan/work/py10torch26

cd /gpfs/work/ext.xinyu.yuan/projects/PerturbReason/VCWorld/src/cli_pipeline
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

tag=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")

# Adjust paths for your experiment.
MODEL_PATH="/projects/AI4D/core-132/PerturbReason_ckpt/Meta-Llama-3.2-3B-Instruct/"
SHARD_ROOT="/projects/AI4D/core-132/PerturbReason_data/C32_k0_m50_DE_shards_32"
PROMPTS_SHARD="${SHARD_ROOT}/prompts_${tag}.txt"
TRUTH_CSV="/projects/AI4D/core-132/PerturbReason_data/C32_k0_m50_DE.csv"
MSLD_CKPT="/projects/AI4D/core-132/PerturbReason_data/C32_k0_m50_DE_msld_phi_star.pt"
OUT_SHARD="${SHARD_ROOT}/msld_llama3b_k0_m50_predictions_${tag}.txt"

# Wait for GPU free memory to avoid startup race with stale processes/cache release.
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

PYTHONUNBUFFERED=1 python -u tools/infer_msld_query.py \
  --model-name "${MODEL_PATH}" \
  --prompts-file "${PROMPTS_SHARD}" \
  --labels-csv "${TRUTH_CSV}" \
  --ckpt "${MSLD_CKPT}" \
  --out "${OUT_SHARD}" \
  --split test \
  --max-input-tokens 4096 \
  --bf16 \
  --trust-remote-code

