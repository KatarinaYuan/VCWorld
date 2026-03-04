#!/bin/bash
#SBATCH --job-name=vcworld-de-ttt-lite-fast-labelpred
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
#SBATCH --output=slurm-fastlp-%A_%a.out
#SBATCH --error=slurm-fastlp-%A_%a.err

set -euo pipefail

echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID}"
/usr/bin/nvidia-smi

module purge
module load conda/3-24.9.0
conda activate /home/ext.xinyu.yuan/work/py10torch26

cd /gpfs/work/ext.xinyu.yuan/projects/PerturbReason/VCWorld/src/cli_pipeline
# Keep Slurm-assigned GPU visibility; do NOT unset CUDA_VISIBLE_DEVICES.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

tag=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")

# Adjust these paths as needed.
SHARD_ROOT="/projects/AI4D/core-132/PerturbReason_data/C32_DE_shards_32"
PROMPTS_SHARD="${SHARD_ROOT}/prompts_${tag}.txt"
TRUTH_CSV="/projects/AI4D/core-132/PerturbReason_data/C32_DE.csv"


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

# Max wait time is 5 minutes. Fail fast if memory is still not ready.
if [ "${MIN_FREE_MIB}" -lt 24500 ]; then
  echo "ERROR: GPU memory not ready after 5 minutes (min_free=${MIN_FREE_MIB} MiB). Abort this shard."
  exit 1
fi

# Fill in your python command below.
# Example:
# PYTHONUNBUFFERED=1 python -u tools/evaluate_predictions.py \
#   --task de \
#   --prompts "${PROMPTS_SHARD}" \
#   --predictions "${PREDICTIONS_FILE}" \
#   --truth-csv "${TRUTH_CSV}" \
#   --out-dir "${OUT_DIR}" \
#   --split test


PYTHONUNBUFFERED=1 python -u tools/run_ttt_lite.py   \
  --model-name /projects/AI4D/core-132/PerturbReason_ckpt/Meta-Llama-3.2-3B-Instruct/  \
  --prompts-file "${PROMPTS_SHARD}"   \
  --labels-csv "${TRUTH_CSV}"   \
  --out /projects/AI4D/core-132/PerturbReason_data/C32_DE_llama3b_ttt_lite_fast_labelpred_predictions_${tag}.txt   \
  --inner-steps 1   --inner-lr 5e-5   --inner-max-tokens 4096   \
  --prediction-mode label-ranking   --label-candidates yes,no,insufficient   \
  --temperature 0.6   --top-p 0.9   --lora-r 16   --lora-alpha 32   \
  --lora-dropout 0.05   --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

