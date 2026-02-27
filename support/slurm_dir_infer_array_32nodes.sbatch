#!/bin/bash
#SBATCH --job-name=vcworld-de-infer
#SBATCH --partition=TrixieLong
#SBATCH --account=ai4d-core-132
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=1-23:59:00
#SBATCH --array=0-31
#SBATCH --output=slurm-infer-%A_%a.out
#SBATCH --error=slurm-infer-%A_%a.err

set -euo pipefail

echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID}"
/usr/bin/nvidia-smi

module purge
module load conda/3-24.9.0
conda activate /home/ext.xinyu.yuan/work/py10torch26

cd /gpfs/work/ext.xinyu.yuan/projects/PerturbReason/VCWorld/src/cli_pipeline
unset CUDA_VISIBLE_DEVICES

# Adjust these paths as needed.
MODEL_PATH="/projects/AI4D/core-132/PerturbReason_ckpt/Meta-Llama-3-8B-Instruct/"
SHARD_ROOT="/projects/AI4D/core-132/PerturbReason_data/C32_DE_shards_32"
PROMPTS_SHARD="${SHARD_ROOT}/prompts_$(printf "%03d" "${SLURM_ARRAY_TASK_ID}").txt"
OUT_SHARD="${SHARD_ROOT}/predictions_maxtoken2k_$(printf "%03d" "${SLURM_ARRAY_TASK_ID}").txt"

PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u cli.py de infer \
  --model "${MODEL_PATH}" \
  --prompts "${PROMPTS_SHARD}" \
  --out "${OUT_SHARD}" \
  --batch-size 4 \
  --max-new-tokens 2048
