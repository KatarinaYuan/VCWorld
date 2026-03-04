#!/bin/bash
#SBATCH --job-name=vcworld-de-eval-ttt-lite
#SBATCH --partition=TrixieMain
#SBATCH --account=ai4d-core-132
#SBATCH --exclude=trixie-cn109
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-11:59:00
#SBATCH --array=0-31
#SBATCH --output=slurm-eval-%A_%a.out
#SBATCH --error=slurm-eval-%A_%a.err

set -euo pipefail

echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID}"

module purge
module load conda/3-24.9.0
conda activate /home/ext.xinyu.yuan/work/py10torch26

cd /gpfs/work/ext.xinyu.yuan/projects/PerturbReason/VCWorld/src/cli_pipeline

tag=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")

# If you have per-shard predictions, replace this with:
# PRED_FILE="/projects/AI4D/core-132/PerturbReason_data/C32_DE_shards_32/predictions_${tag}.txt"
PRED_FILE="/projects/AI4D/core-132/PerturbReason_data/C32_DE_ttt_lite_predictions_fast.txt"

PYTHONUNBUFFERED=1 python tools/run_ttt_lite.py   \
  --model-name /projects/AI4D/core-132/PerturbReason_ckpt/Meta-Llama-3.2-3B-Instruct/  \
  --prompts-file /projects/AI4D/core-132/PerturbReason_data/C32_DE_shards_32/prompts_${tag}.txt   \
  --labels-csv /projects/AI4D/core-132/PerturbReason_data/C32_DE.csv   \
  --out /projects/AI4D/core-132/PerturbReason_data/C32_DE_llama3b_ttt_lite_predictions_fast_labelpred_${tag}.txt   \
  --inner-steps 1   --inner-lr 5e-5   --inner-max-tokens 4096   \
  --prediction-mode label-ranking   --label-candidates yes,no,insufficient   \
  --temperature 0.6   --top-p 0.9   --lora-r 16   --lora-alpha 32   \
  --lora-dropout 0.05   --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
