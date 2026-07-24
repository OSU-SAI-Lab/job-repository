#!/usr/bin/env bash
#SBATCH --job-name=fss_unseen2
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_unseen2_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# 10 MORE unseen images (indices 4-8 in each of March13/March14, no overlap with the
# first batch of 8), current config.py defaults, into the same fss/results folder.
python scripts/run_unseen_visuals.py \
    --support-id 0 --folders March13 March14 \
    --start-index 4 --n-per-folder 5 \
    --dinov3 base --dtype bfloat16 \
    --out-dir /users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/fss/results
echo "[job] DONE"
