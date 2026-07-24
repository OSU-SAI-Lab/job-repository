#!/usr/bin/env bash
#SBATCH --job-name=fss_fert
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_fert_%j.log
set -euo pipefail

REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"

export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1     # weights pre-cached on the login node

module load miniconda3/24.1.2-py310
# shellcheck source=/dev/null
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss

echo "[job] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('[job] cuda available:', torch.cuda.is_available())"

# Crop-mode test: tiny granules need a usable patch grid (support id 0; queries 105,60 disjoint).
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 105 60 \
    --dinov3 base --dtype bfloat16 \
    --unseen-subdir March13 --n-unseen 2

echo "[job] DONE"
