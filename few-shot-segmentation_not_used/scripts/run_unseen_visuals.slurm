#!/usr/bin/env bash
#SBATCH --job-name=fss_unseen
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_unseen_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Render per-granule visuals on truly unseen March13/March14 images using the
# CURRENT config.py defaults verbatim (no knob overrides). Writes overlay + debug
# + mask per image into few-shot-segmentation/fss/results.
python scripts/run_unseen_visuals.py \
    --support-id 0 --folders March13 March14 --n-per-folder 4 \
    --dinov3 base --dtype bfloat16 \
    --out-dir /users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/fss/results
echo "[job] DONE"
