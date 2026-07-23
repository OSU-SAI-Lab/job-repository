#!/usr/bin/env bash
#SBATCH --job-name=sahi_cmp
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/sam3_sahi_fss/sahi_cmp_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Apples-to-apples with sam3_dino_fss: same support (id 0), same 7 query images,
# same 1024 crop. Difference is SAHI-sliced "visual" proposals + proposal-box
# prototypes instead of one whole-image "seed" pass + GT-component prototypes.
python sam3_sahi_fss/run_fertilizer.py \
    --support-id 0 --query-ids 15 30 45 60 75 90 105 \
    --crop-size 1024 --dtype bfloat16 --proposal-text "visual" \
    --slice-size 256 --overlap-ratio 0.2 --score-threshold 0.3 \
    --match-mode prototype --overlay-threshold 0.5 --n-overlays 7 \
    --out-dir "${REPO}/few-shot-segmentation/sam3_sahi_fss/results"
echo "[job] DONE"
