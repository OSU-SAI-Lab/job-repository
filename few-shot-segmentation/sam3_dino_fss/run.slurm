#!/usr/bin/env bash
#SBATCH --job-name=sam_dino
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/sam3_dino_fss/sam_dino_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# SAM3-proposals + DINOv3 cosine-match FSS on fertilizer GT (+ unseen overlays).
python sam3_dino_fss/run_fertilizer.py \
    --support-id 0 --query-ids 15 30 45 60 75 90 105 \
    --dtype bfloat16 --proposal-text "seed" --score-threshold 0.3 \
    --n-unseen 2 --unseen-subdir March13 \
    --out-dir /users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/sam3_dino_fss/results
echo "[job] DONE"
