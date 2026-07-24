#!/usr/bin/env bash
#SBATCH --job-name=sam_sahi
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/sam3_sahi_fss/sam_sahi_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# SAHI-sliced SAM3 "visual" proposals + DINOv3 prototype-set match.
# Start with a handful of queries to check timing, then drop --max-queries for the
# full dataset (a 1024 crop at slice=256/overlap=0.2 is 25 SAM3 forwards per image).
python sam3_sahi_fss/run_fertilizer.py \
    --support-id 0 --max-queries 7 \
    --dtype bfloat16 --proposal-text "visual" \
    --slice-size 256 --overlap-ratio 0.2 --score-threshold 0.3 \
    --match-mode prototype --overlay-threshold 0.5 --n-overlays 5 \
    --out-dir "${REPO}/few-shot-segmentation/sam3_sahi_fss/results"
echo "[job] DONE"
