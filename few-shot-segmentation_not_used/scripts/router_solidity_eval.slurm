#!/usr/bin/env bash
#SBATCH --job-name=fss_rsol
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_rsol_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Calibrate the router's SOLIDITY gate (the decisive dense/scattered discriminator)
# vs boxes/per_granule on all 7 GT images, tuned config. Goal: router routes only
# the solid q60 pile to boxes and all scattered fields to per-granule → match boxes
# on clumped AND per_granule on scattered.
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 15 30 45 60 75 90 105 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.4 --seed-source brightness --granule-min-distance 5 \
    --compare-paths --router-solidities 0.3 0.4 0.5 0.6 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/router_solidity_eval
echo "[job] DONE"
