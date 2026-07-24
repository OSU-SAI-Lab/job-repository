#!/usr/bin/env bash
#SBATCH --job-name=fss_router_cal
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_router_cal_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Calibration follow-up: the cov0.7 router still routed 5/9 q105 components dense.
# Free small components (--router-min-seeds-dense 2) and sweep higher coverage
# thresholds to push tight scattered clusters onto the per-granule path, aiming to
# close the 0.03 IoU gap vs per-granule on the scattered subset while keeping the
# dense q60 pile at ~0.83.
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 60 105 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.45 \
    --compare-paths --router-thresholds 0.7 0.8 0.9 \
    --router-density-metric seed_coverage --granule-min-distance 6 \
    --router-min-seeds-dense 2 \
    --unseen-subdir March13 --n-unseen 0 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/router_calib
echo "[job] DONE"
