#!/usr/bin/env bash
#SBATCH --job-name=fss_eval_all
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_eval_all_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Broad evaluation: compare boxes / per_granule / router (calibrated defaults:
# seed_coverage 0.7, min_seeds_dense 2) on EVERY held-out COCO image (support=0),
# so we can see per-image which approach wins and confirm the router across the
# full set rather than just q60/q105. Plus 3 unseen March13 overlays (qualitative).
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 15 30 45 60 75 90 105 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.45 \
    --compare-paths \
    --router-density-metric seed_coverage --granule-min-distance 6 \
    --unseen-subdir March13 --n-unseen 3 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/router_eval_all
echo "[job] DONE"
