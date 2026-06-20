#!/usr/bin/env bash
#SBATCH --job-name=fss_router_cmp
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_router_cmp_%j.log
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

# TASK 2 — per-region router acceptance comparison. --compare-paths runs THREE
# paths on every query and prints the per-regime IoU/P/R/bgFP table + the router
# verdict (match-or-beat the better pure path on each query):
#   boxes        — original blob+box pipeline (wins on the dense q60 pile)
#   per_granule  — one prompt per granule seed   (wins on scattered q105)
#   router_cov*  — per-component dense/scattered routing, swept over coverage
#                  thresholds for calibration (dense → boxes, scattered → seeds).
# Selection/gating stay at defaults (sam_score, no gate) per Task 3.
# q60 = the dense/pile subset; q105 = the scattered subset. The router should
# recover ~0.82 on q60 AND keep the per-granule recall/precision on q105.
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 60 105 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.45 \
    --compare-paths --router-thresholds 0.4 0.55 0.7 \
    --router-density-metric seed_coverage --granule-min-distance 6 \
    --unseen-subdir March13 --n-unseen 0 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/router_compare

echo "[job] DONE"
