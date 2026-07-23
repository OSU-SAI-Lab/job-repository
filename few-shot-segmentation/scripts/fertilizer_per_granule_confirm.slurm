#!/usr/bin/env bash
#SBATCH --job-name=fss_pg_confirm
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_pg_confirm_%j.log
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

# TASK 1 — confirmation run. Re-render overlays under the RECOMMENDED config:
# --per-granule-prompts ALONE (sam_score selection, no size gate, no granule
# selection). The granule_fix/ unseen overlays used the worse size-gate config;
# these are the clean visuals matching the recommended config. GT metrics for the
# coco queries are recomputed too (cheap) for a clean reference.
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 105 60 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.45 \
    --per-granule-prompts --seed-source both --granule-min-distance 6 \
    --unseen-subdir March13 --n-unseen 2 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/per_granule_confirm

echo "[job] DONE"
