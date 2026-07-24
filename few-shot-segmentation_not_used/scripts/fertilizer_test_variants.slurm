#!/usr/bin/env bash
#SBATCH --job-name=fss_fert_gran
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_fert_gran_%j.log
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

# Per-granule regression fix: per-seed prompts, granule (compact) selection, and a
# granule size gate. --eval-variants saves prior_only / sam3 / granule_sam3 overlays,
# masks, and per-variant precision/recall/IoU/bgFP into fss_results/granule_fix
# (baseline images in the parent dir are left untouched).
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 105 60 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.45 \
    --eval-variants \
    --per-granule-prompts --seed-source both --granule-min-distance 6 \
    --selection granule --size-gate --granule-size-mult 4.0 \
    --intersect-prior-safety \
    --prior-only-threshold 0.45 \
    --unseen-subdir March13 --n-unseen 2 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/granule_fix

echo "[job] DONE"
