#!/usr/bin/env bash
#SBATCH --job-name=fss_align
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_align_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Measure whether the reverse-IoU alignment signal can SELECT the best path per
# image (accuracy) or is only telemetry. compare-paths over all 7 GT images with
# --alignment-check prints, per path/query, reverse-IoU alongside forward-IoU, then
# an alignment-selector vs per_granule vs oracle mIoU and corr(reverse,forward).
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 15 30 45 60 75 90 105 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.45 \
    --compare-paths --alignment-check \
    --router-density-metric seed_coverage --granule-min-distance 6 \
    --unseen-subdir March13 --n-unseen 2 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/alignment_eval
echo "[job] DONE"
