#!/usr/bin/env bash
#SBATCH --job-name=fss_thr06
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_thr06_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Full pipeline with the NEW config thresholds: prior fg threshold 0.6 (passed
# explicitly because the runner's --threshold default is 0.5) and neg_threshold 0.4
# (no flag → taken from config.py). compare-paths runs boxes/per_granule/router on
# all 7 GT images + 2 unseen, into a fresh results folder.
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 15 30 45 60 75 90 105 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.6 \
    --compare-paths \
    --router-density-metric seed_coverage --granule-min-distance 6 \
    --unseen-subdir March13 --n-unseen 2 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/thr06_eval
echo "[job] DONE"
