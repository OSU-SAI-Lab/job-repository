#!/usr/bin/env bash
#SBATCH --job-name=fss_rarea
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_rarea_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Calibrate the router's component-area gate: compare boxes / per_granule / router@
# {area_frac} on all 7 GT images, using the TUNED config (threshold 0.4, brightness
# seeds, min_dist 5; seed_z/box_scale/neg come from config.py). Returns per-query
# IoU so we can read clumped(q60) vs scattered and pick the area_frac that lets the
# router match boxes on q60 AND per_granule on scattered.
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 15 30 45 60 75 90 105 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.4 --seed-source brightness --granule-min-distance 5 \
    --compare-paths --router-area-fracs 0.02 0.04 0.06 0.10 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/router_area_eval
echo "[job] DONE"
