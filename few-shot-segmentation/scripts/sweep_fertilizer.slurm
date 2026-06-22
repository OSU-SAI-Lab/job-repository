#!/usr/bin/env bash
#SBATCH --job-name=fss_sweep
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_sweep_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Coordinate-descent sweep of per-granule knobs over the 7 GT images to maximise
# mean IoU. One model load; reuses support prototypes; each eval flushes so partial
# progress survives a timeout.
python scripts/sweep_fertilizer.py \
    --support-ids 0 --query-ids 15 30 45 60 75 90 105 \
    --dinov3 base --dtype bfloat16 --passes 2 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/sweep_r2
echo "[job] DONE"
