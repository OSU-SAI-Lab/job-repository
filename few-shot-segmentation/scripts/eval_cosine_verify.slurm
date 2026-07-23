#!/usr/bin/env bash
#SBATCH --job-name=fss_cosver
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_cosver_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# DINOv3 cosine-verification of SAM masks: sweep the keep-threshold on the 7 GT
# images, report mIoU/precision/recall/keep%, save baseline-vs-verified overlays.
python scripts/eval_cosine_verify.py \
    --support-ids 0 --query-ids 15 30 45 60 75 90 105 \
    --dinov3 base --dtype bfloat16 \
    --out-dir /users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/fss/results/Cosine_Verify_Eval
echo "[job] DONE"
