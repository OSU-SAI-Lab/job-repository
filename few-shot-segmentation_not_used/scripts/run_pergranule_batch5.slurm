#!/usr/bin/env bash
#SBATCH --job-name=fss_pg_b5
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_pg_b5_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Fixed --per-granule-prompts method (NO --router) on the SAME 6 unseen images that
# the adaptive method ran on (now in fss/results/Batch_4_Adaptive_SAM3): March13 +
# March14 first-3 each. Outputs to a parallel Batch_5_PerGranule_SAM3 folder for
# side-by-side comparison.
python scripts/run_unseen_visuals.py \
    --support-id 0 --folders March13 March14 --start-index 0 --n-per-folder 3 \
    --dinov3 base --dtype bfloat16 \
    --out-dir /users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/fss/results/Batch_5_PerGranule_SAM3
echo "[job] DONE"
