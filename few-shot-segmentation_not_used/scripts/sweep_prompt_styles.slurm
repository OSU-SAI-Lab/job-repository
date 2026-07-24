#!/usr/bin/env bash
#SBATCH --job-name=fss_styles
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_styles_%j.log
set -euo pipefail
REPO=/users/PAS2699/naveenkamath/job-repository
cd "${REPO}/few-shot-segmentation"
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Compare SAM3 prompting styles per regime (clumped pile q60 vs scattered rest) and
# report the best style for each, with per-(style,query) overlays.
python scripts/sweep_prompt_styles.py \
    --support-ids 0 --clumped-ids 60 --scattered-ids 15 30 45 75 90 105 \
    --dinov3 base --dtype bfloat16 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/prompt_styles
echo "[job] DONE"
