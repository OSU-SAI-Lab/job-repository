#!/usr/bin/env bash
#SBATCH --job-name=fss_fert_mi
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/scripts/fss_fert_mi_%j.log
set -euo pipefail
cd /users/PAS2699/naveenkamath/job-repository/few-shot-segmentation
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
python scripts/run_fertilizer_test.py \
    --mode crop --crop-size 1024 \
    --support-ids 0 --query-ids 105 60 \
    --dinov3 base --dtype bfloat16 \
    --multi-instance --threshold 0.45 \
    --unseen-subdir March13 --n-unseen 0 \
    --out-dir /fs/ess/PAS2699/kamath/fertilizer_data/fss_results/multi_instance
echo "[job] DONE"
