#!/usr/bin/env bash
#SBATCH --job-name=sam_diag
#SBATCH --account=PAS2699
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/users/PAS2699/naveenkamath/job-repository/few-shot-segmentation/sam3_dino_fss/sam_diag_%j.log
set -euo pipefail
cd /users/PAS2699/naveenkamath/job-repository/few-shot-segmentation
export HF_HOME=/fs/scratch/PAS2699/${USER}/hf_cache
export HF_HUB_OFFLINE=1
module load miniconda3/24.1.2-py310
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate fss
echo "[job] node=$(hostname)"
python sam3_dino_fss/diag_proposer.py
echo "[job] DONE"
