#!/usr/bin/env bash
# ============================================================================
# SLURM batch script: run fss on one H100 GPU on OSC Cardinal.
#
#   1) Edit the SUPPORT / QUERY / OUT paths below (or pass them as env vars).
#   2) sbatch osc/run_gpu.slurm
#      # or override at submit time:
#      sbatch --export=ALL,QUERY=q.png,OUT=out.png,SUPPORT="a.png:am.png b.png:bm.png" \
#             osc/run_gpu.slurm
#
# Logs land in fss_<jobid>.log. Use --partition=debug (1 h cap) for quick tests.
# ============================================================================
#SBATCH --job-name=fss
#SBATCH --account=PAS2699
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=fss_%j.log
set -euo pipefail

FSS_ENV_NAME="${FSS_ENV_NAME:-fss}"
export HF_HOME="${HF_HOME:-/fs/scratch/PAS2699/${USER}/hf_cache}"
# Weights are pre-cached on the login node; read them offline on the GPU node.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---- inputs (edit these, or pass via --export) -----------------------------
SUPPORT="${SUPPORT:-support1.png:mask1.png}"   # space-separated img:mask pairs
QUERY="${QUERY:-query.png}"
OUT="${OUT:-out.png}"
DINOV3="${DINOV3:-base}"                        # small | base | large | repo_id
DTYPE="${DTYPE:-bfloat16}"                      # bf16 is fast + stable on H100
DEBUG_OUT="${DEBUG_OUT:-debug.png}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                    # e.g. "--multi-instance --corr-4d"
# ----------------------------------------------------------------------------

module load miniconda3/24.1.2-py310
# shellcheck source=/dev/null
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate "${FSS_ENV_NAME}"

cd "${SLURM_SUBMIT_DIR:-.}"
echo "[run] node=$(hostname)  env=${FSS_ENV_NAME}  HF_HOME=${HF_HOME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086  # intentional word-splitting for multi-pair SUPPORT/EXTRA_ARGS
python -m fss \
    --support ${SUPPORT} \
    --query "${QUERY}" \
    --out "${OUT}" \
    --dinov3 "${DINOV3}" \
    --dtype "${DTYPE}" \
    --debug-out "${DEBUG_OUT}" \
    ${EXTRA_ARGS}

echo "[run] done -> ${OUT}"
