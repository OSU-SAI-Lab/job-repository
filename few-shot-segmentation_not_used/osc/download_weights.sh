#!/usr/bin/env bash
# ============================================================================
# Pre-download DINOv3 + SAM 3 weights into a shared HF cache on SCRATCH.
#
# Run on a LOGIN node (internet access). Compute/GPU nodes then read the cache
# offline. facebook/dinov3-* and facebook/sam3 are gated — you must first:
#     huggingface-cli login
# and click "Agree and access" on each model's Hugging Face page.
#
#   bash osc/download_weights.sh                  # defaults: base DINOv3 + sam3
#   bash osc/download_weights.sh facebook/dinov3-vitl16-pretrain-lvd1689m
# ============================================================================
set -euo pipefail

FSS_ENV_NAME="${FSS_ENV_NAME:-fss}"
export HF_HOME="${HF_HOME:-/fs/scratch/PAS2699/${USER}/hf_cache}"
mkdir -p "${HF_HOME}"
echo "[weights] HF_HOME=${HF_HOME}"

DINOV3="${1:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
SAM3="${2:-facebook/sam3}"

module load miniconda3/24.1.2-py310
# shellcheck source=/dev/null
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
conda activate "${FSS_ENV_NAME}"

python - "${DINOV3}" "${SAM3}" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print(f"[weights] downloading {repo} ...")
    path = snapshot_download(repo_id=repo)
    print(f"[weights]   -> {path}")
print(f"[weights] HF cache ready at {os.environ['HF_HOME']}")
PY
