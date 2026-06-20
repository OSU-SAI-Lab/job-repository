#!/usr/bin/env bash
# ============================================================================
# One-time environment setup for `fss` on OSC Cardinal (NVIDIA H100, 80 GB).
#
# Creates a conda env with a CUDA build of PyTorch + transformers>=5.12 (SAM 3)
# and installs the `fss` package in editable mode. Run this ONCE on a LOGIN node
# (login nodes have internet; compute nodes may not).
#
#   bash osc/setup_env.sh
#
# Override defaults via env vars, e.g.:  FSS_ENV_NAME=myenv bash osc/setup_env.sh
# ============================================================================
set -euo pipefail

FSS_ENV_NAME="${FSS_ENV_NAME:-fss}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"          # cu121 / cu124 — H100 (sm_90) supported by both
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] repo:        ${REPO_DIR}"
echo "[setup] conda env:   ${FSS_ENV_NAME} (python ${PYTHON_VERSION})"
echo "[setup] torch wheel: ${TORCH_CUDA}"

module load miniconda3/24.1.2-py310
# shellcheck source=/dev/null
source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${FSS_ENV_NAME}"; then
    echo "[setup] creating conda env..."
    conda create -y -n "${FSS_ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${FSS_ENV_NAME}"

python -m pip install --upgrade pip

# CUDA build of PyTorch + torchvision (torchvision is required by the HF image
# processors). Self-contained wheels — they bundle the CUDA runtime.
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# fss + the rest of its pinned deps (transformers>=5.12, scipy, matplotlib, ...).
python -m pip install -e "${REPO_DIR}[test]"

echo "[setup] verifying install..."
python - <<'PY'
import torch, transformers
print("  torch        :", torch.__version__, "| CUDA build:", torch.version.cuda)
print("  transformers :", transformers.__version__)
try:
    from transformers import Sam3TrackerModel, Sam3Model  # noqa: F401
    print("  SAM 3 classes: OK")
except Exception as e:
    print("  SAM 3 classes: MISSING ->", e, "\n  (need transformers>=5.12.0)")
PY

echo
echo "[setup] done. Next:"
echo "  1) huggingface-cli login        # accept terms on the facebook/sam3 + dinov3 pages"
echo "  2) bash osc/download_weights.sh  # pre-cache weights on the login node"
echo "  3) sbatch osc/run_gpu.slurm      # (after editing the SUPPORT/QUERY paths)"
