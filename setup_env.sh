#!/usr/bin/env bash
# setup_env.sh -- build a Python virtual environment for Chai-1 HPC inference.
#
# Verified requirements (Step A, chai-lab commit 66c38d1 / chai_lab==0.6.1):
#   - Linux, Python >= 3.10 (chai-lab pyproject.toml: requires-python = ">=3.10")
#   - NVIDIA GPU with CUDA + bfloat16 support (README: A100 80GB / H100 80GB /
#     L40S 48GB recommended; A10/A30/RTX 4090 reported to work)
#   - chai_lab==0.6.1, the version pinned in chai_hpc/requirements.txt
#
# This script does NOT guess your cluster's module system, CUDA version, or
# Python interpreter path -- every cluster-specific value is marked
# CHANGE_ME below. It fails fast (set -e) on any installation error rather
# than continuing with a partially-broken environment.
#
# Usage (run from inside this chai_hpc/ directory):
#   bash setup_env.sh
#   source chai_env/bin/activate
#   python run_chai.py --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== setup_env.sh: Chai-1 HPC environment setup ==="
echo "Working directory: $SCRIPT_DIR"

# ---------------------------------------------------------------------------
# CHANGE_ME: cluster module system
# ---------------------------------------------------------------------------
# Uncomment and edit for your HPC's environment-module tool, e.g.:
#   module purge
#   module load cuda/12.1
#   module load python/3.11
# CHANGE_ME: load the CUDA/module configuration required by this HPC
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Python interpreter
# ---------------------------------------------------------------------------
# CHANGE_ME: point this at a Python >=3.10 interpreter on your cluster if the
# default `python3` on PATH resolves to something older (common on HPC login
# nodes). Override by exporting PYTHON_BIN before running this script, e.g.:
#   PYTHON_BIN=/opt/python3.11/bin/python3 bash setup_env.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter '$PYTHON_BIN' not found on PATH." >&2
    echo "       Set PYTHON_BIN, or load the CHANGE_ME module above." >&2
    exit 1
fi

echo "Using Python interpreter: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" --version

PY_OK=$("$PYTHON_BIN" -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
if [ "$PY_OK" != "1" ]; then
    echo "ERROR: chai_lab requires Python >= 3.10 (pyproject.toml: requires-python = \">=3.10\")." >&2
    echo "       '$PYTHON_BIN' is older than that. Set PYTHON_BIN to a newer interpreter." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="chai_env"

if [ -d "$VENV_DIR" ]; then
    echo "Reusing existing virtual environment: ./$VENV_DIR"
else
    echo "Creating virtual environment: ./$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip/setuptools/wheel ..."
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# CHANGE_ME (optional): CUDA-specific torch build
# ---------------------------------------------------------------------------
# chai_lab declares its own dependency as torch>=2.3.1 (any patch release
# 2.3.1-2.7.1 confirmed working per chai-lab's README); the default PyPI
# wheel bundles its own CUDA runtime and works on most modern drivers via
# forward compatibility. If this cluster's NVIDIA driver instead requires a
# specific CUDA build (check `nvidia-smi`'s reported "CUDA Version"),
# install torch explicitly BEFORE the requirements.txt install below, e.g.:
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# See https://pytorch.org/get-started/locally/ to pick the right
# --index-url for this cluster. Do not guess a CUDA build; verify it against
# `nvidia-smi` on this cluster's compute nodes first.
# ---------------------------------------------------------------------------

echo "Installing project requirements (requirements.txt) ..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "=== Verifying install ==="
python -c "import chai_lab; print('chai_lab version:', chai_lab.__version__)"
python -c "
import torch
print('torch version:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('NOTE: no CUDA device visible from this shell -- expected on a login')
    print('      node without a GPU allocation. This is fine; run_chai.slurm')
    print('      re-checks CUDA on the actual GPU compute node at job time.')
"

echo
echo "=== setup_env.sh complete ==="
echo "Next steps:"
echo "  source $VENV_DIR/bin/activate"
echo "  python run_chai.py --help"
echo
echo "Before your first real inference run, also set a persistent model"
echo "cache directory (see README_HPC.md, 'CHAI_DOWNLOADS_DIR setup'):"
echo "  export CHAI_DOWNLOADS_DIR=/path/to/CHANGE_ME/chai_models"
