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

# ---------------------------------------------------------------------------
# Kalign >= 3.3 (only needed for run_chai.py --use-templates)
# ---------------------------------------------------------------------------
# chai_lab/tools/kalign.py runs `kalign -i ... -o ...` from PATH and asserts
# "You need kalign>=3.3". Use the system kalign if it is new enough;
# otherwise build a project-local copy into tools/kalign/bin/kalign, which
# run_chai.slurm prepends to PATH. Building needs curl (or wget), a C/C++
# compiler, and cmake >= 3.18 (installed into chai_env via pip if missing or
# too old). Override the built release with KALIGN_VERSION=x.y.z.
KALIGN_MIN_VERSION="3.3"
KALIGN_VERSION="${KALIGN_VERSION:-3.4.0}"
KALIGN_PREFIX="$SCRIPT_DIR/tools/kalign"

# Prints the X.Y[.Z] version reported by the given kalign binary, or nothing.
# (Kalign 3.x prints "Kalign (3.4.0)"; Kalign 2.x prints "Kalign version 2.04".)
kalign_version() {
    timeout 30 "$1" --version </dev/null 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true
}

# Succeeds if version $1 >= version $2.
version_ge() {
    [ -n "$1" ] && [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

echo
echo "=== Kalign (>= $KALIGN_MIN_VERSION, required for --use-templates) ==="
SYSTEM_KALIGN="$(command -v kalign || true)"
SYSTEM_KALIGN_VERSION=""
[ -n "$SYSTEM_KALIGN" ] && SYSTEM_KALIGN_VERSION="$(kalign_version "$SYSTEM_KALIGN")"

if version_ge "$SYSTEM_KALIGN_VERSION" "$KALIGN_MIN_VERSION"; then
    echo "Using system kalign: $SYSTEM_KALIGN (version $SYSTEM_KALIGN_VERSION)"
elif [ -x "$KALIGN_PREFIX/bin/kalign" ] \
        && version_ge "$(kalign_version "$KALIGN_PREFIX/bin/kalign")" "$KALIGN_MIN_VERSION"; then
    echo "Reusing project-local kalign: $KALIGN_PREFIX/bin/kalign (version $(kalign_version "$KALIGN_PREFIX/bin/kalign"))"
else
    if [ -n "$SYSTEM_KALIGN" ]; then
        echo "System kalign $SYSTEM_KALIGN reports version '${SYSTEM_KALIGN_VERSION:-unknown}' (< $KALIGN_MIN_VERSION)."
    else
        echo "No kalign found on PATH."
    fi
    echo "Building project-local kalign $KALIGN_VERSION into $KALIGN_PREFIX ..."

    if ! command -v cc >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1; then
        echo "ERROR: building kalign needs a C and C++ compiler (cc/c++ not found)." >&2
        echo "       Load a compiler module (CHANGE_ME block above) or install kalign >= $KALIGN_MIN_VERSION system-wide." >&2
        exit 1
    fi
    CMAKE_VERSION_FOUND=""
    command -v cmake >/dev/null 2>&1 \
        && CMAKE_VERSION_FOUND="$(cmake --version | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
    if ! version_ge "$CMAKE_VERSION_FOUND" "3.18"; then
        echo "cmake >= 3.18 not found (found: '${CMAKE_VERSION_FOUND:-none}'); installing cmake into $VENV_DIR via pip ..."
        pip install "cmake>=3.18"
        hash -r
    fi

    rm -rf "$KALIGN_PREFIX"
    mkdir -p "$KALIGN_PREFIX/src" "$KALIGN_PREFIX/bin"
    KALIGN_URL="https://github.com/TimoLassmann/kalign/archive/refs/tags/v${KALIGN_VERSION}.tar.gz"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$KALIGN_URL" -o "$KALIGN_PREFIX/kalign.tar.gz"
    else
        wget -q "$KALIGN_URL" -O "$KALIGN_PREFIX/kalign.tar.gz"
    fi
    tar -xzf "$KALIGN_PREFIX/kalign.tar.gz" -C "$KALIGN_PREFIX/src" --strip-components=1

    # Static build (no libkalign.so to find at runtime), and no AVX/AVX2 so a
    # binary built on a login node also runs on older compute-node CPUs.
    cmake -S "$KALIGN_PREFIX/src" -B "$KALIGN_PREFIX/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DENABLE_AVX=OFF \
        -DENABLE_AVX2=OFF
    cmake --build "$KALIGN_PREFIX/build" --target kalign-bin -j "${KALIGN_BUILD_JOBS:-4}"

    KALIGN_BUILT="$(find "$KALIGN_PREFIX/build" -type f -name kalign -perm -u+x | head -n1)"
    if [ -z "$KALIGN_BUILT" ]; then
        echo "ERROR: kalign build finished but no kalign binary was found under $KALIGN_PREFIX/build." >&2
        exit 1
    fi
    install -m 755 "$KALIGN_BUILT" "$KALIGN_PREFIX/bin/kalign"

    LOCAL_KALIGN_VERSION="$(kalign_version "$KALIGN_PREFIX/bin/kalign")"
    if ! version_ge "$LOCAL_KALIGN_VERSION" "$KALIGN_MIN_VERSION"; then
        echo "ERROR: built kalign reports version '${LOCAL_KALIGN_VERSION:-unknown}', expected >= $KALIGN_MIN_VERSION." >&2
        exit 1
    fi
    echo "Built project-local kalign: $KALIGN_PREFIX/bin/kalign (version $LOCAL_KALIGN_VERSION)"
fi

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
if [ -x "$KALIGN_PREFIX/bin/kalign" ]; then
    echo "For interactive --use-templates runs, also put the local kalign on PATH"
    echo "(run_chai.slurm does this automatically):"
    echo "  export PATH=\"$KALIGN_PREFIX/bin:\$PATH\""
fi
echo
echo "Before your first real inference run, also set a persistent model"
echo "cache directory (see README_HPC.md, 'CHAI_DOWNLOADS_DIR setup'):"
echo "  export CHAI_DOWNLOADS_DIR=/path/to/CHANGE_ME/chai_models"
