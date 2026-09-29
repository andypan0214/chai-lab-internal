#!/bin/bash
# run_chai.slurm -- generic Slurm GPU job template for Chai-1 inference via
# run_chai.py (chai_lab.chai1.run_inference, see Step A/C).
#
# This is a TEMPLATE. No partition, account, or module name is invented --
# every cluster-specific value is marked CHANGE_ME. Submit with:
#   sbatch run_chai.slurm
#
# Before your FIRST submission, create the log directory once (Slurm needs
# the directory in --output/--error to already exist -- it will not create
# it for you, since it opens those files before this script body runs):
#   mkdir -p logs

#SBATCH --job-name=chai1
#SBATCH --output=logs/chai_%j.out
#SBATCH --error=logs/chai_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# CHANGE_ME if required by this cluster:
# #SBATCH --partition=...
# #SBATCH --account=...

# The header above is the GPU execution profile. The CPU profile submits
# run_chai_cpu.slurm instead, which has its own resource request and then
# runs this same body with CHAI_DEVICE=cpu (see execution.py).

set -euo pipefail

# Execution device, forwarded as run_inference(device=...). Only this and
# the GPU checks below differ between profiles.
CHAI_DEVICE="${CHAI_DEVICE:-cuda:0}"
case "$CHAI_DEVICE" in
    cuda:0|cpu) ;;
    *) echo "ERROR: CHAI_DEVICE must be cuda:0 or cpu (got '$CHAI_DEVICE')." >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 1. Move to the submission directory (where chai_env/, run_chai.py,
#    job_input.fasta, etc. live) and create logs/ defensively for any run
#    after the first (see note above about the very first submission).
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=== Chai-1 Slurm job ${SLURM_JOB_ID:-unknown} ==="
echo "Hostname:  $(hostname)"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Date:      $(date)"
echo "Device:    $CHAI_DEVICE"
echo

# ---------------------------------------------------------------------------
# CHANGE_ME: cluster module system (compute-node runtime environment)
# ---------------------------------------------------------------------------
# Uncomment and edit to match what setup_env.sh used to build chai_env, e.g.:
#   module purge
#   module load cuda/12.1
# CHANGE_ME: load the CUDA/module configuration required by this HPC
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. GPU sanity checks (fail fast, before spending time loading model weights)
#    -- GPU profile only
# ---------------------------------------------------------------------------
if [ "$CHAI_DEVICE" != "cpu" ]; then
    echo "=== nvidia-smi ==="
    nvidia-smi
    echo
fi

# ---------------------------------------------------------------------------
# 3. Activate the environment built by setup_env.sh
# ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-chai_env}"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: virtual environment not found at $VENV_DIR/bin/activate." >&2
    echo "       Run setup_env.sh first (see README_HPC.md)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "=== Python ==="
python --version
echo

# Project-local Kalign >= 3.3 built by setup_env.sh when the system kalign is
# missing or too old (needed by chai_lab only for USE_TEMPLATES=1).
if [ -x "$SLURM_SUBMIT_DIR/tools/kalign/bin/kalign" ]; then
    export PATH="$SLURM_SUBMIT_DIR/tools/kalign/bin:$PATH"
fi
echo "kalign: $(command -v kalign || echo '(not found -- only needed for USE_TEMPLATES=1)')"
echo

echo "=== torch / CUDA check ==="
CHAI_DEVICE="$CHAI_DEVICE" python -c "
import os, torch
print('torch version:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
if os.environ['CHAI_DEVICE'] == 'cpu':
    print('CPU profile: CUDA not required; torch threads:', torch.get_num_threads())
elif torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
else:
    raise SystemExit('CUDA not available on this node -- aborting before inference.')
"
echo

# ---------------------------------------------------------------------------
# 4. Model weight cache: point CHAI_DOWNLOADS_DIR at a PERSISTENT path
#    shared across jobs, so weights are downloaded once, not per job
#    (chai_lab/utils/paths.py: downloads_path defaults to CHAI_DOWNLOADS_DIR
#    if set, else <package_root>/downloads).
# ---------------------------------------------------------------------------
# CHANGE_ME: replace with a real, persistent path on this cluster (e.g.
# project/group scratch or a shared filesystem, NOT node-local /tmp, which
# is wiped between jobs and would force a re-download every run).
export CHAI_DOWNLOADS_DIR="${CHAI_DOWNLOADS_DIR:-/path/to/CHANGE_ME/chai_models}"
echo "CHAI_DOWNLOADS_DIR=$CHAI_DOWNLOADS_DIR"
mkdir -p "$CHAI_DOWNLOADS_DIR"
echo

# ---------------------------------------------------------------------------
# 5. Job inputs / output directory
# ---------------------------------------------------------------------------
# CHANGE_ME: point FASTA_PATH at the file generated by input_builder.py for
# the molecules you actually want to fold (see README_HPC.md). It defaults
# to job_input.fasta in the submission directory, matching input_builder.py's
# documented save location.
FASTA_PATH="${FASTA_PATH:-job_input.fasta}"

# Optional: set RESTRAINTS_PATH to a restraint_builder.py CSV to enable
# restraints; leave unset/empty to run without restraints.
RESTRAINTS_PATH="${RESTRAINTS_PATH:-}"

# Optional: SEED=<int>, USE_MSA=1, USE_TEMPLATES=1 (USE_TEMPLATES requires
# USE_MSA=1 -- see run_chai.py's own --use-templates/--use-msa check).
SEED="${SEED:-}"
USE_MSA="${USE_MSA:-0}"
USE_TEMPLATES="${USE_TEMPLATES:-0}"

# Optional precomputed inputs (no server calls; usable with CHAI_OFFLINE_MODE=1):
# MSA_DIRECTORY=<dir of *.aligned.pqt>, TEMPLATE_HITS_PATH=<hits .m8>.
MSA_DIRECTORY="${MSA_DIRECTORY:-}"
TEMPLATE_HITS_PATH="${TEMPLATE_HITS_PATH:-}"

# CHAI_OFFLINE_MODE=1 is read by run_chai.py itself (exported by sbatch's
# default --export=ALL): it refuses server MSA/templates and checks that all
# model files already exist under CHAI_DOWNLOADS_DIR before inference.
CHAI_OFFLINE_MODE="${CHAI_OFFLINE_MODE:-0}"
export CHAI_OFFLINE_MODE

# Each job gets its own isolated, never-reused output directory. The web app
# (job_manager.py) overrides OUTPUT_DIR with its own per-job directory.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/job_${SLURM_JOB_ID}}"

RUN_ARGS=(--fasta "$FASTA_PATH" --output "$OUTPUT_DIR" --device "$CHAI_DEVICE")
[ -n "$SEED" ] && RUN_ARGS+=(--seed "$SEED")
[ -n "$RESTRAINTS_PATH" ] && RUN_ARGS+=(--restraints "$RESTRAINTS_PATH")
[ "$USE_MSA" = "1" ] && RUN_ARGS+=(--use-msa)
[ "$USE_TEMPLATES" = "1" ] && RUN_ARGS+=(--use-templates)
[ -n "$MSA_DIRECTORY" ] && RUN_ARGS+=(--msa-directory "$MSA_DIRECTORY")
[ -n "$TEMPLATE_HITS_PATH" ] && RUN_ARGS+=(--template-hits "$TEMPLATE_HITS_PATH")

echo "=== Configuration ==="
echo "FASTA_PATH:      $FASTA_PATH"
echo "RESTRAINTS_PATH: ${RESTRAINTS_PATH:-(none)}"
echo "SEED:            ${SEED:-(none)}"
echo "USE_MSA:         $USE_MSA"
echo "USE_TEMPLATES:   $USE_TEMPLATES"
echo "MSA_DIRECTORY:   ${MSA_DIRECTORY:-(none)}"
echo "TEMPLATE_HITS:   ${TEMPLATE_HITS_PATH:-(none)}"
echo "OFFLINE_MODE:    $CHAI_OFFLINE_MODE"
echo "DEVICE:          $CHAI_DEVICE"
echo "OUTPUT_DIR:      $OUTPUT_DIR"
echo "run_chai.py args: ${RUN_ARGS[*]}"
echo

# ---------------------------------------------------------------------------
# 6. Run inference. No output redirection here beyond what #SBATCH --output
#    / --error already set up, so run_chai.py's own stdout/stderr (including
#    chai_lab's internal logging) land in logs/chai_${SLURM_JOB_ID}.out/.err
#    unmodified. set -e above means a non-zero exit from run_chai.py fails
#    this job (visible in `sacct`/`squeue`) rather than being swallowed.
# ---------------------------------------------------------------------------
echo "=== Running run_chai.py ==="
python run_chai.py "${RUN_ARGS[@]}"

echo
echo "=== Job complete ==="
echo "Outputs: $OUTPUT_DIR"
