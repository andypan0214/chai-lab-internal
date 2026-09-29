#!/bin/bash
# run_chai_cpu.slurm -- CPU execution profile for Chai-1 inference.
#
# Same job as run_chai.slurm (same inputs, same run_chai.py call); only the
# resource request below and run_inference(device="cpu") differ. Submit with
#   sbatch run_chai_cpu.slurm
# from chai_hpc/, with the same FASTA_PATH / RESTRAINTS_PATH / USE_MSA / ...
# variables as run_chai.slurm. The web app uses it when the server runs with
# CHAI_EXECUTION_PROFILE=cpu (see execution.py).
#
# NOT YET VALIDATED: the time/CPU/memory values are unmeasured starting
# points. CHANGE_ME after the first CPU test run on this cluster.

#SBATCH --job-name=chai1-cpu
#SBATCH --output=logs/chai_%j.out
#SBATCH --error=logs/chai_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# CHANGE_ME if required by this cluster:
# #SBATCH --partition=...
# #SBATCH --account=...

set -euo pipefail

export CHAI_DEVICE=cpu
# Keep torch's thread pool within the CPUs Slurm allocated.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# Slurm runs a spooled copy of this script, so locate the shared body via
# the submission directory (chai_hpc/). Its #SBATCH lines are plain comments
# when run with bash, so they do not change this job's allocation.
exec bash "$SLURM_SUBMIT_DIR/run_chai.slurm"
