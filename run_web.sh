#!/usr/bin/env bash
# run_web.sh -- start the Chai-1 Streamlit web app (see README_HPC.md, "Web app").
#
# Run on a machine that can call sbatch/squeue/sacct (normally the HPC login
# node), from this chai_hpc/ directory, in a Python environment with
# requirements-web.txt installed. Binds to 127.0.0.1:8501 by default
# (.streamlit/config.toml); reach it through an SSH tunnel or the lab's
# reverse proxy. Extra arguments are passed to `streamlit run`, e.g.
#   bash run_web.sh --server.port 8600
#
# Optional environment variables:
#   CHAI_WEB_JOBS_DIR       where web jobs are stored (default ./web_jobs)
#   CHAI_WEB_SBATCH_ARGS    extra sbatch options, e.g. "--partition=gpu --account=lab"
#   CHAI_WEB_SHOW_CANDIDATE_PRESETS=1   also show *_candidate.csv example presets

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "${CHAI_WEB_JOBS_DIR:-web_jobs}"
exec python -m streamlit run streamlit_app.py "$@"
