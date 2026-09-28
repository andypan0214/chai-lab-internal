# Chai-1 HPC inference

A minimal, GPU-only Chai-1 inference pipeline for a Linux HPC cluster with
Slurm and an NVIDIA GPU. No FastAPI, Docker, Redis, Celery, Kubernetes,
auth, or database -- just: build inputs -> `run_chai.py` -> Slurm -> outputs.

Everything here is verified against the official repository
(<https://github.com/chaidiscovery/chai-lab>, commit `66c38d1`,
`chai_lab.__version__ == "0.6.1"`) -- see the module docstrings in
`input_builder.py`, `restraint_builder.py`, and `run_chai.py` for exact
file/line citations of every claim made below.

## Contents

```
chai_hpc/
├── run_chai.py            # CLI wrapper around chai_lab.chai1.run_inference
├── input_builder.py       # UI molecule list -> Chai FASTA
├── restraint_builder.py   # UI restraint rows -> Chai restraint CSV
├── example_input.json     # (Step D+ / not yet created)
├── run_chai.slurm         # generic Slurm GPU job template
├── setup_env.sh           # builds the chai_env/ virtualenv
├── requirements.txt       # chai_lab==0.6.1 + pytest
├── README_HPC.md          # this file
└── tests/                 # non-GPU unit tests (pytest)
```

---

## 1. Prerequisites

- **Linux** HPC login + compute nodes (chai-lab does not support Windows/macOS).
- **NVIDIA GPU** on the compute nodes, with CUDA and **bfloat16** support.
  Per the official README: A100 80GB / H100 80GB / L40S 48GB recommended;
  A10 / A30 work for smaller complexes; RTX 4090 reported to work.
- **Python >= 3.10** available on the cluster (chai-lab `pyproject.toml`:
  `requires-python = ">=3.10"`).
- **Slurm** (`sbatch`, `squeue`, `scancel`) for GPU job submission.
- Outbound **HTTPS** from compute (or at least login) nodes, needed to:
  - download Chai-1's model weights + ESM embeddings on first use, and
  - reach the ColabFold MSA server if you use `--use-msa`.
  If compute nodes are network-isolated on your cluster, see
  "Troubleshooting: model download/network issues" below.
- A **persistent, shared filesystem path** (project/group scratch, not
  node-local `/tmp`) to cache downloaded model weights across jobs.

## 2. Cloning from GitHub

```bash
git clone https://github.com/andypan0214/chai-lab-internal.git   # private repo
cd chai-lab-internal
```

The repository is private: authenticate first (`gh auth login`, or an HTTPS
personal access token / SSH key with access to `andypan0214/chai-lab-internal`).

All commands below assume your current directory is the one containing
`run_chai.py`, `setup_env.sh`, and `requirements.txt` (i.e. `chai_hpc/`).

## 3. Environment setup

```bash
bash setup_env.sh
source chai_env/bin/activate
```

`setup_env.sh`:
- creates a `chai_env/` virtualenv (`python3 -m venv chai_env`),
- installs `requirements.txt` (`chai_lab==0.6.1` + `pytest`, which pulls in
  chai-lab's own transitive deps: torch, rdkit, gemmi, biopython, pandas,
  pandera, etc. -- see `requirements.txt` for details),
- **fails fast** (`set -euo pipefail`) on any installation error,
- prints the installed `chai_lab`/`torch` versions and `torch.cuda.is_available()`
  as a sanity check (expected `False` on a GPU-less login node -- that's fine).

If this cluster needs environment modules loaded first (a specific CUDA
module, a specific Python module, etc.), edit the `CHANGE_ME` block near the
top of `setup_env.sh` -- the script deliberately does not guess module names.

## 4. `CHAI_DOWNLOADS_DIR` setup

Chai-1 downloads its model weights, reference conformers, and ESM2
embedding checkpoint on first use (`chai_lab/utils/paths.py`). By default
they go to `<site-packages>/chai_lab/downloads`; set `CHAI_DOWNLOADS_DIR` to
a **persistent, shared** path instead so weights are downloaded once and
reused by every job, not re-downloaded per job:

```bash
export CHAI_DOWNLOADS_DIR=/path/to/CHANGE_ME/chai_models   # CHANGE_ME
mkdir -p "$CHAI_DOWNLOADS_DIR"
```

`run_chai.slurm` already exports this (with its own `CHANGE_ME` placeholder)
for every job -- edit it there once, and every subsequent job reuses the
same cache. Put this same `export` in your shell profile too if you want it
set for interactive sessions.

## 5. Building job inputs

`input_builder.py` / `restraint_builder.py` are libraries (Step B); there is
no generation CLI yet (that's the future `streamlit_app.py`, or a small
script you write). Until then, generate `job_input.fasta` /
`restraints.csv` with a short Python snippet using their real, verified API:

```python
from input_builder import build_chai_input, write_fasta_file
from restraint_builder import ContactRestraint, build_restraint_file

result = build_chai_input([
    {"molecule_type": "protein", "copies": 1, "chain_ids": ["A"], "sequence": "..."},
    {"molecule_type": "ligand", "copies": 1, "chain_ids": ["B"], "sequence": "CCO"},
])
write_fasta_file(result, "job_input.fasta")

# Optional restraints:
build_restraint_file(
    [ContactRestraint(chain_a="A", residue_a="C2", chain_b="B", residue_b="Y1", max_distance_angstrom=5.0)],
    result.chains_by_id,
    "restraints.csv",
)
```

Inspect `job_input.fasta` (and `restraints.csv`, if used) before submitting
-- both are plain text.

## 6. First CLI smoke test (no GPU required)

```bash
python run_chai.py --help
```

`run_chai.py` imports `torch`/`chai_lab` **lazily**, so `--help` and all
pre-flight validation (missing files, non-empty output dir, the
`--use-templates`/`--use-msa` dependency) work even before weights are
downloaded or a GPU is allocated. You can also run the non-GPU unit tests
the same way:

```bash
pytest tests/
```

## 7. First Slurm submission

```bash
mkdir -p logs   # required once, before the first sbatch (see note in
                # run_chai.slurm -- Slurm opens --output/--error before the
                # script body runs, so it can't create this directory itself)

sbatch run_chai.slurm
```

By default the job looks for `job_input.fasta` in the submission directory
and writes to `outputs/job_${SLURM_JOB_ID}`. Override via environment
variables passed to `sbatch`, e.g.:

```bash
FASTA_PATH=my_job.fasta RESTRAINTS_PATH=restraints.csv USE_MSA=1 SEED=42 \
    sbatch --export=ALL,FASTA_PATH,RESTRAINTS_PATH,USE_MSA,SEED run_chai.slurm
```

(`--use-templates`/`USE_TEMPLATES=1` requires `USE_MSA=1`.)

## 8. Monitoring / logs / cancellation

```bash
squeue -u $USER                    # job status
cat logs/chai_JOBID.out             # stdout: hostname, nvidia-smi, config, chai_lab logging
cat logs/chai_JOBID.err             # stderr: any tracebacks / failures
scancel JOBID                       # cancel a running/queued job
```

### Optional: interactive GPU test

To debug environment/CUDA problems directly instead of through a batch job
(exact flags depend on this cluster's Slurm configuration -- CHANGE_ME as
needed):

```bash
srun --gres=gpu:1 --pty bash   # CHANGE_ME: this cluster may need --partition/--account too
nvidia-smi
source chai_env/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 9. Expected output files

Per job, `run_chai.py` writes everything under its own `--output` directory
(`outputs/job_${SLURM_JOB_ID}` from the Slurm template), one sample set from
`chai_lab.chai1.run_folding_on_context` (Step A):

```
outputs/job_<id>/
├── pred.model_idx_0.cif       # predicted structure, sample 0 (pLDDT in B-factor column, 0-100 scale)
├── pred.model_idx_1.cif       # ... one per diffusion sample (5 by default)
├── ...
├── scores.model_idx_0.npz     # aggregate score, pTM, ipTM, pLDDT, clashes for sample 0
├── ...
└── msas/                      # only present if --use-msa was set
```

PAE/PDE are returned **in memory** on `run_inference()`'s return value
(`StructureCandidates.pae` / `.pde`), not written to disk by Chai itself --
`run_chai.py` does not currently save them (task scope: "do not convert or
postprocess result files yet unless needed for a minimal successful run").

Other generated/log locations:

| what | where |
|---|---|
| generated FASTA | wherever you called `write_fasta_file(...)` (e.g. `job_input.fasta`, submission dir) |
| generated restraint file | wherever you called `build_restraint_file(...)` (e.g. `restraints.csv`) |
| CIF predictions / scores | `<--output>/pred.model_idx_*.cif`, `<--output>/scores.model_idx_*.npz` |
| Slurm stdout/stderr | `logs/chai_<jobid>.out`, `logs/chai_<jobid>.err` |
| downloaded model weights | `$CHAI_DOWNLOADS_DIR` |

## 10. What has and has not been GPU-tested

**Not yet run on a GPU. Nothing below should be taken as "working" until it
has actually executed on this cluster's hardware.**

Verified so far, by reading the official source and by non-GPU unit tests
(`tests/`, 77 tests total across `input_builder.py`, `restraint_builder.py`,
and `run_chai.py`'s CLI/validation -- 4 of which round-trip generated
restraint CSVs through the real `chai_lab.data.parsing.restraints` parser;
those 4 auto-skip if `pandas`/`pandera` aren't installed, e.g. in a
pytest-only venv):

- FASTA/restraint generation matches chai-lab's actual parsers (Step B).
- `run_chai.py`'s argument parsing, file-existence checks, non-empty
  `--output` rejection, and `--use-templates`/`--use-msa` dependency check
  all behave as intended (Step C).
- `run_chai.py --help` and `pytest tests/` run correctly with **zero**
  GPU/torch/chai_lab dependencies present (proven by running them in a venv
  containing only `pytest`).

**Not yet verified (requires an actual HPC GPU run):**

- `chai_lab.chai1.run_inference()` has never actually been called. Model
  weight download via `CHAI_DOWNLOADS_DIR`, MSA-server networking, real CIF
  and score output, and the exact `assert not use_templates_server, "Server
  should have written a path"` failure path are all verified by reading the
  source only.
- `device=None -> "cuda:0"` default behavior on a real multi-GPU node.
- `fasta_names_as_cif_chains=True` producing correct output CIF chain
  labels end-to-end (verified via source + upstream `tests/test_restraints.py`,
  not via a live run through `run_chai.py`).
- Whether the default PyPI `torch` wheel's bundled CUDA build is compatible
  with this specific cluster's driver (see `setup_env.sh`'s CHANGE_ME note).
- Actual wall-clock time, memory usage, and whether the requested
  `--time=04:00:00` / `--mem=64G` / `--cpus-per-task=8` in `run_chai.slurm`
  are adequate for a real complex.

## 11. Troubleshooting

### CUDA unavailable

`run_chai.slurm` checks `torch.cuda.is_available()` and aborts the job
before inference if it's `False`. Causes to check, in order:
1. Did the job actually get a GPU? `#SBATCH --gres=gpu:1` must match this
   cluster's GPU request syntax (some clusters use `--gpus=1` or a
   partition-specific flag instead -- CHANGE_ME if so).
2. Is `chai_env` activated with the right torch build for this cluster's
   driver? See `setup_env.sh`'s CHANGE_ME note on CUDA-specific torch wheels.
3. Run `nvidia-smi` in an interactive GPU session (section 8 above) to
   confirm the node itself sees a GPU before blaming torch/chai_lab.

### Model download / network issues

Chai-1 downloads weights from `chaiassets.com` on first use
(`chai_lab/utils/paths.py`), and, if `--use-msa` is set, queries
`https://api.colabfold.com`. If compute nodes have no outbound internet:
- Run the **first** inference (which triggers the download) on a node/queue
  that does have internet access, with `CHAI_DOWNLOADS_DIR` pointed at the
  same persistent path compute jobs will use -- subsequent jobs reuse the
  cache and need no network access for weights (MSA server calls still need
  network if `--use-msa` is used).
- A partial/interrupted download can leave a lock file behind
  (`chai_lab/utils/paths.py` uses `FileLock` + a `.download_tmp_*` staging
  file); if a job hangs or fails oddly here, check
  `$CHAI_DOWNLOADS_DIR` for stray `.download_lock` / `.download_tmp_*`
  files from a previous interrupted run and remove them before retrying.

### Non-empty output directory

```
RunChaiError: --output directory is not empty: outputs/job_123
```
`run_chai.py` refuses to reuse or overwrite an existing non-empty
`--output` directory (matching `chai_lab.chai1.run_inference`'s own
requirement, Step A: `chai1.py:530-533`). Each Slurm job already gets a
unique `outputs/job_${SLURM_JOB_ID}` directory, so this should only happen
if you reused `--output` manually. Pick a new path, or remove the old
directory yourself if you intend to discard it -- `run_chai.py` will never
delete it for you.

### Restraint parser errors

If `restraint_builder.build_restraint_file(...)` raised a
`RestraintInputError` while building `restraints.csv`, the message
explains exactly which rule failed (unknown chain, wrong residue identity,
residue given on a pocket restraint's chain-level side, etc.) -- see
`restraint_builder.py`'s module docstring for the full list of enforced
rules and why each one exists. These are caught **before** `run_chai.py` is
even invoked, specifically because a bad restraint does not make Chai-1
itself fail: it is silently dropped inside a `try/except` in
`token_dist_restraint.py`/`token_pair_pocket_restraint.py` (Step A/B
finding), producing a normal-looking prediction that quietly ignored your
restraint. If you bypassed `restraint_builder.py` and hand-wrote a CSV, the
real `chai_lab.data.parsing.restraints.parse_pairwise_table` will raise a
`pandera.errors.SchemaError` for a malformed file at inference time instead
-- read that error's column/type message directly.

### Templates enabled without MSA

```
RunChaiError: --use-templates requires --use-msa. ...
```
`run_chai.py` enforces this before calling Chai (Step A/C:
`chai_lab/chai1.py:423-425` asserts `not use_templates_server` once a
protein is present and no MSA server/directory populated a templates path).
Add `--use-msa` (or `USE_MSA=1` in `run_chai.slurm`), or drop
`--use-templates`.

---

## Exact first-run sequence

```bash
git clone https://github.com/andypan0214/chai-lab-internal.git
cd chai-lab-internal
bash setup_env.sh
source chai_env/bin/activate
python run_chai.py --help
mkdir -p logs
sbatch run_chai.slurm
```
