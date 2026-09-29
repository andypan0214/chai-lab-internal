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
├── run_chai.slurm         # generic Slurm GPU job template (GPU execution profile)
├── run_chai_cpu.slurm     # CPU execution profile: CPU resources, same job body (section 14)
├── execution.py           # execution profiles gpu/cpu (CHAI_EXECUTION_PROFILE)
├── setup_env.sh           # builds the chai_env/ virtualenv
├── requirements.txt       # chai_lab==0.6.1 + pytest
├── streamlit_app.py       # web frontend (section 12)
├── form_state.py          #   web form state <-> job config (validates via the builders)
├── presets.py             #   CSV example presets loader
├── job_manager.py         #   web job registry + sbatch submission/status
├── result_adapter.py      #   reads Chai CIF / scores / PAE outputs
├── examples/presets/      #   example preset CSVs (+ README with sources)
├── requirements-web.txt   #   streamlit, plotly, numpy, pandas
├── run_web.sh             #   starts the web app
├── .streamlit/config.toml #   binds to 127.0.0.1:8501, light theme
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
- ensures **Kalign >= 3.3** is available (needed only for `--use-templates`;
  `chai_lab/tools/kalign.py` runs `kalign` from `PATH`):
  - if the system `kalign` reports a version >= 3.3, it is used as-is;
  - otherwise Kalign `v3.4.0` (override with `KALIGN_VERSION=x.y.z`) is
    downloaded from GitHub and built into the project-local
    `tools/kalign/bin/kalign` (static, no AVX/AVX2, so a binary built on the
    login node also runs on compute nodes). This needs `curl`/`wget`, a
    C/C++ compiler (`cc`/`c++`), and `cmake >= 3.18` -- if cmake is missing
    or too old, it is `pip install`ed into `chai_env`. An existing adequate
    `tools/kalign/bin/kalign` is reused on re-runs; `tools/kalign/` is git-ignored.
  - `run_chai.slurm` prepends `tools/kalign/bin` to `PATH` automatically. For
    interactive runs, do it yourself: `export PATH="$PWD/tools/kalign/bin:$PATH"`.
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

(`--use-templates`/`USE_TEMPLATES=1` requires `USE_MSA=1` and Kalign >= 3.3,
which `setup_env.sh` sets up -- see section 3.)

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
├── confidence.model_idx_0.npz # pae (N,N), pde (N,N), plddt (N,) for sample 0 -- saved by run_chai.py
├── ...
└── msas/                      # only present if --use-msa was set
```

PAE/PDE/per-token pLDDT are returned **in memory** on `run_inference()`'s
return value (`StructureCandidates.pae` / `.pde` / `.plddt`), not written to
disk by Chai itself. `run_chai.py` saves candidate *i*'s arrays unchanged to
`confidence.model_idx_<i>.npz` next to `pred.model_idx_<i>.cif`, so the web
app can show the PAE heatmap later.

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
(`tests/`, 142 tests total across `input_builder.py`, `restraint_builder.py`,
the web modules (`tests/test_web.py`), offline mode (`tests/test_offline.py`),
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

### Kalign missing or outdated

```
RunChaiError: --use-templates requires kalign >= 3.3 ...
```
`run_chai.py` checks `kalign --version` before calling Chai whenever
`--use-templates` is set (chai_lab itself would only fail mid-inference).
Re-run `bash setup_env.sh` to build `tools/kalign/bin/kalign`, and for
interactive runs `export PATH="$PWD/tools/kalign/bin:$PATH"`. If the build
fails, check that `cc`/`c++` are available (load a compiler module) and that
the login node can reach `github.com`.

## 12. Web app (Streamlit)

Lab members only need a browser: they fill in molecules/restraints (or load
an example), press **Predict 3D structure**, watch the job on
**Predictions**, and open the result page (ipTM/pTM, pLDDT-coloured 3D
structure, PAE heatmap, input tables, downloads).

### Setup (admin, once)

On a machine that can run `sbatch`/`squeue`/`sacct` (normally the login
node), in `chai_hpc/`, after `setup_env.sh` has built `chai_env`:

```bash
source chai_env/bin/activate            # or a separate light venv: the app
pip install -r requirements-web.txt     # never imports torch/chai_lab
bash run_web.sh                         # http://127.0.0.1:8501
```

`.streamlit/config.toml` binds to `127.0.0.1` only. Users reach it through an
SSH tunnel (`ssh -L 8501:127.0.0.1:8501 <login-node>`, then open
<http://localhost:8501>) or a lab reverse proxy; pass
`--server.address 0.0.0.0` only if the cluster's network policy allows it.
There is no login: everyone who can reach the URL sees all predictions.

If `run_chai.slurm` needs `--partition`/`--account` on this cluster, either
edit its `#SBATCH` CHANGE_ME lines or set
`CHAI_WEB_SBATCH_ARGS="--partition=... --account=..."` before `run_web.sh`.
Whether jobs run on a GPU or a CPU is also a server setting, not a form
field: see section 14.

### How a web job runs

1. The form is validated by `input_builder.build_chai_input` and
   `restraint_builder.validate_restraints` (the same rules as the CLI path;
   templates require MSAs, pocket restraints never take a residue on the
   whole-chain side).
2. `job_manager.py` creates `web_jobs/<job_id>/` (override with
   `CHAI_WEB_JOBS_DIR`) holding `config.json`, `input.fasta`,
   `restraints.csv`, `job.json` (status) and, later, `output/` and
   `slurm.out`/`slurm.err`.
3. It runs `sbatch run_chai.slurm` from `chai_hpc/` with `FASTA_PATH`,
   `RESTRAINTS_PATH`, `USE_MSA`, `USE_TEMPLATES` and `OUTPUT_DIR` set, i.e.
   the unchanged CLI/Slurm path.
4. Status (QUEUED / RUNNING / SUCCESS / FAILED) comes from `squeue`/`sacct`
   plus a check that Chai's CIF + scores files exist.
5. The result page reads Chai's own files through `result_adapter.py`:
   candidates ranked by `aggregate_score` (best first, switchable), ipTM/pTM
   from `scores.model_idx_*.npz`, structure coloured by the per-atom pLDDT
   Chai writes into the CIF B-factor column, PAE from
   `confidence.model_idx_*.npz` (colour range 0 to the matrix maximum).

**The browser needs no internet access.** The 3D viewer uses a vendored
copy of 3Dmol.js 2.5.5 (`vendor/3Dmol/3Dmol-min.js`, BSD-3-Clause, hash in
`vendor/3Dmol/README.md`), inlined into the viewer's iframe by
`streamlit_app.py`. The Streamlit UI, its fonts/icons and the Plotly PAE
heatmap are bundled with the `streamlit`/`plotly` Python packages and served
by the app itself. See section 13 to verify this.

### Example presets

See `examples/presets/README.md`. Only the compact 3-chain restraints demo is
shown to users; the small MSA + template demo stays hidden
(`small_msa_template_candidate.csv`) until it has passed a real HPC run.
Start the app with `CHAI_WEB_SHOW_CANDIDATE_PRESETS=1` to submit it for that
validation, then rename it to `small_msa_template.csv`.

## 13. Offline mode

The deployment is offline-first on two levels.

**Browser (always, no setting needed).** Everything a lab member's browser
loads comes from the Streamlit server: UI, molecule input, presets,
contact/pocket restraints, prediction history, result pages, ipTM/pTM, the
PAE heatmap, the interactive pLDDT-coloured 3D structure and downloads. No
CDN or other external host is contacted.

**Server + HPC jobs: `CHAI_OFFLINE_MODE=1`.** Set it in the environment of
the web server (`CHAI_OFFLINE_MODE=1 bash run_web.sh`) and/or of CLI/Slurm
jobs. Web jobs inherit it through `sbatch` (default `--export=ALL`). With it
set:

- MSA and template **server** search (`--use-msa`, `--use-templates`; the
  web "Use MSAs"/"Use Templates" boxes) is refused with a clear message.
  Nothing falls back to the internet.
- `run_chai.py` checks, before inference, that every file chai_lab would
  otherwise download already exists (`offline.py`), and fails with an
  administrator-facing list of what is missing instead of downloading:

  ```
  $CHAI_DOWNLOADS_DIR/models_v2/feature_embedding.pt
  $CHAI_DOWNLOADS_DIR/models_v2/bond_loss_input_proj.pt
  $CHAI_DOWNLOADS_DIR/models_v2/token_embedder.pt
  $CHAI_DOWNLOADS_DIR/models_v2/trunk.pt
  $CHAI_DOWNLOADS_DIR/models_v2/diffusion_module.pt
  $CHAI_DOWNLOADS_DIR/models_v2/confidence_head.pt
  $CHAI_DOWNLOADS_DIR/conformers_v1.apkl
  $CHAI_DOWNLOADS_DIR/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt
  ```

  (chai_lab only downloads a file when it is absent, so this guarantees no
  download.) `CHAI_DOWNLOADS_DIR` must be set explicitly, on a filesystem the
  GPU nodes can read. The web app runs the same check before submitting and
  tells users to contact the administrator if files are missing (details go
  to the web server log only).
- Unset / `0`: online behaviour exactly as before (server MSA/templates,
  on-demand model download).

**Populating `CHAI_DOWNLOADS_DIR` once.** On any machine with internet
access that writes to the same shared path, run one normal (online)
prediction with `CHAI_DOWNLOADS_DIR` exported; chai_lab downloads all files
above on first use. Or copy an existing, complete downloads directory. Model
weights are never committed to Git.

**MSAs / templates offline (precomputed data, CLI/Slurm only).** No local
ColabFold/MMseqs database is provided. Precomputed inputs can be passed
straight to chai_lab:

| input | run_chai.py | run_chai.slurm | format |
|---|---|---|---|
| MSAs | `--msa-directory DIR` | `MSA_DIRECTORY=DIR` | `<sha256(uppercase sequence)>.aligned.pqt` per protein sequence (chai_lab `expected_basename`); e.g. the `msas/` folder of an earlier online run |
| template hits | `--template-hits FILE.m8` | `TEMPLATE_HITS_PATH=FILE.m8` | tab-separated m8, query ID = chain ID; offline, every hit's `<PDBID>.cif.gz` must already be in `$CHAI_TEMPLATE_CIF_FOLDER` (default `$CHAI_DOWNLOADS_DIR/template_cifs`), otherwise the preflight lists the missing IDs |

They cannot be combined with the matching server flag. Template hits need
Kalign >= 3.3 like `--use-templates` (handled by `setup_env.sh`). The web
form does not accept precomputed MSAs/templates; in offline mode it shows
that MSAs/templates are unavailable.

**Still online-only:** server MSA search and server template search
(ColabFold), downloading missing model files or template structures, and
installation itself (`setup_env.sh` needs PyPI and, when building Kalign,
github.com).

**Verifying the browser needs no internet.** With the app running, open a
result page with the browser's developer tools open (Network tab): every
request goes to the app's own host, and there is no request to cdnjs,
unpkg, jsdelivr or any other host. Or block outbound traffic for the
browser (e.g. disconnect the client from the internet while keeping the SSH
tunnel) and reload a result page: the structure still renders and can be
rotated (left-drag), zoomed (scroll) and translated (right- or middle-drag).
`tests/test_offline.py` also asserts the app has no external `<script src>`
and that the vendored file matches its recorded hash.

## 14. Execution profiles (GPU / CPU)

Where a job runs is a deployment decision, made by the administrator in the
web server's environment. Lab members never see it: the form, the submitted
job and the result pages are the same for both profiles.

| `CHAI_EXECUTION_PROFILE` | Slurm script         | `run_inference(device=...)` | extra sbatch options       |
|--------------------------|----------------------|-----------------------------|----------------------------|
| `gpu` (default)          | `run_chai.slurm`     | `"cuda:0"`                  | `CHAI_WEB_GPU_SBATCH_ARGS` |
| `cpu`                    | `run_chai_cpu.slurm` | `"cpu"`                     | `CHAI_WEB_CPU_SBATCH_ARGS` |

```bash
CHAI_EXECUTION_PROFILE=cpu CHAI_WEB_CPU_SBATCH_ARGS="--partition=..." bash run_web.sh
```

`CHAI_WEB_SBATCH_ARGS` still applies to both. An unknown profile name makes
every submission fail with a "contact the administrator" message; nothing
is submitted. There is **no automatic fallback** from GPU to CPU.

What changes between profiles: only the Slurm resource request and the
`device` argument. `run_chai_cpu.slurm` sets `CHAI_DEVICE=cpu` (and
`OMP_NUM_THREADS` to the allocated CPUs) and runs the same `run_chai.slurm`
body, which skips `nvidia-smi` / the CUDA check for `cpu` and passes
`--device "$CHAI_DEVICE"` to `run_chai.py`. The FASTA, restraint CSV,
MSA/template settings, output files, ranking (`aggregate_score`), PAE,
pLDDT and scores come from the same `run_inference()` call and the same
readers. `device` is an existing `run_inference()` parameter; chai_lab
loads its model components and ESM on CPU for any device other than
`cuda:0` (`chai_lab/chai1.py:143`, `data/dataset/embeddings/esm.py:38`).

Each web job records its profile in `web_jobs/<job_id>/job.json`
(`execution_profile`); `slurm.out` prints `Device:`. On the CLI:
`sbatch run_chai_cpu.slurm` (same variables as `run_chai.slurm`) or
`python run_chai.py ... --device cpu`.

Expect the same scientific result, not bit-identical numbers: CPU and GPU
kernels (and different GPU models) round floating-point differently, so
coordinates and scores may differ in the last digits, and with a fixed seed
the sampled candidates can differ slightly.

**Not yet validated:** no CPU run has been made. Whether a full CPU run
completes (the ESM model is fp16), how long it takes, and whether the
starting request in `run_chai_cpu.slurm` (`--time=24:00:00`,
`--cpus-per-task=16`, `--mem=128G`) is adequate must be measured on this
cluster with a compact job first.

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
