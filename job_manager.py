"""Filesystem job registry + Slurm submission for the Streamlit app.

Each web job gets its own directory under JOBS_ROOT (CHAI_WEB_JOBS_DIR,
default <chai_hpc>/web_jobs):

    <job_id>/
        job.json          metadata + status (the registry record)
        config.json       normalized submitted configuration + resolved chains
        input.fasta       written by input_builder.write_fasta_file
        restraints.csv    written by restraint_builder.build_restraint_file
        output/           run_chai.py --output (created by run_chai.py)
        slurm.out/.err    Slurm stdout/stderr

Submission goes through the existing backend path unchanged: `sbatch
run_chai.slurm` from the chai_hpc directory, with FASTA_PATH /
RESTRAINTS_PATH / USE_MSA / USE_TEMPLATES / OUTPUT_DIR passed as environment
variables (run_chai.slurm reads exactly these). No inference logic lives here.

Statuses: QUEUED, RUNNING, SUCCESS, FAILED. A job is SUCCESS only when Slurm
reports it finished AND its output directory holds Chai's CIF + scores files.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from form_state import ValidatedJob, restraint_to_builder_dict
from input_builder import write_fasta_file
from restraint_builder import build_restraint_file

CHAI_HPC_DIR = Path(__file__).resolve().parent

QUEUED, RUNNING, SUCCESS, FAILED = "QUEUED", "RUNNING", "SUCCESS", "FAILED"
TERMINAL_STATUSES = (SUCCESS, FAILED)
UNREACHABLE = "UNREACHABLE"

# Slurm job states (squeue %T / sacct State) -> web status.
_SLURM_ACTIVE = {
    "PENDING": QUEUED, "CONFIGURING": QUEUED, "REQUEUED": QUEUED, "REQUEUE_HOLD": QUEUED,
    "REQUEUE_FED": QUEUED, "RESV_DEL_HOLD": QUEUED, "SUSPENDED": QUEUED,
    "RUNNING": RUNNING, "COMPLETING": RUNNING, "STAGE_OUT": RUNNING, "SIGNALING": RUNNING,
    "RESIZING": RUNNING,
}
_SLURM_FAILED = {
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT",
}

_JOB_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")

Runner = Callable[..., subprocess.CompletedProcess]


class JobError(RuntimeError):
    """Submission/registry failure with a message safe to show users."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def default_jobs_root() -> Path:
    return Path(os.environ.get("CHAI_WEB_JOBS_DIR", CHAI_HPC_DIR / "web_jobs")).resolve()


@dataclass
class SlurmBackend:
    """Thin wrapper over sbatch/squeue/sacct. Commands are overridable via
    CHAI_WEB_SBATCH / CHAI_WEB_SQUEUE / CHAI_WEB_SACCT, and extra sbatch
    options (e.g. --partition/--account) via CHAI_WEB_SBATCH_ARGS."""

    workdir: Path = CHAI_HPC_DIR
    script: str = "run_chai.slurm"
    runner: Runner = subprocess.run

    def _cmd(self, env_var: str, default: str) -> list[str]:
        return shlex.split(os.environ.get(env_var, default))

    def submit(self, job_dir: Path, env: dict[str, str], job_name: str) -> str:
        cmd = [
            *self._cmd("CHAI_WEB_SBATCH", "sbatch"),
            "--parsable",
            f"--job-name={job_name}",
            f"--output={job_dir / 'slurm.out'}",
            f"--error={job_dir / 'slurm.err'}",
            *shlex.split(os.environ.get("CHAI_WEB_SBATCH_ARGS", "")),
            self.script,
        ]
        try:
            proc = self.runner(
                cmd, cwd=self.workdir, env={**os.environ, **env},
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise JobError(f"Could not submit to the scheduler ({type(e).__name__}).") from e
        if proc.returncode != 0:
            raise JobError(f"The scheduler rejected the job: {proc.stderr.strip()[-500:]}")
        # --parsable prints "<jobid>" or "<jobid>;<cluster>"
        slurm_id = proc.stdout.strip().split(";")[0]
        if not slurm_id.isdigit():
            raise JobError(f"Unexpected sbatch output: {proc.stdout.strip()[:200]!r}")
        return slurm_id

    def state(self, slurm_id: str) -> str | None:
        """Raw Slurm state; None if the scheduler answered but no longer
        knows the job; UNREACHABLE if the scheduler could not be queried."""
        answered = False
        for cmd in (
            [*self._cmd("CHAI_WEB_SQUEUE", "squeue"), "-h", "-j", slurm_id, "-o", "%T"],
            [*self._cmd("CHAI_WEB_SACCT", "sacct"), "-n", "-X", "-P", "-j", slurm_id, "-o", "State"],
        ):
            try:
                proc = self.runner(cmd, capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if proc.returncode == 0 and proc.stdout.strip():
                # "CANCELLED by 123" -> "CANCELLED"
                return proc.stdout.strip().splitlines()[0].split()[0].upper()
            # squeue exits non-zero with "Invalid job id" once a job has left
            # the queue; any other non-zero exit means we learned nothing.
            if proc.returncode == 0 or "invalid job id" in proc.stderr.lower():
                answered = True
        return None if answered else UNREACHABLE


def outputs_complete(output_dir: Path) -> bool:
    cifs = list(output_dir.rglob("pred.model_idx_*.cif")) if output_dir.is_dir() else []
    return bool(cifs) and all(
        c.with_name(c.name.replace("pred.", "scores.", 1)).with_suffix(".npz").is_file() for c in cifs
    )


class JobStore:
    def __init__(self, root: Path | None = None, backend: SlurmBackend | None = None):
        self.root = Path(root) if root is not None else default_jobs_root()
        self.backend = backend or SlurmBackend()

    # --- paths -----------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        if not _JOB_ID_RE.match(job_id):
            raise JobError(f"Unknown prediction {job_id!r}.")
        return self.root / job_id

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output"

    # --- registry --------------------------------------------------------
    def list_jobs(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        jobs = []
        for d in self.root.iterdir():
            if _JOB_ID_RE.match(d.name) and (d / "job.json").is_file():
                try:
                    jobs.append(json.loads((d / "job.json").read_text()))
                except (OSError, ValueError):
                    continue
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)

    def get_job(self, job_id: str) -> dict:
        path = self.job_dir(job_id) / "job.json"
        if not path.is_file():
            raise JobError(f"Unknown prediction {job_id!r}.")
        return json.loads(path.read_text())

    def get_config(self, job_id: str) -> dict:
        return json.loads((self.job_dir(job_id) / "config.json").read_text())

    def _save(self, job: dict) -> dict:
        job["updated_at"] = _now()
        _write_json_atomic(self.job_dir(job["job_id"]) / "job.json", job)
        return job

    # --- submission ------------------------------------------------------
    def create_and_submit(self, validated: ValidatedJob) -> dict:
        """Write the job's inputs with the backend builders, then sbatch it.
        Returns the job record (FAILED with an error message if sbatch fails)."""
        config = validated.config
        job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)

        fasta_path = write_fasta_file(validated.build_result, job_dir / "input.fasta")
        restraints_path = None
        if config["specify_restraints"] and config["restraints"]:
            restraints_path = build_restraint_file(
                [restraint_to_builder_dict(r) for r in config["restraints"]],
                validated.build_result.chains_by_id,
                job_dir / "restraints.csv",
            )

        saved_config = {
            **config,
            "resolved_chains": [
                {"chain_id": c.chain_id, "molecule_type": c.molecule_type, "sequence": c.sequence}
                for c in validated.build_result.chains
            ],
        }
        _write_json_atomic(job_dir / "config.json", saved_config)

        now = _now()
        job = {
            "job_id": job_id,
            "name": config["name"],
            "use_msa": config["use_msa"],
            "use_templates": config["use_templates"],
            "specify_restraints": bool(restraints_path),
            "created_at": now,
            "updated_at": now,
            "status": QUEUED,
            "slurm_job_id": None,
            "error": None,
        }
        _write_json_atomic(job_dir / "job.json", job)

        env = {
            "FASTA_PATH": str(fasta_path),
            "RESTRAINTS_PATH": str(restraints_path) if restraints_path else "",
            "USE_MSA": "1" if config["use_msa"] else "0",
            "USE_TEMPLATES": "1" if config["use_templates"] else "0",
            "SEED": "",
            "OUTPUT_DIR": str(job_dir / "output"),
        }
        try:
            job["slurm_job_id"] = self.backend.submit(job_dir, env, job_name=f"chai1-web-{job_id}")
        except JobError as e:
            job["status"] = FAILED
            job["error"] = self.sanitize(str(e))
        return self._save(job)

    # --- status ----------------------------------------------------------
    def refresh(self, job: dict) -> dict:
        """Update a non-terminal job's status from Slurm + its outputs."""
        if job["status"] in TERMINAL_STATUSES or not job.get("slurm_job_id"):
            return job
        state = self.backend.state(job["slurm_job_id"])
        if state == UNREACHABLE:
            return job  # scheduler temporarily unavailable: keep current status
        done = outputs_complete(self.output_dir(job["job_id"]))

        if state in _SLURM_ACTIVE:
            new_status, error = _SLURM_ACTIVE[state], None
        elif state == "COMPLETED" or (state is None and done):
            new_status, error = (SUCCESS, None) if done else (
                FAILED, "The job finished but produced no structure files."
            )
        elif state in _SLURM_FAILED:
            new_status, error = FAILED, f"The job ended with state {state}."
        elif state is None:
            new_status, error = FAILED, "The job is no longer known to the scheduler and produced no results."
        else:
            return job  # unfamiliar transient state: keep the current status

        if new_status != job["status"] or error != job.get("error"):
            job["status"], job["error"] = new_status, error
            self._save(job)
        return job

    # --- logs ------------------------------------------------------------
    def sanitize(self, text: str) -> str:
        """Hide server filesystem paths from user-visible messages."""
        for path, label in ((self.root, "<jobs>"), (CHAI_HPC_DIR, "<app>")):
            text = text.replace(str(path), label)
        return text

    def log_tail(self, job_id: str, name: str = "slurm.err", lines: int = 40) -> str:
        path = self.job_dir(job_id) / name
        if not path.is_file():
            return ""
        tail = path.read_text(errors="replace").splitlines()[-lines:]
        return self.sanitize("\n".join(tail))
