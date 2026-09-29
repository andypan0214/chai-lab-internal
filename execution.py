"""Execution profiles: where a submitted Chai-1 job runs (deployment layer).

The scientific job (FASTA, restraints, MSA/template flags) is identical for
every profile; a profile only chooses the Slurm script (resource request)
and the `device` forwarded to chai_lab.chai1.run_inference(device=...),
a parameter chai_lab already has (chai_lab/chai1.py:518, None -> "cuda:0";
non-"cuda:0" devices load weights on CPU first, chai1.py:143 / esm.py:38).

Chosen by the administrator in the web server's environment, never by
users:

    CHAI_EXECUTION_PROFILE=gpu   (default) run_chai.slurm,     device cuda:0
    CHAI_EXECUTION_PROFILE=cpu             run_chai_cpu.slurm, device cpu

Per-profile extra sbatch options (e.g. a different --partition):
CHAI_WEB_GPU_SBATCH_ARGS / CHAI_WEB_CPU_SBATCH_ARGS, applied after the
shared CHAI_WEB_SBATCH_ARGS. There is deliberately no automatic fallback
from one profile to the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

PROFILE_ENV = "CHAI_EXECUTION_PROFILE"
DEFAULT_PROFILE = "gpu"


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    device: str  # run_inference(device=...)
    slurm_script: str
    sbatch_args_env: str


PROFILES = {
    "gpu": ExecutionProfile("gpu", "cuda:0", "run_chai.slurm", "CHAI_WEB_GPU_SBATCH_ARGS"),
    "cpu": ExecutionProfile("cpu", "cpu", "run_chai_cpu.slurm", "CHAI_WEB_CPU_SBATCH_ARGS"),
}
DEVICES = tuple(p.device for p in PROFILES.values())


def execution_profile(environ: Mapping[str, str] = os.environ) -> ExecutionProfile:
    name = environ.get(PROFILE_ENV, "").strip().lower() or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ValueError(
            f"{PROFILE_ENV}={name!r} is not a known execution profile "
            f"(choose one of: {', '.join(PROFILES)})."
        )
    return PROFILES[name]
