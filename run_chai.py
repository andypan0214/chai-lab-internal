#!/usr/bin/env python3
"""Thin CLI wrapper around chai_lab.chai1.run_inference() for HPC/Slurm use.

Usage:
    python run_chai.py --fasta job_input.fasta --output outputs/test1
    python run_chai.py --fasta job_input.fasta --output outputs/test1 \\
        --seed 42 --restraints restraints.csv --use-msa --use-templates

Source of truth: https://github.com/chaidiscovery/chai-lab
Commit inspected: 66c38d1 (chai_lab.__version__ == "0.6.1") -- see Step A.

This file intentionally does NOT rewrite or wrap any chai_lab internals; it
only builds the exact keyword arguments run_inference() already accepts
(chai_lab/chai1.py:499-522) and adds the pre-flight checks the task asked
for. Verified facts this wrapper depends on:

  - run_inference()'s current signature (chai_lab/chai1.py:499-522) is
    keyword-only after `fasta_file`, and includes exactly:
    use_esm_embeddings, use_msa_server, msa_server_url, msa_directory,
    constraint_path, use_templates_server, template_hits_path,
    recycle_msa_subsample, num_trunk_recycles, num_diffn_timesteps,
    num_diffn_samples, num_trunk_samples, seed, device, low_memory,
    fasta_names_as_cif_chains. We expose --seed, --restraints, --use-msa,
    --use-templates, and (for offline use) --msa-directory ->
    msa_directory and --template-hits -> template_hits_path, and pass
    every other run_inference argument as its own default -- we do not
    invent extra parameters or flags.

  - CHAI_OFFLINE_MODE=1: check_offline() (see offline.py) rejects the
    server flags and requires every file chai_lab would otherwise download
    to exist already, so no download is ever attempted.

  - `output_dir` must not exist, or must be empty
    (chai_lab/chai1.py:530-533: `if output_dir.exists(): assert not any(
    output_dir.iterdir())`). We replicate this exact check ourselves before
    ever calling run_inference, so a non-empty --output produces a clear
    wrapper-level error instead of relying on Chai's internal assert (and so
    we never touch/overwrite whatever is already in that directory).

  - `--use-templates` requires `--use-msa`: when use_templates_server=True
    but no MSA server/directory was used, `templates_path` is never
    populated, and make_all_atom_feature_context() hits
    `assert not use_templates_server, "Server should have written a path"`
    once a protein chain is present (chai_lab/chai1.py:423-425). We enforce
    this unconditionally at the CLI level (regardless of what's in the
    FASTA) rather than letting a job fail deep inside inference.

  - `--use-templates` / `--template-hits` also need `kalign` >= 3.3 on PATH:
    chai_lab/tools/kalign.py shells out to `kalign -i ... -o ...` and only
    asserts its presence ("You need kalign>=3.3") once template hits are
    being aligned, deep inside inference. check_kalign() verifies both
    presence and version up front instead.

  - We always call run_inference(fasta_names_as_cif_chains=True) -- see
    input_builder.py's module docstring for why this is the naming mode our
    generated FASTA/restraint files assume.

  - constraint_path is only included in the call when --restraints is
    given; when omitted, run_inference() falls back to its own default of
    constraint_path=None (chai_lab/chai1.py:508), so restraints are never
    silently invented.

  - device is left at run_inference()'s own default (None -> "cuda:0",
    chai_lab/chai1.py:535). We do not expose a --device flag here (not part
    of this step's requested CLI surface).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from offline import missing_model_assets, missing_template_cifs, offline_mode_enabled

logger = logging.getLogger("run_chai")

KALIGN_MIN_VERSION = (3, 3)


class RunChaiError(RuntimeError):
    """A pre-flight configuration problem this wrapper caught before ever
    calling chai_lab (bad CLI args, missing files, non-empty output dir,
    unmet --use-templates/--use-msa dependency, missing/outdated kalign).

    Deliberately NOT caught anywhere in this module: we want the same
    behavior for these errors as for a real exception raised inside
    chai_lab itself -- a full traceback on stderr and a non-zero exit code,
    with nothing silently suppressed. See module __main__ block.
    """


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_chai.py",
        description=(
            "Run Chai-1 structure prediction (chai_lab.chai1.run_inference) "
            "on a pre-built FASTA and optional restraint file."
        ),
    )
    parser.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="Path to a Chai-1 FASTA input file (e.g. job_input.fasta from input_builder.py).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for this job. Must not exist, or must be empty "
        "(chai_lab requires this; see module docstring).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, forwarded as run_inference(seed=...). "
        "Default: chai_lab's own default (None).",
    )
    parser.add_argument(
        "--restraints",
        type=Path,
        default=None,
        help="Path to a restraint CSV (e.g. from restraint_builder.py), forwarded as "
        "run_inference(constraint_path=...). Omit to run without restraints.",
    )
    parser.add_argument(
        "--use-msa",
        action="store_true",
        help="Use the ColabFold MSA server, forwarded as "
        "run_inference(use_msa_server=True).",
    )
    parser.add_argument(
        "--use-templates",
        action="store_true",
        help="Use the templates server, forwarded as "
        "run_inference(use_templates_server=True). Requires --use-msa.",
    )
    parser.add_argument(
        "--msa-directory",
        type=Path,
        default=None,
        help="Directory of precomputed <sha256(sequence)>.aligned.pqt MSA files, "
        "forwarded as run_inference(msa_directory=...). No server call. "
        "Cannot be combined with --use-msa.",
    )
    parser.add_argument(
        "--template-hits",
        type=Path,
        default=None,
        help="Precomputed template hits (.m8, query IDs = chain IDs), forwarded as "
        "run_inference(template_hits_path=...). Cannot be combined with "
        "--use-templates. In offline mode every hit's <PDBID>.cif.gz must already "
        "be in the template CIF folder.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pre-flight validation (no chai_lab/torch import required -- this is what
# keeps these checks testable on a machine without a GPU or chai_lab
# installed).
# ---------------------------------------------------------------------------
def validate_args(args: argparse.Namespace) -> None:
    if not args.fasta.exists():
        raise RunChaiError(f"--fasta path does not exist: {args.fasta}")
    if not args.fasta.is_file():
        raise RunChaiError(f"--fasta path is not a file: {args.fasta}")

    if args.restraints is not None:
        if not args.restraints.exists():
            raise RunChaiError(f"--restraints path does not exist: {args.restraints}")
        if not args.restraints.is_file():
            raise RunChaiError(f"--restraints path is not a file: {args.restraints}")

    if args.use_templates and not args.use_msa:
        raise RunChaiError(
            "--use-templates requires --use-msa. Without an MSA server/directory, "
            "Chai-1 never populates a templates path, and "
            "make_all_atom_feature_context() asserts "
            "'Server should have written a path' once a protein chain is present "
            "(chai_lab/chai1.py:423-425). Add --use-msa, or drop --use-templates."
        )

    # make_all_atom_feature_context() asserts these pairs are exclusive
    # (chai_lab/chai1.py:352-357).
    if args.use_msa and args.msa_directory is not None:
        raise RunChaiError("--use-msa and --msa-directory cannot be combined; choose one MSA source.")
    if args.use_templates and args.template_hits is not None:
        raise RunChaiError(
            "--use-templates and --template-hits cannot be combined; choose one template source."
        )
    if args.msa_directory is not None and not args.msa_directory.is_dir():
        raise RunChaiError(f"--msa-directory is not a directory: {args.msa_directory}")
    if args.template_hits is not None and not args.template_hits.is_file():
        raise RunChaiError(f"--template-hits is not a file: {args.template_hits}")

    if args.output.exists():
        if not args.output.is_dir():
            raise RunChaiError(f"--output path exists and is not a directory: {args.output}")
        if any(args.output.iterdir()):
            raise RunChaiError(
                f"--output directory is not empty: {args.output}\n"
                "chai_lab.chai1.run_inference() requires output_dir to not exist, "
                "or to be empty (chai_lab/chai1.py:530-533). Refusing to overwrite "
                "an existing job -- choose a new --output path, or remove the "
                "existing directory yourself if you intend to discard it."
            )


def parse_kalign_version(text: str) -> tuple[int, ...] | None:
    """Extract the first X.Y[.Z] version from `kalign --version` output
    (Kalign 3.x: "Kalign (3.4.0)"; Kalign 2.x: "Kalign version 2.04")."""
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def check_offline(args: argparse.Namespace, environ=os.environ) -> None:
    """In offline mode (CHAI_OFFLINE_MODE=1), refuse anything that would reach
    the internet and require every file chai_lab would otherwise download
    (see offline.py). No-op in online mode."""
    if not offline_mode_enabled(environ):
        return
    if args.use_msa or args.use_templates:
        raise RunChaiError(
            "CHAI_OFFLINE_MODE is set: --use-msa / --use-templates call the external "
            "ColabFold server and are unavailable offline. Use precomputed inputs "
            "instead (--msa-directory DIR, --template-hits FILE.m8), or run without "
            "MSAs/templates."
        )
    problems = missing_model_assets(environ)
    if args.template_hits is not None:
        problems += missing_template_cifs(args.template_hits, environ)
    if problems:
        raise RunChaiError(
            "CHAI_OFFLINE_MODE is set, but local resources are missing, and chai_lab "
            "would try to download them:\n  - "
            + "\n  - ".join(problems)
            + "\nAdministrator: populate CHAI_DOWNLOADS_DIR once from a machine with "
            "internet access (e.g. one online run with the same CHAI_DOWNLOADS_DIR), "
            "see README_HPC.md 'Offline mode'."
        )


def check_kalign(args: argparse.Namespace) -> None:
    """With --use-templates or --template-hits, require kalign >= 3.3 on PATH
    (see module docstring). No-op otherwise."""
    if not (args.use_templates or args.template_hits is not None):
        return

    flag = "--use-templates" if args.use_templates else "--template-hits"
    min_version = ".".join(map(str, KALIGN_MIN_VERSION))
    hint = (
        "Run setup_env.sh (it builds tools/kalign/bin/kalign if the system "
        "kalign is missing or too old), then put it on PATH: "
        'export PATH="$PWD/tools/kalign/bin:$PATH" (run_chai.slurm does this '
        f"automatically). Or drop {flag}."
    )

    kalign = shutil.which("kalign")
    if kalign is None:
        raise RunChaiError(
            f"{flag} requires kalign >= {min_version} on PATH, but no "
            f"kalign was found. {hint}"
        )

    try:
        proc = subprocess.run(
            [kalign, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RunChaiError(f"Could not run '{kalign} --version': {e}. {hint}") from e

    version = parse_kalign_version(proc.stdout + proc.stderr)
    if version is None or version < KALIGN_MIN_VERSION:
        found = ".".join(map(str, version)) if version else "unknown"
        raise RunChaiError(
            f"{flag} requires kalign >= {min_version}, but {kalign} "
            f"reports version {found}. {hint}"
        )


def build_run_kwargs(args: argparse.Namespace) -> dict:
    """Build the exact kwargs passed to chai_lab.chai1.run_inference().

    Factored out of main() so the "only pass constraint_path when
    --restraints was given" and "fasta_names_as_cif_chains is always True"
    rules are unit-testable without importing torch/chai_lab.
    """
    kwargs: dict = dict(
        fasta_file=args.fasta,
        output_dir=args.output,
        use_msa_server=args.use_msa,
        use_templates_server=args.use_templates,
        seed=args.seed,
        # Always True: lets a chain's ID double as the Chai FASTA entity
        # name, the restraint-file chainA/chainB identifier, and the output
        # CIF chain label. See input_builder.py module docstring, rule 4.
        fasta_names_as_cif_chains=True,
    )
    if args.restraints is not None:
        kwargs["constraint_path"] = args.restraints
    if args.msa_directory is not None:
        kwargs["msa_directory"] = args.msa_directory
    if args.template_hits is not None:
        kwargs["template_hits_path"] = args.template_hits
    return kwargs


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def confidence_path_for(cif_path: Path) -> Path:
    """pred.model_idx_3.cif -> confidence.model_idx_3.npz (same directory)."""
    return cif_path.with_name(cif_path.stem.replace("pred.", "confidence.", 1) + ".npz")


def persist_confidence_arrays(candidates) -> list[Path]:
    """Save each candidate's PAE / PDE / per-token pLDDT next to its CIF.

    chai_lab writes CIFs and scores.model_idx_*.npz to disk, but returns
    PAE/PDE/pLDDT only in memory on StructureCandidates (chai_lab/chai1.py:
    StructureCandidates, run_folding_on_context). Candidate i of every array
    corresponds to cif_paths[i] (both built in the same loop and concatenated
    in the same order by StructureCandidates.concat). Values are written
    unchanged (np.asarray on the CPU tensors chai_lab already returns).
    """
    import numpy as np

    paths = []
    for i, cif_path in enumerate(candidates.cif_paths):
        out_path = confidence_path_for(Path(cif_path))
        np.savez(
            out_path,
            allow_pickle=False,
            pae=np.asarray(candidates.pae[i]),
            pde=np.asarray(candidates.pde[i]),
            plddt=np.asarray(candidates.plddt[i]),
        )
        paths.append(out_path)
    return paths


# ---------------------------------------------------------------------------
# GPU/environment reporting
# ---------------------------------------------------------------------------
def print_cuda_info() -> None:
    """Print CUDA availability and GPU name. Imports torch lazily so the
    rest of this module stays usable (CLI parsing, validation) on a machine
    without torch installed."""
    import torch

    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "No CUDA device detected. Chai-1 requires a Linux machine with a "
            "GPU that has CUDA and bfloat16 support (README: A100 80GB / "
            "H100 80GB / L40S 48GB recommended; A10/A30/RTX 4090 reported to "
            "work)."
        )


def print_configuration(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("Chai-1 HPC inference run")
    print("=" * 70)
    print(f"fasta:                {args.fasta}")
    print(f"output:               {args.output}")
    print(f"seed:                 {args.seed}")
    print(f"restraints:           {args.restraints if args.restraints is not None else '(none -- constraint_path=None)'}")
    print(f"use_msa_server:       {args.use_msa}")
    print(f"use_templates_server: {args.use_templates}")
    print(f"msa_directory:        {args.msa_directory}")
    print(f"template_hits_path:   {args.template_hits}")
    print(f"offline mode:         {offline_mode_enabled()}")
    print(f"fasta_names_as_cif_chains: True")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    validate_args(args)
    check_offline(args)
    check_kalign(args)

    # Requirement: "create output directory". Safe here -- validate_args()
    # already guaranteed args.output is either absent or empty.
    args.output.mkdir(parents=True, exist_ok=True)

    print_configuration(args)
    print_cuda_info()

    # Imported lazily: keeps --help / argument parsing / validate_args()
    # usable without torch or chai_lab installed (e.g. running the unit
    # tests on a laptop before shipping this to the HPC).
    from chai_lab.chai1 import run_inference

    run_kwargs = build_run_kwargs(args)
    logger.info(f"Calling chai_lab.chai1.run_inference() with: {run_kwargs}")

    # No try/except here: any exception raised by run_inference (or by
    # anything above) is left to propagate untouched, so Python's default
    # top-level handler prints the full traceback to stderr and the process
    # exits with a non-zero status. Nothing from chai_lab is ever caught
    # and silently suppressed.
    candidates = run_inference(**run_kwargs)

    logger.info(f"run_inference() returned {len(candidates.cif_paths)} structure candidate(s):")
    for cif_path in candidates.cif_paths:
        print(f"  {cif_path}")
    for path in persist_confidence_arrays(candidates):
        print(f"  {path}")
    print(f"Outputs written under: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
