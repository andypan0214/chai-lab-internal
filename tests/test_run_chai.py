"""Small, targeted tests for run_chai.py's CLI and pre-flight validation.

None of these import torch or chai_lab (run_chai.py imports both lazily,
inside main(), specifically so this file can run on a machine with neither
installed). GPU / actual inference behavior is NOT covered here -- see the
report accompanying Step C for what remains untested until run on the HPC.
"""

import subprocess

import pytest

import run_chai
from run_chai import (
    RunChaiError,
    build_arg_parser,
    build_run_kwargs,
    check_kalign,
    parse_kalign_version,
    validate_args,
)


def _make_fasta(tmp_path, name="job_input.fasta"):
    p = tmp_path / name
    p.write_text(">protein|name=A\nMKV\n")
    return p


def _make_restraints(tmp_path, name="restraints.csv"):
    p = tmp_path / name
    p.write_text(
        "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
        "min_distance_angstrom,max_distance_angstrom,comment,restraint_id\n"
    )
    return p


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
class TestArgParsing:
    def test_required_args_parsed(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs" / "test1"
        args = build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(out)])
        assert args.fasta == fasta
        assert args.output == out
        assert args.seed is None
        assert args.restraints is None
        assert args.use_msa is False
        assert args.use_templates is False

    def test_all_optional_flags_parsed(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        restraints = _make_restraints(tmp_path)
        out = tmp_path / "outputs" / "test1"
        args = build_arg_parser().parse_args(
            [
                "--fasta", str(fasta),
                "--output", str(out),
                "--seed", "42",
                "--restraints", str(restraints),
                "--use-msa",
                "--use-templates",
            ]
        )
        assert args.seed == 42
        assert args.restraints == restraints
        assert args.use_msa is True
        assert args.use_templates is True

    def test_missing_required_fasta_is_argparse_error(self):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--output", "outputs/test1"])

    def test_missing_required_output_is_argparse_error(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--fasta", str(fasta)])

    def test_seed_must_be_int(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(
                ["--fasta", str(fasta), "--output", "o", "--seed", "not-an-int"]
            )


# ---------------------------------------------------------------------------
# --use-templates without --use-msa fails early
# ---------------------------------------------------------------------------
class TestTemplatesRequireMsa:
    def test_templates_without_msa_rejected(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(
            ["--fasta", str(fasta), "--output", str(out), "--use-templates"]
        )
        with pytest.raises(RunChaiError, match="--use-templates requires --use-msa"):
            validate_args(args)

    def test_templates_with_msa_accepted(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(
            ["--fasta", str(fasta), "--output", str(out), "--use-templates", "--use-msa"]
        )
        validate_args(args)  # must not raise


# ---------------------------------------------------------------------------
# Nonexistent FASTA fails
# ---------------------------------------------------------------------------
class TestFastaValidation:
    def test_nonexistent_fasta_rejected(self, tmp_path):
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(
            ["--fasta", str(tmp_path / "does_not_exist.fasta"), "--output", str(out)]
        )
        with pytest.raises(RunChaiError, match="--fasta path does not exist"):
            validate_args(args)

    def test_fasta_that_is_a_directory_rejected(self, tmp_path):
        fasta_dir = tmp_path / "looks_like_fasta"
        fasta_dir.mkdir()
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(
            ["--fasta", str(fasta_dir), "--output", str(out)]
        )
        with pytest.raises(RunChaiError, match="not a file"):
            validate_args(args)


# ---------------------------------------------------------------------------
# Non-empty output directory fails
# ---------------------------------------------------------------------------
class TestOutputDirValidation:
    def test_nonexistent_output_dir_accepted(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs" / "test1"  # does not exist yet
        args = build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(out)])
        validate_args(args)  # must not raise

    def test_empty_existing_output_dir_accepted(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        out.mkdir()
        args = build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(out)])
        validate_args(args)  # must not raise

    def test_non_empty_output_dir_rejected(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        out.mkdir()
        (out / "pred.model_idx_0.cif").write_text("stale output from a previous job")
        args = build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(out)])
        with pytest.raises(RunChaiError, match="not empty"):
            validate_args(args)
        # and it must NOT have been touched/deleted
        assert (out / "pred.model_idx_0.cif").read_text() == "stale output from a previous job"

    def test_output_path_that_is_a_file_rejected(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        out.write_text("not a directory")
        args = build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(out)])
        with pytest.raises(RunChaiError, match="not a directory"):
            validate_args(args)


# ---------------------------------------------------------------------------
# --restraints path missing fails
# ---------------------------------------------------------------------------
class TestRestraintsValidation:
    def test_missing_restraints_path_rejected(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(
            [
                "--fasta", str(fasta),
                "--output", str(out),
                "--restraints", str(tmp_path / "does_not_exist.csv"),
            ]
        )
        with pytest.raises(RunChaiError, match="--restraints path does not exist"):
            validate_args(args)

    def test_valid_restraints_path_accepted(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        restraints = _make_restraints(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(
            ["--fasta", str(fasta), "--output", str(out), "--restraints", str(restraints)]
        )
        validate_args(args)  # must not raise

    def test_no_restraints_is_valid(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(out)])
        validate_args(args)  # must not raise


# ---------------------------------------------------------------------------
# run_inference kwargs mapping (no chai_lab/torch import required)
# ---------------------------------------------------------------------------
class TestBuildRunKwargs:
    def test_constraint_path_omitted_when_no_restraints(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(out)])
        kwargs = build_run_kwargs(args)
        assert "constraint_path" not in kwargs

    def test_constraint_path_included_when_restraints_given(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        restraints = _make_restraints(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(
            ["--fasta", str(fasta), "--output", str(out), "--restraints", str(restraints)]
        )
        kwargs = build_run_kwargs(args)
        assert kwargs["constraint_path"] == restraints

    def test_fasta_names_as_cif_chains_always_true(self, tmp_path):
        fasta = _make_fasta(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(out)])
        assert build_run_kwargs(args)["fasta_names_as_cif_chains"] is True

    def test_no_invented_kwargs(self, tmp_path):
        # Only the parameters documented in run_chai.py's module docstring
        # (verified against chai_lab/chai1.py:499-522) may appear.
        fasta = _make_fasta(tmp_path)
        restraints = _make_restraints(tmp_path)
        out = tmp_path / "outputs"
        args = build_arg_parser().parse_args(
            [
                "--fasta", str(fasta),
                "--output", str(out),
                "--seed", "7",
                "--restraints", str(restraints),
                "--use-msa",
            ]
        )
        kwargs = build_run_kwargs(args)
        assert set(kwargs) <= {
            "fasta_file", "output_dir", "use_esm_embeddings", "use_msa_server",
            "msa_server_url", "msa_directory", "constraint_path",
            "use_templates_server", "template_hits_path", "recycle_msa_subsample",
            "num_trunk_recycles", "num_diffn_timesteps", "num_diffn_samples",
            "num_trunk_samples", "seed", "device", "low_memory",
            "fasta_names_as_cif_chains",
        }


# ---------------------------------------------------------------------------
# --use-templates requires kalign >= 3.3 on PATH (checked before inference)
# ---------------------------------------------------------------------------
class TestKalignCheck:
    def _args(self, tmp_path, *extra):
        fasta = _make_fasta(tmp_path)
        return build_arg_parser().parse_args(
            ["--fasta", str(fasta), "--output", str(tmp_path / "outputs"), *extra]
        )

    def _fake_kalign(self, monkeypatch, output):
        monkeypatch.setattr(run_chai.shutil, "which", lambda name: "/opt/bin/kalign")
        monkeypatch.setattr(
            run_chai.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=output, stderr=""),
        )

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("\nKalign (3.4.0)\n\nCopyright (C) 2006,2019 Timo Lassmann\n", (3, 4, 0)),
            ("Kalign version 2.04, Copyright (C) 2004, 2005, 2006 Timo Lassmann", (2, 4)),
            ("no version here", None),
        ],
    )
    def test_parse_kalign_version(self, text, expected):
        assert parse_kalign_version(text) == expected

    def test_not_checked_without_templates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_chai.shutil, "which", lambda name: None)
        check_kalign(self._args(tmp_path, "--use-msa"))  # must not raise

    def test_missing_kalign_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_chai.shutil, "which", lambda name: None)
        args = self._args(tmp_path, "--use-msa", "--use-templates")
        with pytest.raises(RunChaiError, match="no kalign was found"):
            check_kalign(args)

    def test_outdated_kalign_rejected(self, tmp_path, monkeypatch):
        self._fake_kalign(monkeypatch, "Kalign version 2.04")
        args = self._args(tmp_path, "--use-msa", "--use-templates")
        with pytest.raises(RunChaiError, match="reports version 2.4"):
            check_kalign(args)

    def test_unparseable_kalign_version_rejected(self, tmp_path, monkeypatch):
        self._fake_kalign(monkeypatch, "usage: kalign ...")
        args = self._args(tmp_path, "--use-msa", "--use-templates")
        with pytest.raises(RunChaiError, match="reports version unknown"):
            check_kalign(args)

    @pytest.mark.parametrize("output", ["Kalign (3.3)", "Kalign (3.4.0)"])
    def test_adequate_kalign_accepted(self, tmp_path, monkeypatch, output):
        self._fake_kalign(monkeypatch, output)
        check_kalign(self._args(tmp_path, "--use-msa", "--use-templates"))  # must not raise
