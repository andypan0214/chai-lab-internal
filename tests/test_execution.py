"""Execution profiles (execution.py): GPU and CPU submit the same Chai job;
only the Slurm script, its sbatch options and run_inference(device=...)
differ. No torch / chai_lab / Slurm required."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import job_manager
from execution import PROFILES, execution_profile
from run_chai import build_arg_parser, build_run_kwargs

pytest.importorskip("numpy")  # test_web imports result_adapter
from test_web import COMPACT, FakeRunner  # noqa: E402
from form_state import validate_config  # noqa: E402
from presets import load_preset  # noqa: E402

CHAI_HPC_DIR = Path(__file__).resolve().parents[1]


class TestProfileSelection:
    def test_default_is_gpu(self):
        assert execution_profile({}) is PROFILES["gpu"]
        assert PROFILES["gpu"].device == "cuda:0"

    @pytest.mark.parametrize("value", ["cpu", "CPU", " cpu "])
    def test_cpu(self, value):
        p = execution_profile({"CHAI_EXECUTION_PROFILE": value})
        assert (p.device, p.slurm_script) == ("cpu", "run_chai_cpu.slurm")

    def test_unknown_profile_rejected(self):
        with pytest.raises(ValueError, match="not a known execution profile"):
            execution_profile({"CHAI_EXECUTION_PROFILE": "auto"})


class TestRunChaiDevice:
    def _args(self, tmp_path, *extra):
        fasta = tmp_path / "in.fasta"
        fasta.write_text(">protein|name=A\nMKV\n")
        return build_arg_parser().parse_args(
            ["--fasta", str(fasta), "--output", str(tmp_path / "out"), *extra]
        )

    def test_device_omitted_keeps_chai_default(self, tmp_path):
        assert "device" not in build_run_kwargs(self._args(tmp_path))

    @pytest.mark.parametrize("device", ["cuda:0", "cpu"])
    def test_device_forwarded_and_nothing_else_changes(self, tmp_path, device):
        base = build_run_kwargs(self._args(tmp_path))
        kwargs = build_run_kwargs(self._args(tmp_path, "--device", device))
        assert kwargs.pop("device") == device
        assert kwargs == base

    def test_other_devices_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            self._args(tmp_path, "--device", "cuda:1")


class TestJobManagerProfiles:
    def _submit(self, tmp_path, monkeypatch, profile=None, **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        runner = FakeRunner()
        backend = job_manager.SlurmBackend(runner=runner, profile=profile)
        store = job_manager.JobStore(tmp_path / "jobs", backend)
        job = store.create_and_submit(validate_config(load_preset(COMPACT)))
        return store, job, runner

    def test_gpu_profile_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CHAI_EXECUTION_PROFILE", raising=False)
        _, job, runner = self._submit(tmp_path, monkeypatch)
        cmd, kw = runner.calls[0]
        assert cmd[-1] == "run_chai.slurm" and kw["env"]["CHAI_DEVICE"] == "cuda:0"
        assert job["execution_profile"] == "gpu"

    def test_cpu_profile_from_env_with_its_sbatch_args(self, tmp_path, monkeypatch):
        _, job, runner = self._submit(
            tmp_path, monkeypatch, CHAI_EXECUTION_PROFILE="cpu",
            CHAI_WEB_SBATCH_ARGS="--account=lab", CHAI_WEB_CPU_SBATCH_ARGS="--partition=cpu",
            CHAI_WEB_GPU_SBATCH_ARGS="--partition=gpu",
        )
        cmd, kw = runner.calls[0]
        assert cmd[-3:] == ["--account=lab", "--partition=cpu", "run_chai_cpu.slurm"]
        assert "--partition=gpu" not in cmd
        assert kw["env"]["CHAI_DEVICE"] == "cpu" and job["execution_profile"] == "cpu"

    def test_same_job_inputs_for_both_profiles(self, tmp_path, monkeypatch):
        results = {}
        for name in ("gpu", "cpu"):
            store, job, runner = self._submit(tmp_path / name, monkeypatch, profile=PROFILES[name])
            job_dir = store.job_dir(job["job_id"])
            env = {k: v for k, v in runner.calls[0][1]["env"].items() if k != "CHAI_DEVICE"}
            results[name] = (
                (job_dir / "input.fasta").read_text(),
                (job_dir / "restraints.csv").read_text(),
                json.loads((job_dir / "config.json").read_text()),
                {k: v.replace(str(job_dir), "<job>") for k, v in env.items()},
            )
        assert results["gpu"] == results["cpu"]

    def test_misconfigured_profile_fails_job_without_sbatch(self, tmp_path, monkeypatch):
        _, job, runner = self._submit(tmp_path, monkeypatch, CHAI_EXECUTION_PROFILE="tpu")
        assert job["status"] == job_manager.FAILED and "administrator" in job["error"]
        assert runner.calls == []


def _posix_bash():
    bash = shutil.which("bash")
    # Windows' System32\bash.exe is WSL, which cannot see this Windows temp dir.
    if bash is None or "system32" in bash.lower():
        return None
    return bash


@pytest.mark.skipif(_posix_bash() is None, reason="needs a POSIX bash")
class TestSlurmScripts:
    """Run the real Slurm scripts (outside Slurm) with fake python/nvidia-smi
    and check which checks run and what run_chai.py receives."""

    def _run(self, tmp_path, script, **env):
        work = tmp_path / "chai_hpc"
        (work / "chai_env" / "bin").mkdir(parents=True)
        for name in ("run_chai.slurm", "run_chai_cpu.slurm"):
            shutil.copy(CHAI_HPC_DIR / name, work / name)
        calls = work / "calls.txt"
        (work / "chai_env" / "bin" / "activate").write_text(
            'export PATH="$PWD/chai_env/bin:$PATH"\n', newline="\n"
        )
        # nvidia-smi runs before the venv is activated: put it on PATH first.
        fakebin = tmp_path / "fakebin"
        fakebin.mkdir()
        for tool, bindir in (("python", work / "chai_env" / "bin"), ("nvidia-smi", fakebin)):
            f = bindir / tool
            f.write_text(
                f'#!/bin/bash\necho "{tool} $*" >> "$SLURM_SUBMIT_DIR/calls.txt"\n', newline="\n"
            )
            f.chmod(0o755)
        proc = subprocess.run(
            [_posix_bash(), script],
            cwd=work,
            env={
                **{k: v for k, v in os.environ.items()
                   if k not in ("CHAI_DEVICE", "CHAI_OFFLINE_MODE")},
                "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
                "SLURM_SUBMIT_DIR": work.as_posix(),
                "SLURM_JOB_ID": "1",
                "CHAI_DOWNLOADS_DIR": (tmp_path / "dl").as_posix(),
                **env,
            },
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        return calls.read_text().splitlines()

    @staticmethod
    def _run_chai_call(calls):
        return next(c for c in calls if c.startswith("python run_chai.py"))

    def test_gpu_profile(self, tmp_path):
        calls = self._run(tmp_path, "run_chai.slurm")
        assert "nvidia-smi " in calls
        assert "--device cuda:0" in self._run_chai_call(calls)

    def test_cpu_profile(self, tmp_path):
        calls = self._run(tmp_path, "run_chai_cpu.slurm", SLURM_CPUS_PER_TASK="4")
        assert not any(c.startswith("nvidia-smi") for c in calls)
        assert "--device cpu" in self._run_chai_call(calls)

    def test_invalid_device_rejected(self, tmp_path):
        with pytest.raises(AssertionError, match="CHAI_DEVICE must be"):
            self._run(tmp_path, "run_chai.slurm", CHAI_DEVICE="cuda:1")

    def test_cpu_script_requests_no_gpu(self):
        directives = [l for l in (CHAI_HPC_DIR / "run_chai_cpu.slurm").read_text().splitlines()
                      if l.startswith("#SBATCH")]
        assert directives and not any("gres" in l or "gpu" in l for l in directives)
