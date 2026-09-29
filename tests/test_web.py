"""Lightweight non-GPU tests for the Streamlit layer: form state conversion,
contact/pocket normalization, preset loading, job submission (fake sbatch),
and result-adapter parsing. No torch/chai_lab/streamlit needed."""

import json
import subprocess
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")  # result_adapter needs numpy (requirements-web.txt)

import job_manager  # noqa: E402
from form_state import (
    FormValidationError,
    config_to_form,
    empty_form,
    form_to_config,
    input_restraint_rows,
    input_sequence_rows,
    new_restraint_row,
    set_restraint_type,
    total_known_tokens,
    validate_config,
)
from presets import PRESETS_DIR, PresetError, list_presets, load_preset, parse_preset_rows
from result_adapter import load_candidates
from run_chai import confidence_path_for, persist_confidence_arrays

COMPACT = PRESETS_DIR / "compact_3chain_restraints.csv"
CANDIDATE = PRESETS_DIR / "small_msa_template_candidate.csv"


def _config(**overrides):
    config = {
        "name": "t",
        "use_msa": False,
        "use_templates": False,
        "specify_restraints": False,
        "molecules": [{"molecule_type": "protein", "copies": 1, "chain_ids": ["A"], "sequence": "MKCY"}],
        "restraints": [],
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# form state
# ---------------------------------------------------------------------------
class TestFormState:
    def test_form_to_config_normalizes_user_text(self):
        form = empty_form()
        form["molecules"][0].update(chain_ids="A, B", copies=2, sequence="MK\nCY \n")
        config = form_to_config(form)
        assert config["molecules"][0]["chain_ids"] == ["A", "B"]
        assert config["molecules"][0]["sequence"] == "MKCY"
        assert config["name"] == "protein-2"  # generated when left blank
        assert config["restraints"] == []  # restraints off -> none submitted

    def test_ligand_smiles_only_stripped(self):
        form = empty_form()
        form["molecules"][0].update(molecule_type="ligand", sequence="  CC(=O)O \n")
        assert form_to_config(form)["molecules"][0]["sequence"] == "CC(=O)O"

    def test_round_trip_config_form_config(self):
        config = load_preset(COMPACT)
        again = form_to_config(config_to_form(config))
        assert again == config

    def test_contact_to_pocket_clears_residue1_and_back(self):
        row = new_restraint_row("contact", "A", "C387", "B", "Y101", 5.0)
        set_restraint_type(row, "pocket")
        assert row["residue1"] == "" and row["residue2"] == "Y101" and row["distance"] == 5.0
        set_restraint_type(row, "contact")
        assert row["type"] == "contact" and row["residue1"] == ""

    def test_pocket_row_submits_residue1_none(self):
        form = empty_form()
        form["specify_restraints"] = True
        form["restraints"] = [new_restraint_row("pocket", "A", "", "B", "Y101", 5.0)]
        assert form_to_config(form)["restraints"][0]["residue1"] is None

    def test_templates_require_msa(self):
        with pytest.raises(FormValidationError, match="requires Use MSAs"):
            validate_config(_config(use_templates=True))

    def test_contact_needs_both_residues(self):
        config = load_preset(COMPACT)
        config["restraints"][0]["residue1"] = ""
        with pytest.raises(FormValidationError, match="need both residues"):
            validate_config(config)

    def test_pocket_with_residue1_rejected(self):
        config = load_preset(COMPACT)
        config["restraints"][0].update(type="pocket")
        with pytest.raises(FormValidationError, match="should not specify residue 1"):
            validate_config(config)

    def test_valid_pocket_goes_through_restraint_builder(self):
        config = load_preset(COMPACT)
        config["restraints"][0].update(type="pocket", residue1=None)
        rows = validate_config(config).restraint_rows
        assert (rows[0].chainA, rows[0].res_idxA, rows[0].chainB, rows[0].res_idxB) == ("A", "", "B", "Y101")
        assert rows[0].connection_type == "pocket"

    def test_backend_residue_identity_error_surfaces(self):
        config = load_preset(COMPACT)
        config["restraints"][0]["residue1"] = "C388"
        with pytest.raises(FormValidationError, match="Restraint 1: .*position 388"):
            validate_config(config)

    def test_input_tables(self):
        config = load_preset(COMPACT)
        chains = [
            {"chain_id": c.chain_id, "molecule_type": c.molecule_type, "sequence": c.sequence}
            for c in validate_config(config).build_result.chains
        ]
        rows = input_sequence_rows(config, chains)
        assert [(r["Length"], r["Token offset"]) for r in rows] == [
            (604, "0-604"), (223, "604-827"), (221, "827-1048"),
        ]
        assert total_known_tokens(rows) == 1048
        assert input_restraint_rows(config)[1]["Residue index 1"] == "I32"

    def test_token_offsets_unknown_after_ligand(self):
        config = _config(molecules=[
            {"molecule_type": "ligand", "copies": 1, "chain_ids": ["L"], "sequence": "CCO"},
            {"molecule_type": "protein", "copies": 2, "chain_ids": ["A", "B"], "sequence": "MKCY"},
        ])
        chains = [
            {"chain_id": c.chain_id, "molecule_type": c.molecule_type, "sequence": c.sequence}
            for c in validate_config(config).build_result.chains
        ]
        rows = input_sequence_rows(config, chains)
        assert rows[0]["Length"] is None and rows[0]["Token offset"] is None
        assert rows[1]["Length"] == 4 and rows[1]["Token offset"] is None
        assert rows[1]["Chain IDs"] == "A, B"


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------
HEADER = (
    "record_type,prediction_name,use_msa,use_templates,specify_restraints,molecule_type,"
    "copies,chain_ids,sequence,restraint_type,chain1,residue1,chain2,residue2,distance\n"
)


class TestPresets:
    def test_compact_preset_populates_everything(self):
        config = load_preset(COMPACT)
        assert (config["use_msa"], config["use_templates"], config["specify_restraints"]) == (False, False, True)
        assert [m["chain_ids"] for m in config["molecules"]] == [["A"], ["B"], ["C"]]
        assert [len(m["sequence"]) for m in config["molecules"]] == [604, 223, 221]
        assert [(r["type"], r["chain1"], r["residue1"], r["chain2"], r["residue2"], r["distance"])
                for r in config["restraints"]] == [
            ("contact", "A", "C387", "B", "Y101", 5.0),
            ("contact", "C", "I32", "A", "S483", 5.0),
        ]

    def test_candidate_preset_valid_but_hidden(self):
        config = load_preset(CANDIDATE)
        assert (config["use_msa"], config["use_templates"], config["specify_restraints"]) == (True, True, False)
        assert config["molecules"][0]["sequence"].startswith("MQIFVKTLTGK")
        assert CANDIDATE not in [p for _, p in list_presets()]
        assert CANDIDATE in [p for _, p in list_presets(include_candidates=True)]
        assert list_presets()[0][0] == "Compact 3-chain restraints demo"

    def _write(self, tmp_path, body, header=HEADER):
        p = tmp_path / "p.csv"
        p.write_text(header + body)
        return p

    def test_bad_header_rejected(self, tmp_path):
        with pytest.raises(PresetError, match="header"):
            load_preset(self._write(tmp_path, "", header="a,b\n"))

    def test_missing_config_row_rejected(self, tmp_path):
        with pytest.raises(PresetError, match="no 'config' row"):
            load_preset(self._write(tmp_path, "molecule,,,,,protein,1,A,MKV,,,,,,\n"))

    def test_pocket_with_residue1_rejected(self, tmp_path):
        body = (
            "config,x,false,false,true,,,,,,,,,,\n"
            "molecule,,,,,protein,1,A,MKCY,,,,,,\n"
            "restraint,,,,,,,,,pocket,A,C3,A,Y4,5\n"
        )
        with pytest.raises(PresetError, match="residue1 blank"):
            load_preset(self._write(tmp_path, body))

    def test_stray_field_rejected(self, tmp_path):
        with pytest.raises(PresetError, match="must be blank"):
            load_preset(self._write(tmp_path, "config,x,false,false,false,protein,,,,,,,,,\n"))

    def test_backend_validation_applies(self, tmp_path):
        body = "config,x,false,false,false,,,,,,,,,,\nmolecule,,,,,dna,1,A,QWERTY,,,,,,\n"
        with pytest.raises(PresetError, match="non-DNA letters"):
            load_preset(self._write(tmp_path, body))

    def test_restraints_without_flag_rejected(self):
        rows = [
            {"record_type": "config", "prediction_name": "x", "use_msa": "false",
             "use_templates": "false", "specify_restraints": "false"},
            {"record_type": "molecule", "molecule_type": "protein", "copies": "1", "chain_ids": "A",
             "sequence": "MKCY"},
            {"record_type": "restraint", "restraint_type": "contact", "chain1": "A", "residue1": "M1",
             "chain2": "A", "residue2": "Y4", "distance": "5"},
        ]
        rows = [{k: r.get(k, "") for k in HEADER.strip().split(",")} for r in rows]
        with pytest.raises(PresetError, match="specify_restraints is false"):
            parse_preset_rows(rows)


# ---------------------------------------------------------------------------
# job submission (fake sbatch / squeue)
# ---------------------------------------------------------------------------
class FakeRunner:
    def __init__(self, squeue_out="", sacct_out="", sbatch_out="777\n"):
        self.calls = []
        self.out = {"sbatch": sbatch_out, "squeue": squeue_out, "sacct": sacct_out}

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=self.out[cmd[0]], stderr="")


class TestJobManager:
    def _store(self, tmp_path, runner):
        return job_manager.JobStore(tmp_path / "jobs", job_manager.SlurmBackend(runner=runner))

    def test_submit_writes_inputs_and_calls_sbatch(self, tmp_path):
        runner = FakeRunner()
        store = self._store(tmp_path, runner)
        job = store.create_and_submit(validate_config(load_preset(COMPACT)))
        assert job["status"] == job_manager.QUEUED and job["slurm_job_id"] == "777"
        job_dir = store.job_dir(job["job_id"])
        assert (job_dir / "input.fasta").read_text().startswith(">protein|name=A\nMMADSKLVSL")
        assert "A,C387,B,Y101,contact" in (job_dir / "restraints.csv").read_text()
        saved = json.loads((job_dir / "config.json").read_text())
        assert [c["chain_id"] for c in saved["resolved_chains"]] == ["A", "B", "C"]

        cmd, kw = runner.calls[0]
        assert cmd[0] == "sbatch" and cmd[-1] == "run_chai.slurm" and "--parsable" in cmd
        assert kw["cwd"] == job_manager.CHAI_HPC_DIR
        env = kw["env"]
        assert env["USE_MSA"] == "0" and env["USE_TEMPLATES"] == "0"
        assert env["OUTPUT_DIR"] == str(job_dir / "output")
        assert env["RESTRAINTS_PATH"] == str(job_dir / "restraints.csv")
        assert store.list_jobs()[0]["job_id"] == job["job_id"]

    def test_sbatch_failure_marks_failed(self, tmp_path):
        def runner(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="invalid partition")

        store = self._store(tmp_path, runner)
        job = store.create_and_submit(validate_config(_config()))
        assert job["status"] == job_manager.FAILED and "invalid partition" in job["error"]

    @pytest.mark.parametrize(
        "squeue, sacct, has_outputs, expected",
        [
            ("PENDING\n", "", False, job_manager.QUEUED),
            ("RUNNING\n", "", False, job_manager.RUNNING),
            ("", "COMPLETED\n", True, job_manager.SUCCESS),
            ("", "COMPLETED\n", False, job_manager.FAILED),
            ("", "CANCELLED by 42\n", False, job_manager.FAILED),
            ("", "", True, job_manager.SUCCESS),
        ],
    )
    def test_refresh_maps_slurm_state(self, tmp_path, squeue, sacct, has_outputs, expected):
        store = self._store(tmp_path, FakeRunner())
        job = store.create_and_submit(validate_config(_config()))
        store.backend.runner = FakeRunner(squeue_out=squeue, sacct_out=sacct)
        if has_outputs:
            out = store.output_dir(job["job_id"])
            out.mkdir()
            (out / "pred.model_idx_0.cif").write_text("data_x\n")
            (out / "scores.model_idx_0.npz").write_bytes(b"x")
        assert store.refresh(job)["status"] == expected

    def test_unreachable_scheduler_keeps_status(self, tmp_path):
        store = self._store(tmp_path, FakeRunner())
        job = store.create_and_submit(validate_config(_config()))

        def broken(cmd, **kw):
            raise OSError("no slurm here")

        store.backend.runner = broken
        assert store.refresh(job)["status"] == job_manager.QUEUED

    def test_job_ids_are_validated(self, tmp_path):
        store = self._store(tmp_path, FakeRunner())
        with pytest.raises(job_manager.JobError):
            store.job_dir("../../etc")


# ---------------------------------------------------------------------------
# result persistence + adapter
# ---------------------------------------------------------------------------
def _fake_run(tmp_path, aggregate=(0.3, 0.9, 0.5)):
    """Write files shaped like chai_lab + run_chai.py outputs."""
    rng = np.random.default_rng(0)
    n = len(aggregate)
    cif_paths = [tmp_path / f"pred.model_idx_{i}.cif" for i in range(n)]
    for i, (p, agg) in enumerate(zip(cif_paths, aggregate)):
        p.write_text("data_model\n")
        np.savez(
            tmp_path / f"scores.model_idx_{i}.npz",
            aggregate_score=np.array([agg], dtype=np.float32),
            ptm=np.array([0.1 + i], dtype=np.float32),
            iptm=np.array([0.2 + i], dtype=np.float32),
            per_chain_ptm=np.array([[0.4, 0.5]], dtype=np.float32),
            has_inter_chain_clashes=np.array([False]),
        )
    candidates = SimpleNamespace(
        cif_paths=cif_paths,
        pae=rng.random((n, 6, 6), dtype=np.float32) * 32,
        pde=rng.random((n, 6, 6), dtype=np.float32),
        plddt=rng.random((n, 6), dtype=np.float32),
    )
    return candidates


class TestResults:
    def test_confidence_path(self, tmp_path):
        assert confidence_path_for(tmp_path / "pred.model_idx_3.cif").name == "confidence.model_idx_3.npz"

    def test_persist_then_load_keeps_values_and_orders_by_aggregate(self, tmp_path):
        run = _fake_run(tmp_path)
        persist_confidence_arrays(run)
        cands = load_candidates(tmp_path)
        assert [c.candidate_index for c in cands] == [1, 2, 0]
        assert [c.rank for c in cands] == [1, 2, 3]
        best = cands[0]
        assert best.ptm == pytest.approx(1.1) and best.iptm == pytest.approx(1.2)
        assert best.per_chain_ptm == pytest.approx([0.4, 0.5])
        conf = best.load_confidence()
        np.testing.assert_array_equal(conf["pae"], run.pae[1])
        np.testing.assert_array_equal(conf["plddt"], run.plddt[1])
        assert conf["pae"].dtype == np.float32

    def test_missing_confidence_is_optional(self, tmp_path):
        _fake_run(tmp_path, aggregate=(0.5,))
        [cand] = load_candidates(tmp_path)
        assert cand.confidence_path is None and cand.load_confidence() is None
