"""Offline mode (CHAI_OFFLINE_MODE) and the vendored 3D viewer. No network,
torch or chai_lab needed."""

import hashlib
import re
from pathlib import Path

import pytest

from form_state import OFFLINE_MSA_MESSAGE, FormValidationError, validate_config
from offline import (
    missing_model_assets,
    missing_template_cifs,
    offline_mode_enabled,
    required_asset_relpaths,
    template_pdb_ids,
)
from run_chai import RunChaiError, build_arg_parser, build_run_kwargs, check_offline, validate_args

CHAI_HPC = Path(__file__).resolve().parents[1]
VENDORED_3DMOL = CHAI_HPC / "vendor" / "3Dmol" / "3Dmol-min.js"
# sha256 of cdnjs 3Dmol/2.5.5/3Dmol-min.js, recorded in vendor/3Dmol/README.md
VENDORED_3DMOL_SHA256_RE = re.compile(r"sha256: `([0-9a-f]{64})`")


def _assets(root: Path) -> Path:
    for rel in required_asset_relpaths():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return root


def _args(tmp_path, *extra):
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">protein|name=A\nMKV\n")
    return build_arg_parser().parse_args(["--fasta", str(fasta), "--output", str(tmp_path / "out"), *extra])


class TestOfflineSettings:
    @pytest.mark.parametrize("value, expected", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("", False), ("false", False),
    ])
    def test_flag_parsing(self, value, expected):
        assert offline_mode_enabled({"CHAI_OFFLINE_MODE": value}) is expected

    def test_unset_is_online(self):
        assert offline_mode_enabled({}) is False

    def test_assets_required(self, tmp_path):
        assert "not set" in missing_model_assets({})[0]
        [problem] = missing_model_assets({"CHAI_DOWNLOADS_DIR": str(tmp_path)})
        assert "models_v2/trunk.pt" in problem and "conformers_v1.apkl" in problem
        assert "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt" in problem
        assert missing_model_assets({"CHAI_DOWNLOADS_DIR": str(_assets(tmp_path))}) == []

    def test_empty_asset_file_counts_as_missing(self, tmp_path):
        _assets(tmp_path)
        (tmp_path / "models_v2" / "trunk.pt").write_bytes(b"")
        assert "models_v2/trunk.pt" in missing_model_assets({"CHAI_DOWNLOADS_DIR": str(tmp_path)})[0]

    def test_template_cifs(self, tmp_path):
        m8 = tmp_path / "hits.m8"
        m8.write_text("A\t1ubq_A\t100\nA\t1ubi_A\t98\nA\t1UBQ_B\t90\n")
        assert template_pdb_ids(m8) == ["1UBQ", "1UBI"]
        env = {"CHAI_DOWNLOADS_DIR": str(tmp_path)}
        assert "1UBQ, 1UBI" in missing_template_cifs(m8, env)[0]
        folder = tmp_path / "template_cifs"
        folder.mkdir()
        for pid in ("1UBQ", "1UBI"):
            (folder / f"{pid}.cif.gz").write_bytes(b"x")
        assert missing_template_cifs(m8, env) == []
        # CHAI_TEMPLATE_CIF_FOLDER overrides the default, as in chai_lab
        assert missing_template_cifs(m8, {**env, "CHAI_TEMPLATE_CIF_FOLDER": str(tmp_path / "x")})


class TestRunChaiOffline:
    def test_online_mode_unchanged(self, tmp_path):
        check_offline(_args(tmp_path, "--use-msa", "--use-templates"), environ={})  # no raise

    def test_offline_rejects_servers(self, tmp_path):
        env = {"CHAI_OFFLINE_MODE": "1", "CHAI_DOWNLOADS_DIR": str(_assets(tmp_path / "dl"))}
        with pytest.raises(RunChaiError, match="unavailable offline"):
            check_offline(_args(tmp_path, "--use-msa"), environ=env)

    def test_offline_requires_assets(self, tmp_path):
        env = {"CHAI_OFFLINE_MODE": "1", "CHAI_DOWNLOADS_DIR": str(tmp_path)}
        with pytest.raises(RunChaiError, match="local resources are missing"):
            check_offline(_args(tmp_path), environ=env)

    def test_offline_with_assets_and_local_inputs(self, tmp_path):
        dl = _assets(tmp_path / "dl")
        (dl / "template_cifs").mkdir()
        (dl / "template_cifs" / "1UBQ.cif.gz").write_bytes(b"x")
        msa_dir = tmp_path / "msas"
        msa_dir.mkdir()
        m8 = tmp_path / "hits.m8"
        m8.write_text("A\t1ubq_A\t100\n")
        args = _args(tmp_path, "--msa-directory", str(msa_dir), "--template-hits", str(m8))
        validate_args(args)
        check_offline(args, environ={"CHAI_OFFLINE_MODE": "1", "CHAI_DOWNLOADS_DIR": str(dl)})
        kwargs = build_run_kwargs(args)
        assert kwargs["msa_directory"] == msa_dir and kwargs["template_hits_path"] == m8
        assert kwargs["use_msa_server"] is False and kwargs["use_templates_server"] is False

    def test_offline_missing_template_cif(self, tmp_path):
        m8 = tmp_path / "hits.m8"
        m8.write_text("A\t1ubq_A\t100\n")
        env = {"CHAI_OFFLINE_MODE": "1", "CHAI_DOWNLOADS_DIR": str(_assets(tmp_path / "dl"))}
        with pytest.raises(RunChaiError, match="1UBQ"):
            check_offline(_args(tmp_path, "--template-hits", str(m8)), environ=env)

    def test_local_and_server_sources_exclusive(self, tmp_path):
        msa_dir = tmp_path / "msas"
        msa_dir.mkdir()
        with pytest.raises(RunChaiError, match="cannot be combined"):
            validate_args(_args(tmp_path, "--use-msa", "--msa-directory", str(msa_dir)))

    def test_defaults_add_no_kwargs(self, tmp_path):
        kwargs = build_run_kwargs(_args(tmp_path))
        assert "msa_directory" not in kwargs and "template_hits_path" not in kwargs


class TestWebOffline:
    CONFIG = {
        "name": "t", "use_msa": True, "use_templates": True, "specify_restraints": False,
        "molecules": [{"molecule_type": "protein", "copies": 1, "chain_ids": ["A"], "sequence": "MKCY"}],
        "restraints": [],
    }

    def test_offline_blocks_msa_templates(self):
        with pytest.raises(FormValidationError) as e:
            validate_config(self.CONFIG, offline=True)
        assert OFFLINE_MSA_MESSAGE in e.value.messages

    def test_online_allows_msa_templates(self):
        validate_config(self.CONFIG)  # no raise

    def test_offline_allows_plain_job(self):
        validate_config({**self.CONFIG, "use_msa": False, "use_templates": False}, offline=True)


class TestVendored3Dmol:
    def test_vendored_file_matches_recorded_hash(self):
        data = VENDORED_3DMOL.read_bytes()
        readme = (VENDORED_3DMOL.parent / "README.md").read_text()
        [expected] = VENDORED_3DMOL_SHA256_RE.findall(readme)
        assert hashlib.sha256(data).hexdigest() == expected
        assert b"$3Dmol" in data
        assert (VENDORED_3DMOL.parent / "LICENSE").is_file()

    def test_app_loads_no_external_script(self):
        source = (CHAI_HPC / "streamlit_app.py").read_text(encoding="utf-8")
        assert "cdnjs" not in source and "unpkg" not in source and "jsdelivr" not in source
        assert "<script src=" not in source
        assert "vendor" in source and "3Dmol-min.js" in source
        # st.Page ":material/...:" icons are fetched from fonts.gstatic.com
        assert not re.search(r"st\.Page\([^)]*icon=\":material/", source)
