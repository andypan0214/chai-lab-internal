"""Offline-mode settings and local-asset preflight checks.

CHAI_OFFLINE_MODE=1 (also: true/yes/on) means: never reach the internet.
Used by run_chai.py (before inference) and streamlit_app.py (before
submission). Imports nothing heavy, so it works without torch/chai_lab.

What chai_lab would otherwise download on demand (source: chai-lab 66c38d1):

  - chai_lab/utils/paths.py: download_if_not_exists() fetches from
    chaiassets.com any file missing under CHAI_DOWNLOADS_DIR:
      models_v2/<component>.pt   (chai1_component; the six components
                                  loaded in chai_lab/chai1.py)
      conformers_v1.apkl         (cached_conformers, used for every job)
  - chai_lab/data/dataset/embeddings/esm.py: esm/traced_sdpa_esm2_t36_3B_
    UR50D_fp16.pt (run_inference uses ESM embeddings by default).
  - use_msa_server / use_templates_server: ColabFold server
    (chai_lab/data/dataset/msas/colabfold.py).
  - template hits (server or local .m8): each hit's <PDB>.cif.gz from
    files.rcsb.org unless already in the template CIF folder
    (CHAI_TEMPLATE_CIF_FOLDER, default $CHAI_DOWNLOADS_DIR/template_cifs;
    chai_lab/data/dataset/templates/context.py, data/io/rcsb.py).

download_if_not_exists() is a no-op when the file exists, so checking that
every file above is present is sufficient to guarantee no download happens.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

OFFLINE_ENV = "CHAI_OFFLINE_MODE"

# chai_lab/chai1.py: _component_moved_to(...) calls in run_folding_on_context
MODEL_COMPONENTS = (
    "feature_embedding.pt",
    "bond_loss_input_proj.pt",
    "token_embedder.pt",
    "trunk.pt",
    "diffusion_module.pt",
    "confidence_head.pt",
)
CONFORMERS_FILE = "conformers_v1.apkl"
ESM_FILE = "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"


def offline_mode_enabled(environ: Mapping[str, str] = os.environ) -> bool:
    return environ.get(OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def required_asset_relpaths() -> list[str]:
    return [f"models_v2/{c}" for c in MODEL_COMPONENTS] + [CONFORMERS_FILE, ESM_FILE]


def missing_model_assets(environ: Mapping[str, str] = os.environ) -> list[str]:
    """Problems preventing an offline run, as short messages; [] if none.
    Paths are reported relative to CHAI_DOWNLOADS_DIR."""
    downloads = environ.get("CHAI_DOWNLOADS_DIR", "").strip()
    if not downloads:
        return [
            "CHAI_DOWNLOADS_DIR is not set. In offline mode it must point at the "
            "persistent directory that already holds the Chai-1 model files."
        ]
    root = Path(downloads)
    if not root.is_dir():
        return [f"CHAI_DOWNLOADS_DIR does not exist or is not a directory: {root}"]
    missing = []
    for rel in required_asset_relpaths():
        path = root / rel
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(rel)
    if missing:
        return [f"missing from CHAI_DOWNLOADS_DIR: {', '.join(missing)}"]
    return []


def template_cif_folder(environ: Mapping[str, str] = os.environ) -> Path | None:
    """Mirror of chai_lab's TEMPLATE_CIF_FOLDER default."""
    if environ.get("CHAI_TEMPLATE_CIF_FOLDER"):
        return Path(environ["CHAI_TEMPLATE_CIF_FOLDER"])
    if environ.get("CHAI_DOWNLOADS_DIR"):
        return Path(environ["CHAI_DOWNLOADS_DIR"]) / "template_cifs"
    return None


def template_pdb_ids(m8_path: Path) -> list[str]:
    """PDB IDs referenced by an .m8 hits file. Column 2 is the subject ID,
    '<pdbid>_<chain>' (chai_lab/data/parsing/templates/m8.py)."""
    ids = []
    for line in Path(m8_path).read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1].strip():
            pdb_id = fields[1].strip().split("_")[0].upper()
            if pdb_id not in ids:
                ids.append(pdb_id)
    return ids


def missing_template_cifs(m8_path: Path, environ: Mapping[str, str] = os.environ) -> list[str]:
    """Problems preventing an offline run with a local .m8 file; [] if none."""
    folder = template_cif_folder(environ)
    if folder is None:
        return ["set CHAI_TEMPLATE_CIF_FOLDER (or CHAI_DOWNLOADS_DIR) to the folder of cached template CIFs"]
    missing = [pid for pid in template_pdb_ids(m8_path) if not (folder / f"{pid}.cif.gz").is_file()]
    if missing:
        return [
            f"template structures not cached in the template CIF folder "
            f"(expected <PDBID>.cif.gz): {', '.join(missing)}"
        ]
    return []
