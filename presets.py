"""CSV example presets for the Streamlit "Try an example" buttons.

One CSV file per preset, with a `record_type` column selecting what each row
describes:

    record_type,prediction_name,use_msa,use_templates,specify_restraints,
    molecule_type,copies,chain_ids,sequence,
    restraint_type,chain1,residue1,chain2,residue2,distance

  config     exactly one row: prediction_name, use_msa, use_templates,
             specify_restraints (true/false)
  molecule   one row per molecule: molecule_type, copies, chain_ids
             (separated by ';' or ',' -- quote the field if using ','),
             sequence
  restraint  one row per restraint: restraint_type (contact|pocket),
             chain1, residue1 (must be blank for pocket), chain2, residue2,
             distance (Angstrom)

Fields not used by a row's record_type must be blank. load_preset() returns a
normalized job config (see form_state) and then validates it with the backend
builders, so a malformed or scientifically invalid preset fails with a
PresetError naming the file and row instead of reaching the form.

Files named *_candidate.csv are not shown as example buttons unless the
admin sets CHAI_WEB_SHOW_CANDIDATE_PRESETS=1 (used to run a candidate
through the real HPC workflow before promoting it by renaming it).
"""

from __future__ import annotations

import csv
from pathlib import Path

from form_state import FormValidationError, validate_config

PRESETS_DIR = Path(__file__).resolve().parent / "examples" / "presets"

PRESET_COLUMNS = [
    "record_type",
    "prediction_name",
    "use_msa",
    "use_templates",
    "specify_restraints",
    "molecule_type",
    "copies",
    "chain_ids",
    "sequence",
    "restraint_type",
    "chain1",
    "residue1",
    "chain2",
    "residue2",
    "distance",
]

_ROW_FIELDS = {
    "config": {"prediction_name", "use_msa", "use_templates", "specify_restraints"},
    "molecule": {"molecule_type", "copies", "chain_ids", "sequence"},
    "restraint": {"restraint_type", "chain1", "residue1", "chain2", "residue2", "distance"},
}
_REQUIRED_FIELDS = {
    "config": {"prediction_name", "use_msa", "use_templates", "specify_restraints"},
    "molecule": {"molecule_type", "copies", "sequence"},
    "restraint": {"restraint_type", "chain1", "chain2", "residue2", "distance"},
}

# Button labels for known preset files; other files use their prediction name.
PRESET_LABELS = {
    "compact_3chain_restraints.csv": "Compact 3-chain restraints demo",
    "small_msa_template.csv": "Small MSA + template demo",
    "small_msa_template_candidate.csv": "Small MSA + template demo (candidate, admin only)",
}

CANDIDATE_SUFFIX = "_candidate.csv"


class PresetError(ValueError):
    """A preset file is malformed or describes an invalid job."""


def _parse_bool(value: str, *, where: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "yes", "1"):
        return True
    if v in ("false", "no", "0"):
        return False
    raise PresetError(f"{where}: expected true/false, got {value!r}")


def parse_preset_rows(rows: list[dict], *, source: str = "preset") -> dict:
    """Turn raw CSV dict rows into a normalized job config (unvalidated)."""
    config_row = None
    molecules = []
    restraints = []

    for line_no, raw in enumerate(rows, start=2):  # line 1 is the header
        where = f"{source}, line {line_no}"
        if None in raw:
            raise PresetError(f"{where}: more fields than the {len(PRESET_COLUMNS)} header columns")
        row = {k: (v or "").strip() for k, v in raw.items()}
        if not any(row.values()):
            continue  # blank line
        rtype = row["record_type"].lower()
        if rtype not in _ROW_FIELDS:
            raise PresetError(
                f"{where}: record_type must be one of {sorted(_ROW_FIELDS)}, got {row['record_type']!r}"
            )
        stray = sorted(k for k, v in row.items() if v and k != "record_type" and k not in _ROW_FIELDS[rtype])
        if stray:
            raise PresetError(f"{where}: fields {stray} must be blank on a {rtype!r} row")
        missing = sorted(k for k in _REQUIRED_FIELDS[rtype] if not row[k])
        if missing:
            raise PresetError(f"{where}: {rtype!r} row is missing {missing}")

        if rtype == "config":
            if config_row is not None:
                raise PresetError(f"{where}: only one 'config' row is allowed")
            config_row = {
                "name": row["prediction_name"],
                "use_msa": _parse_bool(row["use_msa"], where=f"{where} use_msa"),
                "use_templates": _parse_bool(row["use_templates"], where=f"{where} use_templates"),
                "specify_restraints": _parse_bool(
                    row["specify_restraints"], where=f"{where} specify_restraints"
                ),
            }
        elif rtype == "molecule":
            try:
                copies = int(row["copies"])
            except ValueError:
                raise PresetError(f"{where}: copies must be an integer, got {row['copies']!r}")
            molecules.append(
                {
                    "molecule_type": row["molecule_type"].lower(),
                    "copies": copies,
                    "chain_ids": [c.strip() for c in row["chain_ids"].replace(",", ";").split(";") if c.strip()],
                    "sequence": row["sequence"],
                }
            )
        else:
            restraint_type = row["restraint_type"].lower()
            if restraint_type == "pocket" and row["residue1"]:
                raise PresetError(
                    f"{where}: pocket restraints must leave residue1 blank (chain1 is the whole-chain side)"
                )
            try:
                distance = float(row["distance"])
            except ValueError:
                raise PresetError(f"{where}: distance must be a number, got {row['distance']!r}")
            restraints.append(
                {
                    "type": restraint_type,
                    "chain1": row["chain1"],
                    "residue1": None if restraint_type == "pocket" else row["residue1"],
                    "chain2": row["chain2"],
                    "residue2": row["residue2"],
                    "distance": distance,
                }
            )

    if config_row is None:
        raise PresetError(f"{source}: no 'config' row")
    if not molecules:
        raise PresetError(f"{source}: no 'molecule' rows")
    if restraints and not config_row["specify_restraints"]:
        raise PresetError(f"{source}: has restraint rows but specify_restraints is false")
    return {**config_row, "molecules": molecules, "restraints": restraints}


def load_preset(path: str | Path) -> dict:
    """Read, parse and backend-validate one preset file. Returns the config."""
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = [h.strip() for h in (reader.fieldnames or [])]
            if header != PRESET_COLUMNS:
                raise PresetError(
                    f"{path.name}: header must be exactly {','.join(PRESET_COLUMNS)}; got {','.join(header)}"
                )
            rows = list(reader)
    except OSError as e:
        raise PresetError(f"{path.name}: cannot read preset file ({e.strerror})") from e

    config = parse_preset_rows(rows, source=path.name)
    try:
        validate_config(config)
    except FormValidationError as e:
        raise PresetError(f"{path.name}: " + " ".join(e.messages)) from e
    return config


def list_presets(presets_dir: str | Path = PRESETS_DIR, *, include_candidates: bool = False) -> list[tuple[str, Path]]:
    """(button label, path) for each preset file, known presets first."""
    presets_dir = Path(presets_dir)
    if not presets_dir.is_dir():
        return []
    files = sorted(presets_dir.glob("*.csv"))
    if not include_candidates:
        files = [p for p in files if not p.name.endswith(CANDIDATE_SUFFIX)]
    order = list(PRESET_LABELS)
    files.sort(key=lambda p: (order.index(p.name) if p.name in order else len(order), p.name))
    return [(PRESET_LABELS.get(p.name, p.stem.replace("_", " ")), p) for p in files]
