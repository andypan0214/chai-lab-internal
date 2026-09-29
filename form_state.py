"""User-facing prediction form state for the Streamlit app.

This module only manages what the user typed. It does not serialize FASTA or
restraint CSV and does not re-implement any Chai validation rule: every
submitted configuration is validated by input_builder.build_chai_input() and
restraint_builder.validate_restraints(), which remain the source of truth.

Two shapes are used:

  form    -- editable rows as shown in the UI (chain IDs as free text, a
             stable `uid` per row so Streamlit widget keys survive
             reordering/removal).
  config  -- the normalized job configuration that is saved with a job and
             that presets load into:
               {
                 "name": str,
                 "use_msa": bool,
                 "use_templates": bool,
                 "specify_restraints": bool,
                 "molecules": [{"molecule_type", "copies", "chain_ids", "sequence"}],
                 "restraints": [{"type", "chain1", "residue1", "chain2",
                                 "residue2", "distance"}],
               }
             For pocket restraints residue1 is always None (chain1 is the
             whole-chain side; restraint_builder rule 4).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from input_builder import (
    VALID_MOLECULE_TYPES,
    BuildResult,
    MoleculeInputError,
    build_chai_input,
    parse_modified_sequence,
)
from restraint_builder import (
    VALID_RESTRAINT_TYPES,
    NormalizedRestraintRow,
    RestraintInputError,
    validate_restraints,
)

DEFAULT_DISTANCE_ANGSTROM = 5.0
MAX_NAME_LEN = 100


class FormValidationError(ValueError):
    """The form cannot be submitted. `messages` lists every problem found."""

    def __init__(self, messages: list[str]):
        super().__init__("; ".join(messages))
        self.messages = messages


# ---------------------------------------------------------------------------
# Row constructors
# ---------------------------------------------------------------------------
def _uid() -> str:
    return uuid.uuid4().hex[:10]


def new_molecule_row(
    molecule_type: str = "protein", copies: int = 1, chain_ids: str = "", sequence: str = ""
) -> dict:
    return {
        "uid": _uid(),
        "molecule_type": molecule_type,
        "copies": int(copies),
        "chain_ids": chain_ids,
        "sequence": sequence,
    }


def new_restraint_row(
    restraint_type: str = "contact",
    chain1: str = "",
    residue1: str = "",
    chain2: str = "",
    residue2: str = "",
    distance: float = DEFAULT_DISTANCE_ANGSTROM,
) -> dict:
    row = {
        "uid": _uid(),
        "type": restraint_type,
        "chain1": chain1,
        "residue1": residue1,
        "chain2": chain2,
        "residue2": residue2,
        "distance": float(distance),
    }
    if restraint_type == "pocket":
        row["residue1"] = ""
    return row


def empty_form() -> dict:
    return {
        "name": "",
        "use_msa": False,
        "use_templates": False,
        "specify_restraints": False,
        "molecules": [new_molecule_row()],
        "restraints": [],
    }


def set_restraint_type(row: dict, new_type: str) -> dict:
    """contact -> pocket clears residue1 (the whole-chain side takes no
    residue); pocket -> contact restores an (empty) residue1 that must then
    be filled in. Chains, residue2 and distance are preserved."""
    if new_type not in VALID_RESTRAINT_TYPES:
        raise ValueError(f"restraint type must be one of {VALID_RESTRAINT_TYPES}")
    row["type"] = new_type
    if new_type == "pocket":
        row["residue1"] = ""
    elif row.get("residue1") is None:
        row["residue1"] = ""
    return row


# ---------------------------------------------------------------------------
# form <-> config
# ---------------------------------------------------------------------------
def parse_chain_ids(text: str) -> list[str]:
    """'A', 'A,B', 'A; B' or 'A B' -> ['A', 'B']. Empty -> [] (auto-assign)."""
    return [c for c in re.split(r"[\s,;]+", text or "") if c]


def clean_sequence(sequence: str, molecule_type: str) -> str:
    """Remove line breaks/spaces pasted into a polymer sequence. Ligand
    SMILES is only stripped at the ends."""
    if molecule_type == "ligand":
        return (sequence or "").strip()
    return re.sub(r"\s+", "", sequence or "")


def default_name(molecules: list[dict]) -> str:
    """Name used when the user leaves it blank, e.g. 'protein-1-ligand-1'."""
    parts = [f"{m.get('molecule_type', 'molecule')}-{m.get('copies', 1)}" for m in molecules]
    return "-".join(parts)[:MAX_NAME_LEN] or "prediction"


def form_to_config(form: dict) -> dict:
    molecules = []
    for row in form.get("molecules", []):
        mtype = str(row.get("molecule_type", "")).strip().lower()
        molecules.append(
            {
                "molecule_type": mtype,
                "copies": int(row.get("copies") or 0),
                "chain_ids": parse_chain_ids(row.get("chain_ids", "")),
                "sequence": clean_sequence(row.get("sequence", ""), mtype),
            }
        )

    restraints = []
    if form.get("specify_restraints"):
        for row in form.get("restraints", []):
            rtype = str(row.get("type", "")).strip().lower()
            residue1 = str(row.get("residue1") or "").strip()
            restraints.append(
                {
                    "type": rtype,
                    "chain1": str(row.get("chain1") or "").strip(),
                    "residue1": (residue1 or None) if rtype == "pocket" else residue1,
                    "chain2": str(row.get("chain2") or "").strip(),
                    "residue2": str(row.get("residue2") or "").strip(),
                    "distance": row.get("distance"),
                }
            )

    name = str(form.get("name") or "").strip()[:MAX_NAME_LEN]
    return {
        "name": name or default_name(molecules),
        "use_msa": bool(form.get("use_msa")),
        "use_templates": bool(form.get("use_templates")),
        "specify_restraints": bool(form.get("specify_restraints")),
        "molecules": molecules,
        "restraints": restraints,
    }


def config_to_form(config: dict) -> dict:
    return {
        "name": config.get("name", ""),
        "use_msa": bool(config.get("use_msa")),
        "use_templates": bool(config.get("use_templates")),
        "specify_restraints": bool(config.get("specify_restraints")),
        "molecules": [
            new_molecule_row(
                molecule_type=m["molecule_type"],
                copies=m["copies"],
                chain_ids=",".join(m.get("chain_ids") or []),
                sequence=m["sequence"],
            )
            for m in config.get("molecules", [])
        ],
        "restraints": [
            new_restraint_row(
                restraint_type=r["type"],
                chain1=r["chain1"],
                residue1=r.get("residue1") or "",
                chain2=r["chain2"],
                residue2=r["residue2"],
                distance=r["distance"],
            )
            for r in config.get("restraints", [])
        ],
    }


# ---------------------------------------------------------------------------
# Validation (delegates to the backend builders)
# ---------------------------------------------------------------------------
def restraint_to_builder_dict(restraint: dict) -> dict:
    """Map a config restraint onto restraint_builder's own dict keys."""
    if restraint["type"] == "pocket":
        return {
            "restraint_type": "pocket",
            "pocket_chain": restraint["chain1"],
            "pocket_chain_residue": restraint.get("residue1") or "",
            "target_chain": restraint["chain2"],
            "target_residue": restraint["residue2"],
            "max_distance_angstrom": restraint["distance"],
        }
    return {
        "restraint_type": restraint["type"],
        "chain_a": restraint["chain1"],
        "residue_a": restraint.get("residue1") or "",
        "chain_b": restraint["chain2"],
        "residue_b": restraint["residue2"],
        "max_distance_angstrom": restraint["distance"],
    }


@dataclass
class ValidatedJob:
    config: dict
    build_result: BuildResult
    restraint_rows: list[NormalizedRestraintRow]


def validate_config(config: dict) -> ValidatedJob:
    """Validate a normalized config with the backend builders. Raises
    FormValidationError listing every problem found."""
    messages: list[str] = []

    if config.get("use_templates") and not config.get("use_msa"):
        messages.append(
            "Use Templates requires Use MSAs (templates are found from the MSA search)."
        )

    for i, m in enumerate(config.get("molecules", []), start=1):
        if m.get("molecule_type") not in VALID_MOLECULE_TYPES:
            messages.append(f"Molecule {i}: choose a molecule type.")
        if not m.get("sequence"):
            messages.append(f"Molecule {i}: sequence / SMILES is empty.")
    if not config.get("molecules"):
        messages.append("Add at least one molecule.")

    restraints = config.get("restraints", [])
    if config.get("specify_restraints"):
        if not restraints:
            messages.append("Add at least one restraint, or untick Specify restraints.")
        for i, r in enumerate(restraints, start=1):
            if r.get("type") not in VALID_RESTRAINT_TYPES:
                messages.append(f"Restraint {i}: choose contact or pocket.")
                continue
            if not r.get("chain1") or not r.get("chain2"):
                messages.append(f"Restraint {i}: both chains are required.")
            if r["type"] == "pocket" and r.get("residue1"):
                messages.append(f"Restraint {i}: pocket should not specify residue 1 (whole-chain side).")
            if r["type"] == "contact" and (not r.get("residue1") or not r.get("residue2")):
                messages.append(f"Restraint {i}: contact restraints need both residues.")
            if r["type"] == "pocket" and not r.get("residue2"):
                messages.append(f"Restraint {i}: pocket restraints need a target residue.")
            try:
                float(r.get("distance"))
            except (TypeError, ValueError):
                messages.append(f"Restraint {i}: distance must be a number.")
    if messages:
        raise FormValidationError(messages)

    try:
        build_result = build_chai_input(config["molecules"])
    except (MoleculeInputError, ValueError) as e:
        raise FormValidationError([f"Molecules: {e}"]) from e

    restraint_rows: list[NormalizedRestraintRow] = []
    if config.get("specify_restraints"):
        for i, r in enumerate(restraints, start=1):
            try:
                validate_restraints([restraint_to_builder_dict(r)], build_result.chains_by_id)
            except (RestraintInputError, ValueError) as e:
                messages.append(f"Restraint {i}: {e}")
        if messages:
            raise FormValidationError(messages)
        restraint_rows = validate_restraints(
            [restraint_to_builder_dict(r) for r in restraints], build_result.chains_by_id
        )

    return ValidatedJob(config=config, build_result=build_result, restraint_rows=restraint_rows)


# ---------------------------------------------------------------------------
# Read-only "Input sequences" table
# ---------------------------------------------------------------------------
def _token_count(molecule_type: str, sequence: str) -> int | None:
    """Tokens for one chain when it is unambiguous from the sequence alone:
    one token per standard residue/nucleotide. Ligands and polymers with
    modified (CCD) residues are tokenized per atom by Chai, which cannot be
    counted without chai_lab/rdkit, so None is returned for them."""
    if molecule_type == "ligand":
        return None
    constituents = parse_modified_sequence(sequence)
    if constituents is None or any(len(c) > 1 for c in constituents):
        return None
    return len(constituents)


def input_sequence_rows(config: dict, resolved_chains: list[dict]) -> list[dict]:
    """One row per submitted molecule: type, copies, resolved chain IDs,
    length, token offset range (end exclusive) and sequence. Offsets are
    shown only while every preceding chain has a known token count."""
    rows = []
    offset: int | None = 0
    chain_iter = iter(resolved_chains)
    for m in config["molecules"]:
        chains = [next(chain_iter) for _ in range(m["copies"])]
        per_copy = _token_count(m["molecule_type"], chains[0]["sequence"])
        if m["molecule_type"] == "ligand":
            length = None
        else:
            constituents = parse_modified_sequence(chains[0]["sequence"])
            length = len(constituents) if constituents is not None else None
        if offset is not None and per_copy is not None:
            end = offset + per_copy * m["copies"]
            token_offset = f"{offset}-{end}"
            offset = end
        else:
            token_offset = None
            offset = None
        rows.append(
            {
                "Molecule type": m["molecule_type"],
                "Copies": m["copies"],
                "Chain IDs": ", ".join(c["chain_id"] for c in chains),
                "Length": length,
                "Token offset": token_offset,
                "Sequence": chains[0]["sequence"],
            }
        )
    return rows


def total_known_tokens(rows: list[dict]) -> int | None:
    """End of the last token offset range, if every row's offset is known."""
    if not rows or any(r["Token offset"] is None for r in rows):
        return None
    return int(rows[-1]["Token offset"].split("-")[1])


def input_restraint_rows(config: dict) -> list[dict]:
    return [
        {
            "Type": r["type"],
            "Chain 1": r["chain1"],
            "Residue index 1": r.get("residue1") or ("Whole chain" if r["type"] == "pocket" else ""),
            "Chain 2": r["chain2"],
            "Residue index 2": r["residue2"],
            "Distance (Å)": float(r["distance"]),
        }
        for r in config.get("restraints", [])
    ]
