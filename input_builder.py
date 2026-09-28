"""Convert a UI-level molecule list into Chai-1-compatible FASTA input.

This module is intentionally the ONLY place that knows how to turn our own
internal data model (a plain list of molecule dicts) into the exact text
Chai-1's own FASTA parser expects. It does not import chai_lab (that package
requires torch/CUDA and is not expected to be installed on the machine that
builds the job); instead the parsing rules below are re-implemented from the
verified source, with the exact file/line cited next to each rule so they can
be re-checked against a newer chai-lab release.

Source of truth: https://github.com/chaidiscovery/chai-lab
Commit inspected: 66c38d1 (chai_lab.__version__ == "0.6.1")

Verified rules encoded here
----------------------------
1. FASTA header grammar (chai_lab/data/dataset/inference_dataset.py:227-282,
   function `read_inputs`):
       >{protein|dna|rna|ligand|glycan}|name={entity_name}
   or  >{protein|dna|rna|ligand|glycan}|{entity_name}
   A second "|" is unsupported and raises ValueError inside Chai itself
   (inference_dataset.py:282). We therefore never put anything besides the
   entity name after the type.

2. "copies" has no representation in Chai's FASTA format. Chai-1 has no
   copy-count field at all: N copies of a molecule are N separate FASTA
   records with N distinct entity names. Chai deduplicates records with the
   same (entity_type, sequence) into a single internal "entity" and assigns
   sym_id 0..N-1 to the repeats automatically
   (chai_lab/data/dataset/inference_dataset.py:126-147, and
   chai_lab/data/dataset/structure/all_atom_residue_tokenizer.py:617
   `_make_sym_ids`). We only need to expand copies into repeated records with
   unique names; Chai does the rest.

3. Entity names must be globally unique across the whole job
   (chai_lab/chai1.py:365-369, `UnsupportedInputError` on duplicates).

4. We always build jobs for `run_inference(fasta_names_as_cif_chains=True)`.
   In that mode Chai uses the FASTA entity name as BOTH the restraint-file
   "chainA"/"chainB" identifier AND the output CIF chain label (see
   `run_inference`'s own docstring in chai_lab/chai1.py, and
   `raw_inputs_to_entitites_data` in inference_dataset.py, which sets
   `subchain_id = entity_name` in this mode instead of an auto-generated
   letter). This is what lets a user-chosen "chain ID" map 1:1 onto both the
   FASTA and the restraint syntax with no hidden translation table.

5. That entity name is packed into a FIXED 4-byte tensor
   (`string_to_tensorcode(entity_data.subchain_id, pad_to_length=4)`,
   chai_lab/data/dataset/structure/all_atom_residue_tokenizer.py:509). A
   longer name trips an assert inside `_tokenize_entity`, which
   `load_chains_from_raw` (inference_dataset.py) catches with a bare
   `except Exception` and silently DROPS that chain from the job -- no error
   surfaces at inference time, the prediction just runs without it. So chain
   IDs are validated to <= 4 ASCII characters HERE, before Chai ever sees
   them. We further restrict IDs to plain alphanumerics so they round-trip
   safely through both the "|name=" FASTA header field and the restraint CSV
   (which is comma-separated) without escaping.

6. Ligand rows are SMILES-only. `get_lig_residues()`
   (inference_dataset.py:44-56) stores the record body verbatim as a SMILES
   string on a single `LIG` residue; there is no CCD-code entry point for
   ">ligand|" records in this release. CCD codes only exist via ">glycan|"
   records or inline "(CCD)" blocks in a polymer sequence -- neither of which
   this module exposes, per the task's instruction not to invent ligand CCD
   support. We validate ligand text against the same character set Chai's
   own heuristic uses for recognizing SMILES/glycan text
   (chai_lab/data/parsing/input_validation.py:71-76,
   `identify_potential_entity_types`); we do not attempt full SMILES
   chemistry validation (that requires rdkit).

7. Protein/DNA/RNA sequence syntax: Chai accepts a run of single letters or
   parenthesized/bracketed multi-letter CCD codes, e.g. "AGT(SEP)TG". This is
   `constituents_of_modified_fasta` (chai_lab/data/parsing/input_validation.py
   :15-51); `read_inputs`/`raw_inputs_to_entitites_data`
   (inference_dataset.py:126-133) hard-asserts on a sequence this function
   rejects, so we re-implement the identical grammar here and validate
   up front rather than letting a malformed sequence crash mid-job.
   Chai's own type-vs-sequence mismatch check
   (`identify_potential_entity_types`) is only a warning, not an error, but
   since our UI already asks the user to declare a type explicitly, we treat
   a mismatch (e.g. "QWERTY" declared as `dna`) as a validation error instead
   of silently forwarding it -- this is what actually protects a GPU job from
   producing garbage output.
"""

from __future__ import annotations

import itertools
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

VALID_MOLECULE_TYPES = ("protein", "dna", "rna", "ligand")

# all_atom_residue_tokenizer.py:509 -> string_to_tensorcode(pad_to_length=4)
MAX_CHAIN_ID_LEN = 4

# Chain IDs double as the Chai FASTA entity name (after "name=") and as a
# restraint-file "chainA"/"chainB" CSV field. Restrict to plain alphanumerics
# so no escaping/quoting question ever arises in either format.
_CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]{1,%d}$" % MAX_CHAIN_ID_LEN)

# Nucleotide letters accepted by Chai's own nucleic_acid_1_to_name table
# (chai_lab/data/parsing/fasta.py:19-26).
_DNA_LETTERS = set("AGTC")
_RNA_LETTERS = set("AGUC")

# SMILES/glycan character set used by Chai's own heuristic
# (chai_lab/data/parsing/input_validation.py:76).
_SMILES_CHARSET = set(string.ascii_letters + string.digits + ".-+=#$%:/\\[]()<>@")


class MoleculeInputError(ValueError):
    """Raised when a molecule row cannot be turned into valid Chai-1 input."""


# ---------------------------------------------------------------------------
# Re-implementation of chai_lab.data.parsing.input_validation
#   .constituents_of_modified_fasta (verified byte-for-byte against source),
# so protein/dna/rna sequences can be validated and residue-indexed without
# importing chai_lab itself.
# ---------------------------------------------------------------------------
def parse_modified_sequence(sequence: str) -> list[str] | None:
    """Parse a Chai polymer sequence into its 1-based residue constituents.

    Mirrors chai_lab.data.parsing.input_validation.constituents_of_modified_fasta.
    Returns e.g. ["A", "G", "SEP", "T"] for "AG(SEP)T", or None if the string
    is not valid polymer syntax (unbalanced/empty brackets, non-ASCII-letter
    single residues, etc.) -- exactly the cases where Chai itself would raise
    inside `raw_inputs_to_entitites_data`.
    """
    x = sequence.strip().upper()
    allowed_chars = string.ascii_letters + "()[]" + string.digits
    if not all(letter in allowed_chars for letter in x):
        return None

    current_modified: str | None = None
    constituents: list[str] = []
    for letter in x:
        if letter in "([":
            if current_modified is not None:
                return None  # double open bracket
            current_modified = ""
        elif letter in ")]":
            if current_modified is None:
                return None  # closed without opening
            if len(current_modified) <= 1:
                return None  # empty modification: () or single (K)
            constituents.append(current_modified)
            current_modified = None
        else:
            if current_modified is not None:
                current_modified += letter
            else:
                if letter not in string.ascii_letters:
                    return None
                constituents.append(letter)
    if current_modified is not None:
        return None  # unclosed bracket
    return constituents


# ---------------------------------------------------------------------------
# Internal UI data model
# ---------------------------------------------------------------------------
@dataclass
class Molecule:
    """One molecule row as entered by a user. NOT the Chai wire format."""

    molecule_type: str
    sequence: str
    copies: int = 1
    chain_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Molecule":
        return cls(
            molecule_type=str(d.get("molecule_type", "")).strip().lower(),
            sequence=str(d.get("sequence", "")),
            copies=int(d.get("copies", 1)),
            chain_ids=[str(c).strip() for c in (d.get("chain_ids") or [])],
        )


@dataclass
class ChainRecord:
    """One resolved Chai-1 FASTA record after copy-expansion and chain-ID
    assignment. `chain_id` is simultaneously the Chai entity name, the
    restraint-file chain identifier, and (with fasta_names_as_cif_chains=True)
    the output CIF chain label -- see module docstring, rule 4."""

    chain_id: str
    molecule_type: str
    sequence: str

    @property
    def fasta_header(self) -> str:
        return f"{self.molecule_type}|name={self.chain_id}"

    @property
    def fasta_record(self) -> str:
        return f">{self.fasta_header}\n{self.sequence}\n"


@dataclass
class BuildResult:
    fasta_text: str
    chains: list[ChainRecord]
    chains_by_id: dict[str, ChainRecord]


# ---------------------------------------------------------------------------
# Chain ID handling
# ---------------------------------------------------------------------------
def _auto_chain_id_pool(exclude: set[str]) -> Iterable[str]:
    """Yield short alphanumeric chain IDs (A, B, ..., Z, AA, AB, ...) that are
    not already in `exclude`, never exceeding MAX_CHAIN_ID_LEN."""
    for length in range(1, MAX_CHAIN_ID_LEN + 1):
        for tup in itertools.product(string.ascii_uppercase, repeat=length):
            candidate = "".join(tup)
            if candidate not in exclude:
                yield candidate
    raise MoleculeInputError(
        f"Ran out of auto-generatable chain IDs (<= {MAX_CHAIN_ID_LEN} chars); "
        "specify chain_ids explicitly for this many chains."
    )


def validate_chain_id(chain_id: str) -> str:
    if not _CHAIN_ID_RE.match(chain_id):
        raise MoleculeInputError(
            f"Invalid chain ID {chain_id!r}: must be 1-{MAX_CHAIN_ID_LEN} "
            "alphanumeric ASCII characters (Chai packs the chain/entity name "
            "into a fixed 4-byte field; longer names are silently dropped "
            "from the job at tokenization time -- "
            "all_atom_residue_tokenizer.py:509)."
        )
    return chain_id


# ---------------------------------------------------------------------------
# Per-type sequence validation
# ---------------------------------------------------------------------------
def validate_polymer_sequence(sequence: str, molecule_type: str) -> str:
    """Validate + normalize a protein/dna/rna sequence. Returns the
    upper-cased sequence Chai itself will parse (Chai upper-cases internally
    in constituents_of_modified_fasta, so we store the same normalized form
    to keep residue-position bookkeeping for restraints unambiguous)."""
    if not sequence.strip():
        raise MoleculeInputError(f"{molecule_type} sequence is empty")

    constituents = parse_modified_sequence(sequence)
    if constituents is None:
        raise MoleculeInputError(
            f"{sequence!r} is not valid Chai polymer syntax for {molecule_type} "
            "(letters and (CCD)/[CCD] blocks only, e.g. 'AGT(SEP)TG'); this is "
            "exactly the input Chai's own raw_inputs_to_entitites_data would "
            "assert on and crash the job."
        )

    single_letters = {c for c in constituents if len(c) == 1}
    if molecule_type == "dna" and not single_letters <= _DNA_LETTERS:
        bad = sorted(single_letters - _DNA_LETTERS)
        raise MoleculeInputError(
            f"DNA sequence contains non-DNA letters {bad}; Chai's DNA alphabet "
            "is A/G/T/C (plus modified blocks) -- fasta.py:19-26"
        )
    if molecule_type == "rna" and not single_letters <= _RNA_LETTERS:
        bad = sorted(single_letters - _RNA_LETTERS)
        raise MoleculeInputError(
            f"RNA sequence contains non-RNA letters {bad}; Chai's RNA alphabet "
            "is A/G/U/C (plus modified blocks) -- fasta.py:19-26"
        )
    # protein: any ascii-letter single residue or modified block is accepted
    # by Chai's own parser (restype_1to3_with_x maps unknowns to "UNK"/"X").

    return sequence.strip().upper()


def validate_ligand_sequence(sequence: str) -> str:
    seq = sequence.strip()
    if not seq:
        raise MoleculeInputError("ligand sequence (SMILES) is empty")
    bad = sorted({c for c in seq if c not in _SMILES_CHARSET})
    if bad:
        raise MoleculeInputError(
            f"ligand value {seq!r} contains characters {bad} outside Chai's "
            "SMILES/glycan charset (input_validation.py:76); note Chai's "
            "'>ligand|' records only accept SMILES -- there is no CCD-code "
            "entry point for ligands in this release, only for '>glycan|' "
            "records or inline (CCD) blocks in a polymer sequence."
        )
    return seq


def validate_sequence(sequence: str, molecule_type: str) -> str:
    if molecule_type in ("protein", "dna", "rna"):
        return validate_polymer_sequence(sequence, molecule_type)
    if molecule_type == "ligand":
        return validate_ligand_sequence(sequence)
    raise MoleculeInputError(f"unreachable molecule_type={molecule_type!r}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_chai_input(molecules: list[dict | Molecule]) -> BuildResult:
    """Turn a UI-level molecule list into Chai-1 FASTA text + resolved chain
    records, applying every validation rule documented in the module
    docstring. Raises MoleculeInputError on any problem; never silently
    drops or reinterprets user input."""
    if not molecules:
        raise MoleculeInputError("at least one molecule is required")

    mols = [m if isinstance(m, Molecule) else Molecule.from_dict(m) for m in molecules]

    # --- validate each row's shape ---
    for i, m in enumerate(mols):
        if m.molecule_type not in VALID_MOLECULE_TYPES:
            raise MoleculeInputError(
                f"molecule[{i}]: molecule_type must be one of "
                f"{VALID_MOLECULE_TYPES}, got {m.molecule_type!r}"
            )
        if not isinstance(m.copies, int) or m.copies < 1:
            raise MoleculeInputError(
                f"molecule[{i}]: copies must be a positive integer, got {m.copies!r}"
            )
        if m.chain_ids and len(m.chain_ids) != m.copies:
            raise MoleculeInputError(
                f"molecule[{i}]: {len(m.chain_ids)} chain_ids given but "
                f"copies={m.copies}; provide exactly one chain ID per copy "
                "or leave chain_ids empty to auto-generate them"
            )

    # --- validate + normalize sequences up front (before any chain-id work,
    #     so a bad sequence is reported clearly on its own) ---
    normalized_sequences = [
        validate_sequence(m.sequence, m.molecule_type) for m in mols
    ]

    # --- collect + validate user-specified chain IDs, checking global
    #     uniqueness (chai1.py:365-369) ---
    user_ids: set[str] = set()
    for i, m in enumerate(mols):
        for cid in m.chain_ids:
            validate_chain_id(cid)
            if cid in user_ids:
                raise MoleculeInputError(
                    f"duplicate chain ID {cid!r} in molecule[{i}]: every chain "
                    "ID must be unique across the whole job (Chai requires "
                    "globally unique entity names, chai1.py:365-369)"
                )
            user_ids.add(cid)

    # --- expand copies into resolved ChainRecords, auto-generating IDs for
    #     molecules that did not specify any ---
    taken = set(user_ids)
    auto_pool = _auto_chain_id_pool(exclude=taken)
    chains: list[ChainRecord] = []
    for m, seq in zip(mols, normalized_sequences):
        if m.chain_ids:
            ids_for_this_molecule = list(m.chain_ids)
        else:
            ids_for_this_molecule = []
            for _ in range(m.copies):
                cid = next(auto_pool)
                taken.add(cid)
                ids_for_this_molecule.append(cid)
        for cid in ids_for_this_molecule:
            chains.append(
                ChainRecord(chain_id=cid, molecule_type=m.molecule_type, sequence=seq)
            )

    fasta_text = "".join(c.fasta_record for c in chains)
    chains_by_id = {c.chain_id: c for c in chains}
    return BuildResult(fasta_text=fasta_text, chains=chains, chains_by_id=chains_by_id)


def write_fasta_file(build_result: BuildResult, output_path: str | Path) -> Path:
    """Write the generated FASTA to disk (task requirement: save as
    job_input.fasta so the user can inspect it before inference)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_result.fasta_text)
    return path
