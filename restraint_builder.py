"""Convert UI-level restraint rows into Chai-1's restraint CSV
(the file passed as `constraint_path=` to `run_inference`).

Source of truth: https://github.com/chaidiscovery/chai-lab
Commit inspected: 66c38d1 (chai_lab.__version__ == "0.6.1")

  - chai_lab/data/parsing/restraints.py
      (PairwiseConstraintDataframeModel, PairwiseInteraction,
       _parse_res_idx, parse_pairwise_table, write_pairwise_table)
  - chai_lab/data/dataset/constraints/restraint_context.py
      (load_manual_restraints_for_chai1)
  - chai_lab/data/features/generators/token_dist_restraint.py
  - chai_lab/data/features/generators/token_pair_pocket_restraint.py
  - chai_lab/data/residue_constants.py (restype_1to3_with_x)
  - examples/restraints/{contact,pocket}.restraints, examples/restraints/README.md

Verified rules encoded here
----------------------------
1. The file is a CSV (plain `pd.read_csv`, restraints.py:174) with exactly
   these 10 columns, read by NAME so order does not matter to Chai
   (PairwiseConstraintDataframeModel, restraints.py:27-41). We emit them in
   the same order the shipped examples use:
       chainA,res_idxA,chainB,res_idxB,connection_type,confidence,
       min_distance_angstrom,max_distance_angstrom,comment,restraint_id
   Only `chainA/B`, `res_idxA/B`, `connection_type` and
   `max_distance_angstrom` currently influence the model; `confidence` and
   `min_distance_angstrom` are accepted but ignored, `comment` is never read
   (restraints.py docstring / README). We still fill them with sane defaults
   (confidence=1.0, min_distance_angstrom=0.0) matching the shipped examples,
   since the pandera model requires the columns to exist.

2. `res_idx` grammar (restraints.py:_parse_res_idx, doctested there):
   "<1-letter residue code><1-based index>", e.g. "C387" = Cys at position
   387. An "@<atom>" suffix is also legal but only meaningful for the
   `covalent` connection type, which is a separate code path in chai1.py and
   out of scope for this module's `contact`/`pocket` UI -- we therefore
   require a plain "<letter><digits>" token on every residue field we accept
   and never emit an atom suffix.

3. `contact` (PairwiseInteraction.__post_init__, restraints.py:78-80): both
   sides need a residue and/or atom. This module's UI only exposes
   "chain + residue", so we require a non-empty residue on BOTH sides.

4. `pocket` is asymmetric, NOT the same schema as `contact`
   (restraints.py:73-77, POCKET case):
     - res_idxA MUST be empty (chainA is the "pocket chain", chain-level).
     - res_idxB MUST be non-empty (chainB carries the specific pocket
       residue, token-level).
     - no atom suffix on either side.
   This is why our UI-facing dataclasses for contact vs. pocket expose
   different fields (task section 8: "Do NOT guess that pocket restraints
   use the same schema as contact restraints").

5. Residue-identity enforcement, and why we replicate it ourselves:
   Chai re-derives each residue's 3-letter name from the res_idx's leading
   letter via `restype_1to3_with_x` (residue_constants.py:574-597), which
   covers ONLY the 20 standard amino acids + "X" -> "UNK". That dict lookup
   happens in `load_manual_restraints_for_chai1`
   (restraint_context.py:113,116,124) -- so in this Chai release, `contact`
   and `pocket` restraints are effectively PROTEIN-RESIDUE-ONLY; a DNA/RNA/
   ligand chain on either side cannot be validly referenced by residue. Chai
   then cross-checks that letter against the ACTUAL tokenized sequence at
   that 1-based position (`add_distance_restraint` /
   `add_pocket_restraint`, asserting the true residue name matches).
   Crucially, both of those checks happen inside a feature generator wrapped
   in a bare `except Exception` (token_dist_restraint.py:174,
   token_pair_pocket_restraint.py:156) that on failure just logs an error
   and silently substitutes an all-"ignore" restraint matrix -- so a bad
   restraint does NOT crash the job, it just gets dropped without any
   visible sign in the output. We therefore validate residue identity here,
   against the real chain sequence, and REFUSE to write a restraint that
   Chai would silently ignore.

6. Defensive fix discovered by round-tripping generated files through the
   REAL `parse_pairwise_table` (see tests/test_restraint_builder.py):
   `PairwiseConstraintDataframeModel.comment` is declared
   `pa.Field(nullable=True)` WITHOUT `coerce=True` (restraints.py:41) --
   unlike `res_idxA`/`res_idxB`, which do have `coerce=True`. On pandas
   >=3.0, `pd.read_csv` infers an entirely-empty column as float64 NaN
   rather than object/string dtype, and pandera then raises
   `SchemaError: expected series 'comment' to have type string[python], got
   float64` because there is no coerce to fall back on. Concretely: a
   restraint file where every row's `comment` is "" fails to parse on a
   newer pandas, even though chai-lab's own declared dependency is
   `pandas~=2.1` (requirements.in), where this column would default to
   object dtype and validate fine. Since we cannot control exactly which
   pandas an HPC's dependency resolver picks, we simply never emit an
   all-blank `comment` column: an unset comment gets a small non-empty
   default (e.g. "contact restraint"). This is not a new format -- `comment`
   is documented as free text that Chai "does not read as an input"
   (examples/restraints/README.md) -- it is only a safe choice of default
   value for an existing column.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from input_builder import ChainRecord, parse_modified_sequence

# chai_lab/data/residue_constants.py:574-597 (restype_1to3_with_x), copied
# verbatim so residue-identity checks don't require importing chai_lab.
RESTYPE_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}

VALID_RESTRAINT_TYPES = ("contact", "pocket")

# "<1-letter residue code><1-based index>", per restraints.py:_parse_res_idx,
# with atoms deliberately excluded (see module docstring, rule 2).
_RES_IDX_RE = re.compile(r"^([A-Za-z])(\d+)$")

CSV_FIELDNAMES = [
    "chainA",
    "res_idxA",
    "chainB",
    "res_idxB",
    "connection_type",
    "confidence",
    "min_distance_angstrom",
    "max_distance_angstrom",
    "comment",
    "restraint_id",
]


class RestraintInputError(ValueError):
    """Raised when a restraint row cannot be turned into a valid Chai-1
    restraint that would actually take effect (as opposed to being silently
    dropped by Chai's own error-swallowing feature generators)."""


# ---------------------------------------------------------------------------
# UI-level restraint rows. Deliberately two different shapes, because contact
# and pocket restraints expose different fields in the Chai web workflow
# (task section 8) and in the underlying schema (rule 4 above).
# ---------------------------------------------------------------------------
@dataclass
class ContactRestraint:
    chain_a: str
    residue_a: str  # e.g. "C387"
    chain_b: str
    residue_b: str  # e.g. "Y101"
    max_distance_angstrom: float = 5.0
    comment: str = ""
    restraint_type: str = "contact"

    @classmethod
    def from_dict(cls, d: dict) -> "ContactRestraint":
        return cls(
            chain_a=str(d.get("chain_a", d.get("chain_1", ""))).strip(),
            residue_a=str(d.get("residue_a", d.get("residue_1", ""))).strip(),
            chain_b=str(d.get("chain_b", d.get("chain_2", ""))).strip(),
            residue_b=str(d.get("residue_b", d.get("residue_2", ""))).strip(),
            max_distance_angstrom=float(
                d.get("max_distance_angstrom", d.get("distance_angstrom", 5.0))
            ),
            comment=str(d.get("comment", "")),
        )


@dataclass
class PocketRestraint:
    pocket_chain: str  # chain-level side -> chainA, res_idxA left empty
    target_chain: str  # token-level side -> chainB
    target_residue: str  # e.g. "C387" -> res_idxB
    max_distance_angstrom: float = 5.0
    comment: str = ""
    restraint_type: str = "pocket"

    @classmethod
    def from_dict(cls, d: dict) -> "PocketRestraint":
        # Explicitly reject a residue on the chain-level side instead of
        # silently ignoring it -- a caller who supplies one almost certainly
        # meant a contact restraint, or misunderstands pocket's asymmetric
        # schema (rule 4 above).
        stray_residue = str(
            d.get("pocket_chain_residue", d.get("residue_1", d.get("residue_a", "")))
        ).strip()
        if stray_residue:
            raise RestraintInputError(
                f"pocket restraint: chain-level side (pocket_chain) must not "
                f"specify a residue, got {stray_residue!r}; pocket restraints "
                "are chain-vs-residue, not residue-vs-residue like contact "
                "(restraints.py POCKET case: res_idxA must be empty)"
            )
        return cls(
            pocket_chain=str(d.get("pocket_chain", d.get("chain_1", ""))).strip(),
            target_chain=str(d.get("target_chain", d.get("chain_2", ""))).strip(),
            target_residue=str(
                d.get("target_residue", d.get("residue_2", d.get("residue_b", "")))
            ).strip(),
            max_distance_angstrom=float(
                d.get("max_distance_angstrom", d.get("distance_angstrom", 5.0))
            ),
            comment=str(d.get("comment", "")),
        )


def restraint_from_dict(d: dict) -> ContactRestraint | PocketRestraint:
    rtype = str(d.get("restraint_type", "")).strip().lower()
    if rtype == "contact":
        return ContactRestraint.from_dict(d)
    if rtype == "pocket":
        return PocketRestraint.from_dict(d)
    raise RestraintInputError(
        f"restraint_type must be one of {VALID_RESTRAINT_TYPES}, got {rtype!r} "
        "(Chai's PairwiseInteractionType also defines 'covalent', but that is "
        "a separate covalent-bond code path, not exposed by this restraint UI)"
    )


@dataclass
class NormalizedRestraintRow:
    """One row exactly as it will be written to the Chai restraint CSV."""

    chainA: str
    res_idxA: str
    chainB: str
    res_idxB: str
    connection_type: str
    max_distance_angstrom: float
    comment: str
    min_distance_angstrom: float = 0.0
    confidence: float = 1.0
    restraint_id: str = ""

    def to_csv_row(self) -> dict:
        return {
            "chainA": self.chainA,
            "res_idxA": self.res_idxA,
            "chainB": self.chainB,
            "res_idxB": self.res_idxB,
            "connection_type": self.connection_type,
            "confidence": self.confidence,
            "min_distance_angstrom": self.min_distance_angstrom,
            "max_distance_angstrom": self.max_distance_angstrom,
            "comment": self.comment,
            "restraint_id": self.restraint_id,
        }


# ---------------------------------------------------------------------------
# Residue-identity validation (rule 5)
# ---------------------------------------------------------------------------
def _require_protein_chain(chain_id: str, chains_by_id: dict[str, ChainRecord], *, role: str) -> ChainRecord:
    if chain_id not in chains_by_id:
        raise RestraintInputError(
            f"restraint references chain {chain_id!r} ({role}), which is not "
            f"one of the molecule chains in this job: {sorted(chains_by_id)}"
        )
    chain = chains_by_id[chain_id]
    if chain.molecule_type != "protein":
        raise RestraintInputError(
            f"restraint references chain {chain_id!r} ({role}) of type "
            f"{chain.molecule_type!r}, but contact/pocket restraints can only "
            "reference protein residues in this Chai release: residue "
            "identity is checked via restype_1to3_with_x "
            "(residue_constants.py:574-597), which covers only the 20 "
            "standard amino acids + X, and KeyErrors (silently dropping the "
            "restraint) for any other entity type"
        )
    return chain


def _parse_and_validate_residue(
    token: str, chain: ChainRecord, *, field_name: str
) -> tuple[str, int]:
    """Validate a "<letter><1-based index>" token against the chain's actual
    sequence, exactly as Chai's own add_distance_restraint /
    add_pocket_restraint would (but raising instead of silently degrading)."""
    m = _RES_IDX_RE.match(token)
    if not m:
        raise RestraintInputError(
            f"{field_name}={token!r} is not valid Chai res_idx syntax: expected "
            "'<one-letter residue code><1-based index>', e.g. 'C387' "
            "(restraints.py:_parse_res_idx)"
        )
    letter, pos_str = m.group(1).upper(), m.group(2)
    pos = int(pos_str)
    if pos < 1:
        raise RestraintInputError(f"{field_name}={token!r}: index must be >= 1 (1-based)")
    if letter not in RESTYPE_1TO3:
        raise RestraintInputError(
            f"{field_name}={token!r}: {letter!r} is not one of the 20 standard "
            "amino-acid one-letter codes Chai recognizes for restraints "
            "(restype_1to3_with_x, residue_constants.py:574-597)"
        )

    constituents = parse_modified_sequence(chain.sequence)
    if constituents is None:
        # Should be unreachable: the chain came out of input_builder, which
        # already validated this sequence. Fail loudly if it happens anyway.
        raise RestraintInputError(
            f"internal error: chain {chain.chain_id!r} has unparseable sequence"
        )
    if pos > len(constituents):
        raise RestraintInputError(
            f"{field_name}={token!r}: position {pos} is beyond chain "
            f"{chain.chain_id!r}'s length ({len(constituents)} residues)"
        )

    actual = constituents[pos - 1]
    if len(actual) == 1:
        # Plain residue: Chai asserts an exact letter match
        # (add_distance_restraint / add_pocket_restraint).
        if actual != letter:
            raise RestraintInputError(
                f"{field_name}={token!r}: chain {chain.chain_id!r} has residue "
                f"{actual!r} at position {pos}, not {letter!r} -- this restraint "
                "would fail Chai's own residue-identity check and be silently "
                "dropped (token_dist_restraint.py / "
                "token_pair_pocket_restraint.py)"
            )
    # else: a bracketed modified-residue block (e.g. "(SEP)") occupies this
    # position. Chai maps its *name* through restype_1to3_with_x too, which
    # only recognizes standard amino acids, so a restraint can never
    # correctly target a modified residue here either.
    else:
        raise RestraintInputError(
            f"{field_name}={token!r}: position {pos} of chain {chain.chain_id!r} "
            f"is a modified residue block {actual!r}, not a standard amino "
            "acid; Chai's restraint residue-identity check "
            "(restype_1to3_with_x) cannot reference modified residues"
        )
    return letter, pos


def _validate_distance(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise RestraintInputError(f"max_distance_angstrom must be numeric, got {value!r}")
    if not (v >= 0.0):
        raise RestraintInputError(
            f"max_distance_angstrom must be >= 0.0, got {v} "
            "(PairwiseConstraintDataframeModel: ge=0.0)"
        )
    return v


def _comment_or_default(comment: str, connection_type: str) -> str:
    # See module docstring, rule 6: never emit an empty comment -- an
    # all-blank `comment` column round-trips as float64 NaN on pandas>=3.0
    # and fails Chai's own (coerce-less) schema validation for that column.
    comment = comment.strip()
    return comment if comment else f"{connection_type} restraint"


def normalize_restraint(
    restraint: ContactRestraint | PocketRestraint | dict,
    chains_by_id: dict[str, ChainRecord],
) -> NormalizedRestraintRow:
    if isinstance(restraint, dict):
        restraint = restraint_from_dict(restraint)

    if isinstance(restraint, ContactRestraint):
        chain_a = _require_protein_chain(restraint.chain_a, chains_by_id, role="chainA")
        chain_b = _require_protein_chain(restraint.chain_b, chains_by_id, role="chainB")
        _parse_and_validate_residue(restraint.residue_a, chain_a, field_name="residue_a")
        _parse_and_validate_residue(restraint.residue_b, chain_b, field_name="residue_b")
        dist = _validate_distance(restraint.max_distance_angstrom)
        return NormalizedRestraintRow(
            chainA=chain_a.chain_id,
            res_idxA=restraint.residue_a.strip().upper(),
            chainB=chain_b.chain_id,
            res_idxB=restraint.residue_b.strip().upper(),
            connection_type="contact",
            max_distance_angstrom=dist,
            comment=_comment_or_default(restraint.comment, "contact"),
        )

    if isinstance(restraint, PocketRestraint):
        pocket_chain = _require_protein_chain(
            restraint.pocket_chain, chains_by_id, role="pocket_chain"
        )
        target_chain = _require_protein_chain(
            restraint.target_chain, chains_by_id, role="target_chain"
        )
        _parse_and_validate_residue(
            restraint.target_residue, target_chain, field_name="target_residue"
        )
        dist = _validate_distance(restraint.max_distance_angstrom)
        return NormalizedRestraintRow(
            chainA=pocket_chain.chain_id,
            res_idxA="",  # pocket: chain-level side must be empty (rule 4)
            chainB=target_chain.chain_id,
            res_idxB=restraint.target_residue.strip().upper(),
            connection_type="pocket",
            max_distance_angstrom=dist,
            comment=_comment_or_default(restraint.comment, "pocket"),
        )

    raise RestraintInputError(f"unsupported restraint object: {restraint!r}")


def validate_restraints(
    restraints: list[dict | ContactRestraint | PocketRestraint],
    chains_by_id: dict[str, ChainRecord],
) -> list[NormalizedRestraintRow]:
    if not restraints:
        raise RestraintInputError(
            "no restraints provided; do not call build_restraint_file when "
            "restraints are disabled -- call run_inference with "
            "constraint_path=None instead (task section 5/10)"
        )
    rows = [normalize_restraint(r, chains_by_id) for r in restraints]
    for i, row in enumerate(rows):
        row.restraint_id = f"restraint_{i}"
    ids = [row.restraint_id for row in rows]
    assert len(ids) == len(set(ids)), "restraint_id must be unique (internal invariant)"
    return rows


def build_restraint_file(
    restraints: list[dict | ContactRestraint | PocketRestraint],
    chains_by_id: dict[str, ChainRecord],
    output_path: str | Path,
) -> Path:
    """Validate `restraints` against the actual molecule sequences and write
    the exact 10-column CSV Chai's `parse_pairwise_table` expects."""
    rows = validate_restraints(restraints, chains_by_id)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    return path
