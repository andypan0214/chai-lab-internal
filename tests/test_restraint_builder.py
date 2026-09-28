"""Tests for restraint_builder.py against the rules verified in Step A.

Where a clone of the official chai-lab repository is available (see
conftest.py / Step A's `_chai_src/`), the generated CSV is additionally
round-tripped through the REAL `chai_lab.data.parsing.restraints` module
(parse_pairwise_table / PairwiseInteraction), so these tests check not only
our own re-implementation of the rules but the actual upstream parser.
"""

import csv

import pytest

from input_builder import build_chai_input
from restraint_builder import (
    CSV_FIELDNAMES,
    VALID_RESTRAINT_TYPES,
    ContactRestraint,
    PocketRestraint,
    RestraintInputError,
    build_restraint_file,
    normalize_restraint,
    restraint_from_dict,
    validate_restraints,
)

# All 20 standard amino acids, one per position, so "position N" and
# "the letter at position N" are trivially known for test assertions.
# position: 1=A 2=C 3=D 4=E 5=F 6=G 7=H 8=I 9=K 10=L 11=M 12=N 13=P 14=Q
#           15=R 16=S 17=T 18=V 19=W 20=Y
PROT_A_SEQ = "ACDEFGHIKLMNPQRSTVWY"
PROT_B_SEQ = PROT_A_SEQ[::-1]  # position: 1=Y 2=W ... 20=A


def _two_protein_chains():
    result = build_chai_input(
        [
            {"molecule_type": "protein", "chain_ids": ["A"], "sequence": PROT_A_SEQ},
            {"molecule_type": "protein", "chain_ids": ["B"], "sequence": PROT_B_SEQ},
        ]
    )
    return result.chains_by_id


def _chains_with_dna():
    result = build_chai_input(
        [
            {"molecule_type": "protein", "chain_ids": ["A"], "sequence": PROT_A_SEQ},
            {"molecule_type": "dna", "chain_ids": ["D"], "sequence": "AGTC"},
        ]
    )
    return result.chains_by_id


# ---------------------------------------------------------------------------
# Schema-level tests
# ---------------------------------------------------------------------------
class TestCsvSchema:
    def test_ten_columns_exact_names(self):
        # PairwiseConstraintDataframeModel, restraints.py:27-41
        assert CSV_FIELDNAMES == [
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
        assert len(CSV_FIELDNAMES) == 10

    def test_only_contact_and_pocket_exposed(self):
        assert VALID_RESTRAINT_TYPES == ("contact", "pocket")

    def test_unknown_restraint_type_rejected(self):
        with pytest.raises(RestraintInputError):
            restraint_from_dict({"restraint_type": "covalent"})
        with pytest.raises(RestraintInputError):
            restraint_from_dict({"restraint_type": "docking"})


# ---------------------------------------------------------------------------
# Valid contact restraint
# ---------------------------------------------------------------------------
class TestValidContactRestraint:
    def test_normalizes_correctly(self):
        chains = _two_protein_chains()
        row = normalize_restraint(
            ContactRestraint(
                chain_a="A", residue_a="C2", chain_b="B", residue_b="Y1",
                max_distance_angstrom=5.5, comment="test",
            ),
            chains,
        )
        assert row.chainA == "A"
        assert row.res_idxA == "C2"
        assert row.chainB == "B"
        assert row.res_idxB == "Y1"
        assert row.connection_type == "contact"
        assert row.max_distance_angstrom == 5.5

    def test_matches_chai_example_layout(self):
        # task section 7 example: contact | A | C387 | B | Y101 | 5
        seq_a = "C" * 386 + "C"  # position 387 must be 'C'
        seq_b = "Y" * 100 + "Y"  # position 101 must be 'Y'
        chains = build_chai_input(
            [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq_a},
                {"molecule_type": "protein", "chain_ids": ["B"], "sequence": seq_b},
            ]
        ).chains_by_id
        row = normalize_restraint(
            ContactRestraint(chain_a="A", residue_a="C387", chain_b="B", residue_b="Y101",
                              max_distance_angstrom=5.0),
            chains,
        )
        assert (row.chainA, row.res_idxA, row.chainB, row.res_idxB) == ("A", "C387", "B", "Y101")

    def test_dict_form_accepted(self):
        chains = _two_protein_chains()
        row = normalize_restraint(
            {
                "restraint_type": "contact",
                "chain_1": "A",
                "residue_1": "C2",
                "chain_2": "B",
                "residue_2": "Y1",
                "distance_angstrom": 5.0,
            },
            chains,
        )
        assert row.connection_type == "contact"

    def test_build_restraint_file_writes_csv(self, tmp_path):
        chains = _two_protein_chains()
        out = build_restraint_file(
            [ContactRestraint(chain_a="A", residue_a="C2", chain_b="B", residue_b="Y1")],
            chains,
            tmp_path / "restraints.csv",
        )
        assert out.exists()
        with out.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_FIELDNAMES
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["connection_type"] == "contact"
        assert rows[0]["chainA"] == "A"
        assert rows[0]["res_idxA"] == "C2"
        assert rows[0]["restraint_id"] == "restraint_0"


# ---------------------------------------------------------------------------
# Wrong residue identity
# ---------------------------------------------------------------------------
class TestWrongResidueIdentity:
    def test_wrong_letter_at_position_rejected(self):
        chains = _two_protein_chains()
        # position 2 of PROT_A_SEQ is 'C', not 'D'
        with pytest.raises(RestraintInputError, match="residue"):
            normalize_restraint(
                ContactRestraint(chain_a="A", residue_a="D2", chain_b="B", residue_b="Y1"),
                chains,
            )

    def test_position_out_of_range_rejected(self):
        chains = _two_protein_chains()
        with pytest.raises(RestraintInputError, match="length"):
            normalize_restraint(
                ContactRestraint(chain_a="A", residue_a="C200", chain_b="B", residue_b="Y1"),
                chains,
            )

    def test_this_is_exactly_what_chai_would_silently_drop(self):
        # Document the stakes: Chai's own token_dist_restraint.py wraps this
        # check in `except Exception: log + fall back to all-ignore matrix`,
        # so a wrong residue identity does NOT crash inference -- it just
        # produces a prediction that silently ignored the restraint. We
        # refuse to write such a restraint at all.
        chains = _two_protein_chains()
        with pytest.raises(RestraintInputError):
            build_restraint_file(
                [ContactRestraint(chain_a="A", residue_a="D2", chain_b="B", residue_b="Y1")],
                chains,
                "unused.csv",
            )

    def test_malformed_res_idx_syntax_rejected(self):
        chains = _two_protein_chains()
        with pytest.raises(RestraintInputError, match="syntax"):
            normalize_restraint(
                ContactRestraint(chain_a="A", residue_a="2C", chain_b="B", residue_b="Y1"),
                chains,
            )

    def test_unknown_chain_rejected(self):
        chains = _two_protein_chains()
        with pytest.raises(RestraintInputError, match="not one of"):
            normalize_restraint(
                ContactRestraint(chain_a="Z", residue_a="C2", chain_b="B", residue_b="Y1"),
                chains,
            )

    def test_non_protein_chain_rejected(self):
        # restype_1to3_with_x only covers the 20 standard amino acids + X
        # (residue_constants.py:574-597); DNA/RNA/ligand chains cannot be
        # validly referenced by a contact/pocket residue restraint.
        chains = _chains_with_dna()
        with pytest.raises(RestraintInputError, match="protein"):
            normalize_restraint(
                ContactRestraint(chain_a="A", residue_a="C2", chain_b="D", residue_b="A1"),
                chains,
            )


# ---------------------------------------------------------------------------
# Valid pocket restraint
# ---------------------------------------------------------------------------
class TestValidPocketRestraint:
    def test_normalizes_correctly(self):
        chains = _two_protein_chains()
        row = normalize_restraint(
            PocketRestraint(pocket_chain="B", target_chain="A", target_residue="C2",
                             max_distance_angstrom=5.5),
            chains,
        )
        assert row.connection_type == "pocket"
        assert row.chainA == "B"
        assert row.res_idxA == ""  # chain-level side: MUST be empty
        assert row.chainB == "A"
        assert row.res_idxB == "C2"

    def test_matches_chai_example_layout(self):
        # task section 8 / examples/restraints/pocket.restraints:
        # B,,A,C387,pocket,...
        seq_a = "C" * 386 + "C"
        chains = build_chai_input(
            [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq_a},
                {"molecule_type": "protein", "chain_ids": ["B"], "sequence": "M" * 50},
            ]
        ).chains_by_id
        row = normalize_restraint(
            PocketRestraint(pocket_chain="B", target_chain="A", target_residue="C387"),
            chains,
        )
        assert (row.chainA, row.res_idxA, row.chainB, row.res_idxB) == ("B", "", "A", "C387")

    def test_build_restraint_file_writes_empty_res_idxa(self, tmp_path):
        chains = _two_protein_chains()
        out = build_restraint_file(
            [PocketRestraint(pocket_chain="B", target_chain="A", target_residue="C2")],
            chains,
            tmp_path / "pocket.csv",
        )
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["res_idxA"] == ""
        assert rows[0]["connection_type"] == "pocket"


# ---------------------------------------------------------------------------
# Invalid pocket restraint with residue on chainA
# ---------------------------------------------------------------------------
class TestInvalidPocketResidueOnChainA:
    def test_residue_1_on_pocket_side_rejected(self):
        # restraints.py POCKET case: `assert self.res_idxA == ""`. Our UI
        # models this as two different dataclasses so the mistake can be
        # caught at construction time rather than deep inside validation.
        with pytest.raises(RestraintInputError, match="res_idxA"):
            restraint_from_dict(
                {
                    "restraint_type": "pocket",
                    "chain_1": "B",
                    "residue_1": "C2",  # stray residue on chain-level side
                    "chain_2": "A",
                    "residue_2": "C2",
                }
            )

    def test_pocket_chain_residue_key_rejected(self):
        with pytest.raises(RestraintInputError):
            restraint_from_dict(
                {
                    "restraint_type": "pocket",
                    "pocket_chain": "B",
                    "pocket_chain_residue": "C2",
                    "target_chain": "A",
                    "target_residue": "C2",
                }
            )

    def test_dataclass_form_has_no_residue_field_for_chain_a(self):
        # PocketRestraint simply has no field to put a chainA residue in --
        # the schema difference from ContactRestraint is structural, not
        # just a validation rule.
        assert not hasattr(PocketRestraint, "residue_a")
        assert not hasattr(PocketRestraint, "pocket_chain_residue")


# ---------------------------------------------------------------------------
# Misc validation
# ---------------------------------------------------------------------------
class TestMiscValidation:
    def test_empty_restraints_list_rejected(self):
        chains = _two_protein_chains()
        with pytest.raises(RestraintInputError, match="disabled"):
            validate_restraints([], chains)

    def test_negative_distance_rejected(self):
        chains = _two_protein_chains()
        with pytest.raises(RestraintInputError):
            normalize_restraint(
                ContactRestraint(chain_a="A", residue_a="C2", chain_b="B", residue_b="Y1",
                                  max_distance_angstrom=-1.0),
                chains,
            )

    def test_restraint_ids_are_unique_and_sequential(self):
        chains = _two_protein_chains()
        rows = validate_restraints(
            [
                ContactRestraint(chain_a="A", residue_a="C2", chain_b="B", residue_b="Y1"),
                PocketRestraint(pocket_chain="B", target_chain="A", target_residue="C2"),
            ],
            chains,
        )
        assert [r.restraint_id for r in rows] == ["restraint_0", "restraint_1"]

    def test_mixed_contact_and_pocket_in_one_file(self, tmp_path):
        # "You may specify a mixture of contact and pocket restraints"
        # (examples/restraints/README.md)
        chains = _two_protein_chains()
        out = build_restraint_file(
            [
                ContactRestraint(chain_a="A", residue_a="C2", chain_b="B", residue_b="Y1"),
                PocketRestraint(pocket_chain="B", target_chain="A", target_residue="C2"),
            ],
            chains,
            tmp_path / "mixed.csv",
        )
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert [r["connection_type"] for r in rows] == ["contact", "pocket"]


# ---------------------------------------------------------------------------
# Round-trip through the REAL chai-lab parser, when available
# (see Step A / conftest.py for how _chai_src is located).
# ---------------------------------------------------------------------------
class TestRealChaiParserRoundTrip:
    def test_contact_csv_parses_with_real_chai_parser(self, tmp_path):
        chai_restraints = pytest.importorskip("chai_lab.data.parsing.restraints")
        chains = _two_protein_chains()
        out = build_restraint_file(
            [ContactRestraint(chain_a="A", residue_a="C2", chain_b="B", residue_b="Y1",
                               max_distance_angstrom=5.5, comment="round-trip test")],
            chains,
            tmp_path / "contact.csv",
        )
        parsed = chai_restraints.parse_pairwise_table(out)
        assert len(parsed) == 1
        interaction = parsed[0]
        assert interaction.chainA == "A"
        assert interaction.res_idxA == "C2"
        assert interaction.chainB == "B"
        assert interaction.res_idxB == "Y1"
        assert interaction.connection_type == chai_restraints.PairwiseInteractionType.CONTACT
        assert interaction.max_dist_angstrom == 5.5

    def test_pocket_csv_parses_with_real_chai_parser(self, tmp_path):
        chai_restraints = pytest.importorskip("chai_lab.data.parsing.restraints")
        chains = _two_protein_chains()
        out = build_restraint_file(
            [PocketRestraint(pocket_chain="B", target_chain="A", target_residue="C2",
                              max_distance_angstrom=5.5)],
            chains,
            tmp_path / "pocket.csv",
        )
        parsed = chai_restraints.parse_pairwise_table(out)
        assert len(parsed) == 1
        interaction = parsed[0]
        assert interaction.chainA == "B"
        assert interaction.res_idxA == ""
        assert interaction.chainB == "A"
        assert interaction.res_idxB == "C2"
        assert interaction.connection_type == chai_restraints.PairwiseInteractionType.POCKET

    def test_mixed_csv_parses_and_round_trips_via_real_PairwiseInteraction(self, tmp_path):
        # Also exercise PairwiseInteraction.__post_init__ directly (the
        # actual assertions Chai enforces at load time), not just the
        # dataframe-level pandera schema.
        chai_restraints = pytest.importorskip("chai_lab.data.parsing.restraints")
        chains = _two_protein_chains()
        out = build_restraint_file(
            [
                ContactRestraint(chain_a="A", residue_a="C2", chain_b="B", residue_b="Y1"),
                PocketRestraint(pocket_chain="B", target_chain="A", target_residue="C2"),
            ],
            chains,
            tmp_path / "mixed.csv",
        )
        parsed = chai_restraints.parse_pairwise_table(out)
        assert len(parsed) == 2
        for interaction in parsed:
            # __post_init__ already ran (parse_pairwise_table constructs
            # PairwiseInteraction per row); re-asserting here just documents
            # that construction succeeded without raising.
            assert isinstance(interaction, chai_restraints.PairwiseInteraction)

    def test_our_wrong_residue_rejection_agrees_with_real_chai_semantics(self):
        # We don't have real tokenized structures here to reproduce Chai's
        # exact runtime AssertionError, but we can confirm that Chai's own
        # residue-name table (restype_1to3_with_x) is the SAME 20-aa+X set
        # restraint_builder.RESTYPE_1TO3 was copied from, so our upfront
        # rejection and Chai's downstream KeyError/assert are checking the
        # same thing.
        residue_constants = pytest.importorskip("chai_lab.data.residue_constants")
        from restraint_builder import RESTYPE_1TO3

        assert RESTYPE_1TO3 == residue_constants.restype_1to3_with_x
