"""Tests for input_builder.py against the rules verified in Step A.

Where a claim can be checked mechanically (header grammar, 4-char chain-ID
limit, global name uniqueness) these tests assert the exact behavior, citing
the chai-lab source line the behavior is derived from.
"""

import pytest

from input_builder import (
    MAX_CHAIN_ID_LEN,
    MoleculeInputError,
    build_chai_input,
    parse_modified_sequence,
    validate_chain_id,
)

PROTEIN_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"


class TestOneMolecule:
    def test_single_protein_produces_one_fasta_record(self):
        result = build_chai_input(
            [{"molecule_type": "protein", "copies": 1, "chain_ids": [], "sequence": PROTEIN_SEQ}]
        )
        assert len(result.chains) == 1
        chain = result.chains[0]
        assert chain.chain_id == "A"  # auto-generated
        assert chain.molecule_type == "protein"
        # Exact Chai header grammar: >{type}|name={entity_name}
        # (chai_lab/data/dataset/inference_dataset.py:227-282)
        assert result.fasta_text == f">protein|name=A\n{PROTEIN_SEQ}\n"
        assert set(result.chains_by_id) == {"A"}


class TestMultipleMoleculeTypes:
    def test_protein_dna_rna_ligand_combination(self):
        molecules = [
            {"molecule_type": "protein", "sequence": "AGSHSMRYFST"},
            {"molecule_type": "dna", "sequence": "AGTGGCTA"},
            {"molecule_type": "rna", "sequence": "AGUGGCUA"},
            {"molecule_type": "ligand", "sequence": "CCCCCCCCCCCCCC(=O)O"},
        ]
        result = build_chai_input(molecules)
        assert [c.molecule_type for c in result.chains] == ["protein", "dna", "rna", "ligand"]
        assert [c.chain_id for c in result.chains] == ["A", "B", "C", "D"]
        headers = [line for line in result.fasta_text.splitlines() if line.startswith(">")]
        assert headers == [
            ">protein|name=A",
            ">dna|name=B",
            ">rna|name=C",
            ">ligand|name=D",
        ]

    def test_no_hardcoded_protein_protein_assumption(self):
        # A single-ligand-only job must build fine; nothing should assume a
        # protein is present (task section 1: "Do NOT assume all jobs are
        # protein-protein").
        result = build_chai_input([{"molecule_type": "ligand", "sequence": "C"}])
        assert result.chains[0].molecule_type == "ligand"


class TestCopies:
    def test_copies_greater_than_one_expands_to_repeated_records(self):
        result = build_chai_input(
            [{"molecule_type": "protein", "copies": 3, "sequence": PROTEIN_SEQ}]
        )
        assert len(result.chains) == 3
        assert [c.chain_id for c in result.chains] == ["A", "B", "C"]
        # All copies carry the identical validated sequence; Chai itself
        # dedupes identical (entity_type, sequence) into one entity and
        # assigns sym_id 0..N-1 (all_atom_residue_tokenizer.py:617).
        assert all(c.sequence == PROTEIN_SEQ for c in result.chains)
        assert result.fasta_text.count(">protein|name=") == 3

    def test_copies_must_be_positive_int(self):
        with pytest.raises(MoleculeInputError):
            build_chai_input([{"molecule_type": "protein", "copies": 0, "sequence": PROTEIN_SEQ}])


class TestUserSpecifiedChainIds:
    def test_user_chain_ids_used_verbatim(self):
        result = build_chai_input(
            [
                {
                    "molecule_type": "protein",
                    "copies": 2,
                    "chain_ids": ["H", "L"],
                    "sequence": PROTEIN_SEQ,
                }
            ]
        )
        assert [c.chain_id for c in result.chains] == ["H", "L"]
        assert ">protein|name=H" in result.fasta_text
        assert ">protein|name=L" in result.fasta_text

    def test_chain_id_count_must_match_copies(self):
        with pytest.raises(MoleculeInputError, match="chain_ids"):
            build_chai_input(
                [
                    {
                        "molecule_type": "protein",
                        "copies": 2,
                        "chain_ids": ["H"],
                        "sequence": PROTEIN_SEQ,
                    }
                ]
            )

    def test_user_ids_avoided_by_autogeneration_for_other_molecules(self):
        result = build_chai_input(
            [
                {"molecule_type": "protein", "chain_ids": ["A"], "sequence": PROTEIN_SEQ},
                {"molecule_type": "ligand", "sequence": "C"},  # auto -> must not collide with "A"
            ]
        )
        ids = [c.chain_id for c in result.chains]
        assert ids[0] == "A"
        assert ids[1] != "A"
        assert len(set(ids)) == 2


class TestDuplicateChainIds:
    def test_duplicate_within_single_molecule_rejected(self):
        with pytest.raises(MoleculeInputError, match="duplicate chain ID"):
            build_chai_input(
                [
                    {
                        "molecule_type": "protein",
                        "copies": 2,
                        "chain_ids": ["A", "A"],
                        "sequence": PROTEIN_SEQ,
                    }
                ]
            )

    def test_duplicate_across_molecules_rejected(self):
        # Chai requires globally unique entity names across the whole job
        # (chai_lab/chai1.py:365-369, UnsupportedInputError).
        with pytest.raises(MoleculeInputError, match="duplicate chain ID"):
            build_chai_input(
                [
                    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": PROTEIN_SEQ},
                    {"molecule_type": "ligand", "chain_ids": ["A"], "sequence": "C"},
                ]
            )


class TestChainIdLength:
    def test_chain_id_over_4_chars_rejected(self):
        # all_atom_residue_tokenizer.py:509 packs subchain_id into a fixed
        # 4-byte tensor; longer names are silently dropped from the job at
        # tokenization time instead of raising inside Chai, so we must
        # reject them up front.
        assert MAX_CHAIN_ID_LEN == 4
        with pytest.raises(MoleculeInputError, match="4"):
            build_chai_input(
                [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["ABCDE"],
                        "sequence": PROTEIN_SEQ,
                    }
                ]
            )

    def test_chain_id_exactly_4_chars_accepted(self):
        result = build_chai_input(
            [{"molecule_type": "protein", "chain_ids": ["PROT"], "sequence": PROTEIN_SEQ}]
        )
        assert result.chains[0].chain_id == "PROT"

    def test_validate_chain_id_helper_directly(self):
        assert validate_chain_id("A") == "A"
        assert validate_chain_id("PROT") == "PROT"
        with pytest.raises(MoleculeInputError):
            validate_chain_id("TOOLONG")
        with pytest.raises(MoleculeInputError):
            validate_chain_id("")
        with pytest.raises(MoleculeInputError):
            validate_chain_id("A|B")  # would break FASTA header / CSV parsing


class TestMoleculeTypeValidation:
    def test_unknown_molecule_type_rejected(self):
        with pytest.raises(MoleculeInputError, match="molecule_type"):
            build_chai_input([{"molecule_type": "peptide", "sequence": "AAAA"}])

    def test_empty_molecule_list_rejected(self):
        with pytest.raises(MoleculeInputError):
            build_chai_input([])


class TestSequenceValidation:
    def test_dna_rejects_non_dna_letters(self):
        with pytest.raises(MoleculeInputError, match="DNA"):
            build_chai_input([{"molecule_type": "dna", "sequence": "AGQC"}])

    def test_rna_rejects_non_rna_letters(self):
        with pytest.raises(MoleculeInputError, match="RNA"):
            build_chai_input([{"molecule_type": "rna", "sequence": "AGTC"}])  # T is not RNA

    def test_dna_accepts_t_not_u(self):
        result = build_chai_input([{"molecule_type": "dna", "sequence": "AGTC"}])
        assert result.chains[0].sequence == "AGTC"

    def test_rna_accepts_u_not_t(self):
        result = build_chai_input([{"molecule_type": "rna", "sequence": "AGUC"}])
        assert result.chains[0].sequence == "AGUC"

    def test_protein_accepts_modified_residue_blocks(self):
        # Chai's own modified-residue syntax: AAA(SEP)AAA
        # (input_validation.py:15-51)
        result = build_chai_input([{"molecule_type": "protein", "sequence": "AAA(SEP)AAA"}])
        assert result.chains[0].sequence == "AAA(SEP)AAA"

    def test_malformed_bracket_syntax_rejected(self):
        with pytest.raises(MoleculeInputError):
            build_chai_input([{"molecule_type": "protein", "sequence": "AAA(SEPAAA"}])

    def test_empty_sequence_rejected(self):
        with pytest.raises(MoleculeInputError):
            build_chai_input([{"molecule_type": "protein", "sequence": "   "}])

    def test_ligand_accepts_smiles(self):
        result = build_chai_input(
            [{"molecule_type": "ligand", "sequence": "CCCCCCCCCCCCCC(=O)O"}]
        )
        assert result.chains[0].sequence == "CCCCCCCCCCCCCC(=O)O"

    def test_ligand_rejects_out_of_charset_text(self):
        # e.g. a stray space or non-SMILES punctuation
        with pytest.raises(MoleculeInputError):
            build_chai_input([{"molecule_type": "ligand", "sequence": "this is not smiles!!"}])

    def test_ligand_does_not_invent_ccd_support(self):
        # A bare 3-letter CCD-looking code ("NAG") IS valid under the SMILES
        # charset heuristic (letters only), so it will pass this syntactic
        # check -- exactly like Chai's own identify_potential_entity_types
        # would not reject it either. We only assert that we do not add any
        # *extra* CCD-specific parsing/acceptance beyond what Chai's ligand
        # parser (get_lig_residues, SMILES-only) actually does: the value is
        # forwarded verbatim as the SMILES string, untouched.
        result = build_chai_input([{"molecule_type": "ligand", "sequence": "NAG"}])
        assert result.chains[0].sequence == "NAG"


class TestParseModifiedSequence:
    def test_simple_sequence(self):
        assert parse_modified_sequence("AGT") == ["A", "G", "T"]

    def test_modified_block(self):
        assert parse_modified_sequence("AG(SEP)T") == ["A", "G", "SEP", "T"]

    def test_rejects_empty_modification(self):
        assert parse_modified_sequence("A()T") is None
        assert parse_modified_sequence("A(K)T") is None  # single-char modification

    def test_rejects_unclosed_bracket(self):
        assert parse_modified_sequence("A(SEP") is None

    def test_rejects_unopened_close(self):
        assert parse_modified_sequence("ASEP)T") is None
