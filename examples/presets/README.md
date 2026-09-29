# Example presets

CSV files loaded by the "Try an example" buttons of `streamlit_app.py`.
The format and validation rules are documented in `presets.py`.

| File | Button | Status |
|---|---|---|
| `compact_3chain_restraints.csv` | Compact 3-chain restraints demo | User-facing |
| `small_msa_template_candidate.csv` | (hidden) | Candidate, awaiting HPC validation |

## Sources

- **compact_3chain_restraints.csv**: the three 7SYZ protein chains
  (604 / 223 / 221 residues) from the official chai-lab example
  `examples/restraints/predict_with_restraints.py` (commit `66c38d1`), with
  the two contact restraints `A C387 - B Y101` and `C I32 - A S483`
  (same residues as the official `examples/restraints/contact.restraints`),
  at 5 Å. No MSA, no templates.
- **small_msa_template_candidate.csv**: human ubiquitin, 76 residues, PDB
  [1UBQ](https://www.rcsb.org/structure/1UBQ) chain A
  (`https://www.rcsb.org/fasta/entry/1UBQ`; identical to UniProt
  [P0CG48](https://rest.uniprot.org/uniprotkb/P0CG48.fasta) residues 1-76).
  Chosen because it is small and has many homologs and deposited structures,
  so MSA and template search have a realistic chance to succeed. MSA +
  templates on, no restraints.

## Promoting the MSA + template candidate

The candidate is not shown to users until it has completed the real HPC
workflow (MSA generation, template retrieval, Chai inference, real outputs):

1. Start the app with `CHAI_WEB_SHOW_CANDIDATE_PRESETS=1 bash run_web.sh`,
   load "Small MSA + template demo (candidate, admin only)" and submit it.
2. Check the job's Slurm log for successful MSA and template steps, and the
   result page for real CIF / scores / PAE.
3. Rename the file (`git mv small_msa_template_candidate.csv
   small_msa_template.csv`) and commit. It then appears as
   "Small MSA + template demo" for everyone.
