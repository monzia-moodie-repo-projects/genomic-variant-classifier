# Scientific claim dispositions

**Author: Monzia Moodie**

What a historical result is still permitted to support. Package 1D, stage 1D-0, under the owner's ruling of 2026-11-08.

A reported number and the claim it was meant to support are different objects. Run 15 printed an unseen-gene AUROC (area
under the receiver operating characteristic curve) of 0.9988 and a C3 falsifier PASS. That is a true record of what the
code printed. The claim it was meant to support is not supported, because the experiment broke the assumptions that claim
needs. Each record here keeps the historical observation and states the CURRENT disposition:

| Disposition | Meaning |
|---|---|
| `VALID` | the claim may be cited as evidence |
| `INVALIDATED_EVALUATION` | the experiment violated assumptions its intended conclusion needs: the hypothesis is neither confirmed nor falsified, and how much the figure overstates is NOT estimated |
| `SUPERSEDED` | a later evaluation, named by its claim identifier, replaces this one |

The records are not typed. `scripts/derive_run15_claim_dispositions.py` writes them from Run 15's own files (every input
bound by SHA-256), and with `--check` it re-derives them and compares them byte for byte. The reader is
`src/genomic_variant_classifier/evaluation/claim_disposition.py`.

## The rules the reader enforces

- The file name is `CLAIM-NNNN_<claim_id>.json`, with NNNN = 0001, 0002, and so on, and no gaps. This README is the only
  other file allowed here.
- Each record's `record_sha256` is the canonical digest of the record without that field, and its
  `previous_record_sha256` is the previous record's digest. Editing a past record breaks the chain, so a changed
  disposition is a NEW record. A claim's current disposition is its latest record.
- An invalidated record names at least one measured defect, carries `inflation_estimate: null`, and resolves no hypothesis.
- In the documents that state present claims (`README.md`, `docs/METRICS.md`, `docs/ROADMAP.md`,
  `docs/runs/RUN17_SCOPE.md`), a figure of an inadmissible claim may appear only in a paragraph or table row that also
  carries its disposition and its claim identifier.
- A historical record that stated facts measured false opens with a dated correction:
  `docs/ABLATION_n_pathogenic_in_gene.md`.
- The live unseen-gene holdout design (`scripts/run_phase2_eval.py`) is the one that invalidated C3. It records its verdict
  as WITHHELD, with the defects named, until Package 1D replaces it.

## Current records

| Record | Claim | Disposition |
|---|---|---|
| 0001 | `run15.c3.unseen_gene_generalization` | INVALIDATED_EVALUATION |
| 0002 | `run15.test.ensemble_evaluation` | INVALIDATED_EVALUATION |
| 0003 | `run15.validation.ensemble_evaluation` | INVALIDATED_EVALUATION |

The defects, each measured:
- **`GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE`** (all three records): the gene count was computed over the whole cohort
  before the split.
- **`GENE_IDENTITY_SPLIT_LEAKAGE`** (all three): groups were gene-symbol strings.
- **`GNN_HOLDOUT_LABEL_LEAKAGE`** (C3): the graph neural network was trained on the holdout's labels before the holdout
  was carved.
- **`VARIANT_LEVEL_CLINICAL_CLASS_FEATURE`** (C3 and test): LOVD's own clinical class was an input.
