# Audit -- 2026-07-24 -- AlphaFold structural coverage of the variant cohort

**Status:** measurement complete for the questions it set out to answer. One figure --
the variant-weighted exposure -- is stated against a named denominator and carries an
explicit cross-check that had not yet been run when this document was written. Every
other number here is final.

**Repository state throughout:** `main` at
`715bcfa1bdd8320718c7fa9135834ce1c66d9a59`, "feat(evaluation): typed contracts for Panel
S0 expert identity", 2026-07-23T09:57:33-04:00. No commit was made during the audit.

**Relationship to other documents.** The defect being measured is recorded in
`docs/INCIDENT_2026-07-23_protein_pipeline_alphafold_fetch.md`. That note establishes
*what* is wrong in the code. This document establishes *how much* of the cohort it
touches. The two should be read together and neither supersedes the other.

---

## 1. What was found, in one paragraph

The AlphaFold Protein Structure Database does not model proteins above a length ceiling
that this audit located at 2,699 or 2,700 residues. Of the 20,190 accessions in the
local reviewed-human UniProt index, 215 exceed it. For every one of those 215, the
European Bioinformatics Institute prediction application programming interface returns
either nothing at all or only shorter isoform models --
`src/genomic_variant_classifier/pipelines/protein_pipeline.py:171` takes `data[0]`
unconditionally and therefore attaches an isoform's structure to a canonical protein's
residue numbering, silently. That is 1.065 percent of genes. Because long genes carry
disproportionately many variants, it is **12.725 percent of the ClinVar clean cohort** --
559,786 of 4,399,089 variants -- of which **220,590 receive structural feature values
that are wrong rather than absent**. The four Phase D structural features
(`alphafold_plddt`, `solvent_accessibility`, `secondary_structure_context`,
`dist_to_active_site`) are the affected outputs.

---

## 2. Chronology of the measurement

Every step is timestamped in Coordinated Universal Time as printed by the tools.

| Time (UTC) | Step | Result |
| --- | --- | --- |
| 2026-07-23T17:34:51Z | Endpoint probe, version 1 | AlphaFold parameterless 404, parameterised HEAD 405, parameterised GET 200. LOVD 403 on every form. |
| 2026-07-23T17:50:09Z | Endpoint probe, version 2 | `model_v6` served; `latestVersion` 6; `modelCreatedDate` 2025-08-01T00:00:00Z. 8 records for P38398, 9 for P04637. |
| 2026-07-23T18:17:27Z | UniProt index inspection | 20,190 rows; columns `gene_symbol, uniprot_id, entry_name, sequence`; zero nulls; zero duplicate accessions. Stale-index hypothesis refuted. |
| 2026-07-24T02:23:25Z | Random sample, n=60 | 1 mismatch of 34 informative trials: Q5T4S7. |
| 2026-07-24T02:33:02Z | Record-level probe | Q5T4S7 classified ISOFORM_ONLY. Controls P38398, P04637, Q8IZ96 all CANONICAL_PRESENT at index 0. |
| 2026-07-24T02:33:42Z | Random sample, n=300 | 1 of 159 informative; 5 outright HTTP 404s; all five in the top 1 percent by length. |
| 2026-07-24T03:18:22Z | Length-stratified audit, n=200 | Failure confined to the top length strata. Ceiling bracketed to (2,477, 2,701]. |
| 2026-07-24T03:31:29Z | **Complete census, n=296** | **The definitive measurement. See section 4.** |
| 2026-07-24T03:35:17Z | Record probe on the drift cases | Refuted the causal text the census printed. See section 6. |

Two of the samples must not be pooled: `random.sample` with a fixed seed produces nested
draws, so the 60-accession run is a strict prefix of the 300-accession run. This was
verified rather than assumed, and the 300-run supersedes the 60-run entirely.

---

## 3. The ceiling, located rather than assumed

No value was assumed at any point. The census swept every accession from 2,400 residues
upward, which covers the band the stratified run left unobserved.

    longest canonical length WITH a canonical model : 2,699 residues  (Q9P273, TENM3)
    shortest canonical length WITHOUT one           : 2,701 residues  (Q14571 ITPR2;
                                                                      Q02224 CENPE)

The window contains exactly one integer, 2,700, and the index holds **no protein of that
length** -- the census jumps from 2,701 straight to 2,699. So 2,700 is untestable from
this index, and the ceiling is 2,699 or 2,700. Nothing else is consistent with the data.
The widely cited AlphaFold human-proteome limit of 2,700 residues falls inside that
window; the measurement neither needed it nor contradicts it.

The split is total. Every one of the 81 accessions at or below 2,699 residues returned
`CANONICAL_PRESENT`. Every one of the 215 at or above 2,701 residues did not. There is
no overlap band, so length behaves as a hard cutoff here and may be modelled as one.

---

## 4. The census

Complete enumeration of all 296 index accessions at or above 2,400 residues. This is a
census, not a sample: there is no sampling error in these figures.

| Verdict | Count | Share of 296 |
| --- | --- | --- |
| `CANONICAL_PRESENT` | 81 | 27.36 % |
| `ISOFORM_ONLY` -- wrong data | 102 | 34.46 % |
| `NO_MODEL_404` -- missing data | 109 | 36.82 % |
| `SEQUENCE_VERSION_DRIFT` | 4 | 1.35 % |

81 + 102 + 109 + 4 = 296. Affected = 296 - 81 = **215**, which agrees exactly with the
independent count of index accessions above 2,700 residues (215) computed from the length
distribution. That agreement is an internal consistency check, not a coincidence.

**Wrong data and missing data are different harms and are counted separately throughout.**
A missing structure yields a sentinel and a coverage miss. A substituted isoform yields
numbers that look valid and are not. For Q5T4S7 (UBR4, 5,183 residues) `data[0]` is
`AF-Q5T4S7-6-F1`, a **212-residue** model sharing exactly **one residue** with the
canonical sequence.

---

## 5. Variant-weighted exposure

A gene-weighted rate is the wrong statistic for a variant pathogenicity classifier. Long
genes accumulate more variants.

**Denominator:** `data/processed/clinvar_grch38_clean_seq.parquet`, 4,399,089 rows, every
row carrying a gene symbol. That row count is documented: `docs/CHANGELOG.md` records for
the 2026-05-31 null-key leak remediation, "Emitted
`data/processed/clinvar_grch38_clean.parquet` (4,399,089 rows; 0 null, 0 dup)" and
"Reconciliation identity verified exact (4,420,180 = 21,091 + 4,399,089)". The `_seq`
variant carries the same rows plus sequence-window columns.

| Group | Variants | Share of 4,399,089 |
| --- | --- | --- |
| Wrong data -- isoform substituted | 220,590 | **5.014 %** |
| Missing data -- no model at all | 339,141 | **7.709 %** |
| Sequence drift | 55 | 0.001 % |
| **Combined** | **559,786** | **12.725 %** |
| Gene-weighted, for contrast | 215 / 20,190 | 1.065 % |

**The variant-weighted figure is 11.95 times the gene-weighted figure.** One variant in
every 7.9 sits in an affected gene; one in every 19.9 receives structural values that are
actively wrong.

### An outstanding qualification on this denominator

Runs 15 and later do **not** train on all 4,399,089 variants. `docs/CHANGELOG.md` records
a downstream filter -- "Review-tier <=3 retained 88% (1,686,333 -> 1,490,014)" -- and
`docs/METRICS.md` line 131 carries the same caveat against the area-under-the-receiver-
operating-characteristic-curve column. The review-tier filter is not gene-uniform, so the
exposure on the training cohort may differ from the exposure on the clean cohort. Until
`crosscheck_variant_exposure_2026-07-24.py` has been run, **12.725 percent is the figure
for the clean cohort and must be quoted with that cohort named.**

### Most affected genes by variant count

Wrong data -- structural values are substituted from an isoform:

    NF1 16,807 | DMD 10,058 | USH2A 9,377 | ADGRV1 7,086 | SYNE1 7,048 | VPS13B 6,545
    CDH23 6,184 | FLNC 6,168 | DSP 6,084 | EYS 5,070 | SYNE2 4,807 | CHD7 4,419
    DYNC2H1 4,219 | ANK2 4,191 | DST 4,089 | LYST 4,056 | COL12A1 4,040 | FBN2 4,039
    COL6A3 3,956 | HMCN1 3,858

Missing data -- no structure at all:

    TTN 39,316 | BRCA2 21,166 | ATM 19,121 | APC 16,675 | NEB 12,377 | RYR1 11,050
    RYR2 9,889 | FBN1 9,208 | KMT2D 7,411 | ALMS1 7,342 | PKD1 6,889 | PLEC 6,668
    DNAH11 6,584 | DNAH5 6,564 | PKHD1 6,562 | COL7A1 6,210 | OBSCN 5,822
    DYNC1H1 5,605 | LAMA2 5,420 | ZNF469 5,247

These are not peripheral. BRCA2, ATM and APC are among the most clinically consulted
cancer-predisposition genes in the corpus and have **no AlphaFold structure at all**
through this path. NF1 and DMD have the largest wrong-data counts.

---

## 6. Sequence-version drift, and a correction to a claim this audit itself printed

Four accessions in the census, plus one found earlier at 173 residues, returned a
canonical-shaped entry whose sequence does not match ours:

| Accession | Gene | Our length | AlphaFold length | Common prefix | AlphaFold `sequenceVersionDate` |
| --- | --- | --- | --- | --- | --- |
| Q0P5N6 | ARL16 | 173 | 197 | 1 | 2006-09-19 |
| Q6ZTK2 | APOLTP | 3,320 | 550 | 1 | 2004-07-05 |
| Q9H195 | MUC3B | 13,477 | 1,237 | 122 | 2023-09-13 |
| Q6P3W6 | NBPF10 | 3,795 | not probed individually | -- | -- |
| Q5TI25 | NBPF14 | 2,988 | not probed individually | -- | -- |

**The census printed "Our index is the stale party" for these. That is wrong**, and the
record-level probe of 2026-07-24T03:35:17Z is what refuted it. The `sequenceVersionDate`
fields are 2006, 2004 and 2023 against a local index built on 2026-06-25. AlphaFold's
UniProt snapshot is the older one. Rebuilding the local index would not fix these and is
not a remediation. The text was a causal assertion made without evidence and is corrected
here.

Two consequences follow. First, drift defeats `scripts/build_alphafold_parquet.py` as
well -- the reference implementation matches on canonical sequence, finds nothing, and
records a silent coverage miss for a gene it should cover. Second, and separately:
**for the production path, drift is not a distinct category at all.**
`protein_pipeline.py:171` takes `data[0]` regardless, so a drift case attaches a
550-residue model to a 3,320-residue protein exactly as an isoform case would. Grouped by
production harm rather than by cause, wrong data is 102 + 4 = **106 accessions** and
220,590 + 55 = **220,645 variants**. The census grouping under-reported the wrong-data
class by treating drift as separate.

Numerically the drift correction is small -- 55 variants, 0.001 percent. Conceptually it
matters, because it shows the taxonomy has to be chosen to match the question being asked.

---

## 7. What this means for the four Phase D features

`src/genomic_variant_classifier/data/real_data_prep.py:314` lists
`ProteinStructurePipeline` as step 14 of feature engineering; line 1070 imports it and
line 1085 instantiates it. `protein_pipeline.py:502` is the sole call site of the
defective fetch.

For the 220,590 variants in wrong-data genes, `alphafold_plddt`,
`solvent_accessibility`, `secondary_structure_context` and `dist_to_active_site` are read
at residue indices belonging to a different sequence. The values are dimensionally valid,
within range, and wrong. For the 339,141 variants in missing-data genes they are
sentinels, which is honest but silent -- the fetch failure is logged at `DEBUG` at
`protein_pipeline.py:166`, `:181`, `:299` and `:536`, below the default threshold.

**No claim is made here about the effect on model performance.** Feature importance,
ablation and the effect on the area under the receiver operating characteristic curve are
separate measurements that have not been run. Stating an effect without measuring it
would be the error this audit exists to avoid.

---

## 8. Open items

| # | Question | How it gets answered |
| --- | --- | --- |
| 1 | Exposure on the review-tier <=3 training cohort of 1,490,014, not just the 4,399,089 clean cohort | `crosscheck_variant_exposure_2026-07-24.py` |
| 2 | Do the different cohort files agree, or is 12.725 percent an artefact of auto-selection | same script; it reports the spread and refuses a single number if they diverge |
| 3 | Effect on trained-model performance | ablation, not yet designed |
| 4 | Where the 8.77 gigabyte, 18,079-file AlphaFold cache recorded at `.gitignore:203` now lives | unanswered; it is the population for the cache-invalidation defect |
| 5 | Whether any shipped run artifact was produced through this path | run-artifact trace, not yet designed |

---

## 9. Remediation direction

Unchanged in substance from the incident note, and now quantified.

1. Promote the correct resolution logic out of `scripts/build_alphafold_parquet.py` into
   one shared library function: canonical-sequence match, `None` on no match, filename
   from the server's `cifUrl`.
2. Rewire `protein_pipeline.py::_fetch_alphafold_cif` to call it, so isoform
   substitution becomes structurally impossible rather than merely discouraged.
3. Move the existing assertions in `tests/unit/test_alphafold.py:329-343` onto the shared
   function so both callers are covered.
4. Raise the four `DEBUG` logs to `WARNING` and replace the bare
   `except Exception: pass` at `protein_pipeline.py:275-276`.
5. Add a syntax-tree guard test forbidding any hard-coded `model_v[0-9]+` literal outside
   `tests/fixtures/`, modelled on `tests/unit/test_rnaseq_ablation_native_read.py`.
6. Add a guard test asserting record selection is by canonical-sequence match, so
   `data[0]` cannot return.
7. Decide, explicitly and separately, what a coverage miss should mean for a gene above
   the ceiling. A sentinel is honest; it is not obviously the best available answer for
   215 genes carrying 12.7 percent of the corpus. Alternatives -- an explicit
   `structure_available` indicator feature, or per-domain models for the modelled
   fragments -- are scope decisions for Monzia, not implementation details.

Steps 3, 5 and 6 add tests and each needs its own suite-size ratchet accounting.

---

## 10. Provenance of every figure in this document

| Figure | Source |
| --- | --- |
| Index composition, 20,190 accessions and length quantiles | `audit_alphafold_length_strata_2026-07-24.py`, 2026-07-24T03:18:22Z |
| Ceiling 2,699 / 2,701 | `audit_alphafold_giant_census_2026-07-24.py`, 2026-07-24T03:31:29Z |
| Census counts 81 / 102 / 109 / 4 | same |
| Variant counts and the 12.725 percent | same, joined against `clinvar_grch38_clean_seq.parquet` |
| 4,399,089 clean-cohort row count | `docs/CHANGELOG.md`, 2026-05-31 null-key leak entry |
| 1,686,333 to 1,490,014 tier filter | `docs/CHANGELOG.md`; `docs/METRICS.md:131` |
| Drift sequence lengths and version dates | `probe_alphafold_records_2026-07-23.py`, 2026-07-24T03:35:17Z |
| Clopper-Pearson intervals quoted in earlier turns | computed with `scipy.stats.beta.ppf`; superseded by the census, which has no sampling error |

Raw evidence belongs under `docs/audits/evidence/2026-07-24/`.
