# INCIDENT 2026-09-20 — Review-status VCF join failure silently removed 97% of deletions

## Summary

`augment_reviewstatus.py` reconstructed the cohort's top-level `ReviewStatus`
column by joining ClinVar's VCF on `chrom:pos:ref:alt`, mapping unmatched keys
to `""`. Because ClinVar's `variant_summary` and its VCF use different position
conventions for indels, that join failed for **97.33% of deletions**. An empty
string resolves to the missing-evidence tier, so `DataPrepPipeline._load_and_label`
excluded those variants from every cohort built through this path.

The technical failure was a retrieval failure. It was recorded as an absence of
scientific evidence. That is the defect.

## Measured effect (read-only audit, `outputs/review_status_audit_001/`)

Identity universe: 4,399,089 rows. Label policy identical in both arms.

| Population | Rows |
|---|---:|
| Eligible under legacy (top-level `ReviewStatus`) | 1,490,014 |
| Eligible under corrected (`metadata.review_status`) | 1,620,238 |
| Added by the correction | 130,224 |
| Removed by the correction | 0 |

The corrected cohort is a strict superset. Accounting closes exactly:
1,490,014 + 130,224 + 2,778,851 excluded-both = 4,399,089.

Additions by representation class:

| Class | Added | Share |
|---|---:|---:|
| net_length_loss (deletions) | 129,788 | 99.67% |
| SNV | 422 | 0.32% |
| net_length_gain | 14 | 0.01% |

Label-eligible deletions captured by the legacy cohort: 3,621 of 133,409 = **2.71%**.

Positive-class prevalence moves 14.13% -> 18.53%
(pathogenic 210,549 -> 300,262; benign 1,279,465 -> 1,319,976).

## Root cause (diagnostic, `outputs/review_status_audit_001/join_failure_diagnosis.json`)

Of 20,000 sampled failed deletion joins, **99.92% match the VCF at `pos - 1`**.
SNV failures behave differently: 96.72% match no candidate key form, consistent
with genuine absence from the VCF rather than a coordinate mismatch.

Failure-mode crosstab over the label-eligible population shows **zero** rows with
genuinely absent nested review evidence. Every label-eligible blank top-level
value had substantive nested evidence. All 10 disagreeing status pairs are
blank-top vs populated-nested; there is not one case where both sources hold
different substantive values.

## Repair experiment (`outputs/repair_experiment_v1/`)

Two arms differing ONLY in cohort eligibility policy. One frozen, label-free,
hash-based gene->partition registry (`policy_id review-repair-split-policy-1`,
weights 7/1/2) shared by both arms, so train/validation/test GENE membership is
identical and only row eligibility differs. Feature definitions, constraint
source, model families, preprocessing policy and metrics frozen. Feature space
verified identical across arms (13 columns). Evaluation genes disjoint from
training genes for both arms (overlap 0).

Training: legacy 1,034,163 rows / 12,284 genes; corrected 1,124,720 / 12,385.
Evaluation on the test partition: common 282,638 rows, added 23,151 rows.

Paired Brier deltas (negative = corrected better), gene-cluster bootstrap,
n_boot 2000, verdicts stable across seeds 0-4:

| Model | Cell | Delta | 95% CI | Design effect | Verdict |
|---|---|---:|---|---:|---|
| Logistic regression | common | -0.000696 | [-0.000908, -0.000509] | 4.56 | excludes zero |
| Logistic regression | added | -0.127358 | [-0.157310, -0.096308] | 19.18 | excludes zero |
| LightGBM | common | -0.001348 | [-0.003259, +0.000107] | 16.56 | **INCLUDES ZERO** |
| LightGBM | added | -0.004421 | [-0.005985, -0.003009] | 3.25 | excludes zero |

## What this establishes, by claim level

**Engineering — established.** A representation-dependent join failure changed
cohort eligibility for 130,224 variants, 99.67% of them deletions, removing 97.3%
of the eligible deletion population.

**Statistical — established, with a negative primary result.** The repair does
**not** demonstrably improve prediction on the population that was already
represented: LightGBM's common-cell interval includes zero in 5 of 5 seeds. The
logistic-regression common-cell effect excludes zero but is minuscule (-0.0007).
The repair's demonstrated value is recovering a near-absent variant class, not
improving performance on existing data.

**Biological — not established.** No biological claim is made here.

## Design-effect finding

Gene-cluster resampling widens intervals by **3.2x to 19.2x** over naive row-level
resampling on these same data. Any interval previously computed at row level in
this project understated its uncertainty by approximately that factor. The
LightGBM common-cell result specifically flips from apparently-negative at face
value to indistinguishable from zero once gene dependence is respected.

## Composition differences (reported, not adjusted away)

| Property | common | added |
|---|---:|---:|
| Prevalence | 0.1334 | 0.6571 |
| Genes | 3,618 | 1,090 |
| Rows per gene | 78.1 | 21.2 |
| LOEUF present | 98.23% | 98.67% |
| mis_z present | 98.68% | 98.87% |

Annotation availability is effectively equal, so cell differences are not
attributable to missing constraint data. Prevalence differs ~4.9x, so AUPRC is
**not** comparable across cells; only paired within-cell contrasts are.

Logistic regression's added-cell AUROC is identical across arms to four decimals
(0.9189 vs 0.9189) while Brier improves 0.127. That gain is **calibration**, not
discrimination — a model fit at 14% prevalence scored on a 66% population.

## Superseded and retracted

- An earlier figure of "9.84% of the labeled population" was wrong. The change was
  computed over all 4,399,089 rows (3.77%); the correctly scoped final eligibility
  change is 130,224.
- `is_frameshift` (`abs(len(alt)-len(ref)) % 3 != 0`) is retired. It measures a
  genomic allele-length property, not a transcript frameshift consequence.
- The earlier `alt_len` / frameshift label-rate analysis is superseded: it was
  computed on a cohort whose indels were 94.9% insertions because deletions were
  the missing class.
- The v3 "locked confirmation" is not independent confirmation. Excluding training
  genes does not screen validation rows, test rows, or genes examined during
  development.

## Upstream repair — COMPLETE AND VERIFIED (2026-09-20)

`derive_review_status.py` replaces the VCF-join derivation. It reads the canonical
record already stored at ingestion (`metadata.review_status`), never fabricates a
value, and writes a NEW cohort file rather than modifying in place.

Three properties of the original combined to cause the defect; all three are closed:

| Original behaviour | Replacement |
|---|---|
| Joined a second source (VCF) on `chrom:pos:ref:alt` | Reads the canonical ingestion record |
| Unmatched keys became an empty string | `ReconciliationFailure` raised; never filled |
| Idempotency was mere column presence | Compares content and policy digest |

Run against the production cohort, the content-based idempotency guard detected
disagreement on **424,516 of 4,399,089 rows** — matching the independent audit's
disagreement total exactly — and refused until overwrite was requested deliberately.

Repaired cohort: `data/processed/clinvar_grch38_canonical_review.parquet`
sha256 `66cee5348a8fe9ad277ccc181029acecf882bcd316207dba97f346b194e094eb`
(agrees 3,974,573 / disagrees 424,516).

**Closure proof.** `build_split_registry.py` computes eligibility under both sources
independently and has no knowledge of the repair. On the repaired cohort:

    legacy members:    1,620,238
    corrected members: 1,620,238
    membership cells: {common: 1620238, added: 0, removed: 0, excluded_both: 2778851}
    registry sha256: c0ae0fac1a14a14212554bd4122ec215b811e1cf29c9eed5cd7b9fe66ecb77ba

`added` was 130,224 before the repair and is now **0**. The registry hash is
byte-identical to `split_registry_v1`, confirming the gene population (17,828) did
not shift. The repaired cohort's legacy arm reproduces, from the top-level column
alone, exactly the population that previously required nested metadata: 1,620,238
rows and 300,262 positives, matching to the row.

Acceptance tests (`test_derive_review_status.py`, 12 passing) fail closed on each
failure mode: unmatched join, absent metadata, metadata without the review key,
unknown vocabulary, stale existing column, and output overwrite. The round-trip
test asserts classification, review status and provenance digests survive
materialisation and re-read.

## Remaining open

- Interaction interval: COMPUTED (`outputs/repair_experiment_v1/interaction_ci.json`)
  via `cluster_bootstrap_paired_contrast_ci`, a proposed addition to
  `evaluation.metrics` that reuses the reviewed cluster-resampling design and draws
  both cells inside ONE gene draw. Its point estimates reproduce the values
  `run_repair_experiment.py` already reported, so it adds an interval without
  moving the estimate.

  | Model | Interaction | 95% CI (seed 0) | Closest margin from zero | Reading |
  |---|---:|---|---:|---|
  | Logistic regression | -0.126662 | [-0.157295, -0.096220] | 158% of interval width | robust exclusion |
  | LightGBM | -0.003073 | [-0.005580, -0.000063] | 0.14% of interval width | nominal only |

  Negative means the corrected model improves the added cell more than the common
  cell. Both verdicts were stable across seeds 0-4, but stability is not strength:
  LightGBM's upper bound reached -0.000008 in one seed. That is NOT robust evidence
  of an interaction, and it should not be reported as one. It is also consistent
  with the primary result, where LightGBM's common-cell delta includes zero --
  little or no gain on the existing population, a small gain on the recovered one.

  The estimator currently lives at `baseline/interaction/paired_contrast.py` rather
  than inside `evaluation/metrics.py`. Installing it there is the right destination
  but edits reviewed shared code and is a separate, deliberate decision.
- Exposure ledger: BUILT (`outputs/exposure_ledger_v1/`). 1,430,509 entries --
  1,124,720 training, 305,789 test_feedback -- covering 16,021 of 21,386 genes
  (74.91%). Two gaps are recorded in the summary rather than papered over:
  baseline_run1 exposure (artifacts deleted, split reconstructible but not
  reconstructed), and cohort-wide aggregate label reading that preceded feature
  and cohort decisions.

  The 5,365 unexposed genes decompose exactly: 3,558 hold ONLY never-eligible
  variants (no usable binary labels, so unusable as confirmation regardless of
  exposure) and 1,807 are the validation partition. Within this ClinVar snapshot
  the only usable unexposed population is therefore the validation partition:
  1,807 genes / 189,729 rows / 37,475 positives -- and it still sits under the
  aggregate-label gap above.

  **Consequence: genuine independent confirmation for this project requires a
  population outside this snapshot, realistically a future ClinVar release.**
  That is a real constraint on what can be claimed, not a bookkeeping shortfall.
  `confirmation_screen` reports `passes_recorded_exposure_screen`, never
  `independent`, precisely because an incomplete ledger cannot prove independence.
- Historical runs built through the old path trained on the deletion-depleted
  population. Their results remain valid results on their actual populations;
  their interpretation needs the correction recorded here.

## Artifacts

All twenty-one verified present on 2026-09-20 by `baseline/verify/verify_artifacts.py`;
inventory with digests at `outputs/artifact_verification.json`.

**These artifacts are NOT in version control.** `.gitignore:100` excludes `outputs/`
and `data/processed/.gitignore:2` excludes that directory entirely, so every path
below exists only on the machine that produced it. A clone of this repository will
contain this record but none of the evidence it cites. The digests in the table at
the end of this section are therefore the only portable proof that the reported
numbers came from the files named here; reproducing the artifacts requires re-running
the scripts against the same ClinVar and gnomAD snapshots.

| Path | Contents |
|---|---|
| `outputs/review_status_audit_001/summary.json` | Eligibility reconciliation |
| `outputs/review_status_audit_001/review_source_disagreements.csv` | Row-level disagreements (64 MB) |
| `outputs/review_status_audit_001/join_failure_diagnosis.json` | Coordinate-convention diagnosis |
| `outputs/review_status_audit_001/failure_mode_crosstab.json` | Retrieval failure vs genuine absence |
| `outputs/cohort_corrected_review_v1/cohort_manifest.json` | Superseded two-variable build, retained as evidence |
| `outputs/cohort_corrected_review_v2/cohort_manifest.json` | Corrected cohort manifest |
| `outputs/cohort_corrected_review_v2/cohort_corrected.parquet` | Corrected cohort |
| `outputs/cohort_corrected_review_v2/decision_table.parquet` | Row-level inclusion reasons over the whole universe |
| `outputs/split_registry_v1/split_registry_manifest.json` | Registry manifest (pre-repair cohort) |
| `outputs/split_registry_v1/gene_partition_registry.parquet` | Frozen gene->partition assignment |
| `outputs/split_registry_v1/membership_and_partition.parquet` | Per-row membership and partition |
| `outputs/split_registry_canonical_v1/split_registry_manifest.json` | Closure proof: added 0, removed 0 |
| `outputs/repair_experiment_v1/repair_experiment_report.json` | Two-by-two results |
| `outputs/repair_experiment_v1/evaluation_predictions.parquet` | Paired per-row predictions |
| `outputs/repair_experiment_v1/uncertainty.json` | Gene-cluster bootstrap intervals, design effects |
| `outputs/repair_experiment_v1/stability_and_coverage.json` | Seed stability, annotation availability |
| `outputs/repair_experiment_v1/interaction_ci.json` | Interaction intervals across seeds |
| `outputs/exposure_ledger_v1/exposure_ledger.parquet` | 1,430,509 exposure entries |
| `outputs/exposure_ledger_v1/exposure_ledger_summary.json` | Coverage and recorded gaps |
| `data/processed/clinvar_grch38_canonical_review.parquet` | Repaired cohort from canonical derivation |
| `data/processed/clinvar_grch38_canonical_review.parquet.derivation.json` | Derivation provenance |

Integrity, not merely presence. Three artifacts were re-digested and matched the
values recorded independently at write time, so they are byte-identical to what
produced the reported numbers:

| Artifact | Digest source | Result |
|---|---|---|
| `review_source_disagreements.csv` | `summary.json` `detail_sha256` | match |
| `cohort_corrected.parquet` | `cohort_manifest.json` `corrected_cohort_sha256` | match |
| `clinvar_grch38_canonical_review.parquet` | `.derivation.json` `output_sha256` (full-file, 142 MB) | match |

The remaining eighteen were verified present with size and digest recorded in
`outputs/artifact_verification.json`, but have no independently recorded
write-time digest to compare against.







