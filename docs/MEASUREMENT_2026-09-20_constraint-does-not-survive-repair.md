# MEASUREMENT 2026-09-20 — Constraint extension does not survive cohort repair

Supersedes the legacy-cohort core-versus-constraint result (+0.0088 AUROC, no
uncertainty estimate). Answers the governing ruling question: *"whether gains
from gene constraint persist after cohort repair."* They do not.

## Design

Two feature tiers differing ONLY by `loeuf`, `mis_z` and their explicit
missingness indicators. Everything else frozen: feature definitions, constraint
source policy, model families, preprocessing policy, metrics. Preprocessing
fitted on training rows only.

- Cohort: `clinvar_grch38_canonical_review.parquet` (repaired), gene registry
  `split_registry_canonical_v1`, policy `review-repair-split-policy-1`, weights 7/1/2
- Train: 1,124,720 rows / 12,385 genes / prevalence 0.186605
- Evaluate: **validation** partition, 189,729 rows / 1,807 genes / prevalence 0.197519
- Gene disjointness train/validation verified, overlap 0
- Feature widths: core 9, core_plus_constraint 13 (enforced to differ)

The test partition was deliberately NOT used. It is recorded in
`exposure_ledger_v1` as `test_feedback`; the ruling forbids repeating model
choices against an already-inspected test population.

## Per-tier metrics on validation

| Model | Tier | AUROC | AUPRC | Brier |
|---|---|---:|---:|---:|
| Logistic regression | core | 0.7487 | 0.7130 | 0.08346 |
| Logistic regression | core+constraint | 0.7498 | 0.7325 | 0.08340 |
| LightGBM | core | 0.9642 | 0.8542 | 0.05453 |
| LightGBM | core+constraint | 0.9602 | 0.8745 | 0.06069 |

AUROC and AUPRC are comparable across tiers here because the population is
identical, but they carry NO uncertainty estimate and are not established.
Only the paired Brier contrast below has an interval.

## Paired tier contrast (positive = constraint tier has HIGHER Brier loss)

Gene-cluster bootstrap, n_boot 2000, verdicts stable across seeds 0-4.

| Model | Delta | 95% CI (seed 0) | Design effect | Verdict |
|---|---:|---|---:|---|
| Logistic regression | -0.000068 | [-0.000441, +0.000231] | 8.75 | includes zero |
| LightGBM | **+0.006159** | [+0.000943, +0.010628] | 15.15 | **excludes zero** |

For the stronger model the constraint extension makes probability quality
measurably WORSE. For logistic regression there is no detectable effect.
Design effects of 8-16x are consistent with the 3.2x-19.2x measured in the
repair experiment: any row-level interval on these data would be far too narrow.

## Transfer diagnostic — hypothesis NOT supported

Proposed mechanism: `loeuf` and `mis_z` are GENE-level; every validation gene is
unseen; gene-level constraint might support calibration within seen genes and
mislead on unseen ones. Falsifiable prediction: harm concentrates where the
annotation is PRESENT, contrast positive and excluding zero.

Strata taken from EXECUTION-MATCHED availability persisted by the run, not a
recomputed join.

| Stratum | Rows | Genes | Prevalence | LightGBM delta | Verdict |
|---|---:|---:|---:|---:|---|
| constraint present | 186,297 | 1,569 | 0.197529 | +0.005976 | excludes zero (5/5) |
| constraint absent | 3,432 | 238 | 0.196970 | +0.016115 | includes zero (5/5) |
| contrast (present - absent) | | | | **-0.010139** | **includes zero (5/5)** |

The point estimate runs OPPOSITE to the prediction: absent is larger, not
smaller. The contrast interval spans zero, so no difference between strata is
established.

Logistic regression shows nothing in either stratum (present -0.000085, absent
+0.000840, contrast -0.000925; all include zero).

**Reading.** The hypothesis is not supported. It is also not refuted: the absent
stratum is 1.81% of rows across 238 genes with an interval roughly 4.8x wider
than the present stratum, so an effect of the hypothesised size could not have
been resolved. What IS established is that the harm appears where constraint
data exists and is not shown to depend on whether a gene has constraint data at
all -- which weakens, without excluding, the gene-specific-value story, since
harm appears even where the model sees only an imputed median and a flag.

Strata are not prevalence-confounded (0.197529 vs 0.196970), and the
row-weighted reconstruction of the stratum deltas reproduces the overall
+0.006159 exactly, so the stratification is arithmetically sound.

## Consequence for the ruling's step 7

Step 7's four-arm model-representation comparison assumed constraint features
were in the admissible set. Including them would now confound the representation
question with a feature that measurably harms the stronger model. Constraint
should be excluded from step 7's feature set, or step 7 run at both tiers.

## Provenance

v1 and v2 differ ONLY in persistence: v2 additionally writes execution-matched
feature and availability columns (17 columns vs 7). All contrast values are
identical between them, confirming no behavioural change.

| Artifact | Contents |
|---|---|
| `outputs/constraint_remeasure_v1/` | first run, identity + predictions only |
| `outputs/constraint_remeasure_v2/constraint_remeasure_report.json` | run manifest, input digests |
| `outputs/constraint_remeasure_v2/validation_predictions.parquet` | 189,729 x 17, execution-matched |
| `outputs/constraint_remeasure_v2/transfer_diagnostic.json` | stratified diagnostic |
| `baseline/constraint/run_constraint_remeasure.py` | experiment |
| `baseline/transfer/run_transfer_diagnostic.py` | diagnostic |
| `baseline/transfer/paired_contrast.py` | between-strata estimator (proposed addition to evaluation.metrics) |

Artifacts are gitignored and local-only. A portable copy with a SHA-256 manifest
is at `Downloads/GVC_constraint_remeasure_2026-09-20`.

