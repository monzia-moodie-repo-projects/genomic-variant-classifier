# Measurement 2026-07-24 -- the `pathogenicity` column predates the 2026-07-10 mapping fix

## 1. Claim

In all three cohort artifacts, the `pathogenicity` column's contents are
**arithmetically inconsistent** with `ClinVarConnector._map_pathogenicity` as it
has stood since 2026-07-10, and **consistent** with that function's predecessor.
The difference is that the current version returns `uncertain` for any clinical
significance beginning `conflicting`, and the predecessor did not, so those rows
fell through to a substring test and became `pathogenic`.

Approximately 161,000 variants whose submitters explicitly disagree are labelled
`pathogenic` in that column.

## 2. Method

`pathogenicity` is predicted from the measured `clinical_sig` value counts under
each version of the mapping, and the prediction is subtracted from the observed
`pathogenicity` counts. The residual is the contribution of the value-count
tail -- the distinct values below the reported top twenty -- and is therefore
**bounded below by zero**. A negative residual is impossible and falsifies the
version that produced it.

## 3. `clinvar_grch38_clean.parquet` -- 4,399,089 rows, tail 642

| Category | Current code | Observed | Residual |
| --- | ---: | ---: | ---: |
| pathogenic | 221,716 | 383,277 | +161,561 |
| uncertain | 2,712,021 | 2,550,864 | **-161,157 (impossible)** |

| Category | Pre-2026-07-10 code | Observed | Residual |
| --- | ---: | ---: | ---: |
| pathogenic | 383,035 | 383,277 | +242 |
| uncertain | 2,550,702 | 2,550,864 | +162 |
| benign | 272,612 | 272,688 | +76 |
| likely_benign | 1,081,571 | 1,081,595 | +24 |
| likely_pathogenic | 110,527 | 110,665 | +138 |
| | | **sum** | **+642 = the tail exactly** |

## 4. `clinvar_grch38.parquet` -- 4,420,180 rows, tail 753

| Category | Current code | Observed | Residual |
| --- | ---: | ---: | ---: |
| pathogenic | 229,407 | 391,025 | +161,618 |
| uncertain | 2,718,645 | 2,557,540 | **-161,105 (impossible)** |

| Category | Pre-2026-07-10 code | Observed | Residual |
| --- | ---: | ---: | ---: |
| pathogenic | 390,773 | 391,025 | +252 |
| uncertain | 2,557,279 | 2,557,540 | +261 |
| benign | 276,164 | 276,240 | +76 |
| likely_benign | 1,083,552 | 1,083,576 | +24 |
| likely_pathogenic | 111,659 | 111,799 | +140 |
| | | **sum** | **+753 = the tail exactly** |

## 5. Cross-artifact consistency

Residual deltas from raw to clean are +10, +99, 0, 0, +2, summing to **111**,
which is exactly 753 - 642, the tail rows removed with the 21,091 structural and
copy-number-variant rows. The two reconciliations are not independent
restatements of one fit; they agree on the rows that moved between them.

## 6. Independent confirmation from filesystem timestamps

| Modified | Artifact | Relative to 2026-07-10 |
| --- | --- | --- |
| 2026-03-23T01:22:45 | `clinvar_grch38.parquet` | **before** |
| 2026-06-03T19:39:20 | `clinvar_grch38_clean.parquet` | **before** |
| 2026-07-18T09:42:56 | `clinvar_grch38_clean_seq.parquet` | after |

Both artifacts that could have computed the column predate the fix.
`clinvar_grch38_clean_seq.parquet` postdates it but was derived from
`_clean.parquet` by appending sequence columns and copied `pathogenicity`
verbatim; its `clinical_sig` and `pathogenicity` value counts are identical to
`_clean.parquet` in every category. **Staleness propagates through derive-and-
append steps that do not recompute.**

The Parquet `created_by` stamp is `'parquet-cpp-arrow version 23.0.1'` on all
three and therefore orders nothing.

The timestamp evidence and the residual arithmetic share no assumption and agree.

## 7. Exposure

`src/genomic_variant_classifier/data/spark_etl.py:215-216` derives `acmg_label`
from `pathogenicity` with `ACMG_PATHOGENIC = ["pathogenic", "likely_pathogenic"]`
at line 81. Run against these artifacts it would assign `acmg_label = 1` to every
conflicting variant.

Nothing currently imports `spark_etl` for execution: `tests/unit/test_core.py`
imports only `CHROM_MAP` and `VARIANT_SCHEMA`, and `etl_polars.py` mentions it in
a comment. **The defect is latent, not active** -- one invocation away.

`real_data_prep._load_and_label` labels from `clinical_sig` and is unaffected.

## 8. What this does NOT establish

Which process wrote the column, and when. The arithmetic establishes which
mapping the contents are consistent with; the timestamps establish that the files
predate the fix. Neither identifies the writing run. That requires a provenance
manifest, which these artifacts do not carry.

## 9. Consequence

`docs/PHASE1_SPEC_2026-07-24_deletion-repair.md` section 6, measurement 5 records
that **every figure in its section 2 was measured on `pathogenicity`**. Those
figures therefore rest on a superseded mapping and on a column production does
not label from. Section 2's label-dependent rows -- binary trainable rows and
positive rate -- must be recomputed on `clinical_sig` before they justify
anything. Approved 2026-07-24.
