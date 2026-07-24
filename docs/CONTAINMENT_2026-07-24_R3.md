# Containment 2026-07-24, revision 3 -- the label-column gate is CLEARED

**This document appends to `CONTAINMENT_2026-07-24_R2.md`. It does not replace or
rewrite it.** R2 remains the record of what was believed on 2026-07-24 at the time
it was written. This revision records what was subsequently measured.

## 1. What R2 section 8 said, and why it was stale the moment it was committed

R2 section 8 stated that `probe_label_column_terms_2026-07-24.py` **had not been
run**, that the question it answers **outranked every other open item**, and that
**no repair should begin until it had run**.

By the time R2 was committed at `e3a4795` (2026-07-24T05:22:26-04:00) the probe
had already been run twice. Section 8 was true when authored and false when
committed. It is left in place, uncorrected, because the record of what was
believed is itself evidence; this revision supersedes it.

## 2. The question R2 section 8 asked

Whether `clinical_sig` -- the column `real_data_prep._load_and_label` matches
against with an exact, case-sensitive `.isin()` after `.fillna("").str.strip()`,
at lines 512 and 513 -- carries values in the underscored form. If it did, every
likely-pathogenic and likely-benign variant would fail both term-set tests,
receive no label, and be dropped by the `notna()` filter at line 516, silently.

## 3. Why the original answer was not conclusive

The superseded probe read ONE hardcoded artifact,
`data/processed/clinvar_grch38_clean.parquet`. That is not the artifact
`real_data_prep.py:29` directs a reader to, nor the one
`scripts/preflight_run16_inputs.py:176` defaults to. It measured one of three
candidates. It also exited 1 on any divergence, including one confined to a
column production never labels from, so its exit code could not distinguish the
blocking case from the non-blocking one -- the same defect shape as the Run-16
preflight gate repaired on 2026-07-20.

## 4. The answer, measured across all three artifacts on 2026-07-24

`scripts/probe_label_column_terms.py` reads the term sets and the labelling
column from `real_data_prep.py` by abstract syntax tree (AST) parse at run time,
so it cannot go stale, and it discriminates four exit codes.

| Artifact | Rows | Exact match | Normalised match | Difference |
| --- | ---: | ---: | ---: | ---: |
| `clinvar_grch38.parquet` | 4,420,180 | 1,700,687 (38.476%) | 1,700,687 | **0** |
| `clinvar_grch38_clean.parquet` | 4,399,089 | 1,686,333 (38.334%) | 1,686,333 | **0** |
| `clinvar_grch38_clean_seq.parquet` | 4,399,089 | 1,686,333 (38.334%) | 1,686,333 | **0** |

**`clinical_sig` diverges in no artifact. The feared defect does not exist.**
Section 8's conditional is not met. Aggregate exit 3 -- a finding, not a blocker.

Evidence: `docs/measurements/LABEL_COLUMN_TERMS_2026-07-24.txt`.

## 5. What DOES diverge, and why it is not this

`pathogenicity` diverges in all three artifacts: 0 rows match exactly, 1,848,225
match after normalisation in the clean artifacts and 1,862,640 in the raw one.
Production does not label from that column, so no training row is lost by it.

That column carries a separate and more serious defect, recorded in
`docs/measurements/MEASUREMENT_2026-07-24_pathogenicity-column-staleness.md`: its
contents are arithmetically inconsistent with the mapping code in force since
2026-07-10, and consistent with that code's predecessor.

## 6. Consequence for ordering

R2 section 8's instruction that no repair should begin until the probe had run is
**satisfied**. The Phase 1 repair is unblocked on this ground.

It remains blocked on others, which are not this document's subject: the Step 1a
single-definition guard fails on any complete tree, and specification section 3's
row counts do not survive contact with the cohort. Both are recorded in
`docs/DEFECTS_2026-07-24.md`.
