# Report test fixtures

Each JSON here was produced by the real script it stands in for, run on a small fixture, so its
STRUCTURE is genuine. The runs were independent, so three VALUES were aligned by hand to make the
artifacts describe the same rows, as the generator's cross-output consistency checks require:

| File | Field | Was | Now | Why |
|---|---|---:|---:|---|
| res.json | clinical_sig_disagreement_kinds | {} | {"other": 1} | auth.json has one disagreement, 'Likely benign' vs 'Benign', which is not delimiter-only |
| gi.json | provenance/relation_rows | 3 | 10 | resolution accounts for 10 relation rows |
| components/components_summary.json | genes_only/leakage_restricted/validation_rows | 3 | 4 | the stratified runs evaluate 4 validation rows |

These are NOT measurements. Nothing here should be read as data about the cohort.
