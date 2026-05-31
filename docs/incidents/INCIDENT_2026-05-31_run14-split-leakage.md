# INCIDENT 2026-05-31 -- Run-14 split leakage (quantified) + provenance/ensemble findings

**Status:** ROOT CAUSE KNOWN (null-key cohort, see INCIDENT_2026-05-31_null-key-leak.md);
remediation = regenerate splits from the clean cohort.
**Severity:** HIGH -- the headline ~0.9974 test AUROC is inflated.
**HEAD:** 18bbba1

## Quantified leakage (outputs/run14/full/splits, measured 2026-05-31)
- within-split duplicate variant_id: train 2,125 / val 129 / test 409
- cross-split variant_id overlap: train&test 247, train&val 115, val&test 46
- structural (null-key) variant_ids in splits: 11,320 of 21,091 quarantined
- gene_symbol overlap train&test: 0  (main split IS gene-disjoint)

## Interpretation
GroupShuffleSplit by gene (real_data_prep.py:1154, random_state 42/43) makes the split
gene-disjoint. The ~0.9974 inflation is from variant duplication + structural garbage, NOT gene
leakage. Cross-split overlaps despite gene-disjointness arise from duplicate variant_ids with
inconsistent/null gene labels landing in different gene groups -- removed by cohort dedup.

## Remediation
Regenerate splits from data/processed/clinvar_grch38_clean.parquet (0 null, 0 dup). Removes all
three contamination classes; gene-disjointness preserved. Cohort guard (_assert_clean_cohort)
now aborts on any future null/dup cohort. Post-regen require: within-split dup 0, cross-split 0,
structural-in-splits 0, gene overlap 0.

## Additional findings (run14_master.log)
1. Provenance mismatch: log reports output=/workspace/outputs/run11/full though stored under
   outputs/run14/. Confirm which run produced outputs/run14/ before trusting run labels.
2. Reduced ensemble: skip_cnn=True (cnn_1d closure bug, B.D6) and string_db=None (GNN off).
   The headline came from ~9 of 11 models with two modalities dead.
