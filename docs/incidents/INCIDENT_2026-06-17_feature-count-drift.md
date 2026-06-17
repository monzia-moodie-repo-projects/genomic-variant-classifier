# INCIDENT 2026-06-17 -- CI red: feature-count guardrails not bumped with rnaseq_* family

**Status:** RESOLVED -- verified green at CI run #442 (commit `11e14a3`); both pytest 3.11 and 3.12 legs passing.

## Summary
The RNA-seq ingestion commit `1f3c2e0` (Fork C) widened `TABULAR_FEATURES` from 82 to 87 by adding the
five `rnaseq_*` gene-level columns, but did NOT update the two guardrails that pin that count:
`EXPECTED_TABULAR_FEATURE_COUNT` (variant_ensemble.py) and the `KNOWN_ZERO_DEFAULT` dead-connector
allowlist (correctness_harness.py). The local dev venv was validated with a SUBSET pytest run
("32 passed"), which never exercised those guardrails, so the drift was invisible locally while the
full CI suite failed 5 tests. CI was red from `1f3c2e0` through both data-standard commits.

## Timeline
- `9d941c0` (gtex Fork A): green -- gtex_* features pre-existed in TABULAR_FEATURES, count unchanged.
- `1f3c2e0` (rnaseq Fork C): first RED -- 5 failures (count 82 vs 87).
- `47bc887`, `40f16f0` (data-layout standard): RED, inherited; unrelated to the data work.
- 2026-06-17: detected from the Actions tab (three consecutive red runs); local subset run had masked it.
- `11e14a3`: green -- fix verified at run #442.

## Root cause
Feature-matrix width changed without the SAME-commit guardrail bump. The guardrails are deliberate
tripwires (a count constant + an explicit dead-connector allowlist); they fire only in a full-suite
run. `engineer_features` materializes `rnaseq_*` as 0.0 until an `--rnaseq-path` parquet is supplied
(populated via `annotate_rnaseq_from_parquet`, not via engineer_features inputs) -- so on the synthetic
`build_reference_slice` they are correctly all-zero, exactly like the `gtex_*` entries already
allowlisted. The harness therefore flagged them as new silent-zeros outside the allowlist (working as
designed).

The 5 failures and the edit that cleared each:
- `test_feature_count_contract::test_tabular_features_length_matches_constant`     -> count bump
- `test_feature_count_contract::test_inference_feature_columns_length_matches_constant` -> count bump
  (INFERENCE_FEATURE_COLUMNS derivation in api/pipeline.py was correct -- it derived 87; only the constant was stale)
- `test_esm2_llr_feature_wiring::test_count_constant_tracks_list`                  -> count bump
- `test_api::TestInfoEndpoint::test_info_returns_metadata`                         -> count bump
  (the mock pipeline sets metadata.n_features = EXPECTED_TABULAR_FEATURE_COUNT; the test asserts len(TABULAR_FEATURES) == that)
- `test_correctness_harness::test_complete_slice_only_flags_known_zero_defaults`   -> allowlist add

## Fix (commit 11e14a3)
- `variant_ensemble.py`: `EXPECTED_TABULAR_FEATURE_COUNT` 82 -> 87 (single source of truth).
- `correctness_harness.py`: add `rnaseq_mean_log_tpm`, `rnaseq_detection_rate`, `rnaseq_log2_cv`,
  `rnaseq_log2fc`, `rnaseq_de_neglog10p` to `KNOWN_ZERO_DEFAULT`; doc note 22 -> 27 columns.
- No connector or feature logic changed -- guardrails only. Applied via an EOL-preserving, count-guarded,
  idempotent patcher (`patch_feature_count_87.py`); validated against a pristine pull byte-for-byte.

## Prevention
- Treat ANY change to `TABULAR_FEATURES` width as requiring a same-commit bump of
  `EXPECTED_TABULAR_FEATURE_COUNT` AND a `KNOWN_ZERO_DEFAULT` review (new stub-zero columns must be added).
- NEVER validate a feature-matrix change with a pytest subset. Run the full `pytest tests/unit` (or at
  minimum the `test_feature_count_contract` + `test_correctness_harness` + api contract tests) before push.
- The guardrails worked exactly as intended; the gap was process (subset validation), not coverage.
