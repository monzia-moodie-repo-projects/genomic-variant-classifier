# INCIDENT 2026-06-03 — `--min-review-tier` silently no-op across all runs

**Severity:** High (silent failure affecting result interpretation, multi-run reach)
**Status:** Mitigation in progress (Path A). Part 1 (data) DONE; Parts 2-4 (guard/test/preflight) pending.
**Discovered:** 2026-06-03, during Run 15 pre-flight audit (after G1 PASS / preflight GO).

## Summary
The `--min-review-tier` filter (default 3) was a **silent no-op** on every run that used
the processed ClinVar parquets. `DataPrepPipeline._load_and_label` applies the tier filter
only inside `if "ReviewStatus" in df.columns:` (real_data_prep.py:357). Neither
`clinvar_grch38_clean.parquet` (de-leaked) nor `clinvar_grch38.parquet` (dirty) ever
contained a `ReviewStatus` column — both carry the same 16 columns, none being review status.
The VCF->parquet build never propagated `CLNREVSTAT`. With the guard false, the entire filter
block was skipped and no log line was emitted, so the only signal was an *absent* log line.

## Impact
- Runs that declared tier>=3 (incl. Run 14, AUROC 0.9975) trained on **all review-level**
  labeled ClinVar variants (Pathogenic/Likely-pathogenic/Benign/Likely-benign at any star
  level), not the tier>=3 high-confidence subset their command implied.
- Run 14's headline number is therefore both leakage-inflated AND never tier-filtered.
- Run 15, as configured, would have silently been an all-labeled baseline — not the honest
  tier>=3 baseline the plan describes — while still reporting success.
- No unit test asserted the filter reduces rows; `preflight_run15_baseline.py` checked cohort
  cleanliness (null/dup) but not tier-filterability, so the GO verdict was a false-green on
  this axis.

## Root cause
The processed-cohort build never extracted `CLNREVSTAT` from the ClinVar VCF into a
`ReviewStatus` column. The filter code was written for a column the data pipeline never
produced. De-leaking did NOT drop it (the dirty cohort lacked it too) — it was never present.

## Evidence
- Live guard: `real_data_prep.py:357  if "ReviewStatus" in df.columns:`.
- Clean cohort: 16 cols, no review/status/tier/gold column; 4,399,089 rows.
- Dirty cohort: identical 16 cols, `ReviewStatus present in DIRTY: False`.
- Consumer: `run_phase2_eval.py:214  min_review_tier=args.min_review_tier` ->
  `DataPrepConfig` -> `DataPrepPipeline.run` -> `_load_and_label`.
- Tests: only LOVD test references `min_review_tier`; no tier-reduction assertion.
- Probe (`scripts/probe_review_status.py`, read-only): of 1,686,333 labeled variants,
  1,546,702 (91.7%) join CLNREVSTAT by chrom:pos:ref:alt; tier dist
  {1:13374, 2:334529, 3:1142111, 4:56688}; tier<=3 = 1,490,014.

## Resolution — Path A (make tier>=3 real)
1. **DONE** `scripts/augment_reviewstatus.py`: attach decoded (space-form) `ReviewStatus`
   from the VCF's CLNREVSTAT, unmatched -> "" -> tier 5 (excluded). Backup-first, idempotent,
   atomic. On-disk verification matched the probe exactly: tier<=3 = 1,490,014; rows/null/dup
   unchanged; backup at `clinvar_grch38_clean.parquet.pre_reviewstatus.bak`.
2. **PENDING** Fail-loud guard in `_load_and_label`: when `ReviewStatus` is absent but
   `min_review_tier < 5`, raise instead of silently skipping; drop `review_tier` after
   filtering so it cannot leak into the feature matrix.
3. **PENDING** Unit test: filter reduces rows when `ReviewStatus` present; raises when absent
   with a non-default tier.
4. **PENDING** `preflight_run15_baseline.py` gate: NO-GO if `ReviewStatus` missing from the
   cohort, so a tier-unfilterable cohort can never reach a paid run again.

## Prevention / lessons
- A guarded filter that silently skips on a missing column is a silent-failure pattern; such
  guards must fail loud when the filter was explicitly requested.
- Pre-flight must assert the *effect* (filter reduces rows), not just input presence.
- The honest baseline must differ from Run 14 only by the intended changes (de-leak + tier
  filter), not by an additional silent change.
