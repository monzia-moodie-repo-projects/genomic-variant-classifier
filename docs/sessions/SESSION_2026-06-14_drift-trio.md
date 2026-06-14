# Session 2026-06-14 -- Drift trio activation (Concept + Calibration + FairnessSubgroup)

HEAD at start: 00bbc24 (docs). HEAD at end: 19fb2a0 (trio) + this docs commit.

## What shipped (19fb2a0, 12 files, +693, 20 tests)
Completed the drift-baseline campaign's three RUN-17-DEPENDENT detectors. Each gained the standard
activation pair -- detector `from_baseline` + monitor `from_default_baseline` -- so the orchestrator
`from_default_baseline` hook (6a05481) constructs them from their canonical references automatically; until
those references exist (Run-17) they report `awaiting_baseline` gracefully.

- **Concept** -- baseline = `cbpe_baseline_auroc` + `cbpe_baseline_sigma` (NannyML CBPE on the reference
  window). Monitor resolves the production scalars `cbpe_estimated_auroc` / `bbse_pvalue` / `n_samples` from
  args or `GVC_CONCEPT_CBPE_AUROC` / `GVC_CONCEPT_BBSE_PVALUE` / `GVC_CONCEPT_N_SAMPLES`.
- **Calibration** -- baseline = `classes` + `baseline_ece`. `build_calibration_baseline.py` computes
  `baseline_ece` by calling the detector's OWN `detect()` on the reference predictions (`baseline_ece=0`), so
  the reference ECE and the monitored ECE share ONE code path and cannot silently diverge. Monitor resolves
  `labeled_predictions` from arg or `GVC_CALIBRATION_LABELED_PREDICTIONS` (parquet).
- **FairnessSubgroup** -- baseline = `classes` + `p_train_per_stratum` (the reference predicted-class count
  vector per `(axis, stratum)`, matching `_stratum_metric`'s `observed`; tuple keys serialized as records;
  `high_priority_strata` list -> frozenset). Monitor resolves `predictions` (`GVC_FAIRNESS_PREDICTIONS`
  parquet) + `axes` (`GVC_FAIRNESS_AXES` JSON object).

## Stubs flagged (PHASE_2_FEATURES, pre-existing, NOT changed)
`FairnessSubgroupAgent`: per-stratum AUROC is a confidence proxy (`# placeholder; wire to nannyml.CBPE`) and
`max_dpd_change` is hardcoded `0.0` (`# wire from training-time DPD baseline`). `test_dpd_stub_is_zero`
asserts the `0.0` so a future DPD-baseline wiring trips a failing assertion (the stub becomes self-announcing).

## Bugs caught in build (sandbox, before delivery)
- Calibration test fixture v1 routed all errors to one class -> that class's posterior was miscalibrated
  (p~0.03 but true ~24% when wrong) -> per_class ECE >= 0.10 -> the detector CORRECTLY returned red on
  "green" data. Fixed the generator to distribute errors uniformly across non-predicted classes (every class
  calibrated). The detector was right; the synthetic data was wrong.
- Calibration amber fixture v1 (miscal=0.035) fell just under the 0.02 delta threshold; relocked to
  miscal=0.04 (amber stable across seeds 7/11/23/42, ece 0.034-0.045, margin from both 0.02 and 0.05).
- Trio tests initially imported builders via `from scripts.build_X` (no `scripts/__init__.py`; conftest shim
  only fires for clean_cohort) -> rewrote to the proven `importlib.util.spec_from_file_location(... ../../scripts/...)`
  pattern used by the prior baseline tests.

## Gate
Targeted: 20 passed. Full suite: 1083 passed / 6 skipped. Drift dry-run pipeline: 9 agents run.

## Audit finding -- flaky pre-existing warning (does NOT block; documented)
Two back-to-back `pytest -q` runs reported 41 then 141 warnings. The +100 is a benign FLAKY sklearn
UserWarning (`sklearn.utils.parallel.delayed` ..., parallel.py:144) from `test_correctness_harness.py`:
`run_correctness_harness` builds `EnsembleConfig(skip_svm=...)` with the default `n_jobs=-1`, so the Stage-1
smoke fits the tiny slice through loky `Parallel`; loky emits the warning per worker dispatch (0..~100,
depending on worker spawn/reuse). NOT a trio regression (the trio tests use no sklearn parallelism); no
pass/fail impact. DETERMINISTIC baseline stays 41. Root-cause fix proposed (separate follow-up): force
`n_jobs=1` in the harness smoke.

## Drift set status
8 of 8 WIRED + FeatureCoverageSentinel (9th). ACTIVE: Schema, Infrastructure, AnnotationPolicy,
AdversarialSubmission, FeatureCoverageSentinel. MACHINERY-COMPLETE (awaiting Run-17 model artifacts):
LabelShift, Concept, Calibration, FairnessSubgroup.

## Next
1. (recommended) harness `n_jobs=1` stabilization (apply_harness_njobs1.py) -> deterministic 41-warning gate.
2. Run-17 artifacts: build the 4 model-dependent baselines (LabelShift/Concept/Calibration/Fairness) from
   trained-model predictions; schema_baseline regen 81->82.
3. Remaining ROADMAP drift items: fix drift_monitor.yml (stale splits path + GDrive stub), add the schema
   gate as a yml step, reconcile agent-layer drift vs src/monitoring/ + run_drift_monitor.py.
