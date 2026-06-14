# Session 2026-06-14 -- harness warning fix landed + drift CI repair

Continues the same-day drift work (after 19fb2a0 trio, 993524b docs, fe2289d harness fix).

## Harness fix landed (fe2289d) -- verified
The proposed n_jobs=1 harness-smoke fix shipped and is CONFIRMED: with
`EnsembleConfig(skip_svm=..., n_jobs=1)`, sklearn uses the SequentialBackend (no loky), so the flaky
`parallel.delayed` UserWarning is gone. `test_correctness_harness.py` alone: 6 passed / 8 warnings (lbfgs
ConvergenceWarning only). Two back-to-back full `pytest -q` runs: BOTH 1083 passed / 6 skipped / 41 warnings.
The 41-warning gate baseline is now DETERMINISTIC (was 41-vs-141 flaky).

## drift_monitor.yml repair
The monthly drift-monitor workflow was inert: the GDrive "download" step fabricated a "credentials loaded"
message but wired no real fetch, and pointed at the pre-Run-15 path `outputs/phase2_with_gnomad/splits`, so
the feature-drift job always logged "No reference splits available -- skipping" (drift_level=none -> the
notify job never fired).

- Repointed the stale path -> `outputs/run15_rerun_report/full/splits` (6 occurrences: the mkdir, BOTH
  halves of the `-d ... || -z "$(ls -A ...)"` guard line, the drift command, the Job-2 comment, the Job-3
  issue body).
- Made the GDrive step honest (no fabricated success message; the skip is logged). A real rclone/gdown
  fetch remains a placeholder pending CI data access.
- Added a GUARDED schema-drift gate step: `run_schema_drift_check.py --matrix .../X_train.parquet`
  (exit 0 green / 2 drift / 3 usage). `run_drift_monitor.py` covers PSI/KS/MMD distributional drift but NOT
  schema/column/dtype drift -- this gate is additive. It skips honestly when baseline/matrix are absent
  (GitHub-hosted CI has no data), so it is inert-safe until a self-hosted runner has data.

## Validation boundary (important)
YAML syntax + logic validated in-sandbox (yaml.safe_load: steps parse, schema gate precedes upload, stale
path gone, message honest). GitHub Actions EXECUTION cannot be run in-sandbox -- verify live via the Actions
tab "Run workflow" (workflow_dispatch). The verified split filename (`X_train.parquet`) comes from
run_drift_monitor.py (--reference-splits help + the X_train.parquet load), not a guess.

## Remaining (ROADMAP)
- Real GDrive/rclone fetch for the monthly job (replace the placeholder).
- Tighten the schema gate to gate-the-job on exit-2 (currently continue-on-error, matching the job's
  notify-not-fail design) or feed the notify job.
- Reconcile the two parallel drift systems (agent-layer drift agents vs src/monitoring/ +
  run_drift_monitor.py) -- still OPEN.
- Run-17-gated: build the 4 model-dependent baselines (LabelShift/Concept/Calibration/Fairness) from
  trained-model predictions; schema_baseline regen 81->82.

## Next-agent backlog (user-ranked, fully sandbox-validatable like FeatureCoverageSentinel)
LeakageSentinel, BudgetSentinel, ReclassificationSentinel (wraps clinvar_tracker.py), EvidenceLinkAgent.
