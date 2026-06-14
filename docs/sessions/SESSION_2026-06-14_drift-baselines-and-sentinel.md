# SESSION 2026-06-14 -- drift-baseline campaign + FeatureCoverageSentinel

Continues SESSION_2026-06-14_hetero-eval-drift-baseline.md. Goal: finish the drift-baseline set, then
build FeatureCoverageSentinel to full standard (tested, wired, activated).

## Commits (on top of 6a05481 / 1b69529)

| Commit  | What |
|---------|------|
| c0dec47 | LabelShift activation machinery (from_baseline + builder + from_default_baseline); RUN-17-dependent baseline |
| a7abca0 | Infrastructure activation -- first FULLY model-free agent; ACTIVE NOW |
| 469a7b6 | AnnotationPolicy + AdversarialSubmission activation (model-free, config-threshold); BOTH ACTIVE NOW |
| e4f96df | feature_health shared module + audit refactor (byte-identical; single source of truth) |
| e61a2dc | FeatureCoverageSentinel detector (scores a matrix vs the audit reference) |
| 8206673 | FeatureCoverageSentinel builder + monitor adapter |
| 376aa2e | FeatureCoverageSentinel orchestrator wiring + the `drift` pipeline (ACTIVATED) |

## Drift-baseline set: 6 of 8 active/wired, + a 9th agent

- ACTIVE NOW: Schema (6a05481), Infrastructure (a7abca0), AnnotationPolicy (469a7b6),
  AdversarialSubmission (469a7b6), FeatureCoverageSentinel (376aa2e).
- MACHINERY-READY (real baseline at Run-17): LabelShift (c0dec47).
- REMAINING (RUN-17-DEPENDENT, need model predictions): Concept (NannyML CBPE + BBSE),
  Calibration (per-class posteriors + ECE), FairnessSubgroup (per-subgroup predictions) -- machinery next.

## FeatureCoverageSentinel (the silent-failure auditor)

Catches a column healthy at reference time that has gone degenerate now (the 34/78 and 38/78 dead-feature
regressions of Run 14 / Run 10b) before it reaches training. Built in four steps:

1. **feature_health.py** -- the audit's `_col_health` + `_unique_and_top` extracted verbatim into
   `src/genomic_variant_classifier/data/feature_health.py`, a single source of truth shared by the audit and
   the sentinel (so the sentinel's current-matrix health can never silently diverge from the audit that
   produced its reference). Behavior-preserving: byte-identical health CSV; the refactored audit still gives
   54/42/96 on the real Run-15 splits.
2. **detector** -- `FeatureCoverageSentinelAgent.detect` classifies each column vs the reference into
   regressed (healthy -> degenerate; RED), dropped (RED), recovered, still_degenerate, new (AMBER), scoring
   with the SAME `near_constant_frac` the reference carries.
3. **builder + monitor** -- `build_feature_coverage_baseline.py` turns the audit CSV into the canonical
   reference (cross-file aggregation, guarding the empty->NaN re-read pitfall);
   `FeatureCoverageSentinelMonitorAgent.from_default_baseline` (current_matrix arg -> GVC_FEATURE_MATRIX env).
4. **wiring** -- registered + `PIPELINE_DEFINITIONS["drift"]` = all 9 drift agents, reachable via
   `run_agents.py --pipeline drift`. Verified live: the dry-run drift pipeline runs all 9 agents.

Reference = the existing Run-15 audit (54/42/96), so the sentinel ACTIVATES NOW; point GVC_FEATURE_MATRIX at
a post-regen matrix to score it.

## Bugs caught (nothing shipped wrong)

- **Empty-degenerate -> NaN** in the reference builder: `pd.read_csv` turns the audit's `degenerate=''` into
  NaN, so a naive `degenerate != ''` would flag EVERY healthy column degenerate. Guarded with `fillna('')`;
  proven (naive flags all 7 synthetic columns, guarded gives the correct 4).
- **near_constant_frac honoring**: the reference carries the frac it was audited with; the detector scores
  with that frac (the same near-constant column reads red at 0.95 / green at 0.999).
- Two errors in the validation harness itself (registry name appears 4x not 3x; PIPELINE_DEFINITIONS is an
  AnnAssign not Assign) -- caught + fixed before relying on them; the deliverables were correct.

## Suite

1009 -> 1014 -> 1022 -> 1032 -> 1040 -> 1050 -> 1060 -> 1063 passed / 6 skipped / 41 warnings
(all pre-existing: LGBM feature-names, n_components>n_samples, lbfgs ConvergenceWarning; zero new).
HEAD 376aa2e on origin/main.

## ROADMAP

- `[x]` Pipeline-wire the drift agents -- DONE (376aa2e): the `drift` pipeline now exists with all 9 agents.
- Still open: fix `drift_monitor.yml` (stale path + GDrive stub); add the schema gate as a yml step;
  reconcile the agent-layer drift vs `src/monitoring/` + `run_drift_monitor.py` systems.

## Next

Run-17-dependent trio machinery (Concept / Calibration / FairnessSubgroup) -- read each detector first,
then from_default_baseline + builder + from_baseline like LabelShift, validated synthetically.
