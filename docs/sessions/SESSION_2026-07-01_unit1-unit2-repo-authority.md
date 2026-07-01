# SESSION 2026-07-01 -- Unit 1, Unit 2, and repo-authority record

## Scope
Continuation of the orchestrator-redesign arc. Two units landed; one process gap
(missing repo-move documentation) was surfaced and is being closed.

## Unit 1 -- phantom-drift removal (COMMITTED, PUSHED, CI GREEN)
- Commit `ebbcbbc`. Removed the vestigial `_check_drift` path from
  TrainingLifecycleAgent (imported a phantom `detect_drift` from `ewc_utils` that
  never existed; returned False on every run; no consumer read the result).
- New regression test `tests/unit/test_training_lifecycle_drift_removal.py` (3 tests),
  each confirmed to FAIL against pre-fix code (anti-vacuity).
- Full suite after: 1510 passed, 7 skipped. 22/22 agents active. CI GREEN on origin.

## Unit 2 -- reactivate message-bus suite as collected pytest (COMMITTED, PUSHED)
- Commit `ab56cde`. Moved
  `src/genomic_variant_classifier/agent_layer/test_message_bus.py` ->
  `tests/unit/test_message_bus.py` (rename 80%); quarantine dissolved WITHOUT
  touching pytest `testpaths`.
- Root causes fixed (not papered over):
  - D12 isolation: module-level `sys.modules` mutation (config stub + MagicMock
    torch/shap/ewc_utils/feedparser/requests) that leaked a MagicMock torch into the
    whole pytest collection and broke scipy's array-api import in 12 downstream files
    (INCIDENT_2026-05-26) -> replaced by two autouse, teardown-safe fixtures
    (`_isolated_optional_deps` via monkeypatch.setitem; `_no_interactive_input`
    hard-fails any input() so no test can block, CI has no TTY).
  - Approval control: Group-4 emits test patches DataFreshnessAgent._require_approval
    explicitly (asserts the approved branch deterministically).
  - Removed 5 orphaned `patch.object(..., "_check_drift")` clauses (orphaned by Unit 1).
  - Repointed 4 bare `from orchestrator import Orchestrator` to the full package path.
  - Renamed 35 `_test_*` -> `test_*`; removed the custom `_run`/`TESTS`/`main` harness.
- Verified non-interactively (stdin closed every run): 35/35 message-bus tests collected
  and pass (exit 0); tests/unit collect-only clean (1547; no scipy pollution regression);
  full suite 1545 passed, 7 skipped.
- Incident docs: closed `INCIDENT_2026-05-30_test_message_bus_history_ordering`
  (production already stamps monotonic `seq`; both ordering tests pass); appended a
  collected-pytest verification note to the already-resolved
  `INCIDENT_2026-05-30_test_message_bus_stale_patch_path`. Left
  `INCIDENT_2026-05-26_scipy-torch-array-api-compat` OPEN (different, phylop-triggered
  root cause -- Unit 2 does NOT resolve it).

## Process gap surfaced: repository authority undocumented (ACTION TAKEN)
- Owner stated the authoritative repo was reclaimed to
  `github.com/monzia-moodie/genomic-variant-classifier` days ago; `-repo-projects` to
  become stale. NO written record of that move existed anywhere in the repo.
- Local `origin` still targets `-repo-projects`, and all recent work (through
  `ab56cde`) was pushed there. So the current, most-updated state lives in
  `-repo-projects` regardless of prior intent.
- Created `docs/incidents/INCIDENT_2026-07-01_repo-authority.md` to record the known
  facts, the evidence-based resolution gate (compare `git ls-remote` on both repos),
  and the corrective rule: any remote/repo-identity change MUST be documented in
  incidents + session log AT THE TIME it happens.

## Ensemble roster (VERIFIED from variant_ensemble.py, 2026-07-01)
Confirmed by reading the code (base_models dict): 8 base models --
random_forest, xgboost, lightgbm, svm (RBF, calibrated), logistic_regression,
gradient_boosting, tabular_nn, cnn_1d -- with a Logistic Regression stacking
meta-learner trained on OOF predictions, plus a separate GNN branch (STRING DB) in
gnn.py. Two neural networks are present in the base ensemble: tabular_nn and cnn_1d
(1D-CNN). All implemented models continue to be used; the roster is additive (more
models may be added in future phases). Correction: an earlier working assumption of a
larger roster (CatBoost/KAN/MC-Dropout/Deep Ensemble/svm_bagged_rbf) did not match
this codebase; the code is authoritative and lists the 8 base models above.

## Open / next
- Settle repo authority via the resolution gate in the incident doc; record outcome.
- Confirm `ab56cde` (Unit 2) CI goes green on origin.
- Hygiene: delete stray root `test_training_lifecycle_drift_removal.py` (real copy is
  tracked at tests/unit/); add .gitignore rules for `*.factory_state_*` and
  `data/_pandas3/`; investigate the 0-byte `docs/roadmap` file.
- Note: `Data Freshness Monitor` scheduled workflow shows red on GitHub (a cron, not CI);
  unrelated to Unit 1/2 -- worth a separate look.
