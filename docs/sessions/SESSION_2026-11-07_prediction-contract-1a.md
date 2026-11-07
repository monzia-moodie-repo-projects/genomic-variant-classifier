# Session 2026-11-07 -- PREDICTION-CONTRACT-1A: package 1 (the prediction contract), set 1a of three

Author: Monzia Moodie

## Why

Package 1 makes one predictor authoritative and makes every score traceable to it. The owner's rulings of 2026-11-07 fixed the
science and the delivery:

- The stored convex blend of the fitted members is the single authoritative score. The logistic stacker is a research candidate
  only: never served, never substituted when the blend cannot be computed. Unsupported projections are refused: no
  normalisation, fallback, absolute value or pruning.
- Decision ccffb8cf (revision 5): the kernel is `exact_binary64_products_sum_rne_v1`, the exact real sum of the stored binary64
  products rounded once (nearest, ties to even; canonical +0; a result above one refused). The historical `P @ w` keeps its own
  identity for replay only. Dot2 is a candidate, never selectable. A kernel change is a predictor-identity change. Every score
  candidate in the census gets one disposition.
- Decision 38f15c7a: three sequential installers (1a, 1b, 1c) under one invariant -- each leaves every affected, reachable
  operation either correct or explicitly refused. A declared projection (13 members served as 12) is readable as a diagnostic and
  never ready. The registry record binds the fitted predictor, and inference refuses before any member call unless that binding is
  ready. Gate C10 stays closed.

The frozen acceptance matrix and its revision 2, which records both rulings as amendments, are in docs/package1/.

## What set 1a changes

- `prediction_contract.py` (new): observations and their two identities (`source_observation_id`, `prediction_unit_id`), the
  probability block bound from member to result, replay verification, the one tie rule (`positive_if_p_ge_0.5`), quantities with
  units (nats, probability squared, probability), `ContractError`.
- `models/combiner.py` (new): the exact kernel as an integer accumulator in units of 2^-2148 (65,536-row chunks), rounded
  half-even directly to binary64 bits; the historical kernel for replay; Dot2 as a non-selectable candidate; `ConvexBlendSpec`
  (source, scope, ordered member slots with fitted identities and prediction stages, exact weights each in [0, 1], fit-input
  stage, kernel) and its digest; one member capture per scoring; the migration report (bits, ULP histogram, call changes at every
  declared edge under >= and >, order changes).
- `models/disagreement.py` (new) and `models/mc_dropout.py`: schema-2 uncertainty records. The two decompositions close exactly,
  each in its own unit: the entropy of the mean equals the mean draw entropy plus the mean Bernoulli divergence of the draws from
  their mean (nats); the Bernoulli variance of the mean equals the mean draw variance plus the spread between draws (probability
  squared). In agreeing rows the spread is exactly zero -- `np.var` reported a nonzero spread for ten identical draws of 0.3,
  measured -- and the divergence kernel carries a row status (exact agreement, resolved, unresolved with a reason). A model with
  no stochastic pass reports the uncertainty unavailable, never zero. The historical seed schedule and float32 mean are kept
  bitwise. Draw counts must be integers of at least two; labels must be the integers 0 and 1.
- `models/variant_ensemble.py`: the fitted ensemble declares its combiner; `predict_proba` and `evaluate` score the declared
  blend from one member bundle, with no meta-learner fallback; the evaluation row is `SELECTED_CONVEX_BLEND`
  (`SELECTED_ENSEMBLE_ROW`); the blend optimiser refuses a non-finite or all-zero result; `save` and `load` carry the declaration,
  and a legacy orchestrator loads only with an explicit `legacy_combiner_declaration`. TabularNN derives `classes_` at fit; the
  isotonic calibrator validates its base model's class order.
- Serving (`api/pipeline.py`, `api/main.py`, `api/schemas.py`, `api/attribution.py`, `monitoring/model_registry.py`): one scoring
  path and one serializer for single, batch and offline use; prediction schema 3.0.0 (full-precision score, the descriptive call
  under the tie rule, the predictive entropy in nats, a labelled uniform research view of member disagreement, and explicit
  unavailable statuses for calibration, decision policy, attribution and coverage); a 3.0.0 reader refuses any other schema
  version. `predictor_contract_sha256` binds the combiner digest, the declared features, the scaler and missing-value policy as
  values, and the class order; a new attribution axis, `PredictorContractAlignment` (unknown, not declared by the record,
  matches, differs); ready requires an EXACT roster and MATCHES. `/predict` and `/batch` answer 503 before any member call unless
  ready; `/rsid` returns the locus with `prediction_unavailable_reason`. The rate-limit key is the SHA-256 of the API key. The
  API image now copies `prediction_contract.py` (DOCKERCOPY-1's gate found it missing).
- Transition refusals (`transition_policy.py`, new, separate from the quarantine policy so the admission digest is unchanged):
  `calibrate_thresholds.py` (THRESHOLD_CONSUMER_MIGRATION_REQUIRED; enabled by 1c), `run9_ablations.py`
  (ABLATION_WRITER_MIGRATION_REQUIRED; enabled by 1b; it re-raises `ContainmentError` ahead of its broad handler),
  `export_phase2_predictions_from_saved_models.py` (SCORE_RECONSTRUCTION_REFUSED: it divided the stored weights by their sum,
  a different predictor; disposition in 1c), and `RunArtifactWriter.save_shap_values` (EXPLANATION_TARGET_MIGRATION_REQUIRED,
  before SHAP is imported; enabled by 1b). Each refusal is the script's first statement, before any model library, data or
  output.
- Readers of the evaluation row: run_phase2_eval, run9_ablations, smoke_all_models (and its offline check script), the A1
  boundary probe and the report generator read `SELECTED_ENSEMBLE_ROW`; historical files keep their `ENSEMBLE_STACKER` bytes.

## Measured

- The exact kernel on Run 15's real 883,127 x 13 out-of-fold matrix
  (docs/measurements/MEASUREMENT_2026-11-07_exact-blend-kernel-run15.md): set 1a's implementation equals the owner's reference
  bit for bit on every row, as does an independent exact method (fsum over error-free products); `P @ w` differs on 356,050 rows
  by at most 3 units in the last place (2.22e-16) with 0 call changes at 0.1/0.3/0.5/0.7/0.9 under both >= and >; both score
  vectors hold 883,127 distinct values in one identical order, so every rank-based metric is unchanged.
- A member's joblib bytes are not a stable identity: they change after a dump/load round trip for every member type measured (a
  scaled logistic pipeline: 1,441 versus 1,473 bytes). The fitted parameters are hashed instead, once, at save.
- The census of revision 5 (282 score-surface candidates) reproduced byte-identical from a clean clone of the base, twice. The
  set-1a ledger (docs/package1/PACKAGE1_SET1A_CENSUS_LEDGER.json) gives each candidate one disposition; its stated limit: a
  pattern census plus a symbol cross-check is not a call graph, and the closure review completes with 1b and 1c.

## Verification (development sandbox)

- The whole suite on the candidate (development sandbox: Linux, Python 3.12.3, the environment of the training-and-test Linux lock,
  Git 2.48.1, a full clone; the gate's options apart from coverage; the models offline): 8,815 passed, 54 skipped, 6 xfailed, 5
  subtests passed, 0 failed, in 8 minutes 59 seconds; the network guard 8,875 checked, 0 tried; the warning-filter guard 8,875
  checked, 0 changed. The working-tree guard reported docs/CHANGELOG.md and docs/ROADMAP.md changed: I edited both while that run was
  in progress (my error, not a test's). The frozen candidate is therefore run again, on a clean clone of its tree, and that run is
  recorded with the delivery.
- Test by test against the same measurement of main (0e9db4e1: 8,693 passed, 54 skipped, 6 xfailed; every guard clean): the 29
  retired tests all passed on main and are gone, the 151 new tests pass, and no other test changes its outcome or its reason. The 54
  skips are the sandbox's: no Rscript, no Windows junctions, no cached ESM-2 weights, the three local-data tests, the five
  integration tests marked TODO, and the Windows-targeted postflight checks.
- Forecast for the owner's validation, test by test from set "aj"'s Windows record (its full.xml: 8,665 passed, 83 skipped, 5
  xfailed on this same tree): the 29 retired tests all passed there; the new and changed tests take their sandbox outcome (every one
  passed); every other test keeps its Windows outcome -- 8,787 passed, 83 skipped, 5 xfailed.
- Targeted checks after the fixes: the Docker coverage gate (13), the local-data contract (21), the X_seq refusal contract (11), the
  release reader and the interpretation contract (135, with Git 2.48.1 and full history), the documentation claims and the
  suite-size ratchet (190).

## Errors, and their corrections

- The container was reclaimed mid-session and its temporary folders were wiped (the worktree, the environments and a running
  suite). The 42 recorded edit steps were replayed from the session transcript onto a fresh clone; every replay assertion
  re-passed. The worktree diff is now checkpointed to persistent storage after every milestone.
- The first full run of the candidate found three defects of mine, each fixed and re-run: the API image lacked
  `prediction_contract.py` (the DOCKERCOPY-1 gate); the new lifespan test was not in the measured list of tests that run the API
  startup (it already requested `api_without_local_data`; the list now names it); and the two X_seq refusal tests built an
  ensemble without the combiner declaration that every fitted ensemble now carries, so the declaration refusal fired first --
  their fixture now declares the blend, and the negative control asserts that the failure is the placeholder member being called.
- Nine failures of that run were the sandbox, not the change: its Git was 2.43 (the release-approval reader refuses below 2.45,
  by design) and its clone was shallow (two tests read historical commits). With Git 2.48.1, built from the Ubuntu archive's
  source checked against its SHA-256, and a full clone, all nine pass.
- Two test expectations of mine were wrong before that run and were corrected: evaluate() rounds metrics through pandas, so the
  test compares through the same rounding; and 0.25 x 0.2 + 0.75 x 0.6 is 0.5 exactly while naive float arithmetic gives
  0.49999999999999994, so the test computes the expectation with exact fractions.
