# Package 1 (prediction contract): frozen acceptance matrix

Author: Monzia Moodie
Base: 0e9db4e114efdf174e1b4e3ab1e196062b485d5d. Sources: the owner's rulings of 2026-11-07 (decisions 166a4bfb, 873f6578,
0a0a512a, 6c970843 and the answer "blend authoritative"), review bundle revision 4 (sha256 d095698e..., 111 reference tests),
and the measured refinements R12 (fixed-order compensated blend kernel) and S1-S3. Frozen before implementation: a gate is
changed only by a recorded owner ruling. Each gate names its test. "Refused" means a typed exception before any publication.

## G1. Identity of prediction units (S1, S2, S3)
1. One source row may expand into several prediction units; a repeated prediction unit is refused.
2. `source_observation_id` and `prediction_unit_id` are distinct namespaces (different strings even one-to-one).
3. Split role and fold are outside unit identity; dataset identity change changes both identifiers.
4. NumPy positions and class labels enter only through a recorded `.tolist()` conversion; bool, string and float positions are refused.
5. The ordered unit set is validated and digested once per immutable context; each probability payload is hashed once.
6. Benchmark (recorded, not a pass/fail threshold): construction time, per-block time and peak memory at 883,127 rows.

## G2. Original row identity through training artifacts (R7)
1. OOF rows persist original training-split row identity composed through: original rows -> optional subsample -> fit/calibration
   carve -> OOF rows, using the fitted `oof_fit_indices_`; no sequential-index fallback.
2. A failed identity binding prevents publication (no warning-and-continue).
3. Run 15's historical mapping is a separately identified derived record citing the artifact digest (71f87e32...), the legacy carve
   rule and the equality of the stored `oof_fit_indices_` with the reconstruction; the original OOF file is never rewritten.
4. Forensic helpers that equate `_train_row_idx` with a parent-table row are repaired or retired (list from the census).

## G3. The selected predictor
1. Every artifact declares its combiner explicitly; Run 15's declared combiner is the stored convex blend (exact binary64 weights,
   bound by hex; full ordered 13-member roster; member artifact and prediction-stage identities; fit-input stage identity).
2. Missing, non-finite, negative or mass-defective (|fsum - 1| > 1e-12) weights are refused; no fallback, normalisation, abs,
   pruning or logistic substitution.
3. The logistic stacker is a separately identified candidate and is never called on the authoritative path.
4. Kernel `fixed_order_compensated_dot2_v1` (R12). Historical parity is reported: maximum ulp difference from `base_preds @ w`
   and every decision change at 0.5 (and at legacy tier edges for historical records).
5. Unsupported modalities (for example the sequence input of `cnn_1d`) are refused at construction, before any member call.
6. A golden fixture on which blend and logistic differ substantially returns the blend from every authoritative consumer.

## G4. One shared prediction result, identity-bound end to end
1. Members return ProbabilityBlocks; bare arrays are refused at the publication-level scorer.
2. Reversed member rows, same-sized but different units, and changed input, split, schedule, execution or invocation identity are refused.
3. The combined result is a ProbabilityBlock whose input identity binds the component block digests.
4. Each member is called once and the selected combiner applied once per logical request; `evaluate()` reduces one captured
   member bundle (no second member pass).
5. Class mapping is derived from fitted labels (or a persisted explicit mapping), validated for every member and for the result,
   including the unused column; classes are canonical integers {0, 1}, positive class 1.

## G5. Numerical quantities and states
1. Mixed-unit totals (variance + entropy) and their triage thresholds are removed.
2. Quantities carry value, unit, status and reason; `unavailable` and `unresolved` are null with a reason; a computed zero stays zero.
3. Disagreement uses the exact-complement near/far Kullback-Leibler kernel with row-local status; (1, 1 - 2^-53) is resolved at
   3.8477e-17 nats; subnormal cases are unresolved; inputs outside [0, 1] are refused; no silent clipping anywhere.
4. Blend diagnostics bind the stored weights and member stage; the uniform roster spread is a separately named research view;
   the weighted diagnostic stays `unavailable` until its kernel matches the 800-digit oracle within the declared domain.
5. MC dropout and deep-ensemble mechanisms record supported/requested/applied randomness, ordered member or seed identities and
   degeneracy; schema-2 uncertainty fields with ddof = 0.

## G6. Serving contract 3.0.0
1. API version 3.0.0; the response carries the full-precision selected score, one tie rule (`positive_if_p_ge_0.5`) computed once,
   identities, units and statuses; no computational rounding, re-clipping or reclassification.
2. The legacy fields `uncertainty_epistemic`, `uncertainty_aleatoric`, the five-tier classification and score-distance confidence
   are absent; decision policy is reported as unavailable / unqualified.
3. Single and batch responses come from the same serializer; batch pairing checks exact cardinality and unit identity (strict zip
   only after the check).
4. Batch summary counts are removed or replaced by counts that partition the results (the old counts omit "Likely" labels).
5. C10 still refuses real admission.

## G7. Consumers of the score
1. Evaluation, writer, serving, export metadata and threshold workflows consume the selected score; none calls `meta_learner`.
2. Export metadata carries only evidence bound to the selected predictor (no inherited validation AUROC from another combiner).
3. Historical `ENSEMBLE_STACKER` records keep their bytes; a derived view attributes them to the blend only where provenance proves it.
4. The complete consumer list (from the census) is enumerated in the installer's README with each file's disposition.

## G8. Publication
1. The existing RunArtifactWriter (atomic per-file replacement) is extended with a completion manifest published only after all
   required outputs, identities, digests and replay states validate.
2. Fault injection before the manifest: the run stays inspectable and is refused as completed evidence.

## G9. Replay
1. Replay refuses changed values, ordering, inputs, classes, producer, precision, draw identity or schedule and execution binding.
2. The draw plan is validated separately (expected member/seed/order/count/scope); a replayed prefix of a longer plan is incomplete.

## G10. Process
1. Guarded installer from 0e9db4e1: dry run, -Validate, -Promote, branch, pull request; never commit to main.
2. Real offline and HTTP transport tests and interruption tests pass in the qualified candidate environments and the three CI
   targets; the owner's environments stay unchanged; the validation receipt binds the reviewed patch, candidate tree, environment
   and results.

## G11. Consumer census and dispositions (census of 73 files at the base; claims marked [V] re-verified by direct source reading)
Delivery as three sequential guarded installers (each validated and merged before the next starts), so every pull request stays reviewable:

### Set 1a: contract core and serving (G1, G3-G6, G9)
- models/mc_dropout.py: schema-2 records, stable disagreement with row-local status, explicit unavailable states, n_passes/n_members
  validation, recorded seed schedule, member-distinctness status; legacy total and triage flag removed; float32 cast removed.
- models/variant_ensemble.py: explicit combiner selector persisted at fit and save; classes_ derived and validated (TabularNN); one tie
  rule; `evaluate()` reduces one captured member bundle; label of new evaluation rows names the selected combiner.
- api/pipeline.py: carries the selected blend specification (exact weights, ordered roster, stages); one shared scoring path; refuses
  the cnn_1d projection at construction [V: 13-input stacker called with 12 columns raises]; preprocessing absence reported, never refitted.
- api/schemas.py, api/main.py: response 3.0.0; no reclassification, rounding or clipping [V]; batch pairing by identity; counts that
  partition [V: old counts omit "Likely"]; `top_features` and `coverage_interval` reported unavailable [V: never populated].
- Tests: test_api.py, test_containment_serving.py, test_containment_propagation.py, test_catboost.py (fixtures),
  test_runtime_attribution.py (fixture made width-sensitive: the constant mock hides the width defect), test_mc_dropout_*.py,
  test_tabular_nn_mc_dropout.py, test_smoke_checks.py and scripts/smoke_all_models.py (gate on the new evaluation label).

### Set 1b: identity through training artifacts and publication (G2, G7.3, G8)
- scripts/run_phase2_eval.py: OOF through RunArtifactWriter with original row identity composed via `oof_fit_indices_`; failure blocks
  publication [V: today swallowed as a warning]; the `--max-train` path repaired (WindowAttachment has `subset`, not `iloc` [V]; gene
  groups subsampled with X and y).
- models/variant_ensemble.py: the per-model `{name}_oof_indices.npy` sidecar fixed [V: OOF is row-position ordered, the sidecar is
  fold-concatenated, so element-wise pairing misaligns]; historical sidecars labelled.
- evaluation/prediction_artifacts.py: OOF requires identity columns; completion manifest; SHAP explains the selected predictor (exact
  linear attributions of the blend) [V: today explains meta_learner]; calibration table column semantics corrected [V]; atomic
  ablation append; strict manifest JSON.
- evaluation/model_insights_detector.py, agent_layer/agents/model_insights_agent.py: consume the new OOF schema [V: real OOF lacks `label`].
- scripts/run9_ablations.py: SHAP and OOF through the same owners.
- Run 15 derived mapping record (G2.3). Tests: test_prediction_artifacts.py, test_d1_d2.py (rename and real identity test),
  test_model_insights_*.py, test_base_model_dropout_is_loud.py.

### Set 1c: remaining consumers and historical scripts (G7)
- Through the pipeline (they receive the blend once 1a lands; each is re-pointed at the shared result and its own defect recorded):
  calibration_analysis.py, conformal_prediction.py, validate_external.py, validate_clinvar_temporal.py, validate_lovd_enriched.py,
  build_lovd_index.py, export_model.py (metadata bound to the selected predictor), calibrate_thresholds.py (shared score; no direct
  meta_learner call).
- Historical/forensic: verify_oof_alignment.py retired or repaired (it joins `_train_row_idx` as a train row); export_phase2_predictions_
  from_saved_models.py added to HISTORICAL_EXECUTION_DENIED [V: ungated]; diagnose_phase2_prediction_reconstruction.py labelled
  (computes both combiners); apply_d1_fixes.py / fix_oof_index.py marked superseded; observability scripts' "blend weights versus
  meta-learner" conflation and the never-produced meta_learner_coefs.json corrected. Test: test_containment_scripts.py.

### Recorded for later packages (not expanded into package 1)
- Package 2: calibrate_thresholds double scaling and threshold-stage mismatch (thresholds fitted on calibrated scores, applied to raw);
  build_lovd_index ECE bin misalignment; report_generator UTC labelling.
- Package 5: validate_external threshold broadcasting and ROC frame construction defects; performance_estimator reference guidance.
- Package 6: run_drift_monitor `final_code = max(exit_codes)` lets NOT CHECKED (4) enter the --auto-retrain branch [V]; retraining
  itself remains fail-closed in ContinualLearner [V]; registry promotion hint names a non-existent class.
- Package 7: continual_trainer `predict_proba(...)[:, 1]` on a one-dimensional result [V], behind the fail-closed guard.
