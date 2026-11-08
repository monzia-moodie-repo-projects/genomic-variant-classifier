# Package 1, set 1c: delivery record against the acceptance matrix

Author: Monzia Moodie
Base: merged set 1b, a29af8543dad12a751a785a69e547a541f2a6d15 (tree b0bb918b). Gates: PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md
(revision 1) as amended by revision 2. No gate is changed here; where the implementation refines or goes beyond what a gate asks,
the refinement is named below so the owner can rule on it. Census dispositions after 1c: PACKAGE1_SET1C_CENSUS_LEDGER.json (all
282 candidates; no disposition defers to a later set). Tests: tests/unit/test_prediction_contract_1c.py unless named otherwise.

## The shared input: a completed run's bound split predictions

`evaluation/prediction_artifacts.load_completed_split_predictions(run_dir, role)` is the one way a set-1c consumer reads a run's
scores. The run must verify (`verify_completed_run` re-measures every listed artefact); the predictions file, its block descriptor
and the manifest are each read ONCE and cross-checked -- split, rows, row count, result block and member blocks (and their order)
must agree between the descriptor and the manifest, the columns must be the identity columns followed by exactly the predictor's
members, the labels must reproduce the recorded label digest, and `selected_score` must be binary64, finite and in [0, 1]. The
parsed bytes are exactly the verified bytes. Tests: `test_the_bound_reader_*`, `test_changed_prediction_bytes_are_refused`,
`test_a_block_descriptor_that_disagrees_with_the_manifest_is_refused` (four fields), `test_relabelled_or_reordered_predictions_are_refused`.

## G7.1 and G12: the threshold workflow (scripts/calibrate_thresholds.py) -- migrated

MEASURED on the shipped code (the function executed on a synthetic split of 20,000 rows, prevalence 0.15): its search walked
`reversed(precision[:-1])` and returned the first, i.e. HIGHEST, qualifying threshold while its docstring promised the lowest. It
returned 1.000000 (238 calls) for both the 0.80 and the 0.95 target; the documented thresholds are 0.566 (3,141 calls, positive
predictive value 0.8001) and 0.703 (1,867 calls, 0.9502). The "Likely pathogenic" tier was therefore empty. It also scored the
logistic meta-learner (removed in set 1a), fitted Platt and temperature scaling, chose between them by their error and placed
thresholds on the SAME rows, fell back silently to the best-precision point, and wrote thresholds on a calibrated scale serving
never produces.

| Required (revision 2: "bound predictor, input, calibration transform, score stage") | Implementation | Test |
|---|---|---|
| bound predictor and input | the completion manifest's selected predictor and the bound reader's identities (run, completion manifest digest, file digest, split, rows, result block, label digest) are written into the record | `test_the_threshold_cli_writes_a_bound_policy_from_a_completed_run` |
| selection | the repository's exact sweep over every achievable threshold and Objective A, "the most sensitive operating point whose predictive value clears a floor" (evaluation/thresholds.py), with its declared total order; the negative side is the same machinery on (1 - label, 1 - score) in a declared score space | `test_the_policy_selects_objective_a_exactly_on_both_sides` (against an independent brute force) |
| calibration transform and score stage | `transform: identity`, stage `selected_convex_blend_positive_probability` -- the value serving returns. A predictive value at a threshold depends only on which rows clear it, and a strictly increasing transformation never changes that set, so a calibration cannot change these operating points; it could only move the written number off the served stage | `test_the_policy_is_on_the_served_stage_and_a_monotone_transform_cannot_change_its_calls` (temperature and Platt scaling) |
| no fallback | an unreachable floor refuses with every operating-point outcome; nothing is written (exit 1) | `test_an_unreachable_floor_refuses_with_every_outcome_and_no_fallback`, `test_the_threshold_cli_refuses_without_writing` |
| record integrity | strict JSON, atomic write, `record_sha256`; `load_decision_policy` refuses an edited record, a malformed or inexact threshold, or broken band geometry; `require_predictor` refuses another combiner | `test_the_record_round_trips_verifies_and_refuses_edits` |
| what the record claims | neutral band names, `status: research_operating_points_not_clinically_qualified`, the certification blockers the selection implies (chosen and reported on the same rows; no held-out validation) | `test_the_record_names_its_blockers_and_is_not_a_clinical_classification` |

**MEASURED, and put to the owner: DECISION-POLICY-OVERLAP-1.** Objective A on each side maximizes that side's coverage, so each
side's region extends into the middle as far as its floor allows. On realistic runs the two regions MEET: a completed run of this
repository's synthetic fixture (AUROC 0.906) gives positive scores >= 0.554 clearing 0.80 and negative scores <= 0.618 clearing
0.80 (16 rows in both); a near-separable run (AUROC 0.9997) still overlaps at moderate floors of 0.95. Any rule that assigns the
overlap changes a side's call set and voids its selected operating point, so the policy REFUSES rather than clamping (the old
script clamped silently). The default floors therefore refuse on such data. Choosing the partition rule is a scientific decision;
options: (a) the field-standard calibration of Pejaver et al. 2022 (American Journal of Human Genetics) -- a LOCAL posterior
probability with a stated prior and one-sided bootstrap confidence bounds, mapped to evidence strengths; (b) a sequential rule
(positive side first, the negative side chosen on the remaining rows); (c) positive-side operating points only.

## G7.2: export metadata bound to the selected predictor

| Gate | Implementation | Test |
|---|---|---|
| only evidence bound to the selected predictor | `PipelineMetadata.val_auroc` and `n_train` are removed; reading them raises `ContractError` (PIPEMETA-1), also when an older pickle restores the value (a property outranks the instance dictionary). `evaluation_evidence` is `selected_predictor_evaluation_evidence(run_dir, combiner)`: refused unless the completion manifest names exactly this combiner; AUROC recomputed from each bound split with `evaluation.metrics.auroc`, with the file and row identities; or `{"status": "unavailable", "reason": ...}`. `fit_input_rows` comes from the ensemble; `feature_names` is what the pipeline renders (a different supplied list is refused) | `test_the_evidence_is_recomputed_from_bound_predictions_for_one_predictor`, `test_the_pipeline_metadata_carries_only_bound_evidence` |
| the exporter | `scripts/export_model.py` requires a verified completed run (an export is a deployment candidate); the ensemble file and the scaler bytes must be the ones the manifest lists (the scaler is unpickled from the verified bytes, never a bare `joblib.load` of an unlisted file); a containment refusal is never swallowed by the smoke test. Gate C10 still refuses the load in production | `test_export_refuses_an_incomplete_run_and_writes_nothing`, `test_export_refuses_a_scaler_the_run_did_not_record`, `test_export_is_refused_by_gate_c10_under_the_default_authority`, `test_export_writes_a_pipeline_whose_metadata_is_bound_to_its_predictor` |

## G7.1 and G12: the remaining consumers (each refused in production by gate C10; under an isolated authority they consume the selected score)

| Consumer | Its own defects: resolved here | Test |
|---|---|---|
| api/schemas.py | the threshold loader ran at IMPORT, read a path relative to the current directory (`Path("")` is "."), swallowed every error into five fixed tiers: removed with the five-tier helper | `TestScoreToClassificationRemoved` (test_api.py) |
| validate_external.py | the tiers were passed as (lo, hi) TUPLES to `y_prob >= thr`; the ROC frame appended a threshold to equal-length arrays (it raised); the PR frame invented 1.0; scores rounded; the unbound `pipeline_val_auroc`. Now: an explicit `--decision-policy`, verified and refused for any other predictor before the cohort is read; no threshold without one | `test_external_validation_applies_no_threshold_without_a_policy`, `test_external_validation_applies_a_policy_only_for_its_own_predictor` |
| validate_clinvar_temporal.py | "Temporal filter guarantees no training overlap" was false (a re-reviewed training variant passes the cutoff) and `--splits-dir` was never read; rates were fabricated as 0.0 (`max(denominator, 1)`); a failed enrichment was a warning. Now: overlap MEASURED against the training split's locus identities and removed, or "unmeasured" with the reason; typed rates; recorded enrichment | `test_training_overlap_is_measured_and_removed_or_reported_unmeasured`, `test_fixed_reporting_thresholds_refuse_undefined_rates` |
| validate_lovd_enriched.py | the model input `consequence` was fabricated for every row (every SNV missense, every length change frameshift, in-frame included); fills uncounted; a hard-coded copy into `G:/My Drive`. Now: refused without the cohort's own column unless `--allow-inferred-consequence` (in-frame classes, unknown alleles not guessed, recorded); fills counted; no Drive copy | `test_the_lovd_consequence_guess_is_refused_by_default_and_honest_when_allowed` |
| build_lovd_index.py | MEASURED: substring matching with the pathogenic terms first labelled "not pathogenic" and "probably not pathogenic" as 1 and "Class 10" as 0. A desktop Chrome User-Agent was sent to pass LOVD's challenge for automated clients. The ECE (expected calibration error) zipped non-empty bins with all bins (recorded for package 2). Now: exact normalised terms; the project's client name; the repository kernel; named, logged exceptions | `test_every_lovd_term_maps_to_its_own_class_and_nothing_is_guessed`, `test_the_lovd_client_names_itself_and_uses_the_calibration_kernel` |
| calibration_analysis.py | consequence classes under 50 rows vanished silently: logged and recorded | -- |
| conformal_prediction.py | MEASURED: `np.quantile` at level ceil((n+1)(1-alpha))/n lies between the k-th and (k+1)-th order statistics, and for k > n it returned the largest score where the guarantee needs every label; the "interval" s +/- q is not a conformal object for binary labels. Now: `conformal.split.conformal_quantile`, prediction sets, full precision | `test_the_conformal_script_uses_the_finite_sample_order_statistic` |
| conformal/calibrate.py, run_conformal_calibration.py | default `ensemble_prob` was absent from a 1b file. Now `selected_score`; `--run-dir` reads the bound predictions; every result records its substrate | `test_conformal_calibration_reads_a_completed_run_and_records_its_identity`, `test_the_conformal_command_runs_on_a_completed_run_and_names_it`, test_calibrate.py |
| training/continual_trainer.py | indexed the 1-D score as a matrix (`[:, 1]`), behind its fail-closed guard | `test_no_consumer_indexes_the_one_dimensional_pipeline_score_as_a_matrix` |

## Historical and forensic dispositions (matrix set-1c list; G2.4)

| File | Disposition | Test |
|---|---|---|
| export_phase2_predictions_from_saved_models.py | `quarantine_policy.HISTORICAL_EXECUTION_DENIED` (moved from the transition policy). `quarantine_policy_sha256` changes; no committed decision or record stores it and the default authority decides live | test_containment_completion.py |
| transition_policy.py | RETIRED: no transition refusal remains; nothing imports it | `test_the_transition_policy_is_retired_and_nothing_imports_it` |
| diagnose_phase2_prediction_reconstruction.py | labelled: it computes a NORMALISED-weight blend and the meta-learner; neither is the selected predictor; execution denied | test_containment_completion.py |
| apply_d1_fixes.py, fix_oof_index.py | SUPERSEDED: refuse before reading or writing (measured on a29af854: both already failed without writing; F-13 would reinsert the sequential `_train_row_idx` fallback) | `test_superseded_and_retired_scripts_refuse_before_touching_anything` |
| forensics/verify_oof_alignment.py | RETIRED (G2.4): it joined `y_train[_train_row_idx]`, a position over the OOF subset taken as a training-split row | same |
| forensics/probe_oof_builder.py, scan_oof_join_keys.py | labelled historical (read-only; they join nothing) | -- |
| run14/15/17_observability.py | the blend weights came from three files no run wrote ("meta_learner_coefs.json" among them) under the label "meta-learner contribution"; now the selected predictor's weights from a verified completion manifest, or unavailable with the reason; run17 still generated from run15 | `test_observability_reads_the_selected_blend_from_the_verified_manifest`, `test_the_run17_observability_script_is_still_generated_from_run15` |

## G7.4

The complete consumer list with each file's disposition is in the installer's README (and in PACKAGE1_SET1C_CENSUS_LEDGER.json).

## Refinements for the owner's ruling (implemented; reversible)

1. **The decision policy replaces the five ACMG/AMP-named tiers with neutral bands** (`positive_high_ppv` ... `negative_high_npv`).
   Serving removed the tiers in 3.0.0; a classifier threshold is not an evidence-strength calibration.
2. **Thresholds are on the served (raw) score stage and no calibration transform is fitted by this workflow**, because a strictly
   increasing transform cannot change a predictive-value operating point (tested). This also removes the two package-2 records
   "double scaling" and "threshold-stage mismatch" for this workflow.
3. **Overlapping regions refuse** (DECISION-POLICY-OVERLAP-1 above) instead of a chosen tie rule.
4. **An export requires a completed run** (not only bound metadata), so an artefact's provenance is complete evidence.
5. **The LOVD client identifies itself.** Access through LOVD's challenge then depends on the owner's own session and LOVD's
   policy for automated clients, not on impersonating a browser.
6. **Package-5 defects in validate_external (tuple thresholds, ROC frame) are resolved here** because the rewrite removed the code
   that had them.

## Recorded, not fixed (later packages)

- Package 2: evidence-strength calibration (DECISION-POLICY-OVERLAP-1 option a); report_generator UTC labelling.
- The enrichment fills of the validators (absent gnomAD frequency 0.0, absent AlphaMissense 0.5) are now COUNTED, not changed: the
  same constants engineer_features substitutes for an absent value (models/variant_ensemble.py: `allele_freq` 0.0,
  `alphamissense_score` 0.5), and a constant is a fabricated measurement either way -- a package-level decision.
- Still open from set 1b: GNN-FEATURE-ORDER-1 (blocks Run 17) and MAXTRAIN-SPLIT-REPERSIST-1.
