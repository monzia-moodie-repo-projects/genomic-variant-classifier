# Package 1, set 1b: delivery record against the acceptance matrix

Author: Monzia Moodie
Base: merged set 1a, b051bb3d5d22d74787f0fb4c11ceef129efb606a (tree 6bc717f5). Gates: PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md
(revision 1) as amended by revision 2. No gate is changed here; where the implementation refines what a gate asks, the refinement
is named below so the owner can rule on it. Census dispositions after 1b: PACKAGE1_SET1B_CENSUS_LEDGER.json (all 282 candidates,
each re-checked for every name whose behaviour or presence 1b changes). Tests: tests/unit/test_prediction_contract_1b.py unless
named otherwise.

## G2. Original row identity through training artifacts

| Gate | Implementation | Test |
|---|---|---|
| G2.1 composition, no sequential fallback | `evaluation/split_identity.py` (dataset identity of each split as prepared: role, columns, binary64 features, labels, variant identities; prediction units in two unit contexts) and `evaluation/oof_identity.py` (training-split row -> optional subsample -> `oof_fit_indices_` -> OOF row). `VariantEnsemble.fit` now records `fit_input_rows_`, `oof_labels_`, `oof_fold_` (from the exact splits every member uses; `cross_val_predict` receives them) and `oof_cv_scheme_`, and persists them | `test_the_oof_table_carries_verified_training_split_identity`, `test_a_subsample_is_composed_through_and_verified`, `test_every_fit_row_has_exactly_one_recorded_fold_and_gene_groups_are_disjoint`, `test_without_gene_groups_the_recorded_folds_are_the_declared_stratified_splits` |
| G2.2 failed binding blocks publication | the composed rows' labels must equal the labels the ensemble fitted on, row by row; any mismatch or missing record raises `ContractError`; `run_phase2_eval` exits 2 without a completion manifest | `test_a_failed_identity_binding_blocks_mainline_publication`, `test_an_ensemble_without_recorded_oof_identity_is_refused_not_guessed` |
| G2.3 Run 15 derived record | `RUN15_OOF_IDENTITY_DERIVED_RECORD.json`, written by `scripts/derive_run15_oof_identity.py` from the owner's Run 15 files (ensemble.joblib sha256 71f87e32...929). The orchestrator is read by the new inert reader (`inert_pickle.py`: no class imported, no model deserialized). Stored `oof_fit_indices_` equal the reconstructed legacy carve exactly (sha256 5cc3e4ba... both); AUROC through the mapping 0.85360 to 0.99906 per member, joined naively 0.4978 to 0.4999; all 13 legacy sidecars are the fold concatenation (AUROC 0.4987 to 0.4996 if paired element-wise). The original OOF file is not rewritten | `test_a_historical_oof_identity_is_derived_from_the_artifact_and_verified`, `test_a_historical_mapping_that_does_not_reproduce_is_refused`, inert-reader tests |
| G2.4 forensic helpers | owned by set 1c (revision 2); dispositions in the 1b ledger: `historical_owned_by_1c` | ledger |
| Sidecar (matrix set-1b list) | `{name}_oof_indices.npy` now holds the fit-input row of each element of `{name}_oof.npy`; `{name}_oof_folds.npy` its fold; the meta file names the semantics. Historical sidecars are refused by `read_member_oof_checkpoint` with their meaning, never paired | `test_the_member_sidecar_is_aligned_with_its_oof_vector`, `test_a_historical_sidecar_is_refused_with_its_meaning` |

## G7.3. Historical ENSEMBLE_STACKER records

The derived record carries `code_provenance`: the run's master log names `HEAD: 4b9de1c`; the pre-rewrite commit
4b9de1c319739824955bb165c295e56e444626ce (preserved bundle, sha256 c9b29cb6...) holds variant_ensemble.py blob 615590a4 (identical
in the rewritten 9222a4b), whose `predict_proba` returned `base_preds @ self.blend_weights_` whenever weights were stored and whose
`evaluate()` labelled that row ENSEMBLE_STACKER with calls at `p > 0.5`; the orchestrator stores 13 finite weights. Run 15's
ENSEMBLE_STACKER rows are therefore the stored blend under the historical `@` kernel. Their bytes are unchanged.

## G8. Publication

| Gate | Implementation | Test |
|---|---|---|
| G8.1 completion manifest | `RunArtifactWriter.complete()` writes `completion_manifest.json` last: every output written by the writer (digest measured at write time and re-measured) or registered (models, splits), the selected predictor, the prediction-block digests, the OOF identity evidence and the deliverables. A manifest left by an earlier run is retired (renamed, never onto an earlier retirement: a timestamp can repeat) before anything is written. `verify_completed_run` re-measures every byte | `test_completion_is_published_last_and_verifies_every_byte`, `test_a_file_changed_after_it_was_written_prevents_completion`, `test_a_missing_required_output_or_capture_prevents_completion`, `test_a_rerun_retires_the_previous_completion_before_writing_anything`, `test_a_retirement_never_overwrites_an_earlier_retirement` |
| G8.2 fault injection | a failure before completion leaves every output readable and the run refused as completed evidence; in both drivers a failed requested deliverable or identity binding means no manifest and no aggregate row | `test_fault_injection_before_completion_leaves_an_inspectable_unverifiable_run`, `test_a_failed_deliverable_leaves_the_ablation_incomplete_and_unaggregated` |

## G12 (revision 2): the set-1b row

| Entrypoint | After 1b | Test |
|---|---|---|
| `scripts/run9_ablations.py` | enabled: preflight before any fit; one bound capture per split (`VariantEnsemble.score_split`); identity-bearing test and validation predictions; OOF identity; calibration points; member-score attributions; no broad handler; completion; atomic aggregate of completed runs only; the gene-disjoint carve (it passes `gene_symbol` now) | `test_the_ablation_harness_runs_to_a_verified_completion`, `test_the_ablation_preflight_refuses_before_any_fit`, `test_the_ablation_harness_is_enabled_and_its_deliverables_are_never_swallowed` |
| `scripts/run_phase2_eval.py` | OOF identity and completion manifest; the `--max-train` path fixed (one subsample of features, labels, the sequence attachment and the metadata; the GNN, hetero-GNN and holdout blocks use the aligned in-memory metadata); every output through the writer; a failed requested GNN deliverable is recorded and blocks completion | `test_the_mainline_max_train_path_composes_identity_and_completes` |
| `RunArtifactWriter.save_shap_values` | retired (refused with the standing feature-level reason; nothing imported or written) | `test_the_retired_shap_writer_refuses_before_importing_shap_or_writing` |
| explanation | `save_member_score_attributions`: phi_ij = w_j (p_ij - mu_j), b = sum_j w_j mu_j, mu from a declared background capture (the validation split), exact closure residual against the published score | `test_member_score_attributions_close_on_the_published_score`, `test_a_background_from_another_predictor_is_refused` |
| calibration | `calibration_points.parquet` (bins with edges and real counts, global and per consequence, equal to scikit-learn on non-empty bins and checked against the report) and `consequence_performance.parquet` | `test_calibration_points_reproduce_scikit_learn_and_keep_the_counts`, `test_the_calibration_writer_separates_points_from_consequence_performance` |
| model insights (detector, agent) | completed runs only; the identity schema; members from the run's OOF identity | `test_model_insights_*`, `test_the_insights_detector_reads_only_completed_runs`, `test_folds_that_share_genes_are_flagged_from_a_real_run` |
| `scripts/calibrate_thresholds.py`, the score-reconstruction exporter | still refused (1c) | `test_prediction_contract_transition.py` |

## Refinements for the owner's ruling (implemented; reversible)

1. **The SHAP writer is retired rather than renamed.** The matrix says "SHAP explains the selected predictor (exact linear attributions
   of the blend)". The attributions are exactly that (for a model linear in member scores with an independent background they are
   the interventional Shapley values), but a method named `save_shap_values` that computes no SHAP would mislead; it refuses and
   names `save_member_score_attributions`.
2. **The attribution background is a declared capture of another split (validation), not the explained rows.** It must come from
   the same predictor (combiner digest and member producers checked). Using the explained rows' own mean would make every
   explanation self-referential.
3. **Split identities describe the splits as prepared**, before the subsample and before the GNN blocks rewrite `gnn_score` in place;
   the matrix each member actually received is bound separately in its prediction block's ordered-input digest. So every ablation
   arm shares one source-row identity per variant.
4. **The ablation harness now passes `gene_symbol` to `fit`** (gene-disjoint calibration carve and inner folds, as the mainline run),
   and the holdout ensemble of `--unseen-gene-holdout` does too. Earlier ablations used the legacy label-stratified carve; their
   numbers are not comparable with new arms on calibration.
5. **One definition each** moved into `models/combiner.py` (scope, member modalities, member draw schedule) and
   `prediction_contract.variant_identity_of`, so the offline split capture and serving bind the same combiner identity (the scope
   digest is unchanged: 66b6c895...).

## Findings recorded, not fixed in 1b (owner decision needed)

- **GNN-FEATURE-ORDER-1 (blocks a full mainline run).** `run_phase2_eval` fits the ensemble before the GNN blocks inject `gnn_score`
  and `hetero_gnn_score`. Before the fit both columns are the feature engineering default 0.5 on every row (measured: the census
  reports both dead), so the driver's census aborts (exit 2) unless `--allow-dead-features`, and `VariantEnsemble.fit` itself raises
  for 10,000 rows or more unless `allow_zero_variance_features`; no launcher passes either. With both overrides the members would
  be fitted on a constant and evaluated on injected scores. Run 15 predates both guards and had the same order. Resolving it is a
  design ruling (where the graph scores are computed, from which labels, and how leakage into the training rows is prevented).
- **MAXTRAIN-SPLIT-REPERSIST-1.** Under `--max-train` the GNN block re-persists the subsampled `X_train` into `splits/` beside the
  full `y_train` and `meta_train`. The ablation harness's new preflight refuses such a directory (row counts differ) instead of
  reading it.
