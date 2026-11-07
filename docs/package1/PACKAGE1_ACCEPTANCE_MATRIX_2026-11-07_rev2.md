# Package 1 (prediction contract): acceptance matrix, revision 2 (superseding amendment)

Author: Monzia Moodie
Supersedes, without replacing: PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md, sha256
3669fe07275edadba407f8c160fd8ded1d41edb186e0099f851ef1c009a479f3 (11,887 bytes; the original planning baseline,
preserved unchanged). Base commit 0e9db4e114efdf174e1b4e3ab1e196062b485d5d.
Recorded owner rulings that amend it: decision ccffb8cf (exact numerical contract; census and entrypoint closure) and
decision 38f15c7a (three sequential installers with transition containment; exact record-to-artifact binding).
Every gate of revision 1 stands unless amended below. A gate changes only by a recorded owner ruling.

## Amendments to existing gates

- G3.4 (kernel). Was `fixed_order_compensated_dot2_v1`. Now `exact_binary64_products_sum_rne_v1`: the exact real sum
  of the stored binary64 products, rounded once to binary64, nearest with ties to even; canonical +0; a result above
  one refused. The historical `P @ w` keeps the identity `numpy_float64_matmul_stored_weights_v1` for replay only;
  Dot2 is a non-selectable candidate. A kernel change is a predictor-identity change. The migration report covers
  bits, ULP distances, absolute differences, call changes at every declared edge under both >= and >, and order
  changes; the observed 3-ULP maximum is not a tolerance. Measured on Run 15's 883,127 x 13 matrix: 356,050 rows
  differ from `@` (max 3 ULP), 0 call changes at 0.1/0.3/0.5/0.7/0.9 under both conventions, 0 order changes; an
  independent exact method (fsum over error-free products) equals the integer kernel on every row.
- G6 (serving). Inference requires a READY runtime binding: registered, the declared production record, an EXACT
  roster AND a matching fitted-predictor identity. `/predict` and `/batch` refuse with 503 before any feature
  preparation or member call, outside the broad handlers; `/rsid` returns the locus with
  `prediction_unavailable_reason`. A 3.0.0 reader refuses any other prediction schema version.
- G3.5 extended (projections). `RosterAlignment.SERVING_SUBSET` remains readable as a diagnostic and is never ready;
  `InferencePipeline.load` refuses any roster state but EXACT. Registry records gain `predictor_contract_sha256`
  (`InferencePipeline.predictor_contract_sha256()`: combiner digest -- ordered member slots, fitted member identities,
  prediction stages, exact weights, kernel, scope -- plus declared features, scaler and missing-value policy values,
  and class order); a record without it cannot admit an inference load; a different value refuses the load and makes
  the binding `PredictorContractAlignment.DIFFERS` (never ready).

## G12. Transition invariant (new)

Each installer leaves every affected, reachable operation either correct or explicitly refused; no merged state is a
known broken intermediate. Refusals use the existing containment convention (`require_execution_permitted` as the
first statement, before any model library, artifact, data or output) with a separate, dependency-free
`transition_policy.py`, so the quarantine policy and its admission digest are unchanged. Removing a refusal is the
enabling set's job, together with the migration and its tests.

| Delivery | Implements | Explicitly deferred | Required state after merge |
|---|---|---|---|
| 1a | Contract core, exact kernel, model adapters, serving, readiness gate, record-to-artifact binding, transition guards and refusal propagation | OOF/writer migration; threshold, export-metadata and validator migrations | Supported prediction paths use the authoritative contract; incompatible consumers cannot execute or publish |
| 1b | Original OOF identities, completion manifest, explanation and calibration writers | Remaining consumer migrations | Migrated writers consume bound predictions and name their target; remaining incompatible consumers stay refused |
| 1c | Exporter metadata, threshold workflow, validators, forensic consumers | Separately identified research or historical paths | Every in-scope consumer delegates or keeps a documented refusal; package 1 closes |

## Entrypoint dispositions affected by set 1a

| Entrypoint (callers) | After 1a | After 1b | After 1c | Authoritative producer | Refusal point and reason | Publication behaviour | Test |
|---|---|---|---|---|---|---|---|
| API `/predict`, `/batch` (HTTP clients) | supported when READY; else refused 503 | same | same | InferencePipeline.score -> exact blend | `_require_ready_for_inference`, before rendering | 3.0.0 records, one serializer, identity pairing | test_api.py TestServingContract, test_an_unattributed_model_refuses_...; test_runtime_attribution.py readiness tests |
| API `/rsid` | locus always; prediction only when READY | same | same | same | readiness check; reason field | `prediction_unavailable_reason` | test_api.py TestRsidEndpoint |
| `InferencePipeline.load` (api.main, export_model verify, validators, continual_trainer) | supported only for EXACT roster + matching contract, and C10 still refuses real admission | same | same | registry record + loaded spec | admission (C10), roster EXACT, contract match | none | test_containment_serving.py, test_runtime_attribution.py projection/contract tests |
| `InferencePipeline.predict_proba/score` (offline callers) | supported | same | same | exact blend over one member capture | construction refuses unsupported modalities (cnn_1d) | float64 scores | test_prediction_contract.py golden fixture |
| `VariantEnsemble.predict_proba/evaluate` (run_phase2_eval) | supported; row SELECTED_CONVEX_BLEND | same | same | exact blend; one member bundle | no weights / undeclared combiner refused | per_model_metrics rows | test_prediction_contract.py TestGoldenFixture |
| `VariantEnsemble.load` (run_phase2_eval resume, export_model) | supported for declaring orchestrators; legacy needs explicit declaration | same | same | orchestrator `selected_combiner` | ContractError without declaration | none | TestPersistence |
| scripts/calibrate_thresholds.py (CLI and helpers) | refused | refused | migrated (bound predictor, input, calibration transform, score stage) | none until 1c | import-time `THRESHOLD_CONSUMER_MIGRATION_REQUIRED` | none (refused before load, data, fit, output) | test_prediction_contract_transition.py, test_containment_scripts.py |
| `RunArtifactWriter.save_shap_values` | refused | migrated (blend member-score accounting, closure residual) | supported | none until 1b | before importing SHAP: `EXPLANATION_TARGET_MIGRATION_REQUIRED` | no file | test_the_shap_writer_refuses_... |
| scripts/run9_ablations.py | refused (independent member re-prediction, unbound writers) | migrated onto one bound capture; SHAP refusal propagates | supported | none until 1b | import-time `ABLATION_WRITER_MIGRATION_REQUIRED`; `except ContainmentError: raise` before the broad handler | none | guard + propagation tests |
| scripts/run_phase2_eval.py | supported (training/evaluation; reads SELECTED_ENSEMBLE_ROW) | OOF identity and completion manifest | same | evaluate() selected row | none | metrics of the selected blend; not a completion manifest | test_smoke_all_models.py, scripts/test_smoke_checks.py |
| scripts/export_model.py | supported for runs whose ensemble declares its combiner; legacy run directories refused at VariantEnsemble.load | same | metadata bound to the selected predictor (G7.2) | from_variant_ensemble | cnn_1d member refused at construction; legacy refused | pipeline artifact (not admitted: C10) | TestPersistence, modality refusal test |
| calibration_analysis, conformal_prediction, validate_external, validate_clinvar_temporal, validate_lovd_enriched, build_lovd_index, continual_trainer | refused in production by C10 at InferencePipeline.load; under an isolated authority they consume the selected score | same | each re-pointed and its own defects resolved or recorded | pipeline.predict_proba | C10 | their own outputs (1c) | test_the_default_authority_refuses_even_a_registered_pipeline |
| report_generator (roc_data) | supported for SELECTED_ENSEMBLE_ROW; the retired label is not attributed | same | same | caller-supplied curves | warning, no calibration plot | report | test_core.py |
| scripts/smoke_all_models.py gate | requires SELECTED_ENSEMBLE_ROW; a pre-package-1 file is blocked by name | same | same | evaluate() | gate failure | gate result | scripts/test_smoke_checks.py (7 cases) |
| historical scripts already denied (diagnose_phase2_prediction_reconstruction, run10b) | refused (quarantine policy, unchanged) | same | same | none | HISTORICAL_EXECUTION_DENIED | none | test_containment_completion.py |

The full 282-candidate census of revision 5 (reproduced byte-identical from the base) is dispositioned per file in
the installer's ledger; candidates not listed above have no score effect in set 1a or are owned by 1b/1c with their
current behaviour unchanged by 1a.

## Delivery and validation (decision 38f15c7a)

1a is validated as the changed candidate (earlier results are evidence, not certification), frozen with its
receipts (base commit, installer payload, this revision, tested results), then dry run -> -Validate -> -Promote ->
branch -> pull request -> merge. 1b is built against the merged 1a state, 1c against merged 1b. Rollback respects
the dependencies: 1a is not reverted alone after 1b or 1c has landed.
