# Neural variance mask (TabularNN-based models)

Status: implemented 2026-06-13. Scope: model-internal only. No change to the
81-column feature schema, `TABULAR_FEATURES`, `PHASE_2_FEATURES`, the inference
contract, or the schema-drift baseline.

## Background

The Run 16 feature census found 37 of 81 matrix columns constant (29 unpopulated
/ stub / procurement-blocked sources, plus `gnn_score`, `af_1kg_*`, and the
sparse-but-real `is_mitochondrial` / `lovd_variant_class`). A proposal to trim
`TABULAR_FEATURES` 81 -> 51 was built, validated, and then **reverted**: the full
suite revealed ~40 tests across 10 files enforce a deliberate Phase-4 contract --
a fixed, fully-promoted schema with connector -> matrix wiring and safe defaults,
so a source lights up its feature the moment data is provided, with no schema or
retrain-code change. Those connector-flow tests are themselves silent-failure
guards. The constant columns are a *data-availability state*, not dead code.

The only concern with real teeth was neural inputs: a constant column survives
`StandardScaler` as an all-zero input neuron. The robust fix is to handle that in
the model layer, dynamically, instead of surgically editing the shared schema.

## What it does

`TabularNNClassifier` learns a keep-mask at `fit` time and applies it at predict,
mirroring the existing `self.scaler_` lifecycle:

- `fit`: `feature_mask_ = X.var(axis=0) > 0.0`; `n_features_in_ = X.shape[1]`. The
  net is built at `input_dim = feature_mask_.sum()`. If every column is constant
  (degenerate), the mask falls back to keeping all columns (never 0-width).
- `predict_proba` / `_predict_proba_single_pass`: apply `_apply_feature_mask(X)`
  before scaling.
- `_apply_feature_mask`: returns X unchanged when `feature_mask_` is absent
  (estimators pickled before this change still score), and raises on a column-count
  mismatch rather than silently misaligning.

## Why this scope

- `mc_dropout` and `deep_ensemble` clone-and-fit a `TabularNNClassifier` and pass
  X straight through, so the single change covers all three tabular neural models.
- `cnn_1d` consumes the DNA sequence (`fasta_seq`), not the tabular matrix, and is
  special-cased in the ensemble's fit/OOF/predict dispatch -- untouched.
- Trees, Logistic Regression, CatBoost, and KAN are unaffected (they ignore
  zero-variance columns inherently; CatBoost still receives the full DataFrame).

## Design choices

- **Threshold is exactly `var > 0`.** Only provably-zero-information columns drop.
  Rare-but-real columns survive (`is_mitochondrial`, var ~5e-4, is kept).
- **Dynamic / auto-adapting.** The mask is recomputed every fit, so when a
  connector populates a column it is automatically included next run -- the
  "populate-when-available" principle, honored in the model layer.
- **No leakage in OOF.** `cross_val_predict` re-fits per fold; each fold's mask is
  computed from its own training rows. The always-constant columns drop in every
  fold consistently.
- **Persistence is automatic.** Each estimator is whole-pickled by `joblib.dump`;
  `TabularNN` has no custom `__getstate__` (only `CNN1D` does), so `feature_mask_`,
  `n_features_in_`, the fitted `scaler_`, and the data-sized `model_` all persist.

## Interaction with the zero-audit harness

None. `agent_layer/harness/correctness_harness.py::_stage5_zero_audit` inspects
`engineer_features(df)` output (the 81-column matrix) against `KNOWN_ZERO_DEFAULT`.
The mask is strictly downstream of that matrix and changes nothing the harness
sees. The neural estimators are not in the stage-1 smoke set either.

## Tests

`tests/unit/test_tabularnn_variance_mask.py`: helper pass-through/select/mismatch;
fit drops exact-constants and keeps near-constant; all-constant fallback; predict
width consistency + mismatch raises; joblib round-trip preserves mask and
predictions; `mc_dropout` / `deep_ensemble` inherit the mask.

## Roadmap fit

A Phase-3 modeling refinement and part of the model-comparison deliverable. As the
DNA/RNA/protein/structure branches come online and populate columns, the mask
includes them automatically. Expected accuracy impact is modest (constants already
scale to zero); the value is smaller/cleaner nets and explicit, automatic
adaptation to available signal.
