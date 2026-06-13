# Session 2026-06-13 -- feature-matrix audit, 81->51 trim (reverted), neural variance mask

Author: Monzia Moodie
HEAD at close: 5de7806 (origin/main). Suite: 924 passed / 6 skipped / 0 failed.
Follows SESSION_2026-06-13_run16.md (Run 16 closeout) in the same day.

## 1. What prompted this

The Run 16 feature-matrix census (`outputs/run16/X_test_feature_census.csv`) found
**37 of 81** matrix columns constant on the gene-disjoint test split: 29 truly
unpopulated (no local data / stub / procurement-blocked source), `codon_position`
(an exact duplicate of `protein_pos`, corr=1.0), and a deferred/sparse-but-real
remainder (`gnn_score`, `af_1kg_*` x5, `is_mitochondrial`, `lovd_variant_class`).
The question raised: does a fully-promoted 81-column schema carry dead weight that
should be trimmed?

## 2. The 81->51 schema-trim attempt -- BUILT, VALIDATED, then REVERTED

**Design.** Relocate 29 dead + `codon_position` (30 columns) from `TABULAR_FEATURES`
to `PHASE_2_FEATURES`, keeping `gnn_score` / `af_1kg_*` / `is_mitochondrial` /
`lovd_variant_class` (deferred or sparse-real). Count 81 -> 51. Both independent
feature builders (`variant_ensemble.engineer_features` -- the API path -- and
`real_data_prep.DataPrepPipeline._engineer_features` -- the training path) were to
be unified on a fail-loud select so neither could silently emit 81.

**Validation passed at the unit level.** The patcher was conservation-checked
(union of kept + relocated == 81), idempotent, and produced exactly 51 columns from
both builders. Gate-2 (feature-count contract + `test_api`) passed at 80.

**Full suite revealed it was wrong.** Gate-3 (`pytest -q`) produced **40 failures
across 10 files** (`test_clingen`, `test_core`, `test_dbsnp`, `test_eve`,
`test_hgmd`, `test_omim`, `test_reactome_wiring`, `test_rna_maxentscan_delta`,
`test_splice_ai_promotion`, `test_vep`). These were NOT patcher bugs. They are a
deliberate **Phase-4 contract**: a fixed, fully-promoted 81-column schema with
connector -> matrix wiring and safe defaults (populate-when-available), enforced by
`test_*_in_tabular_features`, `test_*_flows_into_feature_matrix`,
`test_phase_2_features_is_empty`, `test_new_features_in_tabular_features`, and
`test_reactome_is_last_feature_and_columns_match_tabular`. A constant column is a
*data-availability state*, not dead code; the connector-flow tests are themselves
silent-failure guards.

**Decision: REVERTED** via `git checkout --` on the three touched files, deleted the
`.bak` and the scratch patcher. Suite restored to the true baseline
**916 passed / 6 skipped / 0 failed** (the earlier "873/876" figures were a
half-applied trim, now corrected).

## 3. The neural variance mask -- the elegant fix that ships

**The only concern with teeth** was neural inputs: a constant column survives
`StandardScaler` as an all-zero input neuron. Rather than edit the shared schema,
handle it in the model layer, dynamically.

`TabularNNClassifier` now learns a fit-time keep-mask and applies it at predict,
mirroring the existing `self.scaler_` lifecycle:
- `fit`: `feature_mask_ = X.var(axis=0) > 0.0`; `n_features_in_ = X.shape[1]`; net
  built at `input_dim = feature_mask_.sum()`; all-constant fallback keeps all
  columns (never 0-width).
- `_apply_feature_mask`: pass-through when `feature_mask_` is absent (pre-mask
  pickles still score); raises on a column-count mismatch (never silently
  misaligns).
- `predict_proba` / `_predict_proba_single_pass` route through the helper.

**Scope.** `mc_dropout` and `deep_ensemble` clone-and-fit a `TabularNN`, so all
three tabular neural models are covered by the one change. `cnn_1d` consumes the
DNA sequence, not the tabular matrix, and is untouched. Trees / LR / CatBoost / KAN
are unaffected. **No** schema, `TABULAR_FEATURES`, `PHASE_2_FEATURES`, inference
contract, or schema-baseline change.

**Properties.** Threshold is exactly `var > 0` (only provably-zero-information
columns drop; `is_mitochondrial` at var ~5e-4 is kept). Dynamic: recomputed every
fit, so a column auto-rejoins the moment a connector populates it -- the
populate-when-available principle, in the model layer. No OOF leakage
(`cross_val_predict` recomputes per fold). Persistence is automatic (whole-pickle;
no custom `__getstate__` on `TabularNN`).

**No conflict with the zero-audit harness.** Confirmed by source:
`agent_layer/harness/correctness_harness.py::_stage5_zero_audit` inspects
`engineer_features(df)` output (the 81-column matrix) against `KNOWN_ZERO_DEFAULT`;
the mask is strictly downstream of that matrix and the neural estimators are not in
stage-1's smoke set.

**Validation.** Sandbox: patcher applies/compiles/idempotent; AST confirms helper +
masked fit/predict; helper numpy logic (pass-through/select/mismatch) proven.
Machine: Gate-A `8 passed`; Gate-B `924 passed / 6 skipped / 0 failed`, **zero new
warnings**. Origin content re-verified post-push (all 8 mask checks pass; no
leftover unmasked `transform(X)`).

**Commit.** `5de7806` "feat(tabularnn): fit-time variance mask drops constant
columns from neural input (covers mc_dropout/deep_ensemble); no schema change"
(+264/-2 over 4 files). Design note: `docs/design/neural_variance_mask.md`.

## 4. Audited residuals (none silent; all carried forward)

- **GNN tests skip locally.** `test_ablate_gnn` skips at collection on a
  `torch_scatter` / `torch_sparse` `0xc0000139` DLL-load fatal (caught by
  `importorskip`). GNN-ablation coverage is absent on this Windows box. **Matters
  for Run 17, which activates `gnn_score`** -- must be confirmed runnable on the
  vast.ai box (or the local env repaired) before that run is trusted.
- **220 warnings, all pre-existing, zero added by this change.** Dominant: pandas
  `.fillna` downcasting `FutureWarning` at `variant_ensemble.py`
  379/400/404/443/448/456/461/466/473/478 (the score/gtex/gene default loops) --
  will break on a future pandas; wants explicit cast / `.infer_objects(copy=False)`.
  Plus benign sklearn LGBM feature-name `UserWarning` and lbfgs `ConvergenceWarning`
  (tiny test data).
- **Untracked scratch in repo root** accumulating across sessions:
  `apply_meta_test_fix.py`, `apply_tabularnn_variance_mask.py`, `census_matrix.py`,
  `census_raw_features.py`, `check_meta_test_cols.py`, `missense_auroc.py`,
  `read_eval.py`, `xcheck_census.py`. Recommend clearing or gitignoring.
- **Cosmetic/data nits (not fixed this session):** the variance-mask CHANGELOG
  entry renders as an H3 nested under "## 2026-06-13 - Run 16 complete" (readable);
  the Run-16 CHANGELOG entry says "35/81 features constant" but the authoritative
  census is **37/81** (stale, pre-census).

## 5. Next steps (prioritized; none started)

1. **GNN local-test gap** (priority before Run 17): confirm `test_ablate_gnn` runs
   on the vast.ai box, or repair the local `torch_scatter`/`torch_sparse` ABI
   mismatch, so `gnn_score` activation in Run 17 is covered.
2. **pandas `.fillna` `FutureWarning` cleanup** in `variant_ensemble.py` (explicit
   cast / `.infer_objects(copy=False)` on the default-fill loops).
3. **`n_input_features` surfacing** -- record per-neural-model kept-feature count in
   the run report / metrics glossary (NOT a library log line, per the logging
   discipline). Needs the glossary location + surface decision; a small code change.
4. **clingen_validity_score dtype drift** (int in `real_data_prep` vs float in
   `variant_ensemble`) -- fix before the next regen (ROADMAP Section 5).
5. **Run 17 prep** (`RUN17_SCOPE.md`): activate `gnn_score` (`--string-db auto`) and
   `af_1kg_*` (`--kg-path`); the meta=meta_test eval fix (994d248) is already in.
6. **Scratch cleanup** + the two cosmetic CHANGELOG nits.
