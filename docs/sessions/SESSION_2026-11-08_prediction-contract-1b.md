# Session 2026-11-08 -- package 1 (the prediction contract), set 1b

Author: Monzia Moodie
Base: merged set 1a, b051bb3d (pull request #68; tree 6bc717f5). Rulings: decisions ccffb8cf and 38f15c7a of 2026-11-07.
Acceptance: docs/package1/PACKAGE1_SET1B_DELIVERY.md (gate by gate), docs/package1/PACKAGE1_SET1B_CENSUS_LEDGER.json.

## What set 1b does

- **Original out-of-fold (OOF) identity (G2).** `VariantEnsemble.fit` records, for every OOF row, its validation fold (from the
  exact splits all members use: `cross_val_predict` now receives them), its label, the number of rows `fit()` received and the fold
  scheme, and persists them. `evaluation/split_identity.py` gives each split a dataset identity (role, column names, binary64
  features, labels, variant identities) and its prediction units, in two unit contexts so an OOF prediction and a held-out
  prediction of one row never share an identity. `evaluation/oof_identity.py` composes training-split row -> optional subsample ->
  `oof_fit_indices_` -> OOF row and verifies the composed labels against the fitted labels, row by row; any mismatch or missing
  record refuses publication. The OOF table has the identity columns and one column per member, and no blend column (the stored
  weights were fitted on those rows).
- **The completion manifest (G8).** `RunArtifactWriter` measures every file's SHA-256 when it writes it; `complete()` re-measures
  them, adds registered files (models, splits), the selected predictor, the prediction-block digests, the OOF identity evidence and
  the deliverables, and writes `completion_manifest.json` last. A manifest left by an earlier run in the directory is retired
  (renamed) before anything is rewritten. `verify_completed_run` re-measures every listed byte; consumers (the model-insights
  detector and agent, the ablation aggregate) accept only completed runs.
- **Writers consume one bound capture per split.** `VariantEnsemble.score_split` calls every member once on a split and applies the
  declared blend once through `combiner.score_once`; `evaluate_scored` reduces that capture with the same metric code as
  `evaluate()`. Predictions, calibration points (bin edges and real counts, global and per consequence, equal to scikit-learn's on
  non-empty bins and checked against the evaluation report) and the blend's member-score attributions (phi = w (p - mu) against a
  declared background capture, with the closure residual against the published score) are reductions of the capture. Bare arrays
  and bare frames are refused.
- **Entrypoints.** `run9_ablations` is enabled (its set-1a refusal removed with the migration): a preflight before any fit, the
  gene-disjoint carve, one capture per split, no broad handler around requested deliverables, completion, an atomic aggregate of
  completed runs. `run_phase2_eval`: split identities before the subsample, one subsample of every row-aligned input, every output
  through the writer, OOF identity, completion; a failed requested GNN deliverable blocks completion. `save_shap_values` is retired.
- **Run 15's derived record (G2.3, G7.3).** `scripts/derive_run15_oof_identity.py` and
  `evaluation/historical_oof_identity.py`, through the new inert pickle reader (`inert_pickle.py`), wrote
  docs/package1/RUN15_OOF_IDENTITY_DERIVED_RECORD.json from the owner's files.

## Measured

- Run 15 (ensemble.joblib sha256 71f87e327186afe1099f542b01b9936316b8ef7e09f7053cb561aee3f6fb9929, read inertly in 0.18 s): the
  stored `oof_fit_indices_` (883,127 int64) equal `train_test_split(arange(1,038,974), test_size=0.15, stratify=y_train,
  random_state=42)` exactly (sha256 5cc3e4ba... both, scikit-learn 1.8.0 at fit and derivation); the OOF file's member columns equal
  the stored matrix bit for bit and its `_train_row_idx` is 0..883,126. AUROC per member through the mapping 0.85360 (cnn_1d) to
  0.99906 (xgboost, equal to its recorded checkpoint value 0.9990574667382814), joined naively as a training row 0.4978 to 0.4999.
  All 13 legacy `{name}_oof_indices.npy` sidecars are one file (sha256 acfb9c37...): the fold concatenation of StratifiedKFold(5,
  shuffle, 42); paired element-wise with the OOF vector they give AUROC 0.4987 to 0.4996. Fold sizes 176,626 x 2 and 176,625 x 3.
- Run 15's code: the master log names `HEAD: 4b9de1c`; that commit is absent from the rewritten history and present in the
  preserved bundle (Downloads/pre_rewrite_commits.bundle, sha256 c9b29cb6...), where variant_ensemble.py (blob 615590a4),
  run_phase2_eval.py (8ee0d60c) and real_data_prep.py (4daf65b9) equal the rewritten commit 9222a4b's. Its predict_proba returned
  `base_preds @ self.blend_weights_` whenever weights were stored; evaluate() labelled that ENSEMBLE_STACKER with calls at `p > 0.5`.
- The `--max-train` path of run_phase2_eval raised `AttributeError: 'WindowAttachment' object has no attribute 'iloc'` at the base
  (reproduced by executing the driver with the data preparation replaced); it completes after the fix.
- The selected predictor's scope digest is unchanged by moving its definition into models/combiner.py (66b6c895... before and
  after).
- The member-score attributions close within 1.1e-16 (8 units in the last place is the test bound) on the synthetic runs.
- Ten guards were mutated one at a time and each mutation was caught by its test (fold-concatenated sidecar, no label
  verification, no manifest retirement, no byte re-check at completion, the `.iloc` defect, unsubsampled metadata, no calibration
  agreement check, no background predictor check, a swallowed permutation failure, an inert reader that calls REDUCE). Two more
  after the first full run: the retirement counter removed (the second retirement then takes the first one's name), and the
  provenance test's git isolation removed (a hostile `prepare-commit-msg` hook, which `--no-verify` does not skip, then aborts its
  commit).
- Run 15's record re-derived by the installer's own route -- the bundle Downloads/pre_rewrite_commits.bundle (sha256
  c9b29cb6..., 52 preserved commits, verified) fetched into a NEW bare repository -- equals the committed record: 22 fields,
  every value exactly, the 39 AUROC values with 0 units in the last place of difference (the installer allows 4 across systems
  and prints the measured value). The comparison refused each of six corruptions (5 units in the last place, the commit, a fold
  size, an extra field, an integer written as a float, the provenance removed) and accepted 4 units in the last place.

## Findings recorded, not fixed (owner decision)

- **GNN-FEATURE-ORDER-1.** run_phase2_eval fits the ensemble (line "ensemble.fit") before the GNN blocks inject `gnn_score` and
  `hetero_gnn_score`. At fit time both are the feature default 0.5 on every row (measured with engineer_features and
  feature_census: both reported dead), so the driver's census aborts with exit 2 unless `--allow-dead-features`, and `fit` raises
  for 10,000 rows or more unless `allow_zero_variance_features`; no launcher passes either. With both overrides the members would be
  fitted on a constant and evaluated on injected scores. Run 15 predates both guards and had the same order. This blocks a full
  mainline run (Run 17) until the owner rules where the graph scores are computed and how their training rows avoid leakage.
- **MAXTRAIN-SPLIT-REPERSIST-1.** Under `--max-train` the GNN block re-persists the subsampled `X_train` into `splits/` beside the
  full labels; the ablation harness's preflight now refuses such a directory (row counts differ).

## Verification

Recorded in docs/CHANGELOG.md and with the delivery (the frozen candidate is run again on a clean clone of its tree).

## Errors, and their corrections

- The first preflight of the ablation harness refused a writable output directory: it probed the parent of a directory that did not
  exist yet. It now walks to the nearest existing ancestor; the end-to-end run found it before any test was written.
- A test of mine (the sequence-capability refusal) first built a probe whose declared blend was itself invalid, so it would have
  passed on the weight-order refusal; the probe is now a valid declared blend (asserted) and the test names the refusal it expects.
- I left a redundant environment rebuild running for an hour; the existing environment matched the Linux lock exactly (174 pins,
  0 differences), and the rebuild was stopped.
- Found while building the installer, after the first full run: (1) a retired manifest's name was its timestamp alone, and
  os.replace overwrites, so two retirements within one tick of a coarse clock (Python 3.12 on Windows: about 15.6 ms) would have
  destroyed the first -- the name now never repeats; (2) the derivation script decoded git's output in the locale's code page
  (cp1252 on Windows) -- it now decodes UTF-8 strictly everywhere (Run 15's files happen to decode either way; the record is
  byte-identical); (3) the provenance test isolated git with the null device's name, which I had not measured with Git for
  Windows -- it now uses an empty file of its own; (4) the installer would have required Run 15's commit in the owner's object
  store, which I had not verified -- it now verifies the bundle and fetches it into a new bare repository; (5) the installer
  template first expected 47 tests in the 1b file when 46 were collected, and my record comparison miscounted its fields
  (operator precedence).
