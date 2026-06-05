# INCIDENT 2026-06-04 — KAN OOF failed (`name 'test_size' is not defined`) in Run 15

## Status
DIAGNOSED. Root cause confirmed against live source. Fix designed, NOT yet
implemented. KAN is **absent from both Run 15 ensembles** (main + unseen-gene
holdout). Non-fatal: the run completed `TRAIN_OK`, but the model-comparison
goal has a hole (KAN is one of the two newer architectures under study).

## Symptoms
- Main ensemble, 2026-06-04 13:35:58:
  `kan OOF failed: name 'test_size' is not defined — skipping.`
- UGH ensemble, 2026-06-04 17:42:06: identical error.
- Consequence: `kan` produces no OOF, is dropped from `trained_models_`, and
  does not appear in `per_model_metrics.csv` for either ensemble.

## Root cause
Two-part, and only one part was present on the Run 15 VM:

1. **Package bug (imodelsx KAN backend).** The installed imodelsx KAN fit path
   references a bare `test_size` that is not defined in its scope, raising
   `NameError: name 'test_size' is not defined` at fit time (inside
   `cross_val_predict` in `VariantEnsemble.fit`).

2. **The patch that fixes it was never applied on this VM.** The known
   remediation lives in `scripts/launch_run11_vm.sh:193-194`:
   ```bash
   if [ -n "$IMODELSX_KAN" ] && grep -q "test_size=test_size" "$IMODELSX_KAN"; then
       sed -i 's/test_size=test_size/test_size=self.test_size/g' "$IMODELSX_KAN"
   fi
   ```
   `scripts/launch_run15_baseline.sh` deliberately removed launch_run11's
   "brittle source-greps" — but in doing so it **also dropped this required
   imodelsx patch step**. The baseline launcher has no `$IMODELSX_KAN`
   resolution and no `sed`, so imodelsx was left unpatched and KAN tripped the
   bug on first fit.

The instance-attribute half of the historical fix (`self._imodelsx_model.test_size`
/ `random_state` / `shuffle` set before `.fit()`, per `Run14_Preflight.ps1` and
`Run_Preflight_Local.ps1`, validated by `agent_layer/harness/correctness_harness.py`
PM13 check) addresses `self.test_size` existing; the `sed` rewrites the package
call site to reference it. **Both** are needed; the launcher provided neither.

## Why it was not caught before the run
The pre-launch mini-test used `--skip-kan` (along with `--skip-svm/--skip-cnn/--skip-nn`)
and the preflight validated infrastructure (files, key, GPU, import), so KAN's
fit path was never exercised before ~4 h of GPU time was spent. This is the
exact gap the new ALL-MODELS smoke-gate standing law closes (see CHANGELOG
2026-06-04).

## Why it was non-fatal (and correctly handled downstream)
`VariantEnsemble.fit` wraps each model's `cross_val_predict` in
`except Exception`, logs `"%s OOF failed: %s — skipping."` at **ERROR** level,
sets that column's OOF to 0.5, `continue`s, and never adds the model to
`trained_models_`; `valid_cols` then drops the column before the meta-learner
and Nelder-Mead blend. So the blend is computed only over the models that
succeeded — KAN's failure does not corrupt the ensemble, it only removes KAN
as a data point.

## Fix (to implement before the re-run; behind the smoke gate)
1. Port the imodelsx patch into `launch_run15_baseline.sh` setup: resolve the
   installed imodelsx KAN source path (`$IMODELSX_KAN`) and apply the
   `sed 's/test_size=test_size/test_size=self.test_size/g'` (idempotent,
   guarded by the `grep -q` as in launch_run11).
2. Confirm `KANClassifier._fit_imodelsx` (src/.../models/kan.py) sets
   `test_size`/`random_state`/`shuffle` on the instance **before** `.fit()`
   (PM13 correctness-harness invariant).
3. Add `./model/` to `.gitignore` (imodelsx KAN writes a `./model/` dir in CWD).
4. Verify via the ALL-MODELS smoke gate: a tiny `--max-train ~3000` run with no
   `--skip` flags must log a finite, non-degenerate `kan OOF AUROC` and contain
   zero `OOF failed`/`skipping` lines.

## Source needed to finalize the exact patch
`src/genomic_variant_classifier/models/kan.py` and the head of
`scripts/launch_run11_vm.sh` (the `$IMODELSX_KAN` path resolution).

## References
- `scripts/launch_run11_vm.sh:193-194` (the working sed patch)
- `scripts/Run14_Preflight.ps1:71,81,94`, `scripts/Run_Preflight_Local.ps1:51-62`
  (instance-attr injection)
- `src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py`
  (PM13 `test_size` injection check)
- `scripts/run_phase2_eval.py` (`--skip-kan` help text: "do not hardcode
  removal again")
