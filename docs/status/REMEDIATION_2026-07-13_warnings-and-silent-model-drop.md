# Remediation, 2026-07-13 — the warnings were not noise

**Author:** development session, 2026-07-13
**Preceded by:** `REMEDIATION_2026-07-11_test-suite-red.md` (24 red tests → 0)
**Commits:** `7d42409` (silent dropout), `2026-07-13 warnings` (this change)
**Outcome:** test suite **1852 passed, 8 skipped, 0 failed, 0 warnings** — the first
completely clean run in the project's history. Two latent correctness defects in the
pathogenicity classifier were found and fixed; one dangerous "improvement" was
investigated and **rejected on evidence**.

---

## 1. Summary

On 2026-07-12 the test suite was green for the first time (1819 passed) but still emitted
**29 warnings** in every run, and had done so for weeks — through every continuous-integration
run, on Python 3.11 and 3.12 alike. They were widely treated as noise.

They were not noise. Investigating them found:

| Finding | Nature |
|---|---|
| **A base model could be silently erased from the ensemble** (§3) | Real defect. Corrupts the algorithm comparison this project exists to produce. |
| **LightGBM silently returns wrong predictions on mis-ordered columns** (§4) | Real, previously unknown library hazard. Makes an existing line of code load-bearing. |
| **The Nystrom map dimension was clamped against the wrong denominator** (§5) | Real defect. Invisible at production scale. |
| The LightGBM feature-name warning itself | Genuinely spurious — but the tripwire that exposed the other two. |

**The meta-lesson, stated plainly: every warning in this project turned out to be either a
real bug or the visible edge of one.** Not one was safely ignorable. Three of them had been
printing, in plain sight, in every run for weeks.

---

## 2. How the investigation went wrong three times (recorded deliberately)

This section exists because the failures of method are more reusable than the fixes.

The LightGBM warning read:

> `X does not have valid feature names, but LGBMClassifier was fitted with feature names`

From that text, three successive claims were made **without instrumenting anything**:

1. *"LightGBM is silently wrong if column order ever drifts in production."*
   **False.** `VariantEnsemble` dispatches LightGBM a raw ndarray in `fit`,
   `_leakfree_oof`, `predict_proba` **and** `evaluate` — consistently, in every direction.
   No names are passed anywhere, so no mismatch is possible.

2. *"The warning indicates a defect in our code."*
   **False.** A four-call spy showed `LGBMClassifier.fit` receiving `ndarray / names=False`
   on all four calls (three cross-validation folds plus the final fit) — while the warning
   fired four times anyway. The premise of the claim was contradicted by direct measurement.

3. *"The constant-0.5 column poisons the stacking meta-learner."*
   **False.** `variant_ensemble.py` line 1892 (`valid_cols`) drops the column of any model
   absent from `trained_models_` before the meta-learner is fitted. That code was always
   correct. The claim was made without reading forty lines further down the same function.

Each error has the same shape: **a symptom was read, a severity was narrated, and the code
was not opened.** That is precisely the failure mode this project's standing instructions
exist to prevent, and it was committed three times in a row against a single warning.

What broke the cycle was refusing to act until a probe produced numbers. The probes are
reproduced in §4 and §5 and are now permanent tests.

**Two proposed "fixes" were stopped by measuring blast radius first, and both would have
caused real damage:**

- **Scaling `cnn_1d`** (proposed 2026-07-12). It consumes the **one-hot DNA encoding**
  (values in {0, 1}), not the tabular matrix. A `StandardScaler` would have destroyed the
  encoding.
- **Passing DataFrames to LightGBM** (proposed 2026-07-13, to "clean up" the `.values`
  dispatch). This would have armed the silent-corruption bug documented in §4 — a
  0.855-probability error, in the model that classifies variant pathogenicity, with no
  error and no warning.

Both were caught only because the instruction was to investigate before changing.

---

## 3. Defect A — a base model could vanish from the ensemble in silence

### What the code did

`VariantEnsemble.fit`, before this change:

```python
except Exception as exc:
    logger.error("  %s OOF failed: %s - skipping.", name, exc)
    oof_preds[:, model_idx] = 0.5
    continue                      # <-- also skips model.fit() on the next line
```

Any exception from a base model's out-of-fold step was swallowed. The `continue` skipped the
`model.fit(X_input_fit, y_fit)` immediately below it. Consequences, all silent:

* the model was **never fitted**, and **no checkpoint was written** for it;
* it was absent from `trained_models_`, from `oof_model_names_`, from the Nelder-Mead blend,
  and from **every downstream comparison artifact**;
* a 13-model ensemble quietly became a 12-model ensemble;
* the twelve survivors reported **entirely normal metrics**, so the run looked healthy;
* the only trace was a single `logger.error` line in a multi-hour training log.

### Why it matters specifically to *this* project

A first-class goal of this classifier is to **measure and compare the performance of every
machine-learning algorithm** applied to the data. A silently dropped algorithm does not
appear in the report as a *failure*. It appears as an algorithm that **was never a
candidate** — indistinguishable, in the artifacts, from one that was never configured. The
comparison is then wrong in a way no reader can detect.

### What triggered it

Running the suite under `-W error::UserWarning` escalated the **spurious** LightGBM warning
(§6) into an exception. The out-of-fold step "failed"; LightGBM was dropped;
`test_per_model_checkpoints_written` failed because `models/lightgbm.joblib` did not exist.

**Noise was sufficient to delete a model from a paid training run.** The `except Exception`
was broad enough to swallow an out-of-memory error, a transient data fault, or nothing at
all.

### Scope, stated honestly

With default warning filters, the full 1825-test suite triggers this handler **zero times**.
It was a loaded gun that had not yet gone off — *not* a defect that has been silently
corrupting completed runs. There is no evidence any historical run lost a model. That
distinction matters and is not being blurred.

### The fix (`7d42409`)

* New `EnsembleConfig.allow_base_model_dropout: bool = False`.
* The handler now **raises `RuntimeError`**, chaining the original exception (`raise ... from
  exc`), naming the model, carrying the underlying cause, and telling the operator exactly
  what to do.
* Opt-in dropout still exists — but is **loud**: it logs at ERROR with a full traceback and
  records the model **and its cause** in a new `VariantEnsemble.dropped_models_` dict, so the
  incompleteness survives in the run artifacts rather than scrolling past in a log.
* A total wipeout (every model failing) now raises a comprehensible error instead of handing
  an `(n, 0)` matrix to the meta-learner.
* An `ENSEMBLE IS INCOMPLETE: N of M base models were DROPPED` message at ERROR, emitted at
  the last point in `fit` where the fact is still visible.

### Tests — `tests/unit/test_base_model_dropout_is_loud.py` (6)

Including a **meta-test that pins the test fixture itself**. The first version of the
saboteur (`_ExplodingModel`) was a bare duck-typed object; `cross_val_predict` rejected it at
parameter validation *before* the out-of-fold pass began. The ensemble still raised — so the
tests would still have gone green — but they would have been testing scikit-learn's argument
validation rather than a base model failing mid-training. **A test whose fixture fails for
the wrong reason is a gate that reports PASS while guarding nothing.** It was caught only
because the assertion demanded the *specific* underlying cause (`"synthetic OOF failure" in
msg`) rather than merely `pytest.raises(RuntimeError)`.

> **Assert the value, never the shape.** The same lesson as `KNOWN_ZERO_DEFAULT` (27 vs 25)
> and the pytest floor (1485 vs 1815).

---

## 4. Defect B — LightGBM does not enforce its own feature names

This is the most consequential finding of the session, and it was previously unknown.

### The measurement (2026-07-13)

Fit each library on a DataFrame, then predict with **the same data in a different column
order**:

| Library | Result |
|---|---|
| scikit-learn (`RandomForestClassifier`) | `ValueError` — **refuses** |
| XGBoost (`XGBClassifier`) | `ValueError` — **refuses** |
| CatBoost (`CatBoostClassifier`) | reorders by name, identical output — **corrects** |
| **LightGBM (`LGBMClassifier`)** | **SILENTLY WRONG — max delta 0.855 in predicted probability** |

LightGBM accepted the mis-ordered DataFrame, **mapped the columns positionally**, and
returned confident, wrong probabilities — with no error and no warning, **even under
`warnings.simplefilter("error")`**. Its `feature_names_in_` attribute is *decorative*: it is
populated, it is reported, and it is never enforced.

**LightGBM is the sole outlier in the roster.**

### What this means for the code

`VariantEnsemble.fit` uses a three-way dispatch:

```python
if name == "cnn_1d":
    X_input_fit = X_seq_fit          # one-hot DNA sequence
elif name == "catboost":
    X_input_fit = X_tab_fit          # DataFrame — needed for categorical resolution
else:
    X_input_fit = X_tab_fit.values   # raw ndarray — column names DISCARDED
```

That `.values` looks like a wart. **It is load-bearing.** By discarding names it makes column
order positional *by construction*, guaranteed upstream by `engineer_features` — the single
source of truth since 2026-07-11 — which emits `TABULAR_FEATURES` in a fixed order behind the
`EXPECTED_TABULAR_FEATURE_COUNT` fail-loud guard. Passing DataFrames instead would replace
that hard guarantee with a name-based check that **LightGBM silently does not honour.**

CatBoost's DataFrame dispatch is, by the same measurement, **safe** — it reorders by name.

### The fix (2026-07-13)

No production code change was needed — the existing dispatch was already correct. What was
missing was any record of *why*, which is what made it fragile. Added
`tests/unit/test_feature_name_contract.py` (7 tests), which:

* records the measured column-order behaviour of **all four** libraries;
* asserts the dispatch still hands LightGBM an ndarray, read out of the source rather than
  trusted from a comment;
* acts as a **library-upgrade tripwire**: if a future LightGBM release fixes positional
  mapping, `test_lightgbm_does_NOT_enforce_column_order` **fails on purpose** and reports
  that the constraint can be revisited.

> A fact in a comment is a comment. A fact that fails a test is a gate — and a fact that
> **re-derives itself on every run** cannot go stale.

---

## 5. Defect C — the Nystrom map dimension was clamped against the wrong denominator

### What the code did

`ScalableSVM._build_headline`, before this change:

```python
d = int(min(self.n_components, max(n_samples - 1, 1)))   # clamped to the FULL training set
...
if self.calibrate:
    return CalibratedClassifierCV(base, cv=self.calibration_cv)
```

`calibrate=True` and `calibration_cv=3` are the **defaults**, so every headline SVM fit takes
this path. `CalibratedClassifierCV` **refits the entire pipeline — `StandardScaler → Nystroem
→ LinearSVC` — on each cross-validation training fold**, which is strictly smaller than
`n_samples`.

`Nystroem` selects `n_components` **rows** of the data it is fitted on as landmarks, so it
requires `n_components ≤ (rows fitted on)`. When violated, scikit-learn does not raise — it
**silently reduces `n_components`** and warns.

The clamp was therefore **off by the cross-validation factor**.

### The measurement (2026-07-13), n = 100, `n_components` = 1024, old clamp giving d = 99

```
calibrate=True,  calibration_cv=3   ->  3 warnings   (one per fold, ~66 rows each)
calibrate=True,  calibration_cv=5   ->  5 warnings   (one per fold, ~80 rows each)
calibrate=False (no CV refit)       ->  0 warnings
```

**One warning per fold, exactly.** That is the signature of a per-fold refit, and it is
conclusive.

### Why it survived for weeks

**At production scale the clamp never binds.** With n ≈ 1.7 × 10⁶ and D = 1024,
`min(1024, n − 1) = 1024`, and each fold (≈ 1.1 × 10⁶ rows) dwarfs it. **The defect is
invisible at the scale the model is actually trained at**, and surfaces only on the small
fixtures the tests use — where 18 warnings per run were dismissed as "test-scale noise"
rather than read.

This is the same reflex that let a **non-converged logistic regression** feed the stacking
meta-learner for weeks (see `tests/unit/test_logistic_regression_is_scaled.py`): a warning is
not a failure, so nobody looked.

### The fix (2026-07-13)

* `_rows_the_map_is_fitted_on(n)` → `n − ceil(n / k)`, the smallest `StratifiedKFold` training
  fold. The correct denominator.
* `_map_dim(n)` → the **single source of truth** for the map dimension.
* The `fit` log line contained a **second, hand-kept copy of the clamp formula**
  (`min(self.n_components, X.shape[0] - 1)`). Left alone, it would have kept reporting the
  *old* D while the model trained with the *new* one — a log that lies. Both now call
  `_map_dim()`. **This is failure pattern (a) from ROADMAP §7 — caught in the act of being
  created.**
* The class docstring said *"Capped at n_samples"* — the bug, written down as the contract.
  Corrected.

### Tests — `tests/unit/test_scalable_svm_map_dim.py` (20)

The central test does **not** ask "does it warn?". It derives the smallest training fold
**empirically from `StratifiedKFold` itself**, across `n ∈ {40, 60, 100, 250} × cv ∈ {2, 3,
5}`, and asserts the closed form never exceeds it. **A formula asserted against itself proves
nothing.** Also pinned: production scale (D must remain exactly 1024 at n = 1.7 × 10⁶ — the
clamp must not bind where it should not) and a zero-width-map floor.

**Side effect:** the suite runs **486 s → 430 s**. Nystroem is no longer silently rebuilding
a full kernel on every calibration fold.

---

## 6. The spurious warning (for completeness)

LightGBM 4.6.0 populates `feature_names_in_` with **synthetic** names — `Column_0`,
`Column_1`, … — even when fitted on a bare numpy ndarray with no names supplied. scikit-learn
leaves the attribute **unset** in the same situation. At predict time, scikit-learn's
`_check_feature_names` sees an estimator that "was fitted with feature names" being handed a
nameless array, and warns.

Verified directly:

```python
>>> m = lgb.LGBMClassifier().fit(numpy_array, y)   # no names supplied
>>> m.feature_names_in_
array(['Column_0', 'Column_1', 'Column_2'], dtype=object)
```

That asymmetry — and nothing else — produced all 11 warnings.

**Suppressed in `pyproject.toml`, pinned to that exact message string.** Never a blanket
`ignore::UserWarning`: a different `UserWarning` must still reach us. The suppression is safe
**only because** LightGBM is always handed an ndarray, and that premise is now gated by the
tests in §4. Verified after the change that the 11 LightGBM warnings disappeared **and the 18
`n_components` warnings remained visible** — a filter that silenced both would have recreated
the very condition that let these defects survive.

---

## 7. Final state

| | 2026-07-08 | 2026-07-12 | **2026-07-13** |
|---|---|---|---|
| Test suite | 24 red, unnoticed | 1819 passed | **1852 passed, 0 failed** |
| **Warnings** | 41 | 29 | **0** |
| Suite runtime | — | 486 s | **430 s** |
| Continuous Integration | RED, merged past | GREEN | GREEN |
| Silent model dropout | possible, undetectable | possible, undetectable | **impossible by default** |
| LightGBM column-order hazard | unknown | unknown | **measured, gated, documented** |
| Nystrom map dimension | wrong clamp | wrong clamp | **correct, 20 tests** |

**Every warning in the project is now either fixed or suppressed with a written, measured
justification and a test pinning its premise.**

---

## 8. What remains open

Unchanged from `ROADMAP.md` §6, and none of it is closed by this work:

* **6.2 — the largest unlit area.** Continuous Integration runs `tests/unit/` **only**.
  `tests/conformal/` (7 files), `tests/integration/` (1), and 22 root-level `tests/test_*.py`
  — **30 test files** — have never run in continuous integration. Widening will likely go
  RED. **Do it on a branch.**
* **6.5** — correctness-harness stage-3 sanity model convergence.
* **6.7** — `±inf` raises a raw pandas `IntCastingNaNError`.
* **6.9** — the three `conftest.py` autouse guards have no permanent self-test.
* **6.10** — open pull request #1 (`run9a-prep`); repository-authority ambiguity.
* **6.11** — Joint-Embedding Predictive Architecture (JEPA): not started.
* **6.12** — disk free 14.7 GB, below the 20 GB the G1 pre-flight gate recommends.

**Run 17 remains planned and gated, NOT launched.** Run 15 (commit `032a2ab`) is still the
last sealed run.
