# Session record -- 2026-07-20 -- the metric kernel becomes fail-closed

Commits this session, oldest first:

| commit | subject | ratchet |
|---|---|---|
| `fb23543` | WindowAttachment derives its counts from two masks | 1985 -> 1999 |
| `106d107` | sequence-provenance gate | 1999 -> 2017 |
| `3bba87e` | session record: window attachment and the sequence gate | 2017 |
| `bd4d223` | roadmap delta: provenance, monitoring, JEPA, conformal | 2017 |
| `5615cd0` | the metric kernel becomes fail-closed | 2017 -> **2055** |

Continuous Integration #548 GREEN, 13m 43s, all seven jobs. `2048 passed, 7 skipped` in
652.78 seconds under `--assert-suite-size`.

---

## 1. What was measured, and what it overturned

### 1.1 The sequence gate accepts the production path

Roadmap 6.29a, written 2026-07-15, records `data/processed/clinvar_grch38_clean_seq.parquet`
as **"4,399,089 rows, 19 columns, NO `ok`"** and concludes that every Run 17 launcher points
at an artifact carrying no provenance, leaving the whole mechanism of commit 106d107 inert on
the configured path.

Measured 2026-07-20 with `pyarrow.parquet.ParquetFile(...).schema_arrow.names`: **21 columns,
including `ok` and `reason`**. The row count matches 6.29a exactly. The column count and the
provenance claim do not. The artifact gained those columns after the entry was written.

`probe_window_provenance_2026-07-20.py` then ran `attach_delta_windows` against both window
artifacts, keyed off the same 50,000 rows, reading thresholds live from `EnsembleConfig`:

| artifact | provenance | verified | usable | fraction | gate |
|---|---|---|---|---|---|
| `clinvar_grch38_clean_seq.parquet` (21 col) | `parquet+ok` | True | 49,970/50,000 | 0.999400 | **ACCEPT** |
| `seq_windows/seq_windows.parquet` (8 col) | `parquet+ok` | True | 49,970/50,000 | 0.999400 | **ACCEPT** |

Identical because `seq_windows.parquet` is a superset: 4,420,180 - 4,399,089 = **21,091**
alleleless rows absent from the clean cohort.

**A PREDICTION WAS MADE AND WAS WRONG.** Reading 6.29a, it was asserted that
`seq_require_verified_provenance=True` would cause the new gate to REFUSE the launcher path.
It accepts. The contradicting evidence was in this session's own `fb23543` work, which
measured that exact file resolving through the tier-1 `rows+ok` branch -- a branch that cannot
fire without an `ok` column. A document was quoted over a measurement taken the same day.

### 1.2 The base-model roster, derived rather than transcribed

`probe_roster_derivation_2026-07-20.py` constructed a plain `VariantEnsemble()` and read
`.base_estimators`, a dict of thirteen:

```
catboost CatBoostVariantClassifier | cnn_1d CNN1DClassifier (SEQUENCE_MODELS) |
deep_ensemble DeepEnsembleWrapper | gradient_boosting GradientBoostingClassifier (sklearn) |
kan KANClassifier | lightgbm LGBMClassifier (lightgbm) | logistic_regression Pipeline (sklearn) |
mc_dropout MCDropoutWrapper | random_forest RandomForestClassifier (sklearn) |
svm ScalableSVM | svm_bagged_rbf ScalableSVM | tabular_nn TabularNNClassifier |
xgboost XGBClassifier (xgboost)
```

`EXPECTED_TABULAR_FEATURE_COUNT` is 95 and agrees with `len(TABULAR_FEATURES)`.
`svm` and `svm_bagged_rbf` are the SAME class with different configuration -- a generated
table must render configuration, not just class, or it will read as a duplicated row.

**METHODS.md section 3.1 says "Four tabular base models were trained on the 64-feature
matrix"** and lists four with hyperparameters. Nine are absent: catboost, cnn_1d,
deep_ensemble, kan, logistic_regression, mc_dropout, svm, svm_bagged_rbf, tabular_nn. Line 152
additionally claims the sequence convolutional network is "excluded from the inference
pipeline", written before its 2026-07-05 Tier-1 re-architecture. This is NOT yet fixed; it is
commit 3 of the metric programme.

---

## 2. The metric kernel -- commit 5615cd0

An independent audit raised seven defects in `evaluation/metrics.py`. **All seven were
verified by reading the file before any code was written.** Six are repaired; the seventh is
deferred deliberately.

### 2.1 Defect A -- row misalignment between score and probability

`evaluate()` cleaned them with two separate masks:

```python
y_c, s_c = _clean(y, score)
p = s_c if prob is None else _clean(y, prob)[1]
```

`_clean(y, score)` drops rows where the score is non-finite; `_clean(y, prob)` drops rows where
the probability is non-finite. A non-finite score on row 1 and a non-finite probability on
row 2 produce arrays of the **same length describing different observations**, after which
every calibration metric pairs a probability with the wrong label -- silently, because the
downstream length check passes.

Repaired with one joint mask, exposed as `CleanArrays.mask` so no caller reconstructs it, plus
row accounting (`n_input`, `n_dropped`, `dropped_fraction`): a panel computed after silently
discarding a large fraction of the cohort is a different measurement from the one requested.

### 2.2 Defect B -- labels coerced, not validated

`_clean` ended `y[ok].astype(int)`. numpy truncates toward zero, so 0.9 became 0, 1.2 became 1
and 2.0 stayed 2.

**The audit understated this, and the first witness chosen for the test was the wrong one.**
Two distinct corruptions requiring different witnesses:

- `[0, 1, 2]` -- `y.sum() == 3 == y.size`, so `_degenerate` fires SPURIOUSLY and the old code
  returned NaN by accident. Wrong answer, right-looking reason.
- `[0, 1, 3]` -- `y.sum() == 4 != y.size`, so `_degenerate` stays quiet, and
  `(1 - y).sum() == -1` makes the denominator `n_pos * n_neg` NEGATIVE. **A signed AUROC.**

Labels are now rejected with the offending values named, never coerced. Booleans are accepted.

### 2.3 Defect D -- an empty probability vector was valid

`is_probability` returned True for size 0, so `calibration_valid=True` could be recorded for a
vector containing nothing finite. `tests/unit/test_evaluation_metrics.py:282` **asserted this**
-- the defect written down as a requirement, roadmap 6.21a's exact shape.

### 2.4 Defect E -- the calibration solver could not report nonconvergence

The iteratively-reweighted-least-squares loop fell out of `max_iter` and returned coefficients
regardless, so separation, quasi-separation and numerical instability were indistinguishable
from a clean fit.

Now returns `CalibrationFit(slope, intercept, converged, iterations, clipped_fraction)`,
**iterable** so every `slope, intercept = calibration_slope_intercept(...)` call site keeps
working while `.converged` becomes available.

`test_irls_does_not_overflow_on_extreme_logits` asserted finiteness on
`p = np.where(y == 1, 1 - 1e-9, 1e-9)` -- a **perfectly separated** input where the
maximum-likelihood slope does not exist. Its named purpose still holds and is still checked:
reaching the assertion means `np.errstate(over="raise")` raised nothing. What changed is the
second assertion, which conflated "did not overflow" with "produced a number". It now checks
both separately, **with a well-conditioned control** so the new assertion cannot be satisfied
by a solver that simply never converges.

### 2.5 Defect F -- rows with a missing subgroup label vanished

`stratified_evaluate` iterated `g.dropna().unique()`, so those rows counted in ALL and appeared
in NO stratum; the strata did not partition the cohort and nothing said so. Now a
`__MISSING__` stratum, with an assertion that per-stratum row counts sum to the input.
Missingness is informative here: a variant with no consequence annotation is a different object
from one annotated as missense.

### 2.6 Defect G -- subgroup sufficiency used only total n

A stratum of 1,000 rows containing ONE positive passed `min_n=30` and was given an area under
the precision-recall curve and a calibration slope. Positives and negatives now have their own
floors, and an insufficient stratum carries a `status` naming which floor it failed -- never
dropped, never given numbers.

### 2.7 The bootstrap ignored gene clustering -- found independently

`bootstrap_ci` resamples VARIANTS, stratified by class only. Variants within a gene share its
constraint, its network position, its curation history and often its true class. Treating them
as independent **understates variance**, so every confidence interval this project has
published is anti-conservative in a known direction.

`cluster_bootstrap_ci` resamples whole genes, optionally two-stage, and **reports the design
effect** -- the ratio of clustered to naive interval width -- so historical intervals can be
re-read rather than merely superseded. Measured end to end on a cohort where six of thirty
genes have inverted discrimination:

```
naive     [0.7548, 0.8439]   width 0.0891
clustered [0.6611, 0.9228]   width 0.2617
design effect 2.935x   (reported by the code, computed on UNROUNDED bounds)
```

The bounds above are printed to four decimal places; the design effect is computed inside
`cluster_bootstrap_ci` on the full-precision values. Recomputing from the printed widths gives
0.2617 / 0.0891 = 2.937, which is a rounding artefact and not a discrepancy. Stated here
because a reader checking the arithmetic would otherwise find a mismatch and have no way to
tell which number to distrust.

A control pins the effect near 1.0 when gene assignment is arbitrary, so the test cannot pass
on a fixture that merely adds noise.

### 2.8 Deferred deliberately -- defect C, the legacy application programming interface

`compute_classification_metrics` and `ModelEvaluator` are unchanged and the head of the file is
**byte-identical**, verified by hashing both sides of the banner. They are genuinely unsafe:
`confusion_matrix(...).ravel()` unpacks four values and raises on single-class input, and
specificity returns 0 where it is undefined, contradicting the new stack's own design rule.
But adding a deprecation changes what every existing caller emits, and that needs a measured
call-site census of its own. Commit 2.

### 2.9 Also added

`log_loss` -- unbounded, punishing CONFIDENT errors far harder than Brier, which is the failure
that costs most when a pathogenic variant is called benign at probability 0.99. And
`auprc_gain` -- the absolute gain, because the ratio explodes at low prevalence, where a lift
of 10 can be a gain of 0.009.

---

## 3. Three defects in the installers, caught before delivery, none by reading code

1. A post-check string-matched `g.dropna().unique()` and fired on the NEW docstring quoting it
   to explain the fix -- **the twenty-third occurrence this session** of a checker hitting prose
   describing its own rule, inside the installer remediating an audit about exactly this
   blindness. Replaced with an abstract-syntax-tree walk, tested 5/5 against a real call, a
   docstring, a comment, another function, and a call nested in a loop.
2. `evaluate()` rebuilt the keep-mask by hand for the cluster labels and ignored the labels'
   own finiteness -- it would have misaligned the clusters, which is **defect A recreated
   inside its own repair**. Fixed by exposing the mask on `CleanArrays`.
3. A test ended in `or True`, passing unconditionally. Removed; the replacement immediately
   found a bare acronym in prose, and fixing that found a second.

**The durable lesson is not "remember to parse."** It is that outcome-asserting checks catch
what careful reading does not: all three were found by a machine check written to assert a
RESULT, and both bootstrap-fixture errors were found by running the test rather than reviewing
it.

---

## 4. Retractions

- **The gate would refuse the launcher path.** Wrong; it accepts. Quoted a stale roadmap entry
  over a measurement from this session's own commit.
- **AlphaFold `[missing]` meant data loss.** Wrong; 107.1 MB exists at
  `data/external/alphafold/` (`alphafold_cohort.parquet` 110,220,322 bytes,
  `alphafold_coverage.json` 2,094,560 bytes, both 2026-07-03). `monitoring/registry.py:137-140`
  names the stale `.cif` cache path. Stale registry entry, not lost data.
- **`check_agents_active.py` was not doing its job.** Wrong; STALE is warning-only by design
  (docstring lines 25 and 37), the hard failures are ERRORED / DRY_RUN_ONLY / SECTION_ONLY /
  NEVER_RUN / UNSCHEDULED / MISSING_IMPL, and `--strict` exits 1.
- **A silent gate failure had let a performance figure back into the README.** Wrong. The
  performance-figure ban was **deleted deliberately on 2026-07-15**, and
  `test_readme_claims.py` says so at line 698. README lines 319-332 carry the Run 15 figure
  under "Early results" with the caveats. Roadmap 6.23, which records the figures as WITHDRAWN
  2026-07-14, is the stale document -- one day younger than line 41.
- **`push-ghcr` skipping at 0s was a possible silent failure.** It is release-gated by
  `if: github.event_name == 'release' && github.event.action == 'published'` (ci.yml:558).
  Worth checking rather than assuming, because ci.yml:174 records the 2026-07-14 incident where
  the identical symptom meant 1,936 tests never ran.

---

## 5. Open after this session

- **METHODS.md section 3.1** documents four base models against thirteen, and a 64-feature
  matrix against 95. `test_methods_feature_count.py` passes because it checks the count
  sentence, the group-table sum and HGMD's absence -- not the roster.
  `test_readme_claims.py:375` already reads the roster from a live ensemble; METHODS.md needs
  the same treatment.
- **The legacy metric application programming interface** is not yet unified.
  Needs a call-site census first.
- **METRIC_REGISTRY, typed report schema, clinical panels** -- operating points with threshold
  provenance, decision-curve net benefit, selective-prediction risk-at-coverage. Commit 2.
- **`--seq-windows` still means a directory in `train.py:102` and a file in
  `run_phase2_eval.py:49`** -- the surviving half of 6.29a.
- **ESM-2 coverage on the 4,399,089-row cohort is unmeasured.**
- **Monitoring remediation, three separable fixes** -- the stale AlphaFold registry path plus
  the detector's parent-directory and directory-total defects; an exit code that can express a
  finding; a scheduled run that can write telemetry.
- **Monthly Drift Monitor never dispatched** since its 2026-07-14 repair.
- **JEPA V1** blocked on storage: 10.91 GB free against a measured ~14.7 GB minimum.
