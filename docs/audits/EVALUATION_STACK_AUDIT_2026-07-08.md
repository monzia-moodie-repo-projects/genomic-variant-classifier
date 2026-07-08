# EVALUATION STACK AUDIT — 2026-07-08

**Author:** Monzia Moodie
**Trigger:** commit `87e32ad` overwrote `evaluation/__init__.py` and `evaluation/metrics.py`
without reading them, on the strength of a standing note that the package was "UNBUILT".
Restored in `015ff94`. This document is the read-first that should have preceded both.

**Status of the belief that prompted the overwrite:** **FALSE.** The package is substantially
the 10-panel spec, already implemented, and has been since Run 9.

---

## 1. What actually exists

`src/genomic_variant_classifier/evaluation/` — 11 modules, 100 KB:

| module | bytes | contents |
|---|---:|---|
| `evaluator.py` | 24,009 | `ClinicalEvaluator`, `EvaluationReport`, `OperatingPoint`, `ConsequenceBreakdown`, `GeneErrorAnalysis`, `compare_models` |
| `prediction_artifacts.py` | 17,226 | `RunArtifactWriter` — manifest, OOF, test predictions, eval report, calibration, SHAP, permutation importance, graph stats, ablation aggregator |
| `metrics.py` | 13,104 | legacy `compute_classification_metrics`, `ModelEvaluator` + (2026-07-08) the appended primitive stack |
| `ntqr_evaluator.py` | 9,504 | NTQR unsupervised-evaluation bounds, designed to sit alongside `ClinicalEvaluator` |
| `benchmark.py` | 16,205 | (unread) |
| `model_insights_detector.py` | 5,255 | reads a run's `oof_predictions.parquet` in **RunArtifactWriter schema** |
| `agent_ops_detector.py`, `data_readiness_detector.py`, `finops_detector.py`, `model_introspect.py` | — | agent-layer detectors |

### 1.1 The 10-panel spec, mapped

| panel | already in `ClinicalEvaluator`? |
|---|---|
| AUROC + bootstrap CI | **yes** (`_bootstrap_ci`, `n_bootstrap=1000`) |
| AUPRC + bootstrap CI | **yes** |
| MCC, F1, Brier | **yes** (F1 at the same 0.5 threshold as MCC — PHASE5) |
| Calibration ECE / MCE | **yes** (`_calibration_error`) — but see §3.1 |
| Reliability curve | **yes** (`calibration_curve(strategy="quantile")`) |
| Operating point @ sensitivity ≥ 0.90 / ≥ 0.95 | **yes** (`_find_operating_point`) |
| Operating point @ PPV ≥ 0.80 | **yes** (`_find_high_ppv_point`) |
| Per-consequence breakdown | **yes** (`_consequence_breakdown`, LoF / missense / synonymous / splice / inframe) |
| Gene error analysis | **yes** (`_gene_error_analysis`, top-20) |
| Multi-model comparison | **yes** (`compare_models`) |
| Provenance (git sha, versions, config) | **yes** (`RunArtifactWriter.save_manifest`) |
| SHAP + permutation importance | **yes** (`save_shap_values`, `save_permutation_importance`) |
| NTQR bounds | **yes** (`ntqr_evaluator.py`) |

**Nothing on the 10-panel list was missing.** What was missing was the *use* of it (§2).

### 1.2 What the appended primitive stack genuinely adds

Only four things, and each addresses a specific defect found this session:

1. **`no_skill_auprc` / `auprc_lift`.** `EvaluationReport.prevalence` is stored but **AUPRC is never
   quoted against it**. Runs 9-14 ran at prevalence 20.34%; Runs 15-17 at 14.15%. An AUPRC of 0.985
   at prevalence 0.705 (the padded-deletion stratum) is a lift of 1.40, not a triumph.
2. **`calibration_slope_intercept`** (IRLS). `ClinicalEvaluator` reports ECE and MCE, which measure
   *magnitude* of miscalibration but not its *direction*. Slope < 1 = over-confident.
3. **`stratified_evaluate(y, score, groups)`** over an arbitrary grouping. `ClinicalEvaluator`
   stratifies by `consequence` only. We need strata by **variant representation**
   (padded_deletion / padded_insertion / delins / SNV) and by **review tier**.
4. **NaN, never 0.5.** `roc_auc_score` raises on a single-class stratum; the primitives return NaN
   and say so. `_consequence_breakdown` currently *silently drops* strata with `< 20` rows or one
   class (`continue`) — they vanish from the report rather than appearing as "not computable".

**Decision:** do NOT maintain a parallel metric module. The primitives stay as low-level, sklearn-
validated functions; `ClinicalEvaluator` should be extended to consume them and to carry
`auprc_no_skill`, `auprc_lift`, `calibration_slope`, `calibration_intercept`, and a generic
`stratified_breakdown`. That change touches a module with three existing test files
(`test_evaluator_meta.py`, `test_evaluator_phase5.py`, `test_core.py::TestClinicalEvaluator`),
which must be read first. **Read-first is now a hard precondition, not an aspiration.**

---

## 2. ROOT CAUSE: the audited writer exists and production does not use it

```python
# prediction_artifacts.py, save_oof_predictions
required = {"variant_id", "fold", "label"}
missing = required - set(oof_df.columns)
if missing:
    raise ValueError(f"oof_df missing required cols: {missing}")
```

`outputs/run14/full/oof_predictions.parquet` carries **only ten model columns** — no `variant_id`,
no `fold`, no `label`, and 1,017,633 rows against `X_train`'s 1,197,216.

Therefore **Run 14's OOF was not written by `RunArtifactWriter`.** It cannot have been: the writer
would have raised.

`prediction_artifacts.py`'s docstring says *"Integration example (see scripts/run_phase2_eval.py)"*,
yet a recursive grep finds `RunArtifactWriter` only in `scripts/run9_ablations.py` (lines 580, 695)
and **nowhere in `run_phase2_eval.py`** — the launcher every run since 15 has used.

**Consequences, all previously attributed to "missing provenance":**

* the OOF file has no join key, so per-stratum analysis of Run 14's predictions is impossible;
* no `manifest.json` (git sha, package versions, config) accompanies the run;
* `EvaluationReport.prevalence` — the AUPRC floor — is never written, because
  `save_eval_report` is never called;
* `model_insights_detector.py` reads "RunArtifactWriter schema" OOF files and therefore cannot
  read the ones production emits;
* `test_d1_d2.py:24` documents a fix — *"RunArtifactWriter: OOF row-index column"* — that
  production never benefits from.

**Verify (one command):**
```powershell
Select-String -Path scripts\run_phase2_eval.py -Pattern 'RunArtifactWriter|save_oof_predictions|manifest|ClinicalEvaluator|eval_report'
Get-ChildItem outputs,logs -Recurse -Include eval_report.json,manifest.json,test_predictions.parquet |
  Select-Object FullName, Length
```

If `eval_report.json` exists for any run, `prevalence` is in it and the regime table in
`INCIDENT_2026-07-08` can be filled from ground truth rather than recomputed.

---

## 3. Defects found by reading `evaluator.py`

### 3.1 `_calibration_error` silently drops `p == 1.0` (real bug)

```python
bin_edges = np.linspace(0, 1, n_bins + 1)
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (p >= lo) & (p < hi)      # last bin is [0.9, 1.0)
    ...
    bin_weight = mask.sum() / n      # but n counts the excluded rows
```

Predictions of exactly `1.0` fall into **no bin**. They are excluded from both ECE and MCE, while
the weight denominator still counts them. Tree ensembles emit `p == 1.0` routinely (a leaf that is
pure). **ECE is therefore under-reported.** The appended `expected_calibration_error` uses
`np.digitize(p, edges[1:-1], right=True)` and clips, so the top bin is closed.

Fix: `mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)`, or digitize+clip.

### 3.2 `_bootstrap_ci` is unstratified and silently discards resamples

```python
idx = self.rng.integers(0, n, n)
if len(np.unique(y[idx])) < 2:
    continue            # dropped, never counted
```

Two problems. First, an unstratified resample perturbs the class balance, so the CI mixes noise in
the score with noise in `pos_rate` — the very quantity that makes AUPRC comparable. Second, degenerate
resamples are dropped **without accounting**, so at low prevalence `scores` may hold far fewer than
`n_bootstrap` values and `np.percentile` is computed over an unknown sample size. At Run-14
prevalence (20%) this is harmless; at a rare stratum it is not.

Fix: stratified resampling (positives and negatives separately), and raise/report if the retained
count falls below a floor.

### 3.3 Two different binnings are reported side by side

`calibration_curve(..., strategy="quantile")` produces the reliability curve; `_calibration_error`
uses equal-width bins. Both land in the same report. They are not the same partition, and a reader
comparing the curve to the ECE is comparing two different objects. Choose one and say which.

### 3.4 `_consequence_breakdown` silently drops strata

```python
if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
    continue
```
A stratum that cannot be scored *disappears from the report*. It should appear with NaN metrics and
its `n`, so a reader can see that it existed and was not measurable. Guards on rows are not guards on
populations; the same principle applies to reports.

### 3.5 `_find_high_ppv_point` assumes PPV is monotone in the threshold

Walking thresholds high→low, it `break`s at the first `ppv < min_ppv`. PPV is not monotone; a more
permissive threshold further down may recover `ppv >= min_ppv`. The docstring says "the most
permissive threshold that never drops below min_ppv", which the `break` does implement — so this is
a **documented design choice, not a bug**. Recording it because it is invisible at the call site.

### 3.6 Library-level side effects

* `ClinicalEvaluator.evaluate()` calls `self.print_report(report)` unconditionally. A library
  function that prints cannot be used inside a loop or a test without capture.
* `compare_models(..., output_csv="models/model_comparison.csv")` writes a file by default.
* Neither guards `n == 0`: `prevalence=round(n_pos / n, 4)` and the `logger.info` divide by `n`.
* `roc_auc_score` / `average_precision_score` are called unguarded and raise on a single-class input.

### 3.7 `_find_operating_point` cost

`np.linspace(0, 1, 1000)` × full-array comparisons = O(1000·n). For Run 14's 349,067-row test split
that is ~3.5e8 boolean ops per operating point, three operating points per report, inside
`compare_models` per model. Correct, but the sorted-threshold sweep used by `_find_high_ppv_point`
is O(n log n) and would give exact thresholds rather than a 1/1000 grid.

---

## 4. What this changes about the incident record

* **`INCIDENT_2026-07-08` sec 6.2** says "no run artifact records which regime produced it."
  That is true *in practice* — but not because the machinery is missing. `EvaluationReport.prevalence`
  and `manifest.json` exist and are tested. **`run_phase2_eval.py` does not call them.** The finding
  is sharper and more fixable than stated: the audited writer was built, tested, documented as
  integrated, and then bypassed.

* The un-joinable OOF file is not an unexplained anomaly. It is what you get when a run writes its
  own parquet instead of calling `save_oof_predictions`, whose first act is to assert the join key
  exists.

---

## 5. Ordered actions

1. **Identify the 25th test failure.** `test_d1_d2::TestEvaluationPackageImport::test_package_importable`
   broke at `87e32ad` and the restore at `015ff94` did not visibly clear it. Two arithmetics fit
   `25 failed / 1608 passed`; only the FAILED list distinguishes them.
2. **Read** `tests/unit/test_d1_d2.py`, `test_evaluator_meta.py`, `test_evaluator_phase5.py`,
   `test_core.py::TestClinicalEvaluator`, `test_prediction_artifacts.py` before touching
   `evaluator.py`.
3. **Verify** whether `eval_report.json` / `manifest.json` exist for any run. If so, read
   `prevalence` per run and correct the regime table from ground truth.
4. **Wire `run_phase2_eval.py` to `RunArtifactWriter`.** This is the single change that closes the
   provenance gap for every future run, and it is a call, not a rewrite.
5. **Fix §3.1** (ECE drops `p == 1.0`) and **§3.2** (stratified bootstrap), with tests that fail
   before the fix.
6. **Extend `EvaluationReport`** with `auprc_no_skill`, `auprc_lift`, `calibration_slope`,
   `calibration_intercept`, and a generic `stratified_breakdown`; have `ClinicalEvaluator` delegate
   its primitives to the sklearn-validated functions in `metrics.py`. One source of truth.
7. Only then: `cohort-v2`, local smoke, VM smoke, launch.

---

## 6. The lesson, stated plainly

Three times in one session I wrote a file over an existing one I had not opened:
`tests/test_clean_cohort.py`, `evaluation/__init__.py`, `evaluation/metrics.py`. Each time the
justification was a *description* of the repository — a standing note, a summary, an inference from a
filename — rather than the repository itself.

The overwrite was caught by a `git commit` line reading `88 deletions` where a new file should have
reported none. Nothing in the repo announced it. The two regression guards added in `015ff94`
(`test_package_reexports_are_intact`, `test_legacy_metrics_api_is_preserved`) now assert the public
surface directly, and were verified by re-inflicting the overwrite and watching them fail.

**A package's public API is a contract. Until `015ff94`, nothing asserted it.**
