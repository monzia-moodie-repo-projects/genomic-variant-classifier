# Session 2026-11-02 -- CERTIFIED-AUPRC-DEGENERATE-REPLICATES-1 and RECALL-CURVE-FABRICATED-1

Author: Monzia Moodie

The 33 warnings every full run had reported since 2026-08-26 were recorded as expected
(tests/unit/test_warning_filters_are_scoped.py, GATE-WARNING-COUNT-UNSTABLE-1: "every warning in the gate comes from a third-party
library"). The owner's rule is that every warning is resolved. Traced to their causes, they were scikit-learn objecting, correctly,
to calls this project made on input where the quantity is undefined -- and two of those calls changed results.

## What was wrong, measured

1. The census. A measurement plugin recorded the project frames of every scikit-learn metric warning: all 33 come from three call
   sites -- evaluator.py's `roc_curve` and `precision_recall_curve` on single-class cohorts, and the gene-cluster bootstrap's metric
   (`metrics.cluster_bootstrap_ci`, reached through `bootstrap_metric` from the evaluator's certified intervals) on single-class
   replicates.
2. scikit-learn 1.8.0 on single-class input: `roc_auc_score` warns and returns NaN (it no longer raises, so the evaluator's
   `_nan_safe` docstring was stale); `average_precision_score` returns 0.0 on an all-negative input with a UserWarning and 1.0 on
   an all-positive input with NO warning.
3. CERTIFIED-AUPRC-DEGENERATE-REPLICATES-1. `cluster_bootstrap_ci` kept every replicate with a finite value; `bootstrap_metric`'s
   accounting replayed the draws in `_count_valid_replicates` and counted single-class replicates as degenerate. For average
   precision the two disagree: the zeros and ones of single-class replicates were inside the percentile set that the accounting
   said had excluded them. 40 genes, 2 positive genes, 1,000 replicates: reported [0.0000, 0.8333], status OK, certification
   eligible, "863 valid, 137 degenerate"; over the two-class replicates [0.4408, 0.8333]. Also measured: 30 genes / 3 positive
   [0.0000, 0.6835] against [0.2696, 0.6847]; 100 genes / 5 positive, lower endpoint 0.1617 against 0.2015; 40 genes / 20 positive
   unaffected. The design effect moved too: 1.002 reported where the two-class value is 0.656. The area under the receiver
   operating characteristic curve was unaffected (its single-class value is NaN, excluded by both loops). The report-layer shim
   (variant resampling, stratified) and `metrics.evaluate` (this module's own kernels, NaN on one class) were unaffected.
4. RECALL-CURVE-FABRICATED-1. On an all-negative cohort the report persisted recall = 1.0 at every threshold -- 0/0, scikit-learn's
   substitute -- as a usable curve, and the false-positive rate and precision as "usable arrays" without their partners;
   tests/unit/test_explicit_absence.py pinned that.
5. Found while reading: `_absence_maps` accepted `ranking_check` and never read it, so a refused `scores` array beside valid
   probabilities recorded the withheld ranking curves as UNDEFINED_ON_COHORT -- blaming the data; evaluate() built a second set of
   absence maps (with a different reason string) and never used them; metrics.py imported nine scikit-learn names it never used and
   its `__main__` block constructed `ModelEvaluator`, removed 2026-07-21 (NameError if run).
6. Observed, not changed: `bootstrap_metric` derives its tail probability as 1.0 - 0.95 = 0.050000000000000044, so its lower
   percentile is taken at 2.500000000000002 rather than 2.5 (the upper evaluates to exactly 97.5) -- a last-digit effect on the
   lower endpoint, present before and after.

## The change

- metrics.py: `_two_class_replicates`, the ONE replicate loop: a replicate is valid when it is non-empty, carries both classes and
  the metric answers finitely; every other replicate is degenerate and a single-class one never reaches the metric. The valid
  values and the degenerate count come from that loop, so an interval and its accounting are the same replicates by construction.
  `_row_bootstrap` and `_cluster_bootstrap` run it for the row and gene designs; `bootstrap_ci`, `cluster_bootstrap_ci` (two-stage
  included) and `bootstrap_metric` are callers; `_count_valid_replicates` is removed; one `_cluster_draw` serves both stages; on a
  single-class cohort the point estimate is not computed. The unused scikit-learn imports and the dead `__main__` block are
  removed; the module gains `from __future__ import annotations` and its logger.
- evaluator.py: `_CURVE_SUPPORT_REASON` / `_curve_supported` declare what each curve divides by; evaluate() computes a curve only
  where the cohort has those members, and `_absence_maps` records the cause from the same counts
  ("roc_curve_requires_both_classes:positives=0,negatives=200", ...). Each curve takes its cause from its own channel (ranking
  curves: labels then the ranking channel; calibration: labels then probabilities). A supported curve over validated input that
  comes back unusable raises. The duplicate absence maps and duplicate imports are removed; `_nan_safe`'s docstring corrected.
- pyproject.toml: `error::UserWarning:sklearn\\.metrics\\._ranking` -- every warning of that module (five sites in 1.8.0, read)
  reports a quantity undefined or meaningless on its input; selected by module so configuration never imports scikit-learn.

## Verification

- Battery (scratch, kept as evidence): 144 cases by four interval kinds against the pre-change kernel. Only scikit-learn's average
  precision moved (26 interval parts), each equal bit for bit to an independent two-class oracle; every other kernel and interval is
  byte-identical.
- Warning census with warnings as errors on the 10 affected test files: 0 of the 33 remain, all traced to the three call sites.
- Mutation: 11 of 11 mutants killed (8 in the first pass; the 3 survivors each got a test, then 3 of 3 killed).
- ci.yml's exact `pytest tests/` command, each interpreter in its own environment built by scripts/install_locked_profile.py:
  Python 3.11.17, 8,601 passed, 26 skipped, 6 xfailed, 0 failed, NO warnings, coverage 84%, 1,044.83 s; Python 3.12.3, 8,593
  passed, 26 skipped, 6 xfailed, 0 warnings, coverage 84%, and 8 failed -- this sandbox's /usr/bin/git 2.43 lacks
  --no-lazy-fetch; the 86 tests that run git, those 8 among them, pass with git 2.47.1. 8,633 collected both times; the
  working-tree guard reported the tree unchanged.
- ci.yml's `pytest tests_drift/`, in its own drift environment: 19 passed on Python 3.11.17 and 19 on 3.12.3, no warnings.

## Open, recorded

- Persisted reports: affected exactly when the interval for the area under the precision-recall curve is available and records at
  least one degenerate replicate (those replicates' values were in its percentiles, under scikit-learn 1.8.0's behaviour). A
  read-only census of the owner's reports is prepared; not yet run. Checked on reports written by save_report: the verdicts are
  right, but a report written by the corrected code that records degenerate replicates looks the same, so each census row carries
  the file's modification time in UTC.
- REPORT-PROVENANCE-ABSENT-1 (new): an evaluation report records neither the commit nor the library versions that wrote it
  (measured on a schema-version-5 report from save_report), so no report can be attributed to the code that produced it.
- SILENT-FALLBACKS-1 now also names `_nan_safe`: it converts a ValueError on a two-class replicate into NaN, which the engine counts
  degenerate.
