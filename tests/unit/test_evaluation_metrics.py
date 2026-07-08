"""
tests/unit/test_evaluation_metrics.py  (2026-07-08)
==========================================================================
Unit tests for genomic_variant_classifier.evaluation.metrics.

NEGATIVE TESTS AS MUCH AS POSITIVE. A metric that returns 0.5 on a single-class
population, or 0.0 when it cannot be computed, is worse than one that raises: it
launders a non-measurement as a measurement. Each function here is fed the input that
SHOULD make it say "I cannot compute this", and asserted to say so.

Run: python -m pytest tests/unit/test_evaluation_metrics.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.evaluation.metrics import (
    auroc, auprc, no_skill_auprc, brier_score, expected_calibration_error,
    calibration_slope_intercept, bootstrap_ci, evaluate, stratified_evaluate,
)


# --------------------------------------------------------------------------
# AUROC
# --------------------------------------------------------------------------
def test_auroc_perfect_inverted_and_constant():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert auroc(y, [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert auroc(y, [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)   # ties -> average rank


def test_auroc_single_class_is_nan_not_half():
    assert np.isnan(auroc([1, 1, 1], [0.1, 0.5, 0.9]))
    assert np.isnan(auroc([0, 0, 0], [0.1, 0.5, 0.9]))


def test_auroc_matches_mann_whitney_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500) + y * 0.8
    # brute force: P(score_pos > score_neg) + 0.5 P(equal)
    sp, sn = s[y == 1], s[y == 0]
    brute = (np.greater.outer(sp, sn).sum() + 0.5 * np.equal.outer(sp, sn).sum()) / (sp.size * sn.size)
    assert auroc(y, s) == pytest.approx(brute, abs=1e-12)


def test_auroc_ignores_nonfinite():
    assert auroc([0, 1, 1], [0.1, 0.9, np.nan]) == pytest.approx(auroc([0, 1], [0.1, 0.9]))


# --------------------------------------------------------------------------
# AUPRC and its no-skill floor -- the reason this module exists
# --------------------------------------------------------------------------
def test_auprc_perfect_is_one():
    assert auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auprc_of_constant_score_approaches_pos_rate():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.2).astype(int)
    ap = auprc(y, np.zeros_like(y, dtype=float))
    assert ap == pytest.approx(no_skill_auprc(y), abs=0.01)   # no-skill floor == pos_rate


def test_no_skill_auprc_is_the_positive_rate():
    assert no_skill_auprc([0, 0, 0, 1]) == 0.25


def test_auprc_single_class_is_nan():
    assert np.isnan(auprc([1, 1, 1], [0.1, 0.5, 0.9]))


def test_auprc_lift_reported_against_the_floor():
    y = np.array([0] * 80 + [1] * 20)
    s = np.concatenate([np.zeros(80), np.ones(20)])
    r = evaluate(y, s)
    assert r["pos_rate"] == pytest.approx(0.20)
    assert r["auprc"] == pytest.approx(1.0)
    assert r["auprc_lift"] == pytest.approx(5.0)   # 1.0 / 0.20


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def test_brier_bounds():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0


def test_ece_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(2)
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=10) < 0.01


def test_ece_large_for_grossly_miscalibrated():
    y = np.array([0] * 1000)
    p = np.full(1000, 0.9)          # always says 90%, always wrong
    assert expected_calibration_error(y, p) == pytest.approx(0.9, abs=1e-9)


def test_calibration_slope_one_intercept_zero_when_perfect():
    rng = np.random.default_rng(3)
    p = rng.random(60000) * 0.98 + 0.01
    y = (rng.random(60000) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.06)
    assert intercept == pytest.approx(0.0, abs=0.06)


def test_calibration_slope_below_one_when_overconfident():
    rng = np.random.default_rng(4)
    true_p = rng.random(60000) * 0.9 + 0.05
    y = (rng.random(60000) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p))
    over = 1 / (1 + np.exp(-2.0 * logit))          # doubled logit == over-confident
    slope, _ = calibration_slope_intercept(y, over)
    assert slope < 0.75, slope


def test_calibration_single_class_is_nan():
    s, i = calibration_slope_intercept([1, 1, 1], [0.2, 0.5, 0.9])
    assert np.isnan(s) and np.isnan(i)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 2000)
    s = rng.normal(size=2000) + y * 0.7
    point = auroc(y, s)
    lo, hi = bootstrap_ci(auroc, y, s, n_boot=200, seed=0)
    assert lo < point < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_stratified_preserves_positive_rate():
    rng = np.random.default_rng(6)
    y = (rng.random(5000) < 0.1).astype(int)      # heavy imbalance
    s = rng.normal(size=5000)
    lo, hi = bootstrap_ci(auroc, y, s, n_boot=100, seed=1, stratified=True)
    assert np.isfinite(lo) and np.isfinite(hi)    # never degenerates to one class


def test_bootstrap_single_class_is_nan():
    lo, hi = bootstrap_ci(auroc, [1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4], n_boot=10)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------
# Stratified panel -- the whole point
# --------------------------------------------------------------------------
def test_stratified_evaluate_separates_an_easy_stratum_from_a_hard_one():
    rng = np.random.default_rng(7)
    n = 4000
    grp = np.where(rng.random(n) < 0.2, "easy", "hard")
    y = rng.integers(0, 2, n)
    s = np.where(grp == "easy", y * 10.0 + rng.normal(0, 0.1, n), rng.normal(size=n))
    df = stratified_evaluate(y, s, grp)
    assert set(df.index) == {"ALL", "easy", "hard"}
    assert df.loc["easy", "auroc"] > 0.99
    assert df.loc["hard", "auroc"] == pytest.approx(0.5, abs=0.05)
    # the headline sits between the strata and describes neither
    assert df.loc["hard", "auroc"] < df.loc["ALL", "auroc"] < df.loc["easy", "auroc"]


def test_stratified_evaluate_reports_pos_rate_per_stratum():
    y = np.array([1] * 90 + [0] * 10 + [1] * 10 + [0] * 90)
    g = np.array(["a"] * 100 + ["b"] * 100)
    s = np.arange(200, dtype=float)
    df = stratified_evaluate(y, s, g)
    assert df.loc["a", "pos_rate"] == pytest.approx(0.9)
    assert df.loc["b", "pos_rate"] == pytest.approx(0.1)
    assert df.loc["ALL", "pos_rate"] == pytest.approx(0.5)


def test_stratified_evaluate_keeps_small_strata_with_nan_not_dropped():
    y = np.array([0, 1] * 50 + [0, 1])
    g = np.array(["big"] * 100 + ["tiny", "tiny"])
    s = np.arange(102, dtype=float)
    df = stratified_evaluate(y, s, g, min_n=30)
    assert "tiny" in df.index                       # never silently dropped
    assert df.loc["tiny", "n"] == 2
    assert np.isnan(df.loc["tiny", "auroc"])        # and never silently invented


def test_stratified_evaluate_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        stratified_evaluate([0, 1], [0.1, 0.9], ["a"])


# --------------------------------------------------------------------------
# REGRESSION GUARDS -- these two tests would have caught commit 87e32ad, which
# replaced evaluation/__init__.py with a stub and deleted the original metrics
# API. A package's public surface is a contract; nothing asserted it.
# --------------------------------------------------------------------------
def test_package_reexports_are_intact():
    """evaluation/__init__.py must keep re-exporting the clinical evaluator API."""
    import genomic_variant_classifier.evaluation as E
    for name in ("ClinicalEvaluator", "ConsequenceBreakdown", "EvaluationReport",
                 "GeneErrorAnalysis", "OperatingPoint", "compare_models",
                 "RunArtifactWriter"):
        assert hasattr(E, name), f"public API lost: {name}"


def test_legacy_metrics_api_is_preserved():
    """compute_classification_metrics / ModelEvaluator predate the metric stack."""
    from genomic_variant_classifier.evaluation.metrics import (
        compute_classification_metrics, ModelEvaluator,
    )
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.6 * y + 0.4 * rng.beta(2, 5, 300), 0.0, 1.0)
    m = compute_classification_metrics(y, (p >= 0.5).astype(int), p)
    assert set(m) == {"accuracy", "precision", "recall", "specificity", "f1", "auroc",
                      "auprc", "brier_score", "true_positives", "true_negatives",
                      "false_positives", "false_negatives"}
    assert "MODEL EVALUATION REPORT" in ModelEvaluator(y, p).generate_report()
    # the two APIs must agree on the same data
    assert m["auroc"] == pytest.approx(auroc(y, p), abs=1e-12)
    assert m["auprc"] == pytest.approx(auprc(y, p), abs=1e-12)


def test_package_imports_without_sklearn():
    """evaluation/__init__.py must not EAGERLY import sklearn.

    `evaluator.py` lazy-loads sklearn via `_ensure_sklearn()` and
    `prediction_artifacts.py` imports it inside functions, so the package imports
    cleanly in a minimal environment. `metrics.py` imports sklearn at module level.
    Commit 015ff94 added `from ... import metrics` to __init__.py and silently broke
    that contract, surfacing only as
    test_evaluator_phase5.py::test_module_imports_without_sklearn.

    Runs in a SUBPROCESS. Blocking sklearn by mutating this interpreter's
    builtins.__import__ pollutes module identity for every later test -- re-imported
    sklearn classes become new objects and unrelated pickle tests fail. See the note
    at the top of tests/unit/test_evaluator_phase5.py.
    """
    code = textwrap.dedent("""
        import builtins, importlib
        real = builtins.__import__
        def blk(n, *a, **k):
            if n == "sklearn" or n.startswith("sklearn."):
                raise ModuleNotFoundError("No module named 'sklearn' (blocked for test)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        m = importlib.import_module("genomic_variant_classifier.evaluation")
        assert hasattr(m, "ClinicalEvaluator"), "ClinicalEvaluator missing"
        assert hasattr(m, "RunArtifactWriter"), "RunArtifactWriter missing"
        print("PKG_IMPORT_OK")
    """)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert "PKG_IMPORT_OK" in r.stdout, (
        "evaluation/__init__.py must import without sklearn -- do not import metrics.py there.\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
