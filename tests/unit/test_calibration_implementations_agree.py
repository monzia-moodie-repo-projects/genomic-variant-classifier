"""Every calibration-error implementation in this repository, pinned to one definition.

WHY THIS FILE EXISTS
--------------------
A census on 2026-07-20 found TEN independent implementations of expected calibration error
across scripts/ and src/. Evaluating nine of them on identical fixtures found two distinct
defects in six:

  OPEN TOP BIN -- `(p >= lo) & (p < hi)` with `hi == 1.0` drops every prediction of exactly
  1.0, a pure decision-tree or ensemble leaf. Under-reports by 86.7% on a fixture with 20%
  of rows at 1.0. Found in calibrate_thresholds.py, validate_external.py,
  calibration_analysis.py; previously found and repaired in evaluator.py on 2026-07-10.

  COUNTS MISALIGNED WITH BINS -- `calibration_curve` returns only non-empty bins while
  `np.histogram` returns all of them, so zipping the two attaches each weight to the wrong
  bin. Correct whenever every bin is occupied, which is why it survived review. Under-reports
  by 2x on a saturated fixture and 64x on a sparse saturated one. Found in run_benchmark.py,
  validate_clinvar_temporal.py, benchmark.py.

Reading these functions did not catch either defect; evaluating them on a fixture designed to
separate the definitions did. These tests are that fixture, kept.

The expectations are DERIVED from a reference implemented here, at each implementation's own
default bin count -- never hardcoded -- so a change of bin count is not mistaken for a defect
and a change of definition cannot pass by editing a constant.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reference_ece(y, p, n_bins: int, closed_top: bool = True) -> float:
    """Occupancy-weighted |accuracy - confidence|, top bin closed by default."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if closed_top and b == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        total += (mask.sum() / y.size) * abs(y[mask].mean() - p[mask].mean())
    return float(total)


@pytest.fixture(scope="module")
def pure_leaf():
    """20% of rows at exactly p == 1.0, and half of those are negative.

    An implementation with an open top bin cannot see that error at all.
    """
    rng = np.random.default_rng(20260720)
    p = np.concatenate([np.full(1000, 1.0), rng.uniform(0.05, 0.95, 4000)])
    y_leaf = np.zeros(1000, dtype=int)
    y_leaf[:500] = 1
    y = np.concatenate([y_leaf, rng.binomial(1, p[1000:])])
    return y, p


@pytest.fixture(scope="module")
def saturated():
    """Only 0.0 and 1.0. Thirteen of fifteen bins are empty."""
    p = np.concatenate([np.ones(1000), np.zeros(1000)])
    y = np.concatenate([np.ones(750), np.zeros(250),
                        np.ones(250), np.zeros(750)]).astype(int)
    return y, p


@pytest.fixture(scope="module")
def sparse_saturated():
    """Saturated AND sparse: nine empty bins. The misalignment defect is worst here."""
    rng = np.random.default_rng(20260720)
    p = np.concatenate([np.full(600, 1.0),
                        rng.uniform(0.0, 0.2, 700),
                        rng.uniform(0.8, 1.0, 700)])
    y = np.concatenate([np.zeros(600, dtype=int),
                        rng.binomial(1, p[600:1300]),
                        rng.binomial(1, p[1300:])])
    return y, p


def test_the_fixture_actually_separates_the_two_definitions(pure_leaf):
    """A control. If closed-top and open-top agreed here, the other tests would prove nothing."""
    y, p = pure_leaf
    closed = _reference_ece(y, p, n_bins=10, closed_top=True)
    open_ = _reference_ece(y, p, n_bins=10, closed_top=False)
    assert closed > open_ * 5, (
        "the pure-leaf fixture must separate the definitions; "
        "closed={} open={}".format(closed, open_)
    )


def test_kernel_counts_predictions_of_exactly_one(pure_leaf):
    from genomic_variant_classifier.evaluation.metrics import expected_calibration_error

    y, p = pure_leaf
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(
        _reference_ece(y, p, n_bins=10), abs=1e-9)


def test_evaluator_counts_predictions_of_exactly_one(pure_leaf):
    """Repaired 2026-07-10. Pinned so it cannot regress."""
    from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator

    y, p = pure_leaf
    ece, _mce = ClinicalEvaluator._calibration_error(None, y, p, n_bins=10)
    assert ece == pytest.approx(_reference_ece(y, p, n_bins=10), abs=1e-9)


def test_benchmark_aligns_counts_with_the_bins_they_weight(saturated, sparse_saturated):
    """Repaired 2026-07-20. Both fixtures leave bins empty, which is when it bit."""
    from genomic_variant_classifier.evaluation.benchmark import _ece

    for y, p in (saturated, sparse_saturated):
        assert _ece(y, p, n_bins=15) == pytest.approx(
            _reference_ece(y, p, n_bins=15), abs=1e-9)


def test_benchmark_refuses_rather_than_misweighting(saturated):
    """The repair fails loud. It must not silently fall back to a truncated zip."""
    import inspect

    from genomic_variant_classifier.evaluation import benchmark

    source = inspect.getsource(benchmark._ece)
    assert "counts_all" in source, "the aligned selection was removed"
    assert "raise ValueError" in source, (
        "the bin/count length guard was removed; a future mismatch would again be silent")
