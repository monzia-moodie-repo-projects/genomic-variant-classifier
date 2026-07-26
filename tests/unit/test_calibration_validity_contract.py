"""The calibration-validity contract in evaluation/metrics.py.

WHY THIS FILE EXISTS
====================
`calibration_valid` gated on `is_probability(p)` alone until 2026-07-21 -- that
is, on whether the values lay in [0, 1]. It never asked whether a calibration
claim was SUPPORTABLE. Measured on y = [1,1,1,1], p = [.9,.8,.85,.95]:

    auroc              NaN     correct: ranking is undefined with one class
    auprc              NaN
    cal_slope          NaN
    cal_intercept      NaN
    brier              0.01875     <- a number
    ece                0.125       <- a number
    calibration_valid  True        <- asserting those numbers are sound

The ECE of 0.125 is just 1 - 0.875, the gap between the mean prediction and the
only label present. It carries no information about calibration across the
probability range, because the reliability diagram has one occupied row.

Worse, the flag's own documented invariant -- "False => brier/log_loss/ece/cal_*
are NaN by design" -- was already broken in the other direction: cal_slope and
cal_intercept were NaN while the flag read True, so a consumer would read an
undefined estimand as a failed computation.

TWO REPAIRS, AND WHY THEY DIFFER IN SEVERITY

Both classes present is a HARD requirement, because without it the quantity
being computed is not calibration.

Thin support is REPORTED, not refused. DEFAULT_MIN_POS and DEFAULT_MIN_NEG are
the same floors `stratified_evaluate` already applies per subgroup, so identical
data was being called insufficient as a stratum and sound on its own. But
refusing what the predecessor accepted is exactly the regression that broke this
suite earlier the same day on a 427-row cohort, so thin cohorts still get
numbers -- with `calibration_support` saying so.

A bare boolean cannot separate "these are not probabilities" from "this cohort
has one class" from "computed, but on three positives". Those demand different
responses, so the reason is machine-readable.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.metrics import (
    DEFAULT_MIN_NEG,
    DEFAULT_MIN_POS,
    evaluate,
)

CAL_METRICS = ("brier", "log_loss", "ece", "cal_slope", "cal_intercept")


def _ev(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return evaluate(y, p, prob=p)


# --------------------------------------------------------------------------- #
# 1. a single-class cohort cannot support a calibration claim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("y,p,expected", [
    ([1, 1, 1, 1], [.9, .8, .85, .95], "single_class:pos=4,neg=0"),
    ([0, 0, 0, 0], [.1, .2, .15, .05], "single_class:pos=0,neg=4"),
])
def test_a_single_class_cohort_is_not_calibration_valid(y, p, expected):
    out = _ev(y, p)
    assert out["calibration_valid"] is False
    assert out["calibration_support"] == expected


def test_the_single_class_ece_that_used_to_be_reported_is_now_withheld():
    """0.125 was reported for [1,1,1,1] against mean prediction 0.875. It is
    1 - 0.875 and nothing more."""
    assert _ev([1, 1, 1, 1], [.9, .8, .85, .95])["ece"] != _ev(
        [1, 1, 1, 1], [.9, .8, .85, .95])["ece"]  # NaN != NaN


# --------------------------------------------------------------------------- #
# 2. the invariant, enforced rather than merely documented
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("y,p", [
    ([1, 1, 1, 1], [.9, .8, .85, .95]),
    ([0, 0, 0, 0], [.1, .2, .15, .05]),
    ([0, 1, 0, 1, 1, 0], [-2., 3.1, -1.4, 2.2, .7, -.3]),
])
@pytest.mark.parametrize("metric", CAL_METRICS)
def test_invalid_calibration_means_every_calibration_metric_is_nan(y, p, metric):
    """The comment promised this; the code did not deliver it. brier_score and
    the ECE estimator self-guard only on is_probability, so a single-class
    cohort sailed through both."""
    out = _ev(y, p)
    assert out["calibration_valid"] is False
    assert out[metric] != out[metric], f"{metric} = {out[metric]}, expected NaN"


def test_ranking_metrics_keep_their_own_guard():
    """The calibration gate must not reach across and suppress AUROC/AUPRC,
    which have a separate and correct degeneracy check."""
    out = _ev([1, 0, 1, 0, 1, 0], [.9, .1, .8, .2, .7, .3])
    assert out["calibration_valid"] is True
    assert np.isfinite(out["auroc"]) and np.isfinite(out["auprc"])


# --------------------------------------------------------------------------- #
# 3. thin support is reported, not refused
# --------------------------------------------------------------------------- #
def test_thin_support_still_produces_numbers():
    """Refusing what the predecessor accepted is a regression. A 427-row cohort
    broke this suite that way earlier the same day."""
    out = _ev([1, 0, 1, 0, 1, 0], [.9, .1, .8, .2, .7, .3])
    assert out["calibration_valid"] is True
    assert np.isfinite(out["brier"])
    assert out["calibration_support"].startswith("thin:")


def test_thin_support_names_the_floors_it_fell_short_of():
    out = _ev([1, 0, 1, 0, 1, 0], [.9, .1, .8, .2, .7, .3])
    assert f"min {DEFAULT_MIN_POS}" in out["calibration_support"]
    assert f"min {DEFAULT_MIN_NEG}" in out["calibration_support"]


def test_sufficient_support_says_so_plainly():
    y = [1] * 12 + [0] * 12
    p = [.9] * 12 + [.1] * 12
    assert _ev(y, p)["calibration_support"] == "sufficient"


def test_the_thin_boundary_is_exactly_the_stratified_floor():
    """One below the floor is thin; exactly the floor is sufficient. Pinned
    because the two code paths must not drift apart -- the whole point is that
    evaluate() and stratified_evaluate now apply the same policy."""
    n = DEFAULT_MIN_POS
    below = _ev([1] * (n - 1) + [0] * n, [.9] * (n - 1) + [.1] * n)
    at = _ev([1] * n + [0] * n, [.9] * n + [.1] * n)
    assert below["calibration_support"].startswith("thin:")
    assert at["calibration_support"] == "sufficient"


# --------------------------------------------------------------------------- #
# 4. non-probabilities
# --------------------------------------------------------------------------- #
def test_logits_are_rejected_with_their_own_reason():
    out = _ev([0, 1, 0, 1, 1, 0], [-2., 3.1, -1.4, 2.2, .7, -.3])
    assert out["calibration_valid"] is False
    assert out["calibration_support"] == "not_probabilities"


# --------------------------------------------------------------------------- #
# 5. the field is always present
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("y,p", [
    ([1, 1, 1, 1], [.9, .8, .85, .95]),
    ([1, 0, 1, 0, 1, 0], [.9, .1, .8, .2, .7, .3]),
    ([1] * 12 + [0] * 12, [.9] * 12 + [.1] * 12),
    ([], []),
])
def test_calibration_support_is_always_reported(y, p):
    """A consumer must never have to infer WHY from the absence of a key."""
    out = _ev(y, p) if len(y) else evaluate(np.array([]), np.array([]),
                                            prob=np.array([]))
    assert "calibration_support" in out
    assert isinstance(out["calibration_support"], str)
    assert out["calibration_support"]


def test_an_empty_cohort_reports_insufficient_rows():
    out = evaluate(np.array([]), np.array([]), prob=np.array([]))
    assert out["calibration_valid"] is False
    assert out["calibration_support"] == "insufficient_rows"
