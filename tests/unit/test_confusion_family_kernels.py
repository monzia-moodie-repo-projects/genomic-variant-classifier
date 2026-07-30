"""The seven confusion-matrix kernels, checked numerically.

WHY THIS EXISTS
===============
These kernels were hand-verified against a fixture when they were written, and
that verification was never turned into a test. A sabotage matrix then found
three surviving mutations, all of them numerical and all of them dangerous:

    sensitivity returning 0.0 instead of NaN with no positive labels
    the positive likelihood ratio dividing by specificity, not 1 - specificity
    balanced accuracy substituting 0.0 for an undefined component

The second is the exact misstatement the kernel's own docstring warns against. A
warning in a docstring is not a check, and an interactive verification that is
not committed protects nothing.

THE FIXTURE
-----------
Ten observations, threshold 0.5, giving TP=3, FN=1, FP=1, TN=5. Every expected
value below is computed by hand from those counts, not read back from the
implementation:

    sensitivity   3/4      = 0.75
    specificity   5/6      = 0.833333...
    ppv           3/4      = 0.75
    npv           5/6      = 0.833333...
    balanced      (0.75 + 0.833333)/2 = 0.791666...
    LR+           0.75 / (1 - 0.833333) = 4.5
    LR-           (1 - 0.75) / 0.833333 = 0.3
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from genomic_variant_classifier.evaluation import metrics as M

Y = np.array([1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
P = np.array([.9, .8, .7, .3, .6, .2, .1, .1, .1, .1])

EXPECTED = {
    "sensitivity": 3 / 4,
    "specificity": 5 / 6,
    "positive_predictive_value": 3 / 4,
    "negative_predictive_value": 5 / 6,
    "balanced_accuracy": (3 / 4 + 5 / 6) / 2,
    "positive_likelihood_ratio": (3 / 4) / (1 - 5 / 6),
    "negative_likelihood_ratio": (1 - 3 / 4) / (5 / 6),
}


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_the_kernel_matches_the_hand_computed_value(name, expected):
    """Computed from TP=3, FN=1, FP=1, TN=5 by hand, not from the code."""
    observed = getattr(M, name)(Y, P, threshold=0.5)
    assert observed == pytest.approx(expected, rel=1e-12), (
        f"{name}: expected {expected!r} from the confusion counts, got {observed!r}")


def test_the_positive_likelihood_ratio_divides_by_one_minus_specificity():
    """THE MISSTATEMENT THIS PINS. It is sensitivity / (1 - specificity), NOT
    sensitivity / specificity -- a common error that a sabotage mutation
    reproduced and nothing caught.

    On this fixture the two differ by a factor of five, so the assertion
    discriminates rather than passing by coincidence.
    """
    sens, spec = 3 / 4, 5 / 6
    correct = sens / (1 - spec)          # 4.5
    wrong = sens / spec                  # 0.9
    assert not math.isclose(correct, wrong), "the fixture cannot tell them apart"

    observed = M.positive_likelihood_ratio(Y, P, threshold=0.5)
    assert observed == pytest.approx(correct, rel=1e-12)
    assert observed != pytest.approx(wrong, rel=1e-3)


def test_the_negative_likelihood_ratio_divides_by_specificity():
    """Its counterpart: (1 - sensitivity) / specificity. Also distinguishable
    on this fixture from the transposed form."""
    sens, spec = 3 / 4, 5 / 6
    correct = (1 - sens) / spec
    wrong = (1 - sens) / (1 - spec)
    assert not math.isclose(correct, wrong)
    assert M.negative_likelihood_ratio(Y, P, threshold=0.5) == pytest.approx(
        correct, rel=1e-12)


# --------------------------------------------------------------------------- #
# NaN, NEVER ZERO
#
# scikit-learn's zero_division policy reports 0.0 for an empty margin, which is
# indistinguishable from a classifier that was measured and scored nothing. An
# undefined quantity is not a measured zero, and this module refuses to conflate
# them.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,y,p", [
    ("sensitivity", np.zeros(10), np.full(10, .2)),                 # no positives
    ("specificity", np.ones(10), np.full(10, .8)),                  # no negatives
    ("positive_predictive_value", np.array([1., 0.] * 5), np.full(10, .1)),
    ("negative_predictive_value", np.array([1., 0.] * 5), np.full(10, .9)),
])
def test_an_empty_margin_yields_nan_not_zero(name, y, p):
    """A zero here would assert that every case was missed, or that nothing
    flagged was correct -- a measured failure rather than an absent measurement."""
    observed = getattr(M, name)(y, p, threshold=0.5)
    assert math.isnan(observed), (
        f"{name} returned {observed!r} where the margin is empty; 0.0 would be "
        "read as observed performance")


def test_balanced_accuracy_refuses_rather_than_substituting():
    """On an all-negative cohort sensitivity is undefined. Substituting zero for
    it would report balanced accuracy as 0.5 -- a plausible-looking number for a
    quantity that cannot be computed.

    Plain accuracy on the same cohort reports 1.000 and looks excellent, which is
    the reason this metric is in the specification at all.
    """
    y, p = np.zeros(10), np.full(10, .2)
    assert math.isnan(M.sensitivity(y, p, threshold=0.5))
    assert M.specificity(y, p, threshold=0.5) == pytest.approx(1.0)
    assert math.isnan(M.balanced_accuracy(y, p, threshold=0.5)), (
        "balanced accuracy substituted for its undefined component")


def test_a_perfectly_specific_classifier_has_an_unbounded_positive_ratio():
    """Specificity exactly 1.0 makes the ratio infinite. NaN is returned because
    infinity is not a value an evidence artifact can carry -- and because a
    reader seeing a very large finite number would take it as measured."""
    y = np.array([1.] * 5 + [0.] * 5)
    p = np.array([.9] * 5 + [.1] * 5)
    assert M.specificity(y, p, threshold=0.5) == pytest.approx(1.0)
    assert math.isnan(M.positive_likelihood_ratio(y, p, threshold=0.5))
    # its counterpart is finite and zero here: a negative result never errs
    assert M.negative_likelihood_ratio(y, p, threshold=0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_kernel_fails_closed_on_non_finite_probabilities(name):
    """The contract every other kernel in this module follows: a non-finite
    predicted probability is a MODEL-OUTPUT FAILURE, not a missing observation,
    and must not be silently dropped."""
    p = P.copy()
    p[0] = np.nan
    with pytest.raises(Exception):
        getattr(M, name)(Y, p, threshold=0.5)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_kernel_refuses_a_non_probability_input(name):
    """Values outside the unit interval are not probabilities, and a threshold
    sweep across [0, 1] would place every row on one side."""
    assert math.isnan(getattr(M, name)(Y, P * 3.0 - 1.0, threshold=0.5))
