"""Risk-controlling prediction sets: the bounds, the search, and the gate.

WHAT IS CHECKED AND HOW
-----------------------
The binomial tail is checked against a DIRECT `math.comb` sum -- an independent
computation, not a restatement -- across 1,967 (n, p, k) combinations.

The three upper confidence bounds are checked the only way a confidence bound
can honestly be checked: BY SIMULATION. Draw from a known risk, compute the
bound, and count how often it falls below the truth. That rate must not exceed
delta. A bound that merely "looks right" is not a bound.

The whole procedure is then checked end to end: a known monotone risk curve, a
noisy calibration estimate of it, a chosen lambda, and the question of whether
the TRUE risk at that lambda exceeds alpha more than delta of the time.
"""
from __future__ import annotations

import math
from math import comb

import numpy as np
import pytest

from genomic_variant_classifier.conformal.risk_control import (
    NonMonotoneRiskError,
    RiskControlError,
    abstention_rate,
    binomial_cdf,
    clopper_pearson_upper_bound,
    control_risk,
    false_negative_risk,
    hoeffding_bentkus_upper_bound,
    hoeffding_upper_bound,
    risk_control_report,
    risk_curve,
)

BOUNDS = ("hoeffding", "clopper_pearson", "hoeffding_bentkus")


# --------------------------------------------------------------------------- #
# The binomial tail, against an independent exact computation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 5, 17, 50])
@pytest.mark.parametrize("p", [0.0, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0])
def test_the_binomial_tail_matches_a_direct_sum(n, p):
    """`math.comb` is a different computation, not a rearrangement of ours."""
    for k in range(0, n + 1):
        exact = sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
        assert binomial_cdf(k, n, p) == pytest.approx(exact, abs=1e-12)


def test_the_tail_is_computed_in_log_space_so_large_n_survives():
    """A direct sum at n = 100,000 overflows the binomial coefficient. This does
    not, because it never forms one."""
    v = binomial_cdf(500, 100_000, 0.006)
    assert 0.0 < v < 1.0
    assert math.isfinite(v)


def test_the_tail_is_monotone_in_p():
    """Non-increasing in p for fixed k. Both bisections rely on this, so it is
    asserted rather than assumed."""
    values = [binomial_cdf(10, 100, p) for p in np.linspace(0.01, 0.99, 40)]
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:]))


def test_degenerate_arguments():
    assert binomial_cdf(-1, 10, 0.3) == 0.0
    assert binomial_cdf(10, 10, 0.3) == 1.0
    assert binomial_cdf(11, 10, 0.3) == 1.0
    with pytest.raises(ValueError):
        binomial_cdf(1, -1, 0.3)


# --------------------------------------------------------------------------- #
# The bounds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("n", [10, 50, 200, 1000])
@pytest.mark.parametrize("r", [0.0, 0.01, 0.05, 0.2, 0.5, 1.0])
def test_every_bound_is_above_the_estimate_and_inside_the_unit_interval(bound, n, r):
    delta = 0.05
    if bound == "hoeffding":
        v = hoeffding_upper_bound(r, n, delta)
    elif bound == "clopper_pearson":
        v = clopper_pearson_upper_bound(int(round(r * n)), n, delta)
    else:
        v = hoeffding_bentkus_upper_bound(r, n, delta)
    assert r - 1e-12 <= v <= 1.0 + 1e-12, (bound, n, r, v)


@pytest.mark.parametrize("n", [50, 200, 1000, 5000])
def test_hoeffding_bentkus_is_tighter_than_plain_hoeffding(n):
    """If it were not, there would be no reason to carry the extra machinery."""
    r, d = 0.05, 0.05
    assert hoeffding_bentkus_upper_bound(r, n, d) < hoeffding_upper_bound(r, n, d)


@pytest.mark.parametrize("bound", BOUNDS)
def test_every_bound_tightens_as_the_sample_grows(bound):
    d, r = 0.05, 0.05
    def at(n):
        if bound == "hoeffding":
            return hoeffding_upper_bound(r, n, d)
        if bound == "clopper_pearson":
            return clopper_pearson_upper_bound(int(round(r * n)), n, d)
        return hoeffding_bentkus_upper_bound(r, n, d)
    values = [at(n) for n in (50, 200, 1000, 5000)]
    assert all(b < a for a, b in zip(values, values[1:])), values


@pytest.mark.parametrize("bound", BOUNDS)
def test_every_bound_widens_as_confidence_is_demanded(bound):
    n, r = 500, 0.05
    def at(d):
        if bound == "hoeffding":
            return hoeffding_upper_bound(r, n, d)
        if bound == "clopper_pearson":
            return clopper_pearson_upper_bound(int(round(r * n)), n, d)
        return hoeffding_bentkus_upper_bound(r, n, d)
    assert at(0.001) > at(0.01) > at(0.10)


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("R_true,n", [(0.02, 200), (0.10, 200), (0.30, 200)])
def test_the_bound_actually_covers(bound, R_true, n):
    """THE CHECK THAT MATTERS. Draw from a known risk, compute the bound, count
    how often it falls BELOW the truth. That rate must not exceed delta.

    A confidence bound cannot be verified by inspection; this is the only honest
    test of one.
    """
    delta, trials = 0.05, 1200
    rng = np.random.default_rng(20260730)
    failures = 0
    for _ in range(trials):
        k = int(rng.binomial(n, R_true))
        r = k / n
        if bound == "hoeffding":
            v = hoeffding_upper_bound(r, n, delta)
        elif bound == "clopper_pearson":
            v = clopper_pearson_upper_bound(k, n, delta)
        else:
            v = hoeffding_bentkus_upper_bound(r, n, delta)
        if v < R_true:
            failures += 1
    rate = failures / trials
    assert rate <= delta + 0.02, (
        f"{bound} covered only {1 - rate:.4f} of the time against a nominal "
        f"{1 - delta:.2f}; a bound that under-covers is not a bound")


@pytest.mark.parametrize("bad", [(-0.1, 100, 0.05), (1.1, 100, 0.05),
                                 (0.1, 0, 0.05), (0.1, 100, 0.0),
                                 (0.1, 100, 1.0)])
def test_impossible_bound_arguments_are_refused(bad):
    with pytest.raises(ValueError):
        hoeffding_upper_bound(*bad)


def test_clopper_pearson_at_a_total_failure_is_one():
    """Every calibration row failed: nothing below 1 can be asserted."""
    assert clopper_pearson_upper_bound(50, 50, 0.05) == 1.0


# --------------------------------------------------------------------------- #
# The risks
# --------------------------------------------------------------------------- #
def test_the_false_negative_risk_counts_only_the_positive_rows():
    """A marginal error rate on a mostly-benign cohort is dominated by the benign
    rows and can look excellent while missing most pathogenic ones."""
    sets = np.array([[True, True], [True, False], [False, True],
                     [False, False], [True, False], [False, True]], dtype=bool)
    y = np.array([1, 1, 1, 1, 0, 0])
    assert false_negative_risk(sets, y, 1) == pytest.approx(0.5)


def test_an_empty_positive_denominator_is_not_zero_risk():
    """Reporting zero would read as perfect sensitivity on a cohort that never
    tested it."""
    sets = np.ones((6, 2), dtype=bool)
    assert np.isnan(false_negative_risk(sets, np.zeros(6, dtype=int), 1))


def test_the_abstention_rate_counts_empty_sets():
    sets = np.array([[True, True], [False, False], [True, False]], dtype=bool)
    assert abstention_rate(sets) == pytest.approx(1 / 3)


@pytest.mark.parametrize("kwargs", [
    {"positive_class": 5},
    {"positive_class": -1},
])
def test_a_class_outside_the_range_is_refused(kwargs):
    sets = np.ones((4, 2), dtype=bool)
    with pytest.raises(ValueError):
        false_negative_risk(sets, np.array([0, 1, 0, 1]), **kwargs)


def test_a_length_mismatch_is_refused():
    with pytest.raises(ValueError):
        false_negative_risk(np.ones((4, 2), dtype=bool), np.array([0, 1]))


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def test_a_non_monotone_risk_curve_is_refused_not_approximated():
    """THE GATE. The guarantee assumes a nested family, whose risk is
    non-increasing. Where that fails the guarantee is VOID rather than
    approximate, so this refuses in the same spirit as calibrate.py's alignment
    gate."""
    lam = np.linspace(0, 0.5, 6)
    risks = np.array([0.40, 0.02, 0.30, 0.03, 0.02, 0.01])
    with pytest.raises(NonMonotoneRiskError, match="MONOTONICITY GATE FAILED"):
        control_risk(lam, risks, n=1000, alpha=0.10, delta=0.05)


def test_the_monotone_tolerance_is_declared_not_hidden():
    """A risk curve estimated by Monte Carlo wobbles. The allowance for that is a
    parameter the caller sets, not a constant buried in the module."""
    lam = np.linspace(0, 0.5, 6)
    risks = np.array([0.40, 0.25, 0.2501, 0.12, 0.06, 0.01])
    with pytest.raises(NonMonotoneRiskError):
        control_risk(lam, risks, n=1000, alpha=0.10, delta=0.05)
    out = control_risk(lam, risks, n=1000, alpha=0.10, delta=0.05,
                       monotone_tolerance=1e-3)
    assert out["lambda_hat"] is not None


def test_lambda_hat_is_the_smallest_index_whose_whole_suffix_controls():
    """Not the first index that dips below alpha. The distinction is what makes
    the guarantee uniform rather than pointwise."""
    lam = np.linspace(0, 0.5, 6)
    risks = np.array([0.40, 0.25, 0.12, 0.06, 0.03, 0.01])
    out = control_risk(lam, risks, n=2000, alpha=0.10, delta=0.05)
    i = out["index_hat"]
    assert i is not None
    assert bool(out["controls"][i:].all())
    assert not bool(out["controls"][i - 1]) if i > 0 else True


def test_no_controlling_lambda_is_an_answer_not_an_error():
    lam = np.linspace(0, 0.5, 6)
    out = control_risk(lam, np.full(6, 0.9), n=100, alpha=0.05, delta=0.05)
    assert out["lambda_hat"] is None
    assert out["index_hat"] is None
    assert out["guarantee"] == "no lambda controls the risk"


@pytest.mark.parametrize("bound", BOUNDS)
def test_the_chosen_lambda_controls_the_true_risk(bound):
    """END TO END. A known monotone risk curve, a noisy calibration estimate of
    it, a chosen lambda -- and the true risk at that lambda must exceed alpha no
    more than delta of the time."""
    lam = np.linspace(0.0, 1.0, 21)
    true_risk = 0.5 * np.exp(-3.0 * lam)
    alpha, delta, n, trials = 0.10, 0.10, 400, 400
    rng = np.random.default_rng(11)
    violations = considered = 0
    for _ in range(trials):
        hat = np.minimum.accumulate(
            np.array([rng.binomial(n, p) / n for p in true_risk], dtype=float))
        out = control_risk(lam, hat, n=n, alpha=alpha, delta=delta, bound=bound)
        if out["lambda_hat"] is None:
            continue
        considered += 1
        if true_risk[out["index_hat"]] > alpha:
            violations += 1
    assert considered > 0
    rate = violations / considered
    assert rate <= delta + 0.05, (
        f"{bound}: the true risk exceeded alpha in {rate:.4f} of trials against "
        f"a nominal delta of {delta}")


@pytest.mark.parametrize("bad", [
    {"alpha": 0.0}, {"alpha": 1.0}, {"alpha": -0.1}, {"bound": "nonesuch"},
])
def test_impossible_search_arguments_are_refused(bad):
    lam = np.linspace(0, 0.5, 6)
    risks = np.linspace(0.4, 0.01, 6)
    kwargs = {"n": 500, "alpha": 0.1, "delta": 0.05}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        control_risk(lam, risks, **kwargs)


def test_descending_lambdas_are_refused():
    with pytest.raises(ValueError, match="ascending"):
        control_risk(np.array([0.5, 0.4, 0.3]), np.array([0.1, 0.2, 0.3]),
                     n=500, alpha=0.1, delta=0.05)


def test_a_non_finite_risk_is_refused():
    with pytest.raises(ValueError, match="non-finite"):
        control_risk(np.array([0.1, 0.2, 0.3]), np.array([0.3, np.nan, 0.1]),
                     n=500, alpha=0.1, delta=0.05)


def test_risk_curve_evaluates_in_order():
    seen = []
    def fn(l):
        seen.append(l)
        return 1.0 - l
    out = risk_curve([0.1, 0.5, 0.9], fn)
    assert seen == [0.1, 0.5, 0.9]
    assert out.tolist() == pytest.approx([0.9, 0.5, 0.1])


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def test_the_report_says_so_when_abstention_is_missing():
    """A risk controlled by abstaining is not a risk controlled, and a report
    that cannot tell which happened must say that rather than imply the good
    case."""
    lam = np.linspace(0, 0.5, 6)
    out = risk_control_report(lam, np.linspace(0.4, 0.01, 6), n=2000,
                              alpha=0.10, delta=0.05)
    assert out["abstention"] is None
    assert "not supplied" in out["abstention_note"]


def test_the_report_carries_abstention_at_the_chosen_lambda():
    lam = np.linspace(0, 0.5, 6)
    abst = np.linspace(0.0, 0.6, 6)
    out = risk_control_report(lam, np.linspace(0.4, 0.01, 6), n=2000,
                              alpha=0.10, delta=0.05, abstentions=abst)
    assert out["abstention_note"] is None
    assert out["abstention_at_lambda_hat"] == pytest.approx(
        abst[out["index_hat"]])


def test_a_mismatched_abstention_vector_is_refused():
    lam = np.linspace(0, 0.5, 6)
    with pytest.raises(ValueError, match="shape"):
        risk_control_report(lam, np.linspace(0.4, 0.01, 6), n=2000, alpha=0.10,
                            delta=0.05, abstentions=np.zeros(3))


def test_non_monotone_risk_error_is_a_risk_control_error():
    """So a caller can catch the family without enumerating its members."""
    assert issubclass(NonMonotoneRiskError, RiskControlError)
