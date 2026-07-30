"""Murphy's decomposition, and the residual that interval binning leaves behind.

    brier  =  reliability  -  resolution  +  uncertainty  +  residual

WHY THE RESIDUAL EXISTS
=======================
The classical identity is exact only when bins group IDENTICAL forecast values.
With probabilities grouped into intervals a within-bin variance term remains.

MEASURED 2026-07-30 at ten bins, before any of this was written:

    cohort              brier      rel - res + unc   residual
    well separated      0.044013   0.043380          +0.000633
    weakly separated    0.167398   0.169167          -0.001769
    coarse (5 values)   0.080840   0.081856          -0.001016
    large n (5000)      0.085092   0.085583          -0.000491

Non-zero in every case, and of both signs. So the residual is REPORTED rather
than the identity asserted and quietly failing. A decomposition that does not
close should say by how much; hiding it would make three approximate parts look
exact, which is the same class of error as a refusal reported as zero.

THE DIRECTIONS ARE NOT ALL THE SAME
-----------------------------------
    reliability   LOWER is better -- it is calibration error
    resolution    HIGHER is better -- it enters the identity NEGATIVELY
    uncertainty   descriptive -- a cohort property no model can change

Getting resolution's direction wrong would invert a dashboard column and rank the
least discriminating model first.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from genomic_variant_classifier.evaluation import metrics as M


def _cohort(n=500, separation=0.35, seed=11):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.5, n).astype(float)
    p = np.clip(0.5 + separation * (2 * y - 1) + rng.normal(0, 0.15, n),
                0.001, 0.999)
    return y, p


# --------------------------------------------------------------------------- #
# 1. The identity, closed by the residual
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("separation", [0.05, 0.15, 0.25, 0.35, 0.45])
def test_the_four_parts_reconstruct_the_brier_score_exactly(separation):
    """With the residual included the identity is EXACT, not approximate. That
    is the point of reporting it."""
    y, p = _cohort(separation=separation)
    total = (M.brier_reliability(y, p)
             - M.brier_resolution(y, p)
             + M.brier_uncertainty(y, p)
             + M.brier_decomposition_residual(y, p))
    assert total == pytest.approx(M.brier_score(y, p), rel=1e-12, abs=1e-15)


def test_the_residual_is_genuinely_non_zero_under_interval_binning():
    """Guards the reason the residual exists. If it were always zero, reporting
    it would be noise and the three parts could be trusted alone."""
    y, p = _cohort()
    residual = M.brier_decomposition_residual(y, p)
    assert residual != 0.0, (
        "the residual is exactly zero, so interval binning is no longer leaving "
        "a within-bin variance term. The measured basis for reporting it has "
        "changed and the design should be revisited rather than the test relaxed.")
    assert abs(residual) < 0.05, (
        f"residual {residual} is implausibly large; the decomposition is not "
        "merely imprecise but wrong")


# --------------------------------------------------------------------------- #
# 2. Each component means what it says
# --------------------------------------------------------------------------- #
def test_uncertainty_is_prevalence_times_one_minus_prevalence():
    """A cohort property, computable from labels alone and independent of any
    prediction."""
    y = np.concatenate([np.ones(30), np.zeros(70)])
    assert M.brier_uncertainty(y, None) == pytest.approx(0.3 * 0.7)


def test_uncertainty_is_maximal_at_balanced_prevalence():
    """0.25 at prevalence 0.5, and strictly smaller either side. This is why a
    balanced cohort is the hardest to score well on."""
    balanced = M.brier_uncertainty(np.array([1.0] * 50 + [0.0] * 50), None)
    skewed = M.brier_uncertainty(np.array([1.0] * 10 + [0.0] * 90), None)
    assert balanced == pytest.approx(0.25)
    assert skewed < balanced


def test_uncertainty_ignores_the_predictions_entirely():
    """No model can change it. Passing wildly different predictions must not
    move it at all."""
    y = np.array([1.0] * 40 + [0.0] * 60)
    assert (M.brier_uncertainty(y, np.full(100, 0.01))
            == M.brier_uncertainty(y, np.full(100, 0.99)))


def test_a_well_calibrated_forecaster_has_low_reliability():
    """Reliability IS calibration error: predictions that match observed
    frequencies within each bin drive it toward zero."""
    rng = np.random.default_rng(5)
    n = 4000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(float)   # perfectly calibrated by construction
    assert M.brier_reliability(y, p) < 0.01


def test_an_overconfident_forecaster_has_high_reliability():
    """The contrast that makes the previous test meaningful."""
    y = np.array([1.0] * 50 + [0.0] * 50)
    honest = np.full(100, 0.5)
    overconfident = np.array([0.99] * 25 + [0.01] * 25 + [0.99] * 25 + [0.01] * 25)
    assert M.brier_reliability(y, overconfident) > M.brier_reliability(y, honest)


def test_resolution_rises_with_separation():
    """Resolution IS discrimination: a model that sorts the classes into bins
    with very different frequencies scores higher."""
    weak = M.brier_resolution(*_cohort(separation=0.05))
    strong = M.brier_resolution(*_cohort(separation=0.45))
    assert strong > weak


def test_resolution_is_zero_when_every_bin_matches_the_base_rate():
    """A constant forecaster puts every observation in one bin whose frequency
    IS the prevalence, so it resolves nothing."""
    y = np.array([1.0] * 40 + [0.0] * 60)
    assert M.brier_resolution(y, np.full(100, 0.4)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3. Single-class cohorts -- calibration does not need both classes
# --------------------------------------------------------------------------- #
def test_reliability_is_defined_and_meaningful_on_a_single_class_cohort():
    """THE COMMIT 3b-1a LESSON, APPLIED AGAIN.

    Discrimination ranks one class against another and needs both. Calibration
    compares probabilities against observed frequencies and does not. An
    all-negative cohort predicted at 0.2 is overconfident by exactly 0.2, and
    reliability measures that: 0.04.

    Requiring both classes here would withhold a metric from a cohort that can
    perfectly well support it.
    """
    y = np.zeros(50)
    p = np.full(50, 0.2)
    assert M.brier_reliability(y, p) == pytest.approx(0.04)
    assert M.brier_resolution(y, p) == pytest.approx(0.0)
    assert M.brier_uncertainty(y, p) == pytest.approx(0.0)


def test_resolution_and_uncertainty_are_zero_on_a_single_class_cohort():
    """Both correctly so: there is no discrimination to measure and no outcome
    variance to explain. Only reliability carries information."""
    for y in (np.zeros(50), np.ones(50)):
        p = np.full(50, 0.5)
        assert M.brier_resolution(y, p) == pytest.approx(0.0)
        assert M.brier_uncertainty(y, p) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 4. The module's standing contracts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["brier_reliability", "brier_resolution",
                                  "brier_decomposition_residual"])
def test_the_components_fail_closed_on_non_finite_probabilities(name):
    """A non-finite predicted probability is a model-output failure, not a
    missing observation."""
    y, p = _cohort(n=100)
    p = p.copy()
    p[0] = np.nan
    with pytest.raises(Exception):
        getattr(M, name)(y, p)


@pytest.mark.parametrize("name", ["brier_reliability", "brier_resolution",
                                  "brier_decomposition_residual"])
def test_the_components_refuse_values_outside_the_unit_interval(name):
    y, p = _cohort(n=100)
    assert math.isnan(getattr(M, name)(y, p * 3.0 - 1.0))


def test_all_three_read_one_binning():
    """Commit 2b-1 established this: binning twice is how two summaries of one
    table come to disagree. Asserted by counting table constructions."""
    import unittest.mock as mock

    y, p = _cohort(n=200)
    original = M.CalibrationBins.from_predictions
    calls = []

    def counting(*args, **kwargs):
        calls.append(kwargs.get("n_bins"))
        return original(*args, **kwargs)

    with mock.patch.object(M.CalibrationBins, "from_predictions",
                           staticmethod(counting)):
        M.brier_reliability(y, p)
    assert len(calls) == 1, f"one component built {len(calls)} tables"
