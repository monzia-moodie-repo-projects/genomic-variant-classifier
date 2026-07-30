"""The three metrics added 2026-07-30, and the two defects found building them.

EVERY NUMBER HERE WAS MEASURED BEFORE THE KERNELS WERE DELIVERED, not after.

`partial_auroc` is cross-checked against scikit-learn's
`roc_auc_score(max_fpr=...)`, which implements the same McClish standardisation.
That is a genuine independent implementation, not a restatement of ours.

The two calibration metrics have no external reference in this environment, so
they are pinned by PROPERTIES a correct implementation must have: zero on a
perfect forecaster, monotone in the size of an injected miscalibration, and
refusing rather than guessing on a cohort where they are undefined.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from genomic_variant_classifier.evaluation.metrics import (
    _equal_mass_bins,
    adaptive_expected_calibration_error,
    integrated_calibration_index,
    partial_auroc,
)

BANDS = (0.05, 0.1, 0.25, 0.5, 1.0)


def _cohort(seed: int, kind: int, n: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    if kind == 0:
        s = rng.random(n)
    elif kind == 1:
        s = np.clip(0.6 * y + rng.normal(0, 0.4, n), 0, 1)
    elif kind == 2:
        s = np.round(rng.random(n), 1)            # heavy ties
    else:
        s = np.where(rng.random(n) < 0.3,
                     rng.integers(0, 2, n).astype(float), rng.random(n))
    return y, s


# --------------------------------------------------------------------------- #
# partial_auroc -- against an independent implementation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", [0, 1, 2, 3],
                         ids=["continuous", "clipped", "tied", "mixed"])
@pytest.mark.parametrize("band", BANDS, ids=[str(b) for b in BANDS])
def test_partial_auroc_matches_scikit_learn(kind, band):
    """scikit-learn's max_fpr IS McClish standardisation. Agreement must be at
    machine precision, not merely close: a metric that is 1e-3 from an
    independent implementation of the same formula has a defect somewhere."""
    y, s = _cohort(20260730 + kind, kind, 2000)
    mine = partial_auroc(y, s, fpr_low=0.0, fpr_high=band)
    ref = float(roc_auc_score(y, s, max_fpr=(None if band == 1.0 else band)))
    assert abs(mine - ref) < 1e-12, (
        f"partial_auroc over [0, {band}] on the {kind} cohort differs from "
        f"scikit-learn by {abs(mine - ref):.3e}: {mine!r} against {ref!r}")


@pytest.mark.parametrize("low,high", [(0.0, 0.05), (0.0, 0.1), (0.05, 0.3),
                                      (0.0, 1.0)])
def test_a_perfect_classifier_scores_one_on_every_band(low, high):
    y = np.tile([0, 1], 500)
    assert partial_auroc(y, y.astype(float), fpr_low=low, fpr_high=high) == \
        pytest.approx(1.0, abs=1e-12)


def test_a_chance_classifier_scores_one_half():
    """McClish is chosen precisely so this holds. Under the un-standardised
    alternative -- mean true-positive rate across the band -- it would not, and
    the number would not be comparable across bands or against AUROC."""
    y = np.tile([0, 1], 1000)
    assert partial_auroc(y, np.full(y.size, 0.5), fpr_high=0.1) == \
        pytest.approx(0.5, abs=1e-12)


def test_the_full_band_reduces_to_auroc_exactly():
    """Over [0, 1] the standardisation is the identity, so any difference from
    AUROC is a defect in the curve or the integration, not in the scaling."""
    y, s = _cohort(11, 1, 3000)
    assert partial_auroc(y, s, fpr_low=0.0, fpr_high=1.0) == \
        pytest.approx(float(roc_auc_score(y, s)), abs=1e-12)


def test_a_vertical_segment_on_the_band_edge_is_not_dropped():
    """THE DEFECT THIS TEST EXISTS FOR, measured 2026-07-30.

    A receiver operating characteristic curve is vertical wherever a tied block
    is all one class. A strict `fpr < high` discarded four such points sitting
    at fpr = 1.0 and the trapezoid ran a chord across a region the curve does
    not occupy -- over-reporting by 2.5e-07. This cohort ends in a run of
    positives, which is what produces that segment.
    """
    rng = np.random.default_rng(4242)
    n = 3000
    s = np.sort(rng.random(n))[::-1]
    y = rng.integers(0, 2, n)
    y[-40:] = 1                      # the lowest-scoring rows are all positive
    assert partial_auroc(y, s, fpr_low=0.0, fpr_high=1.0) == \
        pytest.approx(float(roc_auc_score(y, s)), abs=1e-12)


@pytest.mark.parametrize("low,high", [(0.1, 0.1), (0.5, 0.2), (-0.1, 0.5),
                                      (0.0, 1.5)])
def test_an_impossible_band_is_refused_not_defaulted(low, high):
    y = np.tile([0, 1], 100)
    with pytest.raises(ValueError, match="band"):
        partial_auroc(y, np.linspace(0, 1, y.size), fpr_low=low, fpr_high=high)


def test_a_non_finite_score_raises():
    y = np.tile([0, 1], 100)
    s = np.linspace(0, 1, y.size).copy()
    s[3] = np.inf
    with pytest.raises(ValueError, match="partial_auroc"):
        partial_auroc(y, s)


def test_a_single_class_cohort_is_undefined_not_zero():
    assert np.isnan(partial_auroc(np.ones(50, dtype=int), np.linspace(0, 1, 50)))


# --------------------------------------------------------------------------- #
# integrated_calibration_index
# --------------------------------------------------------------------------- #
def test_a_perfect_forecaster_has_zero_index():
    y = np.tile([0, 1], 500)
    assert integrated_calibration_index(y, y.astype(float)) == \
        pytest.approx(0.0, abs=1e-12)


def test_the_index_rises_with_the_miscalibration():
    """Monotonicity is the property that makes it a calibration measure at all.
    A metric that did not rise with an injected shift would be measuring
    something else."""
    rng = np.random.default_rng(9)
    p = rng.random(4000)
    y = (rng.random(p.size) < p).astype(int)
    values = [integrated_calibration_index(y, np.clip(p + d, 0, 1))
              for d in (0.0, 0.05, 0.10, 0.20, 0.30)]
    assert all(b >= a for a, b in zip(values, values[1:])), values


def test_the_index_refuses_a_non_probability():
    rng = np.random.default_rng(3)
    assert np.isnan(integrated_calibration_index(
        rng.integers(0, 2, 200), rng.normal(0, 3, 200)))


def test_the_index_refuses_a_single_class_cohort():
    """Isotonic regression on a constant target is a flat line, and the distance
    to it would describe the prevalence rather than the calibration."""
    rng = np.random.default_rng(5)
    assert np.isnan(integrated_calibration_index(
        np.ones(200, dtype=int), rng.random(200)))


# --------------------------------------------------------------------------- #
# adaptive_expected_calibration_error and its binning
# --------------------------------------------------------------------------- #
def test_the_adaptive_error_rises_with_the_miscalibration():
    rng = np.random.default_rng(17)
    p = rng.random(4000)
    y = (rng.random(p.size) < p).astype(int)
    values = [adaptive_expected_calibration_error(y, np.clip(p + d, 0, 1))
              for d in (0.0, 0.05, 0.10, 0.20, 0.30)]
    assert all(b >= a for a, b in zip(values, values[1:])), values


def test_a_saturated_cohort_still_gets_every_bin():
    """THE DEFECT THIS TEST EXISTS FOR, measured 2026-07-30.

    De-duplicated quantile edges collapsed a saturated cohort from ten bins to
    two and put the entire resolvable middle in one of them -- failing on
    exactly the case the metric was added for. Both pure leaves must be
    isolated and the middle must span several bins.
    """
    rng = np.random.default_rng(7)
    n = 5000
    p = np.where(rng.random(n) < 0.85,
                 rng.integers(0, 2, n).astype(float), rng.random(n))
    index, realised = _equal_mass_bins(p, 10)
    middle = (p > 0.0) & (p < 1.0)
    assert realised == 10, f"only {realised} bins realised on a saturated cohort"
    assert len(set(index[p == 0.0].tolist())) == 1
    assert len(set(index[p == 1.0].tolist())) == 1
    assert not set(index[p == 0.0].tolist()) & set(index[middle].tolist())
    assert not set(index[p == 1.0].tolist()) & set(index[middle].tolist())
    assert len(set(index[middle].tolist())) >= 5


def test_a_continuous_vector_still_gets_exactly_equal_counts():
    rng = np.random.default_rng(23)
    p = rng.random(5000)
    index, realised = _equal_mass_bins(p, 10)
    counts = [int((index == b).sum()) for b in range(realised)]
    assert realised == 10
    assert max(abs(c - 500) for c in counts) == 0, counts


@pytest.mark.parametrize("values", [
    np.full(200, 0.3),
    np.repeat([0.2, 0.8], 100),
    np.concatenate([np.zeros(900), np.linspace(0.01, 0.99, 100)]),
    np.array([0.1, 0.4, 0.9]),
], ids=["all_identical", "two_values", "one_dominant", "fewer_than_bins"])
def test_every_row_lands_in_exactly_one_bin_and_ties_share_it(values):
    index, realised = _equal_mass_bins(values, 10)
    assert sum(int((index == b).sum()) for b in range(realised)) == values.size
    for u in np.unique(values):
        assert len(set(index[values == u].tolist())) == 1, (
            f"the value {u} was split across bins; identical predictions must "
            "share a bin or their disagreement is reported as calibration error")


def test_a_bin_count_below_one_is_refused():
    y = np.tile([0, 1], 100)
    with pytest.raises(ValueError, match="n_bins"):
        adaptive_expected_calibration_error(y, np.linspace(0, 1, y.size),
                                            n_bins=0)


def test_the_adaptive_error_refuses_a_non_probability():
    rng = np.random.default_rng(31)
    assert np.isnan(adaptive_expected_calibration_error(
        rng.integers(0, 2, 200), rng.normal(0, 3, 200)))


# --------------------------------------------------------------------------- #
# the registry, which is what makes these metrics real
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["partial_auroc",
                                  "integrated_calibration_index",
                                  "adaptive_expected_calibration_error"])
def test_each_new_metric_is_registered_not_merely_implemented(name):
    """An implemented, unregistered kernel is an orphan. This project already
    carries three of those and does not need three more."""
    from genomic_variant_classifier.evaluation.registry import by_name, names
    assert name in names()
    assert by_name(name) is not None


def test_the_partial_area_declares_its_band():
    """The band is part of the metric's identity. A descriptor that did not
    carry it would let two different metrics share one name."""
    from genomic_variant_classifier.evaluation.registry import by_name
    parameters = by_name("partial_auroc").parameters
    assert parameters["fpr_low"] == 0.0
    assert parameters["fpr_high"] == 0.1
    assert parameters["standardisation"] == "mcclish"


def test_the_adaptive_error_declares_equal_mass_binning():
    """Sharing the equal-WIDTH parameter object would have declared a
    convention this kernel does not use."""
    from genomic_variant_classifier.evaluation.registry import by_name
    assert by_name("adaptive_expected_calibration_error").parameters["binning"] \
        == "equal_mass"


def test_the_index_declares_no_binning_parameters_at_all():
    """Binning-free is the reason it was specified. A declared bin count would
    contradict the metric."""
    from genomic_variant_classifier.evaluation.registry import by_name
    assert by_name("integrated_calibration_index").parameters == {}
