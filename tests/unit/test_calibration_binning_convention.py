"""The calibration binning convention, and the single table both statistics read.

THE DEFECT THIS CLOSES
======================
`metrics.expected_calibration_error` opened with "Equal-width binning, TOP BIN
CLOSED" -- `[lo, hi)` with only the final bin closed at 1.0 -- and implemented
`np.digitize(..., right=True)`, which makes EVERY bin `(lo, hi]`. Every
probability sitting exactly on an interior edge therefore landed one bin LOWER
than documented, and one bin lower than `ClinicalEvaluator._calibration_error`,
which had implemented the documented convention since the 2026-07-10 top-bin
repair. The two disagreed for seventeen days.

Measured on a cohort where an edge-exact value shares a bin with non-edge values
of the OPPOSITE calibration sign: 0.3242857 against 0.0642857, a relative
difference of 404%.

WHY SEVENTEEN DAYS OF TESTS NEVER SAW IT
-----------------------------------------
The expected calibration error is

    (1/N) * sum_b | sum_{i in b} (y_i - p_i) |

so it is INVARIANT to regrouping whenever every merged group shares the sign of
(accuracy - confidence): combining same-sign groups cannot change the total.
Ordinary fixtures land in that regime by default. `test_calibration_implementations_agree`
contains no interior-edge value at all, so it separated the TOP-bin definitions
and was structurally incapable of separating these. Agreement that could not have
been disagreement is not evidence, which is why every agreement assertion below
is paired with a proof that its cohort WOULD expose the superseded convention.

WHAT CHANGED, AND WHAT DID NOT
-------------------------------
No published figure moves. Every published calibration number came from the
evaluator, which was already correct. What changes is that there is now ONE
binning and ONE summation, read by the kernel, by the maximum calibration error,
and by the evaluator, rather than two implementations that happened to agree on
the fixtures anyone had thought to write.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator
from genomic_variant_classifier.evaluation.metrics import (
    CALIBRATION_BINNING,
    CALIBRATION_DEFINITION_VERSION,
    CALIBRATION_INTERVAL_CONVENTION,
    CalibrationBins,
    equal_width_bin_indices,
    expected_calibration_error,
    maximum_calibration_error,
)

N_BINS = 10
EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
INTERIOR_EDGES = [float(e) for e in EDGES[1:-1]]


def _superseded_expected(y, p, n_bins=N_BINS):
    """The pre-2026-07-27 implementation, verbatim, kept ONLY so a fixture can be
    proven to separate the two conventions. Never used in production."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += (m.sum() / y.size) * abs(y[m].mean() - p[m].mean())
    return float(total)


def _separating_cohort_at(edge: float):
    """A cohort whose calibration error depends on which side of `edge` the
    edge-exact rows fall. Rows below are well calibrated; rows at and above are
    miscalibrated in the opposite direction, so the regrouping invariance cannot
    absorb the move."""
    below, above = edge - 0.05, edge + 0.05
    p = np.concatenate([np.full(60, below), np.full(120, edge), np.full(60, above)])
    y = np.concatenate([
        np.repeat([1.0, 0.0], [int(round(below * 60)), 60 - int(round(below * 60))]),
        np.repeat([1.0, 0.0], [110, 10]),
        np.repeat([1.0, 0.0], [4, 56]),
    ])
    return y, p


# --------------------------------------------------------------------------- #
# 1. The convention itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected_bin", [
    (0.0, 0), (0.05, 0), (0.1, 1), (0.15, 1), (0.5, 5), (0.9, 9), (0.95, 9), (1.0, 9),
])
def test_the_documented_worked_example(value, expected_bin):
    assert int(equal_width_bin_indices(np.array([value]))[0]) == expected_bin


def test_the_boundary_vector():
    """An edge-exact value belongs to the bin it OPENS, and the value immediately
    below it to the bin beneath. The top bin stays closed at 1.0."""
    v = np.array([0.0, 0.1, np.nextafter(0.1, 0.0), np.nextafter(0.1, 1.0), 0.9, 1.0])
    assert equal_width_bin_indices(v).tolist() == [0, 1, 0, 1, 9, 9]


def test_every_interior_edge_opens_its_own_bin():
    for i, edge in enumerate(INTERIOR_EDGES, start=1):
        assert int(equal_width_bin_indices(np.array([edge]))[0]) == i, edge
        assert int(equal_width_bin_indices(np.array([np.nextafter(edge, 0.0)]))[0]) == i - 1


def test_predictions_of_exactly_one_are_counted():
    """The 2026-07-10 top-bin repair must survive: a half-open final bin would
    drop every pure decision-tree or ensemble leaf, the rows the model is most
    confident about."""
    assert int(equal_width_bin_indices(np.array([1.0]))[0]) == N_BINS - 1
    y, p = np.zeros(100), np.ones(100)
    assert expected_calibration_error(y, p) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("bad,exc,match", [
    (np.array([0.1, np.nan]), ValueError, "non-finite"),
    (np.array([0.1, np.inf]), ValueError, "non-finite"),
    (np.array([-0.1, 0.5]), ValueError, r"\[0, 1\]"),
    (np.array([0.5, 1.4]), ValueError, r"\[0, 1\]"),
])
def test_the_helper_fails_closed_rather_than_clipping(bad, exc, match):
    """An unguarded clip would place a non-finite or out-of-range value in the
    first or last bin, and the calibration figure would silently describe a
    different population."""
    with pytest.raises(exc, match=match):
        equal_width_bin_indices(bad)


@pytest.mark.parametrize("n_bins,exc", [(0, ValueError), (-3, ValueError), (True, TypeError)])
def test_the_bin_count_is_validated(n_bins, exc):
    with pytest.raises(exc):
        equal_width_bin_indices(np.array([0.5]), n_bins)


# --------------------------------------------------------------------------- #
# 2. One table, two statistics
# --------------------------------------------------------------------------- #
def test_both_statistics_are_read_from_one_binning():
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.5, 500).astype(float)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.15, 500), 0, 1)
    bins = CalibrationBins.from_predictions(y, p, n_bins=N_BINS)

    assert expected_calibration_error(y, p) == bins.expected
    assert maximum_calibration_error(y, p) == bins.maximum
    ece, mce = ClinicalEvaluator._calibration_error(None, y, p, n_bins=N_BINS)
    assert (ece, mce) == (bins.expected, bins.maximum), (
        "the evaluator computes calibration separately from the kernel; that is "
        "exactly how the two came to disagree about every interior edge")


def test_the_maximum_never_falls_below_the_expected():
    """A weighted mean of gaps cannot exceed the largest gap. If it does, the two
    statistics were read from different tables."""
    rng = np.random.default_rng(11)
    for seed_shift in range(8):
        r = np.random.default_rng(11 + seed_shift)
        y = r.binomial(1, 0.4, 400).astype(float)
        p = np.clip(r.random(400), 0, 1)
        bins = CalibrationBins.from_predictions(y, p)
        assert bins.maximum >= bins.expected - 1e-12


def test_only_occupied_bins_contribute():
    """An empty bin has no accuracy and no confidence. Inventing zero for either
    would drag the maximum toward that bin's midpoint and pull the weighted mean
    toward nothing."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.05, 0.95, 0.05, 0.95])          # only bins 0 and 9 occupied
    bins = CalibrationBins.from_predictions(y, p)
    assert bins.n_occupied == 2
    assert set(bins.bin_index.tolist()) == {0, 9}
    assert bins.weight.sum() == pytest.approx(1.0)


def test_the_definition_is_carried_with_the_numbers():
    """A calibration figure without its binning convention is not reproducible:
    the same predictions under the two conventions gave 0.3242857 and 0.0642857."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    definition = CalibrationBins.from_predictions(y, p, n_bins=N_BINS).definition()
    assert definition["binning"] == CALIBRATION_BINNING == "equal_width"
    assert definition["interval_convention"] == CALIBRATION_INTERVAL_CONVENTION
    assert "top bin closed" in definition["interval_convention"].lower()
    assert definition["n_bins"] == N_BINS
    assert definition["metric_definition_version"] == CALIBRATION_DEFINITION_VERSION


# --------------------------------------------------------------------------- #
# 3. The kernel and the evaluator, on cohorts that CAN separate them
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("edge", INTERIOR_EDGES)
def test_the_two_paths_agree_at_every_interior_edge(edge):
    """Each parameter first PROVES its cohort would expose the superseded
    convention, so the agreement assertion cannot pass vacuously the way the
    older agreement module does."""
    y, p = _separating_cohort_at(edge)

    documented, _ = ClinicalEvaluator._calibration_error(None, y, p, n_bins=N_BINS)
    superseded = _superseded_expected(y, p)
    assert not np.isclose(documented, superseded, rtol=1e-9, atol=1e-12), (
        f"the cohort at edge {edge} does not distinguish the two conventions, so "
        "this assertion would hold whichever one the kernel implemented")

    assert expected_calibration_error(y, p, n_bins=N_BINS) == documented, (
        f"at interior edge {edge} the kernel matched the superseded left-open "
        f"convention ({superseded}) rather than the documented one ({documented})")


def test_the_measured_separation_is_reproduced():
    """The exact cohort and figures recorded on 2026-07-27."""
    p = np.array([0.42] * 40 + [0.47] * 40 + [0.50] * 120 + [0.53] * 40 + [0.58] * 40)
    y = np.array([1.0] * 20 + [0.0] * 20 + [1.0] * 20 + [0.0] * 20 +
                 [1.0] * 110 + [0.0] * 10 + [1.0] * 4 + [0.0] * 36 +
                 [1.0] * 4 + [0.0] * 36)
    assert _superseded_expected(y, p) == pytest.approx(0.3242857142857143, abs=1e-15)
    assert expected_calibration_error(y, p) == pytest.approx(0.06428571428571427, abs=1e-15)


def test_the_regrouping_invariance_is_real_and_is_why_this_hid():
    """Not folklore: when every merged group shares the sign of
    (accuracy - confidence), regrouping cannot change the total. That is why
    ordinary fixtures agreed under both conventions and why the defect survived
    seventeen days."""
    p = np.array([0.05] * 50 + [0.15] * 50)     # both below their accuracy
    y = np.array([1.0] * 45 + [0.0] * 5 + [1.0] * 45 + [0.0] * 5)
    merged = abs(y.mean() - p.mean())
    separate = 0.5 * abs(y[:50].mean() - 0.05) + 0.5 * abs(y[50:].mean() - 0.15)
    assert merged == pytest.approx(separate, abs=1e-12)


# --------------------------------------------------------------------------- #
# 4. Representation invariants, added 2026-07-28 by commit 3b-1a
#
# "At least one occupied bin" is NOT an applicability condition. It is a THEOREM
# of the conditions applicability already checks -- a non-empty vector of valid
# probabilities in [0, 1], binned into equal widths with a closed top edge, must
# occupy a bin. Putting it in applicability would create a branch unreachable
# through the public path, and would make INSUFFICIENT_SUPPORT mean two different
# things: a cohort too small to support the estimand, and an implementation that
# broke its own contract.
#
# It is enforced at the REPRESENTATION boundary instead, where a violation is
# FAILED.
# --------------------------------------------------------------------------- #
def test_nonempty_valid_calibration_input_always_occupies_a_bin():
    """The theorem, over a dense deterministic set rather than by example."""
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    probabilities = np.clip(np.concatenate([
        np.linspace(0.0, 1.0, 10001),
        np.nextafter(edges, -np.inf),
        edges,
        np.nextafter(edges, np.inf),
    ]), 0.0, 1.0)

    indices = equal_width_bin_indices(probabilities, N_BINS)
    counts = np.bincount(indices, minlength=N_BINS)

    assert indices.min() >= 0
    assert indices.max() < N_BINS
    assert counts.sum() == probabilities.size, "a probe escaped every bin"
    assert (counts > 0).any()


@pytest.mark.parametrize("label,y,p", [
    ("single observation at zero", np.array([0.0]), np.array([0.0])),
    ("single observation at one", np.array([1.0]), np.array([1.0])),
    ("repeated identical probability", np.zeros(50), np.full(50, 0.37)),
    ("all-negative labels", np.zeros(80), np.full(80, 0.10)),
    ("all-positive labels", np.ones(80), np.full(80, 0.90)),
])
def test_named_boundary_fixtures_occupy_a_bin(label, y, p):
    """The dense test proves the theorem; these explain the contract."""
    bins = CalibrationBins.from_predictions(y, p, n_bins=N_BINS)
    assert bins.n_occupied >= 1, label
    assert np.isfinite(bins.expected)
    assert np.isfinite(bins.maximum)


def test_an_impossible_empty_occupancy_is_an_invariant_failure():
    """Constructed at the representation boundary, NOT reached through valid
    public inputs -- the point of the test is precisely that it is unreachable
    through them."""
    from genomic_variant_classifier.evaluation import metrics as metrics_module
    from genomic_variant_classifier.evaluation.metrics import CalibrationInvariantError

    original = metrics_module.equal_width_bin_indices
    try:
        metrics_module.equal_width_bin_indices = lambda v, n=N_BINS: np.array([], dtype=np.int64)
        with pytest.raises(CalibrationInvariantError, match="no occupied calibration bin"):
            CalibrationBins.from_predictions(np.zeros(10), np.full(10, 0.5), n_bins=N_BINS)
    finally:
        metrics_module.equal_width_bin_indices = original


def test_an_out_of_range_bin_index_is_an_invariant_failure():
    """FOUND BY SABOTAGE. Occupancy alone was not enough.

    Mapping 1.0 to bin 10 of a ten-bin table produced an OCCUPIED, entirely
    plausible table -- expected 0.375, maximum 0.5, status OK, no exception --
    because the occupancy check asks "did anything land in a bin?" and this asks
    "did everything land in a VALID bin?". Only the second catches an index that
    escaped the range.
    """
    from genomic_variant_classifier.evaluation import metrics as metrics_module
    from genomic_variant_classifier.evaluation.metrics import CalibrationInvariantError

    original = metrics_module.equal_width_bin_indices

    def escapes_the_range(values, n_bins=N_BINS):
        idx = original(values, n_bins)
        return np.where(np.asarray(values, dtype=float).ravel() >= 1.0, n_bins, idx)

    try:
        metrics_module.equal_width_bin_indices = escapes_the_range
        with pytest.raises(CalibrationInvariantError, match="out of range"):
            CalibrationBins.from_predictions(
                np.array([0.0, 1.0] * 40),
                np.concatenate([np.full(40, 1.0), np.linspace(0.05, 0.45, 40)]),
                n_bins=N_BINS)
    finally:
        metrics_module.equal_width_bin_indices = original


def test_a_broken_binning_yields_FAILED_and_never_INSUFFICIENT_SUPPORT():
    """The layer assignment, asserted on SEMANTIC outcomes rather than on the
    internal exception site. A binning defect must never be reported as a
    property of the cohort."""
    from genomic_variant_classifier.evaluation import metrics as metrics_module
    from genomic_variant_classifier.evaluation.capabilities import MetricStatus
    from genomic_variant_classifier.evaluation.population import EvaluationPopulation
    from genomic_variant_classifier.evaluation.registry import (
        MetricContext, evaluate_registered)

    original = metrics_module.equal_width_bin_indices

    def escapes_the_range(values, n_bins=N_BINS):
        idx = original(values, n_bins)
        return np.where(np.asarray(values, dtype=float).ravel() >= 1.0, n_bins, idx)

    y = np.array([0.0, 1.0] * 40)
    p = np.concatenate([np.full(40, 1.0), np.linspace(0.05, 0.45, 40)])
    try:
        metrics_module.equal_width_bin_indices = escapes_the_range
        results = evaluate_registered(MetricContext(
            y_true=y, y_prob=p, y_score=p,
            population=EvaluationPopulation.full(80, scope="c", source_id=None)))
    finally:
        metrics_module.equal_width_bin_indices = original

    for name in ("expected_calibration_error", "maximum_calibration_error"):
        r = results[name]
        assert r.status is MetricStatus.FAILED, name
        assert r.status is not MetricStatus.INSUFFICIENT_SUPPORT, (
            f"{name}: a binning defect was reported as a cohort property")
        assert not np.isfinite(r.value), f"{name}: a value was produced anyway"


def test_calibration_never_consults_the_legacy_undefined_resolver():
    """STRUCTURAL, not a check of the policy table's current contents.

    Someone restoring the both-classes applicability rule could mask the
    regression by adding a calibration compatibility substitution. A test that
    only inspected the table would not notice; one that counts calls to the
    resolver does.
    """
    from genomic_variant_classifier.evaluation import legacy_projection
    from genomic_variant_classifier.evaluation.population import EvaluationPopulation
    from genomic_variant_classifier.evaluation.registry import (
        MetricContext, evaluate_registered)

    consulted = []
    original = legacy_projection.resolve_undefined_projection

    def counting(result, *, policy, metric_results=None):
        # Signature tracks the resolver, which gained `metric_results` in commit
        # 3b-1b so the derived single-class AUPRC rule can read prevalence. A spy
        # that silently swallowed extra keywords would keep passing while no
        # longer standing in for the real function.
        consulted.append(policy.field_name)
        return original(result, policy=policy, metric_results=metric_results)

    legacy_projection.resolve_undefined_projection = counting
    try:
        for y, p in ((np.zeros(80), np.full(80, 0.10)),
                     (np.ones(80), np.full(80, 0.90)),
                     (np.array([0.0, 1.0] * 40), np.linspace(0.05, 0.95, 80))):
            consulted.clear()
            results = evaluate_registered(MetricContext(
                y_true=y, y_prob=p, y_score=p,
                population=EvaluationPopulation.full(y.size, scope="c", source_id=None)))
            legacy_projection.project_legacy_fields(results)
            assert consulted.count("calibration_ece") == 0
            assert consulted.count("calibration_mce") == 0
    finally:
        legacy_projection.resolve_undefined_projection = original

