"""Tests for ordinal conformal prediction sets.

WHAT THESE TESTS ARE FOR
========================
The module's central claim is that every prediction set is a CONTIGUOUS interval
of adjacent American College of Medical Genetics and Genomics / Association for
Molecular Pathology (ACMG/AMP) tiers, and that the finite-sample coverage
guarantee survives that restriction. Both halves are asserted here against
OUTCOMES, not against the construction that produced them: contiguity is
recomputed from the returned boolean matrix, and the nested expansion is checked
against a deliberately slow, independent reference implementation that shares no
code with the vectorised one.

Two defects in the module were found this way during development and both are
pinned by tests below, so that neither can return silently:

  1. An empty-set "repair" that forced the modal tier in, inflating coverage by
     exactly the fraction of rows repaired (measured: +8.2 points at alpha=0.30)
     and destroying exact calibration. Now opt-in and counted.
  2. ordinal_report counting ABSTENTIONS as catastrophic errors, because
     distance_to_set scores an empty set as K, which exceeds any threshold. An
     abstention and a confident far-wrong assertion are opposite events and the
     safer one was being penalised as though it were the worst.

DETERMINISM
-----------
Every test uses an explicit seed. The coverage tests average over many trials
with fixed seeds so their outcome is deterministic rather than merely probable;
a statistical test that fails one run in twenty is worse than no test, because it
trains the reader to re-run rather than investigate.

Placement: tests/unit/test_ordinal_conformal.py
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.conformal.ordinal import (
    ACMG5_TIERS,
    MIN_ORDINAL_CLASSES,
    OrdinalConformalClassifier,
    OrdinalLabelError,
    distance_to_set,
    interval_bounds,
    is_contiguous,
    nested_interval_order,
    ordinal_report,
    ordinal_scores_all,
    ordinal_scores_true,
)

K = 5


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #
def make_ordinal_data(n, rng, k=K, signal=2.0, neighbour=0.8):
    """Five-tier data with genuine ordinal structure.

    The true tier gets a probability boost and so do its immediate neighbours,
    which is what makes the data ordinal rather than merely multi-class: an
    error of one tier must be more likely than an error of four. Without that,
    the ordinal machinery would be tested on data it is not designed for and
    would look worse than it is.
    """
    y = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k)) * 1.2
    logits[np.arange(n), y] += signal
    for d in (1, -1):
        nb = np.clip(y + d, 0, k - 1)
        logits[np.arange(n), nb] += neighbour
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return P, y


def slow_nested_order(p, tie_break="low"):
    """Reference implementation of the nested expansion: one row, plain Python.

    Deliberately naive and obviously correct. Its only purpose is to disagree
    with the vectorised version if that version is wrong. It shares no code with
    it beyond numpy indexing.
    """
    k = len(p)
    m = int(np.argmax(p))
    lo = hi = m
    step = [-1] * k
    step[m] = 0
    for t in range(1, k):
        left_ok, right_ok = lo > 0, hi < k - 1
        if left_ok and right_ok:
            if tie_break == "low":
                go_left = p[lo - 1] >= p[hi + 1]
            else:
                go_left = p[lo - 1] > p[hi + 1]
        else:
            go_left = left_ok
        if go_left:
            lo -= 1
            step[lo] = t
        else:
            hi += 1
            step[hi] = t
    return step


# --------------------------------------------------------------------------- #
# 1. the nested expansion
# --------------------------------------------------------------------------- #
def test_nested_order_is_a_permutation_of_steps():
    rng = np.random.default_rng(0)
    P, _ = make_ordinal_data(500, rng)
    step = nested_interval_order(P)
    for row in step:
        assert sorted(row.tolist()) == list(range(K))


def test_every_nested_prefix_is_contiguous():
    """I_0 subset I_1 subset ... and each is an unbroken run.

    This is the property the coverage argument rests on. If a prefix were
    gapped, thresholding the score could return a gapped set and the module's
    entire reason for existing would be void.
    """
    rng = np.random.default_rng(1)
    P, _ = make_ordinal_data(400, rng)
    order = np.argsort(nested_interval_order(P), axis=1)
    for i in range(order.shape[0]):
        for t in range(K):
            idx = np.sort(order[i, : t + 1])
            assert idx[-1] - idx[0] + 1 == len(idx), f"row {i}, step {t}"


@pytest.mark.parametrize("tie_break", ["low", "high"])
def test_vectorised_expansion_matches_slow_reference(tie_break):
    """Independent implementation cross-check -- the strongest test here."""
    rng = np.random.default_rng(2)
    P, _ = make_ordinal_data(300, rng)
    fast = nested_interval_order(P, tie_break=tie_break)
    for i in range(P.shape[0]):
        assert fast[i].tolist() == slow_nested_order(P[i], tie_break), f"row {i}"


def test_tie_break_actually_differs_on_an_exact_tie():
    """A parameter that never changes anything is not a parameter.

    The first draft of this test used [0.25, 0.25, 0.50, 0.00, 0.00] and called
    the neighbours tied. They are not: the modal tier is index 2, whose
    neighbours are 0.25 and 0.00. Both policies went left and the test failed
    against correct code. A symmetric row is required for an actual tie.
    """
    p = np.array([[0.00, 0.25, 0.50, 0.25, 0.00]])   # mode 2, neighbours both 0.25
    assert p[0, 1] == p[0, 3], "fixture must present a genuine tie"
    lo = nested_interval_order(p, tie_break="low")
    hi = nested_interval_order(p, tie_break="high")
    assert lo[0, 1] == 1, "tie_break='low' should absorb the LOWER tier first"
    assert hi[0, 3] == 1, "tie_break='high' should absorb the HIGHER tier first"
    assert not np.array_equal(lo, hi)


def test_tie_break_is_irrelevant_when_there_is_no_tie():
    """The complement. If the policy changed the order on untied rows it would
    be altering the expansion rule itself, not just breaking ties."""
    rng = np.random.default_rng(4)
    P, _ = make_ordinal_data(200, rng)          # continuous, ties have measure zero
    assert np.array_equal(nested_interval_order(P, "low"),
                          nested_interval_order(P, "high"))


def test_mass_at_absorption_is_monotone_in_step():
    """s(k) must be non-decreasing in absorption step, or level sets are not
    nested intervals and contiguity fails."""
    rng = np.random.default_rng(3)
    P, _ = make_ordinal_data(200, rng)
    S = ordinal_scores_all(P, randomize=False)
    step = nested_interval_order(P)
    for i in range(P.shape[0]):
        by_step = S[i][np.argsort(step[i])]
        assert np.all(np.diff(by_step) >= -1e-12), f"row {i} not monotone"


# --------------------------------------------------------------------------- #
# 2. contiguity of prediction sets -- the central claim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.02, 0.05, 0.10, 0.20, 0.35, 0.50])
def test_prediction_sets_are_always_contiguous(alpha):
    """Recomputed from the OUTPUT, not trusted from the construction."""
    rng = np.random.default_rng(int(alpha * 1000))
    P_cal, y_cal = make_ordinal_data(1200, rng)
    P_test, _ = make_ordinal_data(1200, rng)
    model = OrdinalConformalClassifier(alpha=alpha, seed=7).fit(P_cal, y_cal)
    sets = model.predict_set(P_test)
    assert is_contiguous(sets).all()


def test_contiguity_holds_for_adversarially_bimodal_probabilities():
    """The failure mode ordinal conformal exists to prevent.

    A model that is confident the variant is either Benign or Pathogenic and
    nothing between is exactly where an unordered set returns {Benign,
    Pathogenic} -- a clinically meaningless answer. Here it must not.
    """
    rng = np.random.default_rng(11)
    n = 800
    P = np.full((n, K), 0.002)
    P[:, 0] = 0.5
    P[:, K - 1] = 0.494
    P /= P.sum(axis=1, keepdims=True)
    y = rng.choice([0, K - 1], size=n)
    model = OrdinalConformalClassifier(alpha=0.1, seed=1,
                                       allow_degenerate_labels=True).fit(P, y)
    sets = model.predict_set(P)
    assert is_contiguous(sets).all()
    gapped = sets[:, 0] & sets[:, K - 1] & ~sets[:, K // 2]
    assert not gapped.any(), "returned a gapped Benign-or-Pathogenic set"


# --------------------------------------------------------------------------- #
# 3. coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_marginal_coverage_meets_target(alpha):
    """Averaged over 25 fixed-seed trials, so the outcome is deterministic."""
    covs = []
    for t in range(25):
        rng = np.random.default_rng(5000 + int(alpha * 100) * 100 + t)
        P_cal, y_cal = make_ordinal_data(1500, rng)
        P_test, y_test = make_ordinal_data(3000, rng)
        model = OrdinalConformalClassifier(alpha=alpha, seed=t).fit(P_cal, y_cal)
        sets = model.predict_set(P_test)
        covs.append(sets[np.arange(len(y_test)), y_test].mean())
    mean_cov = float(np.mean(covs))
    assert mean_cov >= (1 - alpha) - 0.01, (
        f"mean coverage {mean_cov:.4f} below target {1 - alpha:.3f}")


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_coverage_is_not_wastefully_conservative(alpha):
    """The other half of calibration. Over-coverage is not free -- it is paid for
    in set width, and a method that always returned every tier would pass a
    coverage-only test perfectly while being useless."""
    covs = []
    for t in range(25):
        rng = np.random.default_rng(6000 + int(alpha * 100) * 100 + t)
        P_cal, y_cal = make_ordinal_data(1500, rng)
        P_test, y_test = make_ordinal_data(3000, rng)
        model = OrdinalConformalClassifier(alpha=alpha, seed=t).fit(P_cal, y_cal)
        sets = model.predict_set(P_test)
        covs.append(sets[np.arange(len(y_test)), y_test].mean())
    mean_cov = float(np.mean(covs))
    assert mean_cov <= (1 - alpha) + 0.03, (
        f"mean coverage {mean_cov:.4f} exceeds target {1 - alpha:.3f} by more "
        "than 3 points -- sets are wider than the data supports")


def test_smaller_alpha_gives_wider_sets():
    rng = np.random.default_rng(21)
    P_cal, y_cal = make_ordinal_data(2000, rng)
    P_test, _ = make_ordinal_data(2000, rng)
    widths = []
    for alpha in (0.30, 0.20, 0.10, 0.05):
        m = OrdinalConformalClassifier(alpha=alpha, seed=3).fit(P_cal, y_cal)
        widths.append(m.predict_set(P_test).sum(axis=1).mean())
    assert all(a < b for a, b in zip(widths, widths[1:])), widths


def test_tiny_alpha_returns_every_tier():
    """conformal_quantile returns +inf when alpha < 1/(n+1); the set must then
    be the whole scale rather than something arbitrary."""
    rng = np.random.default_rng(22)
    P_cal, y_cal = make_ordinal_data(20, rng)
    P_test, _ = make_ordinal_data(50, rng)
    m = OrdinalConformalClassifier(alpha=0.001, seed=0).fit(P_cal, y_cal)
    assert np.isinf(m.q_hat_)
    assert m.predict_set(P_test).all()


# --------------------------------------------------------------------------- #
# 4. the binary-labels guard (metric specification, Panel C)
# --------------------------------------------------------------------------- #
def test_binary_labels_are_refused():
    """The current cohort collapses to two classes. Ordinal conformal on
    threshold-derived bands is vacuous, and the module must say so."""
    rng = np.random.default_rng(30)
    P, _ = make_ordinal_data(400, rng)
    y_binary = rng.integers(0, 2, size=400) * (K - 1)
    with pytest.raises(OrdinalLabelError, match="distinct value"):
        OrdinalConformalClassifier().fit(P, y_binary)


def test_single_class_labels_are_refused():
    rng = np.random.default_rng(31)
    P, _ = make_ordinal_data(100, rng)
    with pytest.raises(OrdinalLabelError):
        OrdinalConformalClassifier().fit(P, np.zeros(100, dtype=int))


def test_three_classes_is_the_boundary_and_is_accepted():
    """MIN_ORDINAL_CLASSES is 3; two must fail and three must pass, or the
    constant is decorative."""
    rng = np.random.default_rng(32)
    P, _ = make_ordinal_data(600, rng)
    y3 = rng.integers(0, MIN_ORDINAL_CLASSES, size=600)
    model = OrdinalConformalClassifier().fit(P, y3)
    assert model.q_hat_ is not None
    assert not model.degenerate_labels_


def test_escape_hatch_works_and_records_that_it_was_used():
    """Bypassing the guard must leave a trace on the object, otherwise a result
    produced under it is indistinguishable from a legitimate one."""
    rng = np.random.default_rng(33)
    P, _ = make_ordinal_data(300, rng)
    y_binary = rng.integers(0, 2, size=300) * (K - 1)
    model = OrdinalConformalClassifier(allow_degenerate_labels=True).fit(P, y_binary)
    assert model.degenerate_labels_ is True
    assert model.q_hat_ is not None


def test_two_tier_probability_matrix_is_refused():
    """With K=2 every set is contiguous trivially; the guarantee is vacuous."""
    P = np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])
    with pytest.raises(OrdinalLabelError, match="at least 3"):
        OrdinalConformalClassifier().fit(P, np.array([0, 1, 0]))


# --------------------------------------------------------------------------- #
# 5. the empty-set override -- defect 1, pinned
# --------------------------------------------------------------------------- #
def test_force_nonempty_is_off_by_default():
    rng = np.random.default_rng(40)
    P_cal, y_cal = make_ordinal_data(1500, rng)
    P_test, _ = make_ordinal_data(2000, rng)
    m = OrdinalConformalClassifier(alpha=0.35, seed=4).fit(P_cal, y_cal)
    sets = m.predict_set(P_test)
    assert m.force_nonempty is False
    assert m.n_forced_nonempty_ == 0
    assert (~sets.any(axis=1)).any(), (
        "expected some empty sets at alpha=0.35; if none occur this test proves "
        "nothing and the alpha must be raised")


def test_force_nonempty_eliminates_empty_sets_and_counts_them():
    rng = np.random.default_rng(41)
    P_cal, y_cal = make_ordinal_data(1500, rng)
    P_test, _ = make_ordinal_data(2000, rng)
    m = OrdinalConformalClassifier(alpha=0.35, seed=4,
                                   force_nonempty=True).fit(P_cal, y_cal)
    sets = m.predict_set(P_test)
    assert sets.any(axis=1).all()
    assert m.n_forced_nonempty_ > 0
    assert m.n_predicted_ == P_test.shape[0]
    assert m.frac_forced_nonempty_ == pytest.approx(
        m.n_forced_nonempty_ / m.n_predicted_)


def test_force_nonempty_inflates_coverage_which_is_why_it_is_off():
    """Pins the measured cost. If someone flips the default, this fails and the
    reason is in the assertion message."""
    rng = np.random.default_rng(42)
    P_cal, y_cal = make_ordinal_data(2000, rng)
    P_test, y_test = make_ordinal_data(4000, rng)
    off = OrdinalConformalClassifier(alpha=0.30, seed=5).fit(P_cal, y_cal)
    on = OrdinalConformalClassifier(alpha=0.30, seed=5,
                                    force_nonempty=True).fit(P_cal, y_cal)
    idx = np.arange(len(y_test))
    cov_off = off.predict_set(P_test)[idx, y_test].mean()
    cov_on = on.predict_set(P_test)[idx, y_test].mean()
    assert cov_on > cov_off, (
        "forcing a non-empty set can only add coverage; if it does not, the "
        "override is not doing what it claims")
    assert cov_off <= 0.75, (
        f"without the override coverage should sit near the 0.70 target; got "
        f"{cov_off:.4f}")


def test_counters_reset_between_prediction_batches():
    """A counter that accumulates across calls reports the wrong number for the
    batch the caller just asked about."""
    rng = np.random.default_rng(43)
    P_cal, y_cal = make_ordinal_data(1200, rng)
    m = OrdinalConformalClassifier(alpha=0.35, seed=6,
                                   force_nonempty=True).fit(P_cal, y_cal)
    P_a, _ = make_ordinal_data(500, rng)
    P_b, _ = make_ordinal_data(900, rng)
    m.predict_set(P_a)
    first = m.n_forced_nonempty_
    m.predict_set(P_b)
    assert m.n_predicted_ == 900
    assert m.n_forced_nonempty_ != first or first == 0


# --------------------------------------------------------------------------- #
# 6. abstentions versus catastrophic errors -- defect 2, pinned
# --------------------------------------------------------------------------- #
def test_abstention_is_not_counted_as_catastrophic():
    """An empty set is the method declining to assert. A catastrophic error is a
    confident assertion at the opposite end of the scale. Conflating them
    penalises the safer behaviour, which would push a clinical system towards
    answering when it should decline.
    """
    sets = np.zeros((3, K), dtype=bool)      # three abstentions
    y = np.array([0, 2, 4])
    rep = ordinal_report(sets, y, alpha=0.1, tier_names=ACMG5_TIERS)
    assert rep["n_abstentions"] == 3
    assert rep["n_catastrophic"] == 0, (
        "empty sets were counted as catastrophic errors -- distance_to_set "
        "scores them as K, which exceeds any threshold")


def test_a_real_catastrophic_error_is_counted():
    """The complement: a confident Benign assertion for a truly Pathogenic
    variant must register."""
    sets = np.zeros((1, K), dtype=bool)
    sets[0, 0] = True                         # asserts Benign
    rep = ordinal_report(sets, np.array([K - 1]), alpha=0.1,
                         tier_names=ACMG5_TIERS)
    assert rep["n_catastrophic"] == 1
    assert rep["n_abstentions"] == 0
    assert rep["catastrophic_rate"] == pytest.approx(1.0)


def test_adjacent_tier_error_is_not_catastrophic():
    sets = np.zeros((1, K), dtype=bool)
    sets[0, K - 2] = True                     # Likely pathogenic for Pathogenic
    rep = ordinal_report(sets, np.array([K - 1]), alpha=0.1)
    assert rep["n_catastrophic"] == 0
    assert rep["mean_distance_when_missed"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 7. helpers
# --------------------------------------------------------------------------- #
def test_is_contiguous_detects_a_gap():
    sets = np.array([
        [True, True, False, False, False],    # contiguous
        [True, False, True, False, False],    # gapped
        [False, False, False, False, False],  # empty counts as contiguous
        [True, True, True, True, True],       # full
    ])
    assert is_contiguous(sets).tolist() == [True, False, True, True]


def test_interval_bounds():
    sets = np.array([
        [False, True, True, False, False],
        [True, False, False, False, False],
        [False, False, False, False, False],
    ])
    lo, hi = interval_bounds(sets)
    assert lo.tolist() == [1, 0, -1]
    assert hi.tolist() == [2, 0, -1]


def test_distance_to_set():
    sets = np.array([
        [False, True, True, False, False],
        [True, False, False, False, False],
    ])
    assert distance_to_set(sets, np.array([1, 0])).tolist() == [0, 0]
    assert distance_to_set(sets, np.array([4, 4])).tolist() == [2, 4]


def test_empty_set_distance_is_penalised_not_rewarded():
    """Scoring an abstention as distance 0 would make it look perfect."""
    d = distance_to_set(np.zeros((1, K), dtype=bool), np.array([2]))
    assert d.tolist() == [K]


# --------------------------------------------------------------------------- #
# 8. input validation -- nothing fails silently
# --------------------------------------------------------------------------- #
def test_rows_must_sum_to_one():
    P = np.array([[0.2, 0.2, 0.2, 0.2, 0.1]])     # sums to 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        ordinal_scores_all(P)


def test_negative_probabilities_rejected():
    P = np.array([[0.5, -0.1, 0.3, 0.2, 0.1]])
    with pytest.raises(ValueError, match="negative"):
        ordinal_scores_all(P)


def test_non_finite_probabilities_rejected():
    P = np.array([[0.5, np.nan, 0.2, 0.2, 0.1]])
    with pytest.raises(ValueError, match="non-finite"):
        ordinal_scores_all(P)


def test_labels_outside_the_tier_range_rejected():
    rng = np.random.default_rng(50)
    P, _ = make_ordinal_data(10, rng)
    with pytest.raises(ValueError, match="must lie in"):
        ordinal_scores_true(P, np.full(10, K))


def test_predict_before_fit_raises():
    rng = np.random.default_rng(51)
    P, _ = make_ordinal_data(10, rng)
    with pytest.raises(RuntimeError, match="fit"):
        OrdinalConformalClassifier().predict_set(P)


def test_tier_count_mismatch_between_fit_and_predict_raises():
    rng = np.random.default_rng(52)
    P_cal, y_cal = make_ordinal_data(300, rng, k=5)
    P_other, _ = make_ordinal_data(50, rng, k=4)
    m = OrdinalConformalClassifier().fit(P_cal, y_cal)
    with pytest.raises(ValueError, match="calibrated with K"):
        m.predict_set(P_other)


def test_mismatched_lengths_raise():
    rng = np.random.default_rng(53)
    P, y = make_ordinal_data(40, rng)
    with pytest.raises(ValueError, match="length"):
        OrdinalConformalClassifier().fit(P, y[:20])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_be_strictly_inside_zero_one(alpha):
    with pytest.raises(ValueError, match="alpha"):
        OrdinalConformalClassifier(alpha=alpha)


def test_bad_tie_break_rejected():
    with pytest.raises(ValueError, match="tie_break"):
        OrdinalConformalClassifier(tie_break="middle")


def test_randomize_without_u_raises():
    rng = np.random.default_rng(54)
    P, _ = make_ordinal_data(10, rng)
    with pytest.raises(ValueError, match="requires u"):
        ordinal_scores_all(P, u=None, randomize=True)


# --------------------------------------------------------------------------- #
# 9. reproducibility and the report contract
# --------------------------------------------------------------------------- #
def test_same_seed_gives_identical_sets():
    rng = np.random.default_rng(60)
    P_cal, y_cal = make_ordinal_data(800, rng)
    P_test, _ = make_ordinal_data(800, rng)
    a = OrdinalConformalClassifier(alpha=0.1, seed=99).fit(P_cal, y_cal).predict_set(P_test)
    b = OrdinalConformalClassifier(alpha=0.1, seed=99).fit(P_cal, y_cal).predict_set(P_test)
    assert np.array_equal(a, b)


def test_unrandomized_path_is_deterministic_and_still_contiguous():
    rng = np.random.default_rng(61)
    P_cal, y_cal = make_ordinal_data(900, rng)
    P_test, _ = make_ordinal_data(900, rng)
    m = OrdinalConformalClassifier(alpha=0.1, randomize=False, seed=0).fit(P_cal, y_cal)
    s1, s2 = m.predict_set(P_test), m.predict_set(P_test)
    assert np.array_equal(s1, s2)
    assert is_contiguous(s1).all()


def test_report_fields_are_internally_consistent():
    rng = np.random.default_rng(62)
    P_cal, y_cal = make_ordinal_data(1500, rng)
    P_test, y_test = make_ordinal_data(2000, rng)
    m = OrdinalConformalClassifier(alpha=0.1, seed=8).fit(P_cal, y_cal)
    sets = m.predict_set(P_test)
    rep = ordinal_report(sets, y_test, alpha=0.1, tier_names=ACMG5_TIERS)

    assert rep["n"] == 2000
    assert rep["K"] == K
    assert rep["target_coverage"] == pytest.approx(0.9)
    assert rep["coverage_gap"] == pytest.approx(
        rep["empirical_coverage"] - rep["target_coverage"])
    assert rep["all_sets_contiguous"] is True
    assert rep["n_non_contiguous"] == 0
    assert sum(rep["width_histogram"].values()) == 2000
    assert sum(rep["per_tier_n"].values()) == 2000
    assert set(rep["per_tier_coverage"]) == set(ACMG5_TIERS)
    assert 1 <= rep["mean_width"] <= K or rep["n_abstentions"] > 0


def test_report_rejects_wrong_length_tier_names():
    sets = np.ones((2, K), dtype=bool)
    with pytest.raises(ValueError, match="tier_names"):
        ordinal_report(sets, np.array([0, 1]), alpha=0.1, tier_names=("a", "b"))


def test_predict_interval_agrees_with_predict_set():
    rng = np.random.default_rng(63)
    P_cal, y_cal = make_ordinal_data(600, rng)
    P_test, _ = make_ordinal_data(600, rng)
    m = OrdinalConformalClassifier(alpha=0.15, seed=2).fit(P_cal, y_cal)
    lo, hi = m.predict_interval(P_test)
    sets = m.predict_set(P_test)
    lo2, hi2 = interval_bounds(sets)
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)


def test_acmg5_tiers_are_in_ascending_severity_order():
    """Index order IS the ordinal relation. If this tuple were ever reordered,
    every distance and every interval in the module would silently change
    meaning while remaining numerically valid."""
    assert ACMG5_TIERS == (
        "Benign", "Likely benign", "Uncertain significance",
        "Likely pathogenic", "Pathogenic",
    )
    assert len(ACMG5_TIERS) == K
