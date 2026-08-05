"""The exact threshold sweep: counts at every achievable threshold.

OP-1 step 1, 2026-08-04. A SEPARATE MODULE because a sweep is not vocabulary --
`test_threshold_vocabulary.py` holds ten tests about naming, identity, ownership
and membership, and filing an algorithm's tests there would be the misfiling
OP-0 already paid an erratum for.

These tests stand alone rather than leaning on a shadow comparison, because of
OPCOV-1: `_find_high_ppv_point` was exercised by NOTHING and the operating points
by seven lines across four files, so a shadow that agrees proves agreement on the
cases exercised -- and those are few.

Author: Monzia Moodie
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.population import EvaluationPopulation
from genomic_variant_classifier.evaluation.thresholds import (
    ConfusionCounts,
    ThresholdOperator,
    ThresholdSource,
    sweep_thresholds,
)


def _population(n):
    return EvaluationPopulation.full(n, scope="sweep_test",
                                     source_id="op1-step1-test")


def _brute_force(y, p, threshold, strictly_greater):
    """The definition, computed the slow obvious way.

    The sweep is an OPTIMISATION, and an optimisation is correct only if it
    agrees with the thing it replaces. The oracle is therefore the definition
    itself rather than another clever construction.
    """
    flagged = (p > threshold) if strictly_greater else (p >= threshold)
    return (int((flagged & (y == 1)).sum()),
            int((flagged & (y == 0)).sum()),
            int(((~flagged) & (y == 1)).sum()),
            int(((~flagged) & (y == 0)).sum()))


_COHORTS = {
    "all_distinct": (np.array([1., 0., 1., 0.]), np.array([0.9, 0.7, 0.5, 0.3])),
    "one_tied_pair": (np.array([1., 0., 1., 0.]), np.array([0.9, 0.9, 0.5, 0.3])),
    "all_tied": (np.array([1., 0., 1., 0.]), np.array([0.5, 0.5, 0.5, 0.5])),
    "ties_at_the_top": (np.array([1., 1., 0., 0.]),
                        np.array([0.9, 0.9, 0.4, 0.1])),
    "single_row": (np.array([1.]), np.array([0.7])),
    "all_positive": (np.array([1., 1., 1.]), np.array([0.9, 0.5, 0.1])),
    "all_negative": (np.array([0., 0., 0.]), np.array([0.9, 0.5, 0.1])),
    "boundary_scores": (np.array([1., 0., 1., 0.]),
                        np.array([1.0, 1.0, 0.0, 0.0])),
}


@pytest.mark.parametrize("name", sorted(_COHORTS))
def test_every_candidate_matches_the_brute_force_definition(name):
    """THE ACCEPTANCE CRITERION. Agreement with the definition at every
    candidate, on cohorts chosen to exercise every tie arrangement."""
    y, p = _COHORTS[name]
    sweep = sweep_thresholds(y, p, population=_population(y.size))

    for index in range(len(sweep)):
        candidate = sweep[index]
        expected = _brute_force(
            y, p, candidate.threshold,
            candidate.operator is ThresholdOperator.GREATER)
        actual = (candidate.counts.true_positive,
                  candidate.counts.false_positive,
                  candidate.counts.false_negative,
                  candidate.counts.true_negative)
        assert actual == expected, (
            f"{name}, candidate {index} at threshold {candidate.threshold} "
            f"({candidate.operator.value}): sweep says {actual}, the definition "
            f"says {expected}")


def test_the_empty_candidate_exists_and_flags_nothing():
    """THE OPERATING POINT THE GRID COULD NOT EXPRESS.

    Flagging nothing needs a threshold ABOVE every score, unrepresentable in
    [0, 1] when the maximum is 1.0. GREATER at the maximum expresses it, and
    `np.linspace(0, 1, 1000)` cannot.
    """
    y, p = np.array([1., 0., 1., 0.]), np.array([1.0, 0.7, 0.5, 0.3])
    sweep = sweep_thresholds(y, p, population=_population(4))

    first = sweep[0]
    assert first.operator is ThresholdOperator.GREATER
    assert first.threshold == pytest.approx(1.0)
    assert first.counts.n_flagged == 0
    assert first.counts.n_cleared == 4


def test_the_candidate_domain_is_k_plus_one():
    """k unique scores, plus the empty candidate. NOT both operators at every
    score: for adjacent distinct values `p > s` and `p >= s'` induce the SAME
    partition, so enumerating both would duplicate candidates."""
    y = np.array([1., 0., 1., 0., 1.])
    p = np.array([0.9, 0.9, 0.5, 0.5, 0.1])
    sweep = sweep_thresholds(y, p, population=_population(5))
    assert len(sweep) == np.unique(p).size + 1 == 4


def test_every_candidate_declares_the_sweep_as_its_source():
    """THR-1b exists for this. A candidate is not a selection: every point the
    sweep enumerates carries EVALUATION_SWEEP, and at most one is ever chosen."""
    y, p = _COHORTS["all_distinct"]
    sweep = sweep_thresholds(y, p, population=_population(4))
    for index in range(len(sweep)):
        assert sweep[index].parameters.source is ThresholdSource.EVALUATION_SWEEP


def test_the_actual_class_totals_are_threshold_invariant():
    """D10: the legacy sweep recomputed these INSIDE the loop. Here they are
    computed once, because there is no loop."""
    y = np.array([1., 0., 1., 0., 1.])
    p = np.array([0.9, 0.8, 0.5, 0.4, 0.1])
    sweep = sweep_thresholds(y, p, population=_population(5))

    assert sweep.n_actual_positive == 3
    assert sweep.n_actual_negative == 2
    for index in range(len(sweep)):
        counts = sweep[index]. counts
        assert counts.n_actual_positive == 3
        assert counts.n_actual_negative == 2
        assert counts.n == 5


def test_the_candidates_run_from_conservative_to_permissive():
    """Both legacy selectors walk in this direction, and step 5's shadow
    comparison is far easier to read if the new sweep does too."""
    y, p = _COHORTS["all_distinct"]
    sweep = sweep_thresholds(y, p, population=_population(4))
    flagged = [sweep[i].counts.n_flagged for i in range(len(sweep))]
    assert flagged == sorted(flagged)
    assert flagged[0] == 0
    assert flagged[-1] == 4


def test_the_sweep_is_exact_where_a_thousand_point_grid_is_not():
    """D1, demonstrated rather than asserted.

    On scores spaced at 0.0001 -- finer than the grid's 1/999 step -- the exact
    sweep finds FIVE distinct operating points and the grid reaches TWO. Three of
    five are unreachable by construction.
    """
    y = np.array([1., 0., 1., 0.])
    p = np.array([0.5000, 0.5001, 0.5002, 0.5003])
    sweep = sweep_thresholds(y, p, population=_population(4))

    flagged = [sweep[i].counts.n_flagged for i in range(len(sweep))]
    assert flagged == [0, 1, 2, 3, 4]

    grid_reachable = {int((p >= t).sum()) for t in np.linspace(0.0, 1.0, 1000)}
    assert grid_reachable == {0, 4}, (
        f"the grid reaches {sorted(grid_reachable)}; this fixture no longer "
        "demonstrates its inexactness")


# --------------------------------------------------------------------------- #
# STORAGE: owned immutable arrays, and no object per candidate
# --------------------------------------------------------------------------- #
def test_the_arrays_are_owned_and_read_only():
    """A sweep is EVIDENCE. Evidence that can change after the fact is not
    evidence, so mutating the input afterwards must not alter what it reports."""
    y = np.array([1., 0., 1., 0.])
    p = np.array([0.9, 0.7, 0.5, 0.3])
    sweep = sweep_thresholds(y, p, population=_population(4))

    before = sweep.thresholds.copy()
    p[0] = 0.1                                    # mutate the caller's array
    assert np.array_equal(sweep.thresholds, before)

    with pytest.raises(ValueError):
        sweep.thresholds[0] = 0.0


def test_no_object_is_stored_per_candidate():
    """The array backing exists to avoid one frozen dataclass per candidate. A
    1.5-million-row cohort with distinct scores has 1.5 million achievable
    thresholds; materialising them would cost more than the sweep did."""
    rng = np.random.default_rng(20260804)
    y = (rng.random(2000) < 0.4).astype(float)
    p = rng.random(2000)
    sweep = sweep_thresholds(y, p, population=_population(2000))

    assert len(sweep) == 2001
    for slot in getattr(type(sweep), "__slots__", ()):
        held = getattr(sweep, slot)
        assert not isinstance(held, (list, tuple)), (
            f"{slot} holds a {type(held).__name__}; the sweep must not store a "
            "sequence of per-candidate objects")


def test_array_bytes_scale_linearly_with_the_candidate_count():
    """Doubling the distinct scores must roughly double the bytes held, not
    square them."""
    rng = np.random.default_rng(20260804)
    sizes = {}
    for n in (500, 1000, 2000):
        y = (rng.random(n) < 0.5).astype(float)
        p = rng.random(n)
        sizes[n] = sweep_thresholds(
            y, p, population=_population(n)).array_bytes()

    assert 1.6 < sizes[1000] / sizes[500] < 2.4
    assert 1.6 < sizes[2000] / sizes[1000] < 2.4


def test_slicing_is_refused_rather_than_silently_expensive():
    """Slicing would materialise one object per candidate -- exactly the cost the
    array backing avoids. It refuses and says where to go instead."""
    y, p = _COHORTS["all_distinct"]
    sweep = sweep_thresholds(y, p, population=_population(4))
    with pytest.raises(TypeError, match="index the arrays directly"):
        _ = sweep[1:3]


def test_the_sweep_carries_its_population():
    """POP-1b: a count without its population says nothing about WHICH rows
    produced it. Carried from the first type rather than attached later."""
    y, p = _COHORTS["all_distinct"]
    population = _population(4)
    sweep = sweep_thresholds(y, p, population=population)
    assert sweep.population is population
    assert sweep.population.n == 4


# --------------------------------------------------------------------------- #
# REFUSALS: raise, never filter and continue
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_score_raises_rather_than_becoming_a_negative(bad):
    """`p >= t` evaluates FALSE for a NaN. Measured in evaluator.py: that moved
    an operating point from sensitivity 0.90 to 0.50, with no exception and no
    warning."""
    y = np.array([1., 0., 1., 0.])
    p = np.array([0.9, bad, 0.5, 0.3])
    with pytest.raises(ValueError, match="non-finite"):
        sweep_thresholds(y, p, population=_population(4))


def test_an_empty_cohort_raises_rather_than_returning_an_empty_sweep():
    """An empty sweep would read as 'no operating point was suitable' when the
    truth is that none was examined."""
    with pytest.raises(ValueError, match="no achievable thresholds"):
        sweep_thresholds(np.array([]), np.array([]), population=None)


def test_scores_outside_the_probability_range_raise():
    y = np.array([1., 0.])
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        sweep_thresholds(y, np.array([1.5, 0.5]), population=_population(2))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="rows"):
        sweep_thresholds(np.array([1., 0., 1.]), np.array([0.9, 0.5]),
                         population=None)


def test_a_label_that_is_not_zero_or_one_raises():
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        sweep_thresholds(np.array([1., 2., 0.]), np.array([0.9, 0.5, 0.1]),
                         population=_population(3))


def test_a_population_of_the_wrong_size_raises():
    """The counts must describe the cohort the population declares, or they
    describe a cohort nobody declared."""
    y, p = _COHORTS["all_distinct"]
    with pytest.raises(ValueError, match="rows"):
        sweep_thresholds(y, p, population=_population(7))


def test_confusion_counts_refuses_a_rate_stored_as_a_count():
    """A cardinality is an integer. A float here means something computed a rate
    and stored it where a count belongs -- which is how D2-D5 begin."""
    with pytest.raises(ValueError, match="non-negative integer"):
        ConfusionCounts(true_positive=0.75, false_positive=1,
                        false_negative=1, true_negative=1)
