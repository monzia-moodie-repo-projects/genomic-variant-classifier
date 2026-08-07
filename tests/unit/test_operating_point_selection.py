"""OP-1 step 4: the operating-point selectors, and the closure of D12.

Author: Monzia Moodie
Project: genomic-variant-classifier
Written 2026-08-06.

D12, recorded 2026-08-01: `if diff < best_diff` is strict, so the FIRST
candidate achieving the minimum wins and the answer depends on TRAVERSAL
DIRECTION. The two legacy selectors traverse OPPOSITE ways -- `linspace`
ascending is permissive to conservative, `np.sort(np.unique(p))[::-1]` is
conservative to permissive -- and the exact sweep runs conservative to
permissive. A selector written naturally against the sweep would have INVERTED
the sensitivity-target tie policy while appearing to preserve it.

So D12 does not close by documenting whichever rule falls out. It closes when
the policy is TYPED, PERSISTED, TESTED, and INDEPENDENT OF SWEEP ORDER. The
order-invariance battery below is the load-bearing test in this module.

NOTHING HERE IS WIRED. Step 5 shadows these against the legacy selectors; step
6 cuts over. `evaluator.py` still calls `_find_operating_point` twice and
`_find_high_ppv_point` once, and this commit does not touch them.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.population import (
    EvaluationPopulation, PopulationComparison)
from genomic_variant_classifier.evaluation.thresholds import (
    ExactThresholdSweep,
    MetricStatus,
    OperatingPointCertificationBlocker,
    OperatingPointObjective,
    OperatingPointOutcome,
    OperatingPointSelection,
    OperatingPointSelectionStatus,
    OperatingPointTieBreak,
    ThresholdOperator,
    _OPERATOR_CONSERVATIVENESS,
    _canonical_threshold_key,
    _operator_rank_array,
    metrics_from_counts,
    select_max_sensitivity_at_ppv_floor,
    select_nearest_sensitivity_target,
    sweep_thresholds,
)

_SOURCE_ID = "op1-step4-selection"


def _population(n: int, source_id=_SOURCE_ID) -> EvaluationPopulation:
    return EvaluationPopulation.full(n, scope="step4", source_id=source_id)


def _sweep(y, p, source_id=_SOURCE_ID) -> ExactThresholdSweep:
    y = np.asarray(y, dtype=float)
    return sweep_thresholds(y, np.asarray(p, dtype=float),
                            population=_population(len(y), source_id))


def _permuted(sweep: ExactThresholdSweep, order) -> ExactThresholdSweep:
    """The SAME candidate set, presented in a different order.

    `ExactThresholdSweep` is not a dataclass -- it has __slots__ -- so
    `dataclasses.replace` raises. A new sweep is constructed from permuted
    arrays, which is also the first direct construction anywhere in the suite:
    every other sweep comes from `sweep_thresholds`.
    """
    order = np.asarray(order)
    return ExactThresholdSweep(
        thresholds=sweep.thresholds[order],
        strictly_greater=sweep.strictly_greater[order],
        true_positive=sweep.true_positive[order],
        false_positive=sweep.false_positive[order],
        n_actual_positive=sweep.n_actual_positive,
        n_actual_negative=sweep.n_actual_negative,
        population=sweep.population)


def _chosen(outcome):
    """What a caller compares: the declaration and the counts behind it."""
    if not outcome.is_ok:
        return ("REFUSED", outcome.status.value, outcome.reason)
    counts = outcome.metrics.counts
    return ("OK", outcome.metrics.parameters.threshold,
            outcome.metrics.parameters.operator.value,
            counts.true_positive, counts.false_positive)


# A cohort with DUPLICATE SCORES and overlapping classes, so ties are real
# rather than hypothetical: five positives, five negatives, three tied pairs.
_MIXED_Y = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
_MIXED_P = [0.9, 0.9, 0.8, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1]

# A SEPARABLE cohort, so a positive-predictive-value floor is reachable. On the
# mixed cohort the maximum achievable value is 0.667, so a 0.80 floor is
# correctly refused there -- which is why that fixture cannot test a success.
_SEPARABLE_Y = [1, 1, 1, 0, 0, 0]
_SEPARABLE_P = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]


# --------------------------------------------------------------------------- #
# THE THREE VOCABULARIES, GATED IN BOTH DIRECTIONS
#
# CERT-1 (2026-08-05) added a member to a vocabulary that had no completeness
# gate; THR-1b built one for `ThresholdSource` the day before. These codes are
# persisted beside a chosen threshold, so a stealth addition, removal or RENAME
# must each fail here.
# --------------------------------------------------------------------------- #

_EXPECTED_OBJECTIVES = {
    "NEAREST_SENSITIVITY_TARGET": "nearest_sensitivity_target",
    "MAXIMIZE_SENSITIVITY_AT_PPV_FLOOR":
        "maximize_sensitivity_at_positive_predictive_value_floor",
}

_EXPECTED_TIE_BREAKS = {
    "TARGET_HIGHER_SENSITIVITY_FEWER_FLAGGED":
        "target_higher_sensitivity_fewer_flagged",
    "MAX_SENSITIVITY_FEWER_FALSE_POSITIVES":
        "max_sensitivity_fewer_false_positives",
}

_EXPECTED_SELECTION_STATUSES = {
    "SELECTED": "selected",
    "NO_FEASIBLE_CANDIDATE": "no_feasible_candidate",
    "OBJECTIVE_NOT_APPLICABLE": "objective_not_applicable",
}


@pytest.mark.parametrize("enum_type,expected", [
    (OperatingPointObjective, _EXPECTED_OBJECTIVES),
    (OperatingPointTieBreak, _EXPECTED_TIE_BREAKS),
    (OperatingPointSelectionStatus, _EXPECTED_SELECTION_STATUSES),
], ids=["objective", "tie_break", "selection_status"])
def test_the_selection_vocabulary_is_exactly_this(enum_type, expected):
    actual = {member.name: member.value for member in enum_type}
    assert actual == expected, (
        f"the {enum_type.__name__} vocabulary changed.\n"
        f"  expected: {expected}\n  actual  : {actual}\n"
        "These codes travel with a chosen threshold. If a member was added "
        "deliberately, add it here in the same commit; if a VALUE changed, "
        "stop -- a persisted policy string cannot be renamed silently.")


def test_no_tie_break_name_promises_a_stage_that_cannot_run():
    """The names describe criteria that ACTUALLY EXECUTE.

    A four-stage draft ended each key with the canonical threshold declaration
    and named it in the persisted code. Sabotage could not detect removing it,
    because on a canonical sweep `n_flagged` is strictly increasing, so "fewer
    flagged" and "most conservative threshold" are the SAME order. A policy
    string naming an unreachable stage overstates what chose the candidate.
    """
    for member in OperatingPointTieBreak:
        assert "conservative" not in member.value, (
            f"{member.name} names conservativeness, which cannot decide "
            "anything once fewer-flagged or fewer-false-positives has run")


# --------------------------------------------------------------------------- #
# CANONICAL SWEEP ORDERING -- a property of the SWEEP, not a selector key
# --------------------------------------------------------------------------- #

def test_strict_operator_precedes_inclusive_at_the_same_threshold():
    """Step 1's canonical domain contains the maximum score TWICE:
    `(max p, GREATER)` and `(max p, GREATER_OR_EQUAL)`. At one threshold
    `p > t` flags a strict SUBSET of `p >= t`, so GREATER is the more
    conservative declaration and must rank first.
    """
    maximum = float(max(_MIXED_P))
    assert (_canonical_threshold_key(maximum, ThresholdOperator.GREATER)
            < _canonical_threshold_key(maximum,
                                       ThresholdOperator.GREATER_OR_EQUAL))


def test_the_canonical_key_totally_orders_the_sweep():
    """`-threshold` ALONE is not a total order: the maximum score appears
    twice. The operator rank is what makes the key total, and this asserts it
    over every candidate rather than over the two that motivated it."""
    sweep = _sweep(_MIXED_Y, _MIXED_P)
    keys = [_canonical_threshold_key(
        float(sweep.thresholds[i]),
        ThresholdOperator.GREATER if bool(sweep.strictly_greater[i])
        else ThresholdOperator.GREATER_OR_EQUAL) for i in range(len(sweep))]
    assert len(set(keys)) == len(keys)
    assert keys == sorted(keys), (
        "sweep_thresholds must emit candidates in canonical order, most "
        "conservative first")


def test_the_vectorised_operator_rank_equals_the_declared_mapping():
    """Two encodings of one ordering is the SWEEP-1 shape. The mapping is the
    semantics; the array form is an optimisation and must agree with it."""
    sweep = _sweep(_MIXED_Y, _MIXED_P)
    declared = np.array([
        _OPERATOR_CONSERVATIVENESS[
            ThresholdOperator.GREATER if bool(flag)
            else ThresholdOperator.GREATER_OR_EQUAL]
        for flag in sweep.strictly_greater])
    assert (_operator_rank_array(sweep.strictly_greater) == declared).all()


def test_the_sweep_flagged_count_strictly_increases():
    """Candidate 0 flags nothing; candidate 1 flags every row at the maximum
    score; each later candidate adds at least one more. This is what makes
    "fewer flagged" and "most conservative" the same order -- measured here so
    the simplification of the tie-break rests on an asserted property."""
    sweep = _sweep(_MIXED_Y, _MIXED_P)
    assert np.all(np.diff(np.asarray(sweep.n_flagged)) > 0)


# --------------------------------------------------------------------------- #
# THE ONE ENFORCED INVARIANT: DISTINCT CONFUSION STATES
# --------------------------------------------------------------------------- #

def test_the_sweep_has_unique_confusion_states():
    """The single invariant both selector keys need.

    `(distance, -TP, n_flagged)` ties on the first two exactly when two
    candidates share TP, and the third differs exactly when they differ in FP.
    `(-TP, FP)` is total for the same reason. So distinct `(TP, FP)` pairs make
    BOTH keys total, and no canonicalising suffix is reachable.
    """
    sweep = _sweep(_MIXED_Y, _MIXED_P)
    pairs = list(zip(sweep.true_positive.tolist(),
                     sweep.false_positive.tolist()))
    assert len(set(pairs)) == len(pairs)


def test_a_sweep_with_duplicate_confusion_states_is_refused():
    """Enforced in the TYPE, not assumed of the producer.

    Two candidates with identical counts would leave both selector keys tied,
    and `np.lexsort` is stable -- so the winner would fall back to ARRAY ORDER,
    which is D12 reopening through the back door. Measured 2026-08-06: no test
    in the suite constructs a sweep directly, and `sweep_thresholds` produced
    no violation across 300 cohorts with duplicate scores forced.
    """
    with pytest.raises(ValueError,
                       match="one candidate per confusion state"):
        ExactThresholdSweep(
            thresholds=[0.9, 0.8], strictly_greater=[False, False],
            true_positive=[1, 1], false_positive=[0, 0],
            n_actual_positive=2, n_actual_negative=2,
            population=_population(4))


def test_equal_thresholds_with_distinct_operators_are_accepted():
    """The constructor must not confuse a duplicate THRESHOLD VALUE with a
    duplicate CONFUSION STATE. Step 1's canonical domain contains the maximum
    score twice -- `(max p, GREATER)` and `(max p, GREATER_OR_EQUAL)` -- and
    those classify differently, and they are the pair that exposed the
    canonical-key defect."""
    sweep = ExactThresholdSweep(
        thresholds=[0.9, 0.9], strictly_greater=[True, False],
        true_positive=[0, 1], false_positive=[0, 0],
        n_actual_positive=2, n_actual_negative=2, population=_population(4))
    assert len(sweep) == 2


def test_a_permuted_sweep_is_accepted():
    """The invariant is a property of the SET, so permutation cannot violate
    it. Strictly-increasing flagged counts would have been an ORDERING
    property, and enforcing that would reject the order-invariance battery's
    own inputs."""
    sweep = _sweep(_MIXED_Y, _MIXED_P)
    reversed_sweep = _permuted(sweep, np.arange(len(sweep))[::-1])
    assert len(reversed_sweep) == len(sweep)
    assert set(reversed_sweep.true_positive.tolist()) == set(
        sweep.true_positive.tolist())


# --------------------------------------------------------------------------- #
# ORDER INVARIANCE -- the load-bearing test, and the closure of D12
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(20))
def test_selection_is_invariant_to_candidate_order(seed):
    """The sweep may be presented in ANY order and the answer is the same.

    This is what D12 asks for. A first-minimum rule would return a different
    candidate under a different traversal, and the two legacy selectors
    traverse in OPPOSITE directions -- so a selector written naturally against
    the exact sweep would have silently inverted the sensitivity-target tie
    policy. Twenty deterministic permutations are cheap and they are the only
    thing standing between a declared policy and a rediscovered accident.
    """
    sweep = _sweep(_MIXED_Y, _MIXED_P)
    expected = (_chosen(select_nearest_sensitivity_target(sweep, target=0.90)),
                _chosen(select_max_sensitivity_at_ppv_floor(
                    sweep, positive_predictive_value_floor=0.60)))

    order = np.random.default_rng(seed).permutation(len(sweep))
    shuffled = _permuted(sweep, order)
    assert (_chosen(select_nearest_sensitivity_target(shuffled, target=0.90)),
            _chosen(select_max_sensitivity_at_ppv_floor(
                shuffled, positive_predictive_value_floor=0.60))) == expected


def test_selection_is_invariant_to_a_fully_reversed_sweep():
    """The reversal is called out separately because it is the ACTUAL legacy
    disagreement: `_find_operating_point` walks permissive to conservative and
    the exact sweep walks the other way."""
    sweep = _sweep(_MIXED_Y, _MIXED_P)
    reversed_sweep = _permuted(sweep, np.arange(len(sweep))[::-1])
    assert (_chosen(select_nearest_sensitivity_target(sweep, target=0.90))
            == _chosen(select_nearest_sensitivity_target(reversed_sweep,
                                                         target=0.90)))


# --------------------------------------------------------------------------- #
# THE DECLARED TIE-BREAKS, EACH EXERCISED
# --------------------------------------------------------------------------- #

def test_the_target_selector_prefers_higher_sensitivity_on_a_symmetric_tie():
    """Five positives, so sensitivities land on multiples of 0.2. Against a
    target of 0.90, sensitivity 0.8 and sensitivity 1.0 are EQUALLY distant --
    and the declared policy prefers the higher, which is the one that misses
    fewer true cases."""
    sweep = _sweep(_MIXED_Y, _MIXED_P)
    outcome = select_nearest_sensitivity_target(sweep, target=0.90)
    assert outcome.is_ok
    assert outcome.metrics.counts.true_positive == 5, (
        "an equal-distance tie must go to the higher sensitivity, not to "
        "whichever candidate the sweep happened to present first")


def test_the_target_selector_prefers_fewer_flagged_at_equal_sensitivity():
    """Every candidate recovers the single positive, so the objective and the
    higher-sensitivity rule both tie across all of them. Only "fewer flagged"
    can decide, and it prefers the candidate sending fewer rows downstream."""
    sweep = _sweep([1, 0, 0], [0.9, 0.8, 0.7])
    outcome = select_nearest_sensitivity_target(sweep, target=1.0)
    assert outcome.is_ok
    assert outcome.metrics.counts.false_positive == 0
    assert outcome.metrics.counts.n_flagged == 1


def test_objective_a_is_not_the_legacy_conservative_prefix():
    """THE FIXTURE OP-0 MEASURED, on 2026-08-04.

    y = [1, 0, 1], p = [0.9, 0.8, 0.7], floor 0.60:

        t=0.90   ppv 1.0000   sensitivity 0.5000   feasible, legacy picks this
        t=0.80   ppv 0.5000   sensitivity 0.5000   violates the floor
        t=0.70   ppv 0.6667   sensitivity 1.0000   feasible, legacy never sees

    The legacy break fires at 0.80 and returns the last preceding candidate.
    Objective A evaluates the WHOLE feasible set and takes the most sensitive
    member, which satisfies the same floor at DOUBLE the sensitivity. This test
    is the difference step 5 will shadow.
    """
    sweep = _sweep([1, 0, 1], [0.9, 0.8, 0.7])
    outcome = select_max_sensitivity_at_ppv_floor(
        sweep, positive_predictive_value_floor=0.60)
    assert outcome.is_ok
    assert outcome.metrics.counts.true_positive == 2
    assert outcome.metrics.parameters.threshold == pytest.approx(0.7)


def test_objective_a_prefers_fewer_false_positives_at_equal_sensitivity():
    sweep = _sweep([1, 1, 0, 1], [0.9, 0.8, 0.75, 0.7])
    outcome = select_max_sensitivity_at_ppv_floor(
        sweep, positive_predictive_value_floor=0.70)
    assert outcome.is_ok
    assert outcome.metrics.counts.true_positive == 3
    assert outcome.metrics.counts.false_positive == 1


# --------------------------------------------------------------------------- #
# REFUSALS -- typed, with the stage that failed recorded
# --------------------------------------------------------------------------- #

def test_a_zero_positive_cohort_is_refused_in_the_registry_vocabulary():
    """Sensitivity is TP / P and P is zero, so no candidate has a sensitivity
    to be near the target. The legacy sweep wrote `if n_pos == 0: continue` and
    returned a bare None, which said nothing about why. The reason is the
    registry's, so scalar and composite refusals speak one vocabulary."""
    outcome = select_nearest_sensitivity_target(
        _sweep([0, 0, 0, 0], [0.9, 0.6, 0.3, 0.1]), target=0.90)
    assert not outcome.is_ok
    assert outcome.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert outcome.reason == "positive_class_support_required"
    assert outcome.metrics is None
    assert (outcome.selection.status
            is OperatingPointSelectionStatus.OBJECTIVE_NOT_APPLICABLE)
    assert outcome.selection.n_feasible_candidates is None, (
        "feasibility was never evaluated, so a count would be fabricated")


def test_an_undefined_positive_predictive_value_never_satisfies_a_floor():
    """The empty candidate flags nothing, so its positive predictive value is
    UNDEFINED rather than zero. A floor is a claim about a MEASURED value.
    Even a floor of 0.0 must not admit it -- `(value or 0.0) >= floor` would,
    which conflates undefined with measured zero: the D2-D5 defect."""
    sweep = _sweep(_SEPARABLE_Y, _SEPARABLE_P)
    empty = sweep[0]
    assert empty.counts.n_flagged == 0
    metrics = metrics_from_counts(empty.counts, empty.parameters)
    assert metrics.positive_predictive_value.status is MetricStatus.UNDEFINED
    assert (metrics.positive_predictive_value.reason
            == "empty_predicted_positive_set")

    outcome = select_max_sensitivity_at_ppv_floor(
        sweep, positive_predictive_value_floor=0.0)
    assert outcome.is_ok
    assert outcome.metrics.counts.n_flagged > 0
    assert outcome.selection.n_feasible_candidates == len(sweep) - 1, (
        "every candidate but the empty one has a defined value at a floor of "
        "zero; admitting the empty one would show up here")


def test_no_feasible_candidate_is_a_typed_refusal_recording_the_floor():
    outcome = select_max_sensitivity_at_ppv_floor(
        _sweep([1, 0, 0, 0], [0.9, 0.9, 0.9, 0.9]),
        positive_predictive_value_floor=0.99)
    assert not outcome.is_ok
    assert outcome.status is MetricStatus.NOT_APPLICABLE
    assert (outcome.reason
            == "no_candidate_satisfies_positive_predictive_value_floor")
    assert (outcome.selection.status
            is OperatingPointSelectionStatus.NO_FEASIBLE_CANDIDATE)
    assert outcome.selection.n_feasible_candidates == 0
    assert outcome.selection.target_value == pytest.approx(0.99), (
        "a 'no solution' result is only interpretable beside the floor that "
        "produced it")


@pytest.mark.parametrize("target", [-0.01, 1.01, float("nan")])
def test_a_target_outside_the_unit_interval_is_refused(target):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_nearest_sensitivity_target(_sweep(_MIXED_Y, _MIXED_P),
                                          target=target)


def test_a_sweep_without_a_population_cannot_produce_a_selection():
    """CERT-1: an OK outcome requires an EvaluationPopulation. Refusing here
    names the cause instead of failing later inside the constructor."""
    sweep = sweep_thresholds(np.asarray(_MIXED_Y, dtype=float),
                             np.asarray(_MIXED_P, dtype=float),
                             population=None)
    with pytest.raises(ValueError, match="EvaluationPopulation"):
        select_nearest_sensitivity_target(sweep, target=0.90)


# --------------------------------------------------------------------------- #
# CERTIFICATION -- selection optimism, named rather than implied
# --------------------------------------------------------------------------- #

def test_a_selected_outcome_is_never_certifiable_at_step_4():
    """The threshold is chosen on the rows its performance is reported over,
    and no held-out validation exists. Both block certification, and both are
    NAMED -- a bare `certification_eligible: false` beside an empty blocker
    list would leave a reader unable to see why."""
    outcome = select_max_sensitivity_at_ppv_floor(
        _sweep(_SEPARABLE_Y, _SEPARABLE_P),
        positive_predictive_value_floor=0.80)
    assert outcome.is_ok
    assert not outcome.certification_eligible
    assert set(outcome.certification_blockers) == {
        OperatingPointCertificationBlocker
        .SAME_POPULATION_SELECTION_AND_EVALUATION,
        OperatingPointCertificationBlocker
        .POST_SELECTION_VALIDATION_NOT_IMPLEMENTED}


def test_an_unprovable_independence_gets_its_own_blocker():
    """`compare_membership` is THREE-VALUED. An unattributed population has no
    fingerprint, so independence is UNPROVEN rather than absent -- and marking
    it SAME would assert a sameness the system cannot establish."""
    sweep = _sweep(_SEPARABLE_Y, _SEPARABLE_P, source_id=None)
    assert (sweep.population.compare_membership(sweep.population)
            is PopulationComparison.UNKNOWN)

    outcome = select_max_sensitivity_at_ppv_floor(
        sweep, positive_predictive_value_floor=0.80)
    assert outcome.is_ok
    assert (OperatingPointCertificationBlocker
            .SELECTION_EVALUATION_INDEPENDENCE_NOT_ESTABLISHED
            in outcome.certification_blockers)
    assert (OperatingPointCertificationBlocker
            .SAME_POPULATION_SELECTION_AND_EVALUATION
            not in outcome.certification_blockers)


# --------------------------------------------------------------------------- #
# THE SELECTION RECORD -- what closes D12 in the ARTIFACT
# --------------------------------------------------------------------------- #

def test_a_selected_outcome_states_the_policy_that_chose_it():
    outcome = select_nearest_sensitivity_target(
        _sweep(_MIXED_Y, _MIXED_P), target=0.90)
    selection = outcome.selection
    assert (selection.objective
            is OperatingPointObjective.NEAREST_SENSITIVITY_TARGET)
    assert (selection.tie_break
            is OperatingPointTieBreak.TARGET_HIGHER_SENSITIVITY_FEWER_FLAGGED)
    assert selection.target_value == pytest.approx(0.90)
    assert selection.status is OperatingPointSelectionStatus.SELECTED
    assert selection.n_candidates == len(_sweep(_MIXED_Y, _MIXED_P))
    assert 0 <= selection.selected_index < selection.n_candidates
    assert selection.selected_index is not None


@pytest.mark.parametrize("kwargs,match", [
    (dict(status=OperatingPointSelectionStatus.SELECTED,
          n_feasible_candidates=3, selected_index=None), "must name"),
    (dict(status=OperatingPointSelectionStatus.NO_FEASIBLE_CANDIDATE,
          n_feasible_candidates=2, selected_index=None), "requires"),
    (dict(status=OperatingPointSelectionStatus.OBJECTIVE_NOT_APPLICABLE,
          n_feasible_candidates=0, selected_index=None), "NEVER"),
    (dict(status=OperatingPointSelectionStatus.NO_FEASIBLE_CANDIDATE,
          n_feasible_candidates=0, selected_index=1), "must not name"),
], ids=["selected_without_index", "no_feasible_with_count",
        "not_applicable_with_count", "refused_with_index"])
def test_the_selection_record_refuses_an_incoherent_state(kwargs, match):
    """A record that can describe a selection that never happened is a record
    a reader cannot trust. Each of these is refused at construction."""
    with pytest.raises(ValueError, match=match):
        OperatingPointSelection(
            objective=OperatingPointObjective.NEAREST_SENSITIVITY_TARGET,
            tie_break=(OperatingPointTieBreak
                       .TARGET_HIGHER_SENSITIVITY_FEWER_FLAGGED),
            target_value=0.9, n_candidates=5, **kwargs)


def test_both_a_selection_and_a_refusal_serialise_their_policy():
    """D12 is not closed by implementing a rule; it is closed by an artifact
    that STATES it. A refusal carries no metrics, so the policy cannot ride
    there -- which is why `selection` sits on the outcome."""
    chosen = select_nearest_sensitivity_target(
        _sweep(_MIXED_Y, _MIXED_P), target=0.90).to_dict()
    assert chosen["selection"]["tie_break"] == (
        "target_higher_sensitivity_fewer_flagged")
    assert chosen["selection"]["status"] == "selected"

    refused = select_max_sensitivity_at_ppv_floor(
        _sweep([1, 0, 0, 0], [0.9, 0.9, 0.9, 0.9]),
        positive_predictive_value_floor=0.99).to_dict()
    assert refused["metrics"] is None
    assert refused["selection"]["status"] == "no_feasible_candidate"
    assert refused["selection"]["target_value"] == pytest.approx(0.99)


def test_an_outcome_without_a_selection_still_serialises():
    """`selection` is ADDITIVE and optional. Step 2 created this type before
    selectors existed, and other construction paths may represent an operating
    point that was never chosen by one."""
    outcome = OperatingPointOutcome.refused(
        MetricStatus.NOT_APPLICABLE, "constructed_without_a_selector")
    assert outcome.selection is None
    assert outcome.to_dict()["selection"] is None
