"""Oracle C1: the count path reproduces the registry's scientific semantics.

OP-1 step 3a, 2026-08-05. A VERIFICATION COMMIT -- no production code changes.

WHAT C1 PROVES. For the six estimands both paths compute, `metrics_from_counts`
and `registry.compute` agree on STATUS, REASON and VALUE at the registry's
canonical threshold, on cohorts chosen to exercise every applicability regime.

WHAT C1 DOES NOT PROVE. Full result identity -- metadata, support counts,
population keys, certification eligibility, the serialised form. `registry.compute`
ENRICHES and FINALISES results; `metrics_from_counts` builds plain ones. That is
Oracle C2, and it is deliberately a separate measurement: making it pass by
copying registry metadata into the count path would recreate a second
implementation of the finalisation contract.

WHY ONE THRESHOLD. Measured 2026-08-05: of 24 registered descriptors, NINE carry
a `ThresholdParameters` and all nine carry `(0.5, GREATER_OR_EQUAL,
fixed_default)`. So every fixture here must contain a score EXACTLY equal to 0.5
-- the sweep's candidates are the observed score values, and without one there is
no candidate carrying the registry's canonical parameters.

Author: Monzia Moodie
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.capabilities import MetricStatus
from genomic_variant_classifier.evaluation.population import EvaluationPopulation
from genomic_variant_classifier.evaluation.thresholds import (
    ThresholdOperator,
    metrics_from_counts,
    sweep_thresholds,
)

# --------------------------------------------------------------------------- #
# THE SURFACE PARTITION, CODIFIED
#
# Every omission below is a SCOPE DECISION with a reason, not an accidental gap.
# Left in prose, a future reader could add a metric to one surface without the
# other and create a silent asymmetry; asserted, the asymmetry fails a test.
# --------------------------------------------------------------------------- #

# Computed by BOTH paths. These are what C1 compares.
SHARED_ESTIMANDS = frozenset({
    "sensitivity",
    "specificity",
    "positive_predictive_value",
    "negative_predictive_value",
    "f1",
    "matthews_correlation_coefficient",
})

# On the operating point only. No registry descriptor exists, so it cannot
# participate in Oracle C -- a different estimand surface, not a gap.
OPERATING_POINT_ONLY = frozenset({"flagged_fraction"})

# Threshold-carrying registry metrics DELIBERATELY not duplicated in the count
# path. The formulas are trivial; the applicability, status, reason strings and
# unbounded-value policy are not. `_requires_interior_specificity` already
# decides that a positive likelihood ratio with specificity at 1.0 is UNDEFINED
# with reason `likelihood_ratio_unbounded` -- reimplementing the formula would
# create a second authority for that scientific decision.
REGISTRY_ONLY_THRESHOLD = frozenset({
    "balanced_accuracy",
    "positive_likelihood_ratio",
    "negative_likelihood_ratio",
})

# A POPULATION statistic (Decision 3, 2026-08-04): it does not depend on the
# threshold, on predicted-positive membership, or on the sweep.
REGISTRY_ONLY_POPULATION = frozenset({"prevalence"})


def _metric_context(*, y, probabilities, scope):
    """The REG-2 construction pattern, reused rather than re-invented.

    All three arrays are supplied with `y_score` and `y_prob` bound to the same
    values, so a confusion descriptor thresholding `y_prob` and a ranking
    descriptor consuming `y_score` both work from one context. That is the form
    `test_metric_registry.py` established; coining a second would be the SWEEP-1
    shape applied to test scaffolding.
    """
    import genomic_variant_classifier.evaluation.registry as registry_module

    population = EvaluationPopulation.full(
        y.size, scope=scope, source_id=f"op1-oracle-c1:{scope}")
    return registry_module.MetricContext(
        y_true=y, y_prob=probabilities, y_score=probabilities,
        population=population)


def _candidate_at_registry_threshold(sweep):
    """The one sweep candidate carrying the registry's canonical parameters.

    FAILS LOUDLY if absent. A fixture without a score of exactly 0.5 produces a
    sweep with no such candidate, and the comparison would then be IMPOSSIBLE
    rather than merely empty -- a distinction worth an explicit failure.

    Uses `len(sweep)`, not `sweep.n_candidates`: the adopted design's
    illustration named the latter and `ExactThresholdSweep` implements
    `__len__`. An adopted design's code sketches specify intent, not API.
    """
    matches = [
        sweep[index] for index in range(len(sweep))
        if (sweep[index].parameters.threshold == 0.5
            and sweep[index].parameters.operator
            is ThresholdOperator.GREATER_OR_EQUAL)]

    assert len(matches) == 1, (
        f"the Oracle C fixture must expose exactly one (0.5, >=) candidate, "
        f"found {len(matches)}. Every C1 cohort must contain a score of exactly "
        "0.5, because the sweep's candidates are the observed score values.")
    return matches[0]


# Seven cohorts, each pinning a distinct applicability regime at (0.5, >=).
# EVERY ONE CONTAINS 0.5.
_FIXTURES = {
    "mixed_all_margins_nonzero": (
        np.array([1., 0., 1., 0., 1., 0.]),
        np.array([0.9, 0.5, 0.7, 0.2, 0.5, 0.1])),
    # `no_predicted_positives` IS ABSENT, AND CANNOT BE PRESENT. Nothing flagged
    # at (0.5, >=) requires every score below 0.5; a comparable candidate
    # requires a score of exactly 0.5. Under GREATER_OR_EQUAL those are
    # contradictory, because 0.5 satisfies `p >= 0.5`. The regime is not missing
    # from this matrix -- it is UNEXPRESSIBLE at the registry's threshold, and
    # `test_the_empty_predicted_positive_regime_is_unreachable_at_the_canonical_threshold`
    # asserts that rather than leaving it in a comment.
    #
    # The refusal itself is already pinned at the count level, by
    # `test_an_empty_predicted_positive_set_refuses_rather_than_reporting_zero`
    # in step 2. What C1 cannot do is CORROBORATE it against the registry.
    "no_predicted_negatives": (
        np.array([1., 0., 1., 0.]),
        np.array([0.9, 0.5, 0.7, 0.6])),
    "no_positive_reference_class": (
        np.array([0., 0., 0., 0.]),
        np.array([0.9, 0.5, 0.3, 0.1])),
    "no_negative_reference_class": (
        np.array([1., 1., 1., 1.]),
        np.array([0.9, 0.5, 0.3, 0.1])),
    "zero_confusion_margin": (
        np.array([1., 1., 0., 0.]),
        np.array([0.5, 0.5, 0.5, 0.5])),
    "tied_scores_at_the_threshold": (
        np.array([1., 0., 1., 0.]),
        np.array([0.5, 0.5, 0.9, 0.1])),
}


def _count_path_metrics(y, probabilities):
    """The step-1 sweep plus the step-2 typed record, at the canonical
    threshold."""
    sweep = sweep_thresholds(y, probabilities, population=None)
    candidate = _candidate_at_registry_threshold(sweep)
    return metrics_from_counts(candidate.counts, candidate.parameters)


@pytest.mark.parametrize("fixture_name", sorted(_FIXTURES))
@pytest.mark.parametrize("metric_name", sorted(SHARED_ESTIMANDS))
def test_the_count_path_reproduces_the_registry_status_and_reason(
        fixture_name, metric_name):
    """ORACLE C1. Status, reason and value, on every applicability regime.

    A private vocabulary in the count path would make this compare two dialects
    and report every difference as a defect -- which is why step 2 adopted the
    registry's reason strings and REG-2's status boundary deliberately.
    """
    import genomic_variant_classifier.evaluation.registry as registry_module

    y, probabilities = _FIXTURES[fixture_name]
    ctx = _metric_context(y=y, probabilities=probabilities,
                          scope=f"c1_{fixture_name}")

    from_registry = registry_module.compute(
        registry_module.by_name(metric_name), ctx)
    from_counts = _count_path_metrics(y, probabilities).as_mapping()[metric_name]

    assert from_counts.status is from_registry.status, (
        f"{metric_name} on {fixture_name}: the count path says "
        f"{from_counts.status.value!r} and the registry says "
        f"{from_registry.status.value!r}")

    assert from_counts.reason == from_registry.reason, (
        f"{metric_name} on {fixture_name}: the count path gives reason "
        f"{from_counts.reason!r} and the registry gives {from_registry.reason!r}")

    if from_registry.status is MetricStatus.OK:
        assert from_counts.value == pytest.approx(from_registry.value,
                                                  rel=1e-12, abs=1e-12), (
            f"{metric_name} on {fixture_name}: values disagree")
    else:
        assert math.isnan(from_counts.value), (
            f"{metric_name} on {fixture_name}: a refusal must carry NaN")
        assert math.isnan(from_registry.value)


def test_every_fixture_contains_the_canonical_threshold():
    """THE PRECONDITION OF THE WHOLE ORACLE, asserted rather than assumed.

    Without a score of exactly 0.5 the sweep has no candidate carrying the
    registry's parameters, and C1 would silently compare nothing.
    """
    for name, (_y, probabilities) in _FIXTURES.items():
        assert 0.5 in set(probabilities.tolist()), (
            f"fixture {name!r} contains no score of exactly 0.5, so it cannot "
            "produce a candidate at the registry's canonical threshold")


def test_the_empty_predicted_positive_regime_is_unreachable_at_the_canonical_threshold():
    """WHY ONE REGIME IS ABSENT FROM THE FIXTURE MATRIX, asserted not described.

    Found by the first C1 run (2026-08-05): a `no_predicted_positives` fixture
    failed with "found 0" candidates, because every score had been placed below
    0.5 to make nothing flagged -- which removed the score the comparison
    requires.

    The two requirements are CONTRADICTORY under GREATER_OR_EQUAL:

        a comparable candidate  ->  the cohort must contain exactly 0.5
        nothing flagged at 0.5  ->  every score must be below 0.5

    A score of exactly 0.5 satisfies `p >= 0.5`, so it is flagged. The regime is
    therefore UNEXPRESSIBLE at the registry's threshold, not merely unwritten.

    This asserts the impossibility directly. If it ever becomes false -- the
    registry adopting GREATER, or a descriptor moving off 0.5 -- the regime
    becomes reachable and this test fails, which is the signal to add it back
    DELIBERATELY.
    """
    for name, (_y, probabilities) in _FIXTURES.items():
        flagged = int((probabilities >= 0.5).sum())
        assert flagged >= 1, (
            f"fixture {name!r} flags nothing at (0.5, >=), which contradicts "
            "its containing a score of exactly 0.5")

    # And directly: no cohort containing 0.5 can flag nothing under `>=`.
    for probabilities in (np.array([0.5]), np.array([0.1, 0.5]),
                          np.array([0.5, 0.9]), np.array([0.0, 0.5, 1.0])):
        assert int((probabilities >= 0.5).sum()) >= 1, (
            "a cohort containing exactly 0.5 flagged nothing under >=; the "
            "operator's semantics have changed and the empty-predicted-positive "
            "regime may now be reachable")


def test_the_tied_fixture_actually_exercises_greater_or_equal():
    """The fixture that proves the comparison is not passing by accident.

    On a cohort with no row sitting exactly on the boundary, `>=` and `>` agree
    and the oracle would pass under either. Here two rows carry exactly 0.5, so
    the two operators DISAGREE -- and the candidate must be the `>=` one.
    """
    y, probabilities = _FIXTURES["tied_scores_at_the_threshold"]
    at_or_above = int((probabilities >= 0.5).sum())
    strictly_above = int((probabilities > 0.5).sum())

    assert at_or_above != strictly_above, (
        "this fixture no longer distinguishes >= from >; it cannot prove the "
        "oracle exercises the registry's operator")

    candidate = _candidate_at_registry_threshold(
        sweep_thresholds(y, probabilities, population=None))
    assert candidate.counts.n_flagged == at_or_above


# --------------------------------------------------------------------------- #
# The partition, asserted
# --------------------------------------------------------------------------- #
def test_the_count_path_computes_exactly_the_shared_estimands_plus_its_own():
    """A metric added to `OperatingPointMetrics` without a decision about the
    registry surface fails HERE rather than creating a silent asymmetry."""
    y, probabilities = _FIXTURES["mixed_all_margins_nonzero"]
    computed = set(_count_path_metrics(y, probabilities).as_mapping())

    assert computed == SHARED_ESTIMANDS | OPERATING_POINT_ONLY, (
        f"the count path computes {sorted(computed)}; the partition declares "
        f"{sorted(SHARED_ESTIMANDS | OPERATING_POINT_ONLY)}")


def test_every_shared_estimand_is_a_registered_descriptor():
    """C1 can only compare what the registry also computes."""
    import genomic_variant_classifier.evaluation.registry as registry_module

    registered = set(registry_module.names())
    missing = SHARED_ESTIMANDS - registered
    assert not missing, (
        f"{sorted(missing)} are declared shared but are not registered; C1 has "
        "nothing to compare them against")


def test_the_excluded_registry_metrics_are_registered_and_uncomputed():
    """Each exclusion is a DECISION, and each has a different reason.

    The three threshold metrics are derivable from the same counts and stay out
    because the formulas are not the difficult part -- the applicability, status,
    reason strings and unbounded-value policy are, and the registry already owns
    them. Prevalence stays out because it is a population statistic.
    """
    import genomic_variant_classifier.evaluation.registry as registry_module

    registered = set(registry_module.names())
    excluded = REGISTRY_ONLY_THRESHOLD | REGISTRY_ONLY_POPULATION

    assert excluded <= registered, (
        f"{sorted(excluded - registered)} are declared registry-only but are "
        "not registered")

    y, probabilities = _FIXTURES["mixed_all_margins_nonzero"]
    computed = set(_count_path_metrics(y, probabilities).as_mapping())
    assert not (excluded & computed), (
        f"{sorted(excluded & computed)} are computed by the count path despite "
        "being declared registry-only. If that is intended, the partition and "
        "Oracle C1's scope must be updated in the same commit.")


def test_flagged_fraction_has_no_registry_counterpart():
    """Not a gap in the oracle -- a different estimand surface. It is a
    threshold-dependent decision-burden statistic, and no descriptor computes
    it."""
    import genomic_variant_classifier.evaluation.registry as registry_module

    assert OPERATING_POINT_ONLY.isdisjoint(set(registry_module.names()))


def test_the_four_surfaces_are_disjoint():
    """A metric belongs to exactly one surface. Overlap would make the partition
    unable to say where an estimand's authority lives."""
    surfaces = (SHARED_ESTIMANDS, OPERATING_POINT_ONLY,
                REGISTRY_ONLY_THRESHOLD, REGISTRY_ONLY_POPULATION)
    for index, left in enumerate(surfaces):
        for right in surfaces[index + 1:]:
            assert left.isdisjoint(right), (
                f"{sorted(left & right)} appears on two surfaces")


def test_every_threshold_carrying_descriptor_is_accounted_for():
    """THE COMPLETENESS HALF. A NEW threshold-carrying descriptor added to the
    registry must be placed on a surface deliberately -- either compared by C1 or
    excluded with a reason. This fails until someone decides which.
    """
    import genomic_variant_classifier.evaluation.registry as registry_module

    carrying = set()
    for name in registry_module.names():
        descriptor = registry_module.by_name(name)
        if getattr(descriptor, "threshold_parameters", None) is not None:
            carrying.add(name)

    accounted = SHARED_ESTIMANDS | REGISTRY_ONLY_THRESHOLD
    assert carrying == accounted, (
        f"threshold-carrying descriptors {sorted(carrying)} do not match the "
        f"partition's {sorted(accounted)}. A new one must be placed on a "
        "surface deliberately: compared by C1, or excluded with a reason.")


def test_all_nine_threshold_descriptors_share_one_declaration():
    """C1'S FOUNDING MEASUREMENT, re-derived over the live graph.

    One fixture family serves the entire comparable set ONLY because all nine
    carry the same parameters. A descriptor pinned elsewhere, or using GREATER,
    would need its own fixture -- and `>=` versus `>` differ exactly at
    `prob == threshold`, which is where these fixtures sit.
    """
    import genomic_variant_classifier.evaluation.registry as registry_module

    declarations = set()
    for name in registry_module.names():
        descriptor = registry_module.by_name(name)
        parameters = getattr(descriptor, "threshold_parameters", None)
        if parameters is not None:
            declarations.add((parameters.threshold, parameters.operator.value))

    assert declarations == {(0.5, ">=")}, (
        f"the threshold declarations are {sorted(declarations)}; C1's fixtures "
        "assume a single canonical (0.5, >=) and each additional declaration "
        "needs its own fixture family")
