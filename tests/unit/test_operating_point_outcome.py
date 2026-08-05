"""The typed operating-point outcome: a refusal is not an absence.

OP-1 step 2, 2026-08-05. Closes D2-D5 (fabricated 0.0) and D6 (rounding at
construction) BY CONSTRUCTION.

Author: Monzia Moodie
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.capabilities import (
    MetricMetadataKey, MetricStatus)
from genomic_variant_classifier.evaluation.population import EvaluationPopulation
from genomic_variant_classifier.evaluation.thresholds import (
    ConfusionCounts,
    OperatingPointCertificationBlocker,
    OperatingPointOutcome,
    ThresholdOperator,
    ThresholdParameters,
    ThresholdSource,
    metrics_from_counts,
)


def _parameters(threshold=0.5):
    return ThresholdParameters(threshold=threshold,
                               operator=ThresholdOperator.GREATER_OR_EQUAL,
                               source=ThresholdSource.EVALUATION_SWEEP)


def _counts(tp, fp, fn, tn):
    return ConfusionCounts(true_positive=tp, false_positive=fp,
                           false_negative=fn, true_negative=tn)


# --------------------------------------------------------------------------- #
# D2-D5: a quantity that cannot be computed REFUSES, and cannot fabricate
# --------------------------------------------------------------------------- #
def test_an_empty_predicted_positive_set_refuses_rather_than_reporting_zero():
    """D3. The legacy form was `tp / (tp + fp) if (tp + fp) > 0 else 0.0`, and
    0.0 asserts that every flagged row was wrong -- when nothing was flagged."""
    metrics = metrics_from_counts(_counts(0, 0, 2, 2), _parameters())
    result = metrics.positive_predictive_value

    assert result.status is MetricStatus.UNDEFINED
    assert result.reason == "empty_predicted_positive_set"
    assert math.isnan(result.value)
    assert not result.is_ok


def test_an_empty_predicted_negative_set_refuses():
    """D4, the sibling case."""
    metrics = metrics_from_counts(_counts(2, 2, 0, 0), _parameters())
    result = metrics.negative_predictive_value

    assert result.status is MetricStatus.UNDEFINED
    assert result.reason == "empty_predicted_negative_set"
    assert math.isnan(result.value)


def test_absent_class_support_is_insufficient_support_not_undefined():
    """REG-2's boundary, at the count level. An absent reference class is a
    COHORT-support problem; a vanishing denominator is a fact about a quotient.
    Conflating them was the inconsistency REG-2 corrected."""
    no_positives = metrics_from_counts(_counts(0, 2, 0, 2), _parameters())
    assert no_positives.sensitivity.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert no_positives.sensitivity.reason == "positive_class_support_required"

    no_negatives = metrics_from_counts(_counts(2, 0, 2, 0), _parameters())
    assert no_negatives.specificity.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert no_negatives.specificity.reason == "negative_class_support_required"


def test_f1_is_computed_from_counts_so_it_cannot_inherit_a_fabrication():
    """D5. The legacy form computed F1 from a positive predictive value that
    might itself be a fabricated 0.0 -- a second fabrication derived from the
    first, with no record that either occurred.

    2TP/(2TP+FP+FN) is the same quantity and cannot inherit anything: here the
    positive predictive value REFUSES while F1 is computed and finite.
    """
    metrics = metrics_from_counts(_counts(0, 0, 2, 2), _parameters())

    assert not metrics.positive_predictive_value.is_ok
    assert metrics.f1.is_ok
    assert metrics.f1.value == pytest.approx(0.0)


def test_the_matthews_coefficient_refuses_on_a_vanishing_margin():
    """scikit-learn returns 0.0 and raises UndefinedMetricWarning while doing so
    -- its own warning being the evidence that the 0.0 is a fabrication."""
    metrics = metrics_from_counts(_counts(2, 2, 0, 0), _parameters())
    assert metrics.matthews_correlation_coefficient.status is (
        MetricStatus.UNDEFINED)
    assert metrics.matthews_correlation_coefficient.reason == (
        "zero_confusion_margin")


def test_no_metric_can_carry_a_finite_value_under_a_non_ok_status():
    """THE CONSTRUCTION-LEVEL GUARANTEE. `MetricResult.__post_init__` refuses it,
    so D2-D5 cannot recur in this type however the code is later edited."""
    for counts in (_counts(0, 0, 2, 2), _counts(2, 2, 0, 0),
                   _counts(0, 2, 0, 2), _counts(0, 0, 0, 0)):
        metrics = metrics_from_counts(counts, _parameters())
        for name, result in metrics.as_mapping().items():
            if result.is_ok:
                assert math.isfinite(result.value), name
                assert not result.reason, name
            else:
                assert math.isnan(result.value), name
                assert result.reason, name


# --------------------------------------------------------------------------- #
# D6: nothing is rounded at storage
# --------------------------------------------------------------------------- #
def test_f1_is_reproducible_from_the_stored_values():
    """D6. The legacy selectors stored `round(value, 4)` and computed F1 from
    UNROUNDED inputs, so the stored F1 could not be recomputed from the stored
    positive predictive value and sensitivity. Here it can."""
    metrics = metrics_from_counts(_counts(37, 13, 11, 39), _parameters())

    ppv = metrics.positive_predictive_value.value
    sensitivity = metrics.sensitivity.value
    recomputed = 2 * ppv * sensitivity / (ppv + sensitivity)

    assert metrics.f1.value == pytest.approx(recomputed, rel=1e-12)


def test_rounding_is_a_display_concern_and_leaves_the_record_alone():
    metrics = metrics_from_counts(_counts(37, 13, 11, 39), _parameters())
    stored = metrics.sensitivity.value
    displayed = metrics.round_for_display()["sensitivity"]

    assert displayed == round(stored, 4)
    assert metrics.sensitivity.value == stored, "the record was mutated"
    assert stored != displayed or stored == round(stored, 4)


def test_round_for_display_reports_a_refusal_as_none_not_as_zero():
    metrics = metrics_from_counts(_counts(0, 0, 2, 2), _parameters())
    assert metrics.round_for_display()["positive_predictive_value"] is None


# --------------------------------------------------------------------------- #
# The outcome: a refusal is not an absence
# --------------------------------------------------------------------------- #
def test_an_ok_outcome_must_carry_metrics():
    """`Optional[OperatingPoint]` let `None` mean every kind of unavailability at
    once. An OK status with no operating point is exactly that ambiguity."""
    with pytest.raises(ValueError, match="must carry metrics"):
        OperatingPointOutcome(status=MetricStatus.OK, reason=None,
                              metrics=None, population=None)


def test_a_refusal_must_carry_a_reason_and_must_not_carry_metrics():
    with pytest.raises(ValueError, match="requires a nonempty reason"):
        OperatingPointOutcome(status=MetricStatus.UNDEFINED, reason=None,
                              metrics=None, population=None)

    metrics = metrics_from_counts(_counts(1, 1, 1, 1), _parameters())
    with pytest.raises(ValueError, match="must not carry metrics"):
        OperatingPointOutcome(status=MetricStatus.UNDEFINED,
                              reason="binary_class_support_required",
                              metrics=metrics, population=None)


def test_an_ok_outcome_must_not_carry_a_reason():
    metrics = metrics_from_counts(_counts(1, 1, 1, 1), _parameters())
    with pytest.raises(ValueError, match="must not carry a reason"):
        OperatingPointOutcome(status=MetricStatus.OK, reason="why",
                              metrics=metrics, population=None)


def test_refused_is_the_only_way_to_build_a_refusal():
    outcome = OperatingPointOutcome.refused(
        MetricStatus.INSUFFICIENT_DATA, "no_achievable_candidate")
    assert not outcome.is_ok
    assert outcome.metrics is None
    assert outcome.reason == "no_achievable_candidate"

    with pytest.raises(ValueError, match="cannot construct an OK outcome"):
        OperatingPointOutcome.refused(MetricStatus.OK, "why")


def test_a_refusal_is_never_certification_eligible():
    outcome = OperatingPointOutcome.refused(
        MetricStatus.UNDEFINED, "no_achievable_candidate")
    assert outcome.certification_eligible is False


def test_certification_requires_both_success_and_no_blockers():
    metrics = metrics_from_counts(_counts(1, 1, 1, 1), _parameters())
    clean = OperatingPointOutcome(status=MetricStatus.OK, reason=None,
                                  metrics=metrics, population=None)
    assert clean.certification_eligible is True

    blocked = OperatingPointOutcome(
        status=MetricStatus.OK, reason=None, metrics=metrics, population=None,
        certification_blockers=(
            OperatingPointCertificationBlocker
            .SAME_POPULATION_SELECTION_AND_EVALUATION,))
    assert blocked.certification_eligible is False


def test_a_blocker_must_be_a_member_not_a_string():
    """Artifacts persist CODES; a raw string would defeat the vocabulary."""
    metrics = metrics_from_counts(_counts(1, 1, 1, 1), _parameters())
    with pytest.raises(TypeError, match="OperatingPointCertificationBlocker"):
        OperatingPointOutcome(
            status=MetricStatus.OK, reason=None, metrics=metrics,
            population=None,
            certification_blockers=("threshold_selected_and_evaluated_on_"
                                    "same_population",))


def test_every_blocker_has_prose():
    """A vocabulary-completeness gate. A code with no rendering is a code a
    report cannot explain."""
    for blocker in OperatingPointCertificationBlocker:
        assert blocker.describe(), blocker.name
        assert len(blocker.describe()) > 20, blocker.name


# --------------------------------------------------------------------------- #
# Population identity, carried once
# --------------------------------------------------------------------------- #
def test_the_outcome_carries_population_identity_in_the_shared_vocabulary():
    """POP-1b: "n=980 beside n=980 says nothing about WHICH 980". Carried ONCE
    on the outcome rather than pushed into every result's metadata, because
    `MetricResult` stays generic by a measured decision."""
    population = EvaluationPopulation.full(4, scope="step2_test",
                                           source_id="op1-step2")
    metrics = metrics_from_counts(_counts(1, 1, 1, 1), _parameters())
    outcome = OperatingPointOutcome(status=MetricStatus.OK, reason=None,
                                    metrics=metrics, population=population)

    metadata = outcome.population_metadata()
    assert metadata[MetricMetadataKey.POPULATION_SCOPE] == "step2_test"
    assert metadata[MetricMetadataKey.N_OBSERVATIONS] == 4
    assert isinstance(metadata[MetricMetadataKey.POPULATION_FINGERPRINT], str)


def test_an_outcome_without_a_population_reports_no_identity_rather_than_a_lie():
    outcome = OperatingPointOutcome.refused(
        MetricStatus.UNDEFINED, "no_achievable_candidate")
    assert outcome.population_metadata() == {}


# --------------------------------------------------------------------------- #
# Serialisation: status-aware, never inferring absence from a value
# --------------------------------------------------------------------------- #
def test_a_refused_metric_serialises_its_value_as_null():
    """CI-p (2026-07-29): a refused result carries NaN IN MEMORY, and NaN cannot
    survive strict JavaScript Object Notation. Absence is authorised by the
    STATUS, never inferred from the value."""
    metrics = metrics_from_counts(_counts(0, 0, 2, 2), _parameters())
    outcome = OperatingPointOutcome(status=MetricStatus.OK, reason=None,
                                    metrics=metrics, population=None)

    serialised = outcome.to_dict()
    ppv = serialised["metrics"]["metrics"]["positive_predictive_value"]
    assert ppv["value"] is None
    assert ppv["status"] == "undefined"
    assert ppv["reason"] == "empty_predicted_positive_set"


def test_the_serialised_outcome_states_its_certification_position():
    metrics = metrics_from_counts(_counts(1, 1, 1, 1), _parameters())
    outcome = OperatingPointOutcome(
        status=MetricStatus.OK, reason=None, metrics=metrics, population=None,
        certification_blockers=(
            OperatingPointCertificationBlocker
            .POST_SELECTION_VALIDATION_NOT_IMPLEMENTED,))

    serialised = outcome.to_dict()
    assert serialised["certification_eligible"] is False
    assert serialised["certification_blockers"] == [
        "post_selection_validation_not_implemented"]


def test_prevalence_is_absent_from_the_metrics():
    """Decision 3: prevalence is a POPULATION statistic, registry-derived.
    Computing a second one here would invent two prevalences that agree until a
    population bug makes them diverge."""
    metrics = metrics_from_counts(_counts(1, 1, 1, 1), _parameters())
    assert "prevalence" not in metrics.as_mapping()
