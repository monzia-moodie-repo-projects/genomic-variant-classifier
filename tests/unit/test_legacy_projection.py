"""Verification suite for the schema-version-2 compatibility interpreter.

WHY THIS MODULE EXISTS, AND WHY IT DID NOT
===========================================
`legacy_projection.py` was written without one, on the grounds that it was a
small helper. A sabotage matrix then survived six of eleven mutations, and every
one of the six traced to the same cause: the module had no dedicated
execution-focused tests. It borrowed coverage from the calibration suite, which
proved the resolver was NOT called for calibration and nothing whatever about the
resolver's own contract, the authorisation rules, or the per-field rounding.

That is ONE ARCHITECTURAL BLIND SPOT producing six surviving mutations, not six
independent deficiencies. The distinction matters: the first predicts that a
dedicated module collapses the survivor count, and the second does not.

The module is not a helper. It is an INTERPRETER over a declarative policy:
authorisation by (status, reason), per-field rounding, metric-specific
behaviour, schema-version-2 compatibility. So this suite tests DECISION PATHS
rather than outputs. Not "what number came out" but: which rule fired, was it
authorised, which rules were considered and rejected, was rounding applied,
was substitution bypassed.

THE CASE THAT MAKES OUTPUT-ONLY TESTING INSUFFICIENT
-----------------------------------------------------
Two different rules legitimately produce the same legacy scalar:

    constant_classifier.f1     = 0.0   a MEASUREMENT -- twenty positives
                                       existed and none were found
    degenerate_all_negative.f1 = 0.0   a SUBSTITUTION for canonical UNDEFINED

Identical numbers, opposite meanings. No assertion comparing values can separate
them; `ProjectionDecision` can.
"""
from __future__ import annotations

import math

import pytest

from genomic_variant_classifier.evaluation.capabilities import (
    MetricResult,
    MetricStatus,
)
from genomic_variant_classifier.evaluation.legacy_projection import (
    LEGACY_PROJECTION_POLICIES,
    LegacyProjectionError,
    LegacyProjectionPolicy,
    ProjectionDecision,
    UndefinedProjectionRule,
    legacy_metric_value,
    legacy_values_equal,
    project_legacy_fields,
    projection_decision,
    resolve_undefined_projection,
)


def _result(value=0.5, status=MetricStatus.OK, reason=None):
    return MetricResult(value, status, reason, {})


NAN = float("nan")


# --------------------------------------------------------------------------- #
# 1. THE POLICY-DECISION MATRIX
#
# The legal state space is declared ONCE and the tests are generated from it.
# Adding a rule requires adding exactly one row, which is what keeps the suite
# honest as the policy grows.
# --------------------------------------------------------------------------- #
#   field              status                 reason                     rule                 authorised  source
DECISION_MATRIX = [
    ("auroc",  MetricStatus.OK,                   None,                    UndefinedProjectionRule.NONE, False, "typed_value"),
    ("auprc",  MetricStatus.OK,                   None,                    UndefinedProjectionRule.NONE, False, "typed_value"),
    ("mcc",    MetricStatus.OK,                   None,                    UndefinedProjectionRule.NONE, False, "typed_value"),
    ("f1",     MetricStatus.OK,                   None,                    UndefinedProjectionRule.NONE, False, "typed_value"),

    ("mcc",    MetricStatus.UNDEFINED,            "zero_confusion_margin", UndefinedProjectionRule.ZERO, True,  "substitute"),
    ("f1",     MetricStatus.UNDEFINED,            "zero_f1_denominator",   UndefinedProjectionRule.ZERO, True,  "substitute"),

    # UNDEFINED, but for a cause the policy does not authorise.
    ("mcc",    MetricStatus.UNDEFINED,            "zero_f1_denominator",   UndefinedProjectionRule.ZERO, False, "typed_value"),
    ("f1",     MetricStatus.UNDEFINED,            "zero_confusion_margin", UndefinedProjectionRule.ZERO, False, "typed_value"),
    ("mcc",    MetricStatus.UNDEFINED,            "labels_absent",         UndefinedProjectionRule.ZERO, False, "typed_value"),

    # Not UNDEFINED at all. A substitute authorised for an undefined
    # mathematical form must not be emitted for a cohort that was merely too
    # small, or for an input that failed validation.
    ("mcc",    MetricStatus.INSUFFICIENT_SUPPORT, "too_few_rows",          UndefinedProjectionRule.ZERO, False, "typed_value"),
    # THE CASE A MUTATION FOUND MISSING. Every other non-UNDEFINED row above
    # carries an UNAUTHORISED reason, so removing the status condition from the
    # authorisation changed nothing and the break survived. Authorisation must be
    # conjunctive: the reason alone is not sufficient. A cohort that was merely
    # too small must not receive a substitute authorised for an undefined
    # MATHEMATICAL FORM, even when the reason string happens to match.
    ("mcc",    MetricStatus.INSUFFICIENT_SUPPORT, "zero_confusion_margin", UndefinedProjectionRule.ZERO, False, "typed_value"),
    ("f1",     MetricStatus.FAILED,               "zero_f1_denominator",   UndefinedProjectionRule.ZERO, False, "typed_value"),
    ("mcc",    MetricStatus.NOT_APPLICABLE,       "zero_confusion_margin", UndefinedProjectionRule.ZERO, False, "typed_value"),
    ("mcc",    MetricStatus.FAILED,               "kernel_raised",         UndefinedProjectionRule.ZERO, False, "typed_value"),
    ("mcc",    MetricStatus.NOT_APPLICABLE,       "inputs_missing",        UndefinedProjectionRule.ZERO, False, "typed_value"),
    ("f1",     MetricStatus.INSUFFICIENT_SUPPORT, "too_few_rows",          UndefinedProjectionRule.ZERO, False, "typed_value"),

    # A field with no substitution rule projects its own value, whatever it is.
    ("auroc",  MetricStatus.UNDEFINED,            "binary_class_support_required", UndefinedProjectionRule.NONE, False, "typed_value"),
    ("auprc",  MetricStatus.UNDEFINED,            "binary_class_support_required", UndefinedProjectionRule.NONE, False, "typed_value"),
    ("calibration_ece", MetricStatus.FAILED,      "binning_invariant",     UndefinedProjectionRule.NONE, False, "typed_value"),
    ("calibration_mce", MetricStatus.OK,          None,                    UndefinedProjectionRule.NONE, False, "typed_value"),
    ("calibration_mce", MetricStatus.FAILED,      "binning_invariant",     UndefinedProjectionRule.NONE, False, "typed_value"),
    ("brier_score",     MetricStatus.OK,          None,                    UndefinedProjectionRule.NONE, False, "typed_value"),
    ("brier_score",     MetricStatus.NOT_APPLICABLE, "probabilities_required", UndefinedProjectionRule.NONE, False, "typed_value"),
    ("prevalence",      MetricStatus.OK,          None,                    UndefinedProjectionRule.NONE, False, "typed_value"),
    ("prevalence",      MetricStatus.INSUFFICIENT_SUPPORT, "empty_cohort", UndefinedProjectionRule.NONE, False, "typed_value"),
]


@pytest.mark.parametrize("field,status,reason,rule,authorised,source", DECISION_MATRIX,
                         ids=lambda v: getattr(v, "value", str(v))[:26])
def test_every_declared_decision_path(field, status, reason, rule, authorised, source):
    """Every legal state documented, every illegal substitution tested."""
    value = 0.5 if status is MetricStatus.OK else NAN
    decision = projection_decision(_result(value, status, reason), field_name=field)

    assert decision.rule is rule, f"{field}: wrong rule consulted"
    assert decision.authorised is authorised, f"{field}: wrong authorisation"
    assert decision.source == source, f"{field}: wrong projection source"

    if source == "substitute":
        assert decision.raw_value == 0.0
        assert not math.isnan(decision.rounded_value)
    else:
        if math.isnan(value):
            assert math.isnan(decision.rounded_value), (
                f"{field}: a non-OK result without an authorised substitution "
                "must project as NaN, not as a plausible number")


def test_the_two_zeroes_are_distinguishable_by_decision_not_by_value():
    """THE CASE OUTPUT-ONLY TESTING CANNOT REACH."""
    measured = projection_decision(_result(0.0, MetricStatus.OK), field_name="f1")
    substituted = projection_decision(
        _result(NAN, MetricStatus.UNDEFINED, "zero_f1_denominator"), field_name="f1")

    assert measured.rounded_value == substituted.rounded_value == 0.0
    assert measured.source == "typed_value" and measured.authorised is False
    assert substituted.source == "substitute" and substituted.authorised is True
    assert measured.rule is UndefinedProjectionRule.NONE
    assert substituted.rule is UndefinedProjectionRule.ZERO


# --------------------------------------------------------------------------- #
# 2. POLICY COMPLETENESS -- is every implemented rule DECLARED?
# --------------------------------------------------------------------------- #
def test_every_rule_member_is_reachable_from_some_policy():
    """Catches orphan enum members: a rule nobody uses is dead vocabulary."""
    declared = {p.undefined_rule for p in LEGACY_PROJECTION_POLICIES.values()}
    unreachable = set(UndefinedProjectionRule) - declared
    assert not unreachable, (
        f"rule member(s) {sorted(r.value for r in unreachable)} are declared in "
        "the vocabulary but used by no policy")


def test_every_policy_names_exactly_one_rule_member():
    """Catches a policy carrying something outside the closed vocabulary."""
    for field, policy in LEGACY_PROJECTION_POLICIES.items():
        assert isinstance(policy.undefined_rule, UndefinedProjectionRule), field


def test_every_rule_member_is_exercised_by_the_decision_matrix():
    """Catches a rule that exists, is used, and is never tested."""
    exercised = {row[3] for row in DECISION_MATRIX}
    missing = set(UndefinedProjectionRule) - exercised
    assert not missing, (
        f"rule member(s) {sorted(r.value for r in missing)} appear in no matrix "
        "row; adding a rule must mean adding a row")


def test_every_policy_field_appears_in_the_decision_matrix():
    tested = {row[0] for row in DECISION_MATRIX}
    untested = set(LEGACY_PROJECTION_POLICIES) - tested
    assert not untested, f"policy field(s) {sorted(untested)} are never exercised"


def test_a_substitution_rule_without_authorised_reasons_is_refused():
    """It would apply to EVERY undefined result, including unrelated causes."""
    with pytest.raises(ValueError, match="no authorised reason"):
        LegacyProjectionPolicy("probe", "probe", decimals=5,
                               undefined_rule=UndefinedProjectionRule.ZERO)


def test_authorised_reasons_without_a_rule_are_refused():
    with pytest.raises(ValueError, match="authorise nothing"):
        LegacyProjectionPolicy("probe", "probe", decimals=5,
                               undefined_rule=UndefinedProjectionRule.NONE,
                               authorised_undefined_reasons=frozenset({"r"}))


def test_a_rule_outside_the_closed_vocabulary_is_refused():
    with pytest.raises(TypeError, match="closed UndefinedProjectionRule"):
        LegacyProjectionPolicy("probe", "probe", decimals=5,
                               undefined_rule="zero")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 3. ROUNDING -- extracted per field, never a global rule
# --------------------------------------------------------------------------- #
def test_prevalence_rounds_to_four_and_the_rest_to_five():
    """The asymmetry is not untidiness. `prevalence` became a registered metric
    only in commit 2b-2, so it is precisely the field where a plausible global
    rule silently disagrees with what the report has always emitted."""
    assert LEGACY_PROJECTION_POLICIES["prevalence"].decimals == 4
    for field in ("auroc", "auprc", "brier_score", "calibration_ece",
                  "calibration_mce", "mcc", "f1"):
        assert LEGACY_PROJECTION_POLICIES[field].decimals == 5, field


def test_the_rounding_difference_is_observable():
    """A value chosen so four and five decimals genuinely differ. Without this
    the previous assertion could pass against any implementation."""
    value = 333 / 700          # 0.4757142857...
    assert round(value, 4) != round(value, 5), "the probe cannot separate them"

    as_prevalence = legacy_metric_value(_result(value), field_name="prevalence")
    as_auroc = legacy_metric_value(_result(value), field_name="auroc")
    assert as_prevalence == round(value, 4)
    assert as_auroc == round(value, 5)
    assert as_prevalence != as_auroc


def test_rounding_is_applied_after_substitution_not_before():
    """Order matters and is declared: typed value, then authorised substitution,
    then rounding. Rounding a NaN and inferring compatibility afterwards would
    be a different and wrong pipeline."""
    decision = projection_decision(
        _result(NAN, MetricStatus.UNDEFINED, "zero_confusion_margin"),
        field_name="mcc")
    assert decision.source == "substitute"
    assert decision.raw_value == 0.0
    assert decision.rounded_value == 0.0
    assert not math.isnan(decision.rounded_value)


def test_a_non_finite_value_survives_rounding_unchanged():
    decision = projection_decision(
        _result(NAN, MetricStatus.UNDEFINED, "unauthorised"), field_name="auroc")
    assert math.isnan(decision.rounded_value)


# --------------------------------------------------------------------------- #
# 4. THE STRUCTURAL BRANCH
# --------------------------------------------------------------------------- #
def test_the_resolver_refuses_an_OK_result():
    """Reaching the resolver with an OK result means the structural branch in
    `legacy_metric_value` was bypassed -- a defect in the interpreter, not a
    property of the metric."""
    with pytest.raises(LegacyProjectionError, match="invoked for an OK result"):
        resolve_undefined_projection(
            _result(0.9, MetricStatus.OK),
            policy=LEGACY_PROJECTION_POLICIES["auroc"])


def test_an_OK_result_never_reaches_the_resolver(monkeypatch):
    """Counted, not inferred."""
    from genomic_variant_classifier.evaluation import legacy_projection as module

    calls = []
    original = module.resolve_undefined_projection
    monkeypatch.setattr(module, "resolve_undefined_projection",
                        lambda r, *, policy: (calls.append(policy.field_name),
                                              original(r, policy=policy))[1])
    module.legacy_metric_value(_result(0.9, MetricStatus.OK), field_name="auroc")
    assert calls == []


def test_an_unknown_field_is_refused_rather_than_guessed():
    with pytest.raises(KeyError, match="no legacy projection policy"):
        legacy_metric_value(_result(), field_name="not_a_report_field")


# --------------------------------------------------------------------------- #
# 5. project_legacy_fields
# --------------------------------------------------------------------------- #
def test_a_missing_metric_is_omitted_not_defaulted():
    """A missing metric is not a metric worth zero. Emitting a placeholder would
    let the projection invariant compare a real field against an invented one."""
    projected = project_legacy_fields({"auroc": _result(0.9)})
    assert set(projected) == {"auroc"}
    assert "mcc" not in projected


def test_every_policy_field_is_projected_when_its_metric_is_present():
    results = {p.metric_name: _result(0.5) for p in LEGACY_PROJECTION_POLICIES.values()}
    projected = project_legacy_fields(results)
    assert set(projected) == set(LEGACY_PROJECTION_POLICIES)


# --------------------------------------------------------------------------- #
# 6. legacy_values_equal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("left,right,expected", [
    (0.5, 0.5, True),
    (0.5, 0.50001, False),
    (NAN, NAN, True),
    (NAN, 0.0, False),
    (0.0, NAN, False),
    (0.0, -0.0, True),
])
def test_exact_nan_aware_comparison(left, right, expected):
    """No tolerance. Rounding has already happened, so any remaining difference
    is a projection defect rather than floating-point noise, and a tolerance
    would hide exactly what this comparison exists to surface."""
    assert legacy_values_equal(left, right) is expected
