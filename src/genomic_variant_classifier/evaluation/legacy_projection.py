"""Schema-version-2 compatibility translation.

WHY THIS IS ITS OWN MODULE
==========================
Commit 3b retires evaluator-side computation. Leaving compatibility translation
inside `evaluator.py` would make that module both an orchestration surface and a
legacy translation surface, which weakens the boundary the whole sequence exists
to establish. The layering is:

    registry.py           canonical computation
    capabilities.py       typed results
    legacy_projection.py  schema-version-2 compatibility translation   <- here
    evaluator.py          orchestration

THE TWO SURFACES, AND WHY THEY DIFFER
--------------------------------------
    canonical typed surface   UNDEFINED, with an explicit mathematical reason
    legacy scalar surface     the historical 0.0, where and ONLY where policy
                              authorises it

Both are correct. The typed surface preserves scientific meaning; the legacy
surface preserves historical serialisation. This module is the only place the
two are allowed to disagree, and every disagreement is declared rather than
inferred.

SUBSTITUTION IS AUTHORISED BY IDENTITY *AND* EXACT REASON
----------------------------------------------------------
A substitution keyed on status alone would be a silent falsification. Measured on
2026-07-28 against the frozen report oracle:

    constant_classifier.f1     = 0.0   a MEASUREMENT: twenty positives existed
                                       and none were found
    degenerate_all_negative.f1 = 0.0   a SUBSTITUTION for canonical UNDEFINED

Identical numbers, entirely different meaning. Under a status-keyed rule the
first would have been overwritten by a compatibility zero and no oracle could
have told them apart afterwards, because the value would not have changed. The
reason -- `zero_f1_denominator` -- is what separates them, and it exists only
because commit 3a split the two degenerate conditions into distinct reasons.

An F1 undefined because labels are absent must NOT receive the same substitution
as an F1 undefined because `2*tp + fp + fn == 0`.

THE ORDER IS EXPLICIT
---------------------
    typed registry value
        -> authorised compatibility substitution, when declared
        -> per-field legacy rounding
        -> legacy report field

NOT: round a NaN and infer compatibility behaviour afterwards. Rounding a
substitution is well defined; substituting for a rounded NaN is not.

ROUNDING IS EXTRACTED, NOT CHOSEN
----------------------------------
The decimals below were read out of the landed schema-version-2 implementation,
not invented. `prevalence` rounds to FOUR and every other metric field to FIVE.
That asymmetry is not a mistake to be tidied: `prevalence` became a registered
metric only in commit 2b-2, so it is precisely the field where a plausible global
rule silently disagrees with what the report has always emitted.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional

from .capabilities import MetricResult, MetricStatus

logger = logging.getLogger(__name__)

__all__ = [
    "LEGACY_PROJECTION_POLICIES",
    "LegacyProjectionError",
    "LegacyProjectionPolicy",
    "legacy_metric_value",
    "legacy_values_equal",
    "project_legacy_fields",
    "ProjectionDecision",
    "UndefinedProjectionRule",
    "projection_decision",
    "resolve_undefined_projection",
]


class UndefinedProjectionRule(str, Enum):
    """How a NON-OK result becomes a legacy scalar. A CLOSED vocabulary.

    Closed on purpose. An arbitrary callable stored in configuration would make
    the compatibility policy executable data -- unauditable, unserialisable, and
    impossible to enumerate. A closed enum can be checked for completeness: every
    member must have exactly one policy, and every policy exactly one member.

    NONE means the result projects its own value, whatever that is. It is not the
    absence of a rule; it is the rule that says "do not substitute".
    """

    NONE = "none"
    ZERO = "zero"


@dataclass(frozen=True)
class ProjectionDecision:
    """WHY a legacy value is what it is, not merely what it is.

    Exists because two different rules legitimately produce the same number.
    `constant_classifier.f1 = 0.0` is a measurement -- twenty positives existed
    and none were found -- while `degenerate_all_negative.f1 = 0.0` is a
    compatibility substitution for a canonically UNDEFINED result. Identical
    scalars, opposite meanings, and no test comparing values alone can tell them
    apart.

    This record lets a test ask which rule fired, whether it was authorised, and
    whether rounding was applied. It does not escape the module: the report gets
    a float, and the decision is available to whoever needs to audit it.
    """

    field_name: str
    rule: UndefinedProjectionRule
    authorised: bool
    source: str            # "typed_value" | "substitute"
    raw_value: float
    rounded_value: float
    decimals: Optional[int]


class LegacyProjectionError(RuntimeError):
    """A legacy projection contract was violated.

    Distinct from a metric refusal: this is a defect in the COMPATIBILITY layer,
    not a property of the cohort, and it must never be resolved by emitting a
    plausible-looking number.
    """


@dataclass(frozen=True)
class LegacyProjectionPolicy:
    """How ONE legacy report field is derived from ONE typed result.

    Declarative on purpose. The alternative -- branching on metric name and
    status inside the projection function -- puts the compatibility concessions
    where nobody reviewing the report can see them, and makes each new field an
    invitation to add another branch rather than another row.

    Attributes
    ----------
    field_name
        The legacy report attribute, for example `mcc`.
    metric_name
        The registered descriptor whose result feeds it. Not always the same
        string: the report says `mcc`, the registry says
        `matthews_correlation_coefficient`.
    decimals
        The rounding the landed schema-version-2 implementation applies to this
        field. `None` would mean no rounding; every current field rounds.
    undefined_substitute
        The historical value to emit when the result is UNDEFINED for an
        AUTHORISED reason. `None` means no substitution is permitted and an
        undefined result projects as NaN.
    authorised_undefined_reasons
        The exact reasons that permit the substitution. An UNDEFINED result for
        any other reason does NOT receive it.
    """

    field_name: str
    metric_name: str
    decimals: Optional[int]
    undefined_rule: "UndefinedProjectionRule" = None  # type: ignore[assignment]
    authorised_undefined_reasons: frozenset = frozenset()

    def __post_init__(self) -> None:
        if not self.field_name or not self.metric_name:
            raise ValueError("field_name and metric_name are required")
        if self.decimals is not None and (
                isinstance(self.decimals, bool) or not isinstance(self.decimals, int)
                or self.decimals < 0):
            raise ValueError(
                f"{self.field_name}: decimals must be a non-negative integer or None")
        if self.undefined_rule is None:
            object.__setattr__(self, "undefined_rule", UndefinedProjectionRule.NONE)
        if not isinstance(self.undefined_rule, UndefinedProjectionRule):
            raise TypeError(
                f"{self.field_name}: undefined_rule must be a member of the "
                "closed UndefinedProjectionRule vocabulary")
        if self.undefined_rule is UndefinedProjectionRule.NONE and \
                self.authorised_undefined_reasons:
            raise ValueError(
                f"{self.field_name}: reasons are authorised but the rule is NONE; "
                "the policy would authorise nothing")
        if self.undefined_rule is not UndefinedProjectionRule.NONE and \
                not self.authorised_undefined_reasons:
            raise ValueError(
                f"{self.field_name}: a substitution rule is declared with no "
                "authorised reason, which would apply it to EVERY undefined "
                "result -- including ones undefined for unrelated causes")
        object.__setattr__(self, "authorised_undefined_reasons",
                           frozenset(self.authorised_undefined_reasons))


# The decimals here were EXTRACTED from evaluator.py's report-construction block
# on 2026-07-28, not chosen. See the module docstring.
LEGACY_PROJECTION_POLICIES: Mapping[str, LegacyProjectionPolicy] = MappingProxyType({
    p.field_name: p for p in (
        LegacyProjectionPolicy("auroc", "auroc", decimals=5),
        LegacyProjectionPolicy("auprc", "auprc", decimals=5),
        LegacyProjectionPolicy("brier_score", "brier_score", decimals=5),
        LegacyProjectionPolicy("calibration_ece", "expected_calibration_error",
                               decimals=5),
        LegacyProjectionPolicy("calibration_mce", "maximum_calibration_error",
                               decimals=5),
        LegacyProjectionPolicy("prevalence", "prevalence", decimals=4),
        LegacyProjectionPolicy(
            "mcc", "matthews_correlation_coefficient", decimals=5,
            undefined_rule=UndefinedProjectionRule.ZERO,
            authorised_undefined_reasons=frozenset({"zero_confusion_margin"})),
        LegacyProjectionPolicy(
            "f1", "f1", decimals=5,
            undefined_rule=UndefinedProjectionRule.ZERO,
            authorised_undefined_reasons=frozenset({"zero_f1_denominator"})),
    )
})


_RULE_VALUES = {UndefinedProjectionRule.ZERO: 0.0}


def resolve_undefined_projection(result: MetricResult, *,
                                 policy: LegacyProjectionPolicy) -> float:
    """The compatibility substitution. Called ONLY for a non-OK result.

    A SEPARATE, NAMED FUNCTION ON PURPOSE. Folding this into
    `legacy_metric_value` would make "no compatibility rule was consulted for
    this field" inferable only from the policy table's current contents. As a
    named call it is OBSERVABLE: a test can count invocations and prove that a
    canonically-OK calibration result never reaches compatibility logic at all.

    That matters because the alternative failure is silent. Someone restoring the
    old both-classes applicability rule could mask the regression by adding a
    calibration substitution here, and a test inspecting only the table would not
    notice.
    """
    return _decide(result, policy=policy).rounded_value


def _decide(result: MetricResult, *,
            policy: LegacyProjectionPolicy) -> ProjectionDecision:
    """Every projection decision, recorded rather than merely performed."""
    if result.status is MetricStatus.OK:
        raise LegacyProjectionError(
            f"{policy.field_name}: the compatibility resolver was invoked for an "
            "OK result. An OK result projects its own value; reaching here means "
            "the structural branch in legacy_metric_value was bypassed.")

    authorised = (
        policy.undefined_rule is not UndefinedProjectionRule.NONE
        and result.status is MetricStatus.UNDEFINED
        and result.reason in policy.authorised_undefined_reasons)

    if authorised:
        raw = float(_RULE_VALUES[policy.undefined_rule])
        source = "substitute"
    else:
        # Not UNDEFINED, or UNDEFINED for a cause this policy does not cover.
        # INSUFFICIENT_SUPPORT, FAILED and NOT_APPLICABLE are not UNDEFINED, and
        # a substitute authorised for an undefined mathematical form must not be
        # emitted for a cohort that was merely too small or an input that failed
        # validation.
        raw = float(result.value)
        source = "typed_value"

    return ProjectionDecision(
        field_name=policy.field_name, rule=policy.undefined_rule,
        authorised=authorised, source=source, raw_value=raw,
        rounded_value=_round(raw, policy.decimals), decimals=policy.decimals)


def _round(value: float, decimals: Optional[int]) -> float:
    if decimals is None or math.isnan(value) or math.isinf(value):
        return value
    return round(value, decimals)


def projection_decision(result: MetricResult, *, field_name: str) -> ProjectionDecision:
    """The audit view: which rule produced this value, and was it authorised."""
    policy = _policy_for(field_name)
    if result.status is MetricStatus.OK:
        raw = float(result.value)
        return ProjectionDecision(
            field_name=field_name, rule=UndefinedProjectionRule.NONE,
            authorised=False, source="typed_value", raw_value=raw,
            rounded_value=_round(raw, policy.decimals), decimals=policy.decimals)
    return _decide(result, policy=policy)


def _policy_for(field_name: str) -> LegacyProjectionPolicy:
    policy = LEGACY_PROJECTION_POLICIES.get(field_name)
    if policy is None:
        raise KeyError(
            f"no legacy projection policy for field {field_name!r}; a report "
            "field without a declared policy would be projected by guesswork")
    return policy


def legacy_metric_value(result: MetricResult, *, field_name: str) -> float:
    """Project ONE typed result onto ONE legacy scalar field.

    Order: typed value OR authorised substitution, then rounding. The branch is
    STRUCTURAL rather than conditional inside one expression, so that the two
    paths are separately observable.
    """
    policy = _policy_for(field_name)
    if result.status is MetricStatus.OK:
        return _round(float(result.value), policy.decimals)
    return resolve_undefined_projection(result, policy=policy)


def project_legacy_fields(metric_results: Mapping[str, MetricResult]) -> dict:
    """Every legacy scalar field derivable from the typed results.

    Fields whose metric is absent from the mapping are OMITTED rather than
    defaulted. A missing metric is not a metric worth zero, and emitting a
    placeholder would let the projection invariant compare a real field against
    an invented one.
    """
    out = {}
    for field_name, policy in LEGACY_PROJECTION_POLICIES.items():
        result = metric_results.get(policy.metric_name)
        if result is None:
            continue
        out[field_name] = legacy_metric_value(result, field_name=field_name)
    return out


def legacy_values_equal(left: float, right: float) -> bool:
    """Exact comparison, NaN-aware, with NO tolerance.

    Rounding has already happened by the time two projected values are compared,
    so any remaining difference is a projection defect rather than floating-point
    noise. A tolerance would hide exactly the defects this comparison exists to
    surface.

    NaN is treated as equal to NaN because `float("nan") != float("nan")` would
    otherwise make every refused metric compare unequal to itself.
    """
    left_nan = isinstance(left, float) and math.isnan(left)
    right_nan = isinstance(right, float) and math.isnan(right)
    if left_nan or right_nan:
        return left_nan and right_nan
    return left == right
