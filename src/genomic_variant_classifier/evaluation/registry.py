"""The metric registry: what may be computed, when, and what the answer means.

WHY THIS MODULE EXISTS
======================
The project runs TWO conventions for a metric's answer.

The panels -- clustering_metrics, representation_geometry, norm_angle_probe --
return `MetricResult`: a value that always knows whether it is a value, with a
status and, when there is no value, a reason.

The kernel in `metrics.py` returns BARE FLOATS. `auroc()`, `auprc()`,
`brier_score()`, `log_loss()` and `expected_calibration_error()` each return a
plain number, and `evaluate()` returns a plain `dict`.

A bare float cannot say "I am not a value." What that costs is recorded in
`metrics.evaluate`'s own docstring, measured on `y = [1,1,1,1]` with
`p = [.9,.8,.85,.95]`:

    auroc NaN   auprc NaN              <- correct, ranking is undefined
    cal_slope NaN   cal_intercept NaN
    brier 0.01875   ece 0.125          <- NUMBERS
    calibration_valid True             <- asserting those numbers are sound

That expected calibration error of 0.125 is just `1 - 0.875`: the gap between the
mean prediction and the only label present. It says nothing about calibration
across the probability range, because the reliability diagram has one occupied
row. The number is finite, arithmetically correct, and scientifically empty.

THAT PARTICULAR DEFECT WAS FIXED ON 2026-07-21, inside `evaluate()`, by widening
the calibration gate beyond `is_probability(p)`. Today the same input returns
`brier NaN`, `ece NaN`, `calibration_valid False`. It is quoted here as the
worked example of WHY a bare float is the wrong return type, not as a live bug --
verified against the current implementation on 2026-07-27.

The general problem the fix did not solve is that each such case must be caught
by a bespoke guard inside one function. A bare float still cannot carry a status,
so every new metric must remember to return NaN and every new caller must
remember to check. The registry makes the status structural instead.

APPLICABILITY IS DECIDED BEFORE THE KERNEL IS CALLED
----------------------------------------------------
A registry that ran every metric and then mapped `NaN -> UNDEFINED` could not
have caught that case, because 0.125 is not NaN. So the order is:

    validate the context ONCE
        -> evaluate the descriptor's applicability predicate
            -> if inapplicable, return a typed MetricResult WITHOUT invoking
               the kernel
            -> otherwise invoke, then validate what came back

An inapplicable metric is never computed. A metric that IS applicable and still
returns a non-finite value has FAILED -- that is an implementation defect, not a
property of the cohort, and the two must not be conflated.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
------------------------------------------
`metrics.evaluate()` is UNTOUCHED. It remains the legacy, untyped compatibility
interface; its callers may depend on exact dictionary keys, bare-float values and
the current NaN behaviour. Rewriting it here would mix an architectural addition
with a behaviour change in a historically sensitive function, and a regression
would be impossible to localise. New orchestration and report code consumes the
registry; nothing new should consume the bare dictionary.

`metrics.evaluate()` is also NOT registered as one composite descriptor. Its five
metrics have five different applicability rules -- ranking needs both classes,
calibration needs a meaningful probability partition -- and one capability
decision cannot honestly govern all of them.

ADOPTED AND REJECTED FROM THE 2026-07-27 SCOPE DOCUMENTS
---------------------------------------------------------
ADOPTED: support attachment. Every result records how much evidence stood behind
it, on refusals and failures as well as on values.

ADOPTED: `population_scope`, REQUIRED on every context and therefore present in
every result. Support counts alone do not identify the denominator. This session
produced two cases where correct numbers described different populations and the
difference was invisible: 53 and 63 were both right, over universes that differed
by ten variants; and 85 beside 107 was printed as a breakdown of 107, when
85 + 107 = 192. A number without its population is not evidence, so the context
refuses to be constructed without naming one.

THE PRINCIPLE BEHIND SEVERAL OF THIS SESSION'S DEFECTS
-------------------------------------------------------
    Preserve raw state until diagnostics complete. Canonicalisation occurs only
    after diagnostic measurements have been computed.

Three defects shared one shape -- destroy the distinction, measure the destroyed
distinction, declare success:

  * `n01 + n11 == 203` held only after applicability had already been erased;
  * 85 and 107 printed as a partition after the overlap between them was
    forgotten;
  * quantile bounds sorted by np.minimum/np.maximum BEFORE the crossing rate was
    measured, so that rate is structurally zero and reports perfect health.

It applies to quantile crossings, overlapping populations, duplicate mappings,
cluster identities, calibration exclusions and bootstrap degeneracy alike.

REJECTED, because each would degrade a correction already made:

  * `MetricResult(value=None, status=FAILED, ...)`. The invariant requires NaN
    for a non-OK result; `None` raises `TypeError: ufunc 'isfinite' not
    supported`, verified 2026-07-27. NaN is used.
  * `certification_eligible=True` unconditionally. That is the defect caught in
    this module's own first draft: on a single-class cohort it would certify a
    Brier score. Eligibility is DERIVED. The reviewing document independently
    requires "recording certification eligibility", so the two documents
    disagree and the reviewer is right.
  * `ApplicabilityRule -> MetricResult | None`, None meaning applicable. Such a
    rule may return ANY result, including an OK one carrying a value, so "ruled
    inapplicable" and "computed" become indistinguishable at the type level.
    `Applicability` refuses that structurally.

A NOTE ON A TYPE THAT DOES NOT EXIST
-------------------------------------
The design called for `supported_capabilities: frozenset[EvaluationCapability]`.
There is no `EvaluationCapability` in this project. `CapabilityState` exists but
measures a different axis -- how far a capability has PROGRESSED, from
NOT_IMPLEMENTED to VALIDATED -- and a metric does not "support" NOT_IMPLEMENTED.
Rather than invent a type to match a name, the coarse static filter is expressed
as `required_inputs`, which is concrete and checkable, and the real gate is the
applicability predicate, which sees the actual context. If a genuine capability
axis is needed later it can be added deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from enum import Enum
from typing import Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

from .capabilities import MetricMetadataKey, MetricResult, MetricStatus
from .population import EvaluationPopulation

__all__ = [
    "MetricInput",
    "MetricOutputKind",
    "MetricContext",
    "Applicability",
    "MetricDescriptor",
    "all_metrics",
    "by_name",
    "names",
    "requiring_clusters",
    "requiring_input",
    "evaluate_registered",
    "REGISTRY_SCHEMA_VERSION",
]

REGISTRY_SCHEMA_VERSION = 2


# --------------------------------------------------------------------------- #
# DESCRIPTOR VOCABULARY, completed 2026-07-27
#
# What this commit adds is not "more descriptors". It is the vocabulary every
# descriptor speaks, so that later additions cannot produce a second dialect in
# which some descriptors declare their classification and parameters and others
# leave them implicit.
#
# RESULT KIND IS CLASSIFICATION, NOT DISPATCH. It does not determine
# applicability, certification or required inputs. Those are declared
# separately and remain so; a classification that quietly drove behaviour would
# be a second control path.
#
# RESULT KIND LIVES ON THE DESCRIPTOR AND NOT IN RESULT METADATA. Placing it in
# metadata would perturb every already-serialised result, and the acceptance
# criterion for this commit is that every pre-existing result is byte-identical
# afterwards, with no carve-outs. It joins the serialised surface at schema
# version 3, deliberately.
# --------------------------------------------------------------------------- #
class ResultKind(str, Enum):
    """What KIND of quantity a descriptor produces."""

    PREDICTION_METRIC = "prediction_metric"
    POPULATION_STATISTIC = "population_statistic"


class ThresholdOperator(str, Enum):
    """The comparison that turns a probability into a hard label.

    Declared rather than assumed because `>=` and `>` differ exactly at
    `prob == threshold`, and with the conventional 0.5 that is the value a
    maximally uncertain model emits and the value a two-model average produces
    whenever the pair disagrees. A threshold without its operator is incomplete
    provenance.
    """

    GREATER_OR_EQUAL = ">="
    GREATER = ">"


class ThresholdSource(str, Enum):
    """Where a decision threshold came from.

    A fixed convention and a threshold optimised on a calibration split are not
    the same scientific claim, and a reader of an artifact cannot tell them apart
    from the number alone.
    """

    FIXED_DEFAULT = "fixed_default"
    CALIBRATED = "calibrated"
    USER_SUPPLIED = "user_supplied"


@dataclass(frozen=True)
class ThresholdParameters:
    """The canonical, typed threshold declaration.

    THIS OBJECT IS THE SEMANTICS; the mapping returned by `to_mapping` is merely
    its serialisation. Code should read `descriptor.threshold_parameters.threshold`
    -- type-oriented, checkable, refactorable -- rather than
    `descriptor.parameters["decision_threshold"]`, which is serialisation-oriented
    and silently returns nothing useful when the key is misspelled.

    One instance is shared by a descriptor, its kernel adapter and its
    applicability predicate, and that sharing is asserted BY IDENTITY at import
    time. Three copies of a threshold that merely happen to be equal today is
    how a threshold comes to differ tomorrow.
    """

    threshold: float
    operator: ThresholdOperator
    source: ThresholdSource

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(
                self.threshold, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"decision threshold must be numeric, got "
                f"{type(self.threshold).__name__}")
        value = float(self.threshold)
        if not np.isfinite(value):
            raise ValueError(f"decision threshold must be finite, got {value}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"decision threshold must lie in [0, 1], got {value}; a "
                "threshold outside the probability range would classify every "
                "row identically and report the result as though it had "
                "discriminated")
        object.__setattr__(self, "threshold", value)
        if not isinstance(self.operator, ThresholdOperator):
            raise TypeError("operator must be a ThresholdOperator member")
        if not isinstance(self.source, ThresholdSource):
            raise TypeError("source must be a ThresholdSource member")

    def to_mapping(self) -> dict:
        """Serialisation only. `ThresholdParameters` remains the semantics."""
        return {"decision_threshold": self.threshold,
                "threshold_operator": self.operator.value,
                "threshold_source": self.source.value}


_JSON_SCALARS = (str, int, float, bool, type(None))


def _validate_json_value(value, path: str) -> None:
    """Reject anything that could not survive serialisation, recursively.

    A descriptor is a frozen declaration that ends up in an artifact. Admitting a
    list, an array, a callable or a model object would make the declaration
    mutable in fact while frozen in type, and unserialisable in practice while
    appearing declarative.
    """
    if isinstance(value, bool) or isinstance(value, _JSON_SCALARS):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(
                f"parameter {path}: non-finite floats do not survive JSON")
        return
    if isinstance(value, tuple):
        for i, item in enumerate(value):
            _validate_json_value(item, f"{path}[{i}]")
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str) or not k:
                raise TypeError(
                    f"parameter {path}: mapping keys must be non-empty strings")
            _validate_json_value(v, f"{path}.{k}")
        return
    raise TypeError(
        f"parameter {path}: {type(value).__name__} is not JSON-representable. "
        "Descriptor parameters are a frozen declaration destined for an "
        "artifact; use tuples rather than lists, and never a callable, an "
        "array or a model object.")


class MetricInput(str, Enum):
    """What a metric needs present in the context before it can be attempted.

    A `str`-and-`Enum` subclass, not `StrEnum`: `StrEnum` arrived in Python 3.11
    and pyproject declares requires-python >= 3.10. Pinned by
    test_no_module_uses_strenum_which_would_break_the_declared_python_floor,
    which permits the BACKTICKED spelling so the constraint can be explained
    without violating it -- a lesson this module learned by tripping it.
    """

    LABELS = "labels"
    SCORES = "scores"
    PROBABILITIES = "probabilities"
    CLUSTERS = "clusters"
    SAMPLE_WEIGHT = "sample_weight"


class MetricOutputKind(str, Enum):
    SCALAR = "scalar"
    INTERVAL = "interval"


# --------------------------------------------------------------------------- #
# The context: validated ONCE, before any dispatch
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricContext:
    """Everything a registered metric may read, aligned and validated once.

    Descriptors do NOT reinterpret array lengths or label support. That decision
    is made here, once, so two metrics cannot disagree about how many rows the
    cohort has -- the defect `CleanArrays` was built to remove, where independent
    masks produced two arrays of the same length describing DIFFERENT ROWS and
    every calibration metric silently paired a probability with the wrong label.
    """

    y_true: np.ndarray
    population: EvaluationPopulation
    y_score: Optional[np.ndarray] = None
    y_prob: Optional[np.ndarray] = None
    clusters: Optional[np.ndarray] = None
    sample_weight: Optional[np.ndarray] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.population, EvaluationPopulation):
            raise ValueError(
                "population is REQUIRED and must be an EvaluationPopulation. A "
                "bare scope string is a NAME, not a membership: 53 and 63 were "
                "both correct and described different populations, and 85 and "
                "107 were printed as a partition of 107. A number without its "
                "population is not evidence, and a population named but not "
                "identified cannot be compared with another.")
        n = np.asarray(self.y_true).size
        if n != self.population.n:
            raise ValueError(
                f"y_true has {n} rows but the population declares "
                f"{self.population.n}. The arrays in a context are ALREADY "
                f"PROJECTED, so they carry the length of the population, not of "
                f"the {self.population.n_source}-row source frame. Unprojected "
                "arrays would compute over rows the population had excluded "
                "while reporting the narrower count -- exactly the divergence "
                "between a number and its stated denominator that this stack "
                "exists to remove.")
        for name in ("y_score", "y_prob", "clusters", "sample_weight"):
            v = getattr(self, name)
            if v is not None and np.asarray(v).size != n:
                raise ValueError(
                    f"{name} has {np.asarray(v).size} rows but y_true has {n}; "
                    "the context is aligned ONCE, here, so a descriptor can never "
                    "compute over a mismatched pairing")

    @property
    def population_scope(self) -> str:
        """Derived, never stored.

        A stored copy beside the population would be a second source of truth for
        one fact, and two sources of truth for one fact eventually disagree.
        """
        return self.population.scope

    # --- derived facts every applicability predicate may use -----------------
    @property
    def n(self) -> int:
        return int(np.asarray(self.y_true).size)

    @property
    def classes_observed(self) -> tuple:
        y = np.asarray(self.y_true)
        return tuple(sorted(np.unique(y[np.isfinite(y)]).tolist()))

    @property
    def n_classes_observed(self) -> int:
        return len(self.classes_observed)

    @property
    def has_both_classes(self) -> bool:
        return set(self.classes_observed) >= {0.0, 1.0}

    @property
    def n_clusters(self) -> Optional[int]:
        return None if self.clusters is None else int(
            np.unique(np.asarray(self.clusters)).size)

    def support(self) -> dict:
        """The evidence base a metric was computed over.

        Attached to EVERY result, refusals included. A metric computed on twelve
        rows and one on four hundred thousand are not equally trustworthy, and
        without this an artifact cannot say which it holds. The cohort size
        behind a REFUSAL is equally informative: an INSUFFICIENT_SUPPORT on 3
        rows and one on 300,000 point at different problems.

        `n_clusters` counts DISTINCT clusters, not an effective sample size.
        Effective sample size under clustering is a property of a resampling
        design and already lives in BootstrapResult beside the design effect and
        replicate accounting. Duplicating an approximation here would create a
        second, weaker answer to a question already answered properly.

        NO THRESHOLD IS APPLIED. Whether a minimum observation or cluster count
        should block certification is a scientific policy decision; inventing one
        silently is the class of guess this project removes.
        """
        out = {MetricMetadataKey.POPULATION_SCOPE: self.population_scope,
               MetricMetadataKey.POPULATION_FINGERPRINT:
                   self.population.membership_fingerprint,
               MetricMetadataKey.N_OBSERVATIONS: self.n,
               MetricMetadataKey.N_CLASSES_OBSERVED: self.n_classes_observed}
        if self.clusters is not None:
            out[MetricMetadataKey.N_CLUSTERS] = self.n_clusters
        return out

    def has(self, what: "MetricInput") -> bool:
        return {
            MetricInput.LABELS: self.y_true is not None,
            MetricInput.SCORES: self.y_score is not None,
            MetricInput.PROBABILITIES: self.y_prob is not None,
            MetricInput.CLUSTERS: self.clusters is not None,
            MetricInput.SAMPLE_WEIGHT: self.sample_weight is not None,
        }[what]


# --------------------------------------------------------------------------- #
# Applicability: a decision, with a reason, made BEFORE computing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Applicability:
    """Whether a metric may be attempted, and if not, under which status.

    A bare boolean would force the caller to invent a status and a reason, which
    is how "inapplicable" and "failed" get conflated.
    """

    applicable: bool
    status: Optional[MetricStatus] = None
    reason: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.applicable:
            if self.status is not None or self.reason:
                raise ValueError("an applicable decision carries no status or reason")
        else:
            if self.status is None or self.status is MetricStatus.OK:
                raise ValueError(
                    "an inapplicable decision requires a non-OK status; without one "
                    "the caller must invent it, which is how 'inapplicable' and "
                    "'failed' become indistinguishable")
            if not self.reason:
                raise ValueError("an inapplicable decision requires a nonempty reason")


APPLICABLE = Applicability(applicable=True)


class MetricCallable(Protocol):
    def __call__(self, context: MetricContext) -> float: ...


class ApplicabilityPredicate(Protocol):
    def __call__(self, context: MetricContext) -> Applicability: ...


@dataclass(frozen=True)
class MetricDescriptor:
    name: str
    function: MetricCallable
    required_inputs: frozenset
    applicability: ApplicabilityPredicate
    result_kind: ResultKind
    display_name: str
    description: str
    requires_clusters: bool = False
    output_kind: MetricOutputKind = MetricOutputKind.SCALAR
    include_in_evaluation_report: bool = True
    parameters: Mapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        normalised = dict(self.parameters)
        for key in normalised:
            if not isinstance(key, str) or not key:
                raise TypeError(
                    f"{self.name}: parameter keys must be non-empty strings")
        for key, value in normalised.items():
            _validate_json_value(value, key)
        object.__setattr__(self, "parameters", MappingProxyType(normalised))

    @property
    def threshold_parameters(self):
        """The typed threshold declaration, or `None` for metrics without one.

        Attached to the kernel adapter at construction and read back here, so
        there is exactly ONE object rather than a mapping and a closure that
        happen to agree.
        """
        return getattr(self.function, "_threshold_parameters", None)


# --------------------------------------------------------------------------- #
# Applicability predicates
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The prediction-finiteness contract -- FAIL CLOSED
#
# Ruled 2026-07-27: no numerical kernel may select, filter, normalise or
# redefine its evaluation population. Population construction is an explicit
# upstream operation, and every result must describe exactly that population.
#
# A non-finite predicted probability is a MODEL-OUTPUT FAILURE, not an ordinary
# missing observation. Until this date `metrics._clean` dropped those rows on a
# joint mask inside the kernel, so a value was returned over a silently narrowed
# population while `support()` named the wider one: twenty non-finite
# probabilities in a thousand rows produced a Brier score over 980 rows reported
# as n_observations = 1000, status ok, certification_eligible True.
#
# WHY THIS IS AN APPLICABILITY PREDICATE AND NOT A `-> MetricResult | None`
# VALIDATOR. The reviewing document proposed the latter. This module's docstring
# already REJECTED that shape on the day it was written: such a validator may
# return ANY result, including an OK one carrying a value, so "refused" and
# "computed" stop being distinguishable at the type level. `Applicability`
# refuses it structurally, and MetricStatus.FAILED already covers this case in
# its own definition -- "a prerequisite validated and found contradictory before
# the computation could begin". The ruling is honoured; the rejected type is not
# reintroduced.
#
# The refusal is raised BEFORE the kernel is invoked, so no narrowed population
# is ever computed over, and the support attached to the failure describes the
# population that was ATTEMPTED.
# --------------------------------------------------------------------------- #
def _finiteness_verdict(values, nonfinite_key, finite_key, reason: str) -> Applicability:
    arr = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(arr)
    n_bad = int((~finite).sum())
    if n_bad == 0:
        return APPLICABLE
    return Applicability(
        applicable=False,
        status=MetricStatus.FAILED,
        reason=reason,
        metadata={nonfinite_key: n_bad, finite_key: int(finite.sum())})


def _scores_are_finite(ctx: MetricContext) -> Applicability:
    return _finiteness_verdict(
        ctx.y_score,
        MetricMetadataKey.N_NONFINITE_SCORES,
        MetricMetadataKey.N_FINITE_SCORES,
        "nonfinite_predicted_scores")


def _probabilities_are_finite(ctx: MetricContext) -> Applicability:
    return _finiteness_verdict(
        ctx.y_prob,
        MetricMetadataKey.N_NONFINITE_PROBABILITIES,
        MetricMetadataKey.N_FINITE_PROBABILITIES,
        "nonfinite_predicted_probabilities")


def _requires_both_classes(ctx: MetricContext) -> Applicability:
    """Ranking is undefined when one class is present, and the kernel says so:
    'A single class present -> ranking metrics are undefined. Say so; do not
    guess.' (metrics.py:225)

    The finiteness contract is checked FIRST. A model that emitted a non-finite
    score has failed its output contract, and that is true whatever the cohort's
    class composition; reporting UNDEFINED instead would blame the data for a
    defect in the predictions.
    """
    finite = _scores_are_finite(ctx)
    if not finite.applicable:
        return finite
    if ctx.has_both_classes:
        return APPLICABLE
    return Applicability(
        applicable=False,
        status=MetricStatus.UNDEFINED,
        reason="binary_class_support_required",
        metadata={MetricMetadataKey.N_CLASSES_OBSERVED: ctx.n_classes_observed,
                  "classes_observed": list(ctx.classes_observed)})


def _requires_calibration_support(ctx: MetricContext) -> Applicability:
    """Calibration needs a probability vector AND a meaningful partition.

    THE CASE THIS EXISTS FOR. On a single-class cohort the expected calibration
    error is finite and arithmetically correct and scientifically empty: the
    reliability diagram has one occupied row, so the number is the gap between
    the mean prediction and the only label present. It is INSUFFICIENT_SUPPORT,
    not UNDEFINED -- the quantity is computable, the cohort cannot support its
    interpretation.
    """
    from .metrics import is_probability

    if ctx.y_prob is None:
        return Applicability(applicable=False, status=MetricStatus.NOT_APPLICABLE,
                             reason="probabilities_required")
    if ctx.n == 0:
        return Applicability(applicable=False, status=MetricStatus.INSUFFICIENT_DATA,
                             reason="empty_cohort", metadata={"n": 0})
    # BEFORE is_probability, which documents that it IGNORES non-finite values
    # and would therefore pass a vector containing them.
    finite = _probabilities_are_finite(ctx)
    if not finite.applicable:
        return finite
    if not is_probability(ctx.y_prob):
        return Applicability(applicable=False, status=MetricStatus.NOT_APPLICABLE,
                             reason="values_are_not_probabilities")
    if not ctx.has_both_classes:
        return Applicability(
            applicable=False, status=MetricStatus.INSUFFICIENT_SUPPORT,
            reason="calibration_requires_class_support",
            metadata={MetricMetadataKey.N_CLASSES_OBSERVED: ctx.n_classes_observed,
                      "classes_observed": list(ctx.classes_observed),
                      "note": "a finite value here would be the gap between the "
                              "mean prediction and the only label present"})
    return APPLICABLE


def _requires_probabilities(ctx: MetricContext) -> Applicability:
    """A proper score is numerically defined on one class. It is reported, and
    the certification axis -- not the status -- records that the cohort cannot
    support an inferential claim. Numeric computability, scientific
    interpretability and certification eligibility are three different things."""
    from .metrics import is_probability

    if ctx.y_prob is None:
        return Applicability(applicable=False, status=MetricStatus.NOT_APPLICABLE,
                             reason="probabilities_required")
    if ctx.n == 0:
        return Applicability(applicable=False, status=MetricStatus.INSUFFICIENT_DATA,
                             reason="empty_cohort", metadata={"n": 0})
    finite = _probabilities_are_finite(ctx)
    if not finite.applicable:
        return finite
    if not is_probability(ctx.y_prob):
        return Applicability(applicable=False, status=MetricStatus.NOT_APPLICABLE,
                             reason="values_are_not_probabilities")
    return APPLICABLE


# --------------------------------------------------------------------------- #
# Adapters: translate the uniform context into each kernel's own signature
# --------------------------------------------------------------------------- #
def _auroc(ctx: MetricContext) -> float:
    from .metrics import auroc
    return auroc(ctx.y_true, ctx.y_score)


def _auprc(ctx: MetricContext) -> float:
    from .metrics import auprc
    return auprc(ctx.y_true, ctx.y_score)


def _auprc_gain(ctx: MetricContext) -> float:
    from .metrics import auprc_gain
    return auprc_gain(ctx.y_true, ctx.y_score)


def _brier(ctx: MetricContext) -> float:
    from .metrics import brier_score
    return brier_score(ctx.y_true, ctx.y_prob)


def _log_loss(ctx: MetricContext) -> float:
    from .metrics import log_loss
    return log_loss(ctx.y_true, ctx.y_prob)


def _ece(ctx: MetricContext) -> float:
    from .metrics import expected_calibration_error
    return expected_calibration_error(ctx.y_true, ctx.y_prob)


# --------------------------------------------------------------------------- #
# THE REGISTRY: a frozen declaration, not a mutable table
# --------------------------------------------------------------------------- #
_L, _S, _P = MetricInput.LABELS, MetricInput.SCORES, MetricInput.PROBABILITIES

# --------------------------------------------------------------------------- #
# Threshold-dependent adapters and their applicability
#
# ONE `ThresholdParameters` INSTANCE per metric, shared by the descriptor's
# serialised mapping, its kernel adapter and its applicability predicate, with
# the sharing asserted BY IDENTITY in `_validate_registry`. Three copies of a
# threshold that merely happen to be equal today is precisely how a threshold
# comes to differ tomorrow.
#
# WHY THE DEGENERATE MARGIN IS CAUGHT BY APPLICABILITY AND NOT BY THE KERNEL'S
# RETURN. `compute` already rules, deliberately, that an APPLICABLE metric
# returning a non-finite value is FAILED -- "an implementation defect, not a
# property of the cohort. Calling it UNDEFINED would blame the data." A zero
# confusion-matrix margin IS a property of the cohort, so it must be recognised
# BEFORE dispatch. Letting the kernel's NaN carry that meaning would report a
# constant classifier as a broken implementation.
# --------------------------------------------------------------------------- #
_MCC_THRESHOLD = ThresholdParameters(
    threshold=0.5, operator=ThresholdOperator.GREATER_OR_EQUAL,
    source=ThresholdSource.FIXED_DEFAULT)

_F1_THRESHOLD = ThresholdParameters(
    threshold=0.5, operator=ThresholdOperator.GREATER_OR_EQUAL,
    source=ThresholdSource.FIXED_DEFAULT)

_CALIBRATION_PARAMETERS = {
    "n_bins": 10,
    "binning": "equal_width",
    "interval_convention": "[lo,hi);final_closed",
}


def _threshold_adapter(kernel, tp: ThresholdParameters):
    def adapter(ctx: MetricContext) -> float:
        return kernel(ctx.y_true, ctx.y_prob, threshold=tp.threshold,
                      operator=tp.operator.value)
    adapter._threshold_parameters = tp
    adapter.__name__ = f"_{kernel.__name__}_adapter"
    return adapter


def _requires_nondegenerate_confusion(tp: ThresholdParameters, *, metric: str):
    """Refuse BEFORE dispatch when the confusion matrix has a vanishing margin.

    A constant classifier, or a single-class cohort, gives an undefined
    coefficient rather than a measured one of zero. scikit-learn returns 0.0 and
    raises `UndefinedMetricWarning` while doing so -- its own warning is the
    evidence that the 0.0 is a fabrication, and reporting it as observed
    performance would make a classifier that never discriminated
    indistinguishable from one that discriminated and found nothing.
    """
    from .metrics import apply_decision_threshold, is_probability

    def predicate(ctx: MetricContext) -> Applicability:
        base = _requires_probabilities(ctx)
        if not base.applicable:
            return base
        predicted = apply_decision_threshold(
            ctx.y_prob, threshold=tp.threshold, operator=tp.operator.value)
        y = np.asarray(ctx.y_true, dtype=float).ravel()
        pos = y == 1
        tp_c = int(np.sum(predicted & pos))
        fp_c = int(np.sum(predicted & ~pos))
        fn_c = int(np.sum(~predicted & pos))
        tn_c = int(np.sum(~predicted & ~pos))
        if metric == "mcc":
            margins = ((tp_c + fp_c), (tp_c + fn_c), (tn_c + fp_c), (tn_c + fn_c))
            degenerate = any(m == 0 for m in margins)
        else:
            degenerate = (2 * tp_c + fp_c + fn_c) == 0
        if degenerate:
            return Applicability(
                applicable=False, status=MetricStatus.UNDEFINED,
                reason="degenerate_confusion_margin",
                metadata={"n_predicted_positive": int(np.sum(predicted)),
                          "n_reference_positive": int(np.sum(pos))})
        return APPLICABLE

    predicate._threshold_parameters = tp
    return predicate


def _requires_reference_labels_only(ctx: MetricContext) -> Applicability:
    """Prevalence is defined on a single-class population.

    Deliberately NOT inheriting the ranking metrics' both-classes rule: an
    all-negative cohort has a prevalence of 0.0 and an all-positive one of 1.0,
    and both are correct measurements rather than refusals. Certification
    eligibility remains a separate question -- a prevalence can be numerically
    valid on a cohort where no predictive metric is certifiable.
    """
    if ctx.y_true is None:
        return Applicability(applicable=False, status=MetricStatus.NOT_APPLICABLE,
                             reason="reference_labels_required")
    if ctx.n == 0:
        return Applicability(applicable=False, status=MetricStatus.INSUFFICIENT_DATA,
                             reason="empty_cohort", metadata={"n": 0})
    y = np.asarray(ctx.y_true, dtype=float).ravel()
    if not np.isfinite(y).all():
        return Applicability(
            applicable=False, status=MetricStatus.FAILED,
            reason="nonfinite_reference_labels",
            metadata={"n_nonfinite_labels": int((~np.isfinite(y)).sum())})
    return APPLICABLE


def _mce(ctx: MetricContext) -> float:
    from .metrics import maximum_calibration_error
    return maximum_calibration_error(ctx.y_true, ctx.y_prob,
                                     n_bins=_CALIBRATION_PARAMETERS["n_bins"])


def _prevalence(ctx: MetricContext) -> float:
    from .metrics import prevalence
    return prevalence(ctx.y_true)


def _make_mcc_adapter():
    from .metrics import matthews_correlation_coefficient
    return _threshold_adapter(matthews_correlation_coefficient, _MCC_THRESHOLD)


def _make_f1_adapter():
    from .metrics import f1_at_threshold
    return _threshold_adapter(f1_at_threshold, _F1_THRESHOLD)


_mcc = _make_mcc_adapter()
_f1 = _make_f1_adapter()


_PM = ResultKind.PREDICTION_METRIC
_PS = ResultKind.POPULATION_STATISTIC

REPORT_METRIC_NAMES: tuple = (
    "auroc",
    "auprc",
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "matthews_correlation_coefficient",
    "f1",
    "prevalence",
)

_METRICS: tuple = (
    MetricDescriptor(
        name="auroc", function=_auroc, required_inputs=frozenset({_L, _S}),
        applicability=_requires_both_classes, result_kind=_PM,
        display_name="Area under the receiver operating characteristic curve",
        description="area under the receiver operating characteristic curve"),
    MetricDescriptor(
        name="auprc", function=_auprc, required_inputs=frozenset({_L, _S}),
        applicability=_requires_both_classes, result_kind=_PM,
        display_name="Area under the precision-recall curve",
        description="area under the precision-recall curve"),
    MetricDescriptor(
        name="auprc_gain", function=_auprc_gain, required_inputs=frozenset({_L, _S}),
        applicability=_requires_both_classes, result_kind=_PM,
        display_name="Precision-recall gain over the no-skill floor",
        description="lift of the precision-recall area over the no-skill floor",
        include_in_evaluation_report=False),
    MetricDescriptor(
        name="brier_score", function=_brier, required_inputs=frozenset({_L, _P}),
        applicability=_requires_probabilities, result_kind=_PM,
        display_name="Brier score",
        description="mean squared error of the probability forecast"),
    MetricDescriptor(
        name="log_loss", function=_log_loss, required_inputs=frozenset({_L, _P}),
        applicability=_requires_probabilities, result_kind=_PM,
        display_name="Logarithmic loss",
        description="negative log likelihood of the probability forecast",
        include_in_evaluation_report=False),
    MetricDescriptor(
        name="expected_calibration_error", function=_ece,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_calibration_support, result_kind=_PM,
        display_name="Expected calibration error",
        description="occupancy-weighted binned gap between confidence and accuracy",
        parameters=dict(_CALIBRATION_PARAMETERS)),
    MetricDescriptor(
        name="maximum_calibration_error", function=_mce,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_calibration_support, result_kind=_PM,
        display_name="Maximum calibration error",
        description="largest binned gap between confidence and accuracy, over occupied bins",
        parameters=dict(_CALIBRATION_PARAMETERS)),
    MetricDescriptor(
        name="matthews_correlation_coefficient", function=_mcc,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_nondegenerate_confusion(_MCC_THRESHOLD, metric="mcc"),
        result_kind=_PM,
        display_name="Matthews correlation coefficient",
        description="correlation between hard predictions and reference labels",
        parameters=_MCC_THRESHOLD.to_mapping()),
    MetricDescriptor(
        name="f1", function=_f1, required_inputs=frozenset({_L, _P}),
        applicability=_requires_nondegenerate_confusion(_F1_THRESHOLD, metric="f1"),
        result_kind=_PM,
        display_name="F1 score, positive class",
        description="harmonic mean of positive-class precision and recall",
        parameters={**_F1_THRESHOLD.to_mapping(),
                    "average": "binary", "positive_label": 1,
                    "zero_division": "undefined"}),
    MetricDescriptor(
        name="prevalence", function=_prevalence, required_inputs=frozenset({_L}),
        applicability=_requires_reference_labels_only,
        result_kind=_PS,
        display_name="Prevalence",
        description="proportion of positive reference labels in the evaluation population"),
)


def _validate_registry(metrics: Sequence) -> None:
    """Validation of an IMMUTABLE DECLARATION, run at import.

    Not registration-time validation of a mutable table: there is nothing to
    register into. A malformed declaration fails the import rather than the run.
    """
    seen = set()
    for d in metrics:
        if not isinstance(d, MetricDescriptor):
            raise TypeError(f"registry holds a non-descriptor: {d!r}")
        if not d.name or d.name != d.name.strip().lower():
            raise ValueError(f"metric name must be nonempty and lower-case: {d.name!r}")
        if d.name in seen:
            raise ValueError(f"duplicate metric name in the registry: {d.name!r}")
        seen.add(d.name)
        if not d.required_inputs:
            raise ValueError(f"{d.name}: required_inputs must not be empty")
        if not all(isinstance(i, MetricInput) for i in d.required_inputs):
            raise TypeError(f"{d.name}: required_inputs must hold MetricInput members")
        if MetricInput.LABELS not in d.required_inputs:
            raise ValueError(f"{d.name}: every metric requires LABELS")
        if d.requires_clusters and MetricInput.CLUSTERS not in d.required_inputs:
            raise ValueError(
                f"{d.name}: requires_clusters=True but CLUSTERS is not in "
                "required_inputs; the two would disagree about what is needed")
        if d.applicability is None or not callable(d.applicability):
            raise ValueError(f"{d.name}: an applicability policy is mandatory")
        if not callable(d.function):
            raise ValueError(f"{d.name}: function must be callable")
        # --- descriptor schema, version 2 (2026-07-27) --------------------
        # Every descriptor speaks the same vocabulary. Enforced at IMPORT so a
        # later addition cannot create a second dialect in which some
        # descriptors declare their classification and parameters and others
        # leave them implicit.
        if not isinstance(d.result_kind, ResultKind):
            raise TypeError(f"{d.name}: result_kind must be a ResultKind member")
        if not d.display_name or not d.display_name.strip():
            raise ValueError(f"{d.name}: display_name must be non-empty")
        if not d.description or not d.description.strip():
            raise ValueError(f"{d.name}: description must be non-empty")
        if not isinstance(d.parameters, MappingProxyType):
            raise TypeError(
                f"{d.name}: parameters must be an immutable mapping; a plain "
                "dict on a frozen descriptor is mutable in fact while frozen "
                "in type")
        # ONE ThresholdParameters, shared by identity. Equal-but-separate copies
        # are how a threshold comes to differ later.
        tp = d.threshold_parameters
        declared = "decision_threshold" in d.parameters
        if declared != (tp is not None):
            raise ValueError(
                f"{d.name}: a declared decision_threshold and a typed "
                "ThresholdParameters must appear together; one without the "
                "other means the mapping and the kernel can disagree")
        if tp is not None:
            if getattr(d.applicability, "_threshold_parameters", None) is not tp:
                raise ValueError(
                    f"{d.name}: the applicability predicate does not share the "
                    "SAME ThresholdParameters object as the kernel; two "
                    "thresholds that merely happen to be equal today will not "
                    "stay equal")
            if d.parameters["decision_threshold"] != tp.threshold or \
                    d.parameters["threshold_operator"] != tp.operator.value or \
                    d.parameters["threshold_source"] != tp.source.value:
                raise ValueError(
                    f"{d.name}: the serialised parameters disagree with the "
                    "typed ThresholdParameters they are supposed to serialise")
        if d.result_kind is ResultKind.POPULATION_STATISTIC:
            for forbidden in (MetricInput.SCORES, MetricInput.PROBABILITIES):
                if forbidden in d.required_inputs:
                    raise ValueError(
                        f"{d.name}: a population statistic describes the cohort, "
                        f"not the predictions, and must not require "
                        f"{forbidden.value}")


_validate_registry(_METRICS)


def _validate_report_completeness() -> None:
    """Every quantity the evaluation report carries must come from the registry.

    Stronger than testing each new descriptor individually: it prevents a future
    report field from being added anywhere else. If a quantity appears in a
    report without a descriptor, it has no applicability policy, no certification
    rule, no declared parameters and no population -- which is the condition the
    whole metric stack was built to end.
    """
    declared = {d.name for d in _METRICS if d.include_in_evaluation_report}
    expected = set(REPORT_METRIC_NAMES)
    if declared != expected:
        missing = sorted(expected - declared)
        extra = sorted(declared - expected)
        raise ValueError(
            "the registry's report set does not match REPORT_METRIC_NAMES; "
            f"missing={missing} unexpected={extra}")


_validate_report_completeness()


# --------------------------------------------------------------------------- #
# Accessors -- order-preserving, returning the frozen declaration
# --------------------------------------------------------------------------- #
def all_metrics() -> tuple:
    return _METRICS


def names() -> tuple:
    return tuple(d.name for d in _METRICS)


def by_name(name: str) -> MetricDescriptor:
    for d in _METRICS:
        if d.name == name:
            return d
    raise KeyError(f"no registered metric named {name!r}; registered: {names()}")


def requiring_clusters() -> tuple:
    return tuple(d for d in _METRICS if d.requires_clusters)


def requiring_input(what: MetricInput) -> tuple:
    return tuple(d for d in _METRICS if what in d.required_inputs)


# --------------------------------------------------------------------------- #
# Typed execution
# --------------------------------------------------------------------------- #
def _missing_inputs(d: MetricDescriptor, ctx: MetricContext) -> tuple:
    return tuple(sorted(i.value for i in d.required_inputs if not ctx.has(i)))


def compute(d: MetricDescriptor, ctx: MetricContext) -> MetricResult:
    """Compute ONE registered metric, or explain why it was not computed.

    The order is load-bearing. Inputs, then applicability, then -- only then --
    the kernel. A metric ruled inapplicable is NEVER INVOKED, so a finite but
    scientifically unsupported number cannot be produced and then explained away.
    """
    nan = float("nan")

    missing = _missing_inputs(d, ctx)
    if missing:
        return MetricResult(
            value=nan, status=MetricStatus.NOT_APPLICABLE,
            reason="required_inputs_missing",
            metadata={MetricMetadataKey.METRIC_NAME: d.name, "missing_inputs": list(missing),
                      **ctx.support()})

    verdict = d.applicability(ctx)
    if not verdict.applicable:
        return MetricResult(
            value=nan, status=verdict.status, reason=verdict.reason,
            metadata={MetricMetadataKey.METRIC_NAME: d.name, **ctx.support(),
                      **dict(verdict.metadata)})

    try:
        raw = d.function(ctx)
    except Exception as exc:                                   # noqa: BLE001
        return MetricResult(
            value=nan, status=MetricStatus.FAILED,
            reason="metric_computation_failed",
            metadata={MetricMetadataKey.METRIC_NAME: d.name,
                      "exception_type": type(exc).__name__,
                      # the message is recorded for a human, but the machine-
                      # readable reason above is stable; exception text is not.
                      "exception_message": str(exc)[:200],
                      **ctx.support()})

    value = float(raw)
    if not np.isfinite(value):
        # APPLICABLE and non-finite is an implementation defect, not a property
        # of the cohort. Calling it UNDEFINED would blame the data.
        return MetricResult(
            value=nan, status=MetricStatus.FAILED,
            reason="applicable_metric_returned_non_finite",
            metadata={MetricMetadataKey.METRIC_NAME: d.name, "returned": repr(raw),
                      **ctx.support()})

    # NUMERIC COMPUTABILITY, SCIENTIFIC INTERPRETABILITY AND CERTIFICATION
    # ELIGIBILITY ARE THREE DIFFERENT THINGS. A Brier score on a single-class
    # cohort is a correct proper-score calculation -- status OK -- and is still
    # not something an inferential claim may rest on. An earlier version of this
    # function hard-coded certification_eligible=True for every OK result, which
    # collapsed the third axis into the first.
    eligible, why = _certification_eligibility(d, ctx)
    meta = {MetricMetadataKey.METRIC_NAME: d.name,
            MetricMetadataKey.CERTIFICATION_ELIGIBLE: eligible,
            **ctx.support()}
    if not eligible:
        meta[MetricMetadataKey.CERTIFICATION_BLOCKED_BY] = why
    return MetricResult(value=value, status=MetricStatus.OK, metadata=meta)


def _certification_eligibility(d: MetricDescriptor, ctx: MetricContext) -> tuple:
    """May a computed value support a certified claim?

    Separate from `status`, which answers whether a value exists at all. The
    bootstrap work established the same separation for intervals: `status` asks
    "was an interval produced?", `certification_eligible` asks "is it admissible
    for certified claims?" -- independent axes.

    Returns (eligible, reason_if_not).
    """
    if not ctx.has_both_classes:
        return False, "single_class_cohort"
    if ctx.n == 0:
        return False, "empty_cohort"
    return True, None


def evaluate_registered(ctx: MetricContext, *,
                        only: Optional[Sequence] = None) -> dict:
    """The canonical typed evaluation path.

    Returns one MetricResult per registered metric, in registry order. Every
    entry is present: a metric that could not be computed is reported with a
    status and a reason, never omitted, because an absent key and a refused
    metric are different facts and a caller cannot tell them apart.
    """
    chosen = _METRICS if only is None else tuple(by_name(n) for n in only)
    return {d.name: compute(d, ctx) for d in chosen}
