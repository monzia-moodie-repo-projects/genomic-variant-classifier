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
from enum import Enum
from typing import Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

from .capabilities import MetricResult, MetricStatus

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

REGISTRY_SCHEMA_VERSION = 1


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
    population_scope: str
    y_score: Optional[np.ndarray] = None
    y_prob: Optional[np.ndarray] = None
    clusters: Optional[np.ndarray] = None
    sample_weight: Optional[np.ndarray] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.population_scope, str) or not self.population_scope.strip():
            raise ValueError(
                "population_scope is REQUIRED and must be a nonempty string. "
                "Support counts alone do not identify the denominator: 53 and 63 "
                "were both correct and described different populations, and 85 "
                "and 107 were printed as a partition of 107. A number without its "
                "population is not evidence.")
        n = np.asarray(self.y_true).size
        for name in ("y_score", "y_prob", "clusters", "sample_weight"):
            v = getattr(self, name)
            if v is not None and np.asarray(v).size != n:
                raise ValueError(
                    f"{name} has {np.asarray(v).size} rows but y_true has {n}; "
                    "the context is aligned ONCE, here, so a descriptor can never "
                    "compute over a mismatched pairing")

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
        out = {"population_scope": self.population_scope,
               "n_observations": self.n,
               "n_classes_observed": self.n_classes_observed}
        if self.clusters is not None:
            out["n_clusters"] = self.n_clusters
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
    requires_clusters: bool = False
    output_kind: MetricOutputKind = MetricOutputKind.SCALAR
    description: str = ""


# --------------------------------------------------------------------------- #
# Applicability predicates
# --------------------------------------------------------------------------- #
def _requires_both_classes(ctx: MetricContext) -> Applicability:
    """Ranking is undefined when one class is present, and the kernel says so:
    'A single class present -> ranking metrics are undefined. Say so; do not
    guess.' (metrics.py:225)"""
    if ctx.has_both_classes:
        return APPLICABLE
    return Applicability(
        applicable=False,
        status=MetricStatus.UNDEFINED,
        reason="binary_class_support_required",
        metadata={"n_classes_observed": ctx.n_classes_observed,
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
    if not is_probability(ctx.y_prob):
        return Applicability(applicable=False, status=MetricStatus.NOT_APPLICABLE,
                             reason="values_are_not_probabilities")
    if not ctx.has_both_classes:
        return Applicability(
            applicable=False, status=MetricStatus.INSUFFICIENT_SUPPORT,
            reason="calibration_requires_class_support",
            metadata={"n_classes_observed": ctx.n_classes_observed,
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

_METRICS: tuple = (
    MetricDescriptor(
        name="auroc", function=_auroc, required_inputs=frozenset({_L, _S}),
        applicability=_requires_both_classes,
        description="area under the receiver operating characteristic curve"),
    MetricDescriptor(
        name="auprc", function=_auprc, required_inputs=frozenset({_L, _S}),
        applicability=_requires_both_classes,
        description="area under the precision-recall curve"),
    MetricDescriptor(
        name="auprc_gain", function=_auprc_gain, required_inputs=frozenset({_L, _S}),
        applicability=_requires_both_classes,
        description="lift of the precision-recall area over the no-skill floor"),
    MetricDescriptor(
        name="brier_score", function=_brier, required_inputs=frozenset({_L, _P}),
        applicability=_requires_probabilities,
        description="mean squared error of the probability forecast"),
    MetricDescriptor(
        name="log_loss", function=_log_loss, required_inputs=frozenset({_L, _P}),
        applicability=_requires_probabilities,
        description="negative log likelihood of the probability forecast"),
    MetricDescriptor(
        name="expected_calibration_error", function=_ece,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_calibration_support,
        description="binned gap between confidence and accuracy"),
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


_validate_registry(_METRICS)


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
            metadata={"metric_name": d.name, "missing_inputs": list(missing),
                      **ctx.support()})

    verdict = d.applicability(ctx)
    if not verdict.applicable:
        return MetricResult(
            value=nan, status=verdict.status, reason=verdict.reason,
            metadata={"metric_name": d.name, **ctx.support(),
                      **dict(verdict.metadata)})

    try:
        raw = d.function(ctx)
    except Exception as exc:                                   # noqa: BLE001
        return MetricResult(
            value=nan, status=MetricStatus.FAILED,
            reason="metric_computation_failed",
            metadata={"metric_name": d.name,
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
            metadata={"metric_name": d.name, "returned": repr(raw),
                      **ctx.support()})

    # NUMERIC COMPUTABILITY, SCIENTIFIC INTERPRETABILITY AND CERTIFICATION
    # ELIGIBILITY ARE THREE DIFFERENT THINGS. A Brier score on a single-class
    # cohort is a correct proper-score calculation -- status OK -- and is still
    # not something an inferential claim may rest on. An earlier version of this
    # function hard-coded certification_eligible=True for every OK result, which
    # collapsed the third axis into the first.
    eligible, why = _certification_eligibility(d, ctx)
    meta = {"metric_name": d.name, "certification_eligible": eligible,
            **ctx.support()}
    if not eligible:
        meta["certification_blocked_by"] = why
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
