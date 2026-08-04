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

import math

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
class RegistryInvariantError(RuntimeError):
    """A registry contract was violated by the registry's own configuration.

    Distinct from an applicability refusal, which describes the COHORT. This
    describes a defect in how a metric was DECLARED, and is therefore never a
    result status: it is raised at dispatch so the declaration is fixed rather
    than the number reinterpreted.
    """


class ResultKind(str, Enum):
    """What KIND of quantity a descriptor produces."""

    PREDICTION_METRIC = "prediction_metric"
    POPULATION_STATISTIC = "population_statistic"


# THR-1 (2026-08-04): the canonical threshold vocabulary now lives in
# `thresholds.py`, a module beneath both this one and `metrics.py`.
#
# IT MOVED DOWN so that OP-1's exact threshold sweep can describe each swept
# candidate with a `ThresholdParameters` WITHOUT importing this module. A sweep
# that imported `registry.py` would reverse the layering, and future registry
# count-applicability code could not then import the sweep without a cycle.
# `metrics.py` was the other candidate home and is worse: it imports
# scikit-learn at module level, which would put a reusable algorithm behind the
# import boundary `evaluation/__init__.py` exists to police.
#
# RE-EXPORTED HERE, and the re-export preserves OBJECT IDENTITY rather than mere
# equality. `ThresholdParameters` documents that one instance is shared by a
# descriptor, its kernel adapter and its applicability predicate, asserted BY
# IDENTITY at import time. A re-export producing a distinct class object would
# leave every isinstance() check comparing against a different type, silently,
# until something asserted identity. `test_threshold_vocabulary.py` proves the
# binding with `is`.
#
# Nothing else changed: the classes moved verbatim, and this commit alters no
# behaviour, no value and no test outcome.
from .thresholds import (  # noqa: E402  (re-export, deliberately in place)
    ThresholdOperator,
    ThresholdParameters,
    ThresholdSource,
)


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


# Keys that `support()` also supplies but that a DESCRIPTOR legitimately owns
# when it REFUSES, because on that path the key is the descriptor's ARGUMENT
# rather than a registry fact it would be contradicting.
#
# N_CLASSES_OBSERVED is the only member, and it is not one descriptor's quirk:
# probed 2026-08-03 across four cohort shapes over every registered descriptor,
# 27 refusals were observed and SEVEN metrics reported this key as the ground of
# their refusal -- "there is one class, therefore this metric is undefined". The
# other three `support()` keys (N_OBSERVATIONS, POPULATION_FINGERPRINT,
# POPULATION_SCOPE) were set by NO descriptor on ANY of those refusals.
#
# This is a SUBTRAHEND, not a list of what is protected. The protected set stays
# DERIVED from `support()`, so a future key added there is protected on both
# paths the moment it exists. Only the exception is named, and
# test_the_refusal_ownership_exception_is_still_exhaustive re-derives it from the
# live descriptor graph so an eighth metric adopting a different key fails loudly
# instead of quietly widening this frozenset.
_DESCRIPTOR_OWNED_ON_REFUSAL = frozenset({MetricMetadataKey.N_CLASSES_OBSERVED})


def _reject_registry_owned_keys(d: "MetricDescriptor", ctx: "MetricContext",
                                verdict: "Applicability",
                                protected: frozenset) -> None:
    """Refuse descriptor metadata that would overwrite a registry-owned key.

    EXTRACTED 2026-08-03 (REG-1) FROM THE OK BRANCH, WHERE IT WAS THE ONLY PLACE
    IT RAN. The refusal branch merged `verdict.metadata` LAST with no check, so
    the same metadata that raised on the applicable path was silently accepted on
    the refusal path -- the branch whose whole purpose is saying what evidence
    base the refusal describes. A refusal could claim a membership fingerprint it
    never examined, and "n=980 beside n=980 says nothing about WHICH 980".

    THE PROTECTED SET IS A PARAMETER because the two paths DO NOT OWN THE SAME
    KEYS. A first version of this change derived one set for both and turned 29
    tests red: `auroc` refusing a single-class cohort reports N_CLASSES_OBSERVED
    as the GROUND of its refusal, and the guard called that a violation. The
    derivation is still single-sourced; only the ownership differs, because the
    paths genuinely differ.

    REJECTED, NOT SHADOWED. Merge order would also prevent the overwrite, but
    silently: the descriptor's value would vanish and its author would get no
    signal. That reasoning is recorded in `compute` and is why neither branch's
    merge order was changed.
    """
    overlap = protected & set(verdict.metadata)
    if overlap:
        raise RegistryInvariantError(
            f"{d.name}: applicability metadata attempted to set registry-owned "
            f"key(s) {sorted(str(k) for k in overlap)}. A descriptor states what "
            "is true of the COHORT; the registry states what is true of the "
            "RESULT, and the two must not be able to disagree.")


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
    """Calibration needs a probability vector over a non-empty cohort.

    REVERSED 2026-07-28. This predicate previously ALSO required both reference
    classes, on the reasoning that a single-class calibration figure is
    "computable but scientifically empty". That reasoning conflated two
    different estimands.

    Discrimination asks whether predictions RANK one class against another, and a
    single-class cohort cannot support it. Calibration asks whether predicted
    probabilities MATCH observed event frequencies, and a single-class cohort
    can. Measured on an all-negative cohort where every prediction is 0.10: one
    occupied bin, observed frequency 0.00, mean prediction 0.10, absolute gap
    0.10. That number is not empty -- it measures systematic OVERPREDICTION of
    the event probability in that population. The mirror case, an all-positive
    cohort predicted at 0.90, measures underprediction.

    What a single-class cohort genuinely limits is INTERPRETATION: discrimination
    is unavailable, the outcome distribution is narrow, most bins are unoccupied,
    and generalisation beyond that population is weak. Those limits belong in
    metadata, in certification policy, and in reporting -- not in an applicability
    rule that declares the arithmetic undefined.

    So the result is OK, and carries `reference_class_support` recording the
    structure. The diagnostic is deliberately NEUTRAL rather than a warning: the
    figure is a correct measurement over a narrow distribution, not a defect.

    WHAT IS NOT CHECKED HERE. "At least one occupied bin" is NOT an applicability
    condition. It is a theorem of the conditions above -- a non-empty vector of
    valid probabilities in [0, 1] under equal-width binning with a closed top
    edge must occupy a bin -- and therefore unreachable through this path. Were
    it ever violated, the correct verdict would be FAILED, an implementation
    defect, not INSUFFICIENT_SUPPORT, a cohort property. It is enforced as a
    representation invariant inside `CalibrationBins`, which is the layer that
    owns it.
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
    # The single-class STRUCTURE is recorded, not refused. See the docstring.
    if not ctx.has_both_classes:
        return Applicability(
            applicable=True, status=None, reason=None,
            # N_CLASSES_OBSERVED is deliberately NOT supplied here: ctx.support()
            # already provides it, and a verdict that restates a registry-owned
            # key would collide with the protected set below.
            metadata={MetricMetadataKey.REFERENCE_CLASS_SUPPORT: "single_class"})
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
# ONE declaration for the whole confusion family. The seven metrics below
# describe ONE matrix at ONE cut, and a shared object is what stops them
# disagreeing about where that cut was -- the same identity discipline commit
# 2b-2 applied to the Matthews coefficient and F1.
_CONFUSION_THRESHOLD = ThresholdParameters(
    threshold=0.5,
    operator=ThresholdOperator.GREATER_OR_EQUAL,
    source=ThresholdSource.FIXED_DEFAULT)

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

#: The false-positive band is part of `partial_auroc`'s IDENTITY, exactly as a
#: threshold is for the confusion family. Two partial areas over different bands
#: are two different metrics, and one object shared by the descriptor and the
#: adapter is what stops them disagreeing about which band was used -- the same
#: identity discipline commit 2b-2 applied to the Matthews coefficient and F1.
#:
#: 0.0 to 0.1 is the clinically relevant region the specification names at line
#: 344: "partial AUROC within clinically relevant FPR regions". A variant screen
#: that flags one healthy genome in ten is already at the edge of usable, so the
#: area above that band describes operating points nobody would adopt.
_PARTIAL_AUROC_BAND = {
    "fpr_low": 0.0,
    "fpr_high": 0.1,
    "standardisation": "mcclish",
}

#: Equal-MASS binning, which is a different convention from the equal-width one
#: above and therefore a different parameter object. Sharing
#: `_CALIBRATION_PARAMETERS` would have declared `binning="equal_width"` for a
#: kernel that does no such thing.
_ADAPTIVE_CALIBRATION_PARAMETERS = {
    "n_bins": 10,
    "binning": "equal_mass",
    "interval_convention": "tied groups never split; a heavy group takes a bin",
}


def _partial_auroc_adapter(ctx: "MetricContext") -> float:
    """Bind the declared band. The import is deferred for the reason
    `_threshold_adapter` gives: `metrics` imports scikit-learn and this package
    guarantees that `registry` imports without it."""
    from . import metrics
    return metrics.partial_auroc(
        ctx.y_true, ctx.y_score,
        fpr_low=_partial_auroc_adapter._declared_parameters["fpr_low"],
        fpr_high=_partial_auroc_adapter._declared_parameters["fpr_high"])


_partial_auroc_adapter._declared_parameters = _PARTIAL_AUROC_BAND


def _integrated_calibration_index_adapter(ctx: "MetricContext") -> float:
    """No parameters, and that is the point: this metric is binning-free, so it
    has no interval convention to declare and none to get wrong."""
    from . import metrics
    return metrics.integrated_calibration_index(ctx.y_true, ctx.y_prob)


def _adaptive_ece_adapter(ctx: "MetricContext") -> float:
    from . import metrics
    return metrics.adaptive_expected_calibration_error(
        ctx.y_true, ctx.y_prob,
        n_bins=_adaptive_ece_adapter._declared_parameters["n_bins"])


_adaptive_ece_adapter._declared_parameters = _ADAPTIVE_CALIBRATION_PARAMETERS


def _threshold_adapter(kernel_name: str, tp: ThresholdParameters):
    """Bind a kernel BY NAME, resolving it at call time.

    THE IMPORT MUST NOT HAPPEN AT MODULE SCOPE. `metrics` imports scikit-learn,
    and this package guarantees that `evaluation/__init__` -- and therefore
    `registry` -- imports without it. Every other consumer in this module already
    defers the import inside a function body for exactly that reason.

    The threshold adapters were the exception: they were built by a factory
    invoked at module scope, so `from .metrics import ...` ran at import. That was
    latent while nothing imported `registry` at module level, and became a real
    defect the moment `evaluator` did. Binding by NAME keeps the descriptor
    declaration eager -- which the registry validator requires -- while leaving
    the kernel resolution lazy.
    """
    def adapter(ctx: MetricContext) -> float:
        from . import metrics
        kernel = getattr(metrics, kernel_name)
        return kernel(ctx.y_true, ctx.y_prob, threshold=tp.threshold,
                      operator=tp.operator.value)
    adapter._threshold_parameters = tp
    adapter.__name__ = f"_{kernel_name}_adapter"
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
    def predicate(ctx: MetricContext) -> Applicability:
        # Deferred for the same reason as the adapter above: this factory runs at
        # module scope while building the descriptor table, so an import here
        # would pull scikit-learn into `evaluation/__init__`.
        from .metrics import apply_decision_threshold

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
            # DISTINCT REASONS PER METRIC, 2026-07-28. A single shared reason
            # would let the legacy compatibility projection substitute the
            # Matthews value for an F1 undefined for an entirely different
            # cause. The substitution must be authorised by metric identity AND
            # by the exact undefined reason, so the two conditions -- a
            # vanishing confusion-matrix margin, and a vanishing F1 denominator
            # -- must be nameable apart.
            return Applicability(
                applicable=False, status=MetricStatus.UNDEFINED,
                reason=("zero_confusion_margin" if metric == "mcc"
                        else "zero_f1_denominator"),
                metadata={"n_predicted_positive": int(np.sum(predicted)),
                          "n_reference_positive": int(np.sum(pos))})
        return APPLICABLE

    predicate._threshold_parameters = tp
    return predicate


# --------------------------------------------------------------------------- #
# CONFUSION-FAMILY APPLICABILITY (2026-07-29)
#
# Each of the seven confusion-matrix metrics returns NaN on its own degenerate
# case, and the registry requires an OK result to be finite -- so applicability
# must refuse EXACTLY where the kernel would return NaN. Refusing more widely
# would repeat the calibration over-restriction corrected in commit 3b-1a, where
# a metric was withheld on a cohort that could perfectly well support it.
#
# The requirements differ per metric and are not interchangeable:
#
#   sensitivity                 positive labels        (TP + FN > 0)
#   specificity                 negative labels        (TN + FP > 0)
#   positive predictive value   something flagged      (TP + FP > 0)  threshold-dependent
#   negative predictive value   something cleared      (TN + FN > 0)  threshold-dependent
#   balanced accuracy           BOTH classes
#   likelihood ratios           both classes, and a specificity strictly inside
#                               (0, 1) -- an infinite ratio is not a value an
#                               evidence artifact can carry
# --------------------------------------------------------------------------- #
def _requires_class_support(tp: ThresholdParameters, *, positive: bool,
                            metric: str):
    """Applicability requiring observations of one reference class.

    Takes the ThresholdParameters even though class support does not depend on
    the threshold, because the registry validator asserts BY IDENTITY that a
    descriptor's predicate and kernel share one object. Declaring it here is what
    makes that assertion meaningful rather than vacuous.
    """
    def predicate(ctx: MetricContext) -> Applicability:
        base = _requires_probabilities(ctx)
        if not base.applicable:
            return base
        wanted = 1.0 if positive else 0.0
        if wanted not in ctx.classes_observed:
            label = "positive" if positive else "negative"
            return Applicability(
                applicable=False, status=MetricStatus.INSUFFICIENT_SUPPORT,
                reason=f"{label}_class_support_required",
                metadata={MetricMetadataKey.REFERENCE_CLASS_SUPPORT: "single_class"})
        return APPLICABLE
    predicate.__name__ = f"_requires_{'positive' if positive else 'negative'}_support_{metric}"
    predicate._threshold_parameters = tp
    return predicate


def _requires_flagged_margin(tp: ThresholdParameters, *, flagged: bool,
                             metric: str):
    """Applicability requiring a non-empty predicted-positive or -negative set.

    THRESHOLD-DEPENDENT, unlike class support: whether anything is flagged is a
    property of the threshold and the predictions together, not of the labels.
    """
    def predicate(ctx: MetricContext) -> Applicability:
        base = _requires_probabilities(ctx)
        if not base.applicable:
            return base
        from .metrics import apply_decision_threshold

        predicted = apply_decision_threshold(
            ctx.y_prob, threshold=tp.threshold, operator=tp.operator.value)
        count = int(predicted.sum()) if flagged else int((~predicted).sum())
        if count == 0:
            side = "predicted_positive" if flagged else "predicted_negative"
            # UNDEFINED, not INSUFFICIENT_SUPPORT. Corrected 2026-08-04 (REG-2).
            #
            # A predictive value with an empty denominator is MATHEMATICALLY
            # UNDEFINED: `TP/(TP+FP)` has no value when `TP+FP = 0`. The enum
            # reserves INSUFFICIENT_SUPPORT for "the machinery is ready and the
            # science is not" -- a cohort-support judgement, which this is not.
            #
            # The block above states the intent this restores: applicability must
            # refuse EXACTLY where the kernel would return NaN, and it lists the
            # requirement as `TP + FP > 0`, marked threshold-dependent. That is a
            # denominator condition.
            #
            # NOT A CLOSE CALL. Measured across 24 descriptors and 6 cohort
            # shapes: ten metrics already return UNDEFINED for mathematically
            # undefined states -- binary_class_support_required (7),
            # likelihood_ratio_unbounded (2), zero_confusion_margin (1) -- while
            # only these two predictive values used INSUFFICIENT_SUPPORT for one.
            # The registry contradicted itself; this makes two agree with ten.
            #
            # `_requires_class_support` is DELIBERATELY UNCHANGED: an absent
            # reference class IS a support problem, and its two metrics are
            # correct as they stand.
            #
            # The reason string is unchanged. It was accurate before and after.
            return Applicability(
                applicable=False, status=MetricStatus.UNDEFINED,
                reason=f"empty_{side}_set",
                metadata={"threshold": tp.threshold})
        return APPLICABLE
    predicate.__name__ = f"_requires_{'flagged' if flagged else 'cleared'}_{metric}"
    predicate._threshold_parameters = tp
    return predicate


def _requires_interior_specificity(tp: ThresholdParameters, *, metric: str):
    """Applicability for the likelihood ratios.

    Both classes must be present, and specificity must lie strictly inside
    (0, 1). At specificity exactly 1.0 the positive likelihood ratio is infinite;
    at exactly 0.0 the negative one is. Infinity is not a value an artifact can
    carry, so this refuses rather than producing one.
    """
    def predicate(ctx: MetricContext) -> Applicability:
        base = _requires_both_classes(ctx)
        if not base.applicable:
            return base
        from .metrics import specificity as _spec

        spec = _spec(ctx.y_true, ctx.y_prob, threshold=tp.threshold,
                     operator=tp.operator.value)
        if not math.isfinite(spec):
            return Applicability(applicable=False,
                                 status=MetricStatus.INSUFFICIENT_SUPPORT,
                                 reason="specificity_undefined")
        if spec >= 1.0 or spec <= 0.0:
            return Applicability(
                applicable=False, status=MetricStatus.UNDEFINED,
                reason="likelihood_ratio_unbounded",
                metadata={"specificity": float(spec), "threshold": tp.threshold})
        return APPLICABLE
    predicate.__name__ = f"_requires_interior_specificity_{metric}"
    predicate._threshold_parameters = tp
    return predicate

def _requires_both_classes_at_threshold(tp: ThresholdParameters, *, metric: str):
    """Both reference classes, and nothing more.

    ADDED AFTER A DEFECT OF MINE. Balanced accuracy was first given the
    likelihood-ratio predicate purely to satisfy the identity validator, and it
    inherited that predicate's refusal at specificity exactly 1.0 -- so a PERFECT
    classifier reported balanced accuracy as `undefined` with reason
    `likelihood_ratio_unbounded`. It is perfectly well defined there: (1 + 1) / 2.

    That is the same over-restriction corrected in commit 3b-1a, where calibration
    was withheld on a cohort that could support it. Borrowing a predicate because
    it typechecks is how a metric acquires a restriction nobody intended.
    """
    def predicate(ctx: MetricContext) -> Applicability:
        return _requires_both_classes(ctx)
    predicate.__name__ = f"_requires_both_classes_at_threshold_{metric}"
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


def _calibration_adapter(kernel_name: str):
    """Bind a calibration kernel to the bin count its DESCRIPTOR declares.

    Until 2026-07-28 these adapters read the module constant directly, so a
    descriptor could DECLARE one bin count and COMPUTE with another and only an
    indirect test would notice. The threshold metrics were never exposed to that:
    commit 2b-2 bound them to one shared `ThresholdParameters` object, asserted
    by identity. Calibration is now bound the same way -- the declaration is the
    parameter, not a description of one.
    """
    def adapter(ctx: MetricContext) -> float:
        from . import metrics
        kernel = getattr(metrics, kernel_name)
        return kernel(ctx.y_true, ctx.y_prob,
                      n_bins=adapter._declared_parameters["n_bins"])
    adapter._declared_parameters = _CALIBRATION_PARAMETERS
    adapter.__name__ = f"_{kernel_name}_adapter"
    return adapter


_mce = _calibration_adapter("maximum_calibration_error")
_brier_reliability = _calibration_adapter("brier_reliability")
_brier_resolution = _calibration_adapter("brier_resolution")
_brier_uncertainty = _calibration_adapter("brier_uncertainty")
_brier_residual = _calibration_adapter("brier_decomposition_residual")



def _prevalence(ctx: MetricContext) -> float:
    from .metrics import prevalence
    return prevalence(ctx.y_true)


_sensitivity = _threshold_adapter("sensitivity", _CONFUSION_THRESHOLD)
_specificity = _threshold_adapter("specificity", _CONFUSION_THRESHOLD)
_ppv = _threshold_adapter("positive_predictive_value", _CONFUSION_THRESHOLD)
_npv = _threshold_adapter("negative_predictive_value", _CONFUSION_THRESHOLD)
_balanced_accuracy = _threshold_adapter("balanced_accuracy", _CONFUSION_THRESHOLD)
_lr_positive = _threshold_adapter("positive_likelihood_ratio", _CONFUSION_THRESHOLD)
_lr_negative = _threshold_adapter("negative_likelihood_ratio", _CONFUSION_THRESHOLD)
_mcc = _threshold_adapter("matthews_correlation_coefficient", _MCC_THRESHOLD)
_f1 = _threshold_adapter("f1_at_threshold", _F1_THRESHOLD)


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
        name="sensitivity", function=_sensitivity,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_class_support(
            _CONFUSION_THRESHOLD, positive=True, metric="sensitivity"),
        # REGISTERED BUT NOT IN THE FLAT REPORT (2026-07-29).
        #
        # These are computable now. Adding them to the report surface would move
        # the frozen 480-value oracle, which is a separate and declared change --
        # the same staging that kept commit 3a's schema introduction apart from
        # 3b-2's authority switch. Registered first, surfaced second.
        include_in_evaluation_report=False,
        result_kind=_PM,
        display_name="Sensitivity (recall, true-positive rate)",
        description="proportion of positive reference labels correctly flagged",
        parameters={**_CONFUSION_THRESHOLD.to_mapping(), "zero_division": "undefined"}),
    MetricDescriptor(
        name="specificity", function=_specificity,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_class_support(
            _CONFUSION_THRESHOLD, positive=False, metric="specificity"),
        # REGISTERED BUT NOT IN THE FLAT REPORT (2026-07-29).
        #
        # These are computable now. Adding them to the report surface would move
        # the frozen 480-value oracle, which is a separate and declared change --
        # the same staging that kept commit 3a's schema introduction apart from
        # 3b-2's authority switch. Registered first, surfaced second.
        include_in_evaluation_report=False,
        result_kind=_PM,
        display_name="Specificity (true-negative rate)",
        description="proportion of negative reference labels correctly cleared",
        parameters={**_CONFUSION_THRESHOLD.to_mapping(), "zero_division": "undefined"}),
    MetricDescriptor(
        name="positive_predictive_value", function=_ppv,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_flagged_margin(_CONFUSION_THRESHOLD, flagged=True,
                                               metric="positive_predictive_value"),
        # REGISTERED BUT NOT IN THE FLAT REPORT (2026-07-29).
        #
        # These are computable now. Adding them to the report surface would move
        # the frozen 480-value oracle, which is a separate and declared change --
        # the same staging that kept commit 3a's schema introduction apart from
        # 3b-2's authority switch. Registered first, surfaced second.
        include_in_evaluation_report=False,
        result_kind=_PM,
        display_name="Positive predictive value (precision)",
        description="proportion of flagged variants that are truly positive; "
                    "PREVALENCE-DEPENDENT and not transferable between cohorts",
        parameters={**_CONFUSION_THRESHOLD.to_mapping(),
                    "prevalence_dependent": True}),
    MetricDescriptor(
        name="negative_predictive_value", function=_npv,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_flagged_margin(_CONFUSION_THRESHOLD, flagged=False,
                                               metric="negative_predictive_value"),
        # REGISTERED BUT NOT IN THE FLAT REPORT (2026-07-29).
        #
        # These are computable now. Adding them to the report surface would move
        # the frozen 480-value oracle, which is a separate and declared change --
        # the same staging that kept commit 3a's schema introduction apart from
        # 3b-2's authority switch. Registered first, surfaced second.
        include_in_evaluation_report=False,
        result_kind=_PM,
        display_name="Negative predictive value",
        description="proportion of cleared variants that are truly negative; "
                    "PREVALENCE-DEPENDENT and not transferable between cohorts",
        parameters={**_CONFUSION_THRESHOLD.to_mapping(),
                    "prevalence_dependent": True}),
    MetricDescriptor(
        name="balanced_accuracy", function=_balanced_accuracy,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_both_classes_at_threshold(
            _CONFUSION_THRESHOLD, metric="balanced_accuracy"),
        # REGISTERED BUT NOT IN THE FLAT REPORT (2026-07-29).
        #
        # These are computable now. Adding them to the report surface would move
        # the frozen 480-value oracle, which is a separate and declared change --
        # the same staging that kept commit 3a's schema introduction apart from
        # 3b-2's authority switch. Registered first, surfaced second.
        include_in_evaluation_report=False,
        result_kind=_PM,
        display_name="Balanced accuracy",
        description="mean of sensitivity and specificity; unlike plain accuracy "
                    "it does not reward predicting the majority class",
        parameters={**_CONFUSION_THRESHOLD.to_mapping()}),
    MetricDescriptor(
        name="positive_likelihood_ratio", function=_lr_positive,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_interior_specificity(
            _CONFUSION_THRESHOLD, metric="positive_likelihood_ratio"),
        # REGISTERED BUT NOT IN THE FLAT REPORT (2026-07-29).
        #
        # These are computable now. Adding them to the report surface would move
        # the frozen 480-value oracle, which is a separate and declared change --
        # the same staging that kept commit 3a's schema introduction apart from
        # 3b-2's authority switch. Registered first, surfaced second.
        include_in_evaluation_report=False,
        result_kind=_PM,
        display_name="Positive likelihood ratio",
        description="sensitivity divided by one minus specificity; "
                    "PREVALENCE-INDEPENDENT, so it transfers between cohorts",
        parameters={**_CONFUSION_THRESHOLD.to_mapping(),
                    "prevalence_dependent": False}),
    MetricDescriptor(
        name="negative_likelihood_ratio", function=_lr_negative,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_interior_specificity(
            _CONFUSION_THRESHOLD, metric="negative_likelihood_ratio"),
        # REGISTERED BUT NOT IN THE FLAT REPORT (2026-07-29).
        #
        # These are computable now. Adding them to the report surface would move
        # the frozen 480-value oracle, which is a separate and declared change --
        # the same staging that kept commit 3a's schema introduction apart from
        # 3b-2's authority switch. Registered first, surfaced second.
        include_in_evaluation_report=False,
        result_kind=_PM,
        display_name="Negative likelihood ratio",
        description="one minus sensitivity, divided by specificity; "
                    "prevalence-independent, and LOWER is better",
        parameters={**_CONFUSION_THRESHOLD.to_mapping(),
                    "prevalence_dependent": False}),
    MetricDescriptor(
        name="brier_reliability", function=_brier_reliability,
        required_inputs=frozenset({_L, _P}),
        # CALIBRATION SUPPORT, NOT BOTH CLASSES. Measured 2026-07-30: on an
        # all-negative cohort predicted at 0.2, reliability is 0.040000 and
        # meaningful -- it measures overconfidence. Requiring both classes would
        # repeat the over-restriction corrected in commit 3b-1a, where
        # calibration was withheld on a cohort that could support it.
        applicability=_requires_calibration_support,
        result_kind=_PM,
        include_in_evaluation_report=False,
        display_name="Brier decomposition: reliability",
        description="weighted squared gap between mean predicted probability and "
                    "observed frequency per bin; the CALIBRATION component, lower "
                    "is better",
        parameters=dict(_CALIBRATION_PARAMETERS)),
    MetricDescriptor(
        name="brier_resolution", function=_brier_resolution,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_calibration_support,
        result_kind=_PM,
        include_in_evaluation_report=False,
        display_name="Brier decomposition: resolution",
        description="weighted squared distance of each bin frequency from overall "
                    "prevalence; the DISCRIMINATION component, and the one part "
                    "where HIGHER is better because it enters the identity "
                    "negatively",
        parameters=dict(_CALIBRATION_PARAMETERS)),
    MetricDescriptor(
        name="brier_uncertainty", function=_brier_uncertainty,
        required_inputs=frozenset({_L}),
        applicability=_requires_reference_labels_only,
        # A POPULATION STATISTIC, NOT A PREDICTION METRIC (corrected 2026-07-30).
        #
        # It is prevalence * (1 - prevalence): a property of the COHORT that no
        # model can change, exactly like `prevalence` itself, which carries the
        # same kind. Registering it as a prediction metric made the fail-closed
        # contract test demand that non-finite PROBABILITIES block its
        # certification -- but it never reads the probabilities, so it computed
        # correctly and the test rightly objected.
        #
        # The test was correct and the classification was mine to fix. A metric
        # that ignores the predictions must not be certified or refused on their
        # account.
        result_kind=_PS,
        include_in_evaluation_report=False,
        display_name="Brier decomposition: uncertainty",
        description="prevalence times one minus prevalence; a COHORT PROPERTY no "
                    "model can change, maximal at 0.25 when prevalence is 0.5",
        parameters={}),
    MetricDescriptor(
        name="brier_decomposition_residual", function=_brier_residual,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_calibration_support,
        result_kind=_PM,
        include_in_evaluation_report=False,
        display_name="Brier decomposition residual",
        description="brier minus (reliability - resolution + uncertainty). Exactly "
                    "zero only when bins group identical forecasts; under interval "
                    "binning it is the within-bin variance term, reported so the "
                    "three components can be audited rather than trusted",
        parameters=dict(_CALIBRATION_PARAMETERS)),
    MetricDescriptor(
        name="prevalence", function=_prevalence, required_inputs=frozenset({_L}),
        applicability=_requires_reference_labels_only,
        result_kind=_PS,
        display_name="Prevalence",
        description="proportion of positive reference labels in the evaluation population"),

    # ---------------------------------------------------------------------- #
    # ADDED 2026-07-30. Three metrics the catalogue has declared since commit 1
    # and the registry did not build. None enters REPORT_METRIC_NAMES: they are
    # computed and registered, and whether they join the eight-line printed
    # report is a separate change against a surface test_typed_report_surface.py
    # pins.
    # ---------------------------------------------------------------------- #
    MetricDescriptor(
        name="partial_auroc", function=_partial_auroc_adapter,
        required_inputs=frozenset({_L, _S}),
        applicability=_requires_both_classes, result_kind=_PM,
        include_in_evaluation_report=False,
        # MATCHES THE CATALOGUE EXACTLY. A first draft read "Standardised
        # partial area ...", which is more precise and still wrong to put here:
        # the catalogue is the DECLARATION and the registry implements it, so
        # the registry yields. Caught 2026-07-30 by
        # test_an_implemented_entry_matches_its_descriptor the moment the entry
        # flipped to IMPLEMENTED -- the guard doing exactly its job. The
        # standardisation is not lost: it is in the description below and
        # machine-readable in parameters["standardisation"].
        display_name="Partial area under the receiver operating characteristic "
                     "curve",
        description="area under the curve restricted to a declared "
                    "false-positive band and standardised so a random "
                    "classifier scores 0.5 and a perfect one 1.0",
        parameters=dict(_PARTIAL_AUROC_BAND)),
    MetricDescriptor(
        name="integrated_calibration_index",
        function=_integrated_calibration_index_adapter,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_calibration_support, result_kind=_PM,
        include_in_evaluation_report=False,
        display_name="Integrated calibration index",
        description="mean absolute distance between a predicted probability and "
                    "a smoothed calibration curve; binning-free, so it inherits "
                    "no interval convention",
        parameters={}),
    MetricDescriptor(
        name="adaptive_expected_calibration_error",
        function=_adaptive_ece_adapter,
        required_inputs=frozenset({_L, _P}),
        applicability=_requires_calibration_support, result_kind=_PM,
        include_in_evaluation_report=False,
        display_name="Adaptive expected calibration error",
        description="expected calibration error over equal-MASS bins, which put "
                    "the edges where the predictions are rather than spreading "
                    "them evenly across a range the model does not occupy",
        parameters=dict(_ADAPTIVE_CALIBRATION_PARAMETERS)),
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
        # REG-1, 2026-08-03. Until this date the OK branch below rejected a
        # descriptor that set a registry-owned key and THIS branch accepted it
        # silently, merging `verdict.metadata` LAST so it won.
        #
        # That was the worse branch to leave open: `support()` supplies the
        # population scope and fingerprint, and a refusal's whole job is to say
        # what evidence base it describes -- an INSUFFICIENT_SUPPORT on 3 rows
        # and one on 300,000 point at different problems.
        #
        # THE REFUSAL PATH OWNS LESS THAN THE OK PATH. N_CLASSES_OBSERVED is the
        # descriptor's ARGUMENT here, not a registry fact, so it is subtracted.
        # CERTIFICATION_* are absent because no refusal carries a certification
        # decision. The set is still DERIVED from `support()`, so a new key is
        # protected here automatically.
        #
        # THE MERGE ORDER BELOW IS DELIBERATELY UNCHANGED: reordering would stop
        # the overwrite silently, and the OK branch records why that is inferior.
        _reject_registry_owned_keys(
            d, ctx, verdict,
            frozenset({MetricMetadataKey.METRIC_NAME} | set(ctx.support()))
            - _DESCRIPTOR_OWNED_ON_REFUSAL)
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
    # AN APPLICABLE VERDICT MAY CARRY DIAGNOSTICS TOO (2026-07-28).
    #
    # Until this date only a REFUSAL's metadata was merged, so an
    # `Applicability(applicable=True, metadata={...})` was accepted by the type
    # and then silently discarded -- a structure the class permits and nothing
    # consumed. Found when calibration began recording `reference_class_support`
    # on a single-class cohort and the diagnostic never reached the result.
    #
    # PROTECTED KEYS ARE REJECTED, NOT SHADOWED (2026-07-28).
    #
    # Merge ORDER would also prevent a descriptor from overwriting registry-owned
    # keys, but silently: the descriptor's value would simply vanish, and a
    # descriptor author who believed they were setting the population scope would
    # get no signal at all. An explicit collision check turns "a descriptor cannot
    # overwrite registry identity" from an ordering convention into an enforced
    # invariant with a diagnosable failure.
    #
    # The protected set is DERIVED from what the registry itself supplies, not
    # hand-listed, so a future key added to `support()` is protected the moment it
    # exists rather than the moment somebody remembers to add it here.
    #
    # MOVED to `_reject_registry_owned_keys` 2026-08-03 (REG-1) so the REFUSAL
    # branch above runs the SAME check. The set below is IDENTICAL to what was
    # inlined here, and nothing about this path's behaviour changes: an
    # applicable verdict may not set any registry-owned key, N_CLASSES_OBSERVED
    # included, because by this point the registry has computed the cohort and a
    # descriptor claiming otherwise would be contradicting an established fact.
    _reject_registry_owned_keys(
        d, ctx, verdict,
        frozenset({MetricMetadataKey.METRIC_NAME,
                   MetricMetadataKey.CERTIFICATION_ELIGIBLE,
                   MetricMetadataKey.CERTIFICATION_BLOCKED_BY}
                  | set(ctx.support())))
    meta = {**dict(verdict.metadata), **meta}
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
    # AN UNATTRIBUTED POPULATION CANNOT SUPPORT A CERTIFIED CLAIM (2026-07-28).
    #
    # A certified claim asserts something about a NAMED set of rows. An
    # unattributed population has no source identity, so its membership
    # fingerprint is absent and comparison against any other population returns
    # UNKNOWN rather than SAME or DIFFERENT. A claim that cannot be tied to an
    # identifiable cohort is not certifiable, however sound its arithmetic.
    #
    # This does NOT make the value wrong or the evaluation useless: unattributed
    # evaluation is a legitimate exploratory mode, and every metric still reports
    # its status, value and support. It makes the ADMISSIBILITY explicit rather
    # than leaving a reader to infer it from an absent fingerprint.
    if not ctx.population.is_attributed:
        return False, "unattributed_population"
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
