"""Threshold semantics: the vocabulary beneath the registry and the metrics.

THR-1, 2026-08-04.

This module is the BOTTOM LAYER of the evaluation stack's threshold handling. It
imports neither `registry.py`, nor `metrics.py`, nor scikit-learn, and it must
stay that way -- that constraint is the entire reason it exists.

    capabilities.py / population.py
                 |
            thresholds.py
                 |
        +--------+--------+
    registry.py       metrics.py

OP-1's exact threshold sweep will be built here, so that it can describe each
swept candidate with a `ThresholdParameters` without importing the registry --
which would reverse the layering -- and without sitting behind the scikit-learn
import boundary that `evaluation/__init__.py` documents.

THE THREE CLASSES BELOW MOVED VERBATIM FROM `registry.py` (lines 193-265) and are
re-exported from there, preserving OBJECT IDENTITY. Existing imports continue to
work and continue to return the same objects.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

# THE POPULATION IS IMPORTED, NOT DUCK-TYPED. `population.py` sits
# ABOVE this module in the layering, so the dependency runs downward,
# and it imports no scikit-learn -- verified before this edit, because
# a TRANSITIVE import would defeat the structural guarantee THR-1a
# established without any test noticing.
from .capabilities import (
    MetricMetadataKey,
    MetricResult,
    MetricStatus,
    _is_finite,
)
from .population import EvaluationPopulation

logger = logging.getLogger(__name__)

__all__ = ["ConfusionCounts", "ExactThresholdSweep",
           "ThresholdOperator", "ThresholdParameters",
           "ThresholdSource", "ThresholdSweepCandidate",
           "OperatingPointCertificationBlocker",
           "OperatingPointMetrics", "OperatingPointOutcome",
           "metrics_from_counts", "sweep_thresholds"]


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
    # THR-1b (2026-08-04). A candidate enumerated by an exact threshold sweep.
    #
    # SWEEP, NOT SELECTED. A candidate exists BEFORE selection: every point the
    # sweep enumerates carries this source, and at most one of them is ever
    # chosen. "Selected" would make the vocabulary temporally false for every
    # candidate that was examined and rejected -- which is all but one.
    #
    # And it keeps three facts separate, which is why this is a type rather than
    # a comment: SOURCE is where the candidate came from, POLICY is why it was
    # chosen among candidates, and CERTIFICATION BLOCKERS are why its performance
    # is not independently validated. A field conflating them would leave an
    # artifact unable to distinguish "swept" from "chosen" from "unvalidated".
    EVALUATION_SWEEP = "evaluation_sweep"


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


# --------------------------------------------------------------------------- #
# THE EXACT THRESHOLD SWEEP (OP-1 step 1, 2026-08-04)
#
# The legacy operating-point selectors walk a thousand-point uniform grid, or the
# unique probabilities, recomputing four boolean reductions over the FULL ARRAY
# at every candidate. Measured in the OP-1 defect register: O(k*n), which is
# 2.25e12 element operations on a 1.5-million-variant cohort, against a grid
# whose points are mostly NOT ACHIEVABLE -- no row carries that score.
#
# One sort with cumulative sums gives EXACT counts at every achievable threshold
# in O(n log n), closing four register defects at once rather than four patches:
# D1 (inexact grid), D9 (quadratic cost), D10 (loop-invariant work) and D11 (two
# sweep strategies).
#
# NOT A THIRD IMPLEMENTATION. The same construction exists twice in `metrics.py`
# -- `auprc` and `_roc_points` -- by different index arithmetic. Their raw count
# sequences were compared across eleven cohorts on 2026-08-04 and found
# IDENTICAL (SWEEP-1). This is written to be the one those two can LATER be
# rebuilt on, which is why it lives here and imports no scikit-learn.
#
# NO SELECTOR. This chooses no operating point and implements neither Objective A
# nor B. It returns COUNTS, which are integers and always defined, so it cannot
# fabricate a value for an undefined quantity -- rates, refusals and
# certification belong to the typed outcome in step 2.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfusionCounts:
    """The four confusion counts at one threshold.

    COUNTS ONLY. No rate is stored, because a rate can be undefined while a count
    never is: `TP + FP = 0` makes the positive predictive value undefined, and
    reporting it as 0.0 is the D2-D5 defect. REG-2 established on 2026-08-04 that
    such a state is UNDEFINED rather than INSUFFICIENT_SUPPORT, and step 2's
    typed outcome refuses accordingly.
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    def __post_init__(self) -> None:
        for name in ("true_positive", "false_positive",
                     "false_negative", "true_negative"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                    value, (int, np.integer)) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}. A "
                    "confusion count is a cardinality; a float here means "
                    "something computed a rate and stored it as a count.")
            object.__setattr__(self, name, int(value))

    @property
    def n_flagged(self) -> int:
        """Predicted positives. NOT `n_neg` -- see OP-0 (2026-08-04), where that
        name held this quantity in one selector and the reference-negative count
        in its sibling."""
        return self.true_positive + self.false_positive

    @property
    def n_cleared(self) -> int:
        """Predicted negatives."""
        return self.true_negative + self.false_negative

    @property
    def n_actual_positive(self) -> int:
        """Threshold-INVARIANT: identical at every candidate."""
        return self.true_positive + self.false_negative

    @property
    def n_actual_negative(self) -> int:
        """Threshold-INVARIANT: identical at every candidate."""
        return self.true_negative + self.false_positive

    @property
    def n(self) -> int:
        return (self.true_positive + self.false_positive
                + self.false_negative + self.true_negative)


@dataclass(frozen=True)
class ThresholdSweepCandidate:
    """One achievable operating point: its threshold declaration and its counts.

    BUILT ON DEMAND, NEVER STORED. `ExactThresholdSweep` holds arrays; this is
    the view it constructs when asked for a particular index. A cohort of 1.5
    million distinct scores has 1.5 million achievable thresholds, and
    materialising that many frozen dataclasses would cost more than the sweep
    that computed them.
    """

    index: int
    parameters: ThresholdParameters
    counts: ConfusionCounts

    @property
    def threshold(self) -> float:
        return self.parameters.threshold

    @property
    def operator(self) -> ThresholdOperator:
        return self.parameters.operator


class ExactThresholdSweep:
    """Every achievable operating point, as owned immutable arrays.

    THE ARRAYS ARE COPIED AND MARKED READ-ONLY. A caller who mutates the input
    afterwards cannot retroactively change what this sweep reports -- a sweep is
    evidence, and evidence that can change after the fact is not evidence.

    Candidates run from MOST CONSERVATIVE to MOST PERMISSIVE, which is the order
    both legacy selectors walk; index 0 is the empty candidate.

    SELECTION SHOULD WORK ON THE ARRAYS, not by iterating this object. Iterating
    builds one `ThresholdSweepCandidate` per step, which reintroduces the
    per-candidate object cost the array backing exists to avoid. The arrays are
    public for exactly that reason.
    """

    __slots__ = ("_thresholds", "_strictly_greater", "_true_positive",
                 "_false_positive", "_n_actual_positive", "_n_actual_negative",
                 "_population")

    def __init__(self, *, thresholds, strictly_greater, true_positive,
                 false_positive, n_actual_positive, n_actual_negative,
                 population: EvaluationPopulation | None):
        def owned(array, dtype):
            copy = np.array(array, dtype=dtype, copy=True)
            copy.setflags(write=False)
            return copy

        self._thresholds = owned(thresholds, np.float64)
        self._strictly_greater = owned(strictly_greater, bool)
        self._true_positive = owned(true_positive, np.int64)
        self._false_positive = owned(false_positive, np.int64)
        self._n_actual_positive = int(n_actual_positive)
        self._n_actual_negative = int(n_actual_negative)
        self._population = population

        sizes = {self._thresholds.size, self._strictly_greater.size,
                 self._true_positive.size, self._false_positive.size}
        if len(sizes) != 1:
            raise ValueError(
                f"the sweep arrays have differing lengths {sorted(sizes)}; a "
                "candidate would silently borrow another candidate's counts")

    @property
    def thresholds(self):
        """Read-only. Selection works on this rather than on candidate objects."""
        return self._thresholds

    @property
    def strictly_greater(self):
        return self._strictly_greater

    @property
    def true_positive(self):
        return self._true_positive

    @property
    def false_positive(self):
        return self._false_positive

    @property
    def false_negative(self):
        return self._n_actual_positive - self._true_positive

    @property
    def true_negative(self):
        return self._n_actual_negative - self._false_positive

    @property
    def n_flagged(self):
        return self._true_positive + self._false_positive

    @property
    def n_actual_positive(self) -> int:
        return self._n_actual_positive

    @property
    def n_actual_negative(self) -> int:
        return self._n_actual_negative

    @property
    def population(self) -> EvaluationPopulation | None:
        """The evaluation population these counts describe.

        Carried from the first type rather than attached later, because POP-1b
        established that a count without its population says nothing about WHICH
        rows produced it.
        """
        return self._population

    def __len__(self) -> int:
        return int(self._thresholds.size)

    def __getitem__(self, index: int) -> ThresholdSweepCandidate:
        """Build one candidate view. Nothing is cached; nothing is stored."""
        size = int(self._thresholds.size)
        if isinstance(index, slice):
            raise TypeError(
                "slicing would materialise one object per candidate, which is "
                "the cost this array backing exists to avoid; index the arrays "
                "directly instead")
        position = int(index)
        if position < 0:
            position += size
        if not 0 <= position < size:
            raise IndexError(f"candidate {index} out of range for {size}")
        true_positive = int(self._true_positive[position])
        false_positive = int(self._false_positive[position])
        operator = (ThresholdOperator.GREATER
                    if bool(self._strictly_greater[position])
                    else ThresholdOperator.GREATER_OR_EQUAL)
        return ThresholdSweepCandidate(
            index=position,
            parameters=ThresholdParameters(
                threshold=float(self._thresholds[position]),
                operator=operator,
                source=ThresholdSource.EVALUATION_SWEEP),
            counts=ConfusionCounts(
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=self._n_actual_positive - true_positive,
                true_negative=self._n_actual_negative - false_positive))

    def array_bytes(self) -> int:
        """Total bytes held. Linear in the candidate count, by construction."""
        return int(self._thresholds.nbytes + self._strictly_greater.nbytes
                   + self._true_positive.nbytes + self._false_positive.nbytes)


def sweep_thresholds(y_true, y_prob, *,
                     population: EvaluationPopulation | None
                     ) -> ExactThresholdSweep:
    """Exact confusion counts at every achievable threshold, in O(n log n).

    THE CANONICAL DOMAIN is

        {(max p, GREATER)} union {(s, GREATER_OR_EQUAL) : s in unique(p)}

    and NOT both operators at every score: for adjacent distinct values, `p > s`
    and `p >= s'` induce the SAME partition, so enumerating both would create
    duplicate candidates and make indices representation-dependent.

    The GREATER entry is the EMPTY CANDIDATE. `ThresholdParameters` constrains a
    threshold to [0, 1], so "flag nothing" cannot be expressed as a value above
    the maximum when the maximum is 1.0; GREATER at the maximum expresses it. The
    grid sweep silently lacked this operating point altogether.

    RAISES rather than filtering. `p >= t` evaluates FALSE for a NaN, so a
    non-finite score would become a predicted negative with no exception and no
    warning -- `evaluator.py` records that doing so moved a measured operating
    point from sensitivity 0.90 to 0.50.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()

    if y.size != p.size:
        raise ValueError(
            f"y_true has {y.size} rows and y_prob has {p.size}; a sweep over "
            "mismatched arrays would align the wrong label with the wrong score")
    if y.size == 0:
        raise ValueError(
            "an empty cohort has no achievable thresholds. This raises rather "
            "than returning an empty sweep, because an empty sweep would read "
            "as 'no operating point was suitable' when the truth is that none "
            "was examined")
    if population is not None and int(population.n) != y.size:
        raise ValueError(
            f"the population declares {int(population.n)} rows and the arrays "
            f"carry {y.size}; counts would describe a cohort nobody declared")
    if not np.all(np.isfinite(p)):
        raise ValueError(
            f"{int((~np.isfinite(p)).sum())} of {p.size} scores are non-finite; "
            "`p >= t` evaluates FALSE for a NaN, so these would silently become "
            "predicted negatives")
    if not np.all(np.isfinite(y)):
        raise ValueError(
            f"{int((~np.isfinite(y)).sum())} of {y.size} labels are non-finite; "
            "the label-eligible population is selected upstream (POP-1a)")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError(
            "scores lie outside [0, 1]; this sweeps PROBABILITIES, and a "
            "decision threshold is constrained to that range")
    observed = np.unique(y)
    if not np.all(np.isin(observed, (0.0, 1.0))):
        raise ValueError(f"labels must be 0 or 1; observed {observed.tolist()}")

    # ONE SORT, descending, stable so equal scores keep input order -- which
    # makes the counts reproducible rather than dependent on an unspecified
    # permutation.
    order = np.argsort(-p, kind="mergesort")
    y_ordered = y[order]
    p_ordered = p[order]

    # RUN ENDS. Tied scores form ONE candidate: a threshold cannot separate rows
    # carrying the same score, so treating them as distinct would invent
    # operating points the data cannot express.
    distinct = np.ones(p_ordered.size, dtype=bool)
    distinct[1:] = p_ordered[1:] != p_ordered[:-1]
    run_ends = np.append(np.flatnonzero(distinct)[1:], p_ordered.size) - 1

    cumulative_positive = np.cumsum(y_ordered)
    true_positive = cumulative_positive[run_ends]
    false_positive = (run_ends + 1) - true_positive

    n_actual_positive = int(cumulative_positive[-1])
    n_actual_negative = int(y.size - n_actual_positive)

    thresholds = np.concatenate(([p_ordered[0]], p_ordered[run_ends]))
    strictly_greater = np.zeros(thresholds.size, dtype=bool)
    strictly_greater[0] = True

    logger.debug(
        "swept %d rows into %d achievable candidates (%d actual positives)",
        y.size, thresholds.size, n_actual_positive)

    return ExactThresholdSweep(
        thresholds=thresholds,
        strictly_greater=strictly_greater,
        true_positive=np.concatenate(([0], true_positive)),
        false_positive=np.concatenate(([0], false_positive)),
        n_actual_positive=n_actual_positive,
        n_actual_negative=n_actual_negative,
        population=population)


# --------------------------------------------------------------------------- #
# THE TYPED OPERATING-POINT OUTCOME (OP-1 step 2, 2026-08-05)
#
# Closes D2-D5 and D6 BY CONSTRUCTION rather than by convention.
#
# D2-D5: the legacy selectors computed `tn / n_neg if n_neg > 0 else 0.0`, and
# F1 from a positive predictive value that might itself be a fabricated 0.0 --
# a second fabrication derived from the first, with no record that either
# occurred. Here every quantity is a `MetricResult`, whose constructor REFUSES a
# non-OK status carrying a finite value. There is no way to store 0.0 where a
# refusal belongs.
#
# D6: the legacy selectors stored `round(value, 4)` and computed F1 from
# UNROUNDED inputs, so recomputing it from the STORED positive predictive value
# and sensitivity would not reproduce the STORED F1. Here nothing is rounded at
# storage; `round_for_display` returns a new mapping and leaves the record alone.
#
# STILL NO SELECTOR. This turns counts into a typed record. Choosing among
# candidates is step 4.
# --------------------------------------------------------------------------- #


class OperatingPointCertificationBlocker(str, Enum):
    """Why an operating point's performance is not independently validated.

    STABLE CODES, PROSE ELSEWHERE. Artifacts persist these members; reports
    render `describe()` from them. Serialising mutable prose as the primary
    scientific identifier is how a rewording orphans every historical record --
    the same reasoning that made POP-1b's population scope an enum member rather
    than a sentence.
    """

    # CERT-1 (2026-08-05). The code matches the registry's machine-readable
    # reason CHARACTER FOR CHARACTER, so scalar certification and composite
    # certification share ONE vocabulary rather than two that agree today.
    UNATTRIBUTED_POPULATION = "unattributed_population"

    SAME_POPULATION_SELECTION_AND_EVALUATION = (
        "threshold_selected_and_evaluated_on_same_population")
    POST_SELECTION_VALIDATION_NOT_IMPLEMENTED = (
        "post_selection_validation_not_implemented")

    def describe(self) -> str:
        """The prose, centralised so a rewording touches one place."""
        return _BLOCKER_PROSE[self]


_BLOCKER_PROSE = {
    OperatingPointCertificationBlocker.UNATTRIBUTED_POPULATION: (
        "the evaluation population has no stable external identity, so the "
        "reported operating point cannot be reproduced or compared across "
        "artifacts as a claim about the same rows"),
    OperatingPointCertificationBlocker
    .SAME_POPULATION_SELECTION_AND_EVALUATION: (
        "the threshold was chosen on the same rows its performance is reported "
        "on, so the reported performance is optimistic by an unmeasured amount"),
    OperatingPointCertificationBlocker
    .POST_SELECTION_VALIDATION_NOT_IMPLEMENTED: (
        "no held-out validation of the selected threshold exists, so its "
        "performance has not been confirmed on rows that did not choose it"),
}


@dataclass(frozen=True)
class OperatingPointMetrics:
    """Every quantity derivable from one candidate's counts, each able to refuse.

    NOTHING IS ROUNDED HERE. `round_for_display` returns a new mapping; the
    stored values are the computed ones, so recomputing F1 from the stored
    positive predictive value and sensitivity reproduces the stored F1. That is
    D6 closed.

    PREVALENCE IS ABSENT, DELIBERATELY. It is a POPULATION statistic -- it does
    not depend on the threshold, on predicted-positive membership, or on the
    sweep -- and its canonical value comes from the registry. Computing a second
    one here would invent two prevalences that agree until a population bug makes
    them diverge.
    """

    counts: ConfusionCounts
    parameters: ThresholdParameters

    sensitivity: MetricResult
    specificity: MetricResult
    positive_predictive_value: MetricResult
    negative_predictive_value: MetricResult
    f1: MetricResult
    matthews_correlation_coefficient: MetricResult
    flagged_fraction: MetricResult

    def as_mapping(self) -> dict:
        """Every quantity by name. Ordered as a clinician reads them."""
        return {
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "positive_predictive_value": self.positive_predictive_value,
            "negative_predictive_value": self.negative_predictive_value,
            "f1": self.f1,
            "matthews_correlation_coefficient":
                self.matthews_correlation_coefficient,
            "flagged_fraction": self.flagged_fraction,
        }

    def round_for_display(self, digits: int = 4) -> dict:
        """Rounded values FOR DISPLAY. The record itself is untouched.

        The legacy selectors rounded AT CONSTRUCTION, which is why a stored F1
        could disagree with one recomputed from its stored inputs. Rounding is a
        presentation concern and it stays one.
        """
        return {name: (round(result.value, digits) if result.is_ok else None)
                for name, result in self.as_mapping().items()}

    def to_dict(self) -> dict:
        return {"counts": {"true_positive": self.counts.true_positive,
                           "false_positive": self.counts.false_positive,
                           "false_negative": self.counts.false_negative,
                           "true_negative": self.counts.true_negative},
                "threshold": self.parameters.to_mapping(),
                "metrics": {name: result.to_dict()
                            for name, result in self.as_mapping().items()}}


@dataclass(frozen=True)
class OperatingPointOutcome:
    """An operating point, or a stated reason there is none.

    `Optional[OperatingPoint]` was the defect this replaces: `None` carried every
    kind of unavailability at once, and `evaluate_registered`'s docstring names
    why that matters -- "an absent key and a refused metric are different facts
    and a caller cannot tell them apart".

    Here a refusal carries a STATUS and a REASON, and `metrics` is None only when
    the status says so. The invariant is enforced in __post_init__ so the raw
    constructor cannot bypass it.

    POPULATION IDENTITY IS CARRIED ONCE, HERE, in the `MetricMetadataKey`
    vocabulary -- not pushed into every per-quantity result. `MetricResult` stays
    generic by a measured decision recorded in its own source: 35 of its 53
    construction sites are embedding-space probes for which population scope has
    no epidemiological meaning.
    """

    status: MetricStatus
    reason: Optional[str]
    metrics: Optional[OperatingPointMetrics]
    population: Optional[EvaluationPopulation]
    certification_blockers: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, MetricStatus):
            raise TypeError(
                f"status must be a MetricStatus, got "
                f"{type(self.status).__name__}")
        if self.status is MetricStatus.OK:
            if self.reason:
                raise ValueError(
                    "an OK outcome must not carry a reason; a reason explains "
                    "why an operating point is absent")
            if self.metrics is None:
                raise ValueError(
                    "an OK outcome must carry metrics; an OK status with no "
                    "operating point is the `Optional[OperatingPoint]` "
                    "ambiguity this type replaces")
            # CERT-1 (2026-08-05). Step 2 shipped without this and a test
            # positively asserted that an OK outcome with NO POPULATION was
            # certifiable -- contradicting the registry's rule and step 2's own
            # rationale, which placed population identity here because "n=980
            # beside n=980 says nothing about WHICH 980".
            if self.population is None:
                raise ValueError(
                    "an OK operating-point outcome must carry an "
                    "EvaluationPopulation; without one the reported counts and "
                    "rates have no defined row membership, and a claim that "
                    "cannot be tied to an identifiable cohort is not "
                    "certifiable however sound its arithmetic")
        else:
            if not self.reason:
                raise ValueError(
                    f"status {self.status.value!r} requires a nonempty reason. A "
                    "refusal without an explanation is the silent None this type "
                    "exists to prevent.")
            if self.metrics is not None:
                raise ValueError(
                    f"status {self.status.value!r} must not carry metrics; a "
                    "refusal holding an operating point invites it being used")
        for blocker in self.certification_blockers:
            if not isinstance(blocker, OperatingPointCertificationBlocker):
                raise TypeError(
                    "certification blockers must be "
                    f"OperatingPointCertificationBlocker members, got "
                    f"{type(blocker).__name__}")

    @property
    def is_ok(self) -> bool:
        return self.status is MetricStatus.OK

    @property
    def effective_certification_blockers(self) -> tuple:
        """What the caller declared, PLUS what this outcome's own state implies.

        DECLARED AND DERIVED STAY SEPARATE. Inserting the derived blocker into
        `certification_blockers` inside `__post_init__` is possible in a frozen
        dataclass through `object.__setattr__`, and it would make the
        constructor arguments differ from the stored object -- a caller passing
        `()` and reading back a one-element tuple. The record says what was
        given; this says what follows.
        """
        blockers = list(self.certification_blockers)
        unattributed = (
            OperatingPointCertificationBlocker.UNATTRIBUTED_POPULATION)
        if (self.is_ok and self.population is not None
                and not self.population.is_attributed
                and unattributed not in blockers):
            blockers.append(unattributed)
        return tuple(blockers)

    @property
    def certification_eligible(self) -> bool:
        """Eligible only when it succeeded, its population is ATTRIBUTED, and
        nothing blocks certification.

        CERT-1 (2026-08-05). This previously read
        `self.is_ok and not self.certification_blockers`, which never consulted
        the population at all. The registry's `_certification_eligibility` has
        refused an unattributed population since 2026-07-28, on the reasoning
        that "a certified claim asserts something about a NAMED set of rows" --
        an unattributed population's fingerprint is absent, so comparison with
        any other returns UNKNOWN rather than SAME or DIFFERENT.

        The population check is expressed through
        `effective_certification_blockers` rather than as an extra conjunct, so
        the SERIALISED ARTIFACT CAN EXPLAIN the ineligibility. A boolean alone
        would leave a reader with `certification_eligible: false` beside an
        empty blocker list.
        """
        return (self.is_ok
                and self.population is not None
                and self.population.is_attributed
                and not self.effective_certification_blockers)

    def population_metadata(self) -> dict:
        """Scope, fingerprint and observation count, in the shared vocabulary.

        Carried ONCE on the outcome rather than in each result's metadata. POP-1b
        established why it must be carried at all: "n=980 beside n=980 says
        nothing about WHICH 980".
        """
        if self.population is None:
            return {}
        return {
            MetricMetadataKey.POPULATION_SCOPE: self.population.scope,
            MetricMetadataKey.POPULATION_FINGERPRINT:
                self.population.membership_fingerprint,
            MetricMetadataKey.N_OBSERVATIONS: int(self.population.n),
        }

    @classmethod
    def refused(cls, status: MetricStatus, reason: str, *,
                population=None) -> "OperatingPointOutcome":
        """The only way to build a refusal. Mirrors `MetricResult.not_ok`."""
        if status is MetricStatus.OK:
            raise ValueError("refused() cannot construct an OK outcome")
        return cls(status=status, reason=reason, metrics=None,
                   population=population)

    def to_dict(self) -> dict:
        """Serialisable. STATUS-AWARE, never inferring absence from a value.

        `MetricResult.to_dict` records the rule and the incident behind it
        (CI-p, 2026-07-29): absence is authorised by the STATUS, and an OK result
        whose value is somehow non-finite is a DEFECT that must reach the strict
        writer rather than be disguised as a legitimate absence.
        """
        return {
            "status": self.status.value,
            "reason": self.reason,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "certification_eligible": self.certification_eligible,
            # THE EFFECTIVE set, so an artifact explains its own ineligibility
            # rather than stating it. CERT-1 (2026-08-05).
            "certification_blockers": [
                b.value for b in self.effective_certification_blockers],
            "population": {str(key.value): value
                           for key, value in self.population_metadata().items()},
        }


def _ratio(numerator: int, denominator: int, *, reason: str,
           status: MetricStatus) -> MetricResult:
    """A quotient, or a refusal naming the state that made it undefined.

    THE STATUS IS A PARAMETER because the two cases are different facts, settled
    by REG-2 (afa7a90) against the registry's measured convention: a vanishing
    DENOMINATOR is UNDEFINED, while an absent reference CLASS is
    INSUFFICIENT_SUPPORT.
    """
    if denominator == 0:
        return MetricResult.not_ok(status, reason)
    return MetricResult.ok(numerator / denominator)


def metrics_from_counts(counts: ConfusionCounts,
                        parameters: ThresholdParameters
                        ) -> OperatingPointMetrics:
    """Turn one candidate's counts into a typed record that can refuse.

    EVERY REASON STRING IS ONE THE REGISTRY ALREADY EMITS, so step 3's Oracle C1
    can compare this count path against `registry.compute` on the same cohort. A
    private vocabulary here would make the oracle compare two dialects and report
    every difference as a defect.
    """
    true_positive = counts.true_positive
    false_positive = counts.false_positive
    false_negative = counts.false_negative
    true_negative = counts.true_negative

    sensitivity = _ratio(
        true_positive, counts.n_actual_positive,
        reason="positive_class_support_required",
        status=MetricStatus.INSUFFICIENT_SUPPORT)
    specificity = _ratio(
        true_negative, counts.n_actual_negative,
        reason="negative_class_support_required",
        status=MetricStatus.INSUFFICIENT_SUPPORT)
    positive_predictive_value = _ratio(
        true_positive, counts.n_flagged,
        reason="empty_predicted_positive_set",
        status=MetricStatus.UNDEFINED)
    negative_predictive_value = _ratio(
        true_negative, counts.n_cleared,
        reason="empty_predicted_negative_set",
        status=MetricStatus.UNDEFINED)

    # F1 FROM COUNTS, NOT FROM RATES. The legacy form computed it from a
    # positive predictive value that might itself be a fabricated 0.0 -- D5, a
    # second fabrication derived from the first. 2TP/(2TP+FP+FN) is the same
    # quantity and cannot inherit anything.
    f1 = _ratio(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
        reason="zero_f1_denominator", status=MetricStatus.UNDEFINED)

    # The Matthews correlation coefficient's denominator is the product of four
    # margins; any vanishing margin annihilates it. scikit-learn returns 0.0 and
    # warns while doing so -- its own warning being the evidence that the 0.0 is
    # a fabrication.
    margins = (counts.n_flagged, counts.n_actual_positive,
               counts.n_actual_negative, counts.n_cleared)
    if any(margin == 0 for margin in margins):
        matthews = MetricResult.not_ok(
            MetricStatus.UNDEFINED, "zero_confusion_margin")
    else:
        numerator = (true_positive * true_negative
                     - false_positive * false_negative)
        denominator = np.sqrt(float(margins[0]) * float(margins[1])
                              * float(margins[2]) * float(margins[3]))
        matthews = MetricResult.ok(numerator / denominator)

    flagged_fraction = _ratio(
        counts.n_flagged, counts.n,
        reason="empty_cohort", status=MetricStatus.INSUFFICIENT_DATA)

    return OperatingPointMetrics(
        counts=counts,
        parameters=parameters,
        sensitivity=sensitivity,
        specificity=specificity,
        positive_predictive_value=positive_predictive_value,
        negative_predictive_value=negative_predictive_value,
        f1=f1,
        matthews_correlation_coefficient=matthews,
        flagged_fraction=flagged_fraction)
