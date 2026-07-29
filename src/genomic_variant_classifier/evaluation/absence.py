"""Explicit absence on the artifact surface.

WHY THIS EXISTS
===============
`dump_strict_json` refuses a non-finite number, correctly: *"a non-finite number
in an evidence artifact is either a computation that failed silently or an absent
estimate wearing a number's clothes"*. But the flat report surface had no way to
say a value was absent, so the whole file was rejected rather than the one field
being recorded as missing.

Measured 2026-07-29 on the tree at `2a1e7f6`, three of five cohorts produced
reports that could not be written at all:

    healthy               PERSISTS
    all-negative          REFUSED   auroc, tpr_curve[0], tpr_curve[1]
    all-positive          REFUSED   auroc, fpr_curve[0], fpr_curve[1]
    constant classifier   PERSISTS
    non-finite input      REFUSED   auroc, auprc, mcc, f1, brier_score

A scientifically valid evaluation over a degenerate cohort could not produce an
artifact. That is the defect this vocabulary removes.

BARE NULL IS NOT ENOUGH
-----------------------
A `null` alone says a value is missing and nothing about why. A reader cannot
distinguish "the area under the receiver operating characteristic curve is
mathematically undefined because only one class is present" from "the input was
refused" from "a bug produced nothing". Those demand different responses, and
collapsing them recreates the ambiguity the typed metric surface spent fourteen
commits eliminating.

So absence is EXPLICIT: the value serialises as `null`, and a parallel map carries
the cause and reason.

TWO STRUCTURES, NOT ONE
-----------------------
`field_absence` and `curve_absence` are separate because a scalar and an array
are absent in different ways, and one map with mixed semantics would invite a
reader to treat them alike.

The choice of CURVE-LEVEL rather than element-level absence is measured, not
aesthetic. On 2026-07-29 no curve in any degenerate cohort mixed valid and
non-finite entries -- each array was entirely clean, entirely non-finite, or
empty. Element-level absence would be a representation for a state that cannot
occur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence

__all__ = [
    "AbsenceCause",
    "CurveAbsence",
    "FieldAbsence",
    "absence_for_curve",
    "absence_for_value",
]


class AbsenceCause(str, Enum):
    """WHY a value is absent. A CLOSED vocabulary.

    These are not interchangeable. `UNDEFINED_ON_COHORT` says the quantity has no
    value for this data -- a property of the cohort, and a legitimate scientific
    finding. `WITHHELD_BY_INPUT_GATE` says the inputs were refused before
    computation -- a property of the model output, and a defect to fix. Reporting
    both as "missing" would tell a reader to investigate the wrong thing.
    """

    UNDEFINED_ON_COHORT = "undefined_on_cohort"
    WITHHELD_BY_INPUT_GATE = "withheld_by_input_gate"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class FieldAbsence:
    """Why one scalar field is absent."""

    cause: AbsenceCause
    reason: Optional[str] = None
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.cause, AbsenceCause):
            raise TypeError(
                "cause must be a member of the closed AbsenceCause vocabulary; "
                "a free string would let two artifacts describe the same "
                "situation differently")

    def to_dict(self) -> dict:
        return {"cause": self.cause.value, "reason": self.reason,
                "detail": self.detail}


@dataclass(frozen=True)
class CurveAbsence:
    """Why one curve is absent, and how long it would have been.

    `n_expected` records the length the curve WOULD have had, which is what
    separates an absent curve from a legitimately empty one. An empty curve over
    an empty cohort is not the same as a withheld curve over two hundred rows,
    and an artifact that cannot tell them apart is the silent absence this
    vocabulary exists to remove.
    """

    cause: AbsenceCause
    reason: Optional[str] = None
    n_expected: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.cause, AbsenceCause):
            raise TypeError(
                "cause must be a member of the closed AbsenceCause vocabulary")

    def to_dict(self) -> dict:
        return {"cause": self.cause.value, "reason": self.reason,
                "n_expected": self.n_expected}


def absence_for_value(value, *, cause: AbsenceCause,
                      reason: Optional[str] = None) -> Optional[FieldAbsence]:
    """A `FieldAbsence` when the value is non-finite, otherwise `None`.

    The caller decides the CAUSE, because only the caller knows why: the same
    NaN means "undefined on this cohort" when the labels are single-class and
    "withheld by an input gate" when the probabilities were unusable.
    """
    if value is None:
        return FieldAbsence(cause=cause, reason=reason)
    try:
        finite = math.isfinite(float(value))
    except (TypeError, ValueError):
        return None
    return None if finite else FieldAbsence(cause=cause, reason=reason)


def absence_for_curve(values: Sequence, *, cause: AbsenceCause,
                      reason: Optional[str] = None,
                      n_expected: Optional[int] = None) -> Optional[CurveAbsence]:
    """A `CurveAbsence` when the curve cannot be persisted, otherwise `None`.

    A curve is absent when it is empty OR when any element is non-finite.
    Measured 2026-07-29: no curve mixes valid and non-finite entries, so
    "any element" and "every element" coincide in practice -- but "any" is the
    safe rule, because a partial curve is not a curve and must never be written
    as though it were complete.
    """
    if values is None:
        return CurveAbsence(cause=cause, reason=reason, n_expected=n_expected)
    sequence = list(values)
    if not sequence:
        return CurveAbsence(cause=cause, reason=reason, n_expected=n_expected)
    for item in sequence:
        try:
            if not math.isfinite(float(item)):
                return CurveAbsence(cause=cause, reason=reason,
                                    n_expected=len(sequence))
        except (TypeError, ValueError):
            return CurveAbsence(cause=cause, reason=reason,
                                n_expected=len(sequence))
    return None
