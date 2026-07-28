"""Input validation for the report path, before any library call.

WHY VALIDATE RATHER THAN CATCH
==============================
Five scikit-learn calls sit in the report path on the same `(y, p)` pair, and
their behaviour under bad input is inconsistent. Measured 2026-07-28:

    defect                      roc_  pr_    calibration_  roc_auc  average_
                                curve curve  curve         _score   precision
    non-finite probabilities    raise raise  RETURNS       raise    raise
    outside the unit interval   ok    ok     RAISE         ok       ok
    single-class labels         warn  warn   returns       warn     warn

No consistent rule exists to translate. Wrapping the calls in a handler would
mean the library decides which defects become which statuses, and it does not
agree with itself.

THE SILENT CASE IS THE DANGEROUS ONE. With 40 of 200 probabilities non-finite,
`calibration_curve` raises nothing, WARNS NOTHING, and returns a degenerate
one-point curve carrying NaN -- down from ten points. The failure then surfaces
at persistence, where strict JSON refuses the artifact and names the calibration
curve rather than the corrupt model that caused it. An exception is loud and
stops the run; a poisoned number is neither.

THREE CHANNELS, NOT ONE GATE
-----------------------------
    reference labels   finite, binary, aligned
    ranking scores     finite, aligned, NO range restriction
    probabilities      finite, aligned, within the unit interval

Kept separate because a failed probability check must not suppress a valid
score-based area under the receiver operating characteristic curve. One gate
containing every check cannot express that.

SCORES AND PROBABILITIES ARE DIFFERENT THINGS
----------------------------------------------
An array outside the unit interval may rank perfectly well, but it is not a
probability and cannot support a calibration curve, a Brier score, or any
threshold applied at 0.5. Allowing one array to be both is what produced the
incoherent contract this module removes:

    the same array, invalid as a probability for calibration
                    yet accepted as a probability for the receiver operating
                    characteristic curve

`roc_curve` accepts it because it consumes SCORES, not because the array is a
valid probability.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "InputValidation",
    "validate_probabilities",
    "validate_ranking_scores",
    "validate_reference_labels",
]


@dataclass(frozen=True)
class InputValidation:
    """The verdict on one input channel.

    `ok` is not the same as "no problem anywhere". It means THIS channel is
    usable for the quantities that consume it. A report may legitimately have
    valid scores and invalid probabilities at once, and must then compute the
    ranking quantities while refusing the probability-dependent ones.
    """

    ok: bool
    reason: Optional[str] = None
    detail: Optional[str] = None

    @classmethod
    def valid(cls) -> "InputValidation":
        return cls(ok=True)

    @classmethod
    def invalid(cls, reason: str, detail: str = "") -> "InputValidation":
        return cls(ok=False, reason=reason, detail=detail or None)


def _as_1d(values: Sequence) -> "np.ndarray":
    return np.asarray(values, dtype=float).ravel()


def validate_reference_labels(y_true: Sequence, *,
                              n_expected: Optional[int] = None) -> InputValidation:
    """Reference labels must be finite, binary and aligned."""
    if y_true is None:
        return InputValidation.invalid("reference_labels_absent")
    y = _as_1d(y_true)
    if y.size == 0:
        return InputValidation.invalid("empty_population")
    if n_expected is not None and y.size != n_expected:
        return InputValidation.invalid(
            "length_mismatch", f"labels {y.size} against {n_expected}")
    if not np.isfinite(y).all():
        return InputValidation.invalid(
            "nonfinite_reference_labels",
            f"{int((~np.isfinite(y)).sum())} of {y.size}")
    observed = set(np.unique(y).tolist())
    if not observed <= {0.0, 1.0}:
        return InputValidation.invalid(
            "reference_labels_not_binary", f"observed {sorted(observed)[:6]}")
    return InputValidation.valid()


def validate_ranking_scores(scores: Sequence, *,
                            n_expected: Optional[int] = None) -> InputValidation:
    """Ranking scores must be finite and aligned. NO range restriction.

    A score is an ordering, not a magnitude on any particular scale. Requiring
    the unit interval here would reject the legitimate case this channel exists
    to serve -- a decision-function output, a log-odds, an ensemble margin.
    """
    if scores is None:
        return InputValidation.invalid("ranking_scores_absent")
    s = _as_1d(scores)
    if s.size == 0:
        return InputValidation.invalid("empty_population")
    if n_expected is not None and s.size != n_expected:
        return InputValidation.invalid(
            "length_mismatch", f"scores {s.size} against {n_expected}")
    if not np.isfinite(s).all():
        return InputValidation.invalid(
            "nonfinite_ranking_scores",
            f"{int((~np.isfinite(s)).sum())} of {s.size}")
    return InputValidation.valid()


def validate_probabilities(probabilities: Sequence, *,
                           n_expected: Optional[int] = None) -> InputValidation:
    """Probabilities must be finite, aligned AND within the unit interval.

    THE UNIT INTERVAL IS NOT PEDANTRY HERE. This channel feeds the calibration
    curve, the Brier score and every threshold applied at 0.5. A value of 1.3 is
    not an unusual probability; it is not a probability, and a reliability
    diagram built from one describes nothing.

    Measured 2026-07-28: every production caller obtains this array from
    `predict_proba(...)[:, 1]`, so enforcing the range is a correctness fix and
    not a compatibility break. A caller with genuine ranking scores has the
    `scores` channel, which does not restrict range.
    """
    if probabilities is None:
        return InputValidation.invalid("probabilities_absent")
    p = _as_1d(probabilities)
    if p.size == 0:
        return InputValidation.invalid("empty_population")
    if n_expected is not None and p.size != n_expected:
        return InputValidation.invalid(
            "length_mismatch", f"probabilities {p.size} against {n_expected}")
    if not np.isfinite(p).all():
        return InputValidation.invalid(
            "nonfinite_probabilities",
            f"{int((~np.isfinite(p)).sum())} of {p.size}")
    below = p < 0.0
    above = p > 1.0
    if below.any() or above.any():
        return InputValidation.invalid(
            "probability_out_of_unit_interval",
            f"range [{p.min():.6g}, {p.max():.6g}]")
    return InputValidation.valid()
