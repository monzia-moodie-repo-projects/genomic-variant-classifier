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

logger = logging.getLogger(__name__)

__all__ = ["ThresholdOperator", "ThresholdParameters", "ThresholdSource"]


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
