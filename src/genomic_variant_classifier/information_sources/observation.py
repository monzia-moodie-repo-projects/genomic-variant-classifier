"""Stage 1D-1b.1: the typed observation contract (shadow mode; no producer emits it yet).

Owner ruling f2ec92cd (2026-11-09), section XIII: an observation must say whether a value was MEASURED or why it is
absent, so that "scientifically admitted", "actually observed" and "available to the fitted model" stop being one thing.
Today every producer writes a bare number and fills an absent value with a constant (EVE 0.5, allele frequency 0.0, ...),
so a genuine 0.5 and "no EVE model for this protein" are the same cell (EVE-COVERAGE-IDENTITY-1,
MISSINGNESS-SEMANTICS-1). This module is the contract that stage 1D-1b.2 will make producers emit; nothing imports it yet.

The rules, each refused at construction:

* OBSERVED carries a finite real number. ``True``/``False`` (and NumPy's boolean) are refused: ``math.isfinite(True)`` is
  true and ``True == 1``, so a flag would otherwise pass as the number 1. NumPy integers and floats are accepted (they are
  ``numbers.Real``) and stored as a Python ``float``, so equal measurements compare and serialise equally.
* Every other state carries NO value -- no 0.0, 0.5 or -1 sentinel may ride along with "absent".
* Source identity follows the state (refinement of the owner's reference, which accepted ``NOT_IN_SOURCE`` with no source
  and any non-empty string as an identity): a state that asserts something ABOUT a source (observed in it, absent from it,
  outside its scope) must name that source by the SHA-256 of its bytes; ``SOURCE_ABSENT`` asserts there were no source bytes
  and must name none; a mapping or computation failure may name the source it was attempting.

This module imports only the standard library, so it can be used without the model-training stack.
"""
from __future__ import annotations

import logging
import math
import numbers
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class ObservationState(str, Enum):
    OBSERVED = "observed"                         # a value measured in the named source
    SOURCE_ABSENT = "source_absent"               # the source was not available to this run at all
    NOT_IN_SOURCE = "not_in_source"               # the named source was read; this variant is not in it
    UNSUPPORTED_VARIANT = "unsupported_variant"   # the variant is outside the named source's scope (e.g. not missense)
    BAD_MAPPING = "bad_mapping"                   # the variant could not be mapped to the source's coordinates
    COMPUTATION_FAILED = "computation_failed"     # a computed feature's computation failed for this variant


class SourceRequirement(str, Enum):
    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"


#: What each state asserts about the source, and therefore whether it must name one.
SOURCE_RULE: dict[ObservationState, SourceRequirement] = {
    ObservationState.OBSERVED: SourceRequirement.REQUIRED,
    ObservationState.NOT_IN_SOURCE: SourceRequirement.REQUIRED,
    ObservationState.UNSUPPORTED_VARIANT: SourceRequirement.REQUIRED,
    ObservationState.SOURCE_ABSENT: SourceRequirement.FORBIDDEN,
    ObservationState.BAD_MAPPING: SourceRequirement.OPTIONAL,
    ObservationState.COMPUTATION_FAILED: SourceRequirement.OPTIONAL,
}


def _is_boolean(value: object) -> bool:
    # bool is a subclass of int. NumPy's boolean is neither bool nor numbers.Real; it is refused by its type name so the
    # message is the same whichever library produced it (NumPy 2 names it "bool", NumPy 1 "bool_"; measured on 2.3.3).
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_", "bool8"}


@dataclass(frozen=True, slots=True)
class NumericObservation:
    state: ObservationState
    value: float | None
    source_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ObservationState):
            raise ValueError(f"state must be an ObservationState, not {type(self.state).__name__}")
        if self.state is ObservationState.OBSERVED:
            if _is_boolean(self.value):
                raise ValueError("an observed value must be a number, not a boolean")
            if not isinstance(self.value, numbers.Real):
                raise ValueError(f"an observed value must be a real number, not {type(self.value).__name__}")
            as_float = float(self.value)
            if not math.isfinite(as_float):
                raise ValueError(f"an observed value must be finite, not {as_float!r}")
            object.__setattr__(self, "value", as_float)
        elif self.value is not None:
            raise ValueError(f"a {self.state.value} observation must carry no value (a numeric sentinel is refused)")
        rule = SOURCE_RULE[self.state]
        if self.source_id is None:
            if rule is SourceRequirement.REQUIRED:
                raise ValueError(f"a {self.state.value} observation must name its source by SHA-256")
        else:
            if rule is SourceRequirement.FORBIDDEN:
                raise ValueError(f"a {self.state.value} observation asserts there was no source; it cannot name one")
            if not isinstance(self.source_id, str) or not HEX64.match(self.source_id):
                raise ValueError("a source identity is the lower-case SHA-256 of the source's bytes")

    @property
    def is_observed(self) -> bool:
        return self.state is ObservationState.OBSERVED


def observed(value: float, source_id: str) -> NumericObservation:
    return NumericObservation(ObservationState.OBSERVED, value, source_id)


def absent(state: ObservationState, source_id: str | None = None) -> NumericObservation:
    if state is ObservationState.OBSERVED:
        raise ValueError("absent() takes a state other than OBSERVED")
    return NumericObservation(state, None, source_id)
