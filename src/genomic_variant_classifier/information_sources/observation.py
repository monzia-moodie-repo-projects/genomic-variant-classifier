"""Stage 1D-1b.1 / 1D-1b.2: the typed observation contract (shadow mode; no production module imports it).

Owner rulings f2ec92cd (section XIII) and 7f4b69d0 (section 4, 2026-11-09). An observation says whether a value was MEASURED
or why it is absent, so that "scientifically admitted", "actually observed" and "statically reachable by a model" stop being
one thing. Today every producer writes a bare number and fills an absent value with a constant (EVE 0.5, allele frequency
0.0, ...), so a genuine 0.5 and "no EVE model for this protein" are the same cell (EVE-COVERAGE-IDENTITY-1,
MISSINGNESS-SEMANTICS-1, ABSENCE-AS-EVIDENCE-1).

Stage 1D-1b.2 separates the LOGICAL source from its MATERIALISED bytes (ruling 7f4b69d0, section 4): an observation always
names which source was expected (``dataset:eve``), even when no bytes were available, and names the bytes only when bytes
were read and verified. The rules, each refused at construction:

* OBSERVED carries a finite real number. ``True``/``False`` (and NumPy's boolean) are refused: ``math.isfinite(True)`` is
  true and ``True == 1``, so a flag would otherwise pass as the number 1. NumPy integers and floats are accepted (they are
  ``numbers.Real``) and stored as a Python ``float``, so equal measurements compare and serialise equally.
* Every other state carries NO value -- no 0.0, 0.5 or -1 sentinel may ride along with "absent".
* The content digest follows the state (the owner's table): OBSERVED, NOT_IN_SOURCE and UNSUPPORTED_VARIANT assert something
  about bytes that were read and require their SHA-256; SOURCE_ABSENT asserts no bytes were read and forbids one;
  BAD_MAPPING and COMPUTATION_FAILED depend on the failure stage and accept either.
* A DERIVED value (``derived:<feature>``, computed from other observations) has no source bytes of its own: its identity is
  its inputs' identities and its propagation rule (columns.derive). It is OBSERVED, INPUT_UNOBSERVED (an input was not
  observed) or COMPUTATION_FAILED, and never names a content digest. INPUT_UNOBSERVED exists only for derived values, so a
  derived absence is never confused with a source's.

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
#: A logical source identity: a census dataset ("dataset:<registry id>") or a derived feature ("derived:<feature>").
LOGICAL_ID = re.compile(r"(dataset|derived):[a-z0-9][a-z0-9_]*\Z")


class ObservationState(str, Enum):
    OBSERVED = "observed"                         # a value measured in the named source (or derived from observed inputs)
    SOURCE_ABSENT = "source_absent"               # the source was not available to this run at all
    NOT_IN_SOURCE = "not_in_source"               # the named source was read; this variant is not in it
    UNSUPPORTED_VARIANT = "unsupported_variant"   # the variant is outside the named source's scope (e.g. not missense)
    BAD_MAPPING = "bad_mapping"                   # the variant could not be mapped (or mapped ambiguously) to the source
    COMPUTATION_FAILED = "computation_failed"     # a computed feature's computation failed for this variant
    INPUT_UNOBSERVED = "input_unobserved"         # derived values only: an input was not observed


class SourceRequirement(str, Enum):
    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"


#: For a DATASET source: whether each state must name the SHA-256 of the bytes read (the owner's table, ruling 7f4b69d0
#: section 4). INPUT_UNOBSERVED is not a dataset state.
SOURCE_RULE: dict[ObservationState, SourceRequirement] = {
    ObservationState.OBSERVED: SourceRequirement.REQUIRED,
    ObservationState.NOT_IN_SOURCE: SourceRequirement.REQUIRED,
    ObservationState.UNSUPPORTED_VARIANT: SourceRequirement.REQUIRED,
    ObservationState.SOURCE_ABSENT: SourceRequirement.FORBIDDEN,
    ObservationState.BAD_MAPPING: SourceRequirement.OPTIONAL,
    ObservationState.COMPUTATION_FAILED: SourceRequirement.OPTIONAL,
}
#: The states a derived value may take.
DERIVED_STATES: frozenset[ObservationState] = frozenset({
    ObservationState.OBSERVED, ObservationState.INPUT_UNOBSERVED, ObservationState.COMPUTATION_FAILED})


def _is_boolean(value: object) -> bool:
    # bool is a subclass of int. NumPy's boolean is neither bool nor numbers.Real; it is refused by its type name so the
    # message is the same whichever library produced it (NumPy 2 names it "bool", NumPy 1 "bool_"; measured on 2.3.3).
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_", "bool8"}


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Which source was expected (always), which bytes were read (only if any were) and its release (if known)."""
    logical_id: str
    content_sha256: str | None = None
    release: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logical_id, str) or not LOGICAL_ID.match(self.logical_id):
            raise ValueError(f"a logical source is 'dataset:<id>' or 'derived:<feature>', not {self.logical_id!r}")
        if self.content_sha256 is not None and (
                not isinstance(self.content_sha256, str) or not HEX64.match(self.content_sha256)):
            raise ValueError("a content identity is the lower-case SHA-256 of the bytes read")
        if self.is_derived and self.content_sha256 is not None:
            raise ValueError("a derived value has no source bytes of its own; it cannot name a content digest")
        if self.release is not None and (not isinstance(self.release, str) or not self.release.strip()):
            raise ValueError("a release is a non-empty string or None")

    @property
    def is_derived(self) -> bool:
        return self.logical_id.startswith("derived:")


def check_state_against_source(state: ObservationState, source: SourceIdentity) -> None:
    """Refuse a state the source cannot carry (the digest table for datasets; DERIVED_STATES for derived values)."""
    if not isinstance(state, ObservationState):
        raise ValueError(f"state must be an ObservationState, not {type(state).__name__}")
    if not isinstance(source, SourceIdentity):
        raise ValueError(f"source must be a SourceIdentity, not {type(source).__name__}")
    if source.is_derived:
        if state not in DERIVED_STATES:
            raise ValueError(f"a derived value cannot be {state.value}")
        return
    if state is ObservationState.INPUT_UNOBSERVED:
        raise ValueError("input_unobserved is a derived state; a dataset observation is never input_unobserved")
    rule = SOURCE_RULE[state]
    if source.content_sha256 is None and rule is SourceRequirement.REQUIRED:
        raise ValueError(f"a {state.value} observation must name the SHA-256 of the bytes read")
    if source.content_sha256 is not None and rule is SourceRequirement.FORBIDDEN:
        raise ValueError(f"a {state.value} observation asserts no bytes were read; it cannot name a content digest")


def check_value(state: ObservationState, value: object) -> float | None:
    """The normalised value for a state: a finite float for OBSERVED, None otherwise (sentinels refused)."""
    if state is ObservationState.OBSERVED:
        if _is_boolean(value):
            raise ValueError("an observed value must be a number, not a boolean")
        if not isinstance(value, numbers.Real):
            raise ValueError(f"an observed value must be a real number, not {type(value).__name__}")
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(f"an observed value must be finite, not {as_float!r}")
        return as_float
    if value is not None:
        raise ValueError(f"a {state.value} observation must carry no value (a numeric sentinel is refused)")
    return None


@dataclass(frozen=True, slots=True)
class NumericObservation:
    """One cell. For whole columns use columns.ObservationColumn (one value and one state code per population row)."""
    feature_id: str
    row_id: str
    state: ObservationState
    value: float | None
    source: SourceIdentity

    def __post_init__(self) -> None:
        for field_name in ("feature_id", "row_id"):
            v = getattr(self, field_name)
            if not isinstance(v, str) or not v:
                raise ValueError(f"{field_name} must be a non-empty string")
        check_state_against_source(self.state, self.source)
        object.__setattr__(self, "value", check_value(self.state, self.value))

    @property
    def is_observed(self) -> bool:
        return self.state is ObservationState.OBSERVED


def observed(feature_id: str, row_id: str, value: float, source: SourceIdentity) -> NumericObservation:
    return NumericObservation(feature_id, row_id, ObservationState.OBSERVED, value, source)


def absent(feature_id: str, row_id: str, state: ObservationState, source: SourceIdentity) -> NumericObservation:
    if state is ObservationState.OBSERVED:
        raise ValueError("absent() takes a state other than OBSERVED")
    return NumericObservation(feature_id, row_id, state, None, source)
