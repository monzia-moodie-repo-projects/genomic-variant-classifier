"""src/genomic_variant_classifier/prediction_contract.py

Author: Monzia Moodie

The prediction contract (package 1, set 1a): identity-bound probability payloads from each member to the final result.

Adapted from the owner's review bundle revision 4 (gvc_package1_boundary_review.zip, sha256 d095698e...):
prediction_contract_reference.py, the shared validators of serving_contract_reference.py and the package-1 part of
scientific_boundary_refinements.py. Frozen acceptance matrix: PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md, gates G1, G4,
G5.2 and G9.

What a digest here does and does not mean. A digest binds DECLARED bytes and metadata: the ordered prediction units, the
class order, the producer, the input contract, the split, the draw schedule, the ordered input and the execution. It
detects inconsistency between what was planned and what a producer adapter reports. It does not prove that a producer
told the truth, and it is not evidence of validity, calibration or clinical usefulness.

Two identities per prediction (owner ruling, 2026-11-07):
  source_observation_id  the source row (dataset digest + source row key)
  prediction_unit_id     one prediction made for that row in a declared unit context
One source row may expand into several prediction units; a repeated prediction unit is refused. Split role and fold are
NOT part of either identity; a change of dataset identity changes both.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ContractError",
    "TIE_RULE",
    "PREDICTION_SCHEMA_VERSION",
    "Observation",
    "ObservationSet",
    "Binding",
    "ProbabilityBlock",
    "ReplayVerifier",
    "ScientificScope",
    "DrawSpec",
    "DrawPlan",
    "PlannedReplay",
    "Quantity",
    "binary_entropy",
    "canonical_digest",
    "float_vector",
    "integer_positions",
    "model_call",
    "ordered_rows_digest",
    "positive_column",
    "prediction_records",
    "require_sha256",
    "require_text",
    "select_original_rows",
    "sklearn_binary_classes",
]

#: The ONE tie rule (G6.1). A probability of exactly one half is a positive model call everywhere in the package: the
#: serving response, VariantEnsemble.predict, evaluate() and every member wrapper.
TIE_RULE = "positive_if_p_ge_0.5"

#: The prediction record schema served by API 3.0.0.
PREDICTION_SCHEMA_VERSION = "3.0.0"

#: Row-sum tolerance for a two-class probability payload (owner reference, fixed).
_ROW_SUM_ATOL = 1e-7

_SHA256 = re.compile("[0-9a-f]{64}")


class ContractError(ValueError):
    """A prediction-contract violation. Raised before any value is published; never repaired."""


# ---------------------------------------------------------------------------------------------------------------------
# Canonical scalar validators and digests
# ---------------------------------------------------------------------------------------------------------------------

def require_text(value: Any, name: str) -> str:
    """A nonempty string with no surrounding whitespace (no coercion from other types)."""
    if type(value) is not str or not value or value != value.strip():
        raise ContractError(f"{name}: nonempty canonical string required")
    return value


def require_sha256(value: Any, name: str) -> str:
    """A full lowercase hexadecimal SHA-256 digest."""
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ContractError(f"{name}: full lowercase SHA-256 required")
    return value


def canonical_digest(domain: str, value: Any) -> str:
    """SHA-256 over a domain separator and canonical JSON (sorted keys, ASCII, no NaN or infinity)."""
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                             allow_nan=False).encode("ascii")
    except ValueError as exc:
        raise ContractError(f"{domain}: value is not finite canonical JSON ({exc})") from exc
    return hashlib.sha256(domain.encode("ascii") + b"\x00" + payload).hexdigest()


def _label(value: Any) -> tuple[str, str]:
    """Distinguish integer 1 from string "1"; refuse bool, NumPy scalars and every other implicit coercion."""
    if type(value) is int:
        return ("int", str(value))
    if type(value) is str and value and value == value.strip():
        return ("str", value)
    raise ContractError("class labels must be canonical Python int or str")


def _rows_digest(rows: Iterable["Observation"]) -> str:
    """Same canonical JSON bytes as canonical_digest(domain, list_of_row_dicts), streamed once."""
    h = hashlib.sha256(b"gvc.prediction-rows.v2\x00[")
    for i, row in enumerate(rows):
        if i:
            h.update(b",")
        h.update(json.dumps(vars(row), sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                            allow_nan=False).encode("ascii"))
    h.update(b"]")
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Prediction units (G1)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One prediction unit of one source row."""

    dataset_sha256: str
    source_row_key: str
    variant_identity: str
    unit_context_sha256: str
    source_observation_id: str = field(init=False)
    prediction_unit_id: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.dataset_sha256, "dataset_sha256")
        require_text(self.source_row_key, "source_row_key")
        require_text(self.variant_identity, "variant_identity")
        require_sha256(self.unit_context_sha256, "unit_context_sha256")
        source = canonical_digest("gvc.source-observation.v2", [self.dataset_sha256, self.source_row_key])
        object.__setattr__(self, "source_observation_id", source)
        object.__setattr__(self, "prediction_unit_id",
                           canonical_digest("gvc.prediction-unit.v2", [source, self.unit_context_sha256]))


@dataclass(frozen=True, eq=False)
class ObservationSet:
    """The ordered prediction units of one immutable context, validated and digested ONCE.

    No caller-supplied cached digest is accepted. Deliberately unhashable; equality is identity.
    """

    rows: tuple[Observation, ...]
    digest: str = field(init=False)
    source_count: int = field(init=False)
    max_units_per_source: int = field(init=False)
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or not self.rows:
            raise ContractError("nonempty immutable row tuple required")
        if any(type(r) is not Observation for r in self.rows):
            raise ContractError("Observation rows required")
        if len({r.prediction_unit_id for r in self.rows}) != len(self.rows):
            raise ContractError("duplicate prediction units")
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.source_observation_id] = counts.get(row.source_observation_id, 0) + 1
        object.__setattr__(self, "source_count", len(counts))
        object.__setattr__(self, "max_units_per_source", max(counts.values()))
        object.__setattr__(self, "digest", _rows_digest(self.rows))

    def __len__(self) -> int:
        return len(self.rows)


def ordered_rows_digest(rows: Any) -> str:
    if type(rows) is not ObservationSet:
        raise ContractError("construct an ObservationSet once before producing predictions")
    return rows.digest


def integer_positions(raw: Any) -> tuple[list[int], str]:
    """Positions as Python ints through ONE recorded conversion; bool, string and float positions are refused (G1.4)."""
    if isinstance(raw, np.ndarray):
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ContractError("integer position vector required")
        return raw.tolist(), f"numpy.{raw.dtype.name}.tolist_to_python_int"
    if type(raw) not in (list, tuple) or any(type(i) is not int for i in raw):
        raise ContractError("explicit integer positions required")
    return list(raw), "none"


def select_original_rows(rows: ObservationSet, original_positions: Iterable[int]) -> ObservationSet:
    """Select by parent-table positions once, without reassigning source row keys."""
    ordered_rows_digest(rows)
    positions = tuple(original_positions)
    if not positions or any(type(i) is not int for i in positions):
        raise ContractError("nonempty explicit Python integer positions required")
    if len(set(positions)) != len(positions):
        raise ContractError("duplicate fit positions")
    if any(i < 0 or i >= len(rows) for i in positions):
        raise ContractError("fit position outside the parent table")
    return ObservationSet(tuple(rows.rows[i] for i in positions))


# ---------------------------------------------------------------------------------------------------------------------
# Identity-bound probability payloads (G4)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Binding:
    """What produced a payload, from what, under which split, schedule, input and execution."""

    producer_sha256: str
    input_contract_sha256: str
    split_sha256: str
    draw_schedule_sha256: str
    ordered_input_sha256: str
    execution_sha256: str

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            require_sha256(value, name)


@dataclass(frozen=True, eq=False)
class ProbabilityBlock:
    """Immutable two-class probabilities with explicit identity; deliberately unhashable.

    Only float32 and float64 producer arrays are admitted. Values are stored as little-endian float64 bytes; the
    original precision is bound separately. No clipping, normalisation or missing-row deletion.
    """

    rows: ObservationSet
    class_order: tuple[Any, Any]
    positive_class: Any
    binding: Binding
    draw_id: str
    producer_precision: str
    payload: bytes
    payload_sha256: str = field(init=False)
    _block_digest: str = field(init=False, repr=False)
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if type(self.rows) is not ObservationSet or type(self.class_order) is not tuple:
            raise ContractError("ObservationSet and immutable class tuple required")
        ordered_rows_digest(self.rows)
        labels = tuple(_label(x) for x in self.class_order)
        if len(labels) != 2 or len(set(labels)) != 2 or _label(self.positive_class) not in labels:
            raise ContractError("two distinct classes and a present positive class required")
        if type(self.binding) is not Binding:
            raise ContractError("Binding required")
        require_text(self.draw_id, "draw_id")
        if self.producer_precision not in ("float32", "float64"):
            raise ContractError("unsupported producer precision")
        if type(self.payload) is not bytes or len(self.payload) != len(self.rows) * 2 * 8:
            raise ContractError("invalid immutable probability payload")
        p = self.probabilities()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ContractError("probabilities must be finite and in [0, 1]; no clipping")
        if not np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=_ROW_SUM_ATOL):
            raise ContractError("class probabilities do not sum to one")
        object.__setattr__(self, "payload_sha256", hashlib.sha256(self.payload).hexdigest())
        object.__setattr__(self, "_block_digest", canonical_digest("gvc.probability-block.v2", self.descriptor()))

    @classmethod
    def from_array(cls, *, rows: ObservationSet, class_order: tuple[Any, Any], positive_class: Any, binding: Binding,
                   draw_id: str, probabilities: Any) -> "ProbabilityBlock":
        p = np.asarray(probabilities)
        if type(rows) is not ObservationSet:
            raise ContractError("prevalidated ObservationSet required")
        if p.dtype.kind != "f" or p.dtype.itemsize not in (4, 8):
            raise ContractError("float32/float64 probabilities required; no coercion")
        if p.shape != (len(rows), 2):
            raise ContractError("probability matrix must have shape (rows, 2)")
        precision = "float32" if p.dtype.itemsize == 4 else "float64"
        return cls(rows, tuple(class_order), positive_class, binding, draw_id, precision,
                   np.asarray(p, dtype="<f8", order="C").tobytes(order="C"))

    def probabilities(self) -> np.ndarray:
        # The bytes own the storage, so writeability cannot be re-enabled on this view.
        return np.frombuffer(self.payload, dtype="<f8").reshape(len(self.rows), 2)

    def positive_probability(self) -> np.ndarray:
        positive = _label(self.positive_class)
        column = tuple(_label(x) for x in self.class_order).index(positive)
        return self.probabilities()[:, column]

    def descriptor(self) -> dict[str, Any]:
        return {"schema": "gvc.probability-block.v2",
                "n_rows": len(self.rows), "rows_sha256": ordered_rows_digest(self.rows),
                "class_order": [_label(x) for x in self.class_order],
                "positive_class": _label(self.positive_class),
                "binding": vars(self.binding), "draw_id": self.draw_id,
                "producer_precision": self.producer_precision,
                "payload_sha256": self.payload_sha256}

    @property
    def digest(self) -> str:
        return self._block_digest


def prediction_records(block: ProbabilityBlock):
    """Rows for an existing writer: identity and values travel together (no rounding)."""
    if type(block) is not ProbabilityBlock:
        raise ContractError("ProbabilityBlock required")
    for row, p in zip(block.rows.rows, block.positive_probability(), strict=True):
        yield {"source_observation_id": row.source_observation_id,
               "prediction_unit_id": row.prediction_unit_id,
               "dataset_sha256": row.dataset_sha256,
               "source_row_key": row.source_row_key,
               "variant_identity": row.variant_identity,
               "unit_context_sha256": row.unit_context_sha256,
               "positive_probability": float(p)}


# ---------------------------------------------------------------------------------------------------------------------
# Producer-output validators (G4.5)
# ---------------------------------------------------------------------------------------------------------------------

def float_vector(values: Any) -> np.ndarray:
    """A nonempty finite float32/float64 probability vector in [0, 1], widened to float64. Never clipped."""
    p = np.asarray(values)
    if p.dtype.kind != "f" or p.dtype.itemsize not in (4, 8):
        raise ContractError("float32/float64 probabilities required")
    if p.ndim != 1 or p.size == 0:
        raise ContractError("nonempty probability vector required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ContractError("invalid probability; clipping is forbidden")
    return p.astype(np.float64, copy=False)


def binary_entropy(values: Any) -> np.ndarray:
    """Exact Bernoulli entropy in nats, 0 log 0 := 0; log1p for the complement term."""
    p = float_vector(values)
    result = np.zeros_like(p)
    inside = (p > 0) & (p < 1)
    q = p[inside]
    result[inside] = -q * np.log(q) - (1 - q) * np.log1p(-q)
    return result


def sklearn_binary_classes(estimator: Any) -> tuple[tuple[int, int], str]:
    """The fitted classes, converted once and recorded; exactly the canonical integers {0, 1}, in either order."""
    raw = getattr(estimator, "classes_", None)
    if isinstance(raw, np.ndarray):
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ContractError("binary integer classes_ required")
        classes = raw.tolist()
        conversion = f"numpy.{raw.dtype.name}.tolist_to_python_int"
    elif type(raw) in (list, tuple):
        classes = list(raw)
        conversion = "none"
    else:
        raise ContractError("fitted classes_ required; column 1 is never assumed")
    if len(classes) != 2 or any(type(v) is not int for v in classes) or set(classes) != {0, 1}:
        raise ContractError("this adapter requires exactly the integer classes {0, 1}")
    return (classes[0], classes[1]), conversion


def positive_column(estimator: Any, values: Any, n: int) -> tuple[np.ndarray, tuple[int, int], str]:
    """Validate EVERY column of a producer's (n, 2) output, then return the class-1 column by fitted class order."""
    classes, conversion = sklearn_binary_classes(estimator)
    p = np.asarray(values)
    if p.shape != (n, 2) or p.dtype.kind != "f" or p.dtype.itemsize not in (4, 8):
        raise ContractError(f"producer must return float32/64 of shape ({n}, 2); got {p.dtype} {p.shape}")
    for j in range(2):
        float_vector(p[:, j])
    # The selected column alone can look valid, so the row sums are checked too.
    if not np.allclose(p.astype(np.float64).sum(axis=1), 1.0, atol=_ROW_SUM_ATOL, rtol=0.0):
        raise ContractError("invalid class probability sums")
    return p[:, classes.index(1)], classes, conversion


def model_call(p: Any) -> np.ndarray:
    """The binary model call under TIE_RULE. Descriptive only: not a clinical classification."""
    return (float_vector(np.ravel(p)) >= 0.5).astype(np.int64)


# ---------------------------------------------------------------------------------------------------------------------
# Quantities with units and states (G5.2)
# ---------------------------------------------------------------------------------------------------------------------

_UNITS = frozenset({"nats", "probability_squared", "probability"})


@dataclass(frozen=True)
class Quantity:
    """A value with its unit and state. Unavailable and unresolved values are null WITH a reason; a computed zero
    stays zero."""

    value: float | None
    unit: str
    status: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.unit not in _UNITS:
            raise ContractError(f"unknown unit {self.unit!r}")
        if self.status == "defined":
            if type(self.value) is not float or not math.isfinite(self.value) or self.value < 0:
                raise ContractError("a defined quantity requires a finite nonnegative float")
            if self.reason is not None:
                raise ContractError("a defined quantity cannot carry an unavailability reason")
        elif self.status in {"unavailable", "unresolved"}:
            if self.value is not None:
                raise ContractError("an unavailable or unresolved value must be null")
            require_text(self.reason, "reason")
        else:
            raise ContractError(f"unknown quantity status {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------------------
# Replay (G9)
# ---------------------------------------------------------------------------------------------------------------------

class ReplayVerifier:
    """O(draw count) digest ledger; no draw arrays retained.

    Feed first-pass blocks to observe(), seal(), then check() second-pass blocks. Only finish() returns replay
    evidence; a replayed prefix is incomplete.
    """

    def __init__(self) -> None:
        self._expected: list[str] = []
        self._ids: set[str] = set()
        self._state = "first_pass"
        self._next = 0
        self._context: dict[str, Any] | None = None

    @staticmethod
    def _context_of(block: Any) -> dict[str, Any]:
        if type(block) is not ProbabilityBlock:
            raise ContractError("ProbabilityBlock required")
        d = block.descriptor()
        return {k: v for k, v in d.items() if k not in ("draw_id", "payload_sha256")}

    def observe(self, block: ProbabilityBlock) -> None:
        if self._state != "first_pass":
            raise ContractError("first pass is closed")
        try:
            context = self._context_of(block)
        except ContractError:
            self._state = "failed"
            raise
        if self._context is not None and context != self._context:
            self._state = "failed"
            raise ContractError("draw context changed within the mixture")
        if block.draw_id in self._ids:
            self._state = "failed"
            raise ContractError("duplicate draw identity")
        self._context = context
        self._ids.add(block.draw_id)
        self._expected.append(block.digest)

    def seal(self) -> None:
        if self._state != "first_pass" or len(self._expected) < 2:
            raise ContractError("need at least two first-pass draws, sealed once")
        self._state = "replay"

    def check(self, block: ProbabilityBlock) -> None:
        if self._state != "replay":
            raise ContractError("replay not active")
        if type(block) is not ProbabilityBlock or self._next >= len(self._expected):
            self._state = "failed"
            raise ContractError("unexpected replay draw")
        if block.digest != self._expected[self._next]:
            self._state = "failed"
            raise ContractError("replay differs in values, identity or order")
        self._next += 1

    def finish(self) -> dict[str, Any]:
        if self._state != "replay" or self._next != len(self._expected):
            self._state = "failed"
            raise ContractError("replay incomplete or invalid")
        self._state = "complete"
        return {"draws_verified": self._next,
                "replay_sha256": canonical_digest("gvc.replay.v1", self._expected),
                "status": "identical_declared_blocks"}


@dataclass(frozen=True)
class ScientificScope:
    """References to authoritative definitions (task, method, numerical policy), not duplicated prose."""

    task_sha256: str
    method_sha256: str
    numerical_policy_sha256: str

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            require_sha256(value, name)

    @property
    def digest(self) -> str:
        return canonical_digest("gvc.scientific-scope.v1", asdict(self))


@dataclass(frozen=True)
class DrawSpec:
    draw_id: str
    member_sha256: str
    seed: int | None

    def __post_init__(self) -> None:
        require_text(self.draw_id, "draw_id")
        require_sha256(self.member_sha256, "member_sha256")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**31):
            raise ContractError("seed must be a Python int in the declared 31-bit adapter domain")


_DRAW_MECHANISMS = frozenset({"mc_dropout", "fitted_ensemble", "algorithm_roster"})


@dataclass(frozen=True)
class DrawPlan:
    """The declared, equally weighted draw schedule of one mixture (G9.2)."""

    owner_sha256: str
    execution_sha256: str
    scope: ScientificScope
    mechanism: str
    draws: tuple[DrawSpec, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.owner_sha256, "owner_sha256")
        require_sha256(self.execution_sha256, "execution_sha256")
        if type(self.scope) is not ScientificScope:
            raise ContractError("ScientificScope required")
        if self.mechanism not in _DRAW_MECHANISMS:
            raise ContractError("unknown draw mechanism")
        if type(self.draws) is not tuple or len(self.draws) < 2:
            raise ContractError("at least two immutable draws required for a mixture")
        if any(type(d) is not DrawSpec for d in self.draws):
            raise ContractError("DrawSpec entries required")
        if len({d.draw_id for d in self.draws}) != len(self.draws):
            raise ContractError("duplicate draw identity")
        if self.mechanism == "mc_dropout":
            seeds = [d.seed for d in self.draws]
            if None in seeds or len(set(seeds)) != len(seeds):
                raise ContractError("distinct declared MC seeds required")
            if len({d.member_sha256 for d in self.draws}) != 1:
                raise ContractError("MC draws must refer to one fitted member")
        value = {"owner_sha256": self.owner_sha256, "execution_sha256": self.execution_sha256,
                 "scope": asdict(self.scope), "mechanism": self.mechanism,
                 "draws": [asdict(d) for d in self.draws],
                 # Equal weights by construction; a weighted mixture needs a separately qualified method.
                 "weighting": "uniform_over_declared_draws"}
        object.__setattr__(self, "digest", canonical_digest("gvc.draw-plan.v1", value))


class PlannedReplay:
    """Schedule validation composed with the exact replay checker.

    actual_draw and actual_scope must come from the executed producer adapter, not be copied from the plan.
    """

    def __init__(self, plan: DrawPlan) -> None:
        if type(plan) is not DrawPlan:
            raise ContractError("DrawPlan required")
        self._plan = plan
        self._replay = ReplayVerifier()
        self._state = "first"
        self._first = 0
        self._second = 0

    @property
    def plan(self) -> DrawPlan:
        return self._plan

    def _check_binding(self, block: Any, actual_draw: Any, actual_scope: Any, index: int) -> None:
        if type(block) is not ProbabilityBlock or type(actual_draw) is not DrawSpec:
            raise ContractError("typed block and executed draw required")
        if type(actual_scope) is not ScientificScope or actual_scope != self.plan.scope:
            raise ContractError("scientific scope differs from the planned task/method/numerics")
        if index >= len(self.plan.draws) or actual_draw != self.plan.draws[index]:
            raise ContractError("executed draw differs from the planned schedule")
        if block.draw_id != actual_draw.draw_id:
            raise ContractError("block draw identity differs from the invocation")
        b = block.binding
        if (b.producer_sha256 != self.plan.owner_sha256 or b.draw_schedule_sha256 != self.plan.digest
                or b.execution_sha256 != self.plan.execution_sha256):
            raise ContractError("block binding differs from the execution plan")

    def observe(self, block: ProbabilityBlock, *, actual_draw: DrawSpec, actual_scope: ScientificScope) -> None:
        try:
            if self._state != "first":
                raise ContractError("first pass is closed")
            self._check_binding(block, actual_draw, actual_scope, self._first)
            self._replay.observe(block)
            self._first += 1
        except Exception:
            self._state = "failed"
            raise

    def seal(self) -> None:
        try:
            if self._state != "first" or self._first != len(self.plan.draws):
                raise ContractError("first pass did not execute the complete planned schedule")
            self._replay.seal()
            self._state = "second"
        except Exception:
            self._state = "failed"
            raise

    def check(self, block: ProbabilityBlock, *, actual_draw: DrawSpec, actual_scope: ScientificScope) -> None:
        try:
            if self._state != "second":
                raise ContractError("replay is not active")
            self._check_binding(block, actual_draw, actual_scope, self._second)
            self._replay.check(block)
            self._second += 1
        except Exception:
            self._state = "failed"
            raise

    def finish(self) -> dict[str, Any]:
        try:
            if self._state != "second" or self._second != len(self.plan.draws):
                raise ContractError("planned replay is incomplete")
            evidence = self._replay.finish()
            self._state = "complete"
            return {**evidence, "planned_draws": len(self.plan.draws), "draw_plan_sha256": self.plan.digest,
                    "scientific_scope_sha256": self.plan.scope.digest,
                    "schedule_status": "matches_declared_execution_plan"}
        except Exception:
            self._state = "failed"
            raise


def frozen_digest_items(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Sorted immutable (name, digest) pairs with validated names and digests."""
    items = []
    for name, digest in value.items():
        items.append((require_text(name, "name"), require_sha256(digest, "digest")))
    return tuple(sorted(items))
