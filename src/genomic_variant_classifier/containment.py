"""Containment boundary for quarantined scientific inputs.

Merges two reviewed reference packages, verified 2026-09-22 against their manifests and
tests on the lock environment:
  GVC_containment_boundary_reference.zip  sha256 7a065f32ce3cfb796a8040ddd37ac5f8db31a2cad9281a15e65a6b667671240e
  GVC_quarantine_reference.zip            sha256 da78cb50c6eb8836ee0f3455c5753ee921572cc4ea56319ecdcbce7f770ebff4
Their logic is carried over unchanged, with ONE deliberate change: QuarantineError now
subclasses ContainmentError. In the references they were unrelated types -- a RuntimeError
and a ValueError -- so the prescribed ``except ContainmentError: raise`` did not re-raise a
quarantine refusal, and an ordinary ``except ValueError`` fallback swallowed it (demonstrated
2026-09-22; docs: CONTAINMENT_INTEGRATION_ADDENDUM_2026-09-22.md section 3).

Deliberately NOT ported: the quarantine reference's ``guarded_annotation``, which checked only
the DECLARED outputs and was reproduced allowing a forbidden column when an innocent output was
declared. ``checked_annotation`` checks the ACTUAL output.

POLICY IS NOT DEFINED HERE. Which features are active and which are quarantined is decided in
one place, the feature-contract authority; every function takes that policy as an argument.

Trust boundary (unchanged from the references): these checks do not discover callers,
authenticate manifests, infer lineage, or implement the cohort gates C9 and C10. A SHA-256
establishes byte identity, not scientific validity. Admission callbacks and digests must come
from reviewed project code, never from an API client.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
from tempfile import SpooledTemporaryFile
from typing import BinaryIO, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)


class ContainmentError(RuntimeError):
    """Must propagate through optional-annotation and fallback handlers.

    Every broad handler on a scientific path needs ``except ContainmentError: raise``
    ahead of it. Unsupported or quarantined input is not missing evidence.
    """


class QuarantineError(ContainmentError):
    """A contract, lineage or binding refusal. A ContainmentError, so it is re-raised by the
    same handlers -- and, being a RuntimeError, it is not caught by ``except ValueError``."""


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------
def _names(values: Iterable[str] | None, where: str) -> tuple[str, ...]:
    if values is None or isinstance(values, (str, bytes)):
        raise ContainmentError(f"{where}: explicit ordered names required")
    result = tuple(values)
    if not result or any(type(x) is not str or not x or x != x.strip()
                         for x in result):
        raise ContainmentError(f"{where}: nonempty canonical names required")
    if len(result) != len(set(result)):
        raise ContainmentError(f"{where}: duplicate names")
    return result


def _sha(value: str | None, where: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContainmentError(f"{where}: full lowercase SHA-256 required")
    return value


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _qnames(values) -> tuple[str, ...]:
    """The quarantine reference's name rule, raising QuarantineError."""
    result = tuple(values)
    if not result or any(not isinstance(x, str) or not x or x != x.strip()
                         for x in result) or len(result) != len(set(result)):
        raise QuarantineError("Require nonempty, unique, explicit names")
    return result


# ---------------------------------------------------------------------------
# Boundary checks (from the containment boundary reference)
# ---------------------------------------------------------------------------
def require_producer_enabled(producer_id: str, blocked_producers: Iterable[str]) -> None:
    """Use the producer's own constant identity at every public entry point.

    In a constructor this precedes super().__init__, mkdir, session creation,
    cache lookup, and input inspection. Existing instances need method guards too.
    This is a deny rule for KNOWN producers, not authorization of unknown code.
    """
    _names((producer_id,), "producer")
    if producer_id in _names(blocked_producers, "blocked producers"):
        raise ContainmentError(f"Quarantined producer: {producer_id}")


def require_scientific_contract(columns: Iterable[str], quarantined: Iterable[str]) -> None:
    """A contract containing quarantined names is suspended explicitly."""
    cols = _names(columns, "contract")
    blocked = set(cols) & set(_names(quarantined, "quarantine"))
    if blocked:
        raise ContainmentError(f"Scientific contract suspended: {sorted(blocked)}")


def require_contract_binding(recorded: str | None, expected: str) -> None:
    """Necessary compatibility check; the project's other admission gates remain."""
    if _sha(recorded, "recorded contract") != _sha(expected, "expected contract"):
        raise ContainmentError("Artifact belongs to a different feature contract")


def require_matrix(frame: pd.DataFrame, expected: Iterable[str],
                   quarantined: Iterable[str]) -> None:
    """Check before any conversion to an anonymous NumPy array. Never zero-fill."""
    expected = _names(expected, "expected matrix")
    require_scientific_contract(expected, quarantined)
    if type(frame) is not pd.DataFrame:
        raise ContainmentError("Model boundary requires an explicit DataFrame schema")
    actual = _names(frame.columns, "actual matrix")
    if actual != expected:
        raise ContainmentError("Matrix names/order differ from the bound contract")


def _working_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """Isolate nested Python object cells too; DataFrame.copy(deep=True) does not.

    Prefer small annotation batches and a narrow, explicitly selected working
    view. This is not protection against malicious code or external side effects.
    """
    result = frame.copy(deep=True)
    for name in frame.select_dtypes(include=["object"]).columns:
        result[name] = frame[name].map(deepcopy)
    result.attrs = deepcopy(frame.attrs)
    return result


def checked_annotation(frame: pd.DataFrame, *, producer_id: str,
                       producer: Callable[[pd.DataFrame], pd.DataFrame],
                       declared_outputs: Iterable[str],
                       identity_columns: Iterable[str],
                       quarantined: Iterable[str],
                       blocked_producers: Iterable[str]) -> pd.DataFrame:
    """Check ACTUAL output on an isolated working view.

    Existing historical columns belong in the preserved source artifact. The
    caller must explicitly construct a quarantine-free active view, with its
    projection recorded. No columns are silently dropped by this function.
    Declared outputs may be added or replaced; other columns and row order must
    survive exactly. Producers may not return a reordered/subset population.
    """
    require_producer_enabled(producer_id, blocked_producers)
    if type(frame) is not pd.DataFrame or not frame.index.is_unique:
        raise ContainmentError("Annotation requires a DataFrame with unique row index")
    original = _names(frame.columns, "annotation input")
    forbidden = set(_names(quarantined, "quarantine"))
    outputs = set(_names(declared_outputs, "declared outputs"))
    identities = set(_names(identity_columns, "observation identity"))
    if not identities <= set(original) or identities & outputs:
        raise ContainmentError("Observation identity must exist and cannot be an output")
    keys = frame[sorted(identities)]
    if keys.isna().any().any() or keys.duplicated().any():
        raise ContainmentError("Observation identity is missing or duplicated")
    if forbidden & (set(original) | outputs):
        raise ContainmentError("Quarantined columns in active input or declaration")
    result = producer(_working_copy(frame))
    if type(result) is not pd.DataFrame:
        raise ContainmentError("Producer did not return a DataFrame")
    actual = set(_names(result.columns, "annotation output"))
    if forbidden & actual:
        raise ContainmentError("Producer emitted a quarantined column")
    if actual != set(original) | outputs:
        raise ContainmentError("Producer added undeclared or omitted required columns")
    if not result.index.equals(frame.index):
        raise ContainmentError("Producer changed row identity/order")
    protected = [c for c in original if c not in outputs]
    if not result[protected].equals(frame[protected]):
        raise ContainmentError("Producer changed an undeclared input column")
    return result


T = TypeVar("T")


def load_after_admission(path: Path, *, admit: Callable[[], None],
                         artifact_sha256: str,
                         deserialize: Callable[[BinaryIO], T]) -> T:
    """Admit BEFORE opening a model; deserialize precisely the checked bytes.

    admit() is the PROJECT'S trusted admission function: current contract,
    recorded binding, complete dependencies, cohort/use permissions, provenance.
    This helper does not substitute for it, and must not accept a user callback.
    The digest must be supplied by that trusted, verified manifest context.
    Hashes establish byte identity; they do not establish scientific validity.
    """
    admit()
    expected = _sha(artifact_sha256, "model artifact")
    h = hashlib.sha256()
    # Snapshot avoids hashing one path version and deserializing a later one.
    # Large artifacts spill to temporary storage rather than filling memory.
    with SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b") as snapshot:
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                h.update(block)
                snapshot.write(block)
        if h.hexdigest() != expected:
            raise ContainmentError("Model bytes differ from the admitted manifest")
        snapshot.seek(0)
        return deserialize(snapshot)


# ---------------------------------------------------------------------------
# Contract, lineage and model binding (from the quarantine reference)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Contract:
    version: str
    catalog: tuple[str, ...]
    quarantined: frozenset[str]
    # Digest of reviewed feature semantics/producer policy, not merely names.
    semantics_sha256: str

    def __post_init__(self):
        _qnames((self.version,))
        object.__setattr__(self, "catalog", _qnames(self.catalog))
        object.__setattr__(self, "quarantined", frozenset(self.quarantined))
        if not self.quarantined <= set(self.catalog):
            raise QuarantineError("Quarantine names absent from catalog")
        if not re.fullmatch(r"[0-9a-f]{64}", self.semantics_sha256):
            raise QuarantineError("Require full lowercase semantics SHA-256")
        if not self.active:
            raise QuarantineError("No active features")

    @property
    def active(self):
        return tuple(x for x in self.catalog if x not in self.quarantined)

    @property
    def sha256(self):
        return _json_sha({"version": self.version, "catalog": self.catalog,
                          "quarantined": sorted(self.quarantined),
                          "semantics_sha256": self.semantics_sha256})

    def require_raw_matrix(self, columns):
        columns = _qnames(columns)
        if columns != self.active:
            raise QuarantineError("Raw matrix differs from active ordered contract")

    def require_annotation_request(self, names):
        names = _qnames(names)
        if set(names) - set(self.active):
            raise QuarantineError("Annotation request contains unknown/quarantined names")

    def require_lineage(self, columns, derived):
        """Resolve transitive raw dependencies for every transformed column.

        Each derived entry is name -> nonempty tuple of immediate parents.
        Raw catalog names cannot be redefined. Unknown nodes and cycles refuse.
        Extra graph nodes are validated too; disconnected bad lineage is refused.
        """
        columns = _qnames(columns)
        graph = {name: _qnames(parents) for name, parents in derived.items()}
        if graph:
            _qnames(graph)
        if set(graph) & set(self.catalog):
            raise QuarantineError("Cannot redefine raw feature lineage")
        memo, visiting = {}, set()

        def roots(name):
            if name in memo:
                return memo[name]
            if name in self.catalog:
                answer = frozenset((name,))
            else:
                if name in visiting:
                    raise QuarantineError("Cyclic feature lineage")
                if name not in graph:
                    raise QuarantineError(f"Unknown feature lineage: {name}")
                visiting.add(name)
                answer = frozenset().union(*(roots(p) for p in graph[name]))
                visiting.remove(name)
            if answer & self.quarantined:
                raise QuarantineError(f"Quarantined dependency in {name}")
            memo[name] = answer
            return answer

        for name in (*graph, *columns):
            roots(name)
        return {name: tuple(sorted(memo[name])) for name in columns}


@dataclass(frozen=True)
class ModelBinding:
    contract_sha256: str
    columns: tuple[str, ...]
    lineage_sha256: str
    preprocessor_sha256: str


def bind_model(contract, columns, derived, preprocessor_sha256):
    """Called after fitting a NEW permitted pipeline, never to relabel an old one."""
    columns = _qnames(columns)
    if not re.fullmatch(r"[0-9a-f]{64}", preprocessor_sha256):
        raise QuarantineError("Require full preprocessor artifact SHA-256")
    roots = contract.require_lineage(columns, derived)
    # Bind the full graph too: equal raw roots do not imply equal transforms.
    lineage = _json_sha({"roots": roots, "graph": derived})
    return ModelBinding(contract.sha256, columns, lineage, preprocessor_sha256)


def require_model_compatibility(binding, contract, columns, derived,
                                preprocessor_sha256):
    proposed = bind_model(contract, columns, derived, preprocessor_sha256)
    if proposed != binding:
        raise QuarantineError("Model/preprocessor/contract/schema mismatch")
