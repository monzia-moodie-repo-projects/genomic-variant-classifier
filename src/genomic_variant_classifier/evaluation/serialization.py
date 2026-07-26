"""Strict serialization for scientific artifacts.

WHY THIS MODULE EXISTS
======================
Two defects were measured in this project's artifact writers on 2026-07-26, both
of which corrupt evidence silently.

DEFECT 1 -- `default=str` coerces numbers into strings.
`json.dumps(payload, default=str)` calls `str()` on anything the encoder cannot
serialize. NumPy scalars are exactly that. Measured:

    json.dumps({"auroc": np.float64(0.9123), "n": np.int64(7)}, default=str)
    -> {"auroc": 0.9123, "n": "7"}

The float survived as a number because NumPy floats subclass `float`; the integer
came back as the STRING "7", because `np.int64` does not subclass `int`. A reader
that arithmetics over that column gets string concatenation or a TypeError far
downstream, and nothing at write time said a word. Removing `default=str` makes
the same payload raise `TypeError: Object of type int64 is not JSON
serializable` -- loud, at the point of the defect.

DEFECT 2 -- `NaN` is written as a bare literal that is not valid JSON.
`json.dumps` defaults to `allow_nan=True` and emits `NaN`, `Infinity` and
`-Infinity`. Python's own parser reads them back, which is why this survived, but
they are not JSON number literals: strict parsers, schema validators, other
languages and most databases reject them. Persisting a non-finite value also
conflates "no estimate was made" with "a computation produced a non-number",
which schema version 2 of the evaluation report exists to separate.

WHAT THIS MODULE DOES
---------------------
`to_json_compatible` converts the types this project actually produces --
NumPy scalars, NumPy arrays, dataclasses, enums, Paths, sets -- into JSON types
EXPLICITLY, and refuses anything it does not recognise rather than stringifying
it. `validate_json_finite` then walks the converted payload and raises naming the
exact field path, because `ValueError: Out of range float values are not JSON
compliant` from `json.dumps` tells a reader that something somewhere in a
multi-megabyte report is non-finite and nothing more.

Author: written for Monzia Moodie, 2026-07-26.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "NonFiniteArtifactValue",
    "UnserializableArtifactValue",
    "to_json_compatible",
    "validate_json_finite",
    "dump_strict_json",
]


class NonFiniteArtifactValue(ValueError):
    """A persisted artifact contained NaN, Infinity or -Infinity."""


class UnserializableArtifactValue(TypeError):
    """A persisted artifact contained a type with no defined JSON representation."""


def to_json_compatible(value: Any, *, _path: str = "$") -> Any:
    """Convert a payload to JSON-native types, refusing anything unrecognised.

    The refusal is the point. A permissive fallback such as `default=str` turns
    every unforeseen type into a plausible-looking string, so the artifact stays
    writable and becomes wrong. Raising instead means an unforeseen type is
    noticed once, at the write that introduced it.

    Non-finite floats are PRESERVED here rather than rejected, so that
    `validate_json_finite` can report every offending path in one pass instead of
    aborting at the first.
    """
    # Order matters: bool before int (bool IS an int), and the NumPy scalar
    # checks before the Python ones, because np.float64 subclasses float and
    # would otherwise be passed through as-is with its NumPy type intact.
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Enum):
        # str-mixin enums already serialize to their value, but converting
        # explicitly means a plain Enum added later behaves identically instead
        # of raising only in production.
        return to_json_compatible(value.value, _path=_path)
    if isinstance(value, np.ndarray):
        return [to_json_compatible(v, _path=f"{_path}[{i}]") for i, v in enumerate(value.tolist())]
    if isinstance(value, PurePath):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_json_compatible(dataclasses.asdict(value), _path=_path)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                raise UnserializableArtifactValue(
                    f"{_path}: dictionary key {k!r} of type {type(k).__name__} has no "
                    "defined JSON representation.")
            key = k if isinstance(k, str) else json.dumps(to_json_compatible(k))
            out[key] = to_json_compatible(v, _path=f"{_path}.{key}")
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [to_json_compatible(v, _path=f"{_path}[{i}]") for i, v in enumerate(seq)]

    raise UnserializableArtifactValue(
        f"{_path}: value of type {type(value).__name__} has no defined JSON "
        f"representation. Add an explicit conversion rather than allowing it to "
        f"be stringified, which would persist {str(value)[:60]!r} as evidence.")


def _walk_non_finite(node: Any, path: str, out: list) -> None:
    if isinstance(node, float):
        if not math.isfinite(node):
            out.append((path, node))
        return
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_non_finite(v, f"{path}.{k}", out)
        return
    if isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _walk_non_finite(v, f"{path}[{i}]", out)


def validate_json_finite(payload: Any, *, artifact: str = "artifact") -> None:
    """Raise if any float anywhere in the payload is not finite, naming its path.

    `json.dumps(..., allow_nan=False)` already refuses these, but its message is
    'Out of range float values are not JSON compliant' with no location. In a
    report that embeds full receiver-operating-characteristic,
    precision-recall and calibration curves -- between four and fifteen megabytes
    in this project -- that message is not actionable. This names every offending
    path, and reports all of them rather than only the first.
    """
    offenders: list = []
    _walk_non_finite(payload, "$", offenders)
    if offenders:
        shown = "; ".join(f"{p} = {v}" for p, v in offenders[:10])
        more = f" (and {len(offenders) - 10} more)" if len(offenders) > 10 else ""
        raise NonFiniteArtifactValue(
            f"{artifact}: {len(offenders)} non-finite value(s) cannot be persisted "
            f"as JSON: {shown}{more}. A non-finite number in an evidence artifact "
            "is either a computation that failed silently or an absent estimate "
            "wearing a number's clothes; both must be represented explicitly, not "
            "written as NaN.")


def dump_strict_json(payload: Any, *, artifact: str = "artifact", indent: int = 2) -> str:
    """Convert, audit, and serialize. The only writer path artifacts should use.

    Three gates in a fixed order: explicit type conversion, a finite-value audit
    that names paths, and `allow_nan=False` as a final backstop in case a future
    conversion rule reintroduces one.
    """
    converted = to_json_compatible(payload)
    validate_json_finite(converted, artifact=artifact)
    return json.dumps(converted, indent=indent, allow_nan=False)
