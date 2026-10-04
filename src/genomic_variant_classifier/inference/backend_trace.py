"""Post-processing of the DANDELION actual-call backend trace (owner rulings 2026-10-03b / 2026-10-04).

The R recorder (scripts/dandelion/dandelion_backend_recorder.R) instruments the INSTALLED runtime with trace() -- the pinned
source checkout is unchanged, but the runtime is instrumented, and every event says so. For each real safe_qvalues call it
writes exact hexadecimal input, post-clamp and output values and one event line. This module turns that directory into typed,
verified records:

  * strict parsing: duplicate keys, unknown or missing keys and wrong types are refused;
  * SHA-256 digests computed from the EXACT value files (base R has no SHA-256);
  * a directory holding anything the recorder did not write is refused;
  * COMPLETENESS: a caller holding a reference taken before trace() silently escapes the recorder (measured 2026-10-04), so
    when the run's planned exposures are known, each must be observed exactly once -- otherwise the trace is incomplete;
  * an "unclassified" or abnormal call is never admissible as an observation of the backend.

An actual-call trace records what the production call did; an independent replay only checks agreement (ruling
2026-10-03b). Backends are never inferred from output equality.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from genomic_variant_classifier.inference.exact_confirmation import InferenceError

logger = logging.getLogger(__name__)

__all__ = ["RECORDER_VERSION", "BackendEvent", "read_trace", "admissible_events"]

RECORDER_VERSION = "gvc.dandelion-backend-recorder/2"   # 2: writability probe at start; forked-worker guard
_KEYS = {"call", "exposure_id", "backend", "fallback_reason", "last_step_reached", "n_values", "n_distinct", "qvalue_entered",
         "p_adjust_inside_qvalue", "observation_kind", "runtime_instrumented", "method_commit", "recorder_version"}
_BACKENDS = {"BH", "qvalue", "none", "unclassified"}
_REASONS = {None, "fewer_than_10_values", "fewer_than_4_distinct_values", "qvalue_not_installed", "qvalue_warning",
            "qvalue_error", "qvalue_abnormal_exit_class_unobserved", "qvalue_abnormal_exit_multiple_conditions",
            "safe_qvalues_abnormal_exit", "observation_inconsistent_with_measured_branches"}
_CALL = re.compile(r"call-\d{4}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
# MEASURED R sprintf("%a") forms (R 4.3.3, 2026-10-04): 0x0p+0, -0x0p+0, 0x1.999999999999ap-4, subnormal 0x0.0000000000001p-1022,
# Inf, -Inf, NA, NaN. R never writes a bare "0", so none is accepted.
_HEXFLOAT = re.compile(rb"(-?0x[0-9a-f](\.[0-9a-f]+)?p[+-]\d+|Inf|-Inf|NaN|NA)\n")


def _strict(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise InferenceError("trace_duplicate_key", k)
        out[k] = v
    return out


def _refuse_constant(name):
    raise InferenceError("trace_non_json_constant", name)


@dataclass(frozen=True)
class BackendEvent:
    call: str
    exposure_id: str | None
    backend: str
    fallback_reason: str | None
    last_step_reached: int
    n_values: int | None
    n_distinct: int | None
    qvalue_entered: bool
    p_adjust_inside_qvalue: int
    method_commit: str
    input_sha256: str
    clamped_sha256: str | None
    output_sha256: str | None


def _digest(path: Path) -> str:
    raw = path.read_bytes()
    pos = 0
    while pos < len(raw):                       # every line must be one exact R value representation
        m = _HEXFLOAT.match(raw, pos)
        if m is None:
            raise InferenceError("trace_value_file_malformed", path.name)
        pos = m.end()
    return hashlib.sha256(raw).hexdigest()


def read_trace(directory, *, expected_exposures=None) -> tuple:
    """Parse and verify one recorder directory. `expected_exposures`, when given, is the run's planned exposure IDs: each
    must be observed exactly once (a missing or repeated exposure means the trace is incomplete or ambiguous)."""
    directory = Path(directory)
    events_path = directory / "events.jsonl"
    if not events_path.is_file():
        raise InferenceError("trace_events_missing", str(directory))
    lines = events_path.read_bytes().decode("utf-8").split("\n")
    if lines[-1] != "":
        raise InferenceError("trace_events_truncated")
    events, allowed = [], {"events.jsonl"}
    for n, line in enumerate(lines[:-1], 1):
        doc = json.loads(line, object_pairs_hook=_strict, parse_constant=_refuse_constant)
        if type(doc) is not dict or set(doc) != _KEYS:
            raise InferenceError("trace_event_keys", "line {}".format(n))
        call = doc["call"]
        if type(call) is not str or call != "call-{:04d}".format(n):
            raise InferenceError("trace_call_sequence", "line {}".format(n))
        if not (doc["backend"] in _BACKENDS and doc["fallback_reason"] in _REASONS
                and doc["observation_kind"] == "actual_call_trace" and doc["runtime_instrumented"] is True
                and doc["recorder_version"] == RECORDER_VERSION
                and type(doc["method_commit"]) is str and _COMMIT.fullmatch(doc["method_commit"])
                and type(doc["last_step_reached"]) is int and 2 <= doc["last_step_reached"] <= 5
                and type(doc["qvalue_entered"]) is bool and type(doc["p_adjust_inside_qvalue"]) is int
                and doc["p_adjust_inside_qvalue"] >= 0
                and all(doc[k] is None or (type(doc[k]) is int and doc[k] >= 0) for k in ("n_values", "n_distinct"))
                and (doc["exposure_id"] is None or (type(doc["exposure_id"]) is str and doc["exposure_id"]))):
            raise InferenceError("trace_event_value", "line {}".format(n))
        files = {kind: directory / "{}.{}.txt".format(call, kind) for kind in ("input", "clamped", "output")}
        if not files["input"].is_file():
            raise InferenceError("trace_value_file_missing", files["input"].name)
        clamped = files["clamped"].is_file()
        output = files["output"].is_file()
        if output != (doc["backend"] != "none") or (doc["last_step_reached"] >= 3) != clamped:
            raise InferenceError("trace_value_files_inconsistent", call)
        allowed.update(p.name for p in files.values() if p.is_file())
        events.append(BackendEvent(
            call, doc["exposure_id"], doc["backend"], doc["fallback_reason"], doc["last_step_reached"], doc["n_values"],
            doc["n_distinct"], doc["qvalue_entered"], doc["p_adjust_inside_qvalue"], doc["method_commit"],
            _digest(files["input"]), _digest(files["clamped"]) if clamped else None,
            _digest(files["output"]) if output else None))
    stray = sorted(p.name for p in directory.iterdir() if p.name not in allowed)
    if stray:
        raise InferenceError("trace_unexpected_files", repr(stray[:5]))
    if expected_exposures is not None:
        expected = list(expected_exposures)
        if len(set(expected)) != len(expected):
            raise InferenceError("expected_exposures_duplicated")
        observed = [e.exposure_id for e in events]
        if sorted(observed, key=lambda x: (x is None, x or "")) != sorted(expected):
            missing = sorted(set(expected) - set(observed))
            raise InferenceError("trace_incomplete", "missing {} / observed {} of {} expected".format(
                missing[:5], len(observed), len(expected)))
    return tuple(events)


def admissible_events(events) -> tuple:
    """Events that may be cited as observations of the backend. An unclassified or abnormal call refuses the WHOLE trace:
    the backend of that exposure is unknown, and dropping it would silently shrink the record."""
    bad = [e.call for e in events if e.backend in ("unclassified", "none")]
    if bad:
        raise InferenceError("trace_not_admissible", repr(bad[:5]))
    return tuple(events)
