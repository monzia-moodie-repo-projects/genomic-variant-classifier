"""ONE definition of admission, reused by every orchestration script (owner rulings 2026-10-08, 2026-10-08b).

The repairs of ten defects found in separate scripts converge here: each script previously redefined what "accepted", "the same plan", or
"the qualified library" meant (the downloader accepted "already_accepted" while its consumers selected only "accepted"; consumers compared a
plan's digest LABEL instead of recomputing it; an empty manifest passed the contents check). This module reuses r_runtime's owners --
strict_json, canonical, validate_lock, admit_runtime_change, sha256_file -- and adds only what did not exist:

  admit_artifact_set     every planned artifact exactly once, an admissible acquisition status, bytes still matching the recorded digest
  plan_digest            the ONE recomputation convention (canonical serialization without the plan_sha256 key -- byte-compatible with
                         every recorded replay-plan digest)
  library_digest         the ONE directory content digest (sorted relative paths + file digests; links refused)
  exact_field_diff       every differing lockfile field with EXPLICIT presence (ABSENT is not null) and exact JSON values incl. types
  admit_lock_transition  exactly the approved R change, exactly the reviewed additions, no removals or version changes, and exactly the
                         individually approved field transitions (any unexpected difference stops admission)
  RunRecord              an outer failure-recording boundary: a "started" record first, the active stage, an atomically written terminal
                         record; a MISSING terminal record means INCOMPLETE (forced termination or power loss can prevent the write)
  artifact_readiness     artifact-input readiness judged against requirements derived INDEPENDENTLY from the admitted installation plan
                         (ruling 2026-10-08e section 4): an inventory record that defines both what was required and what was observed
                         could agree with itself while omitting a required artifact. readiness_decision binds the result to the plan
                         documents, the record's exact bytes, the evaluation time and the IMPLEMENTATION that determined it (ruling
                         2026-10-08f section 4): the verified repository tree, the collector's exact bytes, and the record owner's and
                         this module's canonical LF text identities -- digest domains declared, never interchangeable. It is HISTORICAL
                         readiness only; installation admission must recheck the bytes it is about to consume.

An archive checksum identifies THAT archive, not equivalence to a historical artifact.

Author: Monzia Moodie
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from genomic_variant_classifier.environment_qualification.r_runtime import (
    AdmissionError, admit_runtime_change, canonical, require, sha256_file, validate_lock)

logger = logging.getLogger(__name__)

__all__ = ["SUCCESS", "ABSENT", "admit_artifact_set", "plan_digest", "verify_plan", "library_digest", "exact_field_diff",
           "admit_lock_transition", "RunRecord", "EvidenceState", "CheckRequirement", "CheckResult", "QualificationDecision", "decide_qualification",
           "artifact_readiness", "readiness_decision", "canonical_text_sha256", "DIGEST_DOMAINS"]

SUCCESS = frozenset({"accepted", "already_accepted"})       # the downloader's own success set (download_artifacts.py)
_KINDS = frozenset({"source", "windows_binary"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


class _Absent:
    """A field that is NOT PRESENT -- distinct from a present field whose JSON value is null."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "ABSENT"


ABSENT = _Absent()


def _artifact_key(row) -> tuple:
    require(isinstance(row, dict), "artifact.row_shape")
    values = tuple(row.get(k) for k in ("package", "version", "kind"))
    require(all(type(v) is str and v for v in values), "artifact.identity_invalid")
    require(values[2] in _KINDS, "artifact.kind_invalid")
    return values


def admit_artifact_set(planned, observed) -> dict:
    """planned: rows with package / version / kind (the reviewed scope); observed: acquisition-manifest rows. Returns key -> row."""
    planned, observed = list(planned), list(observed)
    expected, actual = Counter(_artifact_key(r) for r in planned), Counter(_artifact_key(r) for r in observed)
    require(len(expected) > 0, "artifact.plan_empty")
    require(all(n == 1 for n in expected.values()), "artifact.plan_duplicate")
    require(all(n == 1 for n in actual.values()), "artifact.manifest_duplicate")
    require(actual == expected, "artifact.coverage_mismatch")
    for row in observed:
        require(type(row.get("decision")) is str and row["decision"] in SUCCESS, "artifact.not_admitted")   # [] would be unhashable
        digest = row.get("sha256")
        require(type(digest) is str and _SHA256.fullmatch(digest) is not None, "artifact.digest_invalid")
        path = row.get("accepted_path")
        require(type(path) is str and Path(path).is_file(), "artifact.file_missing")
        require(sha256_file(path) == digest, "artifact.bytes_changed")
    return {_artifact_key(r): r for r in observed}


def plan_digest(plan: dict) -> str:
    """The plan's digest RECOMPUTED from its body (the plan_sha256 key excluded)."""
    require(isinstance(plan, dict), "plan.shape")
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    return hashlib.sha256(canonical(body).encode("ascii")).hexdigest()


def verify_plan(plan: dict, expected_digest: str) -> str:
    """The plan BODY must hash to expected_digest; its own label must agree. A modified body keeping the old label is refused."""
    computed = plan_digest(plan)
    require(computed == expected_digest, "plan.body_digest_mismatch")
    require(plan.get("plan_sha256") == computed, "plan.label_mismatch")
    return computed


def library_digest(root) -> str:
    """Content digest of a directory: sorted relative paths + file SHA-256s; links (and Windows junctions) refused; no timestamps."""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "library.missing")
    require(not (hasattr(os.path, "isjunction") and os.path.isjunction(root)), "library.root_junction")
    h = hashlib.sha256()
    # os.walk IGNORES scanning errors unless onerror is given -- a failed scan would silently OMIT content (measured: an injected error gave the
    # digest of NOTHING). The encoding below is unchanged, so valid historical identities stay comparable.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=_reject_walk_error):
        dirnames.sort()
        here = Path(dirpath)
        for name in dirnames + filenames:
            item = here / name
            require(not item.is_symlink() and not (hasattr(os.path, "isjunction") and os.path.isjunction(item)),
                    "library.link:" + item.relative_to(root).as_posix())
        for name in sorted(filenames):
            item = here / name
            require(item.is_file(), "library.non_regular_file:" + item.relative_to(root).as_posix())     # a FIFO would block the hash
            h.update(item.relative_to(root).as_posix().encode("utf-8") + b"\0" + sha256_file(item).encode("ascii") + b"\n")
    return h.hexdigest()


def _reject_walk_error(error):
    raise AdmissionError("library.traversal_failed:" + str(getattr(error, "filename", "")))


def exact_field_diff(before: dict, after: dict) -> list:
    """Every differing field of packages present in BOTH locks: (package, field, old, new) with ABSENT for a missing field. Values are
    compared as canonical JSON, so types are distinguished (true vs 1, "1" vs 1). Sorted, deterministic."""
    validate_lock(before)
    validate_lock(after)
    out = []
    for package in sorted(set(before["Packages"]) & set(after["Packages"])):
        b, a = before["Packages"][package], after["Packages"][package]
        for field in sorted(set(b) | set(a)):
            old, new = b.get(field, ABSENT), a.get(field, ABSENT)
            same = old is not ABSENT and new is not ABSENT and canonical(old) == canonical(new)
            if not same and not (old is ABSENT and new is ABSENT):
                out.append((package, field, old, new))
    return out


def _transition_key(t) -> tuple:
    require(isinstance(t, dict), "transition.shape")
    for k in ("package", "field", "old", "new", "classification", "evidence"):
        require(k in t, "transition.missing:" + k)
    require(type(t["package"]) is str and type(t["field"]) is str and t["package"] and t["field"], "transition.identity")
    require(type(t["classification"]) is str and t["classification"] and type(t["evidence"]) is str and t["evidence"], "transition.justification")
    old, new = _transition_side(t["old"]), _transition_side(t["new"])
    return (t["package"], t["field"], old, new)


def _transition_side(value) -> tuple:
    """STRICT shapes: {"absent": true} exactly (1, 1.0, false, null are NOT true -- `v == {"absent": True}` accepted 1 and 1.0), or {"value": v}."""
    require(type(value) is dict, "transition.value_shape")
    if set(value) == {"absent"}:
        require(value["absent"] is True, "transition.value_shape")
        return ("absent",)
    require(set(value) == {"value"}, "transition.value_shape")
    return ("present", canonical(value["value"]))


def admit_lock_transition(before: dict, after: dict, *, observed_version: str, additions: dict, approved_transitions) -> dict:
    """Exactly: the approved R change (via admit_runtime_change), the reviewed additions at their versions, no removals or version
    changes, and the field differences EQUAL to the reviewed exact transitions. A transition is
    {"package", "field", "old": {"absent": true} | {"value": <json>}, "new": ..., "classification", "evidence"}."""
    validate_lock(before)
    validate_lock(after)
    require(isinstance(additions, dict) and all(type(k) is str and type(v) is str for k, v in additions.items()), "lock.additions_shape")
    added, removed = set(after["Packages"]) - set(before["Packages"]), set(before["Packages"]) - set(after["Packages"])
    require(not removed, "lock.removed:" + ",".join(sorted(removed)))
    require(added == set(additions), "lock.additions_mismatch")
    for name in sorted(added):
        require(after["Packages"][name].get("Version") == additions[name], "lock.addition_version:" + name)
    for name in sorted(set(before["Packages"]) & set(after["Packages"])):
        require(before["Packages"][name].get("Version") == after["Packages"][name].get("Version"), "lock.version_changed:" + name)
    approved = [_transition_key(t) for t in approved_transitions]
    require(len(approved) == len(set(approved)), "transition.duplicate")
    actual = [(p, f, ("absent",) if o is ABSENT else ("present", canonical(o)), ("absent",) if n is ABSENT else ("present", canonical(n)))
              for p, f, o, n in exact_field_diff(before, after)]
    require(set(actual) == set(approved), "lockfile.unapproved_transition")
    # The remainder after removing the additions AND reverting exactly the approved transitions must be the approved R change alone.
    reverted = copy.deepcopy(after)
    for name in added:
        del reverted["Packages"][name]
    for p, f, o, n in exact_field_diff(before, after):
        if o is ABSENT:
            del reverted["Packages"][p][f]
        else:
            reverted["Packages"][p][f] = copy.deepcopy(o)
    delta = admit_runtime_change(before, reverted, observed_version)
    return {"schema": "gvc.lock-transition/1", "runtime": delta, "additions": dict(sorted(additions.items())),
            "transitions": len(approved), "packages_after": len(after["Packages"])}


class EvidenceState(str, Enum):
    """Two axes kept unambiguous (ruling 2026-10-08c): were the bytes accessible, and did they match. A historical receipt is NOT a MATCH."""
    MATCH = "match"                         # bytes read and matched the expected identity
    MISMATCH = "mismatch"                   # bytes read and did not match
    UNAVAILABLE = "unavailable"             # required bytes could not be obtained or read
    INVALID = "invalid_evidence"            # the evidence record could not be interpreted under its schema


@dataclass(frozen=True)
class CheckRequirement:
    check_id: str
    subject_sha256: str


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    subject_sha256: str
    specification_sha256: str
    verifier_sha256: str
    state: EvidenceState
    evidence_sha256: str                    # the digest of the EVIDENCE record (which may document unavailable bytes) -- never of the subject
    reason: str


@dataclass(frozen=True)
class QualificationDecision:
    admitted: bool
    reasons: tuple


def _digest(value, reason: str) -> None:
    require(type(value) is str and _SHA256.fullmatch(value) is not None, reason)


def decide_qualification(requirements, results, *, specification_sha256: str, verifier_sha256: str) -> QualificationDecision:
    """PURE (no filesystem access, no side effects): every required check exactly once, each result bound to the SAME subject, the sealed
    specification and the verifier implementation, and in state MATCH. Malformed inputs refuse (AdmissionError); a well-formed but
    insufficient result set returns admitted=False with reasons -- a checker can execute perfectly and correctly refuse."""
    _digest(specification_sha256, "qualification.specification_digest_invalid")
    _digest(verifier_sha256, "qualification.verifier_digest_invalid")
    requirements, results = tuple(requirements), tuple(results)
    require(len(requirements) > 0, "qualification.requirements_empty")
    expected = {}
    for item in requirements:
        require(type(item) is CheckRequirement, "qualification.requirement_type")
        require(type(item.check_id) is str and item.check_id != "", "qualification.check_id_invalid")
        _digest(item.subject_sha256, "qualification.subject_digest_invalid")
        require(item.check_id not in expected, "qualification.requirement_duplicate")
        expected[item.check_id] = item
    observed = {}
    for result in results:
        require(type(result) is CheckResult, "qualification.result_type")
        require(type(result.check_id) is str and result.check_id != "", "qualification.check_id_invalid")
        require(result.check_id not in observed, "qualification.result_duplicate")
        observed[result.check_id] = result
    if set(observed) != set(expected):
        return QualificationDecision(False, ("qualification.coverage_mismatch",))
    reasons = []
    for check_id in sorted(expected):
        result = observed[check_id]
        for value in (result.subject_sha256, result.specification_sha256, result.verifier_sha256, result.evidence_sha256):
            _digest(value, "qualification.result_digest_invalid")
        require(type(result.state) is EvidenceState, "qualification.state_invalid")
        require(type(result.reason) is str and result.reason != "", "qualification.reason_missing")
        if result.subject_sha256 != expected[check_id].subject_sha256:
            reasons.append(check_id + ":subject_mismatch")
        if result.specification_sha256 != specification_sha256:
            reasons.append(check_id + ":specification_mismatch")
        if result.verifier_sha256 != verifier_sha256:
            reasons.append(check_id + ":verifier_mismatch")
        if result.state is not EvidenceState.MATCH:
            reasons.append(check_id + ":" + result.state.value)
    return QualificationDecision(not reasons, tuple(reasons))


class RunRecord:
    """An outer failure-recording boundary that keeps EXECUTION separate from ADMISSION (ruling 2026-10-08c): a checker can execute perfectly
    and correctly REFUSE admission. The run directory is created EXCLUSIVELY (an exists() check followed by a write is not multi-writer
    exclusion). A "started" record is written first; the terminal record is written atomically; a record still saying "started" (or none)
    means INCOMPLETE -- forced termination, power loss or storage failure can prevent the terminal write. Callers derive their exit status from
    admission_status, never from execution alone."""

    def __init__(self, run_dir, name: str = "run_record.json"):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / name
        self.data = {"execution_status": "started", "admission_status": "undetermined", "reason": None, "stage": None, "started_utc": _utc()}

    def _write(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes((json.dumps(self.data, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii"))
        os.replace(tmp, self.path)

    def stage(self, name: str):
        self.data["stage"] = name
        self._write()

    def decide(self, admitted: bool, reason: str | None = None):
        require(type(admitted) is bool, "record.admitted_type")
        require(admitted or (type(reason) is str and reason != ""), "record.refusal_reason_required")
        self.data.update(admission_status="admitted" if admitted else "refused", reason=reason)
        self._write()

    def __enter__(self):
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)          # EXCLUSIVE ownership of the run directory
        except FileExistsError:
            raise AdmissionError("record.run_directory_exists")
        self._write()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.data["ended_utc"] = _utc()
        if exc_type is None:
            self.data["execution_status"] = "completed"
        else:
            self.data.update(execution_status="failed", failed_stage=self.data.get("stage"), exception_class=exc_type.__name__,
                             diagnostic=str(exc)[:2000])
        self._write()
        return False                                    # never swallow the exception

    @property
    def exit_code(self) -> int:
        return 0 if self.data["execution_status"] == "completed" and self.data["admission_status"] == "admitted" else 1


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------------------- artifact-input readiness
def artifact_readiness(record, required) -> dict:
    """Artifact-input readiness against INDEPENDENTLY admitted requirements (owner reference, ruling 2026-10-08e section 4).

    record: a validated repository_records.artifact_inventory.ArtifactInventoryRecord (its results are derived from its locations).
    required: a non-empty tuple of (entry_id, expected_sha256) derived from the sealed installation plan -- never from the record.
    Establishes artifact-input availability ONLY: not runtime, dependency, behavioural or scientific validity. A missing historical
    evidence bundle or unselected alternative is a preservation finding of the record, not a readiness failure, unless required here."""
    require(type(required) is tuple and len(required) > 0, "readiness.empty_or_untyped_requirements")
    ids = []
    for row in required:
        require(type(row) is tuple and len(row) == 2 and type(row[0]) is str and row[0] != "" and type(row[1]) is str
                and _SHA256.fullmatch(row[1]) is not None, "readiness.invalid_requirement")
        ids.append(row[0])
    require(len(ids) == len(set(ids)), "readiness.duplicate_requirement")
    entries = {r.entry_id: r for r in record.requirements}
    results = {r.entry_id: r for r in record.results()}
    require(len(entries) == len(record.requirements) and len(results) == len(entries), "readiness.duplicate_record_entry")
    require(set(entries) == set(results), "readiness.record_coverage")
    rows = []
    for entry_id, wanted in sorted(required):
        entry, result = entries.get(entry_id), results.get(entry_id)
        if entry is None:
            reason = "requirement_not_recorded"
        elif entry.sha256 != wanted:
            reason = "expected_digest_not_plan_digest"
        elif result.state is not EvidenceState.MATCH:
            reason = "content_not_matched:" + result.reason
        else:
            reason = "matched"
        rows.append({"entry_id": entry_id, "ready": reason == "matched", "reason": reason})
    return {"artifact_inputs_ready": all(r["ready"] for r in rows), "requirements": rows}


#: How each implementation digest is computed (ruling 2026-10-08f section 4): the domains are NOT interchangeable.
DIGEST_DOMAINS = {"repository_tree": "git_tree_object_id", "collector_sha256": "exact_bytes_sha256",
                  "record_owner_sha256": "canonical_lf_text_sha256", "admission_sha256": "canonical_lf_text_sha256",
                  "loaded_repository_modules": "canonical_lf_text_sha256"}
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_MODULE_PATH = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*\.py")


def readiness_decision(record, record_bytes: bytes, required, *, plan_documents: dict, evaluated_at: str, implementation: dict) -> dict:
    """The BOUND decision: an unbound "artifact_inputs_ready: true" must never become an installation authorization. Binds the plan
    documents the requirements came from (label -> SHA-256, verified by the caller against the reviewed contract), the record's exact
    bytes (which the record object must render), the evaluation time, and every piece of code that determined the decision:

      implementation = {"repository_tree": <git tree id the CALLER verified as its clean checkout>, "collector_sha256": <exact bytes of
      the collector -- it must be the record's own verifier_sha256>, "loaded_repository_modules": <[{"path", "sha256"}] sorted by
      path: every checkout module the caller loaded, repository-relative, with the canonical LF text digest the caller verified to
      equal its blob in that tree>}

    to which this function adds the record owner's and this module's canonical LF text digests, computed from the modules actually
    running (the record's own class and this file); each must EQUAL the verified digest of exactly one listed module."""
    from genomic_variant_classifier.repository_records.artifact_inventory import ArtifactInventoryRecord   # deferred: that module imports this one
    require(type(record) is ArtifactInventoryRecord, "readiness.record_type")
    require(type(record_bytes) is bytes and record.render() == record_bytes, "readiness.record_bytes_not_this_record")
    require(type(plan_documents) is dict and len(plan_documents) > 0, "readiness.plan_documents_missing")
    for label, digest in plan_documents.items():
        require(type(label) is str and label != "", "readiness.plan_document_label")
        _digest(digest, "readiness.plan_document_digest_invalid")
    require(type(evaluated_at) is str and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", evaluated_at) is not None,
            "readiness.evaluated_at")
    require(type(implementation) is dict and set(implementation) == {"repository_tree", "collector_sha256", "loaded_repository_modules"},
            "readiness.implementation_shape")
    tree, collector, modules = (implementation[k] for k in ("repository_tree", "collector_sha256", "loaded_repository_modules"))
    require(type(tree) is str and _GIT_OBJECT.fullmatch(tree) is not None, "readiness.repository_tree_invalid")
    _digest(collector, "readiness.collector_digest_invalid")
    require(collector == record.verifier_sha256, "readiness.collector_not_the_records_verifier")
    require(type(modules) is list and len(modules) > 0, "readiness.loaded_modules_invalid")
    for m in modules:
        require(type(m) is dict and set(m) == {"path", "sha256"} and type(m["path"]) is str and _MODULE_PATH.fullmatch(m["path"]) is not None,
                "readiness.loaded_modules_invalid")
        _digest(m["sha256"], "readiness.loaded_modules_invalid")
    paths = [m["path"] for m in modules]
    require(paths == sorted(set(paths)), "readiness.loaded_modules_invalid")
    owner_file = Path(sys.modules[ArtifactInventoryRecord.__module__].__file__)
    running = {"record_owner_sha256": canonical_text_sha256(owner_file), "admission_sha256": canonical_text_sha256(Path(__file__))}
    for key, module_file in (("record_owner_sha256", owner_file), ("admission_sha256", Path(__file__))):
        suffix = "/" + "/".join(module_file.parts[-3:])          # genomic_variant_classifier/<subpackage>/<module>.py
        listed = [m for m in modules if ("/" + m["path"]).endswith(suffix)]
        require(len(listed) == 1 and listed[0]["sha256"] == running[key], "readiness.interpreting_module_not_verified")
    result = artifact_readiness(record, required)
    return {"schema": "gvc.artifact-readiness-decision/1", "evaluated_at": evaluated_at,
            "plan_documents": dict(sorted(plan_documents.items())), "record_id": record.record_id.value,
            "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
            "implementation": {"repository_tree": tree, "collector_sha256": collector,
                               "record_owner_sha256": running["record_owner_sha256"], "admission_sha256": running["admission_sha256"],
                               "loaded_repository_modules": [dict(m) for m in modules], "digest_domains": dict(DIGEST_DOMAINS)},
            "artifact_inputs_ready": result["artifact_inputs_ready"], "requirements": result["requirements"],
            "claim": ("HISTORICAL readiness: the selected installation inputs' content was found in the store during this measurement. "
                      "Artifact-input availability only -- not runtime, dependency, behavioural or scientific validity -- and no "
                      "authorization of later use: installation admission must recheck the bytes it is about to consume.")}


def canonical_text_sha256(path) -> str:
    """A text file's identity in the canonical LF domain (CRLF -> LF; a lone CR refused), so a Windows checkout (core.autocrlf) and a
    Linux runner agree. NOT the exact-bytes domain: the two are never compared with each other."""
    raw = Path(path).read_bytes()
    require(b"\r" not in raw.replace(b"\r\n", b""), "readiness.code_lone_cr")
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
