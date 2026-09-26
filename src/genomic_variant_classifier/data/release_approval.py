"""Release-approval validation: an immutable record selected by the manifest.

Created 2026-09-26 (change A, owner rulings of 2026-09-25 and 2026-09-26).

THREE RESPONSIBILITIES, KEPT APART
----------------------------------
- The MANIFEST (`configs/data_manifest.yaml`, `release_approvals:`) SELECTS the
  active approval: exactly a record path and that record's full SHA-256.
- The RECORD (`docs/approvals/*.json`, never edited) holds the approval's facts:
  target, approved release, a narrow scope, dates, evidence, predecessor.
- The VERIFIER'S EXPECTATION (`request_verifier.APPROVED_*`) independently checks
  that the intended approval was selected (`require_verifier_pin`). Two matching
  declarations detect unintended divergence; they do not prove authorization.

Authorization is the owner's deliberate ratification through the protected
pull-request history. A digest proves byte identity, not who approved it.

WHAT AN APPROVAL IS NOT. Scope `release_monitoring_baseline` moves the source
monitor's comparison point. It grants no acquisition, product qualification,
production adoption, transcript policy, model authorization, or approval of any
other release. `approved_release` is validated here only as text: change B owns
the release grammar.

EVIDENCE. Each required evidence entry is a commit-pinned Git blob (the owner's
reference descriptor) with a narrow role. It is verified from Git OBJECTS --
commit/path membership first, then size, SHA-256 and the blob identifier --
never from a working-tree file, whose line endings a Windows checkout may change.
The Git reader disables replacement objects and lazy fetching and REQUIRES Git
2.45 or later (`--no-lazy-fetch`; measured 2026-09-26 in the 2.45.0 release
notes). An older Git refuses; there is no fallback.

Reference kernels integrated here (owner packages, 2026-09-25):
GVC_approval_control_refinements approval_contract.py and
GVC_public_approval_evidence git_evidence.py.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass, fields
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

APPROVAL_SCHEMA_VERSION = 1
APPROVAL_SCOPE = "release_monitoring_baseline"
EVIDENCE_ROLES = frozenset({"historical_approval_record"})
MAX_RECORD_BYTES = 65536
MAX_EVIDENCE_BYTES = 1024 * 1024
_RECORD_PATH = re.compile(r"docs/approvals/[A-Za-z0-9][A-Za-z0-9_.-]*\.json", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", re.ASCII)
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", re.ASCII)
_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)
_RECORD_FIELDS = frozenset({"schema_version", "target", "approved_release", "scope",
                            "recorded_at_utc", "approval_documented_on",
                            "approval_granted_at_utc", "evidence", "supersedes_sha256"})


class PolicyError(ValueError):
    """A policy input cannot support the requested use."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PolicyError(f"{label}: expected a complete lowercase SHA-256")
    return value


def _exact(value: object, expected: frozenset, label: str) -> dict:
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise PolicyError(f"{label}: expected a string-keyed object")
    if set(value) != expected:
        raise PolicyError(f"{label}: missing={sorted(expected - set(value))}; "
                          f"unexpected={sorted(set(value) - expected)}")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise PolicyError(f"{label}: expected nonempty unpadded text")
    return value


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise PolicyError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value: str):
    raise PolicyError(f"non-JSON numeric constant: {value}")


def strict_json(raw: bytes, limit: int = MAX_RECORD_BYTES) -> object:
    """UTF-8 without a byte-order mark; duplicate keys and NaN/Infinity refused."""
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise PolicyError(f"expected 1..{limit} bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PolicyError("a byte-order mark is not permitted")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, PolicyError):
            raise
        raise PolicyError(f"invalid JSON: {exc}") from exc


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise PolicyError(f"{label}: expected UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PolicyError(f"{label}: invalid calendar time") from exc


# --------------------------------------------------------------------------- evidence
@dataclass(frozen=True)
class GitEvidence:
    """The owner's commit-pinned Git-blob descriptor, plus its narrow role."""

    role: str
    kind: str
    repository: str
    commit: str
    path: str
    git_blob_oid: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if self.role not in EVIDENCE_ROLES:
            raise PolicyError(f"evidence role {self.role!r} is not one of {sorted(EVIDENCE_ROLES)}")
        if self.kind != "git_blob":
            raise PolicyError("unsupported evidence kind")
        if type(self.repository) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise PolicyError("evidence repository must be owner/name")
        for label, oid in (("commit", self.commit), ("git_blob_oid", self.git_blob_oid)):
            if type(oid) is not str or _OID.fullmatch(oid) is None:
                raise PolicyError(f"evidence {label}: a full Git object identifier is required")
        if len(self.commit) != len(self.git_blob_oid):
            raise PolicyError("evidence commit and blob object formats differ")
        _digest(self.sha256, "evidence sha256")
        if type(self.bytes) is not int or not 0 < self.bytes <= MAX_EVIDENCE_BYTES:
            raise PolicyError(f"evidence must contain 1..{MAX_EVIDENCE_BYTES} bytes")
        p = self.path
        if (type(p) is not str or not p or p.startswith("/") or "\\" in p or ":" in p
                or any(part in {"", ".", ".."} for part in p.split("/"))
                or any(ord(c) < 32 or ord(c) == 127 for c in p)):
            raise PolicyError("evidence path must be a literal repository-relative path")
        try:
            p.encode("utf-8")
        except UnicodeError as exc:
            raise PolicyError("evidence path must be valid UTF-8") from exc


_EVIDENCE_FIELDS = frozenset(f.name for f in fields(GitEvidence))


def verify_evidence_bytes(ref: GitEvidence, raw: bytes) -> None:
    if type(raw) is not bytes or len(raw) != ref.bytes:
        raise PolicyError("evidence byte count differs")
    if sha256(raw) != ref.sha256:
        raise PolicyError("evidence SHA-256 differs")
    framed = b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    algorithm = hashlib.sha1 if len(ref.git_blob_oid) == 40 else hashlib.sha256
    if algorithm(framed).hexdigest() != ref.git_blob_oid:
        raise PolicyError("evidence Git blob identifier differs")


def git_reader(repo: str | Path) -> Callable[..., bytes]:
    """Binary Git object reader: no shell, filters, textconv, replacement objects or
    lazy fetching. Requires Git 2.45+ for --no-lazy-fetch; an older Git REFUSES."""
    repo = Path(repo)
    if not repo.is_dir():
        raise PolicyError(f"repository directory unavailable: {repo}")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")

    def run(*args: str) -> bytes:
        command = ["git", "--no-replace-objects", "--no-lazy-fetch", "--literal-pathspecs",
                   "-C", str(repo), *args]
        try:
            result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PolicyError(f"Git reader could not complete: {type(exc).__name__}") from exc
        if result.returncode:
            raise PolicyError("Git reader refused (exit {}): {}".format(
                result.returncode, result.stderr.decode("utf-8", errors="replace").strip()[:400]))
        return result.stdout
    return run


def read_blob_at(git: Callable[..., bytes], commit: str, path: str,
                 expected_oid: Optional[str] = None, expected_size: Optional[int] = None,
                 max_size: Optional[int] = None) -> Tuple[str, bytes]:
    """Commit/path MEMBERSHIP first, then the object's SIZE, and only then its bytes.

    The size is checked BEFORE the contents are read (reference verifier contract):
    a wrong or oversized blob is refused without being loaded. Returns (oid, bytes).
    """
    if git("cat-file", "-t", commit).strip() != b"commit":
        raise PolicyError(f"{commit} is not a commit")
    entries = git("ls-tree", "-z", commit, "--", path).split(b"\0")
    if len(entries) != 2 or entries[-1] != b"":
        raise PolicyError(f"{commit}:{path}: expected exactly one tree entry")
    try:
        header, listed = entries[0].split(b"\t", 1)
        mode, kind, oid = header.split(b" ")
    except ValueError as exc:
        raise PolicyError("malformed tree entry") from exc
    if mode not in {b"100644", b"100755"} or kind != b"blob":
        raise PolicyError(f"{path}: must be a regular file, not a symlink or submodule")
    if listed != path.encode("utf-8"):
        raise PolicyError(f"{path}: tree entry names a different path")
    oid_text = oid.decode("ascii")
    if expected_oid is not None and oid_text != expected_oid:
        raise PolicyError("commit/path does not identify the pinned blob")
    try:
        size = int(git("cat-file", "-s", oid_text).strip())
    except ValueError as exc:
        raise PolicyError("invalid Git object size") from exc
    if expected_size is not None and size != expected_size:
        raise PolicyError("Git object size differs")
    if max_size is not None and size > max_size:
        raise PolicyError(f"{path}: {size} bytes exceeds the {max_size}-byte limit")
    return oid_text, git("cat-file", "blob", oid_text)


def verify_git_evidence(ref: GitEvidence, git: Callable[..., bytes]) -> bytes:
    _, raw = read_blob_at(git, ref.commit, ref.path, expected_oid=ref.git_blob_oid,
                          expected_size=ref.bytes)
    verify_evidence_bytes(ref, raw)
    return raw


# --------------------------------------------------------------------------- approval
@dataclass(frozen=True)
class Approval:
    target: str
    approved_release: str
    scope: str
    recorded_at_utc: str
    approval_documented_on: str
    approval_granted_at_utc: Optional[str]
    evidence: Tuple[GitEvidence, ...]
    supersedes_sha256: Optional[str]
    record_sha256: str


def parse_approval(raw: bytes) -> Approval:
    d = _exact(strict_json(raw), _RECORD_FIELDS, "approval")
    if type(d["schema_version"]) is not int or d["schema_version"] != APPROVAL_SCHEMA_VERSION:
        raise PolicyError("unsupported approval schema")
    target = _text(d["target"], "target")
    release = _text(d["approved_release"], "approved_release")  # change B owns the grammar
    if d["scope"] != APPROVAL_SCOPE:
        raise PolicyError("approval scope does not authorize release monitoring")
    recorded = _utc(d["recorded_at_utc"], "recorded_at_utc")
    day = d["approval_documented_on"]
    if type(day) is not str or _DAY.fullmatch(day) is None:
        raise PolicyError("approval_documented_on: expected YYYY-MM-DD")
    try:
        documented = date.fromisoformat(day)
    except ValueError as exc:
        raise PolicyError("approval_documented_on: invalid calendar date") from exc
    if documented > recorded.date():
        raise PolicyError("documentation date is after record creation")
    granted = d["approval_granted_at_utc"]
    if granted is not None:
        grant_time = _utc(granted, "approval_granted_at_utc")
        if grant_time > recorded or grant_time.date() > documented:
            raise PolicyError("grant date contradicts documentation/creation")
    if type(d["evidence"]) is not list or not d["evidence"]:
        raise PolicyError("approval needs nonempty evidence references")
    evidence, seen = [], set()
    for item in d["evidence"]:
        ref = GitEvidence(**_exact(item, _EVIDENCE_FIELDS, "evidence"))
        key = (ref.commit, ref.path)
        if key in seen:
            raise PolicyError(f"duplicate evidence reference: {ref.commit}:{ref.path}")
        seen.add(key)
        evidence.append(ref)
    previous = d["supersedes_sha256"]
    if previous is not None:
        previous = _digest(previous, "supersedes_sha256")
    return Approval(target, release, d["scope"], d["recorded_at_utc"], day, granted,
                    tuple(evidence), previous, sha256(raw))


def load_approval(target: str, record: str, record_sha256: str,
                  read_record: Callable[[str], bytes]) -> Approval:
    """Read ONCE; hash and parse the SAME bytes."""
    if type(record) is not str or _RECORD_PATH.fullmatch(record) is None:
        raise PolicyError("approval record must be a direct docs/approvals/*.json path")
    expected = _digest(record_sha256, "approval pointer")
    try:
        raw = read_record(record)
    except (OSError, KeyError) as exc:
        raise PolicyError(f"approval record unavailable: {record}") from exc
    if type(raw) is not bytes or sha256(raw) != expected:
        raise PolicyError("approval record digest mismatch")
    approval = parse_approval(raw)
    if approval.target != target:
        raise PolicyError("approval target mismatch")
    return approval


def worktree_record_reader(repo_root: str | Path) -> Callable[[str], bytes]:
    """Reads a record from the working tree, refusing symlinks and non-regular files."""
    root = Path(repo_root)

    def read(record: str) -> bytes:
        path = root / record
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode):
            raise PolicyError(f"approval record is not a regular file: {record}")
        return path.read_bytes()
    return read


def require_verifier_pin(approval: Approval, *, target: str, approved_release: str,
                         record_sha256: str) -> None:
    """The verifier's separately reviewed expectation -- checked at RUNTIME."""
    if (approval.target, approval.approved_release, approval.record_sha256) != (
            target, approved_release, _digest(record_sha256, "verifier pin")):
        raise PolicyError("verifier expectation and active approval disagree")


def require_append_only(before: Mapping[str, bytes], after: Mapping[str, bytes]) -> None:
    """Complete directory snapshots at two pinned revisions; detects edit, removal, rename."""
    changed = [p for p, raw in before.items() if after.get(p) != raw]
    if changed:
        raise PolicyError(f"approval records modified or removed: {sorted(changed)}")


def require_successor(previous: Approval, current: Approval) -> None:
    if current.record_sha256 == previous.record_sha256:
        return
    if current.supersedes_sha256 != previous.record_sha256:
        raise PolicyError("new active record does not supersede the old active record")
    if (current.target, current.scope) != (previous.target, previous.scope):
        raise PolicyError("successor changes the approval target or scope")
    if _utc(current.recorded_at_utc, "current") <= _utc(previous.recorded_at_utc, "previous"):
        raise PolicyError("successor must be recorded after its predecessor")


def interpretation_fingerprint(parts: Mapping[str, str]) -> str:
    """Configuration identity only (change B). Execution provenance is checked separately."""
    expected = {"approval", "release_rules", "request_plan", "adapter_code", "verifier_code",
                "environment_lock"}
    if set(parts) != expected:
        raise PolicyError("fingerprint requires every named interpretation dependency")
    validated = {k: _digest(v, k) for k, v in parts.items()}
    encoded = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(b"gvc-monitor-interpretation/v1\0" + encoded)
