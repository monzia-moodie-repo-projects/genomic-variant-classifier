"""The lockfile MIGRATION record: the evidence of one admitted renv.lock authority succession, indexed by one typed owner.

Owner rulings 2026-10-08e / 2026-10-08f: admit the approved transition proposal against the actual baseline and candidate,
preserving the regeneration evidence. ADR-0004: a migration manifest is part of the migration's evidentiary record, so it is a
MIGRATION_RECORD placed by role (records/migrations/...), never documentation about the migration.

AUTHORITY-SUCCESSION-1 (ADR-0004 section D), made checkable:

    old authority identified          `succession.predecessor_canonical_sha256` -- the replaced renv.lock, preserved verbatim
    new authority materialized        `succession.successor_canonical_sha256` -- what renv.lock must now hold (canonical LF)
    succession explicitly declared    this manifest
    behavioural gate green            the admission re-derived from the preserved bytes equals `admission` (the test does this)

The manifest is AUTHORED (deterministic rendering, final newline); the artifacts are PRESERVED (exact bytes, `-text` in
.gitattributes -- the candidate lockfile is CRLF as renv wrote it on Windows, and that is history, not a defect).

THE PARTS are a fixed vocabulary: an archive cannot define its own completeness. Each part is exactly one file,
artifacts/<original basename> -- the basenames are distinct, so a directory per part would add only length. MEASURED 2026-10-10:
with a directory per part the longest artifact path was 169 characters, 260 in the owner's %TEMP% clone, and the installer's apply
failed on Windows; flat, the longest is 148, within the repository path budget (path_budget.py), which every canonical path must
meet (identity.ArtifactInstance enforces it).

This owner validates SHAPE, PLACEMENT, IDENTITY and BYTES. It does not decide admission: environment_qualification.lockfile_admission
does, and the stored `admission` is only ever accepted when that policy, run on these preserved bytes, reproduces it.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .classification import DisclosureClass, PreservationDisposition, ProvenanceRelation, RecordDisposition, RetentionClass
from .identity import ArtifactInstance, RecordId, RecordIdentity
from .roles import ArtifactRole, RecordsOntologyError, canonical_root

logger = logging.getLogger(__name__)

__all__ = ["SCHEMA", "SCHEMA_VERSION", "ROLE", "FAMILY", "PARTS", "SUCCESSION_PATH", "LockfileMigrationError", "PreservedFile",
           "Succession", "LockfileMigrationManifest", "family_root"]

SCHEMA = "gvc.lockfile-migration"
SCHEMA_VERSION = 1
ROLE = ArtifactRole.MIGRATION_RECORD
FAMILY = PurePosixPath("environment-qualification") / "lockfile"
SUCCESSION_PATH = "renv.lock"

#: part -> the original basename it is preserved under (basename retention: ADR-0004 Consequences).
PARTS = {
    "baseline_lockfile": "renv.lock",
    "candidate_lockfile": "candidate.lock",
    "approved_proposal": "lock_transition_proposal.json",
    "regenerated_proposal": "lock_transition_proposal_regenerated_bound.json",
    "equivalence_record": "lock_transition_equivalence.json",
    "replay_plan": "replay_plan.json",
    "candidate_plan": "candidate_plan.json",
    "candidate_difference": "candidate_difference.json",
}
if len(set(PARTS.values())) != len(PARTS):      # the flat layout is only unambiguous while every basename is distinct
    raise RecordsOntologyError("lockfile migration parts share a basename: {}".format(sorted(PARTS.values())))

_SHA256 = re.compile(r"[0-9a-f]{64}")
_AS_OF = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")


class LockfileMigrationError(RecordsOntologyError):
    """The lockfile migration record does not satisfy its own contract."""


def _require(condition, message):
    if not condition:
        raise LockfileMigrationError(message)


def family_root() -> PurePosixPath:
    return canonical_root(ROLE) / FAMILY


def _strict_load(raw: bytes):
    """Duplicate keys, non-finite constants and a byte-order mark are refused. Integers stay integers; floats are refused (the
    admission section is digests, strings, booleans and integers only)."""
    _require(type(raw) is bytes and raw != b"", "manifest: empty or not bytes")
    _require(not raw.startswith(b"\xef\xbb\xbf"), "manifest: byte-order mark")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "manifest: duplicate key {!r}".format(key))
            result[key] = value
        return result

    def refuse(token):
        raise LockfileMigrationError("manifest: non-integer number {!r}".format(token))

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=refuse, parse_constant=refuse)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockfileMigrationError("manifest: invalid JSON: {}".format(exc)) from None


def _keys(doc, required, what):
    _require(type(doc) is dict, "{}: must be an object".format(what))
    missing, unknown = sorted(set(required) - set(doc)), sorted(set(doc) - set(required))
    _require(not missing, "{}: missing {}".format(what, missing))
    _require(not unknown, "{}: undeclared key(s) {}".format(what, unknown))


def _text(value, what):
    _require(type(value) is str and value.strip() != "" and value == value.strip(), "{}: non-empty trimmed text".format(what))


@dataclass(frozen=True)
class PreservedFile:
    """One preserved byte sequence: its part, its placement and identity, and its four-axis disposition."""

    part: str
    identity: RecordIdentity
    disposition: RecordDisposition

    def __post_init__(self) -> None:
        _require(self.part in PARTS, "file: {!r} is not a migration part".format(self.part))
        _require(type(self.identity) is RecordIdentity and type(self.disposition) is RecordDisposition, "file: types")
        _require(self.disposition.role is self.identity.role is ROLE, "file: role must be {}".format(ROLE.value))
        _require(type(self.identity.instance.size_bytes) is int, "file: size_bytes must be an int")
        _require(self.disposition.is_publicly_preservable, "file {}: not publicly preservable".format(self.part))
        _require(self.disposition.retention is RetentionClass.PERMANENT_EVIDENCE, "file {}: retention".format(self.part))

    def as_record(self) -> dict:
        d = self.disposition
        record = {"part": self.part, "canonical_path": self.identity.instance.canonical_path,
                  "content_sha256": self.identity.instance.content_sha256, "size_bytes": self.identity.instance.size_bytes,
                  "disclosure": d.disclosure.value, "preservation": d.preservation.value,
                  "provenance": sorted(p.value for p in d.provenance), "retention": d.retention.value}
        if d.defect_note.strip():
            record["defect_note"] = d.defect_note
        return record


@dataclass(frozen=True)
class Succession:
    """The authority cutover: which tracked path, what it held before (canonical LF), what it must hold after."""

    path: str
    predecessor_canonical_sha256: str
    successor_canonical_sha256: str

    def __post_init__(self) -> None:
        _require(self.path == SUCCESSION_PATH, "succession: path must be {!r}".format(SUCCESSION_PATH))
        for what, value in (("predecessor", self.predecessor_canonical_sha256), ("successor", self.successor_canonical_sha256)):
            _require(type(value) is str and _SHA256.fullmatch(value) is not None, "succession: {} digest".format(what))
        _require(self.predecessor_canonical_sha256 != self.successor_canonical_sha256, "succession: successor equals predecessor")

    def as_record(self) -> dict:
        return {"path": self.path, "predecessor_canonical_sha256": self.predecessor_canonical_sha256,
                "successor_canonical_sha256": self.successor_canonical_sha256}


def _json_value_ok(value) -> bool:
    """The admission section: objects, lists, strings, booleans and integers only (no float, no null except declared)."""
    if type(value) in (str, bool, int):
        return True
    if type(value) is list:
        return all(_json_value_ok(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str for k in value) and all(_json_value_ok(v) for v in value.values())
    return False


@dataclass(frozen=True)
class LockfileMigrationManifest:
    """The index of one lockfile migration. Construction IS validation."""

    record_id: RecordId
    as_of: str
    approval: str
    files: tuple
    succession: Succession
    admission: dict

    @property
    def root(self) -> PurePosixPath:
        return family_root() / self.record_id.value

    def __post_init__(self) -> None:
        _require(type(self.record_id) is RecordId, "record_id: a RecordId")
        _require(type(self.as_of) is str and _AS_OF.fullmatch(self.as_of) is not None, "as_of: YYYY-MM-DDTHH:MM:SSZ")
        _text(self.approval, "approval")
        _require(type(self.files) is tuple and all(type(f) is PreservedFile for f in self.files), "files: a tuple of PreservedFile")
        _require(type(self.succession) is Succession, "succession: a Succession")
        # Canonical ORDER at construction, so that parse(render(m)) == m (measured for the inventory owner: without it the parsed
        # object differs from the constructed one although both render the same bytes).
        object.__setattr__(self, "files", tuple(sorted(self.files, key=lambda f: f.part)))
        parts = [f.part for f in self.files]
        _require(sorted(parts) == sorted(PARTS), "files: parts must be exactly {} (got {})".format(sorted(PARTS), sorted(parts)))
        for f in self.files:
            _require(f.identity.record_id == self.record_id, "file {}: belongs to another record".format(f.part))
            expected = self.root / "artifacts" / PARTS[f.part]
            _require(PurePosixPath(f.identity.instance.canonical_path) == expected, "file {}: must be {}".format(f.part, expected))
        _require(type(self.admission) is dict and len(self.admission) > 0 and _json_value_ok(self.admission),
                 "admission: a non-empty object of strings, booleans, integers, lists and objects")
        _require(self.admission.get("schema") == "gvc.lockfile-migration-admission/1", "admission: schema")
        by_part = {f.part: f for f in self.files}
        baseline, candidate = self.admission.get("baseline"), self.admission.get("candidate")
        _require(type(baseline) is dict and type(candidate) is dict, "admission: baseline / candidate")
        # the succession is the admitted transition, and the preserved lockfiles are the admitted bytes
        _require(self.succession.predecessor_canonical_sha256 == baseline.get("canonical_sha256"), "succession: predecessor is not the admitted baseline")
        _require(self.succession.successor_canonical_sha256 == candidate.get("canonical_sha256"), "succession: successor is not the admitted candidate")
        _require(by_part["baseline_lockfile"].identity.instance.content_sha256 == baseline.get("exact_sha256"),
                 "baseline_lockfile: not the admitted baseline bytes")
        _require(by_part["candidate_lockfile"].identity.instance.content_sha256 == candidate.get("exact_sha256"),
                 "candidate_lockfile: not the admitted candidate bytes")
        _require(by_part["approved_proposal"].identity.instance.content_sha256 == self.admission.get("approved_proposal_sha256"),
                 "approved_proposal: not the admitted proposal")

    def payload(self) -> dict:
        return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "record_id": self.record_id.value, "role": ROLE.value,
                "as_of": self.as_of, "approval": self.approval,
                "files": sorted((f.as_record() for f in self.files), key=lambda r: r["part"]),
                "succession": self.succession.as_record(), "admission": self.admission}

    def render(self) -> bytes:
        """Deterministic and diffable. AUTHORED, so it ends with a newline."""
        return (json.dumps(self.payload(), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")

    @classmethod
    def parse(cls, raw: bytes) -> "LockfileMigrationManifest":
        doc = _strict_load(raw)
        _keys(doc, ("schema", "schema_version", "record_id", "role", "as_of", "approval", "files", "succession", "admission"), "manifest")
        _require(doc["schema"] == SCHEMA, "schema is {!r}".format(doc["schema"]))
        _require(type(doc["schema_version"]) is int and doc["schema_version"] == SCHEMA_VERSION, "schema_version")
        _require(doc["role"] == ROLE.value, "role")
        _require(type(doc["files"]) is list, "files: must be a list")
        record_id = RecordId(doc["record_id"])
        files = []
        for f in doc["files"]:
            optional = ("defect_note",) if type(f) is dict and "defect_note" in f else ()
            _keys(f, ("part", "canonical_path", "content_sha256", "size_bytes", "disclosure", "preservation", "provenance",
                      "retention") + optional, "file")
            _require(type(f["provenance"]) is list, "file: provenance must be a list")
            try:      # ONLY the vocabulary lookups: a disposition that violates its own rules must keep its own message
                terms = (DisclosureClass(f["disclosure"]), PreservationDisposition(f["preservation"]),
                         tuple(ProvenanceRelation(p) for p in f["provenance"]), RetentionClass(f["retention"]))
            except ValueError as exc:
                raise LockfileMigrationError("file: unrecognised vocabulary term: {}".format(exc)) from None
            disposition = RecordDisposition(role=ROLE, disclosure=terms[0], preservation=terms[1], provenance=terms[2],
                                            retention=terms[3], defect_note=f.get("defect_note", ""))
            _require(type(f["size_bytes"]) is int, "file: size_bytes must be an int")
            identity = RecordIdentity(record_id=record_id, instance=ArtifactInstance(
                content_sha256=f["content_sha256"], canonical_path=f["canonical_path"], size_bytes=f["size_bytes"]), role=ROLE)
            files.append(PreservedFile(f["part"], identity, disposition))
        s = doc["succession"]
        _keys(s, ("path", "predecessor_canonical_sha256", "successor_canonical_sha256"), "succession")
        manifest = cls(record_id, doc["as_of"], doc["approval"], tuple(files),
                       Succession(s["path"], s["predecessor_canonical_sha256"], s["successor_canonical_sha256"]), doc["admission"])
        _require(manifest.render() == raw, "manifest: not in its deterministic rendering (round-trip differs)")
        return manifest

    def read_preserved(self, repository_root) -> dict:
        """STAGE 1 -- exact inventory of artifacts/ and exact bytes. Returns part -> bytes for the admission policy to re-run on."""
        root = Path(repository_root)
        artifacts = root.joinpath(*(self.root / "artifacts").parts)
        _require(artifacts.is_dir() and not artifacts.is_symlink(), "{}: missing".format(artifacts))
        actual = set()
        for p in artifacts.iterdir():       # FLAT: artifacts/ holds the preserved files and nothing else, not even a directory
            _require(not p.is_symlink(), "{}: a symbolic link".format(p))
            _require(p.is_file(), "{}: not a regular file (artifacts/ is flat)".format(p))
            actual.add(p.relative_to(root).as_posix())
        declared = {f.identity.instance.canonical_path for f in self.files}
        _require(actual == declared, "inventory: on disk but not indexed {}; indexed but not on disk {}".format(
            sorted(actual - declared), sorted(declared - actual)))
        out = {}
        for f in self.files:
            raw = root.joinpath(*PurePosixPath(f.identity.instance.canonical_path).parts).read_bytes()
            _require(len(raw) == f.identity.instance.size_bytes, "{}: size differs".format(f.part))
            _require(hashlib.sha256(raw).hexdigest() == f.identity.instance.content_sha256, "{}: digest differs".format(f.part))
            out[f.part] = raw
        return out
