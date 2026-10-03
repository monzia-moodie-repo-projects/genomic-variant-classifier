"""The C2 source-monitor qualification record: exact evidence, indexed by one typed owner.

Owner rulings 2026-10-02 / 2026-10-02b. A VERIFICATION_RESULT record -- placed by role, never by a chosen
directory -- holding the exact bytes of the isolated live qualification and the typed manifest that indexes them.
It is NEVER a live posting authority: preserving historical evidence does not create a second delivery ledger.

COMPOSITION, NOT DUPLICATION
============================
Every file is placed by `RecordIdentity` (containment beneath the role's canonical root) and judged on the four
orthogonal axes by `RecordDisposition` (a defect note exactly when a problem is recorded; restricted bytes never
admitted verbatim). This module adds only what the qualification needs: rounds, the acceptance cases, evidence
gaps, acquisition failures, and the process-deviation statement.

THREE SEPARATE CLAIMS (owner ruling)
====================================
    "these bytes have not changed"                  -> verify(): size + SHA-256 against this manifest (stage 1)
    "these bytes came from this run and attempt"    -> recorded provenance and GitHub's own records inside the evidence
    "these bytes support this qualification result" -> the project adapter's semantic replay (stage 2, in tests)
A checksum supplies only the first.

EXACTNESS
=========
Integer fields are `type(x) is int` -- archive_guard MEASURED 2026-09-08 that the installation-attestation owner
accepts True and 1.0 for integers and renders them back. Parsing is STRICT: duplicate keys, floats and constants are
refused (plain json.loads silently keeps the last duplicate). The manifest is AUTHORED (deterministic, final
newline); the artifacts are PRESERVED (exact bytes, no policy applied).

REQUIRED CASES come from the acceptance CONTRACT, never from the manifest being checked: an incomplete archive must
not define its own passing test.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .classification import (
    DisclosureClass,
    PreservationDisposition,
    ProvenanceRelation,
    RecordDisposition,
    RetentionClass,
)
from .identity import ArtifactInstance, RecordId, RecordIdentity
from .roles import ArtifactRole, RecordsOntologyError, canonical_root

SCHEMA = "gvc.source-monitor-c2-qualification"
SCHEMA_VERSION = 1
ROLE = ArtifactRole.VERIFICATION_RESULT
SUBDIRECTORY = "source-monitor-c2"

#: The acceptance contract (owner ruling 2026-10-02): the five live exercises every qualification round must contain.
REQUIRED_CASES = frozenset({"normal", "claims-disagree", "acquisition-limit", "duplicate-suppression", "manual-preview"})

#: The owner's process statement, retained EXACTLY (ruling 2026-09-30, reaffirmed 2026-10-02).
DEVIATION_STATEMENT = ("Production cutover preceded isolated live qualification. This is a process deviation. "
                       "Subsequent isolated qualification supplies compensating functional evidence.")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TREE = re.compile(r"[0-9a-f]{40}")
_ROUND = re.compile(r"round-[1-9][0-9]*")
_AS_OF = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


class QualificationManifestError(RecordsOntologyError):
    """The qualification record does not satisfy its own contract."""


def _require(condition, message):
    if not condition:
        raise QualificationManifestError(message)


def _text(value, what):
    _require(type(value) is str and value.strip() != "" and value == value.strip(), "{}: non-empty trimmed text".format(what))
    return value


def strict_load(raw: bytes):
    """Duplicate keys, floats, NaN/Infinity and a byte-order mark are REFUSED -- an auditable manifest has one meaning."""
    _require(type(raw) is bytes and raw != b"", "manifest: empty or not bytes")
    _require(not raw.startswith(b"\xef\xbb\xbf"), "manifest: byte-order mark")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "manifest: duplicate key {!r}".format(key))
            result[key] = value
        return result

    def refuse(token):
        raise QualificationManifestError("manifest: non-integer number {!r}".format(token))

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=refuse, parse_constant=refuse)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationManifestError("manifest: invalid JSON: {}".format(exc)) from None


def _keys(doc, required, what, optional=()):
    _require(type(doc) is dict, "{}: must be an object".format(what))
    missing, unknown = sorted(set(required) - set(doc)), sorted(set(doc) - set(required) - set(optional))
    _require(not missing, "{}: missing {}".format(what, missing))
    _require(not unknown, "{}: undeclared key(s) {}".format(what, unknown))


@dataclass(frozen=True)
class EvidenceFile:
    """One preserved byte sequence, placed by role and judged on the four axes."""

    identity: RecordIdentity
    round_id: str
    disposition: RecordDisposition

    def __post_init__(self) -> None:
        _require(type(self.round_id) is str and _ROUND.fullmatch(self.round_id), "file: round_id")
        _require(self.disposition.role is self.identity.role is ROLE, "file: role must be {}".format(ROLE.value))
        _require(type(self.identity.instance.size_bytes) is int, "file: size_bytes must be an int, not {}".format(
            type(self.identity.instance.size_bytes).__name__))

    def as_record(self) -> dict:
        d = self.disposition
        record = {"canonical_path": self.identity.instance.canonical_path, "content_sha256": self.identity.instance.content_sha256,
                  "size_bytes": self.identity.instance.size_bytes, "round": self.round_id, "disclosure": d.disclosure.value,
                  "preservation": d.preservation.value, "provenance": sorted(p.value for p in d.provenance),
                  "retention": d.retention.value}
        if d.defect_note.strip():
            record["defect_note"] = d.defect_note
        return record


@dataclass(frozen=True)
class DefectNote:
    """An ACQUISITION FAILURE: no evidence bytes exist (ArtifactInstance refuses size 0), so it is recorded as a note."""

    round_id: str
    name: str
    size_bytes: int
    content_sha256: str
    note: str

    def __post_init__(self) -> None:
        _require(type(self.round_id) is str and _ROUND.fullmatch(self.round_id), "defect: round_id")
        _require(type(self.name) is str and _NAME.fullmatch(self.name), "defect: name")
        _require(type(self.size_bytes) is int and self.size_bytes >= 0, "defect: size_bytes")
        _require(type(self.content_sha256) is str and _SHA256.fullmatch(self.content_sha256), "defect: content_sha256")
        _text(self.note, "defect: note")

    def as_record(self) -> dict:
        return {"round": self.round_id, "name": self.name, "size_bytes": self.size_bytes,
                "content_sha256": self.content_sha256, "note": self.note}


@dataclass(frozen=True)
class Case:
    """One acceptance exercise in one round, with the evidence that supports it and its recorded result."""

    round_id: str
    name: str
    evidence: tuple
    result: str

    def __post_init__(self) -> None:
        _require(type(self.round_id) is str and _ROUND.fullmatch(self.round_id), "case: round_id")
        _require(self.name in REQUIRED_CASES, "case: {!r} is not an acceptance case".format(self.name))
        _require(type(self.evidence) is tuple and self.evidence and all(type(p) is str for p in self.evidence), "case: evidence")
        _require(len(set(self.evidence)) == len(self.evidence), "case: duplicate evidence")
        _text(self.result, "case: result")

    def as_record(self) -> dict:
        return {"round": self.round_id, "name": self.name, "evidence": sorted(self.evidence), "result": self.result}


@dataclass(frozen=True)
class Round:
    """One qualification round: the runtime it qualified, the checker identities INDEPENDENTLY COMPUTED from its trees at
    preservation, and its explicit evidence gaps. Stage 2 binds archived receipts to these recorded identities -- never to
    an identity re-derived from today's tree (a future legitimate checker change would break the historical record for a
    reason unrelated to it) and never to the receipt's own block (self-binding, the defect C2 repairs 1 removed)."""

    round_id: str
    description: str
    production_trees: tuple
    qualification_trees: tuple
    checker_identities: tuple
    limitations: tuple

    def __post_init__(self) -> None:
        _require(type(self.round_id) is str and _ROUND.fullmatch(self.round_id), "round: round_id")
        _text(self.description, "round: description")
        for what, trees in (("production_trees", self.production_trees), ("qualification_trees", self.qualification_trees)):
            _require(type(trees) is tuple and trees and all(type(t) is str and _TREE.fullmatch(t) for t in trees), "round: " + what)
            _require(len(set(trees)) == len(trees), "round: duplicate " + what)
        _require(type(self.checker_identities) is tuple and self.checker_identities and all(
            type(pair) is tuple and len(pair) == 2 and all(type(x) is str and _SHA256.fullmatch(x) for x in pair)
            for pair in self.checker_identities), "round: checker_identities -- (code_manifest_sha256, policy_sha256) pairs")
        _require(len(set(self.checker_identities)) == len(self.checker_identities), "round: duplicate checker identity")
        _require(type(self.limitations) is tuple and all(type(x) is str and x.strip() for x in self.limitations), "round: limitations")

    def as_record(self) -> dict:
        return {"round": self.round_id, "description": self.description, "production_trees": list(self.production_trees),
                "qualification_trees": list(self.qualification_trees), "limitations": list(self.limitations),
                "checker_identities": [{"code_manifest_sha256": c, "policy_sha256": p} for c, p in sorted(self.checker_identities)]}


@dataclass(frozen=True)
class QualificationManifest:
    """The index. Construction IS validation."""

    record_id: RecordId
    as_of: str
    disclosure_basis: str
    deviation: str
    rounds: tuple
    files: tuple
    cases: tuple
    defects: tuple

    @property
    def root(self) -> PurePosixPath:
        return canonical_root(ROLE) / SUBDIRECTORY / self.record_id.value

    def __post_init__(self) -> None:
        _require(type(self.record_id) is RecordId, "record_id: a RecordId")
        _require(type(self.as_of) is str and _AS_OF.fullmatch(self.as_of), "as_of: YYYY-MM-DDTHH:MM:SSZ")
        _text(self.disclosure_basis, "disclosure_basis")
        _require(self.deviation == DEVIATION_STATEMENT, "deviation: must be the owner's statement EXACTLY")
        for what, items, kind in (("rounds", self.rounds, Round), ("files", self.files, EvidenceFile),
                                  ("cases", self.cases, Case), ("defects", self.defects, DefectNote)):
            _require(type(items) is tuple and all(type(x) is kind for x in items), "{}: a tuple of {}".format(what, kind.__name__))
        _require(self.rounds and self.files and self.cases, "rounds, files and cases must be non-empty")
        round_ids = [r.round_id for r in self.rounds]
        _require(len(set(round_ids)) == len(round_ids), "rounds: duplicate round id")
        artifacts = self.root / "artifacts"
        paths = [f.identity.instance.canonical_path for f in self.files]
        _require(len(set(paths)) == len(paths), "files: two entries claim the same path")
        for f in self.files:
            rel = PurePosixPath(f.identity.instance.canonical_path)
            _require(rel.parent == artifacts / f.round_id, "{}: must lie directly in {}".format(rel, artifacts / f.round_id))
            _require(f.round_id in round_ids, "{}: unknown round".format(rel))
        by_path = {f.identity.instance.canonical_path: f for f in self.files}
        for c in self.cases:
            _require(c.round_id in round_ids, "case {}/{}: unknown round".format(c.round_id, c.name))
            for p in c.evidence:
                _require(p in by_path and by_path[p].round_id == c.round_id,
                         "case {}/{}: evidence {} is not a file of that round".format(c.round_id, c.name, p))
        pairs = [(c.round_id, c.name) for c in self.cases]
        _require(len(set(pairs)) == len(pairs), "cases: duplicate (round, case)")
        for r in round_ids:
            _require({n for (rr, n) in pairs if rr == r} == REQUIRED_CASES,
                     "{}: cases must be exactly the contract's {}".format(r, sorted(REQUIRED_CASES)))
        for d in self.defects:
            _require(d.round_id in round_ids, "defect {}: unknown round".format(d.name))

    def payload(self) -> dict:
        return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "record_id": self.record_id.value, "role": ROLE.value,
                "as_of": self.as_of, "disclosure_basis": self.disclosure_basis, "deviation": self.deviation,
                "rounds": [r.as_record() for r in sorted(self.rounds, key=lambda r: r.round_id)],
                "files": sorted((f.as_record() for f in self.files), key=lambda r: r["canonical_path"]),
                "cases": sorted((c.as_record() for c in self.cases), key=lambda r: (r["round"], r["name"])),
                "defects": sorted((d.as_record() for d in self.defects), key=lambda r: (r["round"], r["name"]))}

    def render(self) -> bytes:
        """Deterministic and diffable. AUTHORED, so it ends with a newline."""
        return (json.dumps(self.payload(), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")

    @classmethod
    def parse(cls, raw: bytes) -> "QualificationManifest":
        doc = strict_load(raw)
        _keys(doc, ("schema", "schema_version", "record_id", "role", "as_of", "disclosure_basis", "deviation", "rounds",
                    "files", "cases", "defects"), "manifest")
        _require(doc["schema"] == SCHEMA, "schema is {!r}".format(doc["schema"]))
        _require(type(doc["schema_version"]) is int and doc["schema_version"] == SCHEMA_VERSION, "schema_version")
        _require(doc["role"] == ROLE.value, "role")
        for key in ("rounds", "files", "cases", "defects"):
            _require(type(doc[key]) is list, "{}: must be a list".format(key))
        record_id = RecordId(doc["record_id"])
        rounds = []
        for r in doc["rounds"]:
            _keys(r, ("round", "description", "production_trees", "qualification_trees", "checker_identities", "limitations"), "round")
            for key in ("production_trees", "qualification_trees", "checker_identities", "limitations"):
                _require(type(r[key]) is list, "round: {} must be a list".format(key))
            pairs = []
            for i in r["checker_identities"]:
                _keys(i, ("code_manifest_sha256", "policy_sha256"), "checker identity")
                pairs.append((i["code_manifest_sha256"], i["policy_sha256"]))
            rounds.append(Round(r["round"], r["description"], tuple(r["production_trees"]), tuple(r["qualification_trees"]),
                                tuple(pairs), tuple(r["limitations"])))
        files = []
        for f in doc["files"]:
            _keys(f, ("canonical_path", "content_sha256", "size_bytes", "round", "disclosure", "preservation", "provenance",
                      "retention"), "file", optional=("defect_note",))
            _require(type(f["provenance"]) is list, "file: provenance must be a list")
            try:
                disposition = RecordDisposition(
                    role=ROLE, disclosure=DisclosureClass(f["disclosure"]), preservation=PreservationDisposition(f["preservation"]),
                    provenance=tuple(ProvenanceRelation(p) for p in f["provenance"]), retention=RetentionClass(f["retention"]),
                    defect_note=f.get("defect_note", ""))
            except ValueError as exc:
                raise QualificationManifestError("file: unrecognised vocabulary term: {}".format(exc)) from None
            identity = RecordIdentity(record_id=record_id, instance=ArtifactInstance(
                content_sha256=f["content_sha256"], canonical_path=f["canonical_path"], size_bytes=f["size_bytes"]), role=ROLE)
            files.append(EvidenceFile(identity, f["round"], disposition))
        cases = []
        for c in doc["cases"]:
            _keys(c, ("round", "name", "evidence", "result"), "case")
            _require(type(c["evidence"]) is list, "case: evidence must be a list")
            cases.append(Case(c["round"], c["name"], tuple(c["evidence"]), c["result"]))
        defects = []
        for d in doc["defects"]:
            _keys(d, ("round", "name", "size_bytes", "content_sha256", "note"), "defect")
            defects.append(DefectNote(d["round"], d["name"], d["size_bytes"], d["content_sha256"], d["note"]))
        manifest = cls(record_id, doc["as_of"], doc["disclosure_basis"], doc["deviation"], tuple(rounds), tuple(files),
                       tuple(cases), tuple(defects))
        _require(manifest.render() == raw, "manifest: not in its deterministic rendering (round-trip differs)")
        return manifest

    def verify(self, repository_root: Path, *, required_cases: frozenset) -> int:
        """STAGE 1 -- inventory and exact-byte integrity. `required_cases` comes from the acceptance contract and must
        equal REQUIRED_CASES; a vacuous or self-defined set is refused. Returns the number of files verified."""
        _require(type(required_cases) is frozenset and required_cases == REQUIRED_CASES, "contract: required_cases")
        artifacts = Path(repository_root).joinpath(*(self.root / "artifacts").parts)
        _require(artifacts.is_dir() and not artifacts.is_symlink(), "{}: missing".format(artifacts))
        actual = set()
        for p in artifacts.rglob("*"):
            _require(not p.is_symlink(), "{}: a symbolic link".format(p))
            if p.is_file():
                actual.add(p.relative_to(repository_root).as_posix())
            else:
                _require(p.is_dir(), "{}: not a regular file or directory".format(p))
        declared = {f.identity.instance.canonical_path for f in self.files}
        _require(actual == declared, "inventory: on disk but not indexed {}; indexed but not on disk {}".format(
            sorted(actual - declared)[:5], sorted(declared - actual)[:5]))
        for f in self.files:
            p = Path(repository_root).joinpath(*PurePosixPath(f.identity.instance.canonical_path).parts)
            _require(p.stat().st_size == f.identity.instance.size_bytes, "{}: size differs".format(p.name))
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            _require(h.hexdigest() == f.identity.instance.content_sha256, "{}: digest differs".format(p.name))
        return len(self.files)
