"""The index over preserved historical evidence, typed rather than hand-built.

ADR-0004 section G. Created 2026-08-23.

WHY THIS IS TYPED
=================
ATTESTATION-SCHEMA-DRIFT-1 happened because nine documents were hand-built as
dictionaries under one unchanging version, and the drift was only detectable
afterwards. **A preservation manifest is itself a durable record.** Introducing
`gvc.installation-attestation-archive` as an unvalidated dictionary would
reproduce that defect one level up, inside the very unit preserving the evidence
of having fixed it.

So one typed object owns construction, serialization, validation and version:

    typed construction -> semantic validation -> deterministic serialization
                       -> schema validation   -> round-trip validation

WHAT A MANIFEST IS FOR
======================
Sixteen install attestations live outside version control while sixteen commit
messages cite them by name. Retaining a historical basename does NOT make a
citation resolve -- git does not turn a filename in a commit message into a
locator. Resolution is a REPOSITORY property, established by recorded aliases
and proven by a test. That is what this index is: the mapping from what history
said to where the bytes now are.

SERIALIZATION
=============
`json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)` plus a
terminating newline. Sorted keys and fixed indentation are fully deterministic;
a compact separator form is no more deterministic and turns a record reviewers
must audit into a single unreadable line. **Determinism does not require
unreadability.**

Note the asymmetry, which is the point of ADR-0004's three policies: this
manifest is AUTHORED and therefore ends with a newline, while the artifacts it
indexes are PRESERVED and -- measured 2026-08-23, sixteen of sixteen -- do not.
Applying one policy to both would refuse every file the archive exists to hold.

GENESIS CARDINALITY
===================
An archive that is empty is internally consistent: manifest entries equal disk
artifacts equal aliases equal the empty set, and every set equality passes. The
manifest therefore carries the count it was born with, and the contract is

    len(entries) >= genesis_cardinality       the archive may grow, never shrink
    every genesis alias still present         no original may be dropped

rather than `== 16` in a test, which would make the archive unable to grow
without editing the assertion -- and an assertion edited whenever it fails is
not an assertion.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .classification import (
    DisclosureClass,
    PreservationDisposition,
    ProvenanceRelation,
    RetentionClass,
)
from .identity import ArtifactInstance, RecordId, RecordIdentity
from .roles import ArtifactRole, RecordsOntologyError

SCHEMA = "gvc.installation-attestation-archive"
SCHEMA_VERSION = 1

_ENTRY_KEYS = frozenset({
    "record_id", "content_sha256", "canonical_path", "size_bytes",
    "legacy_aliases", "cited_by", "role", "disclosure", "preservation",
    "provenance", "retention", "artifact_schema_version",
})
_OPTIONAL_ENTRY_KEYS = frozenset({"defect_note"})
_MANIFEST_KEYS = frozenset({
    "schema", "schema_version", "artifact_class", "genesis_cardinality",
    "genesis_aliases", "entries",
})


class ArchiveManifestError(RecordsOntologyError):
    """A manifest does not satisfy its own contract."""


@dataclass(frozen=True)
class ArchiveEntry:
    """One preserved artifact, with everything needed to resolve a citation.

    `cited_by` holds the short commits whose messages name this artifact. It is
    a COMMITTED fact, not a walk of git history: a test deriving citations from
    `git log --all` would be silently vacuous in a shallow checkout --
    `actions/checkout` defaults to `fetch-depth: 1`, measured 2026-08-22 across
    ten invocations in four workflows, none declaring a depth. Such a walk sees
    one commit, asserts one citation resolves, and passes.
    """

    identity: RecordIdentity
    cited_by: tuple
    disclosure: DisclosureClass
    preservation: PreservationDisposition
    provenance: tuple
    retention: RetentionClass
    artifact_schema_version: int
    defect_note: str = ""

    def __post_init__(self) -> None:
        if not self.provenance:
            raise ArchiveManifestError(
                "{}: an entry with no provenance cannot be audited".format(
                    self.identity.basename))
        if len(set(self.provenance)) != len(self.provenance):
            raise ArchiveManifestError(
                "{}: duplicate provenance".format(self.identity.basename))
        if len(set(self.cited_by)) != len(self.cited_by):
            raise ArchiveManifestError(
                "{}: duplicate citing commit".format(self.identity.basename))
        if self.artifact_schema_version < 1:
            raise ArchiveManifestError(
                "{}: artifact_schema_version must be positive".format(
                    self.identity.basename))
        needs_note = self.preservation in (
            PreservationDisposition.ADMITTED_WITH_DEFECT_NOTE,
            PreservationDisposition.QUARANTINED,
            PreservationDisposition.REJECTED)
        if needs_note and not self.defect_note.strip():
            raise ArchiveManifestError(
                "{}: {} requires a defect_note".format(
                    self.identity.basename, self.preservation.value))
        if not needs_note and self.defect_note.strip():
            raise ArchiveManifestError(
                "{}: a clean disposition may not carry a defect_note".format(
                    self.identity.basename))
        if (self.disclosure is DisclosureClass.RESTRICTED_VERBATIM
                and self.preservation is
                PreservationDisposition.ADMITTED_VERBATIM):
            raise ArchiveManifestError(
                "{}: RESTRICTED_VERBATIM may not be ADMITTED_VERBATIM into a "
                "public repository".format(self.identity.basename))

    def as_record(self) -> dict:
        record = {
            "record_id": self.identity.record_id.value,
            "content_sha256": self.identity.instance.content_sha256,
            "canonical_path": self.identity.instance.canonical_path,
            "size_bytes": self.identity.instance.size_bytes,
            "legacy_aliases": sorted(self.identity.legacy_aliases),
            "cited_by": sorted(self.cited_by),
            "role": self.identity.role.value,
            "disclosure": self.disclosure.value,
            "preservation": self.preservation.value,
            "provenance": sorted(p.value for p in self.provenance),
            "retention": self.retention.value,
            "artifact_schema_version": self.artifact_schema_version,
        }
        if self.defect_note.strip():
            record["defect_note"] = self.defect_note
        return record

    @classmethod
    def from_record(cls, record) -> "ArchiveEntry":
        if not isinstance(record, dict):
            raise ArchiveManifestError("an entry must be an object")
        keys = set(record)
        missing = sorted(_ENTRY_KEYS - keys)
        unknown = sorted(keys - _ENTRY_KEYS - _OPTIONAL_ENTRY_KEYS)
        if missing:
            raise ArchiveManifestError("entry missing {}".format(missing))
        if unknown:
            raise ArchiveManifestError(
                "entry has undeclared key(s) {}. An undeclared field is how "
                "ATTESTATION-SCHEMA-DRIFT-1 happened.".format(unknown))
        try:
            role = ArtifactRole(record["role"])
            disclosure = DisclosureClass(record["disclosure"])
            preservation = PreservationDisposition(record["preservation"])
            retention = RetentionClass(record["retention"])
            provenance = tuple(ProvenanceRelation(p)
                               for p in record["provenance"])
        except ValueError as exc:
            raise ArchiveManifestError(
                "unrecognised vocabulary term: {}".format(exc)) from None
        identity = RecordIdentity(
            record_id=RecordId(record["record_id"]),
            instance=ArtifactInstance(
                content_sha256=record["content_sha256"],
                canonical_path=record["canonical_path"],
                size_bytes=record["size_bytes"]),
            role=role,
            legacy_aliases=tuple(record["legacy_aliases"]))
        return cls(identity=identity, cited_by=tuple(record["cited_by"]),
                   disclosure=disclosure, preservation=preservation,
                   provenance=provenance, retention=retention,
                   artifact_schema_version=record["artifact_schema_version"],
                   defect_note=record.get("defect_note", ""))


@dataclass(frozen=True)
class ArchiveManifest:
    """The index. Construction IS validation."""

    artifact_class: str
    genesis_cardinality: int
    genesis_aliases: tuple
    entries: tuple
    _checked: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not self.artifact_class.strip():
            raise ArchiveManifestError("artifact_class must be non-empty")
        if self.genesis_cardinality < 1:
            raise ArchiveManifestError(
                "a genesis cardinality below one would let an EMPTY archive "
                "satisfy every set equality it declares")
        if len(self.entries) < self.genesis_cardinality:
            raise ArchiveManifestError(
                "the archive holds {} entr(ies) but was born with {}. It may "
                "grow; it may never shrink.".format(
                    len(self.entries), self.genesis_cardinality))
        if len(set(self.genesis_aliases)) != len(self.genesis_aliases):
            raise ArchiveManifestError("duplicate genesis alias")
        if len(self.genesis_aliases) != self.genesis_cardinality:
            raise ArchiveManifestError(
                "{} genesis alias(es) for a cardinality of {}".format(
                    len(self.genesis_aliases), self.genesis_cardinality))

        ids = [e.identity.record_id.value for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ArchiveManifestError(
                "duplicate record identifier; a durable identity is not "
                "durable if two records share it")
        paths = [e.identity.instance.canonical_path for e in self.entries]
        if len(set(paths)) != len(paths):
            raise ArchiveManifestError(
                "two entries claim the same canonical path")

        # Identical CONTENT is legitimate -- the same bytes can occur in two
        # evidentiary contexts -- so content_sha256 is deliberately NOT unique.
        aliases = [a for e in self.entries for a in e.identity.legacy_aliases]
        if len(set(aliases)) != len(aliases):
            raise ArchiveManifestError(
                "an alias resolves to more than one record: {}".format(
                    sorted({a for a in aliases if aliases.count(a) > 1})))
        absent = sorted(set(self.genesis_aliases) - set(aliases))
        if absent:
            raise ArchiveManifestError(
                "genesis alias(es) no longer resolvable: {}. A historical "
                "citation that stops resolving is evidence lost.".format(absent))
        object.__setattr__(self, "_checked", True)

    @property
    def payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "artifact_class": self.artifact_class,
            "genesis_cardinality": self.genesis_cardinality,
            "genesis_aliases": sorted(self.genesis_aliases),
            "entries": sorted((e.as_record() for e in self.entries),
                              key=lambda r: r["canonical_path"]),
        }

    def render(self) -> bytes:
        """Deterministic AND diffable. Authored, so it ends with a newline."""
        return (json.dumps(self.payload, indent=2, sort_keys=True,
                           ensure_ascii=True) + "\n").encode("utf-8")

    def resolve(self, alias: str):
        """Historical basename -> entry, or None. The point of the index."""
        for entry in self.entries:
            if alias in entry.identity.legacy_aliases:
                return entry
        return None

    @classmethod
    def parse(cls, data: bytes) -> "ArchiveManifest":
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ArchiveManifestError("not valid JSON: {}".format(exc)) from None
        if not isinstance(payload, dict):
            raise ArchiveManifestError("a manifest must be an object")
        if payload.get("schema") != SCHEMA:
            raise ArchiveManifestError(
                "schema is {!r}, expected {!r}".format(
                    payload.get("schema"), SCHEMA))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ArchiveManifestError(
                "schema_version is {!r}; this parser judges version {} only"
                .format(payload.get("schema_version"), SCHEMA_VERSION))
        keys = set(payload)
        missing = sorted(_MANIFEST_KEYS - keys)
        unknown = sorted(keys - _MANIFEST_KEYS)
        if missing:
            raise ArchiveManifestError("manifest missing {}".format(missing))
        if unknown:
            raise ArchiveManifestError(
                "manifest has undeclared key(s) {}".format(unknown))
        if not isinstance(payload["entries"], list):
            raise ArchiveManifestError("entries must be a list")
        return cls(
            artifact_class=payload["artifact_class"],
            genesis_cardinality=payload["genesis_cardinality"],
            genesis_aliases=tuple(payload["genesis_aliases"]),
            entries=tuple(ArchiveEntry.from_record(r)
                          for r in payload["entries"]))
