"""The artifact-inventory verification record: what was OBSERVED about the external artifact store, indexed by one typed owner.

Owner rulings 2026-10-08c (QA, QB, QC), 2026-10-08e and 2026-10-08f. Artifacts and runs live OUTSIDE the repository; the repository holds portable,
immutable verification records about them. One typed object owns construction, validation, serialization and the schema version
(ADR-0004 section G); construction IS validation; parsing is strict and must round-trip byte-for-byte.

THREE ENTITIES, NEVER CONFLATED (ruling 2026-10-08e section 1)
    requirement   a plan document's demand for content: (plan, entry id) -> SHA-256 (+ size when the plan records one)
    content       a SHA-256 with ONE consistent byte size, however many requirements name it
    location      (store, relative path) observed ONCE in the measurement, with what it held
"Observe locations once; identify their content; derive satisfaction of every requirement from those observations." The record stores
requirements, location observations and the per-content SEARCH that was performed; every requirement's result is DERIVED here, never
authored -- so repeated references cannot multiply copies and two requirements cannot assign one location contradictory content.

DERIVED RESULT OF A REQUIREMENT (EvidenceState from the shared admission layer -- no competing model -- plus a precise reason)
    MATCH        at least one observed location holds the content                               reason "matched"
    MISMATCH     none does, and the hint location was READ and holds other content (zero bytes included)
                                                                                                 reason "hint_holds_other_content"
    INVALID      none does, and the hint exists but could not be read as a regular file          reason "hint_not_regular_file" / "hint_unreadable"
    UNAVAILABLE  none does, and the hint is absent or there is none                              reason "not_found" (search complete)
                                                                                                        "search_incomplete" (otherwise)
The HINT CONDITION is reported SEPARATELY (no_hint / absent / matched / digest_mismatch / not_regular_file / unreadable): a damaged hint
can coexist with a valid copy elsewhere, and recovery may need to know.
A search is COMPLETE only when every candidate file was read: all files of the content's size (the size from a bound record, or from
the content itself once found), or every file in the store when no size could be established. An unreadable candidate makes it
incomplete -- an availability LIMITATION, never a claim of global absence.

COUNTS (derived) are of requirements, content objects and LOCATIONS -- never "replicas" or redundancy: a second path may be a hard link
or lie on the same failing disk. FILESYSTEM IDENTITY (ruling 2026-10-08f section 3): a READ location carries opaque digests of its volume
identifier and file identifier. Two locations with the same (store, volume, file) identity are the SAME filesystem file during this
observation (a hard link); two file identities on one volume are distinct filesystem files; two volume identities are distinct observed
volumes. None of this establishes separate physical disks, independent failure domains or recoverable backups, and file identifiers may
be reused over time: identities are compared only within ONE measurement and its declared store. SHA-256 remains the content identity;
contradictory content for one filesystem identity is refused (inventory.file_identity_content_conflict).

MEASUREMENT IS AN INTERVAL: measurement_started_at .. measurement_completed_at, with the collector's stability facts (metadata identical
before and after each hash; before finalization the whole file census repeated and compared, every observed absence re-checked). Metadata
comparison is not an atomic snapshot and cannot detect a modification followed by restoration of the original metadata. A record is
valid evidence of what was observed; whether an environment is READY to replay is a separate decision taken against the independently
admitted installation plan (environment_qualification.admission.artifact_readiness).

ROLE AND PLACEMENT. VERIFICATION_RESULT via canonical_root(ROLE) / environment-qualification / artifact-inventory / <REC id>.json.
LOCATIONS are a neutral store id + relative POSIX path; the absolute root lives only in the local binding outside every checkout
(RuntimePaths.artifact_store_bindings), ACCEPTED ONLY AS A PLAIN PATH (checked_store_root: no component from the anchor through the root,
nor below it, may be a symbolic link, a junction or any other Windows reparse point, inspected before anything is resolved). DISCLOSURE: digests and relative locations only; every text field is refused if it carries an
absolute or drive-qualified path, a backslash, a parent-directory step, a home-directory marker or a URL. RETENTION: PERMANENT_EVIDENCE;
a later measurement is a NEW record naming its predecessor; the current index is a REPLACEABLE projection.

Author: Monzia Moodie
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from genomic_variant_classifier.environment_qualification.admission import EvidenceState

from .classification import DisclosureClass, RetentionClass
from .identity import RecordId
from .roles import ArtifactRole, RecordsOntologyError, canonical_root

logger = logging.getLogger(__name__)

__all__ = ["SCHEMA", "SCHEMA_VERSION", "INDEX_SCHEMA", "BINDINGS_SCHEMA", "ROLE", "FAMILY", "ArtifactPurpose", "LocationCondition",
           "ArtifactInventoryError", "PlanDocument", "Requirement", "LocationObservation", "ContentSearch", "Collection",
           "RuntimeSupplied", "KnownGap", "RequirementResult", "ArtifactInventoryRecord", "family_root", "scan_records", "render_index",
           "load_store_bindings", "checked_store_root", "resolve_location", "is_link_or_junction", "is_redirected"]

SCHEMA = "gvc.artifact-inventory-verification"
SCHEMA_VERSION = 1
INDEX_SCHEMA = "gvc.artifact-inventory-index"
BINDINGS_SCHEMA = "gvc.artifact-store-bindings"
ROLE = ArtifactRole.VERIFICATION_RESULT
FAMILY = PurePosixPath("environment-qualification") / "artifact-inventory"
DISCLOSURE = DisclosureClass.HASH_ONLY_PUBLIC
RETENTION = RetentionClass.PERMANENT_EVIDENCE
_KINDS = frozenset({"source", "windows_binary", "local_binary", "evidence_bundle"})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_STORE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,199}")
_UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
_PRIORITY = frozenset({"base", "recommended"})
_SCOPES = frozenset({"same_size_files", "every_file"})
_REPARSE_POINT = stat.FILE_ATTRIBUTE_REPARSE_POINT          # 0x400; defined by the stat module on every platform
_UNDISCLOSABLE = (
    (re.compile(r"\\"), "a backslash"),
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]"), "a drive-qualified path"),
    (re.compile(r"(^|[\s(\"'=])/[^\s]"), "an absolute path"),
    (re.compile(r"(^|/)\.\.(/|$)"), "a parent-directory step"),
    (re.compile(r"(^|[\s/])~"), "a home-directory marker"),
    (re.compile(r"(?i)(^|[/\s])(users|home)/"), "a user directory"),
    (re.compile(r"(?i)[a-z][a-z0-9+.-]*://"), "a URL"),
)


class ArtifactInventoryError(RecordsOntologyError):
    """The artifact-inventory record does not satisfy its own contract."""


def _require(condition, message: str) -> None:
    if not condition:
        raise ArtifactInventoryError(message)


def _disclosable(value: str, what: str) -> str:
    for pattern, name in _UNDISCLOSABLE:
        _require(pattern.search(value) is None, "{}: carries {} -- not disclosable in a public record".format(what, name))
    return value


def _text(value, what: str, *, empty_ok: bool = False) -> str:
    _require(type(value) is str, "{}: text required".format(what))
    _require(value == value.strip() and (empty_ok or value != ""), "{}: non-empty trimmed text".format(what))
    return _disclosable(value, what)


def _sha(value, what: str) -> str:
    _require(type(value) is str and _SHA256.fullmatch(value) is not None, "{}: a lowercase SHA-256".format(what))
    return value


def _label(value, what: str) -> str:
    _require(type(value) is str and _LABEL.fullmatch(value) is not None, "{}: a label".format(what))
    return value


def _count(value, what: str) -> int:
    _require(type(value) is int and value >= 0, "{}: a non-negative integer".format(what))
    return value


def _location(value, what: str) -> str:
    _text(value, what)
    p = PurePosixPath(value)
    normalised = not p.is_absolute() and value == p.as_posix() and "." not in p.parts and ".." not in p.parts and p.parts
    _require(normalised, "{}: a normalised relative POSIX path".format(what))
    return value


def _store(value, what: str) -> str:
    _require(type(value) is str and _STORE.fullmatch(value) is not None, "{}: a neutral store identifier".format(what))
    return value


def _utc(value, what: str) -> str:
    _require(type(value) is str and _UTC.fullmatch(value) is not None, "{}: YYYY-MM-DDTHH:MM:SSZ".format(what))
    try:
        datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ArtifactInventoryError("{}: not a real UTC time".format(what)) from None
    return value


class ArtifactPurpose(str, Enum):
    """Why content is required (a fact from the named plan document, not from the store)."""

    ACQUIRED = "acquired"                                   # an accepted acquisition-manifest row
    SELECTED_UPSTREAM_BINARY = "selected_upstream_binary"   # installed by a sealed plan, officially distributed
    SELECTED_LOCAL_BUILD = "selected_local_build"           # installed by a sealed plan, produced by a recorded local build
    BUILD_SOURCE = "build_source"                           # the source a recorded local build consumed
    BOOTSTRAP = "bootstrap"                                 # installed before the plan's own installer (renv)
    UNSELECTED_ALTERNATIVE = "unselected_alternative"       # preserved, deliberately not selected
    RUN_EVIDENCE = "run_evidence"                           # a run's evidence bundle


class LocationCondition(str, Enum):
    """What one location held when it was observed."""

    READ = "read"                                           # a regular file whose bytes were read (size may be 0)
    ABSENT = "absent"                                       # nothing exists at the location
    NOT_REGULAR_FILE = "not_regular_file"                   # something exists that is not a regular file
    UNREADABLE = "unreadable"                               # a regular file whose bytes could not be read


@dataclass(frozen=True)
class PlanDocument:
    """A document requirements (or a size) were derived from, identified by its digest. The label is a name, not a path."""

    label: str
    sha256: str

    def __post_init__(self) -> None:
        _label(self.label, "plan document: label")
        _sha(self.sha256, "plan document {}: sha256".format(self.label))


@dataclass(frozen=True)
class Requirement:
    """One plan document's demand for content. Several requirements may name the same content."""

    entry_id: str
    kind: str
    package: str
    version: str
    sha256: str
    size_bytes: object          # positive int, or None when the plan document records no size
    purposes: tuple
    expected_from: str          # a PlanDocument label
    location_hint: object       # (store, relative path) or None

    def __post_init__(self) -> None:
        w = "requirement {}".format(self.entry_id)
        _label(self.entry_id, "requirement: entry_id")
        _require(self.kind in _KINDS, "{}: kind {!r}".format(w, self.kind))
        _text(self.package, w + ": package")
        _text(self.version, w + ": version", empty_ok=self.kind == "evidence_bundle")
        _sha(self.sha256, w + ": sha256")
        _require(self.size_bytes is None or (type(self.size_bytes) is int and self.size_bytes > 0), w + ": size_bytes (expected size > 0)")
        _require(type(self.purposes) is tuple and self.purposes and all(type(p) is ArtifactPurpose for p in self.purposes)
                 and len(set(self.purposes)) == len(self.purposes), w + ": purposes")
        object.__setattr__(self, "purposes", tuple(sorted(self.purposes, key=lambda x: x.value)))     # a set: canonical order
        _label(self.expected_from, w + ": expected_from")
        if self.location_hint is not None:
            _require(type(self.location_hint) is tuple and len(self.location_hint) == 2, w + ": location_hint")
            _store(self.location_hint[0], w + ": hint store")
            _location(self.location_hint[1], w + ": hint path")

    def as_record(self) -> dict:
        return {"entry_id": self.entry_id, "kind": self.kind, "package": self.package, "version": self.version, "sha256": self.sha256,
                "size_bytes": self.size_bytes, "purposes": [p.value for p in self.purposes], "expected_from": self.expected_from,
                "location_hint": None if self.location_hint is None else {"store": self.location_hint[0], "path": self.location_hint[1]}}


@dataclass(frozen=True)
class LocationObservation:
    """One location, observed ONCE. A READ location carries its content identity (sha256, size >= 0: an empty file has a digest too)
    and opaque FILESYSTEM identities of the file and its volume (digests of the volume and file identifiers), so hard links and shared
    volumes among matching locations can be counted. These are observation identities of one measurement on one machine -- not
    portable locators, not permanent content identifiers (file identifiers can be reused), not evidence of separate disks."""

    store: str
    path: str
    condition: LocationCondition
    sha256: object              # str for READ, else None
    size_bytes: object          # int >= 0 for READ, else None
    file_identity: object       # str (opaque sha256) for READ, else None
    volume_identity: object     # str (opaque sha256) for READ, else None

    def __post_init__(self) -> None:
        w = "location {}/{}".format(self.store, self.path)
        _store(self.store, w + ": store")
        _location(self.path, w + ": path")
        _require(type(self.condition) is LocationCondition, w + ": condition must be a LocationCondition")
        if self.condition is LocationCondition.READ:
            _sha(self.sha256, w + ": sha256")
            _require(type(self.size_bytes) is int and self.size_bytes >= 0, w + ": size_bytes (observed size >= 0)")
            _sha(self.file_identity, w + ": file_identity")
            _sha(self.volume_identity, w + ": volume_identity")
        else:
            _require(self.sha256 is None and self.size_bytes is None and self.file_identity is None and self.volume_identity is None,
                     w + ": {} records no content".format(self.condition.value))

    @property
    def key(self) -> tuple:
        return (self.store, self.path)

    def as_record(self) -> dict:
        return {"store": self.store, "path": self.path, "condition": self.condition.value, "sha256": self.sha256,
                "size_bytes": self.size_bytes, "file_identity": self.file_identity, "volume_identity": self.volume_identity}


@dataclass(frozen=True)
class ContentSearch:
    """The search performed for ONE required content object. size_from names the plan document that BOUND the size; a size with no
    size_from was established from the content itself (a read location holding it). scope: every file of that size, or every file
    when no size could be established. The search is complete only when no candidate was unreadable."""

    sha256: str
    size_bytes: object          # positive int or None
    size_from: object           # PlanDocument label or None
    scope: str
    candidates_read: int
    candidates_unreadable: int

    def __post_init__(self) -> None:
        w = "search {}".format(self.sha256[:12] if type(self.sha256) is str else self.sha256)
        _sha(self.sha256, w + ": sha256")
        _require(self.size_bytes is None or (type(self.size_bytes) is int and self.size_bytes >= 0), w + ": size_bytes")
        _require(self.size_from is None or (type(self.size_from) is str and _LABEL.fullmatch(self.size_from)), w + ": size_from")
        _require(self.size_from is None or self.size_bytes is not None, w + ": size_from names a document but no size")
        _require(self.scope in _SCOPES, w + ": scope")
        # A BOUND size permits searching only files of that size; with no size at all, only searching every file is complete
        # evidence. A size learned from found content (size_from None) is compatible with either scope.
        _require(self.size_from is None or self.scope == "same_size_files", w + ": a bound size implies the same_size_files scope")
        _require(self.size_bytes is not None or self.scope == "every_file", w + ": with no size, the scope must be every_file")
        _count(self.candidates_read, w + ": candidates_read")
        _count(self.candidates_unreadable, w + ": candidates_unreadable")

    @property
    def complete(self) -> bool:
        return self.candidates_unreadable == 0

    def as_record(self) -> dict:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes, "size_from": self.size_from, "scope": self.scope,
                "candidates_read": self.candidates_read, "candidates_unreadable": self.candidates_unreadable, "complete": self.complete}


@dataclass(frozen=True)
class Collection:
    """Facts about the measurement itself, attested by the collector (the owner checks their arithmetic, not the filesystem)."""

    files_indexed: int
    files_hashed: int
    files_rechecked: int
    final_census_files: int     # regular files whose metadata the repeated census compared before finalization (ruling 2026-10-08f)
    stability: str              # what was compared before / after hashing and before finalization
    assumption: str             # the controlled-workflow assumption the measurement relies on

    def __post_init__(self) -> None:
        for name in ("files_indexed", "files_hashed", "files_rechecked", "final_census_files"):
            _count(getattr(self, name), "collection: " + name)
        _require(self.files_hashed <= self.files_indexed, "collection: more files hashed than indexed")
        _require(self.files_rechecked == self.files_hashed, "collection: every hashed file must be re-checked before finalization")
        _require(self.final_census_files == self.files_indexed,
                 "collection: the repeated census must compare every indexed file, not only the hashed ones")
        _text(self.stability, "collection: stability")
        _text(self.assumption, "collection: assumption")

    def as_record(self) -> dict:
        return {"files_indexed": self.files_indexed, "files_hashed": self.files_hashed, "files_rechecked": self.files_rechecked,
                "final_census_files": self.final_census_files, "stability": self.stability, "assumption": self.assumption}


@dataclass(frozen=True)
class RuntimeSupplied:
    """A package the qualified runtime itself supplies: no store content exists, so it is accounted for here, explicitly."""

    package: str
    version: str
    priority: str
    runtime_record_sha256: str

    def __post_init__(self) -> None:
        _text(self.package, "runtime-supplied: package")
        _text(self.version, "runtime-supplied {}: version".format(self.package))
        _require(self.priority in _PRIORITY, "runtime-supplied {}: priority".format(self.package))
        _sha(self.runtime_record_sha256, "runtime-supplied {}: runtime_record_sha256".format(self.package))

    def as_record(self) -> dict:
        return {"package": self.package, "version": self.version, "priority": self.priority, "runtime_record_sha256": self.runtime_record_sha256}


@dataclass(frozen=True)
class KnownGap:
    """Something a plan or run inventory names but for which NO expected digest exists (so it cannot be a requirement)."""

    name: str
    description: str

    def __post_init__(self) -> None:
        _label(self.name, "gap: name")
        _text(self.description, "gap {}: description".format(self.name))

    def as_record(self) -> dict:
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True)
class RequirementResult:
    """DERIVED by the record from its locations and searches -- never constructed from outside."""

    entry_id: str
    state: EvidenceState
    reason: str
    hint_condition: str
    matching_locations: tuple
    search_complete: bool

    def as_record(self) -> dict:
        return {"entry_id": self.entry_id, "state": self.state.value, "reason": self.reason, "hint_condition": self.hint_condition,
                "matching_locations": [{"store": s, "path": p} for s, p in self.matching_locations], "search_complete": self.search_complete}


def family_root() -> PurePosixPath:
    return canonical_root(ROLE) / FAMILY


@dataclass(frozen=True)
class ArtifactInventoryRecord:
    """One measurement. Construction IS validation; render() is the only serialization; parse() must round-trip."""

    record_id: RecordId
    previous_record_id: object      # RecordId or None
    measurement_started_at: str
    measurement_completed_at: str
    scope: str
    verifier_sha256: str
    stores: tuple
    plan_documents: tuple
    requirements: tuple
    locations: tuple
    searches: tuple
    collection: Collection
    runtime_supplied: tuple
    gaps: tuple

    @property
    def canonical_path(self) -> PurePosixPath:
        return family_root() / (self.record_id.value + ".json")

    def __post_init__(self) -> None:
        _require(type(self.record_id) is RecordId, "record_id: a RecordId")
        _require(self.previous_record_id is None or type(self.previous_record_id) is RecordId, "previous_record_id: a RecordId or None")
        _require(self.previous_record_id != self.record_id, "previous_record_id: a record cannot follow itself")
        _utc(self.measurement_started_at, "measurement_started_at")
        _utc(self.measurement_completed_at, "measurement_completed_at")
        _require(self.measurement_completed_at >= self.measurement_started_at, "measurement interval: completed before it started")
        _text(self.scope, "scope")
        _sha(self.verifier_sha256, "verifier_sha256")
        _require(type(self.collection) is Collection, "collection: a Collection")
        for what, items, kind in (("stores", self.stores, str), ("plan_documents", self.plan_documents, PlanDocument),
                                  ("requirements", self.requirements, Requirement), ("locations", self.locations, LocationObservation),
                                  ("searches", self.searches, ContentSearch), ("runtime_supplied", self.runtime_supplied, RuntimeSupplied),
                                  ("gaps", self.gaps, KnownGap)):
            _require(type(items) is tuple and all(type(x) is kind for x in items), "{}: a tuple of {}".format(what, kind.__name__))
        # CANONICAL ORDER: a record is a set of facts; constructions differing only in order are EQUAL and render identically.
        for name, key in (("stores", None), ("plan_documents", lambda d: d.label), ("requirements", lambda r: r.entry_id),
                          ("locations", lambda o: o.key), ("searches", lambda s: s.sha256), ("runtime_supplied", lambda r: r.package),
                          ("gaps", lambda g: g.name)):
            object.__setattr__(self, name, tuple(sorted(getattr(self, name), key=key)))
        _require(self.stores and self.plan_documents and self.requirements, "stores, plan_documents and requirements must be non-empty")
        for s in self.stores:
            _store(s, "stores")
        _require(len(set(self.stores)) == len(self.stores), "stores: duplicate")
        labels = [d.label for d in self.plan_documents]
        _require(len(set(labels)) == len(labels), "plan_documents: duplicate label")
        ids = [r.entry_id for r in self.requirements]
        _require(len(set(ids)) == len(ids), "requirements: duplicate entry_id")
        # CONTENT: one size per digest across every requirement.
        bound = {}
        for r in self.requirements:
            _require(r.expected_from in labels, "requirement {}: expected_from names no plan document".format(r.entry_id))
            _require(r.location_hint is None or r.location_hint[0] in self.stores, "requirement {}: hint names an undeclared store".format(r.entry_id))
            if r.size_bytes is not None:
                _require(bound.setdefault(r.sha256, r.size_bytes) == r.size_bytes,
                         "content {}: requirements declare incompatible sizes (inventory.digest_size_conflict)".format(r.sha256[:12]))
        # LOCATIONS: each observed once, so one location carries one identity.
        keys = [o.key for o in self.locations]
        _require(len(set(keys)) == len(keys), "locations: a location is observed twice (inventory.location_content_conflict)")
        for o in self.locations:
            _require(o.store in self.stores, "location {}/{}: undeclared store".format(o.store, o.path))
        # FILESYSTEM IDENTITY (ruling 2026-10-08f section 3): READ locations sharing a (store, volume, file) identity are one filesystem
        # file during this measurement, so they must agree about its content; disagreement is a consistency failure, never two files.
        identities = {}
        for o in self.locations:
            if o.condition is LocationCondition.READ:
                seen = identities.setdefault((o.store, o.volume_identity, o.file_identity), (o.sha256, o.size_bytes))
                _require(seen == (o.sha256, o.size_bytes), "location {}/{}: another location with the same filesystem identity holds other "
                         "content (inventory.file_identity_content_conflict)".format(o.store, o.path))
        required = {r.sha256 for r in self.requirements}
        hints = {r.location_hint for r in self.requirements if r.location_hint is not None}
        for o in self.locations:
            relevant = o.key in hints or (o.condition is LocationCondition.READ and o.sha256 in required)
            _require(relevant, "location {}/{}: neither a hint nor a holder of required content".format(o.store, o.path))
        for h in hints:
            _require(h in set(keys), "hint {}/{}: a requirement's hint location was not observed".format(*h))
        # SEARCHES: exactly one per required content object, consistent with every size and every holder.
        searched = [s.sha256 for s in self.searches]
        _require(len(set(searched)) == len(searched), "searches: duplicate content")
        _require(set(searched) == required, "searches: coverage differs from the required content")
        holders = {}
        for o in self.locations:
            if o.condition is LocationCondition.READ:
                holders.setdefault(o.sha256, []).append(o)
        for s in self.searches:
            sizes = {o.size_bytes for o in holders.get(s.sha256, [])}
            _require(len(sizes) <= 1, "content {}: holders of one digest differ in size".format(s.sha256[:12]))
            if s.sha256 in bound:
                _require(s.size_bytes == bound[s.sha256] and s.size_from is not None,
                         "search {}: must use, and attribute, the size the requirements bind".format(s.sha256[:12]))
            if s.size_from is not None:
                _require(s.size_from in labels, "search {}: size_from names no plan document".format(s.sha256[:12]))
            elif s.size_bytes is not None and s.sha256 not in bound:
                _require(sizes == {s.size_bytes}, "search {}: a size not bound by a document must be the found content's size".format(s.sha256[:12]))
            if sizes and s.size_bytes is not None:
                _require(sizes == {s.size_bytes}, "content {}: a holder's size differs from the established size".format(s.sha256[:12]))
            if s.size_bytes is None:
                _require(not sizes, "search {}: content found but its size not recorded".format(s.sha256[:12]))
        names = [r.package for r in self.runtime_supplied]
        _require(len(set(names)) == len(names), "runtime_supplied: duplicate package")
        _require(not set(names) & {r.package for r in self.requirements if r.kind != "evidence_bundle"},
                 "runtime_supplied: a runtime-supplied package also has a content requirement")
        gaps = [g.name for g in self.gaps]
        _require(len(set(gaps)) == len(gaps), "gaps: duplicate name")
        _require(self.collection.files_hashed >= len([o for o in self.locations if o.condition is LocationCondition.READ]),
                 "collection: fewer files hashed than read locations recorded")

    # ------------------------------------------------------------------------------------------------ derived semantics
    def results(self) -> tuple:
        by_key = {o.key: o for o in self.locations}
        search = {s.sha256: s for s in self.searches}
        out = []
        for r in self.requirements:
            matching = tuple(sorted(o.key for o in self.locations if o.condition is LocationCondition.READ and o.sha256 == r.sha256))
            hint = by_key.get(r.location_hint) if r.location_hint is not None else None
            if r.location_hint is None:
                hint_condition = "no_hint"
            elif hint.condition is LocationCondition.READ:
                hint_condition = "matched" if hint.sha256 == r.sha256 else "digest_mismatch"
            else:
                hint_condition = hint.condition.value
            complete = search[r.sha256].complete
            if matching:
                state, reason = EvidenceState.MATCH, "matched"
            elif hint_condition == "digest_mismatch":
                state, reason = EvidenceState.MISMATCH, "hint_holds_other_content"
            elif hint_condition in ("not_regular_file", "unreadable"):
                state, reason = EvidenceState.INVALID, "hint_" + hint_condition
            else:
                state, reason = EvidenceState.UNAVAILABLE, "not_found" if complete else "search_incomplete"
            out.append(RequirementResult(r.entry_id, state, reason, hint_condition, matching, complete))
        return tuple(out)

    def counts(self) -> dict:
        """DERIVED: requirements, content objects and LOCATIONS are distinct quantities; no redundancy is claimed."""
        states = {s.value: 0 for s in EvidenceState}
        for res in self.results():
            states[res.state.value] += 1
        required = {r.sha256 for r in self.requirements}
        matching = [o for o in self.locations if o.condition is LocationCondition.READ and o.sha256 in required]
        by_content = {}
        for o in matching:
            by_content.setdefault(o.sha256, []).append(o)
        return {"requirements": len(self.requirements), "expected_content_objects": len(required),
                "matched_content_objects": len(by_content), "verified_matching_locations": len(matching),
                "additional_matching_locations": sum(len(v) - 1 for v in by_content.values()),
                "distinct_filesystem_files_among_matching_locations": len({(o.store, o.volume_identity, o.file_identity) for o in matching}),
                "distinct_volume_identities_among_matching_locations": len({(o.store, o.volume_identity) for o in matching}),
                "incomplete_searches": sum(1 for s in self.searches if not s.complete),
                "requirement_states": states, "runtime_supplied": len(self.runtime_supplied), "known_gaps": len(self.gaps)}

    def payload(self) -> dict:
        return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "record_id": self.record_id.value,
                "previous_record_id": None if self.previous_record_id is None else self.previous_record_id.value,
                "role": ROLE.value, "disclosure": DISCLOSURE.value, "retention": RETENTION.value,
                "measurement_started_at": self.measurement_started_at, "measurement_completed_at": self.measurement_completed_at,
                "scope": self.scope, "verifier_sha256": self.verifier_sha256, "stores": list(self.stores),
                "plan_documents": [{"label": d.label, "sha256": d.sha256} for d in self.plan_documents],
                "requirements": [r.as_record() for r in self.requirements],
                "locations": [o.as_record() for o in self.locations],
                "searches": [s.as_record() for s in self.searches],
                "collection": self.collection.as_record(),
                "runtime_supplied": [r.as_record() for r in self.runtime_supplied],
                "gaps": [g.as_record() for g in self.gaps],
                "results": [r.as_record() for r in self.results()],
                "counts": self.counts()}

    def render(self) -> bytes:
        """Deterministic (ADR-0004 section G): indent 2, sorted keys, ASCII, one final newline. AUTHORED, so normalised."""
        return (json.dumps(self.payload(), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")

    @classmethod
    def parse(cls, raw: bytes) -> "ArtifactInventoryRecord":
        doc = _strict_load(raw)
        _keys(doc, ("schema", "schema_version", "record_id", "previous_record_id", "role", "disclosure", "retention", "measurement_started_at",
                    "measurement_completed_at", "scope", "verifier_sha256", "stores", "plan_documents", "requirements", "locations",
                    "searches", "collection", "runtime_supplied", "gaps", "results", "counts"), "record")
        _require(doc["schema"] == SCHEMA and type(doc["schema_version"]) is int and doc["schema_version"] == SCHEMA_VERSION, "schema")
        _require(doc["role"] == ROLE.value and doc["disclosure"] == DISCLOSURE.value and doc["retention"] == RETENTION.value,
                 "role / disclosure / retention")
        for key in ("stores", "plan_documents", "requirements", "locations", "searches", "runtime_supplied", "gaps"):
            _require(type(doc[key]) is list, "{}: must be a list".format(key))
        docs = []
        for d in doc["plan_documents"]:
            _keys(d, ("label", "sha256"), "plan document")
            docs.append(PlanDocument(d["label"], d["sha256"]))
        reqs = []
        for e in doc["requirements"]:
            _keys(e, ("entry_id", "kind", "package", "version", "sha256", "size_bytes", "purposes", "expected_from", "location_hint"), "requirement")
            _require(type(e["purposes"]) is list, "requirement: purposes must be a list")
            try:
                purposes = tuple(ArtifactPurpose(p) for p in e["purposes"])
            except ValueError as exc:
                raise ArtifactInventoryError("requirement: unrecognised purpose: {}".format(exc)) from None
            hint = e["location_hint"]
            if hint is not None:
                _keys(hint, ("store", "path"), "location_hint")
                hint = (hint["store"], hint["path"])
            reqs.append(Requirement(e["entry_id"], e["kind"], e["package"], e["version"], e["sha256"], e["size_bytes"], purposes,
                                    e["expected_from"], hint))
        locations = []
        for o in doc["locations"]:
            _keys(o, ("store", "path", "condition", "sha256", "size_bytes", "file_identity", "volume_identity"), "location")
            try:
                condition = LocationCondition(o["condition"])
            except ValueError:
                raise ArtifactInventoryError("location: unrecognised condition {!r}".format(o["condition"])) from None
            locations.append(LocationObservation(o["store"], o["path"], condition, o["sha256"], o["size_bytes"], o["file_identity"],
                                                 o["volume_identity"]))
        searches = []
        for s in doc["searches"]:
            _keys(s, ("sha256", "size_bytes", "size_from", "scope", "candidates_read", "candidates_unreadable", "complete"), "search")
            searches.append(ContentSearch(s["sha256"], s["size_bytes"], s["size_from"], s["scope"], s["candidates_read"], s["candidates_unreadable"]))
        c = doc["collection"]
        _keys(c, ("files_indexed", "files_hashed", "files_rechecked", "final_census_files", "stability", "assumption"), "collection")
        collection = Collection(c["files_indexed"], c["files_hashed"], c["files_rechecked"], c["final_census_files"], c["stability"],
                                c["assumption"])
        runtime = []
        for r in doc["runtime_supplied"]:
            _keys(r, ("package", "version", "priority", "runtime_record_sha256"), "runtime-supplied")
            runtime.append(RuntimeSupplied(r["package"], r["version"], r["priority"], r["runtime_record_sha256"]))
        gaps = []
        for g in doc["gaps"]:
            _keys(g, ("name", "description"), "gap")
            gaps.append(KnownGap(g["name"], g["description"]))
        previous = None if doc["previous_record_id"] is None else RecordId(doc["previous_record_id"])
        record = cls(RecordId(doc["record_id"]), previous, doc["measurement_started_at"], doc["measurement_completed_at"], doc["scope"],
                     doc["verifier_sha256"], tuple(doc["stores"]), tuple(docs), tuple(reqs), tuple(locations), tuple(searches), collection,
                     tuple(runtime), tuple(gaps))
        _require(record.render() == raw, "record: not in its deterministic rendering (round-trip differs; results and counts are derived)")
        return record


def _strict_load(raw: bytes):
    """Duplicate keys, floats, NaN / Infinity and a byte-order mark are REFUSED -- one meaning per record."""
    _require(type(raw) is bytes and raw != b"", "record: empty or not bytes")
    _require(not raw.startswith(b"\xef\xbb\xbf"), "record: byte-order mark")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "record: duplicate key {!r}".format(key))
            result[key] = value
        return result

    def refuse(token):
        raise ArtifactInventoryError("record: non-integer number {!r}".format(token))

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=refuse, parse_constant=refuse)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactInventoryError("record: invalid JSON: {}".format(exc)) from None


def _keys(doc, required, what):
    _require(type(doc) is dict, "{}: must be an object".format(what))
    missing, unknown = sorted(set(required) - set(doc)), sorted(set(doc) - set(required))
    _require(not missing, "{}: missing {}".format(what, missing))
    _require(not unknown, "{}: undeclared key(s) {}".format(what, unknown))


# ---------------------------------------------------------------------------------------------------- the record set and its index
def scan_records(repository_root) -> tuple:
    """Every record of the family, each parsed strictly; its file name must be its own record id."""
    root = Path(repository_root).joinpath(*family_root().parts)
    if not root.exists():
        return ()
    _require(root.is_dir() and not is_redirected(root), "{}: not a plain directory".format(root))
    records = []
    for p in sorted(root.iterdir()):
        _require(not is_redirected(p), "{}: a link, junction or reparse point".format(p.name))
        if p.name == "index.json":
            continue
        _require(p.is_file() and p.name.startswith("REC-") and p.suffix == ".json", "{}: not a record file".format(p.name))
        record = ArtifactInventoryRecord.parse(p.read_bytes())
        _require(p.name == record.record_id.value + ".json", "{}: file name differs from its record id".format(p.name))
        records.append(record)
    return tuple(records)


def render_index(records) -> bytes:
    """The REPLACEABLE current index: a pure function of the records (it reads ids, never mints them). The records must form ONE
    linear chain -- every predecessor present, none followed twice -- so 'current' is unambiguous."""
    records = tuple(records)
    _require(all(type(r) is ArtifactInventoryRecord for r in records), "index: records required")
    ids = {r.record_id.value: r for r in records}
    _require(len(ids) == len(records), "index: duplicate record id")
    followed = [r.previous_record_id.value for r in records if r.previous_record_id is not None]
    _require(all(p in ids for p in followed), "index: a predecessor is missing")
    _require(len(set(followed)) == len(followed), "index: a record is followed twice (a fork)")
    roots = [r for r in records if r.previous_record_id is None]
    _require(len(roots) == (1 if records else 0), "index: the chain must have exactly one first record")
    heads = [i for i in ids if i not in set(followed)]
    _require(len(heads) == (1 if records else 0), "index: the chain must have exactly one current record")
    chain, cursor, nxt = [], roots[0] if roots else None, {r.previous_record_id.value: r for r in records if r.previous_record_id}
    while cursor is not None:
        chain.append(cursor)
        cursor = nxt.get(cursor.record_id.value)
    _require(len(chain) == len(records), "index: the records do not form one chain")
    body = {"schema": INDEX_SCHEMA, "schema_version": 1, "derived": True, "current": heads[0] if heads else None,
            "records": [{"record_id": r.record_id.value, "previous_record_id": None if r.previous_record_id is None else r.previous_record_id.value,
                         "measurement_started_at": r.measurement_started_at, "measurement_completed_at": r.measurement_completed_at,
                         "counts": r.counts()} for r in chain]}
    return (json.dumps(body, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


# ---------------------------------------------------------------------------------------------------- the LOCAL, non-portable binding
def is_link_or_junction(path) -> bool:
    """A symbolic link OR a Windows directory junction (Python 3.12 exposes Path.is_junction separately: a symlink check alone does
    not establish the store boundary). Never follows either. is_redirected is the stricter predicate the store uses."""
    p = Path(path)
    if p.is_symlink():
        return True
    is_junction = getattr(p, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    return bool(hasattr(os.path, "isjunction") and os.path.isjunction(p))


def _redirected(observed) -> bool:
    """From an lstat result: a symbolic link, or ANY Windows reparse point (junctions, mount points, cloud placeholders, ...)."""
    return stat.S_ISLNK(observed.st_mode) or bool(getattr(observed, "st_file_attributes", 0) & _REPARSE_POINT)


def is_redirected(path) -> bool:
    """True when the entry AT path is a symbolic link, a junction or any other Windows reparse point (ruling 2026-10-08f: cloud
    placeholders are not accepted as plain files); never follows. Nothing at path -> False (absence is a separate observation); any
    other inspection failure is REFUSED, never read as "not redirected"."""
    try:
        observed = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        raise ArtifactInventoryError("{}: could not be inspected ({}) (inventory.component_unreadable)".format(path, type(exc).__name__)) from None
    return _redirected(observed) or is_link_or_junction(path)


def load_store_bindings(path) -> dict:
    """{store id: absolute root} from the local binding file (RuntimePaths.artifact_store_bindings). Strict; absolute roots only."""
    p = Path(path)
    _require(p.is_file(), "bindings: {} is not a file".format(p))
    doc = _strict_load(p.read_bytes())
    _keys(doc, ("schema", "schema_version", "stores"), "bindings")
    _require(doc["schema"] == BINDINGS_SCHEMA and type(doc["schema_version"]) is int and doc["schema_version"] == 1, "bindings: schema")
    _require(type(doc["stores"]) is dict and doc["stores"], "bindings: stores must be a non-empty object")
    out = {}
    for store, root in doc["stores"].items():
        _store(store, "bindings: store id")
        _require(type(root) is str and Path(root).is_absolute(), "bindings {}: an absolute root".format(store))
        out[store] = Path(root)
    return out


def checked_store_root(root) -> Path:
    """The bound root, accepted ONLY as a plain path (ruling 2026-10-08f section 2), inspected BEFORE anything is resolved: absolute,
    no parent-directory step, and every component from the anchor through the root an existing directory that is neither a symbolic
    link, a junction nor any other Windows reparse point. The path must also be its own resolution (case aside), so an alias the
    component check cannot see -- a substituted drive, a short 8.3 name, a mapped share -- is refused too: bind the canonical path."""
    p = Path(root)
    _require(p.is_absolute() and ".." not in p.parts, "store root: {} is not a plain absolute path (inventory.root_not_plain_absolute_path)".format(p))
    for component in (*reversed(p.parents), p):             # parents come nearest-first: inspect anchor-first
        try:
            observed = os.lstat(component)
        except OSError as exc:
            raise ArtifactInventoryError("store root: {} could not be inspected ({}) (inventory.root_component_unreadable)".format(
                component, type(exc).__name__)) from None
        _require(not _redirected(observed) and not is_link_or_junction(component),
                 "store root: {} is a link, junction or reparse point (inventory.root_component_redirected)".format(component))
        _require(stat.S_ISDIR(observed.st_mode), "store root: {} is not a directory (inventory.root_component_not_directory)".format(component))
    try:
        resolved = p.resolve(strict=True)
    except OSError as exc:
        raise ArtifactInventoryError("store root: {} could not be resolved ({}) (inventory.root_component_unreadable)".format(
            p, type(exc).__name__)) from None
    _require(os.path.normcase(str(resolved)) == os.path.normcase(str(p)),
             "store root: {} resolves to {}; bind the canonical path (inventory.root_not_canonical)".format(p, resolved))
    return p


def resolve_location(bindings: dict, store: str, relative: str) -> Path:
    """The local file for (store, relative path). The root must be a plain canonical path (checked_store_root) and no component below
    it may be a link, junction or reparse point -- all checked BEFORE any content is read; the result must stay inside the root."""
    _store(store, "store")
    _location(relative, "path")
    _require(store in bindings, "store {!r} has no local binding".format(store))
    root = checked_store_root(bindings[store])
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        _require(not is_redirected(cursor), "{}: passes through a link, junction or reparse point (inventory.location_component_redirected)".format(relative))
    try:
        cursor.resolve(strict=False).relative_to(root)
    except ValueError:
        raise ArtifactInventoryError("{}: resolves outside the store root".format(relative)) from None
    return cursor
