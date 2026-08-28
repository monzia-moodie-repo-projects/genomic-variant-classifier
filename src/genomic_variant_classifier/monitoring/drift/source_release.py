"""What evidence was used, what role it played, and how it was obtained.

DRIFT-1 Phase 1B.1. Created 2026-08-27, replacing the 2026-08-27 original.

THE DEFECT THIS REPLACES
------------------------
The original `SourceRelease` was one flat record whose canonical form hashed
`retrieved_at`. MEASURED against the installed code:

    same release, same bytes, same rows, same build, retrieved a day apart
        -> DIFFERENT manifest digests

The evidence was identical. Only the retrieval EVENT differed. And
`differing_releases` compared the WHOLE record, so the same re-download was
reported as a release change at the interpretation layer too -- repairing the
digest alone would have left that untouched.

THREE FACTS, THREE TYPES
------------------------
    SourceArtifactIdentity      WHAT bytes, from what named release?
    SourceRetrievalProvenance   HOW and WHEN were they obtained?
    SourceDependency            WHAT ROLE do they play in this analysis?

A source does not have one intrinsic role. ClinVar may be an OBSERVATION
source, a LABEL source, or a temporal-validation source depending on the
question. The role belongs to the RELATIONSHIP, not to the artifact -- so
`SourceDependency` carries roles and `SourceArtifactIdentity` does not.

EQUALITY IS NOT REDEFINED
-------------------------
It would be easy to make `SourceRelease.__eq__` ignore provenance. That would
merely reverse the original conflation:

    before        a provenance difference became a scientific difference
    bad repair    scientific equality erased the provenance difference
    correct       both survive independently

So three predicates coexist, and each says something different:

    monday == tuesday                    False -- distinct acquisition records
    monday.identity == tuesday.identity  True  -- same scientific evidence
    monday.provenance == ...             False -- the acquisition event moved

ROW COUNT IS VERIFICATION, NOT IDENTITY
---------------------------------------
`artifact_sha256` identifies the exact bytes read. If two readers derive
different row counts from identical bytes, the PARSER or the schema
interpretation changed -- which belongs to transformation or verification, not
to source identity. So `observed_row_count` lives in provenance, where a
mismatch under one artifact identity becomes a reader-inconsistency signal
rather than a pretence that the source moved.

ORDERING IS EXPLICIT
--------------------
`order=True` is gone. Implicit dataclass ordering over a composed record would
make canonical order depend on FIELD DECLARATION ORDER -- far too implicit for
something that participates in a persistent digest contract. `canonical_key`
names the ordering the domain means.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; UTC = Coordinated Universal
Time.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Iterable, Optional, Tuple

from genomic_variant_classifier.monitoring.drift._digest import domain_digest

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_UTC = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_RELEASE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SOURCE = re.compile(r"\A[A-Za-z][A-Za-z0-9-]*\Z")

#: GRCh37 and GRCh38 coordinates are NOT interchangeable.
GENOME_BUILDS = ("GRCh37", "GRCh38")

EVIDENCE_DOMAIN = "drift-source-evidence-manifest-v2"


class SourceRole(str, Enum):
    """What a source CONTRIBUTES to one analysis.

    On the dependency, never on the artifact: one ClinVar release can be both
    the OBSERVATION population and the LABEL authority, and forcing two
    records would duplicate one artifact identity.
    """

    #: Supplies the rows being studied.
    OBSERVATION = "observation"
    #: Supplies feature values measured on those rows.
    ANNOTATION = "annotation"
    #: Supplies the outcome being predicted.
    LABEL = "label"
    #: Defines the coordinate system.
    REFERENCE_SEQUENCE = "reference_sequence"
    #: Supplies term semantics whose meaning can shift between releases.
    ONTOLOGY = "ontology"


class SourceError(ValueError):
    """A source record that cannot identify the evidence it describes."""


@dataclass(frozen=True)
class SourceArtifactIdentity:
    """One artifact, from one named release. Content-addressed."""

    source: str
    release_id: str
    genome_build: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not _SOURCE.match(self.source):
            raise SourceError(
                "source is {!r}; expected a name like 'ClinVar' or 'gnomAD'"
                .format(self.source))
        if not isinstance(self.release_id, str) or \
                not _RELEASE_ID.match(self.release_id):
            raise SourceError(
                "release_id is {!r}; expected an identifier such as '2026-08' "
                "with no whitespace or separators".format(self.release_id))
        if self.genome_build not in GENOME_BUILDS:
            raise SourceError(
                "genome_build is {!r}; expected one of {}. GRCh37 and GRCh38 "
                "coordinates are NOT interchangeable, and comparing across "
                "them would pair unrelated loci."
                .format(self.genome_build, list(GENOME_BUILDS)))
        if not isinstance(self.artifact_sha256, str) or \
                not _SHA256.match(self.artifact_sha256):
            raise SourceError(
                "artifact_sha256 is {!r}; expected 64 lowercase hexadecimal "
                "characters. A PATH IS NOT IDENTITY, and a prefix is not a "
                "digest.".format(self.artifact_sha256))

    @property
    def canonical_key(self) -> Tuple[str, str, str, str]:
        """The ordering the DOMAIN means, not what field order implies."""
        return (self.source, self.release_id, self.genome_build,
                self.artifact_sha256)

    def as_record(self) -> dict:
        return {"source": self.source, "release_id": self.release_id,
                "genome_build": self.genome_build,
                "artifact_sha256": self.artifact_sha256}

    def describe(self) -> str:
        return "{}@{} [{}] {}".format(self.source, self.release_id,
                                      self.genome_build,
                                      self.artifact_sha256[:12])


@dataclass(frozen=True)
class SourceRetrievalProvenance:
    """WHEN and HOW an artifact was obtained. Never scientific identity.

    `origin_locator` is a path or a URL: evidence of where acquisition
    happened, not of what was acquired. It must never reach a digest.
    """

    retrieved_at: str
    observed_row_count: int
    origin_locator: Optional[str] = None
    transport: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.retrieved_at, str) or \
                not _UTC.match(self.retrieved_at):
            raise SourceError(
                "retrieved_at is {!r}; expected YYYY-MM-DDTHH:MM:SSZ"
                .format(self.retrieved_at))
        n = self.observed_row_count
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise SourceError(
                "observed_row_count is {!r}; expected a non-negative integer. "
                "It is VERIFICATION evidence: two readers deriving different "
                "counts from identical bytes indicates a parser change, not a "
                "source change.".format(n))
        for name in ("origin_locator", "transport"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v):
                raise SourceError(
                    "{} is {!r}; expected a non-empty string or None"
                    .format(name, v))


@dataclass(frozen=True)
class SourceDependency:
    """An artifact and the ROLES it plays in one analysis."""

    identity: SourceArtifactIdentity
    roles: FrozenSet[SourceRole]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SourceArtifactIdentity):
            raise SourceError("identity is {!r}".format(self.identity))
        if not isinstance(self.roles, frozenset):
            raise SourceError(
                "roles is {}; it must be a FROZENSET so a dependency cannot be "
                "mutated after its digest has been quoted"
                .format(type(self.roles).__name__))
        if not self.roles:
            raise SourceError(
                "a dependency must declare at least one role. A source with "
                "no stated role cannot be governed by a protocol.")
        for r in self.roles:
            if not isinstance(r, SourceRole):
                raise SourceError("role {!r} is not a SourceRole".format(r))

    @property
    def canonical_key(self):
        return self.identity.canonical_key

    def as_record(self) -> dict:
        return {"identity": self.identity.as_record(),
                "roles": sorted(r.value for r in self.roles)}

    def describe(self) -> str:
        return "{}  roles {}".format(
            self.identity.describe(), sorted(r.value for r in self.roles))


@dataclass(frozen=True)
class SourceAcquisition:
    """One artifact and one retrieval event. Many may share an identity."""

    identity: SourceArtifactIdentity
    provenance: SourceRetrievalProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SourceArtifactIdentity):
            raise SourceError("identity is {!r}".format(self.identity))
        if not isinstance(self.provenance, SourceRetrievalProvenance):
            raise SourceError("provenance is {!r}".format(self.provenance))

    @property
    def canonical_key(self):
        return self.identity.canonical_key


@dataclass(frozen=True)
class SourceEvidenceManifest:
    """WHICH scientific evidence was used. Retrieval cannot reach it."""

    dependencies: Tuple[SourceDependency, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dependencies, tuple):
            raise SourceError(
                "dependencies is {}; it must be a TUPLE"
                .format(type(self.dependencies).__name__))
        if not self.dependencies:
            raise SourceError(
                "a manifest must name at least one dependency. An EMPTY "
                "manifest would digest to a constant and make every "
                "representation compare equal on its sources.")
        for d in self.dependencies:
            if not isinstance(d, SourceDependency):
                raise SourceError("entry {!r} is not a SourceDependency"
                                  .format(d))
        sources = [d.identity.source for d in self.dependencies]
        if len(set(sources)) != len(sources):
            duplicated = sorted({s for s in sources if sources.count(s) > 1})
            raise SourceError(
                "source(s) {} appear more than once. One analysis reads ONE "
                "artifact per source; two would leave which was used "
                "undetermined. A source in several ROLES is one dependency "
                "with several roles.".format(duplicated))
        builds = {d.identity.genome_build for d in self.dependencies}
        if len(builds) > 1:
            raise SourceError(
                "the manifest mixes genome builds {}. Coordinates from "
                "different builds are not comparable, and a join across them "
                "would be silently wrong.".format(sorted(builds)))
        if list(self.dependencies) != sorted(
                self.dependencies, key=lambda d: d.canonical_key):
            raise SourceError(
                "dependencies are not in canonical order. `of()` sorts them; "
                "ORDER IS ENFORCED HERE AND NOWHERE ELSE, so the digest can "
                "consume the verified tuple verbatim.")

    @classmethod
    def of(cls, dependencies: Iterable[SourceDependency]
           ) -> "SourceEvidenceManifest":
        return cls(dependencies=tuple(
            sorted(dependencies, key=lambda d: d.canonical_key)))

    @property
    def genome_build(self) -> str:
        return self.dependencies[0].identity.genome_build

    @property
    def sources(self) -> Tuple[str, ...]:
        return tuple(d.identity.source for d in self.dependencies)

    @property
    def digest(self) -> str:
        """Scientific evidence identity. DERIVED and DOMAIN-SEPARATED."""
        return domain_digest(EVIDENCE_DOMAIN, {
            "schema_version": 2,
            "dependencies": [d.as_record() for d in self.dependencies]})

    def dependency_of(self, source: str) -> SourceDependency:
        for d in self.dependencies:
            if d.identity.source == source:
                return d
        raise SourceError("{!r} is not in this manifest; it names {}".format(
            source, list(self.sources)))

    def identity_of(self, source: str) -> SourceArtifactIdentity:
        return self.dependency_of(source).identity

    def describe(self) -> str:
        return "{} dependenc{} [{}] evidence {}\n  {}".format(
            len(self.dependencies),
            "y" if len(self.dependencies) == 1 else "ies",
            self.genome_build, self.digest[:12],
            "\n  ".join(d.describe() for d in self.dependencies))


@dataclass(frozen=True)
class SourceManifest:
    """Evidence and acquisition together, with the two kept distinct.

    `digest` is the EVIDENCE digest, and the name is retained because callers
    already spell it that way. New code should prefer `evidence_digest`, which
    says which of the two questions it answers.
    """

    evidence: SourceEvidenceManifest
    acquisitions: Tuple[SourceAcquisition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, SourceEvidenceManifest):
            raise SourceError("evidence is {!r}".format(self.evidence))
        if not isinstance(self.acquisitions, tuple):
            raise SourceError("acquisitions must be a TUPLE")
        declared = set(self.evidence.sources)
        for a in self.acquisitions:
            if not isinstance(a, SourceAcquisition):
                raise SourceError("entry {!r} is not a SourceAcquisition"
                                  .format(a))
            if a.identity.source not in declared:
                raise SourceError(
                    "an acquisition names {!r}, which the evidence manifest "
                    "does not declare. Recording how something was obtained "
                    "that was never used is a record of nothing."
                    .format(a.identity.source))

    @property
    def evidence_digest(self) -> str:
        return self.evidence.digest

    @property
    def digest(self) -> str:
        """Compatibility spelling for `evidence_digest`."""
        return self.evidence.digest

    @property
    def sources(self) -> Tuple[str, ...]:
        return self.evidence.sources

    @property
    def genome_build(self) -> str:
        return self.evidence.genome_build

    def describe(self) -> str:
        return "{}\n  {} acquisition record(s)".format(
            self.evidence.describe(), len(self.acquisitions))
