"""What evidence was used, what role it played, and how it was obtained.

DRIFT-1 Phase 1B.3. Created 2026-08-28, replacing the 2026-08-27 version.

WHAT CHANGED, AND WHY EACH WAS MEASURED FIRST
---------------------------------------------
The 2026-08-27 version keyed identity on `source` alone and stated:
"One analysis reads ONE artifact per source."

MEASURED 2026-08-28 across 3,420 artifact files and every tracked module:

    authorities holding more than one artifact kind      10
    MAX distinct kinds consumed by ONE module             3

    monitoring/registry.py   ClinVar  index.parquet + parquet + variant_summary
    monitoring/registry.py   gnomAD   index.parquet + parquet
    data/protein_coords.py   AlphaMissense  index.parquet + tsv
    agent_layer/config.py    ClinVar  variant_summary.txt + vcf

The invariant is FALSE for this repository, and false in the package's own
modules -- not only in scripts. So the key is now `SourceArtifactKey`, a
(source, artifact_kind) pair. Forcing uniqueness on `source` would have
required faking names such as `ClinVarVCF`, turning an ARTIFACT distinction
into a SOURCE distinction and losing that both came from one release.

`genome_build` was mandatory and restricted to GRCh37/GRCh38. MEASURED: SIX of
sixteen authorities carry no genomic coordinates at all -- UniProt, Reactome,
OMIM, STRING-DB, AlphaFold, ESM-2. `CoordinateContext` makes
build-independence a POSITIVE claim rather than a missing value.

`source` was validated by pattern, so `ClinVar`, `clinvar` and `NCBI-ClinVar`
were three identities. MEASURED: no registry existed. `SourceName` is now a
controlled vocabulary and `resolve_source_name` is the ingestion boundary.

WHAT DID NOT CHANGE
-------------------
The 2026-08-27 separation of scientific identity from acquisition provenance
stands, and every test of it stands with it: `retrieved_at` is unreachable from
a digest, `source_deltas` takes an EVIDENCE manifest, and equality is not
redefined to hide the acquisition event.

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
from genomic_variant_classifier.monitoring.drift.coordinate import (
    CoordinateContext,
    assemblies_in,
)
from genomic_variant_classifier.monitoring.drift.source_vocabulary import (
    ArtifactKind,
    SourceName,
    resolve_source_name,
)

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_UTC = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_RELEASE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")

EVIDENCE_DOMAIN = "drift-source-evidence-manifest-v3"


class SourceRole(str, Enum):
    """What a source CONTRIBUTES to one analysis.

    On the DEPENDENCY, never on the artifact: one ClinVar release can be both
    the OBSERVATION population and the LABEL authority.
    """

    OBSERVATION = "observation"
    ANNOTATION = "annotation"
    LABEL = "label"
    REFERENCE_SEQUENCE = "reference_sequence"
    ONTOLOGY = "ontology"


class SourceError(ValueError):
    """A source record that cannot identify the evidence it describes."""


@dataclass(frozen=True)
class SourceArtifactKey:
    """WHICH artifact of WHICH authority. The uniqueness key.

    `source` alone is insufficient: measured, one module consumes up to THREE
    distinct ClinVar artifacts. Two artifacts of one authority are two
    dependencies; two dependencies of one KEY are a contradiction.
    """

    source: SourceName
    artifact_kind: ArtifactKind

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceName):
            raise SourceError(
                "source is {!r}; it must be a SourceName. Resolve a raw "
                "spelling at the ingestion boundary with resolve_source_name, "
                "so an unregistered name cannot mint an identity."
                .format(self.source))
        if not isinstance(self.artifact_kind, ArtifactKind):
            raise SourceError(
                "artifact_kind is {!r}; one authority publishes several kinds "
                "and they are not interchangeable".format(self.artifact_kind))

    @classmethod
    def of(cls, source, artifact_kind) -> "SourceArtifactKey":
        """Build from a raw spelling, resolving it at the boundary."""
        return cls(source=resolve_source_name(source),
                   artifact_kind=ArtifactKind(artifact_kind)
                   if not isinstance(artifact_kind, ArtifactKind)
                   else artifact_kind)

    @property
    def canonical_key(self) -> Tuple[str, str]:
        return (self.source.value, self.artifact_kind.value)

    def as_record(self) -> dict:
        return {"source": self.source.value,
                "artifact_kind": self.artifact_kind.value}

    def describe(self) -> str:
        return "{}/{}".format(self.source.value, self.artifact_kind.value)


@dataclass(frozen=True)
class SourceArtifactIdentity:
    """One artifact, of one kind, from one named release. Content-addressed."""

    key: SourceArtifactKey
    release_id: str
    coordinate_context: CoordinateContext
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, SourceArtifactKey):
            raise SourceError("key is {!r}".format(self.key))
        if not isinstance(self.release_id, str) or \
                not _RELEASE_ID.match(self.release_id):
            raise SourceError(
                "release_id is {!r}; expected an identifier such as '2026-08' "
                "with no whitespace or separators".format(self.release_id))
        if not isinstance(self.coordinate_context, CoordinateContext):
            raise SourceError(
                "coordinate_context is {!r}; an artifact must state whether "
                "its records carry genomic coordinates. Six of sixteen "
                "authorities carry none.".format(self.coordinate_context))
        if not isinstance(self.artifact_sha256, str) or \
                not _SHA256.match(self.artifact_sha256):
            raise SourceError(
                "artifact_sha256 is {!r}; expected 64 lowercase hexadecimal "
                "characters. A PATH IS NOT IDENTITY, and a prefix is not a "
                "digest.".format(self.artifact_sha256))

    @property
    def source(self) -> SourceName:
        return self.key.source

    @property
    def artifact_kind(self) -> ArtifactKind:
        return self.key.artifact_kind

    @property
    def canonical_key(self):
        """Ordering the DOMAIN means, not what field order implies."""
        return self.key.canonical_key + (
            self.release_id, self.coordinate_context.describe(),
            self.artifact_sha256)

    def as_record(self) -> dict:
        return {"key": self.key.as_record(), "release_id": self.release_id,
                "coordinate_context": self.coordinate_context.as_record(),
                "artifact_sha256": self.artifact_sha256}

    def describe(self) -> str:
        return "{}@{} [{}] {}".format(
            self.key.describe(), self.release_id,
            self.coordinate_context.describe(), self.artifact_sha256[:12])


@dataclass(frozen=True)
class SourceRetrievalProvenance:
    """WHEN and HOW an artifact was obtained. Never scientific identity."""

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
                "roles is {}; it must be a FROZENSET"
                .format(type(self.roles).__name__))
        if not self.roles:
            raise SourceError(
                "a dependency must declare at least one role. A source with "
                "no stated role cannot be governed by a protocol.")
        for r in self.roles:
            if not isinstance(r, SourceRole):
                raise SourceError("role {!r} is not a SourceRole".format(r))

    @property
    def key(self) -> SourceArtifactKey:
        return self.identity.key

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
        keys = [d.key for d in self.dependencies]
        if len(set(keys)) != len(keys):
            duplicated = sorted({k.describe() for k in keys
                                 if keys.count(k) > 1})
            raise SourceError(
                "artifact key(s) {} appear more than once. One analysis reads "
                "ONE artifact of each KIND per authority -- several kinds from "
                "one authority are several dependencies, which is measured "
                "practice: one module consumes three distinct ClinVar "
                "artifacts.".format(duplicated))
        assemblies = assemblies_in(d.identity.coordinate_context
                                   for d in self.dependencies)
        if len(assemblies) > 1:
            raise SourceError(
                "the manifest mixes genome assemblies {}. Coordinates from "
                "different assemblies are not comparable, and a join across "
                "them would be silently wrong. Build-independent evidence is "
                "unaffected and may accompany any assembly."
                .format(sorted(assemblies)))
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
    def assemblies(self) -> FrozenSet[str]:
        """Every genome assembly present. Build-independent adds nothing."""
        return assemblies_in(d.identity.coordinate_context
                             for d in self.dependencies)

    @property
    def keys(self) -> Tuple[SourceArtifactKey, ...]:
        return tuple(d.key for d in self.dependencies)

    @property
    def sources(self) -> Tuple[SourceName, ...]:
        """Distinct authorities, which may be fewer than the dependencies."""
        seen = []
        for d in self.dependencies:
            if d.identity.source not in seen:
                seen.append(d.identity.source)
        return tuple(seen)

    @property
    def digest(self) -> str:
        """Scientific evidence identity. DERIVED and DOMAIN-SEPARATED."""
        return domain_digest(EVIDENCE_DOMAIN, {
            "schema_version": 3,
            "dependencies": [d.as_record() for d in self.dependencies]})

    def dependency_of(self, key: SourceArtifactKey) -> SourceDependency:
        for d in self.dependencies:
            if d.key == key:
                return d
        raise SourceError("{} is not in this manifest; it names {}".format(
            key.describe(), [k.describe() for k in self.keys]))

    def artifacts_of(self, source) -> Tuple[SourceArtifactIdentity, ...]:
        """Every artifact of one authority. May legitimately be several."""
        name = resolve_source_name(source)
        return tuple(d.identity for d in self.dependencies
                     if d.identity.source is name)

    def describe(self) -> str:
        return "{} dependenc{} | {} authorit{} | assemblies {} | evidence {}\n  {}".format(
            len(self.dependencies),
            "y" if len(self.dependencies) == 1 else "ies",
            len(self.sources), "y" if len(self.sources) == 1 else "ies",
            sorted(self.assemblies) or "none", self.digest[:12],
            "\n  ".join(d.describe() for d in self.dependencies))


@dataclass(frozen=True)
class SourceManifest:
    """Evidence and acquisition together, with the two kept distinct."""

    evidence: SourceEvidenceManifest
    acquisitions: Tuple[SourceAcquisition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, SourceEvidenceManifest):
            raise SourceError("evidence is {!r}".format(self.evidence))
        if not isinstance(self.acquisitions, tuple):
            raise SourceError("acquisitions must be a TUPLE")
        declared = set(self.evidence.keys)
        for a in self.acquisitions:
            if not isinstance(a, SourceAcquisition):
                raise SourceError("entry {!r} is not a SourceAcquisition"
                                  .format(a))
            if a.identity.key not in declared:
                raise SourceError(
                    "an acquisition names {}, which the evidence manifest does "
                    "not declare. Recording how something was obtained that "
                    "was never used is a record of nothing."
                    .format(a.identity.key.describe()))

    @property
    def evidence_digest(self) -> str:
        return self.evidence.digest

    @property
    def digest(self) -> str:
        """Compatibility spelling for `evidence_digest`."""
        return self.evidence.digest

    @property
    def keys(self):
        return self.evidence.keys

    @property
    def sources(self):
        return self.evidence.sources

    @property
    def assemblies(self):
        return self.evidence.assemblies

    def describe(self) -> str:
        return "{}\n  {} acquisition record(s)".format(
            self.evidence.describe(), len(self.acquisitions))
