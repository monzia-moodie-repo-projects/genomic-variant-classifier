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
were three identities. The 2026-08-28 repair replaced the pattern with a
`SourceName` enum -- eighteen members and twenty-six aliases, all invented.

THAT REPAIR RESTED ON A FALSE MEASUREMENT, and this module said so in its own
docstring: "no registry existed". `configs/data_manifest.yaml` calls itself the
"Canonical registry of every data source under data/" on its own THIRD LINE,
declares 32 sources, and is read by five scripts. The authority search that
missed it looked only at Python files.

MEASURED 2026-08-29 against the manifest:

    declared sources     32      SourceName members    18
    it cannot name       16      aliases it accepted    0
    declared nowhere      2      aliases it invented   26

Four of the sixteen are `irreplaceable` and constrained: `tcga` and `topmed`
are `controlled`, `rnaseq` and `validation_cohort` are `review`.

`source` IS NOW A VALIDATED STRING, and membership is an ADMISSION question
answered by `genomic_variant_classifier.data.source_registry`. Identity stays
constructible without a readable file -- threading a registry through every
construction would make `SourceArtifactKey` depend on `configs/` being present,
which is the collapse this package has twice repaired: `RepresentationIdentity`
carries no source state, and `SourceArtifactIdentity` carries no retrieval time.

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

from genomic_variant_classifier.provenance.serialization import domain_digest
from genomic_variant_classifier.provenance.coordinate import (
    CoordinateContext,
    assemblies_in,
)
from genomic_variant_classifier.provenance.artifact import (
    ArtifactKind,
)

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_UTC = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_RELEASE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
#: SYNTAX only. Whether a name is a DECLARED source is a separate
#: question, answered by SourceRegistry against the manifest. This
#: pattern existed before the enum replaced it on 2026-08-28 and is
#: restored because a name with a space or a separator could not be a
#: directory under data/external/, whatever the manifest declares.
_SOURCE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]*\Z")
#: A product name, when one is carried. SYNTAX only, same shape as a source
#: name. The EMPTY string is refused so that "" can mean ABSENT in
#: `canonical_key` without ever colliding with a real product.
_PRODUCT = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

#: v4 since 2026-09-01. `SourceArtifactKey` gained an optional PRODUCT
#: coordinate, which alters EQUALITY -- GENCODE release 50 publishes three
#: transcript FASTA products that were one key under v3 and are three under
#: v4. A v3 digest and a v4 digest are therefore incomparable, which is the
#: point: a legacy record that cannot say WHICH FASTA it describes must be
#: REFUSED rather than silently given a product of "unknown".
EVIDENCE_DOMAIN = "drift-source-evidence-manifest-v4"


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


class SourceIdentityError(SourceError):
    """A key that cannot BE built, refused at the admission boundary.

    Phase 1C Unit 1, 2026-09-01. `SourceArtifactKey.of` is becoming the
    admission boundary for scientific evidence, so it must stop permissively
    stringifying whatever it is handed: `of(Path("x"), ...)` produced a source
    named `x` on POSIX and `x` on Windows but `WindowsPath('x')` under repr,
    and `of(None, ...)` produced the literal string "None".

    It subclasses `SourceError`, which is already a `ValueError`. The design
    authority specifies `ValueError`; subclassing the narrower existing error
    satisfies that AND keeps every `pytest.raises(SourceError)` in the suite
    catching it, so hardening the factory moves no test identity.
    """


@dataclass(frozen=True)
class SourceArtifactKey:
    """WHICH artifact of WHICH authority. The uniqueness key.

    `source` alone is insufficient: measured, one module consumes up to THREE
    distinct ClinVar artifacts. Two artifacts of one authority are two
    dependencies; two dependencies of one KEY are a contradiction.
    """

    source: str
    artifact_kind: ArtifactKind
    #: WHICH product of this authority, when the kind alone does not say.
    #:
    #: MEASURED 2026-08-28: GENCODE release 50 publishes `transcripts`,
    #: `pc_transcripts` and `lncRNA_transcripts` -- one authority, one release,
    #: one assembly, one artifact_kind, THREE distinct scientific products.
    #: Under `(source, artifact_kind)` all three collapsed to one key and
    #: `SourceEvidenceManifest` refused a legitimate state.
    #:
    #: OPTIONAL, and LAST. Absence means "this artifact kind does not require a
    #: product coordinate", not "the product is unknown". A mandatory field
    #: would put `product="default"` on every ClinVar and gnomAD record in the
    #: corpus, which carries no information and would have to be maintained.
    #: Last, because eighteen existing construction sites pass two positional
    #: arguments and must keep working.
    product: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not _SOURCE.match(self.source):
            raise SourceError(
                "source is {!r}; expected a name such as 'clinvar' or "
                "'gnomad'. WHETHER IT IS A DECLARED SOURCE IS A SEPARATE "
                "QUESTION, answered by SourceRegistry.canonical_for against "
                "configs/data_manifest.yaml. Identity must stay constructible "
                "without a readable file.".format(self.source))
        if not isinstance(self.artifact_kind, ArtifactKind):
            raise SourceError(
                "artifact_kind is {!r}; one authority publishes several kinds "
                "and they are not interchangeable".format(self.artifact_kind))
        if self.product is not None:
            if not isinstance(self.product, str) or not _PRODUCT.match(self.product):
                raise SourceError(
                    "product is {!r}; expected a name such as 'transcripts' or "
                    "'pc_transcripts', or None where the artifact kind already "
                    "identifies the artifact. The EMPTY string is refused: it "
                    "encodes ABSENCE in canonical_key and must not also name a "
                    "product.".format(self.product))

    @classmethod
    def of(cls, source, artifact_kind, product=None) -> "SourceArtifactKey":
        """Build from a raw spelling. It does NOT consult the registry.

        The pre-2026-08-29 version called `resolve_source_name`, which refused
        any spelling absent from an invented eighteen-member enum -- including
        all eight aliases this project declares and sixteen of its thirty-two
        sources. Registry membership is checked where the manifest is
        available, not where an identity is constructed.

        `product` is optional and defaults to None: absence means the artifact
        kind already identifies the artifact.

        HARDENED 2026-09-01. It no longer calls `str()` on whatever it is
        handed. This factory is becoming the admission boundary for scientific
        evidence, and `str(Path("clinvar.vcf"))`, `str(None)` and
        `str(3)` all produce a plausible source name for something that is not
        one. Whitespace is stripped, empties are refused, and an unrecognised
        `ArtifactKind` raises rather than propagating a bare ValueError.
        """
        if not isinstance(source, str):
            raise SourceIdentityError(
                "source must be a string, got {}. The factory no longer "
                "stringifies: str(Path(...)) and str(None) both produce a "
                "plausible-looking name for something that is not a source."
                .format(type(source).__name__))
        source = source.strip()
        if not source:
            raise SourceIdentityError("source cannot be empty or whitespace")

        if product is not None:
            if not isinstance(product, str):
                raise SourceIdentityError(
                    "product must be a string or None, got {}"
                    .format(type(product).__name__))
            product = product.strip()
            if not product:
                raise SourceIdentityError(
                    "an empty product is forbidden; use None. Absence means "
                    "the artifact kind already identifies the artifact, and "
                    "canonical_key renders it as the empty string -- so an "
                    "empty product would collide with absence.")

        try:
            kind = (artifact_kind if isinstance(artifact_kind, ArtifactKind)
                    else ArtifactKind(artifact_kind))
        except (TypeError, ValueError) as exc:
            raise SourceIdentityError(
                "unknown artifact kind {!r}. ArtifactKind is a LOCAL "
                "vocabulary with no external authority, so an unrecognised "
                "value is a defect rather than a source this project has not "
                "declared.".format(artifact_kind)) from exc

        return cls(source=source, artifact_kind=kind, product=product)

    @property
    def canonical_key(self) -> Tuple[str, str, str]:
        """THREE fields, always. An absent product renders as the empty string.

        Fixed arity, deliberately. A variable-length tuple would make a
        two-field key and a three-field key structurally different, which is
        the concatenation ambiguity `RepresentationIdentity` was
        length-prefixed to prevent on 2026-08-28. The empty string is safe
        because `__post_init__` refuses it as a product.
        """
        return (self.source, self.artifact_kind.value, self.product or "")

    def as_record(self) -> dict:
        return {"source": self.source,
                "artifact_kind": self.artifact_kind.value,
                "product": self.product or ""}

    def describe(self) -> str:
        if self.product:
            return "{}/{}/{}".format(self.source, self.artifact_kind.value,
                                     self.product)
        return "{}/{}".format(self.source, self.artifact_kind.value)


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
    def source(self) -> str:
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
    def sources(self) -> Tuple[str, ...]:
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
        return tuple(d.identity for d in self.dependencies
                     if d.identity.source == str(source))

    def describe(self) -> str:
        return "{} dependenc{} | {} authorit{} | assemblies {} | evidence {}\n  {}".format(
            len(self.dependencies),
            "y" if len(self.dependencies) == 1 else "ies",
            len(self.sources), "y" if len(self.sources) == 1 else "ies",
            sorted(self.assemblies) or "none", self.digest[:12],
            "\n  ".join(d.describe() for d in self.dependencies))


def _describe_identity_mismatch(candidate: "SourceArtifactIdentity",
                                declared) -> str:
    """Name WHAT differs, not merely THAT something does.

    SOURCE-ACQUISITION-KEY-ONLY-MATCH-1, measured 2026-09-02. The previous
    check compared `a.identity.key` against `evidence.keys`. A key is
    `(source, artifact_kind, product)`; `release_id`, `coordinate_context` and
    `artifact_sha256` are NOT in it. So a JULY retrieval record satisfied
    AUGUST evidence, and -- worse -- a GRCh37 record satisfied GRCh38
    evidence, though `CoordinateContext` exists precisely because those
    coordinates are not interchangeable.

    A caller who reaches this has made a scientific error, not a typo, so the
    message names the fields that differ against the nearest declared identity
    sharing the same key. Reporting only "does not match" would leave them to
    diff two objects by hand.
    """
    same_key = [d for d in declared if d.key == candidate.key]
    if not same_key:                                    # pragma: no cover
        return "no declared identity shares that key"
    parts = []
    for d in sorted(same_key, key=lambda x: x.release_id):
        differs = []
        if d.release_id != candidate.release_id:
            differs.append("release_id {!r} declared vs {!r} acquired"
                           .format(d.release_id, candidate.release_id))
        if d.coordinate_context != candidate.coordinate_context:
            differs.append("coordinates {} declared vs {} acquired"
                           .format(d.coordinate_context.describe(),
                                   candidate.coordinate_context.describe()))
        if d.artifact_sha256 != candidate.artifact_sha256:
            differs.append("digest {}... declared vs {}... acquired"
                           .format(d.artifact_sha256[:16],
                                   candidate.artifact_sha256[:16]))
        parts.append("; ".join(differs) if differs else "no field differs")
    return " | ".join(parts)


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
        declared_keys = set(self.evidence.keys)
        declared = {d.identity for d in self.evidence.dependencies}
        for a in self.acquisitions:
            if not isinstance(a, SourceAcquisition):
                raise SourceError("entry {!r} is not a SourceAcquisition"
                                  .format(a))
            if a.identity.key not in declared_keys:
                raise SourceError(
                    "an acquisition names {}, which the evidence manifest does "
                    "not declare. Recording how something was obtained that "
                    "was never used is a record of nothing."
                    .format(a.identity.key.describe()))
            if a.identity not in declared:
                raise SourceError(
                    "an acquisition describes a DIFFERENT MATERIALIZATION of "
                    "{}: {}. The evidence manifest declares that artifact, but "
                    "not these bytes. A retrieval record for one release does "
                    "not describe how another was obtained."
                    .format(a.identity.key.describe(),
                            _describe_identity_mismatch(a.identity, declared)))

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
