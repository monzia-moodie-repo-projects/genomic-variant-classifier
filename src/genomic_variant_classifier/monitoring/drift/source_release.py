"""Which releases produced the evidence, and what their identity is.

DRIFT-1 Phase 1B. Created 2026-08-27.

WHY THIS EXISTS
---------------
MEASURED 2026-08-27: of the eight facts DRIFT-1 Phase 1 needs, seven already
have owners in this repository. THIS ONE DOES NOT. A Layer-B authority scan
reported an owner and was WRONG -- it matched `anchor_manifest_sha256` in
`evaluation/moe_identity.py`, a field of `ExpertLineage` about mechanistic
anchor sets, unrelated to data releases. Registered as
PROBE-AUTHORITY-MATCH-UNVERIFIED-1: a substring match is not a concept match.

WHY A SET, NOT A SINGLE RELEASE
-------------------------------
A single release manifest is insufficient, and the reason is scientific rather
than architectural.

The semantic feature plane is a JOIN across many sources -- ClinVar, gnomAD,
dbNSFP, SpliceAI, AlphaMissense, phyloP and others -- each with its own release
cadence. A distribution change must be attributable to one of:

    delta V     the biological observation population
    delta S     the source-release state
    delta T     the transformation pipeline

Same ClinVar variants, new dbNSFP release, CADD moves: the POPULATION did not
drift, the MEASUREMENT PROCESS did. A representation carrying only ClinVar's
release identity cannot tell those apart -- the dbNSFP bump would be invisible
in the identity and would surface as unexplained feature drift.

So `SourceRelease` identifies ONE source at ONE release, and `SourceManifest`
carries the COMPLETE set with a derived digest. Any one release moving changes
the manifest digest, changes the representation identity, and the comparison is
REFUSED with a named cause instead of reported as drift.

WHY THE DIGEST IS DERIVED
-------------------------
The same argument as `RepresentationIdentity.feature_contract_digest`: a digest
stored beside the data it digests is two fields for one fact, and they can come
apart with nothing to notice. `SourceManifest.digest` is a property.

The attestation schema records `pre_head` AND `pre_head_oid` and BINDS them,
because an abbreviation cannot be derived from an identifier -- git chooses the
length. A manifest digest can be derived, so it is.

A PATH IS NOT IDENTITY
----------------------
`clinvar_2026_08.parquet` names where a file sits, not what it contains.
`ClinVarTracker.compare` currently records `new_cohort_path: Optional[str]` and
nothing else, so two runs over different bytes at one path are
indistinguishable. Every release here carries `artifact_sha256`.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; UTC = Coordinated Universal
Time.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Tuple

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_UTC = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
#: A release identifier is used in filenames and record keys, so it may not
#: carry a separator or whitespace.
_RELEASE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SOURCE = re.compile(r"\A[A-Za-z][A-Za-z0-9-]*\Z")

#: The genome builds this project admits. GRCh37 and GRCh38 coordinates are NOT
#: interchangeable, and a comparison across them would pair unrelated loci.
GENOME_BUILDS = ("GRCh37", "GRCh38")

_FIELD = "\x1f"      # unit separator: cannot occur in any validated field
_RECORD = "\x1e"     # record separator


@dataclass(frozen=True, order=True)
class SourceRelease:
    """One source, at one release, with the bytes that were actually read."""

    source: str
    release_id: str
    genome_build: str
    artifact_sha256: str
    row_count: int
    retrieved_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not _SOURCE.match(self.source):
            raise ValueError(
                "source is {!r}; expected a name like 'ClinVar' or 'gnomAD'"
                .format(self.source))
        if not isinstance(self.release_id, str) or \
                not _RELEASE_ID.match(self.release_id):
            raise ValueError(
                "release_id is {!r}; expected an identifier such as '2026-08' "
                "with no whitespace or separators".format(self.release_id))
        if self.genome_build not in GENOME_BUILDS:
            raise ValueError(
                "genome_build is {!r}; expected one of {}. GRCh37 and GRCh38 "
                "coordinates are NOT interchangeable, and comparing across "
                "them would pair unrelated loci."
                .format(self.genome_build, list(GENOME_BUILDS)))
        if not isinstance(self.artifact_sha256, str) or \
                not _SHA256.match(self.artifact_sha256):
            raise ValueError(
                "artifact_sha256 is {!r}; expected 64 lowercase hexadecimal "
                "characters. A path is not identity, and a prefix is not a "
                "digest.".format(self.artifact_sha256))
        if not isinstance(self.row_count, int) or \
                isinstance(self.row_count, bool) or self.row_count < 0:
            raise ValueError(
                "row_count is {!r}; expected a non-negative integer"
                .format(self.row_count))
        if not isinstance(self.retrieved_at, str) or \
                not _UTC.match(self.retrieved_at):
            raise ValueError(
                "retrieved_at is {!r}; expected YYYY-MM-DDTHH:MM:SSZ"
                .format(self.retrieved_at))

    def as_record(self) -> str:
        """The canonical serialisation this release contributes to a digest."""
        return _FIELD.join((self.source, self.release_id, self.genome_build,
                            self.artifact_sha256, str(self.row_count),
                            self.retrieved_at))

    def describe(self) -> str:
        return "{}@{} [{}] {} rows, {}".format(
            self.source, self.release_id, self.genome_build,
            self.row_count, self.artifact_sha256[:12])


class SourceManifestError(ValueError):
    """A manifest that cannot identify the evidence it claims to describe."""


@dataclass(frozen=True)
class SourceManifest:
    """EVERY release a representation depends on, and one derived digest."""

    releases: Tuple[SourceRelease, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.releases, tuple):
            raise SourceManifestError(
                "releases is {}; it must be a TUPLE so a manifest cannot be "
                "mutated after its digest has been quoted"
                .format(type(self.releases).__name__))
        if not self.releases:
            raise SourceManifestError(
                "a manifest must name at least one release. An EMPTY manifest "
                "would digest to a constant and make every representation "
                "compare equal on its sources.")
        for item in self.releases:
            if not isinstance(item, SourceRelease):
                raise SourceManifestError(
                    "manifest entry {!r} is not a SourceRelease".format(item))
        sources = [r.source for r in self.releases]
        if len(set(sources)) != len(sources):
            duplicated = sorted({s for s in sources if sources.count(s) > 1})
            raise SourceManifestError(
                "source(s) {} appear more than once. One representation reads "
                "ONE release per source; two would leave which was used "
                "undetermined.".format(duplicated))
        if list(self.releases) != sorted(self.releases):
            raise SourceManifestError(
                "releases are not in canonical order. `SourceManifest.of()` "
                "sorts them; constructing directly with an arbitrary order "
                "would make two equal manifests compare unequal. ORDER IS "
                "ENFORCED HERE and nowhere else -- an earlier version also "
                "sorted inside `digest`, so neither sort could be shown to "
                "matter and removing either changed nothing.")
        builds = {r.genome_build for r in self.releases}
        if len(builds) > 1:
            raise SourceManifestError(
                "the manifest mixes genome builds {}. Coordinates from "
                "different builds are not comparable, and a join across them "
                "would be silently wrong.".format(sorted(builds)))

    @classmethod
    def of(cls, releases: Iterable[SourceRelease]) -> "SourceManifest":
        """Build from any iterable, SORTED so member order cannot alter identity.

        Unlike a feature vector, a manifest has no meaningful order -- the set
        of releases is the fact. Sorting at construction means two manifests
        assembled in different orders are the SAME manifest, which is what a
        reader would expect and what a digest must reflect.
        """
        return cls(releases=tuple(sorted(releases)))

    @property
    def genome_build(self) -> str:
        return self.releases[0].genome_build

    @property
    def sources(self) -> Tuple[str, ...]:
        return tuple(r.source for r in self.releases)

    @property
    def digest(self) -> str:
        """DERIVED, never stored. See the module docstring."""
        # NOT re-sorted: __post_init__ has already enforced canonical order,
        # and sorting here too would be a second authority for one fact.
        payload = _RECORD.join(
            r.as_record() for r in self.releases).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def release_of(self, source: str) -> SourceRelease:
        for r in self.releases:
            if r.source == source:
                return r
        raise SourceManifestError(
            "{!r} is not in this manifest; it names {}".format(
                source, list(self.sources)))

    def describe(self) -> str:
        return "{} source(s) [{}] digest {}\n  {}".format(
            len(self.releases), self.genome_build, self.digest[:12],
            "\n  ".join(r.describe() for r in self.releases))


def differing_releases(reference: SourceManifest,
                       candidate: SourceManifest) -> Tuple[str, ...]:
    """Which sources moved between two manifests.

    Returned so a refusal can say WHICH release changed. "The manifests differ"
    sends a reader to diff two objects; "dbNSFP moved from 4.7a to 4.8a" is a
    scientific statement about measurement-process drift.
    """
    moved = []
    for source in sorted(set(reference.sources) | set(candidate.sources)):
        in_ref = source in reference.sources
        in_cand = source in candidate.sources
        if not in_ref or not in_cand:
            moved.append(source)
        elif reference.release_of(source) != candidate.release_of(source):
            moved.append(source)
    return tuple(moved)
