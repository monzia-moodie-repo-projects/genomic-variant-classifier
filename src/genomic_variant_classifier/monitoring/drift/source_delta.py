"""What moved between two source states, and what kind of movement it was.

DRIFT-1 Phase 1B.1. Created 2026-08-27.

WHY A KIND AND NOT A NAME
-------------------------
The original `differing_releases` returned `tuple[str, ...]` -- the names of
sources that moved. "dbNSFP differs" is useful; "dbNSFP moved from 4.7a to
4.8a, and the artifact digest changed with it" is a scientific statement a
protocol can reason about.

THE CASE THAT MOST NEEDS A NAME
-------------------------------
    same source, same release_id, DIFFERENT artifact digest

That is not ordinary release movement. It means one of: an upstream source
silently replaced a same-labelled release, a download was corrupted, packaging
is not reproducible, or acquisition is defective. Reporting it as
`RELEASE_MOVED` would hide a serious problem inside a routine one, so it has
its own kind.

WHY THE UNCLASSIFIED BRANCH RAISES
----------------------------------
If two identities differ and no rule above classified the difference, a field
was added to `SourceArtifactIdentity` without a corresponding delta kind. The
alternative -- falling through silently -- would make that new field invisible
to every comparison, which is exactly how a change becomes undetectable. It
raises instead.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from genomic_variant_classifier.monitoring.drift.source_release import (
    SourceArtifactIdentity,
    SourceEvidenceManifest,
    SourceManifest,
)


class SourceDeltaKind(str, Enum):
    """How one source's evidence differs between two states."""

    #: Present in the candidate only.
    SOURCE_ADDED = "source_added"
    #: Present in the reference only.
    SOURCE_REMOVED = "source_removed"
    #: A different named release. Ordinary temporal movement.
    RELEASE_MOVED = "release_moved"
    #: SAME release label, DIFFERENT bytes. Never ordinary.
    ARTIFACT_CHANGED_UNDER_SAME_RELEASE = "artifact_changed_under_same_release"
    #: Coordinates are no longer comparable.
    GENOME_BUILD_CHANGED = "genome_build_changed"


@dataclass(frozen=True)
class SourceDelta:
    """One source, one kind of movement, both identities where they exist."""

    source: str
    kind: SourceDeltaKind
    reference: Optional[SourceArtifactIdentity]
    candidate: Optional[SourceArtifactIdentity]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SourceDeltaKind):
            raise ValueError("kind is {!r}".format(self.kind))
        if self.kind is SourceDeltaKind.SOURCE_ADDED:
            if self.reference is not None or self.candidate is None:
                raise ValueError(
                    "SOURCE_ADDED requires no reference and a candidate")
        elif self.kind is SourceDeltaKind.SOURCE_REMOVED:
            if self.reference is None or self.candidate is not None:
                raise ValueError(
                    "SOURCE_REMOVED requires a reference and no candidate")
        elif self.reference is None or self.candidate is None:
            raise ValueError(
                "{} describes a CHANGE and requires both identities"
                .format(self.kind.value))

    def describe(self) -> str:
        if self.kind is SourceDeltaKind.SOURCE_ADDED:
            return "{}: added at {}".format(self.source,
                                            self.candidate.release_id)
        if self.kind is SourceDeltaKind.SOURCE_REMOVED:
            return "{}: removed (was {})".format(self.source,
                                                 self.reference.release_id)
        if self.kind is SourceDeltaKind.ARTIFACT_CHANGED_UNDER_SAME_RELEASE:
            return ("{}: release {} UNCHANGED but bytes differ, {} -> {}"
                    .format(self.source, self.reference.release_id,
                            self.reference.artifact_sha256[:12],
                            self.candidate.artifact_sha256[:12]))
        if self.kind is SourceDeltaKind.GENOME_BUILD_CHANGED:
            return "{}: {} -> {}".format(self.source,
                                         self.reference.genome_build,
                                         self.candidate.genome_build)
        return "{}: {} -> {}".format(self.source, self.reference.release_id,
                                     self.candidate.release_id)


def source_deltas(reference: SourceEvidenceManifest,
                  candidate: SourceEvidenceManifest) -> Tuple[SourceDelta, ...]:
    """Every movement between two EVIDENCE manifests.

    It takes evidence manifests, not `SourceManifest`, so acquisition
    provenance is STRUCTURALLY INACCESSIBLE here. That is stronger than
    remembering to compare `.identity`: a re-download cannot be reported as a
    release change because retrieval time is not reachable from this argument.
    """
    by_ref = {d.identity.source: d.identity for d in reference.dependencies}
    by_cand = {d.identity.source: d.identity for d in candidate.dependencies}
    out = []
    for source in sorted(set(by_ref) | set(by_cand)):
        a, b = by_ref.get(source), by_cand.get(source)
        if a is None:
            out.append(SourceDelta(source, SourceDeltaKind.SOURCE_ADDED,
                                   None, b))
            continue
        if b is None:
            out.append(SourceDelta(source, SourceDeltaKind.SOURCE_REMOVED,
                                   a, None))
            continue
        if a == b:
            continue
        if a.genome_build != b.genome_build:
            kind = SourceDeltaKind.GENOME_BUILD_CHANGED
        elif a.release_id == b.release_id and \
                a.artifact_sha256 != b.artifact_sha256:
            kind = SourceDeltaKind.ARTIFACT_CHANGED_UNDER_SAME_RELEASE
        elif a.release_id != b.release_id:
            kind = SourceDeltaKind.RELEASE_MOVED
        else:
            raise RuntimeError(
                "source identities differ in an UNCLASSIFIED way: {!r} vs "
                "{!r}. A field was added to SourceArtifactIdentity without a "
                "SourceDeltaKind, and it would otherwise be invisible to every "
                "comparison.".format(a, b))
        out.append(SourceDelta(source, kind, a, b))
    return tuple(out)


def differing_releases(reference: SourceManifest,
                       candidate: SourceManifest) -> Tuple[str, ...]:
    """Which sources moved, by name. Compatibility spelling.

    Delegates to `source_deltas` over the EVIDENCE manifests, so a re-download
    can never appear here.
    """
    return tuple(d.source for d in source_deltas(reference.evidence,
                                                 candidate.evidence))
