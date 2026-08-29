"""What moved between two source states, and EVERY kind of movement.

DRIFT-1 Phase 1B.3. Created 2026-08-28, replacing the 2026-08-27 version.

TWO DEFECTS, BOTH CONFIRMED BY MEASUREMENT AGAINST THE 2026-08-27 CODE
----------------------------------------------------------------------
A ROLE CHANGE MOVED THE DIGEST AND PRODUCED NO DELTA.

    reference roles {observation}
    candidate roles {observation, label}
    same artifact identity   True
    digests differ           True
    source_deltas            ()

`SourceDependency.as_record()` includes roles, so the manifest identity moved --
and `source_deltas` reduced dependencies to identities, so nothing attributed
it. Every identity movement must have an attributable delta, and that one did
not.

THE DELTA WAS PRECEDENCE-BASED, NOT COMPLETE.

    reference   ClinVar 2026-08 GRCh38 aaaa...
    candidate   ClinVar 2026-09 GRCh37 bbbb...
    THREE facts moved; ONE was reported: genome_build_changed

Branch ORDER became scientific interpretation. `representation_differences`
returns EVERY difference and is tested for it; this returned one. The two
halves of the same package answered the same question differently, and both
were written on the same day.

WHAT REPLACES THEM
------------------
`SourceTransition` carries a FROZENSET of changes, so a release move that is
also an assembly change is two facts rather than a precedence winner. Roles are
compared, so a role change is `ROLE_CHANGED` rather than silence.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional, Tuple

from genomic_variant_classifier.monitoring.drift.source_release import (
    SourceArtifactIdentity,
    SourceArtifactKey,
    SourceDependency,
    SourceEvidenceManifest,
    SourceManifest,
)


class SourceDeltaKind(str, Enum):
    """How one artifact's evidence or use differs between two states."""

    #: Present in the candidate only.
    ARTIFACT_ADDED = "artifact_added"
    #: Present in the reference only.
    ARTIFACT_REMOVED = "artifact_removed"
    #: A different named release. Ordinary temporal movement.
    RELEASE_MOVED = "release_moved"
    #: SAME release label, DIFFERENT bytes. Never ordinary: an upstream
    #: replacement, a corrupted download, or non-reproducible packaging.
    ARTIFACT_CHANGED_UNDER_SAME_RELEASE = "artifact_changed_under_same_release"
    #: The coordinate system moved. Coordinates are no longer comparable.
    COORDINATE_CONTEXT_CHANGED = "coordinate_context_changed"
    #: The same artifact is used for a different purpose. It moves the
    #: manifest digest, so it must move a delta too.
    ROLE_CHANGED = "role_changed"


@dataclass(frozen=True)
class SourceTransition:
    """One artifact key, and EVERY way it moved.

    A frozenset, not a single kind: a release move that is also an assembly
    change is two facts, and letting branch order pick one makes the reporting
    order into an interpretation.
    """

    key: SourceArtifactKey
    reference: Optional[SourceDependency]
    candidate: Optional[SourceDependency]
    changes: FrozenSet[SourceDeltaKind]

    def __post_init__(self) -> None:
        if not isinstance(self.key, SourceArtifactKey):
            raise ValueError("key is {!r}".format(self.key))
        if not isinstance(self.changes, frozenset) or not self.changes:
            raise ValueError(
                "a transition must name at least one change; an empty set "
                "describes no movement and should not have been constructed")
        for c in self.changes:
            if not isinstance(c, SourceDeltaKind):
                raise ValueError("change {!r} is not a SourceDeltaKind"
                                 .format(c))
        added = SourceDeltaKind.ARTIFACT_ADDED in self.changes
        removed = SourceDeltaKind.ARTIFACT_REMOVED in self.changes
        if added and removed:
            raise ValueError("an artifact cannot be both added and removed")
        if added:
            if self.reference is not None or self.candidate is None:
                raise ValueError(
                    "ARTIFACT_ADDED requires no reference and a candidate")
            if self.changes != frozenset({SourceDeltaKind.ARTIFACT_ADDED}):
                raise ValueError(
                    "an added artifact has nothing to have moved FROM; {} "
                    "describes a change between two states"
                    .format(sorted(c.value for c in self.changes)))
        elif removed:
            if self.reference is None or self.candidate is not None:
                raise ValueError(
                    "ARTIFACT_REMOVED requires a reference and no candidate")
            if self.changes != frozenset({SourceDeltaKind.ARTIFACT_REMOVED}):
                raise ValueError("a removed artifact has nothing to move TO")
        elif self.reference is None or self.candidate is None:
            raise ValueError(
                "{} describes a CHANGE and requires both sides"
                .format(sorted(c.value for c in self.changes)))

    @property
    def source(self):
        return self.key.source

    def describe(self) -> str:
        name = self.key.describe()
        if SourceDeltaKind.ARTIFACT_ADDED in self.changes:
            return "{}: added at {}".format(name,
                                            self.candidate.identity.release_id)
        if SourceDeltaKind.ARTIFACT_REMOVED in self.changes:
            return "{}: removed (was {})".format(
                name, self.reference.identity.release_id)
        parts = []
        a, b = self.reference.identity, self.candidate.identity
        for change in sorted(self.changes, key=lambda c: c.value):
            if change is SourceDeltaKind.RELEASE_MOVED:
                parts.append("release {} -> {}".format(a.release_id,
                                                       b.release_id))
            elif change is SourceDeltaKind.ARTIFACT_CHANGED_UNDER_SAME_RELEASE:
                parts.append("release {} UNCHANGED but bytes differ, {} -> {}"
                             .format(a.release_id, a.artifact_sha256[:12],
                                     b.artifact_sha256[:12]))
            elif change is SourceDeltaKind.COORDINATE_CONTEXT_CHANGED:
                parts.append("coordinates {} -> {}".format(
                    a.coordinate_context.describe(),
                    b.coordinate_context.describe()))
            elif change is SourceDeltaKind.ROLE_CHANGED:
                parts.append("roles {} -> {}".format(
                    sorted(r.value for r in self.reference.roles),
                    sorted(r.value for r in self.candidate.roles)))
        return "{}: {}".format(name, "; ".join(parts))


def source_transitions(reference: SourceEvidenceManifest,
                       candidate: SourceEvidenceManifest
                       ) -> Tuple[SourceTransition, ...]:
    """Every movement between two EVIDENCE manifests, keyed by ARTIFACT.

    It takes evidence manifests, not `SourceManifest`, so acquisition
    provenance is STRUCTURALLY INACCESSIBLE: a re-download cannot be reported
    as a release change because retrieval time is not reachable here.
    """
    by_ref = {d.key: d for d in reference.dependencies}
    by_cand = {d.key: d for d in candidate.dependencies}
    out = []
    for key in sorted(set(by_ref) | set(by_cand), key=lambda k: k.canonical_key):
        a, b = by_ref.get(key), by_cand.get(key)
        if a is None:
            out.append(SourceTransition(
                key, None, b, frozenset({SourceDeltaKind.ARTIFACT_ADDED})))
            continue
        if b is None:
            out.append(SourceTransition(
                key, a, None, frozenset({SourceDeltaKind.ARTIFACT_REMOVED})))
            continue
        if a == b:
            continue
        changes = set()
        ai, bi = a.identity, b.identity
        if ai.release_id != bi.release_id:
            changes.add(SourceDeltaKind.RELEASE_MOVED)
        if ai.coordinate_context != bi.coordinate_context:
            changes.add(SourceDeltaKind.COORDINATE_CONTEXT_CHANGED)
        if ai.artifact_sha256 != bi.artifact_sha256 and \
                ai.release_id == bi.release_id:
            changes.add(SourceDeltaKind.ARTIFACT_CHANGED_UNDER_SAME_RELEASE)
        if a.roles != b.roles:
            changes.add(SourceDeltaKind.ROLE_CHANGED)
        if not changes:
            raise RuntimeError(
                "dependencies differ in an UNCLASSIFIED way: {!r} vs {!r}. A "
                "field was added without a corresponding SourceDeltaKind, and "
                "it would otherwise be invisible to every comparison."
                .format(a, b))
        out.append(SourceTransition(key, a, b, frozenset(changes)))
    return tuple(out)


def differing_releases(reference: SourceManifest,
                       candidate: SourceManifest) -> Tuple[str, ...]:
    """Which AUTHORITIES moved, by name. Compatibility spelling.

    Deduplicated: one authority may contribute several artifacts, so two
    transitions can name one source. Reporting it twice would suggest two
    authorities moved.
    """
    seen = []
    for t in source_transitions(reference.evidence, candidate.evidence):
        if t.source not in seen:
            seen.append(t.source)
    return tuple(seen)
