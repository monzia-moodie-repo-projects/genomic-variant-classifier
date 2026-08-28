"""Typed substrate for temporal evidence comparison.

DRIFT-1. Phase 1B created the first identity types; Phase 1B.1 decomposed them
after two defects were reproduced against the installed code on 2026-08-27.

THE EVIDENCE STATE IS FOUR INDEPENDENT THINGS
---------------------------------------------
    P   population        WHICH ROWS       evaluation.population
    R   representation    WHAT SPACE       drift.representation
    T   transformation    WHAT SEMANTICS   drift.transformation
    S   source state      WHICH EVIDENCE   drift.source_release

A drift comparison is a statement about which of them moved. The first version
of this package folded S into R, so "same representation, different source
state" -- the single most common temporal case -- could not be expressed at
all: `assert_same_representation` refused it.

It also folded acquisition into evidence, so two byte-identical downloads of
one release produced different manifest digests, and `differing_releases`
reported a re-download as a release change.

WHAT EACH LAYER MAY NOT DO
--------------------------
No layer may author a fact owned by another. `representation` cannot see source
state; `source_delta` takes EVIDENCE manifests so retrieval time is
structurally unreachable; population identity stays with
`EvaluationPopulation` and is not duplicated here.

Admission -- whether two evidence views may be compared under a protocol --
belongs to a later unit. This package establishes the coordinate axes; it does
not reason over them.

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.monitoring.drift.representation import (
    RepresentationDelta,
    RepresentationDeltaKind,
    RepresentationIdentity,
    RepresentationMismatch,
    RepresentationPlane,
    assert_same_representation,
    render_representation_differences,
    representation_differences,
)
from genomic_variant_classifier.monitoring.drift.source_delta import (
    SourceDelta,
    SourceDeltaKind,
    differing_releases,
    source_deltas,
)
from genomic_variant_classifier.monitoring.drift.source_release import (
    GENOME_BUILDS,
    SourceAcquisition,
    SourceArtifactIdentity,
    SourceDependency,
    SourceError,
    SourceEvidenceManifest,
    SourceManifest,
    SourceRetrievalProvenance,
    SourceRole,
)
from genomic_variant_classifier.monitoring.drift.transformation import (
    TransformationComponent,
    TransformationComponentKind,
    TransformationError,
    TransformationIdentity,
    differing_components,
)

#: Domain concepts only. Serialization machinery -- `_digest` -- stays private.
__all__ = [
    "GENOME_BUILDS",
    "RepresentationDelta",
    "RepresentationDeltaKind",
    "RepresentationIdentity",
    "RepresentationMismatch",
    "RepresentationPlane",
    "SourceAcquisition",
    "SourceArtifactIdentity",
    "SourceDelta",
    "SourceDeltaKind",
    "SourceDependency",
    "SourceError",
    "SourceEvidenceManifest",
    "SourceManifest",
    "SourceRetrievalProvenance",
    "SourceRole",
    "TransformationComponent",
    "TransformationComponentKind",
    "TransformationError",
    "TransformationIdentity",
    "assert_same_representation",
    "differing_components",
    "differing_releases",
    "render_representation_differences",
    "representation_differences",
    "source_deltas",
]
