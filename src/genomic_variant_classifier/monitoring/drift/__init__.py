"""Typed substrate for temporal evidence comparison.

DRIFT-1. Phase 1B created the identity types; 1B.1 decomposed them after two
defects were reproduced; 1B.3 corrected the source kernel after four more were
measured.

THE EVIDENCE STATE IS FOUR INDEPENDENT THINGS
---------------------------------------------
    P  population       WHICH ROWS       evaluation.population -- and
                                         CanonicalVariantTable owns the ordered
                                         row-universe identity its indices
                                         address. NOT duplicated here.
    R  representation   WHAT SPACE       drift.representation
    T  transformation   WHAT SEMANTICS   drift.transformation
    S  source state     WHICH EVIDENCE   drift.source_release

WHAT 1B.3 CORRECTED, EACH MEASURED FIRST
----------------------------------------
    one artifact per source     FALSE: 10 authorities hold several kinds and
                                one module consumes THREE ClinVar artifacts
    mandatory genome build      FALSE for 6 of 16 authorities
    free-form source names      three spellings of ClinVar were three
                                identities. The 2026-08-28 repair replaced the
                                pattern with an INVENTED enum, on the false
                                basis that "no registry existed" -- see below

WHAT 1B.5 CORRECTED, 2026-08-29
-------------------------------
`configs/data_manifest.yaml` calls itself the "Canonical registry of every data
source under data/" on its own third line and declares 32 sources. The enum
named 18, could not name 16 -- including `tcga` and `topmed`, both controlled
and irreplaceable -- and refused all 8 declared aliases while carrying 26
invented ones.

`SourceArtifactKey.source` is now a validated STRING. Registry membership is an
ADMISSION question answered by
`genomic_variant_classifier.data.source_registry`, so identity stays
constructible without a readable file
    role change without delta   the digest moved and nothing attributed it
    precedence-based deltas     three facts moved, one was reported

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.monitoring.drift.coordinate import (
    GENOME_ASSEMBLIES,
    CoordinateContext,
    CoordinateContextKind,
    CoordinateError,
    assemblies_in,
)
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
    SourceDeltaKind,
    SourceTransition,
    differing_releases,
    source_transitions,
)
from genomic_variant_classifier.monitoring.drift.source_release import (
    SourceAcquisition,
    SourceArtifactIdentity,
    SourceArtifactKey,
    SourceDependency,
    SourceError,
    SourceEvidenceManifest,
    SourceManifest,
    SourceRetrievalProvenance,
    SourceRole,
)
from genomic_variant_classifier.monitoring.drift.source_vocabulary import (
    ArtifactKind,
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
    "ArtifactKind",
    "CoordinateContext",
    "CoordinateContextKind",
    "CoordinateError",
    "GENOME_ASSEMBLIES",
    "RepresentationDelta",
    "RepresentationDeltaKind",
    "RepresentationIdentity",
    "RepresentationMismatch",
    "RepresentationPlane",
    "SourceAcquisition",
    "SourceArtifactIdentity",
    "SourceArtifactKey",
    "SourceDeltaKind",
    "SourceDependency",
    "SourceError",
    "SourceEvidenceManifest",
    "SourceManifest",
    "SourceRetrievalProvenance",
    "SourceRole",
    "SourceTransition",
    "TransformationComponent",
    "TransformationComponentKind",
    "TransformationError",
    "TransformationIdentity",
    "assemblies_in",
    "assert_same_representation",
    "differing_components",
    "differing_releases",
    "render_representation_differences",
    "representation_differences",
    "source_transitions",
]
