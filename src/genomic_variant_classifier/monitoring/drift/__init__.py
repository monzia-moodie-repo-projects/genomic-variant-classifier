"""Typed substrate for temporal evidence comparison.

DRIFT-1 Phase 1B. Created 2026-08-27.

Each layer owns exactly one fact, and no downstream statement may be
manufactured from the absence of an upstream result:

    capability  !=  observation  !=  comparability  !=  measurement  !=  decision

This package currently holds the two identity types Phase 1B establishes.
`representation` says WHAT the columns are; `source_release` says WHICH
releases produced them. Population identity -- WHICH ROWS -- is owned by
`evaluation.population` and is deliberately not duplicated here.

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.monitoring.drift.representation import (
    RepresentationIdentity,
    RepresentationMismatch,
    RepresentationPlane,
    assert_same_representation,
)
from genomic_variant_classifier.monitoring.drift.source_release import (
    GENOME_BUILDS,
    SourceManifest,
    SourceManifestError,
    SourceRelease,
    differing_releases,
)

__all__ = [
    "GENOME_BUILDS",
    "RepresentationIdentity",
    "RepresentationMismatch",
    "RepresentationPlane",
    "SourceManifest",
    "SourceManifestError",
    "SourceRelease",
    "assert_same_representation",
    "differing_releases",
]
