"""Scientific evidence provenance.

Phase 1C. Created 2026-09-02.

This package answers ONE question:

    What exact scientific evidence demonstrably contributed to this
    computation, in what materialized form, through what transformation
    lineage, under what coordinate and release context, and with what
    completeness guarantee?

NOT what existed on disk, NOT what was configured, NOT what was instantiated,
and NOT what a capture layer happened to notice. Four candidate attachment
points were measured and rejected before this package was created:

    every read site        93 sites in 53 modules -- too dispersed
    digest-computing       5 of 36 modules, two source families, and NO
      loaders              principal training input among them
    AnnotationConfig       `vep_path` is read NOWHERE, and line 942 reads
                           `vep = VEPConnector()` with no path argument
    BaseConnector          `_load_cache` opens the project's own parquet
                           CACHE, not a publisher artifact; `fetch` is
                           abstract; 15 of 32 connectors do not inherit it

So the abstraction is introduced DELIBERATELY rather than discovered. The
seam is semantic -- evidence resolution -- not mechanical, because scientific
evidence arrives as local files, compressed archives, remote responses,
derived caches, generated indices and sharded collections, and filesystem
access is an implementation detail of only some of those.

DEPENDENCY DIRECTION
--------------------
    SourceRegistry -> provenance -> connectors -> pipelines -> evaluation
                                                            -> monitoring

Core provenance value objects depend on very little, and NOTHING here imports
from `data`, `models`, `training` or `monitoring`. `test_provenance_hashing`
asserts that by parsing, so the direction cannot invert silently.

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.provenance.artifact import ArtifactKind
from genomic_variant_classifier.provenance.coordinate import (
    GENOME_ASSEMBLIES,
    CoordinateContext,
    CoordinateContextKind,
    CoordinateError,
    assemblies_in,
)
from genomic_variant_classifier.provenance.hashing import (
    FileChangedDuringDigest,
    FileDigest,
    digest_file,
)
from genomic_variant_classifier.provenance.serialization import (
    canonical_json,
    domain_digest,
)
from genomic_variant_classifier.provenance.source import (
    EVIDENCE_DOMAIN,
    SourceAcquisition,
    SourceArtifactIdentity,
    SourceArtifactKey,
    SourceDependency,
    SourceError,
    SourceEvidenceManifest,
    SourceIdentityError,
    SourceManifest,
    SourceRetrievalProvenance,
    SourceRole,
)
from genomic_variant_classifier.provenance.transformation import (
    TRANSFORMATION_DOMAIN,
    TransformationComponent,
    TransformationComponentKind,
    TransformationError,
    TransformationIdentity,
)

#: OUTWARD-FACING ONLY. Internal provenance modules import each other by their
#: own paths -- `from genomic_variant_classifier.provenance.coordinate import
#: CoordinateContext` -- never through this file. Importing through the
#: package `__init__` would make every module depend on the whole export
#: surface and create hidden cycles as that surface grows.
__all__ = [
    "ArtifactKind",
    "CoordinateContext",
    "CoordinateContextKind",
    "CoordinateError",
    "EVIDENCE_DOMAIN",
    "FileChangedDuringDigest",
    "FileDigest",
    "GENOME_ASSEMBLIES",
    "SourceAcquisition",
    "SourceArtifactIdentity",
    "SourceArtifactKey",
    "SourceDependency",
    "SourceError",
    "SourceEvidenceManifest",
    "SourceIdentityError",
    "SourceManifest",
    "SourceRetrievalProvenance",
    "SourceRole",
    "TRANSFORMATION_DOMAIN",
    "TransformationComponent",
    "TransformationComponentKind",
    "TransformationError",
    "TransformationIdentity",
    "assemblies_in",
    "canonical_json",
    "digest_file",
    "domain_digest",
]
