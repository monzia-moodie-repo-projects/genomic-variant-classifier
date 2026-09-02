"""What KINDS of artifact this project reads.

DRIFT-1. Created 2026-08-28; the authority vocabulary was removed 2026-08-29.

WHAT THIS MODULE NO LONGER DECLARES
-----------------------------------
It declared `SourceName`, `_ALIASES`, `resolve_source_name`, `known_aliases`
and `SourceVocabularyError`, on the stated basis that "no source registry
exists anywhere in the repository".

THAT MEASUREMENT WAS WRONG. `configs/data_manifest.yaml` calls itself the
"Canonical registry of every data source under data/" on its own THIRD LINE,
declares 32 sources, and is read by five scripts under `scripts/maintenance/`.
The authority search that missed it looked only at Python files.

MEASURED 2026-08-29, the enum against the manifest:

    declared sources     32      SourceName members    18
    it cannot name       16      aliases it accepted    0
    declared nowhere      2      aliases it invented   26

Four of the sixteen it could not name are `irreplaceable` and constrained:
`tcga` and `topmed` are `controlled`, `rnaseq` and `validation_cohort` are
`review`. A vocabulary that cannot name `tcga` cannot express a manifest
containing it, so `SourceEvidenceManifest` would have refused a governed source
because of a missing enum member rather than a scientific judgement.

Authority naming now belongs to
`genomic_variant_classifier.data.source_registry`, which reads the manifest,
types every field, records the path it read, and RAISES rather than defaulting
-- one cannot invent 32 declarations.

WHY `ArtifactKind` STAYS
------------------------
It is not in the manifest, and nothing else declares it. The manifest declares
`location`, `tier`, `class`, `aliases`, `version`, `acquire`, `regenerate`,
`sync` and `notes` -- nothing about Variant Call Format versus parquet versus
FASTA. `ArtifactKind` is a genuine LOCAL vocabulary with no external authority,
so removing it would create a gap rather than close a duplication.

MEASURED on disk 2026-08-28: ten authorities hold more than one artifact kind,
and one module names THREE distinct ClinVar artifacts. That is why the kind is
part of the identity key at all.

Acronyms: VCF = Variant Call Format; GTF = Gene Transfer Format; GFF = General
Feature Format; FASTA is a sequence format.

Author: Monzia Moodie
"""
from __future__ import annotations

from enum import Enum


class ArtifactKind(str, Enum):
    """WHAT an artifact is, semantically. One authority may publish several.

    MEASURED on disk 2026-08-28. Each member corresponds to a kind actually
    observed, not a format that might appear.
    """

    #: The authority's own primary release file, as published.
    PRIMARY_RELEASE = "primary_release"
    #: A join-ready index this project derives from a primary release.
    DERIVED_INDEX = "derived_index"
    #: Variant Call Format, as published.
    VCF = "vcf"
    #: ClinVar's tab-separated variant summary.
    VARIANT_SUMMARY = "variant_summary"
    #: Gene annotation, GTF flavour.
    ANNOTATION_GTF = "annotation_gtf"
    #: Gene annotation, GFF3 flavour. NOT interchangeable with GTF.
    ANNOTATION_GFF3 = "annotation_gff3"
    #: Nucleotide or protein sequences.
    SEQUENCE_FASTA = "sequence_fasta"
    #: Per-gene or per-transcript constraint statistics.
    CONSTRAINT_TABLE = "constraint_table"
    #: Genome-wide per-base scores in a binary interval format.
    SCORE_TRACK = "score_track"
    #: Interaction or pathway edges.
    NETWORK_EDGES = "network_edges"
