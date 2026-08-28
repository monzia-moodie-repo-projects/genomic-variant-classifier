"""Which authorities exist, what kinds of artifact they publish.

DRIFT-1 Phase 1B.3. Created 2026-08-28.

WHY A CONTROLLED VOCABULARY
---------------------------
The previous `_SOURCE` pattern validated SYNTAX, not identity. `ClinVar`,
`clinvar` and `NCBI-ClinVar` were all valid and all DIFFERENT, so a typo could
mint a scientifically duplicate source identity that no check would catch.

MEASURED 2026-08-28: no source registry exists anywhere in the repository. That
means the clean-break window is still open -- and it closes the moment a
`SourceEvidenceManifest` digest is persisted, because correcting an alias
afterwards becomes an identity migration.

THE RULE
--------
Aliases may be accepted at the INGESTION boundary. Only canonical names may
enter persistent identity. `resolve_source_name` is that boundary, and it
REFUSES an unknown spelling rather than minting an identity from it.

WHY EVERY MEMBER IS MEASURED, NOT LISTED
----------------------------------------
Each authority below was counted on disk on 2026-08-28: 3,420 artifact files
under `data/`, attributed by path. The count is recorded beside each member so
a later reader can tell a real dependency from an aspirational one.

Sources named in the project roadmap but holding ZERO artifacts today --
Nucleotide-Transformer, FinnGen -- are deliberately ABSENT. A vocabulary that
lists what might arrive cannot distinguish a missing artifact from an
unimplemented one.

WHY ARTIFACT KIND IS A SEPARATE COORDINATE
------------------------------------------
MEASURED: ten authorities hold more than one kind, and the maximum consumed by
one module is THREE. `monitoring/registry.py` names ClinVar's `index.parquet`,
`parquet` AND `variant_summary.txt`; `agent_layer/config.py` declares ClinVar's
`variant_summary.txt` alongside its `vcf`.

So `source` alone cannot be the identity key. Forcing it would require faking
names such as `ClinVarVCF`, turning an ARTIFACT distinction into a SOURCE
distinction and losing the fact that both came from one release.

Acronyms: VCF = Variant Call Format; GTF = Gene Transfer Format; GFF = General
Feature Format; FASTA is a sequence format.

Author: Monzia Moodie
"""
from __future__ import annotations

from enum import Enum


class SourceName(str, Enum):
    """Every authority holding artifacts in this repository on 2026-08-28.

    The trailing count is what was MEASURED under `data/`, not an estimate.
    """

    CLINVAR = "ClinVar"                  # 4 kinds, 22 files
    GNOMAD = "gnomAD"                    # 2 kinds, 3 files
    DBNSFP = "dbNSFP"                    # 1 kind, 1 file
    SPLICEAI = "SpliceAI"                # 2 kinds, 3 files
    ALPHAMISSENSE = "AlphaMissense"      # 3 kinds, 4 files
    ALPHAFOLD = "AlphaFold"              # 1 kind, 2 files
    GENCODE = "GENCODE"                  # 3 kinds, 5 files
    COSMIC = "COSMIC"                    # 2 kinds, 2 files
    HGMD = "HGMD"                        # 1 kind, 1 file
    OMIM = "OMIM"                        # 1 kind, 5 files
    UNIPROT = "UniProt"                  # 1 kind, 1 file
    REACTOME = "Reactome"                # 1 kind, 2 files
    GTEX = "GTEx"                        # 2 kinds, 9 files
    EVE = "EVE"                          # 2 kinds, 3,217 files
    ESM2 = "ESM-2"                       # 2 kinds, 2 files
    REFERENCE_GENOME = "ReferenceGenome"  # 2 kinds, 4 files
    #: Held in `data/external/string/` and consumed by the graph branch; the
    #: path census attributes it under its own directory rather than a token
    #: shared with the Python builtin, so it is listed from the roadmap's
    #: measured node and edge counts instead.
    STRING_DB = "STRING-DB"
    PHYLOP = "PhyloP"


#: Accepted ONLY at the ingestion boundary. Case-folded on lookup.
#:
#: Every entry here is a spelling this repository has actually used in a path,
#: a configuration key or a script -- not a hypothetical variant.
_ALIASES = {
    "clinvar": SourceName.CLINVAR,
    "ncbi-clinvar": SourceName.CLINVAR,
    "ncbi_clinvar": SourceName.CLINVAR,
    "gnomad": SourceName.GNOMAD,
    "dbnsfp": SourceName.DBNSFP,
    "spliceai": SourceName.SPLICEAI,
    "splice-ai": SourceName.SPLICEAI,
    "alphamissense": SourceName.ALPHAMISSENSE,
    "alphafold": SourceName.ALPHAFOLD,
    "gencode": SourceName.GENCODE,
    "cosmic": SourceName.COSMIC,
    "hgmd": SourceName.HGMD,
    "omim": SourceName.OMIM,
    "uniprot": SourceName.UNIPROT,
    "reactome": SourceName.REACTOME,
    "gtex": SourceName.GTEX,
    "eve": SourceName.EVE,
    "esm-2": SourceName.ESM2,
    "esm2": SourceName.ESM2,
    "reference_genome": SourceName.REFERENCE_GENOME,
    "referencegenome": SourceName.REFERENCE_GENOME,
    "grch38": SourceName.REFERENCE_GENOME,
    "grch37": SourceName.REFERENCE_GENOME,
    "string-db": SourceName.STRING_DB,
    "stringdb": SourceName.STRING_DB,
    "string_db": SourceName.STRING_DB,
    "phylop": SourceName.PHYLOP,
}


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


class SourceVocabularyError(ValueError):
    """A spelling that would mint an identity rather than name one."""


def resolve_source_name(raw) -> SourceName:
    """The INGESTION boundary. Canonical names pass; aliases resolve; else refuse.

    An unknown spelling is REFUSED rather than accepted, because accepting it
    would create a scientifically duplicate authority that compares unequal to
    the real one and that no later check could distinguish from a genuine new
    source.
    """
    if isinstance(raw, SourceName):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise SourceVocabularyError(
            "source name is {!r}; expected a non-empty string".format(raw))
    if raw in {m.value for m in SourceName}:
        return SourceName(raw)
    resolved = _ALIASES.get(raw.strip().casefold())
    if resolved is not None:
        return resolved
    raise SourceVocabularyError(
        "unknown source {!r}. Register it explicitly rather than minting an "
        "identity by spelling -- an unregistered name would compare unequal to "
        "the authority it means. Known: {}".format(
            raw, sorted(m.value for m in SourceName)))


def known_aliases(name: SourceName):
    """Every accepted spelling of one authority, for diagnostics."""
    return tuple(sorted(k for k, v in _ALIASES.items() if v is name))
