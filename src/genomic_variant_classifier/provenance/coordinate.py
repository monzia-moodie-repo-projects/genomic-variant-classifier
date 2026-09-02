"""Whether an artifact inhabits a genomic coordinate system at all.

DRIFT-1 Phase 1B.3. Created 2026-08-28.

WHY THIS EXISTS
---------------
`SourceArtifactIdentity` required `genome_build` to be GRCh37 or GRCh38, and
`SourceEvidenceManifest` refused a mixed manifest. That is correct for
coordinate-bearing genomic artifacts and WRONG as a universal source property.

MEASURED 2026-08-28 across the sixteen authorities holding artifacts:

    coordinate-bearing   ClinVar, gnomAD, dbNSFP, SpliceAI, AlphaMissense,
                         GENCODE, COSMIC, PhyloP, ReferenceGenome
    build-independent    UniProt, Reactome, OMIM, STRING-DB, AlphaFold, ESM-2

A UniProt accession, a Reactome pathway and a STRING-DB interaction edge do not
sit on GRCh37 or GRCh38. Requiring an assembly for them would force a lie.

WHY NOT `genome_build: Optional[str]`
-------------------------------------
That is the nullable-union defect this programme has removed three times. A
field that is sometimes absent becomes a field nobody supplies, and `None`
would then mean BOTH "this evidence has no coordinate system" and "nobody
recorded one" -- two states that must never be confused, because the first is a
fact and the second is a gap.

A typed context makes the distinction structural: `BUILD_INDEPENDENT` is a
POSITIVE claim, and it REFUSES an identifier rather than ignoring one.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

#: The assemblies this project admits. GRCh37 and GRCh38 coordinates are NOT
#: interchangeable: one position denotes different loci under each.
GENOME_ASSEMBLIES = ("GRCh37", "GRCh38")


class CoordinateContextKind(str, Enum):
    """WHETHER an artifact's records carry genomic coordinates."""

    #: Records are addressed by assembly, chromosome and position.
    GENOMIC_ASSEMBLY = "genomic_assembly"
    #: Records are addressed by an accession, an identifier or a term, and
    #: carry no genomic position. A POSITIVE claim, not a missing value.
    BUILD_INDEPENDENT = "build_independent"


class CoordinateError(ValueError):
    """A coordinate context that describes no coordinate system."""


@dataclass(frozen=True)
class CoordinateContext:
    """The coordinate system an artifact's records inhabit, or its absence."""

    kind: CoordinateContextKind
    identifier: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CoordinateContextKind):
            raise CoordinateError(
                "kind is {!r}; an artifact must state whether it carries "
                "genomic coordinates".format(self.kind))
        if self.kind is CoordinateContextKind.GENOMIC_ASSEMBLY:
            if self.identifier not in GENOME_ASSEMBLIES:
                raise CoordinateError(
                    "identifier is {!r}; a genomic assembly must be one of {}. "
                    "GRCh37 and GRCh38 coordinates are NOT interchangeable, "
                    "and comparing across them would pair unrelated loci."
                    .format(self.identifier, list(GENOME_ASSEMBLIES)))
        elif self.identifier is not None:
            raise CoordinateError(
                "build-independent evidence carries identifier {!r}. It must "
                "not INVENT an assembly: a UniProt accession or a Reactome "
                "pathway has no genomic position, and recording one would be "
                "a claim about coordinates that do not exist."
                .format(self.identifier))

    @classmethod
    def assembly(cls, identifier: str) -> "CoordinateContext":
        return cls(kind=CoordinateContextKind.GENOMIC_ASSEMBLY,
                   identifier=identifier)

    @classmethod
    def build_independent(cls) -> "CoordinateContext":
        return cls(kind=CoordinateContextKind.BUILD_INDEPENDENT)

    @property
    def is_genomic(self) -> bool:
        return self.kind is CoordinateContextKind.GENOMIC_ASSEMBLY

    def as_record(self) -> dict:
        return {"kind": self.kind.value, "identifier": self.identifier}

    def describe(self) -> str:
        return self.identifier if self.is_genomic else "build-independent"


def assemblies_in(contexts) -> frozenset:
    """Every distinct assembly among coordinate-bearing contexts.

    Build-independent evidence contributes NOTHING here -- which is the whole
    point: it may coexist with any assembly, because it has no coordinates to
    conflict.
    """
    return frozenset(c.identifier for c in contexts if c.is_genomic)
