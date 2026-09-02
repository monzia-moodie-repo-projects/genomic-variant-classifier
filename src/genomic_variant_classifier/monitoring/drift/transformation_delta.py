"""What moved between two transformation identities.

Phase 1C Unit 3A. Created 2026-09-02.

PROVENANCE DEFINES STATES; MONITORING COMPARES THEM.
----------------------------------------------------
`TransformationIdentity` and its components describe WHAT a computation is.
That is a scientific identity, and it moved to
`genomic_variant_classifier.provenance.transformation` so that connectors,
pipelines and evaluation can name a transformation without importing a
monitoring package.

`differing_components` answers a different question -- what CHANGED between a
reference and a candidate -- and that is drift monitoring. It stays here.

The rule is the same one that keeps `source_delta` in monitoring while
`SourceArtifactIdentity` moves down: an identity is a fact about one thing;
a delta is a judgement about two.

This module holds no definitions of its own beyond that comparison, and it
imports the identities from their canonical owner rather than redeclaring
them -- there is exactly ONE `TransformationIdentity` class object in the
process.

Author: Monzia Moodie
"""
from __future__ import annotations

from typing import Tuple

from genomic_variant_classifier.provenance.transformation import (
    TransformationComponentKind,
    TransformationIdentity,
)

__all__ = ["differing_components"]


def differing_components(reference: TransformationIdentity,
                         candidate: TransformationIdentity
                         ) -> Tuple[TransformationComponentKind, ...]:
    """Which aspects moved. Named, so a refusal can say WHAT changed.

    "the transformation differs" sends a reader to diff two objects;
    "the join policy moved" is a statement about the pipeline.
    """
    ref = {c.kind: c for c in reference.components}
    cand = {c.kind: c for c in candidate.components}
    moved = []
    for kind in sorted(set(ref) | set(cand), key=lambda k: k.value):
        if kind not in ref or kind not in cand or ref[kind] != cand[kind]:
            moved.append(kind)
    return tuple(moved)
