"""What mathematical space a feature matrix inhabits. Nothing else.

DRIFT-1 Phase 1B.1. Created 2026-08-27, replacing the 2026-08-27 original.

THE DEFECT THIS REPLACES
------------------------
The original carried `source_manifest_sha256`, and `assert_same_representation`
refused a comparison whose source manifests differed. MEASURED against the
installed code:

    reference   ClinVar 2026-07, dbNSFP 4.7a
    candidate   ClinVar 2026-08, dbNSFP 4.7a
    same plane, same feature names, same policy
        -> REFUSED

That is exactly the temporal comparison DRIFT-1 exists to make. The prose was
right -- an annotation release moving IS measurement-process drift -- but the
FIELD PLACEMENT was wrong: it collapsed a source-state difference into a
representation incompatibility, so the two could not be told apart at all.

The evidence state is FOUR independent things:

    P   population        WHICH ROWS          evaluation.population
    R   representation    WHAT SPACE          here
    T   transformation    WHAT SEMANTICS      drift.transformation
    S   source state      WHICH EVIDENCE      drift.source_release

A comparison must be able to express any combination of movements. The old
type could not express "same representation, different source state", which is
the single most common case in temporal drift.

WHAT THIS TYPE NOW IGNORES
--------------------------
    source state           moving it is measured by `source_deltas`
    population rows        owned by `EvaluationPopulation`
    acquisition provenance owned by `SourceRetrievalProvenance`

The type is now INCAPABLE of conflating them, which is stronger than supplying
two differing source digests and asserting they are ignored.

IDENTITY KERNEL
---------------
    equal iff   plane, ordered feature contract, transformation identity
    ignores     source state, population, acquisition provenance

ONE DIFFERENCE AUTHORITY
------------------------
`representation_differences` computes a COMPLETE typed description;
`assert_same_representation` is the strict adapter that raises on the first
non-empty result. Admission needs the whole delta -- a first-failure exception
discards the fact that three things moved -- while existing callers keep a
fail-closed primitive.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from genomic_variant_classifier.monitoring.drift.transformation import (
    TransformationIdentity,
    differing_components,
)

#: A feature name may not contain this, because it joins the contract digest.
_JOIN = "\n"


class RepresentationPlane(str, Enum):
    """WHICH space a comparison is made in.

    These answer different scientific questions and must never share a result
    object.
    """

    #: Engineered features as they MEAN something. Interpretable.
    SEMANTIC_TABULAR = "semantic_tabular"
    #: What the estimator consumes after imputation and scaling.
    MODEL_INPUT = "model_input"
    #: A learned latent space. Reveals joint shifts, least interpretable.
    LEARNED_REPRESENTATION = "learned_representation"


@dataclass(frozen=True)
class RepresentationIdentity:
    """The contract two feature frames must share to be comparable at all."""

    plane: RepresentationPlane
    feature_names: Tuple[str, ...]
    transformation: TransformationIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.plane, RepresentationPlane):
            raise ValueError(
                "plane is {!r}; a representation must name its plane from the "
                "declared vocabulary, because semantic and model-input drift "
                "are different questions".format(self.plane))
        names = self.feature_names
        if not isinstance(names, tuple):
            raise ValueError(
                "feature_names is {}; it must be a TUPLE, so the order is part "
                "of the identity and cannot be mutated after construction"
                .format(type(names).__name__))
        if not names:
            raise ValueError("a representation must enumerate its features")
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValueError("feature name {!r} is not a non-empty string"
                                 .format(name))
            if _JOIN in name:
                raise ValueError(
                    "feature name {!r} contains a newline, which is the digest "
                    "separator. Two different name tuples could then produce "
                    "one digest.".format(name))
        if len(set(names)) != len(names):
            duplicated = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                "feature names must be unique; duplicated: {}".format(duplicated))
        if not isinstance(self.transformation, TransformationIdentity):
            raise ValueError(
                "transformation is {!r}; a representation must state what "
                "semantics produced its values. Same columns computed "
                "differently are not the same space."
                .format(self.transformation))

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def feature_contract_digest(self) -> str:
        """A digest of the ORDERED names. DERIVED, never stored."""
        return hashlib.sha256(
            _JOIN.join(self.feature_names).encode("utf-8")).hexdigest()

    def describe(self) -> str:
        return "{plane} | {n} features | contract {contract} | {transform}".format(
            plane=self.plane.value, n=self.n_features,
            contract=self.feature_contract_digest[:12],
            transform=self.transformation.describe())


class RepresentationDeltaKind(str, Enum):
    """How two representations differ. Typed, not free-form strings."""

    PLANE = "plane"
    FEATURE_SET = "feature_set"
    FEATURE_ORDER = "feature_order"
    TRANSFORMATION = "transformation"


@dataclass(frozen=True)
class RepresentationDelta:
    """One named difference, with both sides and a rendered explanation."""

    kind: RepresentationDeltaKind
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepresentationDeltaKind):
            raise ValueError("kind is {!r}".format(self.kind))
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a delta must explain itself")


class RepresentationMismatch(ValueError):
    """Two frames do not inhabit one representation."""


def representation_differences(reference: RepresentationIdentity,
                               candidate: RepresentationIdentity
                               ) -> Tuple[RepresentationDelta, ...]:
    """EVERY difference, not the first.

    Admission needs a complete delta: a comparison where the plane matches, the
    features were reordered AND the join policy moved is three facts, and a
    first-failure exception reports one.
    """
    out = []
    if reference.plane is not candidate.plane:
        out.append(RepresentationDelta(
            RepresentationDeltaKind.PLANE,
            "reference {!r}, candidate {!r}. Semantic-feature drift and "
            "model-input drift are different questions and must not be "
            "compared.".format(reference.plane.value, candidate.plane.value)))
    if reference.feature_names != candidate.feature_names:
        if set(reference.feature_names) == set(candidate.feature_names):
            out.append(RepresentationDelta(
                RepresentationDeltaKind.FEATURE_ORDER,
                "the same {} features in a DIFFERENT ORDER. Column position "
                "carries meaning here; a same-width comparison would pair up "
                "the wrong features.".format(len(reference.feature_names))))
        else:
            missing = sorted(set(reference.feature_names)
                             - set(candidate.feature_names))
            extra = sorted(set(candidate.feature_names)
                           - set(reference.feature_names))
            out.append(RepresentationDelta(
                RepresentationDeltaKind.FEATURE_SET,
                "candidate is missing {} and adds {}".format(
                    missing[:8] or "nothing", extra[:8] or "nothing")))
    if reference.transformation != candidate.transformation:
        moved = differing_components(reference.transformation,
                                     candidate.transformation)
        out.append(RepresentationDelta(
            RepresentationDeltaKind.TRANSFORMATION,
            "component(s) {} moved. The same values computed differently are "
            "not the same observation.".format([k.value for k in moved])))
    return tuple(out)


def render_representation_differences(
        differences: Tuple[RepresentationDelta, ...]) -> str:
    return "\n".join("{}: {}".format(d.kind.value, d.detail)
                     for d in differences)


def assert_same_representation(reference: RepresentationIdentity,
                               candidate: RepresentationIdentity) -> None:
    """The strict adapter over the one difference authority.

    NOTE what it no longer checks: SOURCE STATE. A ClinVar release moving is
    measured by `source_deltas`, and treating it as a representation mismatch
    made the most common temporal comparison unexpressible.
    """
    differences = representation_differences(reference, candidate)
    if differences:
        raise RepresentationMismatch(
            render_representation_differences(differences))
