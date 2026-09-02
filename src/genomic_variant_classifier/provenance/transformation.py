"""What semantics turned source evidence into feature values.

DRIFT-1 Phase 1B.1. Created 2026-08-27.

WHY THIS IS NOT ONE FINGERPRINT
-------------------------------
`RepresentationIdentity` previously carried `preprocessing_policy_sha256`,
holding `policy_fingerprint(DECLARED_MISSINGNESS)`. That digest is real and
useful -- a pure function of a declared policy mapping, with no fitted state --
but it identifies MISSINGNESS SEMANTICS ONLY.

Same features, same missingness policy and same sources do not prove:

    same feature engineering        same joins
    same normalization              same unit conversion
    same consequence mapping        same aggregation semantics

So a single field named for the whole transformation would be a claim the
digest cannot support. Transformation is COMPOSITIONAL: each component names a
kind, carries its own schema version, and contributes its own fingerprint.

WHY NOT HASH THE SOURCE CODE
----------------------------
`sha256(inspect.getsource(engineer_features))` is tempting and wrong in both
directions. A comment change, a rename, a semantically identical refactor or a
module move would each mint a new transformation identity while the science is
unchanged. And behaviour can depend on external configuration while the source
text stays constant.

Transformation identity fingerprints DECLARATIONS OF SEMANTICS, never Python
source text.

WHAT A COMPONENT IS
-------------------
A component is a declaration that some named aspect of the pipeline behaves a
particular way, plus the digest of that declaration. This module does not
compute those digests: each aspect owns its own, exactly as
`model_preprocessing.policy_fingerprint` owns missingness.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

from genomic_variant_classifier.provenance.digest_schema import (
    CanonicalDigestSchema,
)

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

#: Bumped whenever the canonical record shape changes, never for content.
#: ONE authority for this identity epoch. The domain and the record's
#: `schema_version` are both DERIVED from `version`, so they cannot drift
#: apart -- which is exactly what happened to source evidence, where the
#: domain reached v4 while the embedded literal stayed at 3.
#:
#: SEMANTIC-ZERO by construction: family + version reproduce the previous
#: literal domain EXACTLY, and the stamped record reproduces the previous
#: payload EXACTLY. Every transformation digest is unchanged, which is why
#: this family was chosen to prove the abstraction before source evidence
#: is deliberately migrated.
TRANSFORMATION_SCHEMA = CanonicalDigestSchema(
    family="drift-transformation-identity", version=1)

#: Compatibility spelling. DERIVED, never a second declaration.
TRANSFORMATION_DOMAIN = TRANSFORMATION_SCHEMA.domain


class TransformationComponentKind(str, Enum):
    """The aspects of feature generation that carry independent semantics.

    Each is a separate scientific decision, and two pipelines can agree on any
    subset while differing on the rest. Collapsing them into one digest would
    make "the transformation changed" unattributable.
    """

    #: How raw source columns become semantic features: consequence severity
    #: mapping, frequency transforms, splice distance conventions.
    FEATURE_ENGINEERING = "feature_engineering"

    #: What an absent value means and whether its absence is itself a feature.
    #: Owned today by `model_preprocessing.DECLARED_MISSINGNESS`.
    MISSINGNESS = "missingness"

    #: Scaling, clipping and unit conventions.
    NORMALIZATION = "normalization"

    #: How annotation sources are joined to observations, and what a failed
    #: join produces -- a distinction Run-15 proved is load-bearing.
    JOIN_POLICY = "join_policy"

    #: Coordinate conventions: which base a position refers to, how indels are
    #: anchored. NOT the genome build, which belongs to source identity.
    COORDINATE_POLICY = "coordinate_policy"


@dataclass(frozen=True)
class TransformationComponent:
    """One named aspect, its schema version, and its declaration's digest."""

    kind: TransformationComponentKind
    schema_version: int
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TransformationComponentKind):
            raise ValueError(
                "kind is {!r}; a component must name its aspect from the "
                "declared vocabulary".format(self.kind))
        if not isinstance(self.schema_version, int) or \
                isinstance(self.schema_version, bool) or \
                self.schema_version < 1:
            raise ValueError(
                "schema_version is {!r}; expected a positive integer. A "
                "component whose declaration shape changes without a version "
                "produces two meanings under one fingerprint."
                .format(self.schema_version))
        if not isinstance(self.fingerprint, str) or \
                not _SHA256.match(self.fingerprint):
            raise ValueError(
                "fingerprint is {!r}; expected 64 lowercase hexadecimal "
                "characters".format(self.fingerprint))

    @property
    def canonical_key(self) -> str:
        return self.kind.value

    def as_record(self) -> dict:
        return {"kind": self.kind.value,
                "schema_version": self.schema_version,
                "fingerprint": self.fingerprint}


class TransformationError(ValueError):
    """A transformation identity that cannot describe a pipeline."""


@dataclass(frozen=True)
class TransformationIdentity:
    """The composition of every declared semantic aspect.

    Components are held in canonical order, enforced ONCE here. `of()`
    canonicalises, `__post_init__` verifies, and `digest` consumes the verified
    tuple verbatim -- sorting again inside the digest would make neither sort
    individually load-bearing, which is a defect this package already repaired.
    """

    components: Tuple[TransformationComponent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple):
            raise TransformationError(
                "components is {}; it must be a TUPLE so an identity cannot be "
                "mutated after its digest has been quoted"
                .format(type(self.components).__name__))
        if not self.components:
            raise TransformationError(
                "a transformation must declare at least one component. An "
                "empty identity would digest to a constant and make every "
                "pipeline compare equal.")
        for c in self.components:
            if not isinstance(c, TransformationComponent):
                raise TransformationError(
                    "component {!r} is not a TransformationComponent".format(c))
        kinds = [c.kind for c in self.components]
        if len(set(kinds)) != len(kinds):
            duplicated = sorted({k.value for k in kinds
                                 if kinds.count(k) > 1})
            raise TransformationError(
                "component kind(s) {} appear more than once. One aspect has "
                "one declaration; two would leave which applied undetermined."
                .format(duplicated))
        if list(self.components) != sorted(
                self.components, key=lambda c: c.canonical_key):
            raise TransformationError(
                "components are not in canonical order. `of()` sorts them; "
                "constructing directly with an arbitrary order would make two "
                "equal identities compare unequal. ORDER IS ENFORCED HERE AND "
                "NOWHERE ELSE.")

    @classmethod
    def of(cls, components: Iterable[TransformationComponent]
           ) -> "TransformationIdentity":
        return cls(components=tuple(
            sorted(components, key=lambda c: c.canonical_key)))

    @property
    def kinds(self) -> Tuple[TransformationComponentKind, ...]:
        return tuple(c.kind for c in self.components)

    def component(self, kind: TransformationComponentKind
                  ) -> TransformationComponent:
        for c in self.components:
            if c.kind is kind:
                return c
        raise TransformationError(
            "{!r} is not declared; this identity declares {}".format(
                kind.value, [k.value for k in self.kinds]))

    @property
    def digest(self) -> str:
        """DERIVED, never stored, and DOMAIN-SEPARATED.

        Not re-sorted: `__post_init__` has already enforced canonical order.
        """
        return TRANSFORMATION_SCHEMA.digest(
            components=[c.as_record() for c in self.components])

    def describe(self) -> str:
        return "{} component(s) [{}] digest {}".format(
            len(self.components), ", ".join(k.value for k in self.kinds),
            self.digest[:12])
