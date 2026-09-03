"""One canonical identity epoch, one version authority.

Phase 1C Unit 3A++.1. Created 2026-09-02.

THE DEFECT THIS PREVENTS
------------------------
MEASURED 2026-09-02 in `provenance/source.py`:

    line 105   EVIDENCE_DOMAIN = "drift-source-evidence-manifest-v4"
    line 486   "schema_version": 3,

Two writable declarations represented ONE semantic version. `product` was added
to `SourceArtifactKey` on 2026-09-01, changing key equality and the canonical
record shape; the domain was bumped and the embedded literal was not.

That architecture permits divergence BY CONSTRUCTION. No amount of care fixes
it, because nothing binds the two numbers together. `CanonicalDigestSchema`
binds them: the domain is DERIVED from family and version, and the record is
stamped with that same version. The invalid state

    domain v5, record schema 4

cannot be expressed through this API.

WHY THE FAMILY MUST BE UNVERSIONED
----------------------------------
`family="drift-transformation-identity"`, never
`family="drift-transformation-identity-v1"`. If the family carried a version
too, there would again be two places to change and one could be forgotten --
the exact defect this type exists to remove.

ONE GRAMMAR, CROSS-CHECKED
--------------------------
Phase 1C Unit 3A++.3. `__post_init__` now parses the domain it would emit
through `parse_versioned_domain` -- the same function `domain_digest` uses --
and refuses if that parser disagrees.

Before this, the two layers validated the epoch grammar independently: this
class required `version >= 1` while the primitive accepted any digits, or none
at all. MEASURED 2026-09-03: `domain_digest("family-v", ...)` was ACCEPTED,
though no `CanonicalDigestSchema` could ever produce it. Two validators for one
language is exactly the dual-authority pathology this arc removed from source
evidence.

The family check also tightened. `-v` followed by ANY digits is refused,
including `family-v0`, which the previous `rstrip` predicate also caught but
for an accidental reason.

WHY THIS IS NOT IN serialization.py
-----------------------------------
`serialization.py` knows only: object -> canonical JSON -> domain-separated
digest. It must NOT know source, transformation or materialization schema
versions, or the lowest layer becomes an identity-policy registry.

    serialization
          ^
    digest_schema
          ^
    source / transformation / materialization / derivation

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

from genomic_variant_classifier.provenance.serialization import (
    DigestDomainError,
    domain_digest,
    parse_versioned_domain,
)


class DigestSchemaError(ValueError):
    """An invalid canonical digest-schema declaration."""


@dataclass(frozen=True)
class CanonicalDigestSchema:
    """One authority for one canonical identity epoch.

    `domain` and the record's `schema_version` are both derived from `version`,
    so they cannot drift apart. That is the whole point of the type.
    """

    family: str
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise DigestSchemaError("family must be a non-empty string")
        family = self.family.strip()
        if "\x00" in family:
            raise DigestSchemaError(
                "family cannot contain the domain separator byte; the "
                "boundary between domain and payload would be ambiguous")
        if re.search(r"-v[0-9]*\Z", family):
            raise DigestSchemaError(
                "family {!r} is VERSIONED. The family must be unversioned: "
                "version has exactly one authority, and a version in the "
                "family name would be a second place to change."
                .format(self.family))
        if not family.isascii():
            raise DigestSchemaError(
                "family {!r} is not ASCII; the domain is encoded as ASCII "
                "into the digest prefix".format(self.family))
        if (not isinstance(self.version, int)
                or isinstance(self.version, bool)
                or self.version < 1):
            raise DigestSchemaError(
                "version is {!r}; expected a positive integer. A boolean is "
                "refused explicitly because `True == 1` in Python, so `True` "
                "would silently become schema version 1."
                .format(self.version))
        object.__setattr__(self, "family", family)

        # ONE GRAMMAR. Phase 1C Unit 3A++.3: the domain this object emits is
        # cross-checked against the SAME parser `domain_digest` uses, rather
        # than trusting that two independent rules agree. Two validators for
        # one language is the dual-authority pathology this arc removed from
        # source evidence, and it must not reappear one layer down.
        try:
            parsed = parse_versioned_domain(self.domain)
        except DigestDomainError as exc:
            raise DigestSchemaError(
                "this schema would emit {!r}, which the canonical domain "
                "grammar refuses: {}".format(self.domain, exc)) from exc
        if parsed.family != family or parsed.version != self.version:
            raise DigestSchemaError(          # pragma: no cover
                "the emitted domain {!r} parses back as {!r}/{}, not {!r}/{}"
                .format(self.domain, parsed.family, parsed.version,
                        family, self.version))

    @property
    def domain(self) -> str:
        """The versioned domain string `domain_digest` requires."""
        return "{}-v{}".format(self.family, self.version)

    def record(self, **payload: Any) -> Dict[str, Any]:
        """The canonical record, stamped with THIS schema's version.

        A caller passing `schema_version` is refused rather than overridden:
        silently ignoring it would let a caller believe they had set something.
        """
        if "schema_version" in payload:
            raise DigestSchemaError(
                "schema_version is owned by CanonicalDigestSchema and cannot "
                "be supplied by a caller; that is the duplication this type "
                "exists to prevent")
        stamped = {"schema_version": self.version}
        stamped.update(payload)
        return stamped

    def digest(self, **payload: Any) -> str:
        """A digest over the stamped record, under the derived domain."""
        return domain_digest(self.domain, self.record(**payload))

    def describe(self) -> str:
        return "{} (schema {})".format(self.domain, self.version)
