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

from dataclasses import dataclass
from typing import Any, Dict

from genomic_variant_classifier.provenance.serialization import domain_digest


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
        if family.rstrip("0123456789").endswith("-v"):
            raise DigestSchemaError(
                "family {!r} is VERSIONED. The family must be unversioned: "
                "version has exactly one authority, and a version in the "
                "family name would be a second place to change."
                .format(self.family))
        if (not isinstance(self.version, int)
                or isinstance(self.version, bool)
                or self.version < 1):
            raise DigestSchemaError(
                "version is {!r}; expected a positive integer. A boolean is "
                "refused explicitly because `True == 1` in Python, so `True` "
                "would silently become schema version 1."
                .format(self.version))
        object.__setattr__(self, "family", family)

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
