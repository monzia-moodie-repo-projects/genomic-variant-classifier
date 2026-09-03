"""Domain-separated, versioned digests over canonical JSON.

DRIFT-1 Phase 1B.1. Created 2026-08-27.

WHY DOMAIN SEPARATION
---------------------
A 64-character SHA-256 digest does not say what it identifies. Two digests of
different KINDS are both 64 lowercase hexadecimal characters, so a source
manifest digest and a transformation digest are structurally interchangeable --
and version 3 of the attestation schema types every digest field as exactly
that shape, which means a caller could pass one where the other belongs and
every validator would accept it.

This is not about collision resistance. It is about TYPE CONFUSION.

WHY CANONICAL JSON
------------------
The previous implementation joined fields with unit and record separators and
validated every field against a pattern excluding them -- a careful design,
with a test proving the collision it prevents is real.

It does not scale to nested identities. Once a manifest holds dependencies
holding identities holding role sets, a flat separator scheme needs a new rule
per level. `json.dumps(sort_keys=True, separators=(",", ":"),
ensure_ascii=True)` gives one deterministic serialisation for any depth.

WHY THE DOMAIN CARRIES A VERSION
--------------------------------
`drift-source-evidence-manifest-v2` rather than `source-manifest`. When the
canonical record shape changes, the domain changes with it, so digests from two
schema versions cannot silently compare equal. That is the same boundary
`DriftReferenceProfile.load` enforces with `format_version`, and the same one
`install_attestation.validate` enforces by refusing versions it does not judge.

ONE GRAMMAR, ONE PARSER
-----------------------
Phase 1C Unit 3A++.3. The version requirement was previously expressed as
`domain.rstrip("0123456789").endswith("-v")`, and MEASURED 2026-09-03 that
predicate accepts `family-v` -- a domain carrying no version at all, which has
nothing to increment. It also accepted `family-v0` and `family-v01`, so `v1`,
`v01` and `v001` would have been three byte-distinct namespaces for one
numerical epoch.

`parse_versioned_domain` is now the ONLY definition of that grammar:

    VersionedDomain = Family "-v" PositiveCanonicalInteger
    PositiveCanonicalInteger = [1-9][0-9]*

`CanonicalDigestSchema` cross-checks the domain it emits against this same
parser rather than re-implementing the rule. Two independent validators is the
dual-authority pathology this arc removed from source evidence; it must not
reappear one layer down.

MEASURED BEFORE CHANGING: every live digest domain in the repository already
satisfies this grammar. The five `-v0` occurrences are prose in incident
documents ("regime-v0 runs"), there are no leading-zero forms anywhere, and the
only frozen historical domain is `drift-source-evidence-manifest-v4`. So this
is a HARDENING -- the accepted set narrows and no valid digest moves.

GVC CANONICAL JSON v1
---------------------
The serialisation contract, stated so it can be cited rather than inferred:

    1. Unicode input values are permitted.
    2. Object keys are sorted lexicographically.
    3. Compact separators, no whitespace.
    4. Non-ASCII characters are emitted as JSON escapes.
    5. Output bytes are ASCII.
    6. No platform newline participates.
    7. No locale participates.
    8. No filesystem encoding participates.
    9. Non-finite numbers are REFUSED.
   10. No Unicode normalisation is performed implicitly.

Rule 9 was added at 3A++.3 after MEASURING that no live canonical record
contains a float at all: the evidence payload holds only `NoneType`, `int` and
`str`, and the transformation payload only `int` and `str`. Both digests are
byte-identical with `allow_nan=False`, so the refusal is semantic-zero.

Rule 10 is deliberate. `"\u00e9"` and `"e\u0301"` are different code point
sequences, and a registry may distinguish them. Normalising here would decide
an admission-policy question inside a serialiser.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation; ASCII = American Standard Code for Information Interchange;
NUL = the zero byte.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

#: Prefixed to every payload. The NUL byte cannot occur in the ASCII JSON that
#: follows, so the boundary between domain and payload is unambiguous.
_NAMESPACE = "genomic-variant-classifier:"
_SEPARATOR = "\x00"

#: The ONE definition of a canonical versioned domain. A positive integer with
#: no leading zero, so each epoch has exactly one spelling.
_VERSIONED_DOMAIN = re.compile(r"\A(?P<family>.+)-v(?P<version>[1-9][0-9]*)\Z")


class DigestDomainError(ValueError):
    """A digest domain is not a canonical versioned namespace.

    Subclasses `ValueError`, so every existing `pytest.raises(ValueError)`
    against `domain_digest` keeps catching it.
    """


@dataclass(frozen=True)
class ParsedDigestDomain:
    """A domain split into the two things it declares."""

    family: str
    version: int


def parse_versioned_domain(domain: str) -> ParsedDigestDomain:
    """Refuse anything that is not exactly one spelling of one epoch.

    REFUSED, and why each matters:

        ""              nothing is identified
        "family"        no version, so a schema change cannot be expressed
        "family-v"      a version marker with no version
        "family-v0"     `CanonicalDigestSchema` requires version >= 1, so this
                        is a domain the canonical authority can never produce
        "family-v01"    a second spelling of epoch 1
        "-v1"           an empty family
        "bad\x00-v1"    the separator byte, making the payload boundary
                        ambiguous
        non-ASCII       the prefix is encoded as ASCII
    """
    if not isinstance(domain, str):
        raise DigestDomainError(
            "domain must be a string, got {}".format(type(domain).__name__))
    if not domain:
        raise DigestDomainError("domain must not be empty")
    if _SEPARATOR in domain:
        raise DigestDomainError(
            "domain {!r} contains the separator byte; the boundary between "
            "domain and payload would be ambiguous".format(domain))
    if not domain.isascii():
        raise DigestDomainError(
            "domain {!r} is not ASCII; it is encoded as ASCII into the digest "
            "prefix".format(domain))
    match = _VERSIONED_DOMAIN.match(domain)
    if match is None:
        raise DigestDomainError(
            "domain {!r} is not a canonical versioned domain. The required "
            "form is '<family>-vN', where the version N is a positive integer "
            "with no leading zero. A digest whose domain never changes cannot "
            "express a schema change, and two spellings of one version would "
            "be two namespaces for one meaning.".format(domain))
    family = match.group("family")
    if not family:                                          # pragma: no cover
        raise DigestDomainError("domain family must not be empty")
    version = int(match.group("version"))
    if version < 1:                                         # pragma: no cover
        raise AssertionError(
            "the grammar admitted a non-positive version: {!r}".format(domain))
    return ParsedDigestDomain(family=family, version=version)


def canonical_json(obj) -> bytes:
    """One deterministic serialisation, at any nesting depth.

    `ensure_ascii=True` so the bytes cannot depend on a locale or a filesystem
    encoding; `sort_keys=True` so key order cannot alter identity; no spaces,
    so whitespace cannot either; `allow_nan=False` so a non-finite number is
    REFUSED rather than emitted as the non-standard `NaN`, `Infinity` or
    `-Infinity` tokens, which no conforming JSON reader accepts.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def domain_digest(domain: str, obj) -> str:
    """A digest that says what KIND of thing it identifies.

    `domain` must carry a version suffix. A digest whose domain never changes
    cannot express a schema change, and two records under one domain would then
    be comparable across incompatible shapes.
    """
    parse_versioned_domain(domain)
    prefix = (_NAMESPACE + domain + _SEPARATOR).encode("ascii")
    return hashlib.sha256(prefix + canonical_json(obj)).hexdigest()
