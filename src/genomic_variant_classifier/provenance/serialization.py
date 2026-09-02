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

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json

#: Prefixed to every payload. The NUL byte cannot occur in the ASCII JSON that
#: follows, so the boundary between domain and payload is unambiguous.
_NAMESPACE = "genomic-variant-classifier:"
_SEPARATOR = "\x00"


def canonical_json(obj) -> bytes:
    """One deterministic serialisation, at any nesting depth.

    `ensure_ascii=True` so the bytes cannot depend on a locale or a filesystem
    encoding; `sort_keys=True` so key order cannot alter identity; no spaces,
    so whitespace cannot either.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def domain_digest(domain: str, obj) -> str:
    """A digest that says what KIND of thing it identifies.

    `domain` must carry a version suffix. A digest whose domain never changes
    cannot express a schema change, and two records under one domain would then
    be comparable across incompatible shapes.
    """
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty string")
    if _SEPARATOR in domain:
        raise ValueError(
            "domain {!r} contains the separator byte; the boundary between "
            "domain and payload would be ambiguous".format(domain))
    if not domain.rstrip("0123456789").endswith("-v"):
        raise ValueError(
            "domain {!r} carries no version suffix. A digest whose domain "
            "never changes cannot express a schema change, and records of two "
            "incompatible shapes would compare across it.".format(domain))
    prefix = (_NAMESPACE + domain + _SEPARATOR).encode("ascii")
    return hashlib.sha256(prefix + canonical_json(obj)).hexdigest()
