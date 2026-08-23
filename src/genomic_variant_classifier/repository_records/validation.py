"""Three validation policies that must never be aliases for one another.

ADR-0004 section C.

    AUTHORING      what NEWLY AUTHORED artifacts must look like
    PRESERVATION   what HISTORICAL artifacts must RETAIN
    INTERCHANGE    what structured data must satisfy to be interpreted

VERBATIM-IMPORT-NOT-AUTHORING-1, measured 2026-08-22: every install attestation
was written with `json.dumps()` or `to_json()`, neither of which appends a
newline, so ELEVEN OF ELEVEN end without one. The authoring predicate requires a
trailing newline; an importer reusing it would refuse every file it exists to
preserve, and adding one would change the bytes -- destroying the byte identity
that is the entire preservation claim.

    An archival importer verifies the artifact AS FOUND. It does not retrofit
    current repository formatting conventions onto historical evidence.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib

from .roles import RecordsOntologyError


class AuthoringPolicyError(RecordsOntologyError):
    """A newly authored artifact does not meet the house convention."""


class PreservationPolicyError(RecordsOntologyError):
    """A historical artifact would not be preserved as found."""


def validate_authored_text(name: str, data: bytes) -> None:
    """The AUTHORING policy. For files this repository writes fresh.

    Never apply this to an imported artifact. It demands properties the
    historical corpus does not have and cannot be given without mutation.
    """
    if data[:3] == b"\xef\xbb\xbf":
        raise AuthoringPolicyError("{}: byte-order mark".format(name))
    if b"\r\n" in data:
        raise AuthoringPolicyError("{}: CRLF".format(name))
    if any(b > 0x7F for b in data):
        raise AuthoringPolicyError("{}: non-ASCII".format(name))
    if not data.endswith(b"\n"):
        raise AuthoringPolicyError("{}: no trailing newline".format(name))
    if not data.strip():
        raise AuthoringPolicyError("{}: empty".format(name))


def validate_verbatim_artifact(name: str, source: bytes, preserved: bytes,
                               expected_sha256: str = "") -> None:
    """The PRESERVATION policy. Forbids MUTATION, not house style.

    It does NOT require a trailing newline, pure ASCII, or absence of a
    byte-order mark: a historical artifact having any of those properties is a
    fact about history, and correcting it would falsify the record.

    What it forbids is the preserved copy differing from the source in any way.
    """
    if not source:
        raise PreservationPolicyError(
            "{}: the source is empty; there is nothing to preserve".format(name))
    if preserved != source:
        raise PreservationPolicyError(
            "{}: the preserved bytes differ from the source. {} bytes vs {}. "
            "Preservation forbids mutation absolutely -- no newline added, no "
            "schema upgraded, no key reordered, no path redacted.".format(
                name, len(preserved), len(source)))
    actual = hashlib.sha256(source).hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise PreservationPolicyError(
            "{}: digest {} does not match the declared {}".format(
                name, actual, expected_sha256))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
