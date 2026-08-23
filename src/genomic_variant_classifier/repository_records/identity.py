"""A record, an artifact instance, and a filename are three different things.

ADR-0004 section B3.

Retaining a historical basename does NOT make a commit citation resolve. Git
does not turn a filename in a commit message into a locator. Resolution is a
repository property, established by recorded aliases and proven by a test.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

from .roles import ArtifactRole, RecordsOntologyError, is_within_canonical_root

#: `REC-` plus a uuid4 hexadecimal. NOT sequential: `REC-0001` implies a global
#: ordering nothing enforces and invites renumbering, and a renumbered durable
#: identity is not durable.
#:
#: uuid7 would be preferable for sortability. MEASURED 2026-08-22: uuid6, uuid7
#: and uuid8 entered the standard library in Python 3.14, and continuous
#: integration runs 3.11 and 3.12. uuid4 it is.
_RECORD_ID = re.compile(r"^REC-[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def allocate_record_id() -> str:
    """Mint a new durable identity. ONCE, AT PRESERVATION, and never again.

    Because uuid4 is random, a deterministic projection over records is only
    possible if it READS identifiers rather than minting them. A renderer that
    allocated would make `actual == render(scan(...))` fail on every run for a
    reason unrelated to drift -- a check failing for the wrong reason, which
    this project treats as a defect in its own right.
    """
    return "REC-" + uuid.uuid4().hex


@dataclass(frozen=True)
class RecordId:
    """Durable logical identity.

    Never reused, never renumbered, never derived from a path, a filename or an
    ordinal. A path is where a thing is; this is what it is.
    """

    value: str

    def __post_init__(self) -> None:
        if not _RECORD_ID.match(self.value):
            raise RecordsOntologyError(
                "{!r} is not a record identifier. Expected REC- followed by 32 "
                "lowercase hexadecimal characters.".format(self.value))


@dataclass(frozen=True)
class ArtifactInstance:
    """One concrete byte sequence at one location.

    Separate from RecordId because a record may later be copied, mirrored or
    re-encoded for interchange and exist as several byte sequences. Fusing them
    would force a rename or a renumber the first time that happens.
    """

    content_sha256: str
    canonical_path: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not _SHA256.match(self.content_sha256):
            raise RecordsOntologyError(
                "{!r} is not a SHA-256 digest: expected 64 lowercase "
                "hexadecimal characters".format(self.content_sha256))
        if self.size_bytes < 0:
            raise RecordsOntologyError(
                "negative size {}".format(self.size_bytes))
        if self.size_bytes == 0:
            raise RecordsOntologyError(
                "a zero-length artifact preserves nothing; if the historical "
                "file was genuinely empty, record that as a defect note rather "
                "than as evidence")
        p = str(self.canonical_path)
        if p != p.replace("\\", "/"):
            raise RecordsOntologyError(
                "{!r} uses backslashes. Repository paths are POSIX; a Windows "
                "separator in a durable record makes it unresolvable "
                "elsewhere.".format(self.canonical_path))
        if PurePosixPath(p).is_absolute():
            raise RecordsOntologyError(
                "{!r} is absolute. A durable record locates artifacts relative "
                "to the repository, not to one workstation.".format(p))


@dataclass(frozen=True)
class RecordIdentity:
    """The three identities, bound together, with historical compatibility.

    `legacy_aliases` is what makes a historical citation resolvable, and it is a
    COMMITTED fact rather than a walk of git history. A test collecting
    citations from `git log --all` would be silently vacuous in a shallow
    checkout -- `actions/checkout` defaults to `fetch-depth: 1`, MEASURED
    2026-08-22 across ten invocations in four workflows, none declaring a depth.
    Such a walk sees one commit, asserts one citation resolves, and passes.
    """

    record_id: RecordId
    instance: ArtifactInstance
    role: ArtifactRole
    legacy_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not is_within_canonical_root(self.instance.canonical_path, self.role):
            raise RecordsOntologyError(
                "{!r} is not beneath the canonical root for {}. Placement "
                "follows role; a record filed elsewhere is how six evidence "
                "locations happened.".format(
                    self.instance.canonical_path, self.role.value))
        if len(set(self.legacy_aliases)) != len(self.legacy_aliases):
            raise RecordsOntologyError(
                "duplicate legacy aliases: {}".format(list(self.legacy_aliases)))
        for alias in self.legacy_aliases:
            if "/" in alias or "\\" in alias:
                raise RecordsOntologyError(
                    "{!r} is a path, not an alias. Aliases are the BASENAMES "
                    "historical citations used; a path would be a second "
                    "locator competing with canonical_path.".format(alias))
            if not alias.strip():
                raise RecordsOntologyError("an empty alias resolves nothing")

    @property
    def basename(self) -> str:
        return PurePosixPath(self.instance.canonical_path).name
