"""What kind of record this is, and therefore where it belongs.

ADR-0004 section B. The inversion is the point:

    destination(artifact) = f(ArtifactRole)

not

    meaning(artifact) ~= directory someone happened to choose

Author: Monzia Moodie
"""
from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath
from typing import Mapping

#: The plane root. A sibling of `docs/`, never a child: documentation is what
#: humans author to explain or govern the project; records are durable facts
#: emitted, captured or preserved about what actually happened.
RECORDS_ROOT = PurePosixPath("records")


class RecordsOntologyError(ValueError):
    """A record's role, placement or identity does not satisfy the ontology."""


class ArtifactRole(str, Enum):
    """The semantic kind of a durable repository record."""

    INSTALLATION_ATTESTATION = "installation_attestation"
    EXECUTION_MEASUREMENT = "execution_measurement"
    AUDIT_RESULT = "audit_result"
    INCIDENT_EVIDENCE = "incident_evidence"
    MIGRATION_RECORD = "migration_record"
    VERIFICATION_RESULT = "verification_result"
    RECOVERY_ARTIFACT = "recovery_artifact"


#: THE OPERATIONAL AUTHORITY. ADR-0004 is normative about the architecture and
#: deliberately does not restate this mapping, because two copies of a mapping
#: is the defect the record exists to prevent.
CANONICAL_RECORD_ROOTS: Mapping[ArtifactRole, PurePosixPath] = {
    ArtifactRole.INSTALLATION_ATTESTATION:
        RECORDS_ROOT / "attestations" / "installations",
    ArtifactRole.EXECUTION_MEASUREMENT: RECORDS_ROOT / "measurements",
    ArtifactRole.AUDIT_RESULT: RECORDS_ROOT / "audits",
    ArtifactRole.INCIDENT_EVIDENCE: RECORDS_ROOT / "incidents",
    ArtifactRole.MIGRATION_RECORD: RECORDS_ROOT / "migrations",
    ArtifactRole.VERIFICATION_RESULT: RECORDS_ROOT / "verification",
    ArtifactRole.RECOVERY_ARTIFACT: RECORDS_ROOT / "recovery",
}


def canonical_root(role: ArtifactRole) -> PurePosixPath:
    """Where records of this role live. Raises rather than guessing."""
    try:
        return CANONICAL_RECORD_ROOTS[role]
    except KeyError:                      # pragma: no cover - guarded by a test
        raise RecordsOntologyError(
            "no canonical root for {!r}. A role without a root cannot place "
            "anything, and a default would place it wrongly and silently."
            .format(role)) from None


def is_within_canonical_root(path: str, role: ArtifactRole) -> bool:
    """Whether `path` lies beneath the role's root.

    Containment, never a string prefix: `records/audits2/x` starts with
    `records/audits` as TEXT while being no descendant of it. ADR-0002 records
    the same distinction for runtime paths.
    """
    candidate = PurePosixPath(str(path).replace("\\", "/"))
    root = canonical_root(role)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate != root


def role_for_path(path: str) -> ArtifactRole:
    """The role a path implies, or a refusal.

    This is the INVERSE of the authority direction and exists only for auditing
    existing placement. It is never used to decide where something goes: that
    would reinstate paths as the source of meaning.
    """
    matches = [r for r in ArtifactRole if is_within_canonical_root(path, r)]
    if not matches:
        raise RecordsOntologyError(
            "{!r} lies beneath no canonical record root. Placement follows "
            "role; a path outside every root has no role to infer."
            .format(path))
    if len(matches) > 1:                  # pragma: no cover - guarded by a test
        raise RecordsOntologyError(
            "{!r} lies beneath {} roots: {}. Roots must be mutually exclusive "
            "or placement is ambiguous.".format(
                path, len(matches), sorted(m.value for m in matches)))
    return matches[0]
