"""Four orthogonal questions about one artifact.

ADR-0004 sections B2, B2b, B4, B5. Answering one of these does not answer any
other, and collapsing them is how a measured ruling about eleven benign
attestations would become a universal licence.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .roles import ArtifactRole, RecordsOntologyError


class DisclosureClass(str, Enum):
    """May these exact bytes be published?

    NOT the same question as whether the artifact may be preserved. The
    repository is public; verbatim import is permanent and cannot later be
    edited without destroying the byte identity that justified it.
    """

    PUBLIC_VERBATIM = "public_verbatim"
    PUBLIC_DERIVATIVE = "public_derivative"
    RESTRICTED_VERBATIM = "restricted_verbatim"
    HASH_ONLY_PUBLIC = "hash_only_public"


class PreservationDisposition(str, Enum):
    """May this artifact be preserved verbatim at all?

    ADMITTED_WITH_DEFECT_NOTE is the one that matters. A malformed historical
    attestation is still historical evidence; the defect belongs in the
    manifest, not in the bytes. Preservation validity and interchange validity
    are orthogonal:

        preservation_valid = True    these are exactly the bytes that existed
        interchange_valid  = False   these bytes do not satisfy the schema

    A preservation system that cannot express both at once will eventually be
    asked to repair evidence in order to store it.
    """

    ADMITTED_VERBATIM = "admitted_verbatim"
    ADMITTED_WITH_DEFECT_NOTE = "admitted_with_defect_note"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class ProvenanceRelation(str, Enum):
    """Where the artifact came from.

    A fact about the artifact, not about its location -- encoding it in a path
    loses it the moment the artifact moves.
    """

    EMITTED_BY_INSTALLER = "emitted_by_installer"
    CAPTURED_FROM_EXTERNAL_TOOL = "captured_from_external_tool"
    IMPORTED_FROM_STAGING = "imported_from_staging"
    DERIVED_FROM_RECORD = "derived_from_record"
    RECOVERED_FROM_INCIDENT = "recovered_from_incident"

    #: The artifact was never emitted, and this record establishes afterwards
    #: what surviving evidence supports. PROOF-AFTER-IRREVERSIBILITY-1,
    #: 2026-08-25: an installer committed, then refused to write its own
    #: attestation, leaving a commit message citing a file that does not exist.
    #:
    #: DISTINCT FROM ITS TWO NEIGHBOURS, and the distinction is the reason this
    #: member exists rather than an overload:
    #:
    #:     DERIVED_FROM_RECORD      derived from a record that EXISTS
    #:     RECOVERED_FROM_INCIDENT  bytes that once existed were recovered
    #:
    #: Neither is true here. There is no source record to derive from, and no
    #: bytes were ever written to recover -- the installer raised before
    #: writing. Using either would assert something that did not happen.
    RECONSTRUCTS_MISSING_ARTIFACT = "reconstructs_missing_artifact"


class RetentionClass(str, Enum):
    """How long it is kept. Immutability is not the same as permanence.

    Without a policy, `records/` accumulates forever and someone eventually
    deletes material under time pressure with nothing to appeal to.
    """

    PERMANENT_EVIDENCE = "permanent_evidence"
    SUPERSEDABLE_SNAPSHOT = "supersedable_snapshot"
    TRANSIENT_DIAGNOSTIC = "transient_diagnostic"


@dataclass(frozen=True)
class RecordDisposition:
    """The four axes together, for one artifact.

    The order of decision is fixed: classify the role, classify disclosure,
    validate preservation, then choose storage. Never "this is evidence,
    therefore commit its bytes".
    """

    role: ArtifactRole
    disclosure: DisclosureClass
    preservation: PreservationDisposition
    provenance: tuple[ProvenanceRelation, ...]
    retention: RetentionClass
    defect_note: str = ""

    def __post_init__(self) -> None:
        if not self.provenance:
            raise RecordsOntologyError(
                "an artifact with no recorded provenance cannot be audited; "
                "'where did this come from' must not be inferable only from a "
                "directory name")
        if len(set(self.provenance)) != len(self.provenance):
            raise RecordsOntologyError(
                "duplicate provenance relations: {}".format(
                    [p.value for p in self.provenance]))
        needs_note = self.preservation in (
            PreservationDisposition.ADMITTED_WITH_DEFECT_NOTE,
            PreservationDisposition.QUARANTINED,
            PreservationDisposition.REJECTED)
        if needs_note and not self.defect_note.strip():
            raise RecordsOntologyError(
                "{} requires a defect_note. A disposition that records a "
                "problem without recording what it was is not a record."
                .format(self.preservation.value))
        if not needs_note and self.defect_note.strip():
            raise RecordsOntologyError(
                "{} carries a defect_note but declares no defect".format(
                    self.preservation.value))
        if (self.disclosure is DisclosureClass.RESTRICTED_VERBATIM
                and self.preservation is PreservationDisposition.ADMITTED_VERBATIM):
            raise RecordsOntologyError(
                "RESTRICTED_VERBATIM may not be ADMITTED_VERBATIM into a public "
                "repository. Route it to a restricted channel; a redacted copy "
                "presented as the original is a forgery, however well "
                "intentioned")

    @property
    def is_publicly_preservable(self) -> bool:
        return (self.disclosure in (DisclosureClass.PUBLIC_VERBATIM,)
                and self.preservation in (
                    PreservationDisposition.ADMITTED_VERBATIM,
                    PreservationDisposition.ADMITTED_WITH_DEFECT_NOTE))
