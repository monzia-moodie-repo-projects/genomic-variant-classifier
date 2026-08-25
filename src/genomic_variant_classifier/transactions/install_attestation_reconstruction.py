"""What can be established later, when the original was never validly emitted.

PROOF-AFTER-IRREVERSIBILITY-1. Created 2026-08-25.

WHY THIS IS NOT AN INSTALLATION ATTESTATION
===========================================
On 2026-08-25 the DRIFT-1 installer applied eight targets, proved its suite
transition by node identity, passed a 5,479-case acceptance gate, and committed
`abcb22e`. Then it refused:

    ATTESTATION INVALID AFTER A SUCCESSFUL COMMIT:
        a deliberate retirement requires a justification

The installation is correct. The attestation was never written, and the commit
message cites a filename that does not exist. `gvc.install-attestation` v2
requires `started_at`, and MEASURED 2026-08-25 that value is UNRECOVERABLE
within a 1,434-second interval: the installer samples the clock after its heavy
package imports, and no witness closes the upper end of the window.

A v2 attestation carrying an invented `started_at` would pass `validate()`,
because that validator checks PRESENCE, not semantic validity. **A schema
accepting a false value does not make that value evidence.** So this is a
different artifact class, with a different question:

    gvc.install-attestation        what the installer EMITTED
    gvc.install-attestation-       what can be ESTABLISHED afterwards from
      reconstruction               surviving evidence, and what cannot

It does not claim to be the missing original. It resolves the historical
citation while preserving the fact that no valid v2 artifact was ever emitted.

WHY NOT AN `amendments` ENTRY
=============================
MEASURED 2026-08-25: exactly one preserved attestation uses `amendments`, and
its established shape is per-artifact mutation --
`{artifact, finding, kind, preimage_sha256, postimage_sha256}`. A document that
was never emitted has no preimage and no postimage. The shape cannot express
this case, independently of the missing `started_at`.

THE EPISTEMIC CORE
==================
`KnowledgeState` distinguishes three things v2 cannot:

    OBSERVED        read directly from a surviving witness
    DERIVED_EXACT   uniquely determined at a stated resolution by a squeeze
                    between two independent witnesses -- NOT observed
    BOUNDED         an interval is known; no point estimate is supported
    UNRECOVERABLE   no witness constrains it at all

`finished_at` is DERIVED_EXACT at one-second resolution: the installer samples
it after `git commit` returns and before the failure line prints, and both
witnesses -- the committer date and the apply log's last write -- fall in the
same second. That is a derivation, not an observation, and saying so is the
point of the distinction.

`started_at` is BOUNDED. No invented point estimate.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; OID = object identifier;
JSON = JavaScript Object Notation; UTC = Coordinated Universal Time.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

SCHEMA = "gvc.install-attestation-reconstruction"
SCHEMA_VERSION = 1

#: Full Git object identifiers. The v2 attestation format historically accepted
#: abbreviated commit identifiers -- the seven-character forms are in the
#: preserved corpus -- and this class is new, so there is no compatibility
#: reason to inherit that weakness. Identity is not a display string.
_OID = re.compile(r"^[0-9a-f]{40}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class ReconstructionSchemaError(ValueError):
    """A reconstruction does not satisfy its own contract."""


class KnowledgeState(str, Enum):
    """How well a field is known. `(str, Enum)`: MEASURED 2026-08-24, 79 of 79
    enumerations in `src/` use this form and none uses `StrEnum`."""

    #: Read directly from a surviving witness.
    OBSERVED = "observed"

    #: Uniquely determined at a stated resolution by two independent witnesses
    #: bounding it to one value. Not observed -- derived.
    DERIVED_EXACT = "derived_exact"

    #: An interval is supported. A point estimate is not.
    BOUNDED = "bounded"

    #: No witness constrains it. Recorded as absent rather than guessed.
    UNRECOVERABLE = "unrecoverable"


class EvidenceKind(str, Enum):
    """What sort of witness supports a field."""

    GIT_OBJECT = "git_object"
    GIT_METADATA = "git_metadata"
    FILESYSTEM_METADATA = "filesystem_metadata"
    FILE_CONTENT = "file_content"
    COMMIT_MESSAGE = "commit_message"
    PROCESS_TRANSCRIPT = "process_transcript"


class ReconstructionStatus(str, Enum):
    """Whether every required field was recoverable exactly."""

    COMPLETE = "complete"
    PARTIAL = "partial"


@dataclass(frozen=True)
class EvidenceRef:
    """One witness, with what it is claimed to establish.

    `claim` is mandatory. A witness listed without saying what it supports is
    decoration: a later reader cannot tell whether it was load-bearing or
    merely nearby.
    """

    kind: EvidenceKind
    locator: str
    claim: str
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.locator.strip():
            raise ReconstructionSchemaError(
                "an evidence reference requires a locator")
        if not self.claim.strip():
            raise ReconstructionSchemaError(
                "{}: a witness must state what it establishes. A reference "
                "without a claim cannot be audited.".format(self.locator))
        if self.sha256 and not re.match(r"^[0-9a-f]{64}$", self.sha256):
            raise ReconstructionSchemaError(
                "{}: {!r} is not a SHA-256 digest".format(
                    self.locator, self.sha256))

    def as_record(self) -> dict:
        record = {"kind": self.kind.value, "locator": self.locator,
                  "claim": self.claim}
        if self.sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True)
class EvidencedField:
    """One field of the missing document, with how well it is known.

    The invariants make the three impossible combinations unconstructible:
    an exact value with bounds, bounds with a point estimate, and an
    unrecoverable field carrying invented data.
    """

    name: str
    state: KnowledgeState
    value: str = ""
    lower_bound: str = ""
    upper_bound: str = ""
    resolution: str = ""
    derivation: str = ""
    witnesses: tuple = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ReconstructionSchemaError("a field requires a name")
        has_bounds = bool(self.lower_bound or self.upper_bound)

        if self.state in (KnowledgeState.OBSERVED, KnowledgeState.DERIVED_EXACT):
            if not self.value:
                raise ReconstructionSchemaError(
                    "{}: {} requires a value".format(
                        self.name, self.state.value))
            if has_bounds:
                raise ReconstructionSchemaError(
                    "{}: {} may not carry bounds. A value that is known "
                    "exactly is not also an interval.".format(
                        self.name, self.state.value))
        elif self.state is KnowledgeState.BOUNDED:
            if self.value:
                raise ReconstructionSchemaError(
                    "{}: BOUNDED may not claim an exact value. That is the "
                    "invented point estimate this state exists to "
                    "prevent.".format(self.name))
            if not (self.lower_bound and self.upper_bound):
                raise ReconstructionSchemaError(
                    "{}: BOUNDED requires BOTH bounds; one bound is not an "
                    "interval".format(self.name))
            if self.lower_bound > self.upper_bound:
                raise ReconstructionSchemaError(
                    "{}: lower bound {!r} exceeds upper bound {!r}".format(
                        self.name, self.lower_bound, self.upper_bound))
        else:
            if self.value or has_bounds:
                raise ReconstructionSchemaError(
                    "{}: UNRECOVERABLE may not carry invented data".format(
                        self.name))

        if self.state is KnowledgeState.DERIVED_EXACT:
            if not (self.resolution.strip() and self.derivation.strip()):
                raise ReconstructionSchemaError(
                    "{}: DERIVED_EXACT requires a resolution and a derivation. "
                    "'Uniquely determined at one-second resolution by an "
                    "interval squeeze' is a different claim from 'observed', "
                    "and the difference must be recorded.".format(self.name))
        elif self.resolution.strip() or self.derivation.strip():
            raise ReconstructionSchemaError(
                "{}: only DERIVED_EXACT carries a resolution or a "
                "derivation".format(self.name))

        if self.state is not KnowledgeState.UNRECOVERABLE and not self.witnesses:
            raise ReconstructionSchemaError(
                "{}: a field claimed to be known requires at least one "
                "witness".format(self.name))
        if self.state is KnowledgeState.UNRECOVERABLE and self.witnesses:
            raise ReconstructionSchemaError(
                "{}: an unrecoverable field cites witnesses. If a witness "
                "constrains it, it is not unrecoverable.".format(self.name))

    def as_record(self) -> dict:
        record = {"name": self.name, "state": self.state.value}
        for key, val in (("value", self.value),
                         ("lower_bound", self.lower_bound),
                         ("upper_bound", self.upper_bound),
                         ("resolution", self.resolution),
                         ("derivation", self.derivation)):
            if val:
                record[key] = val
        if self.witnesses:
            record["witnesses"] = [w.as_record() for w in self.witnesses]
        return record


@dataclass(frozen=True)
class GitIdentity:
    """Full object identifiers. Abbreviations are presentation, not identity."""

    commit_oid: str
    tree_oid: str
    parent_oid: str

    def __post_init__(self) -> None:
        for label, value in (("commit_oid", self.commit_oid),
                             ("tree_oid", self.tree_oid),
                             ("parent_oid", self.parent_oid)):
            if not _OID.match(value):
                raise ReconstructionSchemaError(
                    "{} is {!r}; a full 40-character lowercase object "
                    "identifier is required. A seven-character abbreviation "
                    "is a display string, and this class is new enough to "
                    "have no reason to inherit that weakness.".format(
                        label, value))
        if self.commit_oid == self.parent_oid:
            raise ReconstructionSchemaError(
                "a commit cannot be its own parent")

    def as_record(self) -> dict:
        return {"commit_oid": self.commit_oid, "tree_oid": self.tree_oid,
                "parent_oid": self.parent_oid}


@dataclass(frozen=True)
class PublicationFailure:
    """Why a reconstruction exists at all.

    Recording only the recovered good data would lose the reason -- and the
    reason is the finding.
    """

    finding: str
    publication_error: str
    original_artifact_validly_emitted: bool = False

    def __post_init__(self) -> None:
        if not self.finding.strip():
            raise ReconstructionSchemaError(
                "a publication failure requires a finding identifier")
        if not self.publication_error.strip():
            raise ReconstructionSchemaError(
                "a publication failure requires the error it raised")
        if self.original_artifact_validly_emitted:
            raise ReconstructionSchemaError(
                "if the original was validly emitted there is nothing to "
                "reconstruct; preserve the original instead")

    def as_record(self) -> dict:
        return {
            "finding": self.finding,
            "publication_error": self.publication_error,
            "original_artifact_validly_emitted":
                self.original_artifact_validly_emitted,
        }


@dataclass(frozen=True)
class ReconstructionDocument:
    """A typed reconstruction. Construction IS validation."""

    subject_unit: str
    intended_legacy_alias: str
    repository: GitIdentity
    failure: PublicationFailure
    fields: tuple
    suite_transition: dict
    acceptance: dict
    targets: tuple
    reconstructed_at: str
    _checked: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not self.subject_unit.strip():
            raise ReconstructionSchemaError("a reconstruction names its unit")
        alias = self.intended_legacy_alias
        if not alias.strip():
            raise ReconstructionSchemaError(
                "a reconstruction states the citation it resolves")
        if "/" in alias or "\\" in alias:
            raise ReconstructionSchemaError(
                "{!r} is a path, not a basename. The cited alias is what a "
                "commit message said, and a path would compete with the "
                "canonical location.".format(alias))
        if not _UTC.match(self.reconstructed_at):
            raise ReconstructionSchemaError(
                "reconstructed_at is {!r}; expected YYYY-MM-DDTHH:MM:SSZ"
                .format(self.reconstructed_at))
        if not self.fields:
            raise ReconstructionSchemaError(
                "a reconstruction with no recovered fields establishes "
                "nothing")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ReconstructionSchemaError(
                "duplicate field name(s): {}".format(
                    sorted({n for n in names if names.count(n) > 1})))
        if not self.targets:
            raise ReconstructionSchemaError(
                "an installation that wrote nothing is not an installation")
        for required in ("kind", "before_digest", "after_digest"):
            if required not in self.suite_transition:
                raise ReconstructionSchemaError(
                    "the suite transition record lacks {!r}".format(required))
        object.__setattr__(self, "_checked", True)

    @property
    def status(self) -> ReconstructionStatus:
        """PARTIAL unless every field is exactly known.

        Derived, never declared: a status a caller could assert independently
        of the fields would eventually disagree with them.
        """
        exact = (KnowledgeState.OBSERVED, KnowledgeState.DERIVED_EXACT)
        if all(f.state in exact for f in self.fields):
            return ReconstructionStatus.COMPLETE
        return ReconstructionStatus.PARTIAL

    @property
    def payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "artifact_class": "installation_attestation_reconstruction",
            "subject_unit": self.subject_unit,
            "intended_legacy_alias": self.intended_legacy_alias,
            "reconstruction_status": self.status.value,
            "reconstructed_at": self.reconstructed_at,
            "failure": self.failure.as_record(),
            "repository": self.repository.as_record(),
            "recovered_fields": [f.as_record() for f in self.fields],
            "suite_transition": self.suite_transition,
            "acceptance": self.acceptance,
            "targets": list(self.targets),
        }

    def render(self) -> bytes:
        """Deterministic AND diffable. AUTHORED, so it ends with a newline.

        Unlike the seventeen preserved artifacts beside it, every one of which
        ends WITHOUT one because `json.dumps` does not append one. ADR-0004
        section C: applying one policy to both would refuse every file the
        archive exists to hold.
        """
        return (json.dumps(self.payload, indent=2, sort_keys=True,
                           ensure_ascii=True) + "\n").encode("utf-8")

    @classmethod
    def parse(cls, data: bytes) -> dict:
        """Validate rendered bytes as a well-formed reconstruction payload.

        Returns the payload rather than reconstructing the typed object: the
        round-trip this class needs is `parse(render()) == payload`, and a
        partial rehydration would be a second, weaker constructor.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ReconstructionSchemaError(
                "not valid JSON: {}".format(exc)) from None
        if not isinstance(payload, dict):
            raise ReconstructionSchemaError("a reconstruction must be an object")
        if payload.get("schema") != SCHEMA:
            raise ReconstructionSchemaError(
                "schema is {!r}, expected {!r}".format(
                    payload.get("schema"), SCHEMA))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ReconstructionSchemaError(
                "schema_version is {!r}; this parser judges version {} only"
                .format(payload.get("schema_version"), SCHEMA_VERSION))
        for required in ("subject_unit", "intended_legacy_alias",
                         "reconstruction_status", "reconstructed_at",
                         "failure", "repository", "recovered_fields",
                         "suite_transition", "acceptance", "targets"):
            if required not in payload:
                raise ReconstructionSchemaError(
                    "reconstruction missing {!r}".format(required))
        states = {f.get("state") for f in payload["recovered_fields"]}
        unknown = sorted(s for s in states
                         if s not in {k.value for k in KnowledgeState})
        if unknown:
            raise ReconstructionSchemaError(
                "unrecognised knowledge state(s) {}".format(unknown))
        return payload
