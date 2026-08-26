"""An attestation is a schema, not a dictionary each installer invents.

ATTESTATION-SCHEMA-DRIFT-1, 2026-08-22.

WHY
===
Nine install attestations exist. All nine declare `"schema_version": 1`. They
have THREE different shapes, because every installer hand-built its own
dictionary and each one learned something the previous had not recorded:

    D0a         acceptance: {failed, passed, returncode, scope, seconds,
                             skipped, summary}
    D0b, D3a    the same, PLUS deselected, errors, measured_by, warning_kinds,
                warnings, xfailed, xpassed
    later units the same again, PLUS a top-level suite_transition; and one
                PLUS a top-level amendments

A version that does not change when the shape changes cannot be used to
interpret the document. A reader cannot know which fields to expect, and a
consumer written against one shape breaks silently on another -- the same
disease as four installers each carrying a private notion of "neutral", cured
in the same way: ONE typed owner.

The suite-transition unit at `a60f18f` produced identity digests that BELONG in
an attestation, and deliberately did not add them, because widening a corrupt
format to carry better evidence corrupts the evidence. This module is the
prerequisite that was owed first.

WHAT VERSION 2 ADDS BEYOND A FIXED FIELD LIST
=============================================
A field list alone would only prevent SHAPE drift. Version 2 also enforces
CROSS-FIELD CONSISTENCY, so an internally contradictory attestation cannot be
written at all:

    counter.after - counter.before  ==  |added| - |removed|
    suite_transition counts         ==  counter counts
    acceptance passed+skipped+xfailed == counter.after
    NEUTRAL                         =>  both observed sets empty AND the
                                        before and after identity digests equal
    ADDITION                        =>  added non-empty, removed empty, count rose
    DELIBERATE_RETIREMENT           =>  removed non-empty, justification present
    publication_error present       <=> status is INSTALL_APPLIED_PUBLICATION_PENDING

The acceptance-to-counter binding is the one that matters most. A gate summary
and a collection count are two measurements of the same suite, and until now
nothing required them to agree INSIDE the recorded evidence.

VERSION 1 DOCUMENTS ARE NOT MIGRATED
====================================
The nine existing attestations are historical evidence of what happened. They
are not rewritten, not upgraded, and not validated against version 2 -- a record
corrected in place is no longer a record. `validate` refuses to judge them and
says why. They are also, separately, ATTESTATION-NOT-PRESERVED-1: they exist
only in a downloads directory and are referenced by nine commit messages.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

SCHEMA = "gvc.install-attestation"
SCHEMA_VERSION = 3

#: MEASURED 2026-08-26, and the reason version 3 exists.
#:
#: Version 2 enforced CROSS-FIELD consistency -- counter deltas against
#: identity deltas, gate totals against collection counts, kind against the
#: observed sets -- and almost nothing about PRIMITIVE TYPES. An audit of the
#: preserved corpus applied the typing that
#: `install_attestation_reconstruction.py` had used since 2026-08-25:
#:
#:     version-2 documents preserved            8
#:     their ONLY typing failure                repository.pre_head, post_head
#:     every other typed field                  already conformed
#:
#: So the corpus was already fully typed but for one field pair -- and that
#: pair was the one that could not be tightened without changing every
#: producer. MEASURED across the delivered installers: 102 `rev-parse --short`
#: call sites and ZERO full ones. Not one installer ever captured a full object
#: identifier.
#:
#: VERSION 3 RECORDS BOTH. `pre_head` keeps the abbreviation git itself prints
#: and every historical attestation carries; `pre_head_oid` carries the full
#: forty characters. That follows the reconstruction schema, which already
#: records `commit_oid` AND `COMMIT_ABBREV`, and resolves a real tension: the
#: same repository was typing one concept two ways.
#:
#: AND THE TWO ARE CHECKED AGAINST EACH OTHER. Recording both is worth nothing
#: if they may disagree -- two independent fields would simply double the
#: surface for a wrong value. `_check_repository` requires the abbreviation to
#: be a PREFIX of the full identifier, which is the only relationship that
#: makes the pair evidence rather than decoration.
_OID = re.compile(r"\A[0-9a-f]{40}\Z")
_ABBREV = re.compile(r"\A[0-9a-f]{7,40}\Z")
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_UTC = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


class AttestationSchemaError(ValueError):
    """A document does not satisfy the attestation schema."""


class InstallStatus(str, Enum):
    PUBLISHED = "PUBLISHED"
    INSTALL_APPLIED_PUBLICATION_PENDING = "INSTALL_APPLIED_PUBLICATION_PENDING"


REQUIRED_TOP_LEVEL = frozenset({
    "schema", "schema_version", "unit", "plan_digest", "status",
    "started_at", "finished_at", "python", "platform",
    "repository", "counter", "acceptance", "targets", "suite_transition",
})
OPTIONAL_TOP_LEVEL = frozenset({
    "amendments", "invariant_migrations", "publication_error",
})

REQUIRED_REPOSITORY = frozenset({
    "pre_head", "post_head", "pre_head_oid", "post_head_oid",
})
REQUIRED_COUNTER = frozenset({"scope", "before", "after"})
REQUIRED_ACCEPTANCE = frozenset({
    "scope", "returncode", "passed", "skipped", "failed", "errors",
    "xfailed", "xpassed", "warnings", "summary", "seconds", "measured_by",
})
REQUIRED_TRANSITION = frozenset({
    "kind", "expected_added_nodeids", "expected_removed_nodeids",
    "observed_added_nodeids", "observed_removed_nodeids",
    "before_count", "after_count", "before_digest", "after_digest",
})
OPTIONAL_TRANSITION = frozenset({"justification"})
REQUIRED_TARGET = frozenset({"path", "action", "post_sha256", "post_size"})

VALID_KINDS = frozenset({"addition", "neutral", "deliberate_retirement"})


def _exact_keys(where: str, obj, required: frozenset,
                optional: frozenset = frozenset()) -> None:
    if not isinstance(obj, dict):
        raise AttestationSchemaError("{} must be an object, not {}".format(
            where, type(obj).__name__))
    keys = set(obj)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise AttestationSchemaError("{}: missing {}".format(where, missing))
    if unknown:
        raise AttestationSchemaError(
            "{}: unknown key(s) {}. An attestation with an undeclared field is "
            "how ATTESTATION-SCHEMA-DRIFT-1 happened: nine documents, one "
            "declared version, three shapes. Add the field to the schema and "
            "raise the version, or do not write it.".format(where, unknown))


def _typed(where: str, value, pattern, want: str) -> None:
    """A string field whose SHAPE is part of its meaning."""
    if not isinstance(value, str) or not pattern.match(value):
        raise AttestationSchemaError(
            "{} is {!r}; expected {}. Version 2 accepted any value here, and "
            "an attestation whose digest is not a digest is not evidence."
            .format(where, value, want))


def _counted(where: str, value) -> None:
    """An integer field. `True` is an int in Python and is not a count."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise AttestationSchemaError(
            "{} is {!r} of type {}; expected an integer".format(
                where, value, type(value).__name__))


def _check_repository(repo: dict, status: str) -> None:
    """Both identifiers, and the relationship that makes the pair evidence.

    An abbreviation and a full identifier recorded independently would double
    the surface for a wrong value while proving nothing. Requiring the short
    form to be a PREFIX of the long one means a single transcription error is
    caught here rather than by a reader years later.

    `post_head` and `post_head_oid` are BOTH null when publication did not
    reach a commit -- and both must be, together. One null beside one value
    would describe a state that cannot exist.
    """
    _typed("repository.pre_head", repo["pre_head"], _ABBREV,
           "7 to 40 lowercase hexadecimal characters")
    _typed("repository.pre_head_oid", repo["pre_head_oid"], _OID,
           "a full 40-character lowercase object identifier")
    if not repo["pre_head_oid"].startswith(repo["pre_head"]):
        raise AttestationSchemaError(
            "repository.pre_head {!r} is not a prefix of pre_head_oid {!r}. "
            "Recording both is evidence only if they agree."
            .format(repo["pre_head"], repo["pre_head_oid"]))

    pending = status == InstallStatus.INSTALL_APPLIED_PUBLICATION_PENDING.value
    short, full = repo["post_head"], repo["post_head_oid"]
    if short is None or full is None:
        if not pending:
            raise AttestationSchemaError(
                "repository.post_head is null but the status is {!r}. A "
                "published install has a commit.".format(status))
        if not (short is None and full is None):
            raise AttestationSchemaError(
                "repository.post_head is {!r} and post_head_oid is {!r}. A "
                "pending install has neither; one of each describes a state "
                "that cannot exist.".format(short, full))
        return
    _typed("repository.post_head", short, _ABBREV,
           "7 to 40 lowercase hexadecimal characters")
    _typed("repository.post_head_oid", full, _OID,
           "a full 40-character lowercase object identifier")
    if not full.startswith(short):
        raise AttestationSchemaError(
            "repository.post_head {!r} is not a prefix of post_head_oid {!r}"
            .format(short, full))


def validate(doc) -> None:
    """Refuse any document that is not a well-formed version-3 attestation."""
    if not isinstance(doc, dict):
        raise AttestationSchemaError("an attestation must be an object")
    if doc.get("schema") != SCHEMA:
        raise AttestationSchemaError(
            "schema is {!r}, expected {!r}".format(doc.get("schema"), SCHEMA))
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise AttestationSchemaError(
            "schema_version is {!r}, and this validator judges version {} only."
            " Versions 1 and 2 are historical evidence: they are not migrated, "
            "not upgraded, and not judged against a schema written after they "
            "were recorded. MEASURED 2026-08-26: nine version-1 and eight "
            "version-2 documents are preserved, and they stay exactly as they "
            "were emitted.".format(version, SCHEMA_VERSION))

    _exact_keys("attestation", doc, REQUIRED_TOP_LEVEL, OPTIONAL_TOP_LEVEL)
    _exact_keys("repository", doc["repository"], REQUIRED_REPOSITORY)
    _exact_keys("counter", doc["counter"], REQUIRED_COUNTER)
    _exact_keys("acceptance", doc["acceptance"], REQUIRED_ACCEPTANCE)
    _exact_keys("suite_transition", doc["suite_transition"],
                REQUIRED_TRANSITION, OPTIONAL_TRANSITION)

    status = doc["status"]
    if status not in {s.value for s in InstallStatus}:
        raise AttestationSchemaError("unknown status {!r}".format(status))

    # ---- version-3 typing, in the order a reader would check it ----------
    _check_repository(doc["repository"], status)
    for field in ("started_at", "finished_at"):
        _typed(field, doc[field], _UTC, "YYYY-MM-DDTHH:MM:SSZ")
    _typed("plan_digest", doc["plan_digest"], _HEX64,
           "a 64-character lowercase SHA-256 digest")
    for field in ("before_digest", "after_digest"):
        _typed("suite_transition." + field, doc["suite_transition"][field],
               _HEX64, "a 64-character lowercase SHA-256 digest")
    for field in ("before_count", "after_count"):
        _counted("suite_transition." + field, doc["suite_transition"][field])
    for field in ("before", "after"):
        _counted("counter." + field, doc["counter"][field])
    for field in ("passed", "skipped", "failed", "errors", "xfailed",
                  "xpassed", "warnings", "returncode"):
        _counted("acceptance." + field, doc["acceptance"][field])
    if not isinstance(doc["acceptance"]["seconds"], (int, float)) or \
            isinstance(doc["acceptance"]["seconds"], bool):
        raise AttestationSchemaError(
            "acceptance.seconds is {!r}; expected a number".format(
                doc["acceptance"]["seconds"]))
    pending = status == InstallStatus.INSTALL_APPLIED_PUBLICATION_PENDING.value
    has_error = "publication_error" in doc
    if pending != has_error:
        raise AttestationSchemaError(
            "publication_error must be present exactly when status is "
            "INSTALL_APPLIED_PUBLICATION_PENDING (status={!r}, present={})"
            .format(status, has_error))

    if not isinstance(doc["targets"], list) or not doc["targets"]:
        raise AttestationSchemaError(
            "targets must be a non-empty list: an install that wrote nothing "
            "is not an install")
    for i, t in enumerate(doc["targets"]):
        _exact_keys("targets[{}]".format(i), t, REQUIRED_TARGET)
        _typed("targets[{}].post_sha256".format(i), t["post_sha256"], _HEX64,
               "a 64-character lowercase SHA-256 digest")
        _counted("targets[{}].post_size".format(i), t["post_size"])

    tr = doc["suite_transition"]
    if tr["kind"] not in VALID_KINDS:
        raise AttestationSchemaError(
            "unknown transition kind {!r}; expected one of {}".format(
                tr["kind"], sorted(VALID_KINDS)))

    added = set(tr["observed_added_nodeids"])
    removed = set(tr["observed_removed_nodeids"])
    if added & removed:
        raise AttestationSchemaError(
            "identities observed both added and removed: {}".format(
                sorted(added & removed)))
    if set(tr["expected_added_nodeids"]) != added:
        raise AttestationSchemaError(
            "expected added identities do not equal the observed ones")
    if set(tr["expected_removed_nodeids"]) != removed:
        raise AttestationSchemaError(
            "expected removed identities do not equal the observed ones")

    cbefore, cafter = doc["counter"]["before"], doc["counter"]["after"]
    if (tr["before_count"], tr["after_count"]) != (cbefore, cafter):
        raise AttestationSchemaError(
            "suite_transition counts {} do not equal counter counts {}".format(
                (tr["before_count"], tr["after_count"]), (cbefore, cafter)))
    if cafter - cbefore != len(added) - len(removed):
        raise AttestationSchemaError(
            "counter delta {:+d} disagrees with identity delta {:+d}".format(
                cafter - cbefore, len(added) - len(removed)))

    if tr["kind"] == "neutral":
        if added or removed:
            raise AttestationSchemaError(
                "a neutral transition observed {} added and {} removed "
                "identities".format(len(added), len(removed)))
        if tr["before_digest"] != tr["after_digest"]:
            raise AttestationSchemaError(
                "a neutral transition recorded different identity digests: "
                "{} -> {}".format(tr["before_digest"][:16],
                                  tr["after_digest"][:16]))
    elif tr["kind"] == "addition":
        if not added or removed:
            raise AttestationSchemaError(
                "an addition observed {} added and {} removed".format(
                    len(added), len(removed)))
    else:
        if not removed:
            raise AttestationSchemaError(
                "a deliberate retirement observed no removals")
        if not str(tr.get("justification", "")).strip():
            raise AttestationSchemaError(
                "a deliberate retirement requires a justification")

    acc = doc["acceptance"]
    accounted = acc["passed"] + acc["skipped"] + acc["xfailed"]
    if accounted != cafter:
        raise AttestationSchemaError(
            "the gate accounts for {} case(s) but the counter records {}. A "
            "gate summary and a collection count are two measurements of one "
            "suite; an attestation in which they disagree is not evidence."
            .format(accounted, cafter))
    if acc["returncode"] != 0 and status == InstallStatus.PUBLISHED.value:
        raise AttestationSchemaError(
            "status PUBLISHED with a nonzero gate return code {}".format(
                acc["returncode"]))


@dataclass(frozen=True)
class AttestationDocument:
    """A validated attestation. Construction is the validation."""

    payload: dict

    def __post_init__(self) -> None:
        validate(self.payload)

    def to_json(self) -> str:
        return json.dumps(self.payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "AttestationDocument":
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise AttestationSchemaError(
                "not valid JSON: {}".format(exc)) from exc
        return cls(payload=payload)


class PublicationError(RuntimeError):
    """Evidence could not be written to the place it was asked to go."""


def publish(document: "AttestationDocument", destination) -> "Path":
    """Write a VALIDATED attestation. The only way evidence reaches disk.

    PENDING-ATTESTATION-BYPASSES-SCHEMA-VALIDATION-1, repaired 2026-08-26.

    Until this function existed, `AttestationDocument` validated and serialised
    but did not WRITE -- so every caller opened a file itself, and two paths
    diverged:

        success   payload -> AttestationDocument -> to_json -> caller writes
        pending   payload -> json.dumps -------------------> caller writes

    MEASURED 2026-08-26 across THIRTY-THREE delivered installers: every single
    `except PublicationPending` handler took the second path. Not twenty-two, as
    a stale census claimed -- and the count includes four installers written on
    2026-08-25 by an author who had just applied the opposing rule to the
    transition record and the target record in those same files.

    THE PENDING STATE WAS ALWAYS VALIDATABLE. `InstallStatus` declares
    INSTALL_APPLIED_PUBLICATION_PENDING; `validate` requires `publication_error`
    to be present exactly when the status is pending; and nothing constrains
    `post_head`'s VALUE, only that the key exists. The schema anticipated this
    state from the start. The pending path simply never used it.

    So there is no second function here for pending documents. There is ONE
    path, and a pending attestation reaches it by being CONSTRUCTED like any
    other -- which is where validation happens.

    WHAT THIS REFUSES, AND WHY EACH REFUSAL IS NOT PARANOIA:

      - a payload that is not an AttestationDocument. A dict cannot be
        published, because a dict has not been validated. The type IS the
        proof.
      - a destination that already exists. Evidence is written once; silently
        overwriting an attestation would destroy the record of what an earlier
        install claimed.
      - a destination whose parent does not exist. Creating directories to
        store evidence hides a misconfigured path until an audit cannot find
        the artifact.

    AND IT RE-PARSES ITS OWN OUTPUT before returning. `to_json` could in
    principle emit text that no longer validates -- a non-serialisable value
    coerced by `default=str`, for instance. Rendering and re-reading proves the
    BYTES are a valid attestation, not merely the object that produced them.
    """
    from pathlib import Path as _Path

    if not isinstance(document, AttestationDocument):
        raise PublicationError(
            "publish() takes an AttestationDocument, not {}. A raw payload has "
            "not been validated, and the type is what proves it was."
            .format(type(document).__name__))

    target = _Path(destination)
    if target.exists():
        raise PublicationError(
            "{} already exists. Evidence is written once; overwriting an "
            "attestation would destroy the record of what an earlier install "
            "claimed.".format(target))
    if not target.parent.is_dir():
        raise PublicationError(
            "the parent directory {} does not exist. Creating directories to "
            "store evidence hides a misconfigured path until an audit cannot "
            "find the artifact.".format(target.parent))

    text = document.to_json()
    # Proves the BYTES validate, not merely the object that produced them.
    AttestationDocument.from_json(text)
    with open(str(target), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return target
