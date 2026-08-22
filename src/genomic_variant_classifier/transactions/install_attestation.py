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
from dataclasses import dataclass
from enum import Enum

SCHEMA = "gvc.install-attestation"
SCHEMA_VERSION = 2


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

REQUIRED_REPOSITORY = frozenset({"pre_head", "post_head"})
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


def validate(doc) -> None:
    """Refuse any document that is not a well-formed version-2 attestation."""
    if not isinstance(doc, dict):
        raise AttestationSchemaError("an attestation must be an object")
    if doc.get("schema") != SCHEMA:
        raise AttestationSchemaError(
            "schema is {!r}, expected {!r}".format(doc.get("schema"), SCHEMA))
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise AttestationSchemaError(
            "schema_version is {!r}, and this validator judges version {} only."
            " Version 1 documents are historical evidence: they are not "
            "migrated, not upgraded, and not judged against a schema written "
            "after they were recorded.".format(version, SCHEMA_VERSION))

    _exact_keys("attestation", doc, REQUIRED_TOP_LEVEL, OPTIONAL_TOP_LEVEL)
    _exact_keys("repository", doc["repository"], REQUIRED_REPOSITORY)
    _exact_keys("counter", doc["counter"], REQUIRED_COUNTER)
    _exact_keys("acceptance", doc["acceptance"], REQUIRED_ACCEPTANCE)
    _exact_keys("suite_transition", doc["suite_transition"],
                REQUIRED_TRANSITION, OPTIONAL_TRANSITION)

    status = doc["status"]
    if status not in {s.value for s in InstallStatus}:
        raise AttestationSchemaError("unknown status {!r}".format(status))
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
