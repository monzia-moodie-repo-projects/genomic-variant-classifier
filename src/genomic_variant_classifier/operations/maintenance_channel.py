"""The maintenance-evidence channel, and its qualification MEASURED.

Author: Monzia Moodie

THE TERMINATING OBLIGATION
==========================
    A verified archive-admission operation creates a MAINTENANCE-EVIDENCE
    obligation. Satisfying it creates no new installation-archive admission
    obligation.

That is not "no evidence required". It is a finite obligation with a defined
destination, and completion depends on the evidence being ACTUALLY PRESERVED.

WHY THIS IS NOT A CONTRACT DICTIONARY
=====================================
An earlier attempt expressed the channel as a mapping of descriptive strings
whose self-test established that keys existed. That made nothing
machine-verified: retention, cleanup exclusion and replication were prose.

Here the two concepts are separated:

    POLICY                 the required behaviour, declared
    QUALIFICATION EVIDENCE measurements showing the configured implementation
                           provides it

`ChannelQualification`'s Booleans are produced ONLY by `qualify_channel`,
which runs the measurements and binds their transcript's digest. A caller
cannot set them to True.

FOUR ASSURANCES, NEVER COLLAPSED INTO ONE
=========================================
    process-crash recovery  an interrupted writer and a lost acknowledgment
                            recover correctly
    cleanup survival        transaction and cache cleanup preserve retained
                            evidence
    storage-loss recovery   a separate backup restores the evidence AND the
                            dependencies needed to interpret it
    power-loss durability   platform and filesystem persistence behaviour is
                            qualified

`durable: true` would hide which of these was actually established. The fourth
is NOT attempted here and is reported False with its reason.

WHERE IT LIVES
==============
`<operation intent root>/<operation id>/evidence/`, owned by OperationRecord.
No new RuntimePaths root: `operation_intent` already provides the root
authority, and directory composition alone does not justify duplicating it.

EXIT STATUS when run as a script
  0  every declared measurement held
  2  a measurement did not hold
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

EVIDENCE_DIRNAME = "evidence"
EVIDENCE_SCHEMA = "gvc.maintenance-evidence"
EVIDENCE_VERSION = 1
RETAINED_MARKER = ".retained"

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_OID = re.compile(r"\A[0-9a-f]{40}\Z")

REQUIRED_EVIDENCE = {
    "schema", "schema_version", "operation_id", "operation_kind",
    "policy_identity", "plan_sha256", "predecessor_commit",
    "candidate_commit", "integrated_commit", "preimage_manifest_sha256",
    "postimage_manifest_sha256", "approved_entries_sha256",
    "validation_evidence_sha256", "admitted_records", "preserved_records",
    "resulting_records", "completion_route", "recorded_at_utc",
}


class ChannelError(RuntimeError):
    """The channel refuses. It never repairs and never defaults."""


def evidence_directory(operation_directory: Path) -> Path:
    """Owned by the operation, not by a new runtime root."""
    return Path(operation_directory) / EVIDENCE_DIRNAME


def _exact_int(value) -> bool:
    return type(value) is int


def validate_evidence(document) -> dict:
    if not isinstance(document, dict):
        raise ChannelError("the maintenance evidence is not an object")
    if set(document) != REQUIRED_EVIDENCE:
        raise ChannelError(
            "evidence field set is not exactly the supported one: unexpected "
            "{}, missing {}".format(
                sorted(set(document) - REQUIRED_EVIDENCE),
                sorted(REQUIRED_EVIDENCE - set(document))))
    if document["schema"] != EVIDENCE_SCHEMA:
        raise ChannelError("unsupported schema {!r}".format(document["schema"]))
    if not _exact_int(document["schema_version"]) or \
            document["schema_version"] != EVIDENCE_VERSION:
        raise ChannelError(
            "schema_version must be the integer {}, not {!r} of type "
            "{}".format(EVIDENCE_VERSION, document["schema_version"],
                        type(document["schema_version"]).__name__))
    if document["operation_kind"] != "archive_admission":
        raise ChannelError(
            "this channel preserves archive-admission evidence, not {!r}"
            .format(document["operation_kind"]))
    if document["completion_route"] not in ("normal", "recovery_finalization"):
        raise ChannelError(
            "unsupported completion route {!r}".format(
                document["completion_route"]))
    for field in ("predecessor_commit", "candidate_commit",
                  "integrated_commit"):
        value = document[field]
        if type(value) is not str or not _OID.fullmatch(value):
            raise ChannelError(
                "{} must be 40 lowercase hexadecimal digits".format(field))
    for field in ("plan_sha256", "policy_identity", "preimage_manifest_sha256",
                  "postimage_manifest_sha256", "approved_entries_sha256",
                  "validation_evidence_sha256"):
        value = document[field]
        if type(value) is not str or not _SHA256.fullmatch(value):
            raise ChannelError(
                "{} must be 64 lowercase hexadecimal digits".format(field))
    for field in ("admitted_records", "preserved_records",
                  "resulting_records"):
        value = document[field]
        if not _exact_int(value) or value < 0:
            raise ChannelError(
                "{} must be a nonnegative integer".format(field))
    if document["preserved_records"] + document["admitted_records"] != \
            document["resulting_records"]:
        raise ChannelError(
            "{} preserved plus {} admitted is not {} resulting".format(
                document["preserved_records"], document["admitted_records"],
                document["resulting_records"]))
    return document


#: The publication phases a crash experiment can stop at. Named so an
#: experiment states WHICH boundary it interrupted rather than "somewhere in
#: publication" -- before synchronisation, after it, and after the final name
#: exists are three different recovery situations.
PHASES = ("before_synchronisation", "after_synchronisation",
          "after_final_install")


def publish_evidence(operation_directory: Path, document,
                     on_phase=None) -> str:
    """Stage, synchronise, then install under the final name WITHOUT replacing.

    The same ordering the receipt publisher uses, and for the same measured
    reason: writing directly to the final name leaves a syntactically valid
    record there when the synchronisation raises.
    """
    validate_evidence(document)
    directory = evidence_directory(operation_directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RETAINED_MARKER).write_bytes(b"retained\n")
    final = directory / "maintenance_evidence.json"
    payload = (json.dumps(document, indent=2, sort_keys=True,
                          ensure_ascii=True) + "\n").encode("utf-8")
    staging = directory / ".evidence-{}.pending".format(uuid.uuid4().hex)
    with staging.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        # `on_phase` is INERT unless a caller supplies it. It exists so a
        # crash experiment can stop inside the REAL publication sequence: a
        # child that re-implemented these steps would be testing a copy, and
        # the copy is exactly where a divergence would hide.
        if on_phase is not None:
            on_phase("before_synchronisation")
        os.fsync(handle.fileno())
        if on_phase is not None:
            on_phase("after_synchronisation")
    if staging.read_bytes() != payload:
        raise ChannelError("the staged evidence does not match what was "
                           "written")
    try:
        os.link(staging, final)
    except FileExistsError as exc:
        raise ChannelError(
            "{} already exists. That is a reconciliation case, not an "
            "overwrite.".format(final)) from exc
    except OSError as exc:
        raise ChannelError(
            "no-replace installation is unsupported here: {}. Copying into "
            "the final name is not an acceptable fallback.".format(exc)
        ) from exc
    if on_phase is not None:
        on_phase("after_final_install")
    if final.read_bytes() != payload:
        raise ChannelError("the installed evidence does not match the staging "
                           "bytes")
    return hashlib.sha256(payload).hexdigest()


def read_evidence(operation_directory: Path):
    final = evidence_directory(operation_directory) / "maintenance_evidence.json"
    # ABSENCE, DAMAGE AND INACCESSIBILITY ARE THREE ANSWERS.
    #
    # `if not final.is_file(): return None` reports a DIRECTORY at that path
    # as absent -- the same category error unit S was installed to eliminate.
    try:
        raw = final.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ChannelError(
            "{} exists or is inaccessible and could not be inspected: "
            "{}".format(final, exc)) from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChannelError(
            "{} is unreadable: {}. The bytes are preserved.".format(final, exc)
        ) from exc
    return validate_evidence(document)


def is_retained(operation_directory: Path) -> bool:
    """Retention is a MARKER ON DISK, not a value a cleanup routine infers."""
    return (evidence_directory(operation_directory)
            / RETAINED_MARKER).is_file()


def enumerate_evidence(intent_root: Path) -> list:
    """Every operation's evidence, INDEPENDENT of transaction journals.

    Unreadable records are reported, never skipped -- an enumeration that
    skipped them would be blind to exactly the damage it exists to find.
    """
    root = Path(intent_root)
    if root.exists() and not root.is_dir():
        raise ChannelError(
            "the intent root {} exists and is not a directory".format(root))
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        directory = evidence_directory(child)
        if not directory.is_dir():
            continue
        row = {"operation_id": child.name, "retained": is_retained(child),
               "state": "absent", "detail": None}
        try:
            document = read_evidence(child)
            if document is None:
                row["detail"] = "no evidence record"
            else:
                row.update({"state": "readable",
                            "integrated_commit": document["integrated_commit"],
                            "admitted_records": document["admitted_records"]})
        except ChannelError as exc:
            row.update({"state": "unreadable", "detail": str(exc)[:160]})
        found.append(row)
    return found


class CleanupRefused(ChannelError):
    """Cleanup declined to remove an operation. Never a silent skip."""


def cleanup_operations(intent_root: Path, *, remove_resolved,
                       exclusion_for=None) -> dict:
    """Cleanup that acquires the SAME coordination lifecycle mutations use.

    MEASURED 2026-09-08 against the previous implementation: it checked
    `.retained` and later called `shutil.rmtree` with nothing held in between,
    so a publication could land in that window. It also treated a MISSING
    marker beside existing evidence as permission to delete.

    The rules now are:

        acquire the operation's exclusion, then RE-OBSERVE under it
        refuse any operation that bears evidence, retained or not
        refuse a missing marker beside evidence -- that is DAMAGE to
            investigate, not permission
        refuse anything that cannot be observed

    `exclusion_for` is a callable returning a context manager for an operation
    identifier. When it is None no coordination is available and cleanup
    refuses everything rather than proceeding uncoordinated: an uncoordinated
    delete is the failure mode this function exists to prevent.

    Returns a report of what was removed AND what was refused with reasons.
    Testing THIS function does not demonstrate that the repository's actual
    transaction cleanup or cache eviction exclude the subtree; those are
    separate claims about separate code.
    """
    root = Path(intent_root)
    report = {"removed": [], "refused": {}}
    if not root.is_dir():
        return report
    if exclusion_for is None:
        for name in sorted(remove_resolved):
            report["refused"][name] = (
                "no coordination mechanism was supplied; an uncoordinated "
                "delete is what this function exists to prevent")
        return report

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name not in remove_resolved:
            continue
        try:
            with exclusion_for(child.name):
                # RE-OBSERVE under the exclusion. The earlier observation was
                # made before anything was held and may be stale.
                directory = evidence_directory(child)
                if directory.exists():
                    if not is_retained(child):
                        report["refused"][child.name] = (
                            "evidence is present and the retention marker is "
                            "ABSENT. That is damage to investigate, not "
                            "permission to delete.")
                    else:
                        report["refused"][child.name] = (
                            "the operation bears retained evidence")
                    continue
                shutil.rmtree(child)
                report["removed"].append(child.name)
        except Exception as exc:                          # noqa: BLE001
            report["refused"][child.name] = (
                "could not be observed under coordination: {}: {}".format(
                    type(exc).__name__, exc))
    return report


SUPPORTED_BACKUP_FILES = frozenset({
    "specification.json", "progress.json", "completion_receipt.json",
    "{}/maintenance_evidence.json".format(EVIDENCE_DIRNAME),
    "{}/{}".format(EVIDENCE_DIRNAME, RETAINED_MARKER),
})

#: What a COMPLETED operation's backup must carry to remain interpretable.
#: MEASURED 2026-09-08: back_up_evidence silently skipped an absent
#: specification and receipt, so "backup succeeded" established nothing about
#: whether the record could later be understood.
REQUIRED_BACKUP_FILES = frozenset({
    "specification.json", "completion_receipt.json",
    "{}/maintenance_evidence.json".format(EVIDENCE_DIRNAME),
})


def require_backup_population(files, required=REQUIRED_BACKUP_FILES) -> None:
    if type(files) is not dict:
        raise ChannelError("backup files must be an object")
    names = set(files)
    unsupported = sorted(names - SUPPORTED_BACKUP_FILES)
    if unsupported:
        raise ChannelError(
            "unsupported backup member(s): {}. An allowlist is used because "
            "a manifest path from ordinary json.loads is caller-controlled: "
            "MEASURED 2026-09-08, '../source.txt' was accepted.".format(
                unsupported[:5]))
    missing = sorted(required - names)
    if missing:
        raise ChannelError(
            "required interpreting dependenc(ies) absent from the backup: "
            "{}".format(missing))


def back_up_evidence(operation_directory: Path, destination: Path, *,
                     required=None) -> dict:
    """Back up the evidence AND the dependencies needed to interpret it.

    An isolated receipt whose specification is unavailable cannot be verified
    after a restore, so the specification and completion receipt travel with
    it. What is backed up is enumerated in the returned manifest rather than
    implied by a directory copy.
    """
    source = Path(operation_directory)
    destination = Path(destination)
    if destination.exists():
        raise ChannelError(
            "{} already exists; a backup never replaces one".format(
                destination))
    if required is None:
        required = REQUIRED_BACKUP_FILES
    destination.mkdir(parents=True)
    manifest = {"schema": "gvc.maintenance-evidence-backup",
                "schema_version": 1, "files": {}}
    for relative in ("specification.json", "progress.json",
                     "completion_receipt.json",
                     "{}/maintenance_evidence.json".format(EVIDENCE_DIRNAME),
                     "{}/{}".format(EVIDENCE_DIRNAME, RETAINED_MARKER)):
        origin = source / relative
        if not origin.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = origin.read_bytes()
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        manifest["files"][relative] = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}
    require_backup_population(manifest["files"], required)
    raw = (json.dumps(manifest, indent=2, sort_keys=True,
                      ensure_ascii=True) + "\n").encode("utf-8")
    with (destination / "backup_manifest.json").open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    # The manifest's OWN digest, so a restore can bind it to something
    # recorded elsewhere rather than trusting the file beside the payload.
    manifest["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return manifest


def restore_evidence(backup: Path, operation_directory: Path, *,
                     expected_manifest_sha256: str,
                     required=REQUIRED_BACKUP_FILES) -> dict:
    """Verify EVERYTHING, then publish without replacing.

    MEASURED 2026-09-08 against the previous implementation:

        a manifest naming '../source.txt'   ACCEPTED
        an existing destination b"old"      REPLACED with b"new"

    It trusted relative paths from an unvalidated manifest, wrote with
    `write_bytes`, and could publish earlier files before discovering
    corruption in a later one.

    The manifest's own identity is now bound to a digest recorded elsewhere,
    every member is checked against an allowlist, ALL source bytes are
    verified before anything is published, the restore lands in a newly
    created staging directory, and an existing destination is a reconciliation
    or conflict -- never an overwrite.
    """
    backup = Path(backup)
    target = Path(operation_directory)
    raw = (backup / "backup_manifest.json").read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_manifest_sha256:
        raise ChannelError(
            "the backup manifest is {} and the approved digest is {}".format(
                observed, expected_manifest_sha256))
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema") != "gvc.maintenance-evidence-backup" or \
            manifest.get("schema_version") != 1:
        raise ChannelError("unsupported backup manifest schema")
    files = manifest.get("files")
    require_backup_population(files, required)

    # VERIFY ALL SOURCE BYTES FIRST. Publishing a prefix of a corrupt backup
    # leaves a half-restored operation that looks present.
    payloads = {}
    for relative in sorted(files):
        spec = files[relative]
        source = backup / relative
        try:
            payload = source.read_bytes()
        except OSError as exc:
            raise ChannelError(
                "{}: the backup copy could not be read: {}".format(
                    relative, exc)) from exc
        if len(payload) != spec.get("size_bytes") or \
                hashlib.sha256(payload).hexdigest() != spec.get("sha256"):
            raise ChannelError(
                "{}: the backup copy does not match its manifest".format(
                    relative))
        payloads[relative] = payload

    if target.exists():
        raise ChannelError(
            "{} already exists. An existing operation is a reconciliation "
            "case, not a restore destination.".format(target))
    staging = target.parent / ".restore-{}.pending".format(uuid.uuid4().hex)
    staging.mkdir(parents=True)
    for relative, payload in sorted(payloads.items()):
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    # The restored record must be readable through its OWNER before the
    # operation is published under its real name.
    validate_evidence(json.loads(
        (staging / "{}/maintenance_evidence.json".format(EVIDENCE_DIRNAME))
        .read_bytes().decode("utf-8")))
    try:
        os.rename(staging, target)
    except OSError as exc:
        raise ChannelError(
            "the restored operation could not be published without "
            "replacement: {}".format(exc)) from exc
    return {r: files[r]["sha256"] for r in sorted(files)}


@dataclass(frozen=True)
class ChannelQualification:
    """What a qualification RUN recorded.

    THIS IS NOT A CAPABILITY. MEASURED 2026-09-08: a caller constructed one
    claiming full qualification, including power-loss durability, in a single
    expression. An earlier docstring here said "a caller cannot set them to
    True"; that was false, and a frozen dataclass never prevented it.

    What the fields mean is that SOME run recorded them. A consumer must bind
    `test_evidence_sha256` to a transcript it independently trusts, and check
    `platform` and `scope` -- a Linux result does not qualify Windows, and a
    backup beside its original in one workspace does not establish survival of
    storage-device loss.

    The destructive orchestration that produces one lives in test support or
    an explicitly invoked qualification command, NOT in this library: killing
    processes is not something importing an evidence module should enable.
    """

    policy_sha256: str
    implementation_sha256: str
    test_evidence_sha256: str
    platform: str
    scope: str
    process_crash_recovery_verified: bool
    cleanup_exclusion_verified: bool
    restoration_verified: bool
    power_loss_durability_verified: bool
    unverified_reasons: tuple

    def requires_operator_review(self) -> tuple:
        """Everything this run did NOT establish, named."""
        return self.unverified_reasons


POLICY = {
    "schema": "gvc.maintenance-channel-policy", "schema_version": 1,
    "terminating_rule": "a verified archive-admission operation creates a "
                        "maintenance-evidence obligation; satisfying it "
                        "creates no new installation-archive admission "
                        "obligation",
    "location": "<operation intent root>/<operation id>/evidence/",
    "publication": "validate, stage, synchronise, install without replacing, "
                   "verify",
    "retention": "permanent while the archive history it explains is "
                 "retained; marked on disk and excluded from cleanup",
    "discovery": "enumerated independently of transaction journals",
    "assurances_required": ["process_crash_recovery", "cleanup_survival",
                            "storage_loss_recovery"],
    "assurances_not_claimed": ["power_loss_durability"],
}


def policy_digest() -> str:
    return hashlib.sha256(
        (json.dumps(POLICY, indent=2, sort_keys=True, ensure_ascii=True)
         + "\n").encode("utf-8")).hexdigest()
