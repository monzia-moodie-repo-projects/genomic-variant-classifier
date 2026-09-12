"""Retain what was observed, before anything interprets it.

Author: Monzia Moodie

WHY THIS EXISTS
===============
MEASURED 2026-09-11 against the installer at
66772b3a9de7779087c2380e865194489c91a8644427ebfbfb35bd4a2e96bd7a:

    collect_output   line 388  capture_output=True -- the bytes EXIST
                     line 392  proc.stdout.decode("utf-8", "replace")
                     returns a str; `proc` leaves scope; the bytes are gone

    three call sites  baseline 690, prospective-in-rollback 706, recheck 880
    run_gate line 403 has the same shape for EXECUTION evidence

So the problem is RETENTION, not an inability to obtain the output. An outer
transcript is not a substitute: it is captured separately and need not contain
the inner subprocess output.

The consequence is already on the record. The unit-O interpretation migration
had to be RECONSTRUCTED from ten identities recorded in prose, because the raw
collection output from f808944 was never kept. A reconstruction is examples; it
is not the corpus.

WHAT THIS DOES NOT DO
=====================
It does not establish repository-subject binding, approval, or admission. It
retains bytes and the facts of their production. Binding those to an approved
operation is the admission boundary's work, and this module deliberately cannot
do it.

Nor does it improve the decoding. `decode("utf-8", "replace")` is PRESERVED and
DECLARED, because replacement decoding maps distinct invalid byte sequences to
the same text -- so the decoder is part of the interpretation contract, and
changing it is a measured behaviour change, not a tidy-up inside a retention
unit. Keeping the raw bytes is what makes that later comparison possible.

ORDER MATTERS
=============
Retention happens BEFORE interpretation and BEFORE the exit-status refusal, so
a parse failure or a nonzero exit has its evidence already persisted rather
than discarded by the raise that reports it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

#: The decoding procedure this capture DECLARES. Named, not assumed.
#:
#: "replace" can map distinct invalid byte sequences to one text, so two
#: different raw outputs can yield one snapshot. That is a property of the
#: procedure, and it belongs in the record rather than in a reader's memory.
DECODE_PROCEDURE = "utf-8/replace"


class CaptureError(RuntimeError):
    """Capture or retention failed. Distinct from the subprocess failing."""


class RetentionFailed(CaptureError):
    """The bytes could not be retained. The operation must not continue.

    Separate from the subprocess outcome: a successful collection whose
    evidence could not be stored is not an acceptable input to acceptance.
    """

    code = "COLLECTION_EVIDENCE_RETENTION_FAILED"


class SubjectKind(str, Enum):
    """What the observation was taken OVER.

    MEASURED 2026-09-11: the installer collects three times, and TWO of those
    run while repository content is modified inside a transaction. Stamping the
    predecessor commit onto those would describe files that are not the ones
    that were read.
    """

    #: A clean checkout at a named commit. The commit identifies the content.
    COMMITTED_TREE = "committed_tree"

    #: A working tree modified inside a transaction. The commit does NOT
    #: identify the content; a separate measurement must.
    MEASURED_TRANSACTION_STATE = "measured_transaction_state"


class CollectionPhase(str, Enum):
    """Which observation this is. Named so two cannot be confused."""

    BASELINE_COLLECTION = "baseline_collection"
    PROSPECTIVE_COLLECTION = "prospective_collection"
    APPLY_COLLECTION = "apply_collection"
    ACCEPTANCE_EXECUTION = "acceptance_execution"


@dataclass(frozen=True)
class Subject:
    """What the observation was taken over, and how that was established."""

    kind: SubjectKind
    commit: str = ""
    #: For a modified tree: how the content was measured. NOT a commit.
    state_reference: str = ""

    def __post_init__(self) -> None:
        if type(self.kind) is not SubjectKind:
            raise CaptureError(
                "subject kind must be a SubjectKind, not {!r}".format(
                    self.kind))
        if self.kind is SubjectKind.COMMITTED_TREE:
            if not self.commit:
                raise CaptureError(
                    "a committed-tree subject must name its commit")
            if self.state_reference:
                raise CaptureError(
                    "a committed-tree subject may not also carry a "
                    "state_reference; the commit identifies the content")
        else:
            if not self.state_reference:
                raise CaptureError(
                    "a measured-transaction-state subject must carry a "
                    "state_reference. The predecessor commit does NOT describe "
                    "modified files, and stamping it here would assert that "
                    "it does.")


@dataclass(frozen=True)
class CaptureRecord:
    """The facts of one collection, with its bytes retained beside it."""

    phase: CollectionPhase
    subject: Subject
    argv: tuple
    cwd: str
    returncode: int
    stdout_sha256: str
    stdout_bytes: int
    stderr_sha256: str
    stderr_bytes: int
    decode_procedure: str
    interpreter: str
    started_at: str
    seconds: float
    timed_out: bool = False
    environment: dict = field(default_factory=dict)

    def as_document(self) -> dict:
        return {
            "schema": "gvc.collection-capture",
            "schema_version": 1,
            "phase": self.phase.value,
            "subject": {"kind": self.subject.kind.value,
                        "commit": self.subject.commit,
                        "state_reference": self.subject.state_reference},
            "argv": list(self.argv),
            "cwd": self.cwd,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout": {"sha256": self.stdout_sha256, "bytes": self.stdout_bytes},
            "stderr": {"sha256": self.stderr_sha256, "bytes": self.stderr_bytes},
            "decode_procedure": self.decode_procedure,
            "interpreter": self.interpreter,
            "started_at": self.started_at,
            "seconds": self.seconds,
            "environment": dict(self.environment),
            "does_not_establish": [
                "that these bytes describe the approved operation's subject; "
                "binding is the admission boundary's work",
                "that the producer is trustworthy; a digest identifies bytes, "
                "not their truthfulness",
            ],
        }


class DirectorySink:
    """Retain bytes on disk, then publish. Never the other way round.

    Publication happens ONLY after both streams are written AND re-read and
    verified against their digests. A record naming bytes that were not stored
    would be the same defect as a pin that is never compared.
    """

    def __init__(self, root) -> None:
        self.root = Path(root)

    def publish_completed(self, record: CaptureRecord,
                          stdout: bytes, stderr: bytes) -> dict:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            stem = "{}-{}".format(record.phase.value, record.stdout_sha256[:16])
            out_path = self.root / "{}.stdout".format(stem)
            err_path = self.root / "{}.stderr".format(stem)
            out_path.write_bytes(stdout)
            err_path.write_bytes(stderr)
            # RE-READ. Writing is not retaining.
            if hashlib.sha256(out_path.read_bytes()).hexdigest() != \
                    record.stdout_sha256:
                raise RetentionFailed("retained stdout does not match its digest")
            if hashlib.sha256(err_path.read_bytes()).hexdigest() != \
                    record.stderr_sha256:
                raise RetentionFailed("retained stderr does not match its digest")
            document = record.as_document()
            document["retained"] = {"stdout_path": out_path.name,
                                    "stderr_path": err_path.name}
            raw = (json.dumps(document, indent=2, sort_keys=True,
                              ensure_ascii=True) + "\n").encode("utf-8")
            rec_path = self.root / "{}.json".format(stem)
            rec_path.write_bytes(raw)
        except RetentionFailed:
            raise
        except OSError as exc:
            raise RetentionFailed(
                "collection evidence could not be retained: {}".format(
                    exc)) from exc
        return document


def capture_collection(*, argv, cwd, env, timeout, phase, subject,
                       interpreter, sink):
    """Run a collection and RETAIN its output before anything reads it.

    Returns (record_document, stdout_bytes). The caller decodes and interprets;
    this function does neither, and it does not examine the exit status --
    because a nonzero exit is exactly when the evidence matters most.
    """
    if type(phase) is not CollectionPhase:
        raise CaptureError(
            "phase must be a CollectionPhase, not {!r}".format(phase))
    if type(subject) is not Subject:
        raise CaptureError(
            "subject must be a Subject, not {!r}".format(subject))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    clock = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(list(argv), cwd=str(cwd), env=env,
                              capture_output=True, timeout=timeout)
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        # A timeout still produced output, and it is the output that explains
        # the timeout. Retaining nothing here would discard the only evidence.
        timed_out = True
        out = exc.stdout or b""
        err = exc.stderr or b""
        rc = -1
    seconds = round(time.perf_counter() - clock, 3)
    record = CaptureRecord(
        phase=phase, subject=subject, argv=tuple(argv), cwd=str(cwd),
        returncode=rc,
        stdout_sha256=hashlib.sha256(out).hexdigest(), stdout_bytes=len(out),
        stderr_sha256=hashlib.sha256(err).hexdigest(), stderr_bytes=len(err),
        decode_procedure=DECODE_PROCEDURE, interpreter=str(interpreter),
        started_at=started, seconds=seconds, timed_out=timed_out)
    document = sink.publish_completed(record, out, err)
    return document, out


def decode_retained(raw: bytes) -> str:
    """Apply the DECLARED decoding procedure. Named, so it can be compared."""
    return raw.decode("utf-8", "replace")
