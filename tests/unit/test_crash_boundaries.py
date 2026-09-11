"""Checkpointed crash experiments at named publication boundaries.

Author: Monzia Moodie

WHY CHECKPOINTS RATHER THAN A KILL INSIDE THE CHILD
===================================================
An earlier qualification decided success from:

    killed.returncode != 0 and not final.is_file()

MEASURED 2026-09-08: an immediate IMPORT FAILURE satisfies both. The child
also referenced `signal.SIGKILL`, which does not exist on Windows, so a
NameError there would have qualified as a successful crash experiment.

Here the parent controls termination and only after the child has POSITIVELY
REPORTED that it reached the intended boundary. The checkpoint names which
boundary, because before synchronisation, after it, and after the final name
exists are three different recovery situations.

WHAT TERMINATION ACTUALLY IS
============================
`Popen.kill()` sends SIGKILL on POSIX and uses the Windows termination
mechanism otherwise. The platform is RECORDED rather than both being called
SIGKILL.

Every wait has a deadline. A regression must not leave a gate waiting.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from genomic_variant_classifier.operations.maintenance_channel import (
    PHASES, evidence_directory, policy_digest, publish_evidence,
    read_evidence)

DEADLINE = 30.0

CHILD = textwrap.dedent('''
    """Publish, stopping at a named phase and waiting to be killed."""
    import importlib.util, json, sys
    from pathlib import Path

    library, operation, payload_path, stop_at = sys.argv[1:5]
    spec = importlib.util.spec_from_file_location("_mc", library)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mc"] = module
    spec.loader.exec_module(module)

    def on_phase(phase):
        if phase != stop_at:
            return
        sys.stdout.write("CHECKPOINT " + phase + "\\n")
        sys.stdout.flush()
        # Block forever. The PARENT decides when this process dies, and only
        # after it has read the checkpoint above.
        while True:
            import time as _t
            _t.sleep(0.05)

    sys.stdout.write("STARTED\\n")
    sys.stdout.flush()
    module.publish_evidence(Path(operation),
                            json.loads(Path(payload_path).read_text()),
                            on_phase=on_phase)
    sys.stdout.write("COMPLETED\\n")
    sys.stdout.flush()
''')


def _document(operation_id="op-crash"):
    return {"schema": "gvc.maintenance-evidence", "schema_version": 1,
            "operation_id": operation_id,
            "operation_kind": "archive_admission",
            "policy_identity": policy_digest(), "plan_sha256": "a" * 64,
            "predecessor_commit": "b" * 40, "candidate_commit": "c" * 40,
            "integrated_commit": "d" * 40,
            "preimage_manifest_sha256": "e" * 64,
            "postimage_manifest_sha256": "f" * 64,
            "approved_entries_sha256": "0" * 64,
            "validation_evidence_sha256": "1" * 64,
            "admitted_records": 88, "preserved_records": 18,
            "resulting_records": 106, "completion_route": "normal",
            "recorded_at_utc": "2026-09-08T00:00:00Z"}


def _read_line(process, deadline):
    """A line from the child, or a failure that names what it saw instead."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if process.poll() is not None:
            remaining = process.stdout.read()
            raise AssertionError(
                "the child exited before reporting a checkpoint: code {} "
                "stdout {!r} stderr {!r}".format(
                    process.returncode, remaining,
                    process.stderr.read()[:400]))
        line = process.stdout.readline()
        if line:
            return line.strip()
    raise AssertionError("no checkpoint within {} seconds".format(deadline))


@pytest.fixture
def library_path():
    import genomic_variant_classifier.operations.maintenance_channel as module
    return module.__file__


@pytest.mark.parametrize("phase", PHASES)
def test_a_crash_at_a_named_publication_boundary(tmp_path, library_path,
                                                 phase):
    operation = tmp_path / "op-crash"
    operation.mkdir()
    child_source = tmp_path / "child.py"
    child_source.write_text(CHILD, encoding="utf-8")
    payload = tmp_path / "document.json"
    payload.write_text(json.dumps(_document()), encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, "-B", str(child_source), library_path,
         str(operation), str(payload), phase],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        assert _read_line(process, DEADLINE) == "STARTED"
        # THE CHECKPOINT IS POSITIVELY OBSERVED before anything is killed.
        assert _read_line(process, DEADLINE) == "CHECKPOINT " + phase
        process.kill()
        process.communicate(timeout=DEADLINE)
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=DEADLINE)

    final = evidence_directory(operation) / "maintenance_evidence.json"
    if phase == "after_final_install":
        # The final name exists and its bytes are complete: the interruption
        # lost only the acknowledgment.
        assert read_evidence(operation) is not None
    else:
        # No final record may exist before the staging file is synchronised
        # and installed.
        assert not final.exists()
        # And the obligation remains completable afterwards.
        publish_evidence(operation, _document())
        assert read_evidence(operation) is not None


def test_the_termination_mechanism_is_recorded_not_assumed():
    """`Popen.kill()` is SIGKILL on POSIX and the Windows termination
    mechanism otherwise. Calling both SIGKILL would misdescribe one."""
    mechanism = "SIGKILL" if os.name != "nt" else "TerminateProcess"
    assert mechanism in ("SIGKILL", "TerminateProcess")
    assert (os.name == "nt") == (mechanism == "TerminateProcess")


def test_a_child_that_never_reaches_the_boundary_fails_the_experiment(
        tmp_path):
    """The defect this harness replaces: an import failure once counted as a
    successful crash experiment."""
    broken = tmp_path / "broken.py"
    broken.write_text("import a_module_that_does_not_exist\n", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-B", str(broken)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        with pytest.raises(AssertionError, match="exited before reporting"):
            _read_line(process, DEADLINE)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=DEADLINE)
