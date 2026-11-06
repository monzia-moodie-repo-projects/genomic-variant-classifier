"""PROGRAM-START-RECORD-PLATFORM-FORM-1 (2026-11-05): the program_starts fixture (tests/unit/conftest.py), calibrated on the
machine that runs it.

MEASURED 2026-11-05 on the owner's Windows machine: the first recorder of program starts read a "subprocess.Popen" audit event's
command as a list of arguments -- the POSIX form; CPython on Windows reports one command line (subprocess.list2cmdline) and
executable=None -- and test_inference_backend_trace's R-start test failed there, having been verified only on Linux. The recorder
now keeps what the platform raised, and its prediction of that form, program_starts.subprocess_start, is compared here with a REAL
start on whichever machine runs the suite:
* a start let through runs (its exit status is the program's own) and is recorded exactly as predicted -- on Windows this compares
  the predicted command line, quoting included, with CPython's own;
* a start blocked (the default) is recorded as predicted and never runs: the file it would write does not exist;
* os.system is recorded and blocked (its command is bytes on POSIX; it is compared decoded);
* an audit hook added while recording is refused and recorded, and is never called afterwards (no hook leaks).

Author: Monzia Moodie
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_a_start_let_through_runs_and_is_recorded_in_this_platforms_own_form(program_starts, tmp_path):
    program_starts.block = False
    argv = [sys.executable, "-I", "-S", "-c", "import sys; sys.exit(7)"]
    run = subprocess.run(argv, cwd=tmp_path, capture_output=True, timeout=120)
    assert run.returncode == 7, run.stderr
    assert program_starts.seen == [program_starts.subprocess_start(argv, cwd=tmp_path)]


def test_a_blocked_start_is_recorded_and_never_runs(program_starts, tmp_path):
    marker = tmp_path / "the program ran"
    argv = [sys.executable, "-I", "-S", "-c", "import sys; open(sys.argv[1], 'w').close()", str(marker)]
    with pytest.raises(RuntimeError, match=r"^program start blocked by the test \(recorded\): subprocess\.Popen$"):
        subprocess.run(argv, cwd=tmp_path, capture_output=True, timeout=120)
    assert program_starts.seen == [program_starts.subprocess_start(argv, cwd=tmp_path)]
    assert not marker.exists()


def test_os_system_is_recorded_and_blocked(program_starts):
    with pytest.raises(RuntimeError, match=r"^program start blocked by the test \(recorded\): os\.system$"):
        os.system("exit 3")
    assert [(r["event"], [os.fsdecode(a) for a in r["arguments"]]) for r in program_starts.seen] == [("os.system", ["exit 3"])]


def test_an_audit_hook_added_while_recording_is_refused_and_never_called(program_starts):
    calls = []
    sys.addaudithook(lambda event, args: calls.append(event))       # CPython drops it: an existing hook raised RuntimeError
    assert program_starts.seen == [{"event": "sys.addaudithook", "arguments": ()}]
    sys.audit("gvc.program_start_recorder.probe")
    assert calls == []
