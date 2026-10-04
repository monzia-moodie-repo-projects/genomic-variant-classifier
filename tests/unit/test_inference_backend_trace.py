"""The DANDELION actual-call backend trace: strict post-processing (always) and the real recorder against the installed
DANDELION in every qvalue mode (skipped, with the reason, when Rscript or the pinned package is absent -- no workflow
installs R, so it skips in continuous integration).

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from genomic_variant_classifier.inference.backend_trace import RECORDER_VERSION, admissible_events, read_trace
from genomic_variant_classifier.inference.exact_confirmation import InferenceError

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "f471153bfa3c0069cd68a67565000889c7cdf5d1"


def event(n, **kw):
    doc = {"call": "call-{:04d}".format(n), "exposure_id": "rs{}".format(n), "backend": "BH",
           "fallback_reason": "fewer_than_10_values", "last_step_reached": 3, "n_values": 3, "n_distinct": 3,
           "qvalue_entered": False, "p_adjust_inside_qvalue": 0, "observation_kind": "actual_call_trace",
           "runtime_instrumented": True, "method_commit": COMMIT, "recorder_version": RECORDER_VERSION}
    doc.update(kw)
    return doc


def write_trace(directory, docs, *, values=b"0x1.999999999999ap-4\n0x1p+0\n"):
    directory.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        for kind in ("input", "clamped", "output"):
            (directory / "{}.{}.txt".format(doc["call"], kind)).write_bytes(values)
    (directory / "events.jsonl").write_bytes(b"".join(json.dumps(d).encode() + b"\n" for d in docs))
    return directory


def code_of(call):
    with pytest.raises(InferenceError) as exc:
        call()
    return exc.value.code


def test_a_valid_trace_is_read_with_exact_digests(tmp_path):
    d = write_trace(tmp_path / "t", [event(1), event(2, backend="qvalue", fallback_reason=None, last_step_reached=4)])
    events = read_trace(d, expected_exposures=("rs1", "rs2"))
    assert [e.backend for e in events] == ["BH", "qvalue"]
    assert events[0].input_sha256 == hashlib.sha256((d / "call-0001.input.txt").read_bytes()).hexdigest()
    assert admissible_events(events) == events


@pytest.mark.parametrize("mutate, code", [
    (lambda d: (d / "events.jsonl").write_bytes((d / "events.jsonl").read_bytes().replace(b'"call"', b'"backend": "BH", "call"', 1)),
     "trace_duplicate_key"),
    (lambda d: (d / "events.jsonl").write_bytes((d / "events.jsonl").read_bytes().replace(b'"call"', b'"extra": 1, "call"', 1)),
     "trace_event_keys"),
    (lambda d: (d / "events.jsonl").write_bytes((d / "events.jsonl").read_bytes().replace(b"call-0002", b"call-0007")),
     "trace_call_sequence"),
    (lambda d: (d / "events.jsonl").write_bytes((d / "events.jsonl").read_bytes().replace(RECORDER_VERSION.encode(), b"other/1", 1)),
     "trace_event_value"),
    (lambda d: (d / "call-0001.input.txt").unlink(), "trace_value_file_missing"),
    (lambda d: (d / "call-0002.output.txt").unlink(), "trace_value_files_inconsistent"),
    (lambda d: (d / "call-0001.input.txt").write_bytes(b"0\n"), "trace_value_file_malformed"),       # R never writes a bare 0
    (lambda d: (d / "events.jsonl").write_bytes((d / "events.jsonl").read_bytes()[:-1]), "trace_events_truncated"),
    (lambda d: (d / "notes.txt").write_bytes(b"x"), "trace_unexpected_files"),
])
def test_malformed_traces_are_refused(tmp_path, mutate, code):
    d = write_trace(tmp_path / "t", [event(1), event(2)])
    mutate(d)
    assert code_of(lambda: read_trace(d)) == code


def test_completeness_requires_each_planned_exposure_exactly_once(tmp_path):
    d = write_trace(tmp_path / "t", [event(1), event(2)])
    assert code_of(lambda: read_trace(d, expected_exposures=("rs1", "rs2", "rs3"))) == "trace_incomplete"   # rs3 escaped
    assert code_of(lambda: read_trace(d, expected_exposures=("rs1",))) == "trace_incomplete"                 # an extra call
    assert code_of(lambda: read_trace(d, expected_exposures=("rs1", "rs1"))) == "expected_exposures_duplicated"
    repeated = write_trace(tmp_path / "r", [event(1), event(2, exposure_id="rs1")])
    assert code_of(lambda: read_trace(repeated, expected_exposures=("rs1", "rs2"))) == "trace_incomplete"


@pytest.mark.parametrize("backend, reason", [("unclassified", "observation_inconsistent_with_measured_branches"),
                                             ("none", "safe_qvalues_abnormal_exit")])
def test_unclassified_or_abnormal_calls_refuse_admission(tmp_path, backend, reason):
    docs = [event(1), event(2, backend=backend, fallback_reason=reason)]
    d = write_trace(tmp_path / "t", docs)
    if backend == "none":
        (d / "call-0002.output.txt").unlink()
    assert code_of(lambda: admissible_events(read_trace(d))) == "trace_not_admissible"


# ------------------------------------------------------------------ the real recorder against the installed DANDELION

def _rscript():
    return shutil.which("Rscript")


def _dandelion_installed():
    exe = _rscript()
    if exe is None:
        return False
    r = subprocess.run([exe, "-e", 'quit(status = if (requireNamespace("DANDELION", quietly = TRUE)) 0 else 1)'],
                       capture_output=True, timeout=120)
    return r.returncode == 0


@pytest.mark.skipif(_rscript() is None, reason="Rscript not on PATH")
@pytest.mark.skipif(not _dandelion_installed(), reason="the pinned DANDELION package (f471153) is not installed in R's library")
@pytest.mark.parametrize("mode, many_backend, many_reason", [
    ("absent", "BH", "qvalue_not_installed"), ("success", "qvalue", None),
    ("warning", "BH", "qvalue_warning"), ("error", "BH", "qvalue_error")])
def test_the_recorder_observes_every_branch_of_the_real_safe_qvalues(tmp_path, mode, many_backend, many_reason):
    env = dict(os.environ)
    if mode != "absent":
        lib = tmp_path / "rlib_double"
        lib.mkdir()
        r_exe = shutil.which("R")
        if r_exe is None:
            pytest.skip("R (for R CMD INSTALL of the qvalue test double) not on PATH")
        inst = subprocess.run([r_exe, "CMD", "INSTALL", "-l", str(lib), str(ROOT / "tests/fixtures/dandelion/qvalue_testdouble")],
                              capture_output=True, text=True, timeout=300)
        assert inst.returncode == 0, inst.stderr[-500:]
        env["R_LIBS"] = os.pathsep.join(x for x in (str(lib), env.get("R_LIBS", "")) if x)
        env["GVC_QVALUE_DOUBLE_MODE"] = mode
    out = tmp_path / "trace"
    run = subprocess.run([_rscript(), str(ROOT / "tests/fixtures/dandelion/run_recorder_fixture.R"),
                          str(ROOT / "scripts/dandelion/dandelion_backend_recorder.R"), str(out)],
                         capture_output=True, text=True, timeout=300, env=env)
    assert run.returncode == 0 and "fixture OK" in run.stdout, run.stderr[-800:]
    events = admissible_events(read_trace(out, expected_exposures=("rs_small", "rs_few", "rs_many")))
    assert [(e.exposure_id, e.backend, e.fallback_reason) for e in events] == [
        ("rs_small", "BH", "fewer_than_10_values"), ("rs_few", "BH", "fewer_than_4_distinct_values"),
        ("rs_many", many_backend, many_reason)]


def _probe(tmp_path, scenario, trace_dir):
    result = tmp_path / ("result-" + scenario + ".txt")
    run = subprocess.run([_rscript(), str(ROOT / "tests/fixtures/dandelion/run_recorder_probe.R"),
                          str(ROOT / "scripts/dandelion/dandelion_backend_recorder.R"), scenario, str(trace_dir), str(result)],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-800:]
    return result.read_bytes()


@pytest.mark.skipif(_rscript() is None, reason="Rscript not on PATH")
@pytest.mark.skipif(not _dandelion_installed(), reason="the pinned DANDELION package (f471153) is not installed in R's library")
def test_recording_does_not_change_the_science_across_fresh_processes(tmp_path):
    """Owner ruling 2026-10-04b: recorder OFF and recorder ON in two FRESH R processes; exact byte equality of the outputs."""
    off = _probe(tmp_path, "off", tmp_path / "unused")
    on = _probe(tmp_path, "on", tmp_path / "trace")
    assert off == on and off.count(b"\n") == 3 + 3 + 12 + 25
    assert not (tmp_path / "unused").exists()
    assert len(read_trace(tmp_path / "trace")) == 3


@pytest.mark.skipif(_rscript() is None, reason="Rscript not on PATH")
@pytest.mark.skipif(not _dandelion_installed(), reason="the pinned DANDELION package (f471153) is not installed in R's library")
def test_an_unusable_destination_is_refused_before_any_tracing(tmp_path):
    parent = tmp_path / "a_regular_file"
    parent.write_bytes(b"x")
    out = _probe(tmp_path, "badpath", parent / "sub").decode("utf-8").split("\n")
    assert "refusing before any traced call" in out[0] and out[1] == "traced FALSE"


@pytest.mark.skipif(os.name == "nt", reason="parallel::mclapply cannot fork on Windows")
@pytest.mark.skipif(_rscript() is None, reason="Rscript not on PATH")
@pytest.mark.skipif(not _dandelion_installed(), reason="the pinned DANDELION package (f471153) is not installed in R's library")
def test_a_forked_worker_is_refused_before_writing(tmp_path):
    out = _probe(tmp_path, "fork", tmp_path / "trace").decode("utf-8").split("\n")
    assert out[:2] == ["refused 2", "files 0"]


@pytest.mark.skipif(os.name == "nt", reason="a read-only directory attribute does not block file creation on Windows")
@pytest.mark.skipif(os.name != "nt" and os.geteuid() == 0, reason="root ignores directory permissions, so an unwritable directory cannot be made")
@pytest.mark.skipif(_rscript() is None, reason="Rscript not on PATH")
@pytest.mark.skipif(not _dandelion_installed(), reason="the pinned DANDELION package (f471153) is not installed in R's library")
def test_an_existing_but_unwritable_directory_is_refused_by_the_write_probe(tmp_path):
    """The probe's own case (the directory exists, so only the write probe can refuse it)."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        out = _probe(tmp_path, "badpath", locked).decode("utf-8").split("\n")
    finally:
        locked.chmod(0o755)
    assert "is not writable -- refusing before any traced call" in out[0] and out[1] == "traced FALSE"
