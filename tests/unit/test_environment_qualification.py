"""Environment qualification (owner ruling 2026-10-05b): the runtime-only lockfile admission against the REAL renv.lock, the clean R
probe against the real Rscript and controlled stand-ins, the required-test outcome gate, and the qualification receipt.

Author: Monzia Moodie
"""
from __future__ import annotations

import copy
import dataclasses
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from genomic_variant_classifier.environment_qualification.r_runtime import (
    RUNTIME_TARGET, AdmissionError, admit_runtime_change, canonical, probe_r, run_r_file, strict_json)
from genomic_variant_classifier.environment_qualification.receipt import QualificationReceipt, require_applicable
from genomic_variant_classifier.environment_qualification.required_tests import Case, QualificationError, admit_junit

ROOT = Path(__file__).resolve().parents[2]


def code_of(exc_type, call):
    with pytest.raises(exc_type) as exc:
        call()
    return str(exc.value)


def real_lock():
    return strict_json((ROOT / "renv.lock").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ runtime-only lockfile admission (the REAL renv.lock)

def test_the_single_permitted_change_is_admitted():
    before = real_lock()
    after = copy.deepcopy(before)
    after["R"]["Version"] = RUNTIME_TARGET
    delta = admit_runtime_change(before, after, RUNTIME_TARGET)
    assert (delta["before_r"], delta["after_r"], delta["package_count"], delta["qualification_complete"]) == ("4.6.0", "4.6.1", len(before["Packages"]), False)


def _mutations():
    first = sorted(real_lock()["Packages"])[0]
    def set_(path, value):
        def apply(d):
            node = d
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
        return apply
    return [
        ("package version", set_(("Packages", first, "Version"), "999.0")),
        ("added package", lambda d: d["Packages"].__setitem__("Zzz", {"Package": "Zzz", "Version": "1.0", "Source": "Repository"})),
        ("removed package", lambda d: d["Packages"].pop(first)),
        ("bioconductor", set_(("Bioconductor", "Version"), "3.24")),
        ("repository url", lambda d: d["R"]["Repositories"][0].__setitem__("URL", "https://example.invalid")),
        ("renv version", set_(("Packages", "renv", "Version"), "1.2.4")),
        ("extra top-level key", lambda d: d.__setitem__("Extra", {})),
        ("true replaced by 1", lambda d: d["Packages"][first].__setitem__("GVCFlag", True)),
    ]


@pytest.mark.parametrize("label, mutate", _mutations(), ids=[m[0] for m in _mutations()])
def test_any_change_outside_R_Version_is_refused(label, mutate):
    before = real_lock()
    after = copy.deepcopy(before)
    after["R"]["Version"] = RUNTIME_TARGET
    if label == "true replaced by 1":       # the baseline carries True, the candidate 1: serialised comparison must tell them apart
        before["Packages"][sorted(before["Packages"])[0]]["GVCFlag"] = 1
    mutate(after)
    assert code_of(AdmissionError, lambda: admit_runtime_change(before, after, RUNTIME_TARGET)) == "change_outside_R.Version"


def test_wrong_observed_runtime_is_refused():
    before = real_lock()
    after = copy.deepcopy(before)
    after["R"]["Version"] = "4.6.2"
    assert code_of(AdmissionError, lambda: admit_runtime_change(before, after, "4.6.2")) == "unexpected_observed_runtime"


@pytest.mark.parametrize("text, code", [('{"a": 1, "a": 2}', "duplicate_json_key:a"), ('{"a": NaN}', "nonfinite_json:NaN"),
                                        ('{"a": Infinity}', "nonfinite_json:Infinity"), ('{"a": -Infinity}', "nonfinite_json:-Infinity")])
def test_strict_json_refuses_duplicates_and_nonfinite_constants(text, code):
    assert code_of(AdmissionError, lambda: strict_json(text)) == code


def test_canonical_distinguishes_true_from_one():
    assert canonical({"x": True}) != canonical({"x": 1})


# ------------------------------------------------------------------ the clean R probe

RSCRIPT = shutil.which("Rscript")


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
def test_the_probe_is_clean_even_with_a_contaminating_profile(tmp_path, monkeypatch):
    version_program = tmp_path / "version.R"
    version_program.write_text("cat(as.character(getRversion()))\n", encoding="utf-8")
    actual = subprocess.run([RSCRIPT, "--vanilla", str(version_program)], cwd=tmp_path,
                            capture_output=True, text=True, check=True).stdout
    profile = tmp_path / "profile.R"
    profile.write_text('cat("CONTAMINATION\\n")\n', encoding="utf-8")
    monkeypatch.setenv("R_PROFILE_USER", str(profile))
    monkeypatch.setenv("RENV_PATHS_ROOT", str(tmp_path / "nowhere"))
    monkeypatch.chdir(ROOT)                          # the repository root, whose .Rprofile contaminated an unisolated probe
    result = probe_r(RSCRIPT, tmp_path / "probe", expected=actual)
    assert result["version"] == actual and result["release_status"] == "" and len(result["launcher_sha256"]) == 64


def _fake(tmp_path, body):
    exe = tmp_path / "Rscript"
    exe.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


GOOD = 'printf "GVC_R_VERSION=4.6.1\\nGVC_R_STATUS=\\nGVC_R_PLATFORM=x86_64-w64-mingw32\\nGVC_R_HOME=C:/R\\n"'


@pytest.mark.skipif(os.name == "nt", reason="the stand-in executables are POSIX shell scripts")
@pytest.mark.parametrize("body, code", [
    (GOOD, None),
    ('printf "CONTAMINATION\\n"; ' + GOOD, "r_probe_shape"),
    (GOOD.replace("4.6.1", "4.6.0"), "r_probe_version"),
    (GOOD.replace("STATUS=", "STATUS=Patched"), "r_not_plain_release"),
    (GOOD + '; echo "a warning" >&2', "r_probe_stderr"),
    (GOOD + "; exit 3", "r_probe_exit:3"),
    ('[ -n "$R_LIBS$RENV_PATHS_ROOT" ] && exit 9; ' + GOOD, None),   # R_* / RENV_* never reach the child
])
def test_probe_outcomes_with_stand_in_executables(tmp_path, monkeypatch, body, code):
    monkeypatch.setenv("R_LIBS", "/somewhere")
    monkeypatch.setenv("RENV_PATHS_ROOT", "/somewhere")
    exe = _fake(tmp_path, body)
    if code is None:
        assert probe_r(exe, tmp_path / "probe")["version"] == "4.6.1"
    else:
        assert code_of(AdmissionError, lambda: probe_r(exe, tmp_path / "probe")) == code
    assert (tmp_path / "probe" / "process.json").is_file()        # every outcome leaves evidence


FILE_ONLY = 'case "$2" in */probe.R) [ "$1" = "--vanilla" ] && [ -f "$2" ] && [ "$#" -eq 2 ] || exit 7 ;; *) exit 8 ;; esac; '


@pytest.mark.skipif(os.name == "nt", reason="the stand-in executables are POSIX shell scripts")
def test_the_probe_runs_a_file_program_never_minus_e(tmp_path):
    exe = _fake(tmp_path, FILE_ONLY + GOOD)
    assert probe_r(exe, tmp_path / "probe")["version"] == "4.6.1"
    assert "GVC_R_HOME=" in (tmp_path / "probe" / "probe.R").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the stand-in executables are POSIX shell scripts")
def test_a_refusal_keeps_the_complete_evidence(tmp_path):
    exe = _fake(tmp_path, GOOD + '; echo "R said something" >&2; exit 3')
    assert code_of(AdmissionError, lambda: probe_r(exe, tmp_path / "probe")) == "r_probe_exit:3"
    record = strict_json((tmp_path / "probe" / "process.json").read_text(encoding="utf-8"))
    assert (record["status"], record["returncode"]) == ("exited_nonzero", 3)
    assert (tmp_path / "probe" / "stderr.bin").read_bytes() == b"R said something\n"
    assert (tmp_path / "probe" / "stdout.bin").read_bytes().startswith(b"GVC_R_VERSION=4.6.1")


def test_a_missing_executable_is_recorded_as_start_failed(tmp_path):
    assert code_of(AdmissionError, lambda: probe_r(tmp_path / "no-such-Rscript", tmp_path / "probe")) == "r_probe_start_failed"
    record = strict_json((tmp_path / "probe" / "process.json").read_text(encoding="utf-8"))
    assert record["status"] == "start_failed" and record["executable_sha256"] is None and record["error_type"] == "FileNotFoundError"


@pytest.mark.skipif(os.name == "nt", reason="the stand-in executables are POSIX shell scripts")
def test_non_utf8_output_is_refused(tmp_path):
    exe = _fake(tmp_path, "printf '\\377\\376'")
    assert code_of(AdmissionError, lambda: probe_r(exe, tmp_path / "probe")) == "r_probe_encoding"


@pytest.mark.skipif(os.name == "nt", reason="the stand-in executables are POSIX shell scripts")
def test_run_r_file_records_a_timeout_with_its_partial_output(tmp_path):
    exe = _fake(tmp_path, 'printf "partial"; sleep 5')
    record = run_r_file(exe, "cat(1)\n", tmp_path / "run", child_env=dict(os.environ), timeout_seconds=1)
    assert record["status"] == "timeout" and record["returncode"] is None
    assert strict_json((tmp_path / "run" / "process.json").read_text(encoding="utf-8"))["status"] == "timeout"


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
def test_valid_looking_output_followed_by_an_r_error_is_refused_with_both_streams_kept(tmp_path):
    record = run_r_file(RSCRIPT, 'cat("GVC_R_VERSION=4.6.1\\n")\nstop("deliberate failure after printing")\n', tmp_path / "run",
                        child_env=dict(os.environ))
    assert (record["status"], record["returncode"]) == ("exited_nonzero", 1)
    assert (tmp_path / "run" / "stdout.bin").read_bytes() == b"GVC_R_VERSION=4.6.1\n"
    assert b"deliberate failure after printing" in (tmp_path / "run" / "stderr.bin").read_bytes()


# ------------------------------------------------------------------ the required-test outcome gate

PLAN = frozenset({Case("tests.unit.test_x", "test_a"), Case("tests.unit.test_x", "test_b[1]")})


def report(*cases):
    body = "".join('<testcase classname="{}" name="{}" time="0.1">{}</testcase>'.format(c, n, extra) for c, n, extra in cases)
    return ('<?xml version="1.0"?><testsuites><testsuite name="pytest">' + body + "</testsuite></testsuites>").encode()


GOOD_REPORT = report(("tests.unit.test_x", "test_a", ""), ("tests.unit.test_x", "test_b[1]", ""))


def test_a_complete_passing_report_is_admitted():
    out = admit_junit(GOOD_REPORT, expected_cases=PLAN, process_exit_code=0)
    assert out["required_cases_passed"] == 2 and len(out["report_sha256"]) == 64


@pytest.mark.parametrize("xml, exit_code, plan, code", [
    (report(("tests.unit.test_x", "test_a", "<skipped/>"), ("tests.unit.test_x", "test_b[1]", "")), 0, PLAN, "test_skipped"),
    (report(("tests.unit.test_x", "test_a", "<failure/>"), ("tests.unit.test_x", "test_b[1]", "")), 0, PLAN, "test_failure"),
    (report(("tests.unit.test_x", "test_a", "<error/>"), ("tests.unit.test_x", "test_b[1]", "")), 0, PLAN, "test_error"),
    (report(("tests.unit.test_x", "test_a", "")), 0, PLAN, "case_set_mismatch"),
    (report(("tests.unit.test_x", "test_a", ""), ("tests.unit.test_x", "test_b[2]", "")), 0, PLAN, "case_set_mismatch"),
    (report(("tests.unit.test_x", "test_a", ""), ("tests.unit.test_x", "test_a", ""), ("tests.unit.test_x", "test_b[1]", "")), 0, PLAN, "duplicate_case"),
    (b"<testsuites><testsuite>", 0, PLAN, "report_xml"),
    (GOOD_REPORT, 1, PLAN, "test_process_failed"),
    (GOOD_REPORT, True, PLAN, "test_process_failed"),
    (GOOD_REPORT, 0, frozenset(), "expected_cases_required"),
])
def test_incomplete_or_unsuccessful_reports_are_refused(xml, exit_code, plan, code):
    assert code_of(QualificationError, lambda: admit_junit(xml, expected_cases=plan, process_exit_code=exit_code)) == code


# ------------------------------------------------------------------ the receipt

def receipt(**kw):
    args = dict(checkpoint="runtime", repository_tree="e" * 40, baseline_lock_sha256="a" * 64, candidate_lock_sha256="b" * 64,
                runtime_version="4.6.1", runtime_release_status="", runtime_platform="x86_64-w64-mingw32", launcher_sha256="c" * 64,
                qualification_code_sha256="d" * 64, inventory_sha256="f" * 64, required_test_report_sha256="9" * 64, required_cases=6)
    args.update(kw)
    return QualificationReceipt(**args)


def test_receipt_renders_deterministically_and_applies_to_itself():
    assert receipt().render() == receipt().render() and receipt().render().endswith(b"\n")
    require_applicable(receipt(), receipt())


@pytest.mark.parametrize("name", [f.name for f in dataclasses.fields(QualificationReceipt)])
def test_a_receipt_for_another_candidate_is_inapplicable(name):
    changed = {"checkpoint": "dependency", "required_cases": 7, "runtime_version": "4.6.0", "runtime_release_status": "Patched",
               "runtime_platform": "other", "repository_tree": "0" * 40}.get(name, "0" * 64)
    assert code_of(AdmissionError, lambda: require_applicable(receipt(), receipt(**{name: changed}))) == "receipt_inapplicable:" + name


@pytest.mark.parametrize("kw, code", [({"checkpoint": "both"}, "receipt_checkpoint"), ({"candidate_lock_sha256": "x"}, "receipt_candidate_lock_sha256"),
                                      ({"required_cases": True}, "receipt_required_cases"), ({"repository_tree": "e" * 39}, "receipt_repository_tree")])
def test_malformed_receipts_are_refused(kw, code):
    assert code_of(AdmissionError, lambda: receipt(**kw)) == code
