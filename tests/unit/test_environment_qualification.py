"""Environment qualification (owner ruling 2026-10-05b): the runtime-only lockfile admission against the REAL renv.lock, the clean R
probe against the real Rscript and controlled stand-ins, the required-test outcome gate, and the qualification receipt.

Author: Monzia Moodie
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from genomic_variant_classifier.environment_qualification.r_runtime import (
    RUNTIME_TARGET, AdmissionError, admit_runtime_change, canonical, probe_r, run_r_file, runtime_component_manifest, strict_json)
from genomic_variant_classifier.environment_qualification.receipt import QualificationReceipt, require_applicable
from genomic_variant_classifier.environment_qualification.required_tests import (
    Case, QualificationError, admit_junit, admit_loaded_namespaces, admit_qualification_rows)

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


# ------------------------------------------------------------------ the RUNTIME component manifest (owner ruling 2026-10-07, section 5)

def _fake_r_home(root):
    home = Path(root) / "R-4.6.1"
    for rel, data in (("bin/x64/R.dll", b"dll"), ("bin/x64/Rblas.dll", b"blas"), ("bin/Rscript.exe", b"launcher"), ("etc/x64/Makeconf", b"CC = gcc\n"),
                      ("library/base/DESCRIPTION", b"Package: base\n")):
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_bytes(data)
    return home


def test_runtime_manifest_identifies_components_not_only_the_launcher(tmp_path):
    home = _fake_r_home(tmp_path)
    m = runtime_component_manifest(home)
    assert [f["path"] for f in m["files"]] == ["bin/Rscript.exe", "bin/x64/R.dll", "bin/x64/Rblas.dll", "etc/x64/Makeconf"]   # library excluded
    assert runtime_component_manifest(home)["manifest_sha256"] == m["manifest_sha256"]                                       # deterministic
    (home / "bin/x64/R.dll").write_bytes(b"DLL")               # same launcher, different runtime component
    assert runtime_component_manifest(home)["manifest_sha256"] != m["manifest_sha256"]


def _symlinks_allowed() -> bool:
    """Whether THIS process may create symbolic links: Windows needs the SeCreateSymbolicLinkPrivilege (Developer Mode or an administrator).
    Measured 2026-10-07: these tests passed on Linux and failed the targeted stage on the owner's Windows machine."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "t"
        target.write_bytes(b"x")
        try:
            os.symlink(target, Path(d) / "l")
        except (OSError, NotImplementedError, AttributeError):
            return False
    return True


needs_symlinks = pytest.mark.skipif(not _symlinks_allowed(), reason="this process may not create symbolic links (Windows: Developer Mode or an administrator)")
needs_fifos = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs do not exist on this platform (os.mkfifo is Unix-only)")


@needs_symlinks
def test_runtime_manifest_records_a_file_link_by_its_target_content(tmp_path):
    """Measured 2026-10-07: Ubuntu's R keeps etc/Makeconf (and five other configuration files) as links into /etc/R."""
    home = _fake_r_home(tmp_path)
    config = tmp_path / "etc_R" / "Makeconf"
    config.parent.mkdir()
    config.write_bytes(b"CC = gcc-14\n")
    (home / "etc/x64/Makeconf").unlink()
    os.symlink(config, home / "etc/x64/Makeconf")
    m = runtime_component_manifest(home)
    entry = next(f for f in m["files"] if f["path"] == "etc/x64/Makeconf")
    assert entry["link_target"] == config.resolve().as_posix() and entry["size"] == len(b"CC = gcc-14\n")
    config.write_bytes(b"CC = gcc-15\n")                      # the TARGET changes -> the runtime identity changes
    changed = runtime_component_manifest(home)["manifest_sha256"]
    assert changed != m["manifest_sha256"]
    moved = tmp_path / "other_R" / "Makeconf"                  # same content, a DIFFERENT target path -> also a different identity
    moved.parent.mkdir()
    moved.write_bytes(b"CC = gcc-15\n")
    (home / "etc/x64/Makeconf").unlink()
    os.symlink(moved, home / "etc/x64/Makeconf")
    assert runtime_component_manifest(home)["manifest_sha256"] != changed


def _refused(home, reason):
    with pytest.raises(AdmissionError) as error:
        runtime_component_manifest(home)
    assert str(error.value) == reason


@needs_fifos
def test_runtime_manifest_refuses_a_fifo(tmp_path):
    home = _fake_r_home(tmp_path)
    os.mkfifo(home / "etc/pipe")                               # hashing a FIFO would block forever
    _refused(home, "runtime_manifest.non_regular_file:etc/pipe")


@needs_fifos
@needs_symlinks
def test_runtime_manifest_refuses_a_link_to_a_fifo(tmp_path):
    home = _fake_r_home(tmp_path)
    os.mkfifo(tmp_path / "fifo_target")
    os.symlink(tmp_path / "fifo_target", home / "etc/linked_pipe")
    _refused(home, "runtime_manifest.link_to_non_regular_file:etc/linked_pipe")


@needs_symlinks
def test_runtime_manifest_refuses_a_directory_link(tmp_path):
    home = _fake_r_home(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", home / "etc/linked_dir", target_is_directory=True)   # Windows needs the flag for a directory link
    _refused(home, "runtime_manifest.link_to_directory:etc/linked_dir")


@needs_symlinks
def test_runtime_manifest_refuses_a_dangling_link(tmp_path):
    home = _fake_r_home(tmp_path)
    os.symlink(tmp_path / "does_not_exist", home / "etc/dangling")
    _refused(home, "runtime_manifest.dangling_link:etc/dangling")


def test_runtime_manifest_refuses_a_missing_etc(tmp_path):
    home = _fake_r_home(tmp_path)
    shutil.rmtree(home / "etc")
    _refused(home, "runtime_manifest.missing:etc")


# ------------------------------------------------------------------ EXACT membership of replay qualification rows (owner ruling 2026-10-07b)

_EXPECTED_ROWS = (("IRanges", "2.46.0", "fresh"), ("S4Vectors", "0.50.1", "fresh"), ("Matrix", "1.7-5", "runtime"))


def _observed(*rows):
    return tuple(r + ("OK", "loaded") if len(r) == 3 else r for r in rows)


def test_exact_qualification_rows_are_admitted():
    assert admit_qualification_rows(_EXPECTED_ROWS, _observed(*_EXPECTED_ROWS))["qualified"] == 3


@pytest.mark.parametrize("observed, reason", [
    (_observed(_EXPECTED_ROWS[0], _EXPECTED_ROWS[1]), "qualification.identity_mismatch"),                          # a missing row
    (_observed(_EXPECTED_ROWS[0], _EXPECTED_ROWS[0], _EXPECTED_ROWS[2]), "qualification.identity_mismatch"),       # a DUPLICATE replaces a missing row (same count)
    (_observed(("IRanges", "2.46.1", "fresh"), _EXPECTED_ROWS[1], _EXPECTED_ROWS[2]), "qualification.identity_mismatch"),   # wrong version
    (_observed(_EXPECTED_ROWS[0], _EXPECTED_ROWS[1], ("Matrix", "1.7-5", "fresh")), "qualification.identity_mismatch"),     # wrong role
    ((("IRanges", "2.46.0", "fresh", "OK"),) + _observed(_EXPECTED_ROWS[1], _EXPECTED_ROWS[2]), "qualification.observed_shape"),   # malformed
    (_observed(_EXPECTED_ROWS[0], _EXPECTED_ROWS[1], ("Matrix", "1.7-5", "runtime", "MISMATCH", "path")), "qualification.result_failed"),
])
def test_qualification_row_refusals(observed, reason):
    with pytest.raises(QualificationError) as error:
        admit_qualification_rows(_EXPECTED_ROWS, observed)
    assert str(error.value) == reason


@pytest.mark.parametrize("expected, reason", [
    ((), "qualification.expected_empty"),
    ((("IRanges", "2.46.0"),), "qualification.expected_shape"),
    ((("IRanges", "2.46.0", "fresh"), ("IRanges", "2.46.0", "runtime")), "qualification.expected_duplicate"),
])
def test_qualification_expectation_refusals(expected, reason):
    with pytest.raises(QualificationError) as error:
        admit_qualification_rows(expected, ())
    assert str(error.value) == reason


# ------------------------------------------------------------------ what the fixture process ACTUALLY loaded (owner ruling 2026-10-07b, section 3)

# The REAL in-process identity record of the owner's admitted fixture run (fixtures_v2_20261007T042436Z; Windows, R 4.6.1, 47 namespaces),
# with expectations derived INDEPENDENTLY: versions from renv.lock, approved roots = that run's replay library / R's own library.
REAL_IDENTITY = json.loads('{"loaded_namespaces":[{"package":"abind","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/abind","version":"1.4-8"},{"package":"base","path":"C:/Program Files/R/R-4.6.1/library/base","version":"4.6.1"},{"package":"Biobase","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/Biobase","version":"2.72.0"},{"package":"BiocGenerics","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/BiocGenerics","version":"0.58.1"},{"package":"BiocIO","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/BiocIO","version":"1.22.0"},{"package":"BiocParallel","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/BiocParallel","version":"1.46.0"},{"package":"Biostrings","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/Biostrings","version":"2.80.1"},{"package":"bitops","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/bitops","version":"1.0-9"},{"package":"cigarillo","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/cigarillo","version":"1.2.0"},{"package":"codetools","path":"C:/Program Files/R/R-4.6.1/library/codetools","version":"0.2-20"},{"package":"compiler","path":"C:/Program Files/R/R-4.6.1/library/compiler","version":"4.6.1"},{"package":"crayon","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/crayon","version":"1.5.3"},{"package":"curl","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/curl","version":"7.1.0"},{"package":"datasets","path":"C:/Program Files/R/R-4.6.1/library/datasets","version":"4.6.1"},{"package":"DelayedArray","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/DelayedArray","version":"0.38.2"},{"package":"generics","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/generics","version":"0.1.4"},{"package":"GenomicAlignments","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/GenomicAlignments","version":"1.48.0"},{"package":"GenomicRanges","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/GenomicRanges","version":"1.64.0"},{"package":"graphics","path":"C:/Program Files/R/R-4.6.1/library/graphics","version":"4.6.1"},{"package":"grDevices","path":"C:/Program Files/R/R-4.6.1/library/grDevices","version":"4.6.1"},{"package":"grid","path":"C:/Program Files/R/R-4.6.1/library/grid","version":"4.6.1"},{"package":"httr","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/httr","version":"1.4.8"},{"package":"IRanges","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/IRanges","version":"2.46.0"},{"package":"lattice","path":"C:/Program Files/R/R-4.6.1/library/lattice","version":"0.22-9"},{"package":"Matrix","path":"C:/Program Files/R/R-4.6.1/library/Matrix","version":"1.7-5"},{"package":"MatrixGenerics","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/MatrixGenerics","version":"1.24.0"},{"package":"matrixStats","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/matrixStats","version":"1.5.0"},{"package":"methods","path":"C:/Program Files/R/R-4.6.1/library/methods","version":"4.6.1"},{"package":"parallel","path":"C:/Program Files/R/R-4.6.1/library/parallel","version":"4.6.1"},{"package":"R6","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/R6","version":"2.6.1"},{"package":"RCurl","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/RCurl","version":"1.98-1.19"},{"package":"restfulr","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/restfulr","version":"0.0.17"},{"package":"rjson","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/rjson","version":"0.2.23"},{"package":"Rsamtools","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/Rsamtools","version":"2.28.0"},{"package":"rtracklayer","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/rtracklayer","version":"1.72.0"},{"package":"S4Arrays","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/S4Arrays","version":"1.12.0"},{"package":"S4Vectors","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/S4Vectors","version":"0.50.1"},{"package":"Seqinfo","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/Seqinfo","version":"1.2.0"},{"package":"SparseArray","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/SparseArray","version":"1.12.2"},{"package":"stats","path":"C:/Program Files/R/R-4.6.1/library/stats","version":"4.6.1"},{"package":"stats4","path":"C:/Program Files/R/R-4.6.1/library/stats4","version":"4.6.1"},{"package":"SummarizedExperiment","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/SummarizedExperiment","version":"1.42.0"},{"package":"tools","path":"C:/Program Files/R/R-4.6.1/library/tools","version":"4.6.1"},{"package":"utils","path":"C:/Program Files/R/R-4.6.1/library/utils","version":"4.6.1"},{"package":"XML","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/XML","version":"3.99-0.23"},{"package":"XVector","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/XVector","version":"0.52.0"},{"package":"yaml","path":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library/yaml","version":"2.3.12"}],"platform":"x86_64-w64-mingw32","r_version":"4.6.1"}')
REAL_EXPECTED = json.loads('{"Biobase":"2.72.0","BiocGenerics":"0.58.1","BiocIO":"1.22.0","BiocParallel":"1.46.0","Biostrings":"2.80.1","DelayedArray":"0.38.2","GenomicAlignments":"1.48.0","GenomicRanges":"1.64.0","IRanges":"2.46.0","Matrix":"1.7-5","MatrixGenerics":"1.24.0","R6":"2.6.1","RCurl":"1.98-1.19","Rsamtools":"2.28.0","S4Arrays":"1.12.0","S4Vectors":"0.50.1","Seqinfo":"1.2.0","SparseArray":"1.12.2","SummarizedExperiment":"1.42.0","XML":"3.99-0.23","XVector":"0.52.0","abind":"1.4-8","bitops":"1.0-9","cigarillo":"1.2.0","codetools":"0.2-20","crayon":"1.5.3","curl":"7.1.0","generics":"0.1.4","httr":"1.4.8","lattice":"0.22-9","matrixStats":"1.5.0","restfulr":"0.0.17","rjson":"0.2.23","rtracklayer":"1.72.0","yaml":"2.3.12"}')
REAL_ROOTS = json.loads('{"Biobase":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","BiocGenerics":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","BiocIO":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","BiocParallel":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","Biostrings":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","DelayedArray":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","GenomicAlignments":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","GenomicRanges":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","IRanges":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","Matrix":"C:/Program Files/R/R-4.6.1/library","MatrixGenerics":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","R6":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","RCurl":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","Rsamtools":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","S4Arrays":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","S4Vectors":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","Seqinfo":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","SparseArray":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","SummarizedExperiment":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","XML":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","XVector":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","abind":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","bitops":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","cigarillo":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","codetools":"C:/Program Files/R/R-4.6.1/library","crayon":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","curl":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","generics":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","httr":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","lattice":"C:/Program Files/R/R-4.6.1/library","matrixStats":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","restfulr":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","rjson":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","rtracklayer":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library","yaml":"C:/Users/monzi/GVC_artifacts/runs/replay_20261006T230555Z/library"}')
REAL_BASE = frozenset(['base', 'compiler', 'datasets', 'grDevices', 'graphics', 'grid', 'methods', 'parallel', 'splines', 'stats', 'stats4', 'tcltk', 'tools', 'utils'])


def _admit_real(identity=None, **changes):
    kw = dict(expected_r_version="4.6.1", expected_platform="x86_64-w64-mingw32", expected=dict(REAL_EXPECTED), approved_roots=dict(REAL_ROOTS),
              r_home="C:/Program Files/R/R-4.6.1", base_packages=REAL_BASE, case_insensitive_paths=True)
    kw.update(changes)
    return admit_loaded_namespaces(identity if identity is not None else copy.deepcopy(REAL_IDENTITY), **kw)


def test_the_real_admitted_fixture_identity_is_admitted():
    result = _admit_real()
    assert (result["loaded"], result["expected"]) == (47, 35)


def _with(package, **fields):
    ident = copy.deepcopy(REAL_IDENTITY)
    for row in ident["loaded_namespaces"]:
        if row["package"] == package:
            row.update(fields)
    return ident


@pytest.mark.parametrize("identity, changes, reason", [
    (_with("IRanges", version="2.46.1"), {}, "identity.version:IRanges"),
    (_with("IRanges", path="C:/Users/someone/R/win-library/4.6/IRanges"), {}, "identity.location:IRanges"),        # right version, WRONG installation
    (_with("Matrix", path=REAL_ROOTS["IRanges"] + "/Matrix"), {}, "identity.location:Matrix"),                     # runtime package from the wrong library
    (_with("utils", path="C:/elsewhere/library/utils"), {}, "identity.base_location:utils"),
    (None, {"expected_r_version": "4.6.0"}, "identity.r_version"),
    (None, {"expected_platform": "x86_64-pc-linux-gnu"}, "identity.platform"),
])
def test_loaded_identity_refusals(identity, changes, reason):
    with pytest.raises(QualificationError) as error:
        _admit_real(identity, **changes)
    assert str(error.value).split(":")[0] == reason.split(":")[0]


def test_an_unexpected_or_missing_namespace_is_refused():
    extra = copy.deepcopy(REAL_IDENTITY)
    extra["loaded_namespaces"].append({"package": "ggplot2", "version": "4.0.3", "path": REAL_ROOTS["IRanges"] + "/ggplot2"})
    with pytest.raises(QualificationError, match="identity.unexpected_namespace:ggplot2"):
        _admit_real(extra)
    missing = copy.deepcopy(REAL_IDENTITY)
    missing["loaded_namespaces"] = [r for r in missing["loaded_namespaces"] if r["package"] != "S4Vectors"]
    with pytest.raises(QualificationError, match="identity.required_not_loaded:S4Vectors"):
        _admit_real(missing)
    dup = copy.deepcopy(REAL_IDENTITY)
    dup["loaded_namespaces"].append(dict(dup["loaded_namespaces"][0]))
    with pytest.raises(QualificationError, match="identity.duplicate_namespace"):
        _admit_real(dup)


def test_case_insensitive_comparison_applies_only_when_declared():
    """Measured 2026-10-07: the real record's paths share R_HOME's exact casing, so the flag is tested with an explicit casing difference."""
    assert _admit_real(r_home="c:/program files/r/r-4.6.1", case_insensitive_paths=True)["loaded"] == 47
    with pytest.raises(QualificationError, match="identity.base_location:base"):
        _admit_real(r_home="c:/program files/r/r-4.6.1", case_insensitive_paths=False)
