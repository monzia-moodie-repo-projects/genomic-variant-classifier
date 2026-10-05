"""The pre-install gate (owner ruling 2026-10-06): archives built here with the MEASURED layouts of a CRAN source tarball and a CRAN Windows
binary (one top-level directory; a binary holds <pkg>/Meta/package.rds and a Built field; a pure-R binary's platform is empty).

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from genomic_variant_classifier.environment_qualification.artifact_inspector import (
    check_dependencies, inspect_archive, r_version_key, select_candidates)
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError, sha256_file

COMPILED_BUILT = "R 4.6.1; x86_64-w64-mingw32; 2026-09-22 15:16:14 UTC; windows"
PURE_R_BUILT = "R 4.7.0; ; 2026-09-24 04:00:23 UTC; windows"


def desc(package="S4Arrays", version="1.12.0", **extra):
    lines = ["Package: " + package, "Version: " + version] + ["{}: {}".format(k.replace("_", "/"), v) for k, v in extra.items()]
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_source(path, package="S4Arrays", description=None, extra=()):
    with tarfile.open(path, "w:gz") as tf:
        for name, data in [(package + "/DESCRIPTION", description if description is not None else desc(package))] + list(extra):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def make_binary(path, package="S4Arrays", description=None, meta=True, extra=()):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(package + "/DESCRIPTION", description if description is not None else desc(package, Built=COMPILED_BUILT, NeedsCompilation="yes"))
        if meta:
            zf.writestr(package + "/Meta/package.rds", b"rds")
        for name, data in extra:
            zf.writestr(name, data)
    return path


def code_of(call):
    with pytest.raises(AdmissionError) as exc:
        call()
    return str(exc.value)


# ------------------------------------------------------------------ the owner's inspector regression cases

def test_filename_says_1_12_0_but_the_internal_version_is_1_12_1(tmp_path):
    archive = make_binary(tmp_path / "S4Arrays_1.12.0.zip", description=desc(version="1.12.1", Built=COMPILED_BUILT, NeedsCompilation="yes"))
    assert code_of(lambda: inspect_archive(archive, package="S4Arrays", version="1.12.0", kind="windows_binary")) == "artifact.version_mismatch"


def test_correct_version_wrong_approved_hash(tmp_path):
    archive = make_source(tmp_path / "S4Arrays_1.12.0.tar.gz")
    assert code_of(lambda: inspect_archive(archive, package="S4Arrays", version="1.12.0", kind="source", approved_sha256="0" * 64)) == "artifact.digest_mismatch"
    assert inspect_archive(archive, package="S4Arrays", version="1.12.0", kind="source", approved_sha256=sha256_file(archive)).artifact.size > 0


def test_installed_1_12_1_carrying_the_old_remote_sha_is_refused(tmp_path):
    d = desc(version="1.12.1", Built=COMPILED_BUILT, NeedsCompilation="yes", RemoteSha="b1246fd0b81ac137623ee1c0d6587a59e8ad1073")
    archive = make_binary(tmp_path / "x.zip", description=d)
    assert code_of(lambda: inspect_archive(archive, package="S4Arrays", version="1.12.0", kind="windows_binary")) == "artifact.version_mismatch"


# ------------------------------------------------------------------ admitted forms and what they record

def test_a_source_archive_is_admitted_without_binary_identity(tmp_path):
    a = inspect_archive(make_source(tmp_path / "s.tar.gz"), package="S4Arrays", version="1.12.0", kind="source").artifact
    assert (a.kind, a.platform, a.built_r_series, a.needs_compilation) == ("source", None, None, None) and len(a.description_sha256) == 64


def test_a_compiled_binary_records_its_build_identity(tmp_path):
    a = inspect_archive(make_binary(tmp_path / "b.zip"), package="S4Arrays", version="1.12.0", kind="windows_binary").artifact
    assert (a.platform, a.built_r_series, a.needs_compilation) == ("x86_64-w64-mingw32", "4.6", True)


def test_a_pure_r_binary_records_an_empty_platform(tmp_path):
    d = desc("DANDELION", "0.1.0", Built=PURE_R_BUILT, NeedsCompilation="no")
    a = inspect_archive(make_binary(tmp_path / "d.zip", package="DANDELION", description=d), package="DANDELION", version="0.1.0", kind="windows_binary").artifact
    assert (a.platform, a.built_r_series, a.needs_compilation) == ("", "4.7", False)


# ------------------------------------------------------------------ every refusal branch

@pytest.mark.parametrize("builder, kind, code", [
    (lambda p: make_source(p, description=desc(Built=COMPILED_BUILT)), "source", "artifact.source_has_built"),
    (lambda p: make_binary(p, meta=False), "windows_binary", "artifact.binary_layout"),
    (lambda p: make_source(p, extra=[("S4Arrays/Meta/package.rds", b"x")]), "source", "artifact.source_is_installed"),
    (lambda p: make_source(p, extra=[("Other/DESCRIPTION", b"x")]), "source", "artifact.layout"),
    (lambda p: make_source(p, extra=[("S4Arrays/../escape", b"x")]), "source", "artifact.unsafe_member"),
    (lambda p: make_binary(p, extra=[("/abs/path", b"x")]), "windows_binary", "artifact.unsafe_member"),
    (lambda p: make_source(p, description=desc(package="Other")), "source", "artifact.package_mismatch"),
    (lambda p: make_source(p, description=b"Package: S4Arrays\nVersion: 1.12.0\nVersion: 1.12.0\n"), "source", "artifact.description_duplicate_field:Version"),
    (lambda p: make_source(p, description=b"  starts with a continuation\n"), "source", "artifact.description_malformed"),
    (lambda p: make_source(p, description=b"Package: S4Arrays\nVersion: 1.12.0\nTitle: caf\xe9\n"), "source", "artifact.description_encoding"),
    (lambda p: make_binary(p, description=desc(Built="R 4.6.1; x86_64-pc-linux-gnu; 2026-09-22; unix", NeedsCompilation="yes")), "windows_binary", "artifact.binary_built"),
    (lambda p: make_binary(p, description=desc(Built="not a build line", NeedsCompilation="yes")), "windows_binary", "artifact.binary_built"),
    (lambda p: make_binary(p, description=desc(Built=COMPILED_BUILT)), "windows_binary", "artifact.needs_compilation"),
    (lambda p: (p.write_bytes(b"not an archive"), p)[1], "source", "artifact.unreadable"),
    (lambda p: (p.write_bytes(b"not an archive"), p)[1], "windows_binary", "artifact.unreadable"),
])
def test_refusals(tmp_path, builder, kind, code):
    archive = builder(tmp_path / "artifact")
    assert code_of(lambda: inspect_archive(archive, package="S4Arrays", version="1.12.0", kind=kind)) == code


def test_a_missing_description_is_refused(tmp_path):
    archive = tmp_path / "m.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("S4Arrays/NAMESPACE")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    assert code_of(lambda: inspect_archive(archive, package="S4Arrays", version="1.12.0", kind="source")) == "artifact.description_missing"


def test_an_unknown_kind_is_refused(tmp_path):
    assert code_of(lambda: inspect_archive(make_source(tmp_path / "s.tar.gz"), package="S4Arrays", version="1.12.0", kind="mac_binary")) == "artifact.kind"


# ------------------------------------------------------------------ conflicting candidates

def test_conflicting_candidates_for_one_identity_are_refused(tmp_path):
    one = inspect_archive(make_source(tmp_path / "a.tar.gz"), package="S4Arrays", version="1.12.0", kind="source")
    two = inspect_archive(make_source(tmp_path / "b.tar.gz", extra=[("S4Arrays/NEWS", b"different bytes")]), package="S4Arrays", version="1.12.0", kind="source")
    assert len(select_candidates([one, one])) == 1
    assert code_of(lambda: select_candidates([one, two])) == "artifact.conflicting_candidates:S4Arrays"


# ------------------------------------------------------------------ dependency constraints over the COMPLETE planned set

def test_r_version_ordering():
    assert r_version_key("1.10.0") > r_version_key("1.9.0") and r_version_key("1.7-5") > r_version_key("1.7-4")
    assert code_of(lambda: r_version_key("1.2a")) == "artifact.version_syntax:1.2a"


def test_satisfied_dependencies_pass():
    descs = {"S4Arrays": {"Depends": "R (>= 4.3.0), methods, Matrix, abind, BiocGenerics (>= 0.45.2)", "Imports": "stats",
                          "LinkingTo": "S4Vectors"}}
    check_dependencies(descs, {"S4Arrays": "1.12.0", "Matrix": "1.7-5", "abind": "1.4-8", "BiocGenerics": "0.56.0", "S4Vectors": "0.48.0"}, "4.6.1")


@pytest.mark.parametrize("descs, planned, fragment", [
    ({"S4Arrays": {"Depends": "abind"}}, {"S4Arrays": "1.12.0"}, "abind not planned"),
    ({"S4Arrays": {"Depends": "BiocGenerics (>= 0.45.2)"}}, {"S4Arrays": "1.12.0", "BiocGenerics": "0.40.0"}, "BiocGenerics >= 0.45.2 (planned 0.40.0)"),
    ({"S4Arrays": {"Depends": "R (>= 4.7.0)"}}, {"S4Arrays": "1.12.0"}, "R >= 4.7.0 (runtime 4.6.1)"),
    ({"S4Arrays": {"Imports": "weird entry!!"}}, {"S4Arrays": "1.12.0"}, "unparsed:weird entry!!"),
])
def test_unmet_dependencies_are_refused(descs, planned, fragment):
    message = code_of(lambda: check_dependencies(descs, planned, "4.6.1"))
    assert message.startswith("artifact.dependencies:") and fragment in message


def test_a_version_exactly_at_a_lower_bound_satisfies_it():
    check_dependencies({"S4Arrays": {"Depends": "BiocGenerics (>= 0.45.2)", "Imports": "abind (<= 1.4-8), Matrix (== 1.7-5)"}},
                       {"S4Arrays": "1.12.0", "BiocGenerics": "0.45.2", "abind": "1.4-8", "Matrix": "1.7-5"}, "4.6.1")
