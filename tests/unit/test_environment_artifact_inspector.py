"""The pre-install gate (owner rulings 2026-10-06 and 2026-10-07). Archives are built here with the MEASURED layouts of a CRAN source
tarball and a CRAN Windows binary. STRUCTURE tests are pure Python; DESCRIPTION and dependency SEMANTICS are judged by R and skip where
Rscript is not on PATH (the qualification runs pass an explicit Rscript).

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import shutil
import stat
import tarfile
import zipfile

import pytest

from genomic_variant_classifier.environment_qualification import artifact_inspector as ai
from genomic_variant_classifier.environment_qualification.artifact_inspector import (
    Inspected, closure_with_r, inspect_archive, inspect_structure, select_candidates)
from genomic_variant_classifier.environment_qualification.install_plan import Artifact
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R semantics are judged by Rscript, which is not on PATH")
COMPILED_BUILT = "R 4.6.1; x86_64-w64-mingw32; 2026-09-22 15:16:14 UTC; windows"
PURE_R_BUILT = "R 4.7.0; ; 2026-09-24 04:00:23 UTC; windows"


def desc(package="S4Arrays", version="1.12.0", **extra):
    lines = ["Package: " + package, "Version: " + version] + ["{}: {}".format(k, v) for k, v in extra.items()]
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_source(path, package="S4Arrays", description=None, extra=(), members=None):
    with tarfile.open(path, "w:gz") as tf:
        items = members if members is not None else [(package + "/DESCRIPTION", description if description is not None else desc(package))] + list(extra)
        for item in items:
            info = item if isinstance(item, tarfile.TarInfo) else tarfile.TarInfo(item[0])
            data = b"" if isinstance(item, tarfile.TarInfo) else item[1]
            if not isinstance(item, tarfile.TarInfo):
                info.size = len(data)
            tf.addfile(info, io.BytesIO(data) if info.isreg() else None)
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


def inspect(path, tmp_path, version="1.12.0", kind="source", package="S4Arrays", **kw):
    return inspect_archive(path, package=package, version=version, kind=kind, rscript=RSCRIPT, evidence_dir=tmp_path / "r_evidence", **kw)


# ------------------------------------------------------------------ STRUCTURE (pure Python)

def _tar_link(name, kind):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = "S4Arrays/DESCRIPTION"
    return info


def _dir(name):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    return info


@pytest.mark.parametrize("members, code", [
    ([("S4Arrays/DESCRIPTION", desc()), _tar_link("S4Arrays/link", tarfile.SYMTYPE)], "artifact.member_type"),
    ([("S4Arrays/DESCRIPTION", desc()), _tar_link("S4Arrays/hard", tarfile.LNKTYPE)], "artifact.member_type"),
    ([("S4Arrays/DESCRIPTION", desc()), ("S4Arrays/R/a.R", b"1"), ("S4Arrays/R/a.R", b"2")], "artifact.duplicate_member"),
    ([("S4Arrays/DESCRIPTION", desc()), ("S4Arrays/R/a.R", b"1"), ("S4Arrays/R/A.R", b"2")], "artifact.case_collision"),
    ([("S4Arrays/DESCRIPTION", desc()), ("S4Arrays/R", b"file"), ("S4Arrays/R/a.R", b"1")], "artifact.file_directory_conflict"),
    ([("S4Arrays", b"a FILE at the root name"), ("S4Arrays/DESCRIPTION", desc())], "artifact.layout"),
    ([("S4Arrays/DESCRIPTION", desc()), ("Other/x", b"1")], "artifact.layout"),
    ([("S4Arrays/DESCRIPTION", desc()), ("S4Arrays/../escape", b"1")], "artifact.unsafe_member"),
    ([("S4Arrays/DESCRIPTION", desc()), ("S4Arrays/a\\b", b"1")], "artifact.unsafe_member"),
    ([("S4Arrays/NAMESPACE", b"x")], "artifact.description_missing"),
    ([("S4Arrays/DESCRIPTION", desc()), ("S4Arrays/Meta/package.rds", b"x")], "artifact.source_is_installed"),
])
def test_structural_refusals_in_a_source_archive(tmp_path, members, code):
    archive = make_source(tmp_path / "a.tar.gz", members=members)
    assert code_of(lambda: inspect_structure(archive, package="S4Arrays", kind="source")) == code


def test_a_bare_root_directory_header_is_accepted(tmp_path):
    archive = make_source(tmp_path / "a.tar.gz", members=[_dir("S4Arrays"), _dir("S4Arrays/R"), ("S4Arrays/DESCRIPTION", desc())])
    assert inspect_structure(archive, package="S4Arrays", kind="source").installed_layout is False


def test_a_zip_symlink_is_refused(tmp_path):
    archive = make_binary(tmp_path / "b.zip")
    with zipfile.ZipFile(archive, "a") as zf:
        info = zipfile.ZipInfo("S4Arrays/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "S4Arrays/DESCRIPTION")
    assert code_of(lambda: inspect_structure(archive, package="S4Arrays", kind="windows_binary")) == "artifact.member_type"


@pytest.mark.parametrize("builder, kind, code", [
    (lambda p: make_binary(p, meta=False), "windows_binary", "artifact.binary_layout"),
    (lambda p: make_binary(p, extra=[("/abs/path", b"x")]), "windows_binary", "artifact.unsafe_member"),
    (lambda p: (p.write_bytes(b"not an archive"), p)[1], "source", "artifact.unreadable"),
    (lambda p: (p.write_bytes(b"not an archive"), p)[1], "windows_binary", "artifact.unreadable"),
    (lambda p: make_source(p), "mac_binary", "artifact.kind"),
])
def test_more_structural_refusals(tmp_path, builder, kind, code):
    archive = builder(tmp_path / "artifact")
    assert code_of(lambda: inspect_structure(archive, package="S4Arrays", kind=kind)) == code


def test_member_count_and_declared_size_are_bounded(tmp_path, monkeypatch):
    archive = make_source(tmp_path / "a.tar.gz", extra=[("S4Arrays/R/a.R", b"12345")])
    monkeypatch.setattr(ai, "MAX_MEMBERS", 1)
    assert code_of(lambda: inspect_structure(archive, package="S4Arrays", kind="source")) == "artifact.too_many_members"
    monkeypatch.setattr(ai, "MAX_MEMBERS", 50_000)
    monkeypatch.setattr(ai, "MAX_DECLARED_BYTES", 10)
    assert code_of(lambda: inspect_structure(archive, package="S4Arrays", kind="source")) == "artifact.too_large"


def test_the_digest_is_of_the_inspected_bytes(tmp_path):
    archive = make_source(tmp_path / "a.tar.gz")
    with archive.open("rb") as fh:
        expected = hashlib.sha256(fh.read()).hexdigest()
    structure = inspect_structure(archive, package="S4Arrays", kind="source")
    assert structure.sha256 == expected and structure.size == archive.stat().st_size
    assert structure.description_bytes == desc()


def test_conflicting_candidates_for_one_identity_are_refused():
    def item(digest):
        return Inspected(Artifact("S4Arrays", "1.12.0", "source", digest, 1, "b" * 64), {})
    assert len(select_candidates([item("a" * 64), item("a" * 64)])) == 1
    assert code_of(lambda: select_candidates([item("a" * 64), item("c" * 64)])) == "artifact.conflicting_candidates:S4Arrays"


# ------------------------------------------------------------------ SEMANTICS (R) -- the owner's inspector regression cases first

@needs_r
def test_filename_says_1_12_0_but_the_internal_version_is_1_12_1(tmp_path):
    archive = make_binary(tmp_path / "S4Arrays_1.12.0.zip", description=desc(version="1.12.1", Built=COMPILED_BUILT, NeedsCompilation="yes"))
    assert code_of(lambda: inspect(archive, tmp_path, kind="windows_binary")) == "artifact.version_mismatch"


@needs_r
def test_correct_version_wrong_approved_hash(tmp_path):
    archive = make_source(tmp_path / "S4Arrays_1.12.0.tar.gz")
    assert code_of(lambda: inspect(archive, tmp_path, approved_sha256="0" * 64)) == "artifact.digest_mismatch"


@needs_r
def test_installed_1_12_1_carrying_the_old_remote_sha_is_refused(tmp_path):
    d = desc(version="1.12.1", Built=COMPILED_BUILT, NeedsCompilation="yes", RemoteSha="b1246fd0b81ac137623ee1c0d6587a59e8ad1073")
    assert code_of(lambda: inspect(make_binary(tmp_path / "x.zip", description=d), tmp_path, kind="windows_binary")) == "artifact.version_mismatch"


@needs_r
def test_admitted_forms_record_their_identity(tmp_path):
    s = inspect(make_source(tmp_path / "s.tar.gz"), tmp_path).artifact
    assert (s.kind, s.platform, s.built_r_series, s.needs_compilation) == ("source", None, None, None)
    b = inspect_archive(make_binary(tmp_path / "b.zip"), package="S4Arrays", version="1.12.0", kind="windows_binary", rscript=RSCRIPT,
                        evidence_dir=tmp_path / "e2").artifact
    assert (b.platform, b.built_r_series, b.needs_compilation) == ("x86_64-w64-mingw32", "4.6", True)
    d = desc("DANDELION", "0.1.0", Built=PURE_R_BUILT, NeedsCompilation="no")
    p = inspect_archive(make_binary(tmp_path / "d.zip", package="DANDELION", description=d), package="DANDELION", version="0.1.0",
                        kind="windows_binary", rscript=RSCRIPT, evidence_dir=tmp_path / "e3").artifact
    assert (p.platform, p.built_r_series, p.needs_compilation) == ("", "4.7", False)
    assert (tmp_path / "r_evidence" / "probe.R").is_file() and (tmp_path / "r_evidence" / "process.json").is_file()


@needs_r
@pytest.mark.parametrize("builder, kind, code", [
    (lambda p: make_source(p, description=desc(Built=COMPILED_BUILT)), "source", "artifact.source_has_built"),
    (lambda p: make_source(p, description=desc(package="Other")), "source", "artifact.package_mismatch"),
    (lambda p: make_source(p, description=b"Package: S4Arrays\nVersion: 1.12.0\nVersion: 1.12.0\n"), "source",
     "artifact.r_semantics:description.duplicate_field:Version"),
    (lambda p: make_source(p, description=b"Package: S4Arrays\nVersion: 1.12.0\n\nPackage: Other\nVersion: 2.0\n"), "source",
     "artifact.r_semantics:description.record_count"),
    (lambda p: make_source(p, description=b""), "source", "artifact.r_semantics:description.record_count"),
    (lambda p: make_source(p, description=desc(version="1.2a")), "source", "artifact.r_semantics:description.version_syntax:1.2a"),
    (lambda p: make_source(p, description=b"Package: S4Arrays\nVersion: 1.12.0\nTitle: caf\xe9\n"), "source", "artifact.description_encoding:unsupported_by_policy"),
    (lambda p: make_binary(p, description=desc(Built="R 4.6.1; x86_64-pc-linux-gnu; 2026-09-22; unix", NeedsCompilation="yes")), "windows_binary", "artifact.binary_built"),
    (lambda p: make_binary(p, description=desc(Built="not a build line", NeedsCompilation="yes")), "windows_binary", "artifact.binary_built"),
    (lambda p: make_binary(p, description=desc(Built=COMPILED_BUILT)), "windows_binary", "artifact.needs_compilation"),
])
def test_semantic_refusals(tmp_path, builder, kind, code):
    assert code_of(lambda: inspect(builder(tmp_path / "artifact"), tmp_path, kind=kind)) == code


@needs_r
def test_a_leading_continuation_line_is_unparseable(tmp_path):
    archive = make_source(tmp_path / "a.tar.gz", description=b"  starts with a continuation\n")
    assert code_of(lambda: inspect(archive, tmp_path)).startswith("artifact.r_semantics:description.unparseable:")


# ------------------------------------------------------------------ dependency closure, judged by R (the R oracle)

def _d(text):
    return text.encode("utf-8")


@needs_r
@pytest.mark.parametrize("descs, planned", [
    ([_d("Package: A\nVersion: 1.0\nDepends: R (>= 3.0.0), methods, B (>= 1.2.0)\nImports: stats\nLinkingTo: B\n"), _d("Package: B\nVersion: 1.2\n")],
     {"A": "1.0", "B": "1.2"}),                                        # 1.2 satisfies >= 1.2.0 (R: equal)
    ([_d("Package: A\nVersion: 1.0\nDepends: B (>= 0.45.2), C (<= 1.4-8), D (== 1.7-5)\n"), _d("Package: B\nVersion: 0.45.2\n"),
      _d("Package: C\nVersion: 1.4-8\n"), _d("Package: D\nVersion: 1.7.5\n")], {"A": "1.0", "B": "0.45.2", "C": "1.4-8", "D": "1.7.5"}),
    ([_d("Package: A\nVersion: 1.0\nDepends: B (> 1.9.9)\n"), _d("Package: B\nVersion: 1.10.0\n")], {"A": "1.0", "B": "1.10.0"}),   # numeric, not lexical
])
def test_satisfied_closures_pass(tmp_path, descs, planned):
    closure_with_r(RSCRIPT, descs, planned, tmp_path / "closure")


@needs_r
@pytest.mark.parametrize("descs, planned, fragment", [
    ([_d("Package: A\nVersion: 1.0\nImports: stats (>= 999.0)\n")], {"A": "1.0"}, "A:stats:"),
    ([_d("Package: A\nVersion: 1.0\nDepends: B (>= 2.0), C\n"), _d("Package: B\nVersion: 1.0\n")], {"A": "1.0", "B": "1.0"}, "A:B:1.0:>=:2.0 | A:C:missing"),
    ([_d("Package: A\nVersion: 1.0\nDepends: R (>= 999.0.0)\n")], {"A": "1.0"}, "A:R:"),
    ([], {"A": "1.0"}, "dependency.description_coverage"),
    ([_d("Package: A\nVersion: 1.0\n")], {"A": "1.1"}, "dependency.package_version:A"),
    ([_d("Package: A\nVersion: 1.0\nImports: weird entry!!\n")], {"A": "1.0"}, "A:Imports:unparsed:weird entry!!"),
])
def test_unmet_closures_are_refused(tmp_path, descs, planned, fragment):
    message = code_of(lambda: closure_with_r(RSCRIPT, descs, planned, tmp_path / "closure"))
    assert message.startswith("artifact.dependencies:") and fragment in message


# ------------------------------------------------------------------ native libraries are COUNTED from contents (measured 2026-10-10)

@pytest.mark.parametrize("extra, count", [
    ([("S4Arrays/libs/x64/S4Arrays.dll", b"MZ")], 1),
    ([], 0),
    ([("S4Arrays/libs/x64/S4Arrays.DLL", b"MZ"), ("S4Arrays/libs/i386/S4Arrays.dll", b"MZ")], 2),
    ([("S4Arrays/libs/x64/symbols.rds", b"x")], 0),
])
def test_native_libraries_are_counted_from_the_binary_contents(tmp_path, extra, count):
    archive = make_binary(tmp_path / "b.zip", extra=extra)
    assert inspect_structure(archive, package="S4Arrays", kind="windows_binary").native_libraries == count


@needs_r
def test_the_count_reaches_the_artifact_for_binaries_only(tmp_path):
    b = inspect_archive(make_binary(tmp_path / "b.zip", extra=[("S4Arrays/libs/x64/S4Arrays.dll", b"MZ")]), package="S4Arrays", version="1.12.0",
                        kind="windows_binary", rscript=RSCRIPT, evidence_dir=tmp_path / "e1").artifact
    s = inspect(make_source(tmp_path / "s.tar.gz"), tmp_path).artifact
    assert (b.native_libraries, s.native_libraries) == (1, None)
