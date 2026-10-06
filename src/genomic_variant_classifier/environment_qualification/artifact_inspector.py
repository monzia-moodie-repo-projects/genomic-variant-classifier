"""The pre-install gate: requested identity -> acquired archive -> internal DESCRIPTION -> approved SHA-256 -> admit (owner rulings
2026-10-06 and 2026-10-07).

PYTHON checks bytes and archive STRUCTURE; the QUALIFIED R interprets R's own metadata (r_semantics.R_PACKAGE_SEMANTICS, run through
run_r_file with an explicit Rscript). The merged version re-implemented R's DESCRIPTION and version semantics in Python and got them
wrong (measured 2026-10-07: "1.2" vs "1.2.0" unequal; two records merged; base-package constraints unchecked; empty coverage accepted;
duplicate members accepted).

inspect_archive() reads the archive ONCE, computes its SHA-256 and size from THOSE bytes and inspects THOSE bytes (no window in which
the digest and the inspected content could describe different files). Structure: tar members must be regular files or directories and
zip members must not be symbolic links; names must not be absolute, contain "..", or use backslashes; one root named after the package
(a bare "<pkg>" header only as a directory); no duplicate paths, no Windows case-insensitive collisions, no path that is both a file and
a directory; bounded member count and declared size; exactly one regular <pkg>/DESCRIPTION; the kind is taken from the CONTENTS (a
Windows binary holds <pkg>/Meta/package.rds). R then reads that DESCRIPTION strictly; Package and Version must equal the intended
identity as exact recorded STRINGS (semantic version equality is used only inside dependency constraints); Built is parsed in its
measured four-field form. closure_with_r() checks the complete planned set's dependency closure in R against the runtime's measured
base inventory. select_candidates() refuses conflicting candidates for one identity. Nothing is downloaded, extracted or installed.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import stat
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from genomic_variant_classifier.environment_qualification.install_plan import Artifact
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError, run_r_file
from genomic_variant_classifier.environment_qualification.r_semantics import R_PACKAGE_SEMANTICS

logger = logging.getLogger(__name__)

__all__ = ["MAX_ARCHIVE_BYTES", "MAX_MEMBERS", "MAX_DECLARED_BYTES", "Structure", "Inspected", "inspect_structure",
           "describe_with_r", "closure_with_r", "inspect_archive", "select_candidates"]

MAX_ARCHIVE_BYTES = 1 << 30          # 1 GiB on disk
MAX_MEMBERS = 50_000
MAX_DECLARED_BYTES = 4 << 30         # 4 GiB declared uncompressed
_MAX_DESCRIPTION_BYTES = 1 << 20
_BUILT = re.compile(r"R ([0-9]+)\.([0-9]+)\.[0-9]+; ([^;]*); ([^;]+); (windows|unix)")


@dataclass(frozen=True)
class Structure:
    sha256: str
    size: int
    description_bytes: bytes = field(repr=False)
    installed_layout: bool
    # Native libraries actually present (<pkg>/libs/**.dll). Judged from CONTENTS: measured 2026-10-10, tidyselect 1.2.1's DESCRIPTION says
    # NeedsCompilation "yes" while its source has no src/ files and Posit's binary has no DLL -- the metadata was stale, the binary complete.
    native_libraries: int = 0


@dataclass(frozen=True)
class Inspected:
    artifact: Artifact
    description: dict = field(hash=False, compare=False)


def _entries(data: bytes, kind: str):
    """(name, is_dir, declared_size, payload_reader) for every member; member TYPES are judged here."""
    out = []
    if kind == "source":
        try:
            tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
            members = tf.getmembers()
        except (tarfile.TarError, OSError, EOFError):
            raise AdmissionError("artifact.unreadable")
        for m in members:
            if not (m.isreg() or m.isdir()):
                raise AdmissionError("artifact.member_type")
            out.append((m.name, m.isdir(), m.size, (lambda m=m, tf=tf: tf.extractfile(m).read())))
    elif kind == "windows_binary":
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
            infos = zf.infolist()
        except (zipfile.BadZipFile, OSError, EOFError):
            raise AdmissionError("artifact.unreadable")
        for i in infos:
            if stat.S_ISLNK(i.external_attr >> 16):
                raise AdmissionError("artifact.member_type")
            out.append((i.filename, i.is_dir(), i.file_size, (lambda i=i, zf=zf: zf.read(i))))
    else:
        raise AdmissionError("artifact.kind")
    return out


def inspect_structure(path, *, package: str, kind: str) -> Structure:
    """Bytes and structure only. The archive is read ONCE; digest and inspection use the same bytes."""
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise AdmissionError("artifact.too_large")
    with path.open("rb") as stream:
        data = stream.read()
    entries = _entries(data, kind)
    if len(entries) > MAX_MEMBERS:
        raise AdmissionError("artifact.too_many_members")
    if sum(e[2] for e in entries) > MAX_DECLARED_BYTES:
        raise AdmissionError("artifact.too_large")
    seen, folded, files, dirs = set(), set(), set(), set()
    description = []
    for name, is_dir, size, read in entries:
        if name.startswith("/") or "\\" in name or ".." in PurePosixPath(name).parts:
            raise AdmissionError("artifact.unsafe_member")
        clean = name.rstrip("/")
        if clean == package:
            if not is_dir:
                raise AdmissionError("artifact.layout")
        elif not clean.startswith(package + "/"):
            raise AdmissionError("artifact.layout")
        if clean in seen:
            raise AdmissionError("artifact.duplicate_member")
        if clean.casefold() in folded:
            raise AdmissionError("artifact.case_collision")
        seen.add(clean)
        folded.add(clean.casefold())
        (dirs if is_dir else files).add(clean)
        if clean == package + "/DESCRIPTION":
            if is_dir:
                raise AdmissionError("artifact.description_missing")
            if size > _MAX_DESCRIPTION_BYTES:
                raise AdmissionError("artifact.description_too_large")
            description.append(read())
    for f in files:                      # a path may not be a file AND contain other members
        parts = PurePosixPath(f).parts
        if any(str(PurePosixPath(*parts[:i])) in files for i in range(1, len(parts))):
            raise AdmissionError("artifact.file_directory_conflict")
    if files & dirs:
        raise AdmissionError("artifact.file_directory_conflict")
    if len(description) != 1:
        raise AdmissionError("artifact.description_missing")
    installed = (package + "/Meta/package.rds") in files
    if kind == "windows_binary" and not installed:
        raise AdmissionError("artifact.binary_layout")
    if kind == "source" and installed:
        raise AdmissionError("artifact.source_is_installed")
    native = sum(1 for f in files if f.startswith(package + "/libs/") and f.lower().endswith(".dll"))
    return Structure(sha256=hashlib.sha256(data).hexdigest(), size=len(data), description_bytes=description[0], installed_layout=installed,
                     native_libraries=native)


def _r_string(path: Path) -> str:
    text = path.resolve().as_posix()
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _r_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.upper().startswith(("R_", "RENV_"))}


def _r_failure(run: dict) -> str:
    """R's COMPLETE error text (a multi-line stop() keeps every line), without the "Error ... :" prefix and "Execution halted"."""
    lines = [line.rstrip() for line in run["stderr"].decode("utf-8", errors="replace").splitlines() if line.strip()]
    lines = [line for line in lines if line.strip() != "Execution halted" and not line.startswith("Calls:")]
    start = next((i for i, line in enumerate(lines) if line.startswith("Error")), None)
    if start is None:
        return " | ".join(lines) or "status " + str(run["status"])
    first = lines[start]
    if first.startswith("Error in ") and first.rstrip().endswith(":") and " : " not in first.rstrip()[:-1] + " ":
        first = ""                          # R wrapped "Error in <call> :" onto its own line; the message follows
    elif " : " in first:
        first = first.split(" : ", 1)[1]
    elif first.rstrip().endswith(" :"):
        first = ""
    else:
        first = first.split(": ", 1)[-1]
    return " | ".join(part.strip() for part in [first] + lines[start + 1:] if part.strip()) or "status " + str(run["status"])


def _unescape(value: str) -> str:
    out, i = [], 0
    while i < len(value):
        if value[i] == "\\" and i + 1 < len(value):
            out.append({"\\": "\\", "t": "\t", "n": "\n"}.get(value[i + 1], value[i + 1]))
            i += 2
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def _stage_descriptions(descriptions, inputs: Path) -> Path:
    inputs.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, raw in enumerate(descriptions):
        target = inputs / "description_{}.dcf".format(index)
        target.write_bytes(raw)
        rows.append("{}\t{}".format(index, target.resolve().as_posix()))
    listing = inputs / "listing.tsv"
    listing.write_text("".join(row + "\n" for row in rows), encoding="utf-8")     # EMPTY file for no rows (not one blank line)
    return listing


def describe_with_r(rscript, descriptions, evidence_dir) -> list:
    """Each DESCRIPTION read strictly by the qualified R; returns one field dictionary per input, in order."""
    for raw in descriptions:             # PROJECT POLICY (ruling 2026-10-07): UTF-8 DESCRIPTION bytes; others = unsupported, not malformed
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            raise AdmissionError("artifact.description_encoding:unsupported_by_policy")
    evidence_dir = Path(evidence_dir)
    inputs = evidence_dir.with_name(evidence_dir.name + ".inputs")
    listing = _stage_descriptions(descriptions, inputs)
    out = inputs / "fields.tsv"
    run = run_r_file(rscript, R_PACKAGE_SEMANTICS + "\ngvc_describe({}, {})\n".format(_r_string(listing), _r_string(out)),
                     evidence_dir, child_env=_r_env(), timeout_seconds=300)
    if run["status"] != "exited_zero":
        raise AdmissionError("artifact.r_semantics:" + _r_failure(run))
    try:
        text = out.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise AdmissionError("artifact.description_encoding")
    fields = [dict() for _ in descriptions]
    for line in text.splitlines():
        index, name, value = line.split("\t")
        fields[int(index)][_unescape(name)] = _unescape(value)
    return fields


def closure_with_r(rscript, descriptions, planned: dict, evidence_dir) -> None:
    """The COMPLETE planned set's dependency closure, judged by R against the runtime's measured base inventory."""
    evidence_dir = Path(evidence_dir)
    inputs = evidence_dir.with_name(evidence_dir.name + ".inputs")
    listing = _stage_descriptions(descriptions, inputs)
    planned_file = inputs / "planned.tsv"
    planned_file.write_text("".join("{}\t{}\n".format(p, planned[p]) for p in sorted(planned)), encoding="utf-8")
    run = run_r_file(rscript, R_PACKAGE_SEMANTICS + "\ngvc_closure({}, {})\n".format(_r_string(listing), _r_string(planned_file)),
                     evidence_dir, child_env=_r_env(), timeout_seconds=300)
    if run["status"] != "exited_zero" or b"GVC_CLOSURE_OK" not in run["stdout"]:
        raise AdmissionError("artifact.dependencies:" + _r_failure(run))


def inspect_archive(path, *, package: str, version: str, kind: str, rscript, evidence_dir, approved_sha256: str | None = None) -> Inspected:
    """Inspect one acquired archive against the INTENDED identity. The filename is never consulted."""
    structure = inspect_structure(path, package=package, kind=kind)
    if approved_sha256 is not None and structure.sha256 != approved_sha256:
        raise AdmissionError("artifact.digest_mismatch")
    description = describe_with_r(rscript, [structure.description_bytes], evidence_dir)[0]
    if description.get("Package") != package:
        raise AdmissionError("artifact.package_mismatch")
    if description.get("Version") != version:
        raise AdmissionError("artifact.version_mismatch")
    platform = built_series = needs = None
    if kind == "source":
        if "Built" in description:
            raise AdmissionError("artifact.source_has_built")
    else:
        built = _BUILT.fullmatch(description.get("Built", ""))
        if built is None or built.group(5) != "windows":
            raise AdmissionError("artifact.binary_built")
        built_series = built.group(1) + "." + built.group(2)
        platform = built.group(3).strip()
        nc = description.get("NeedsCompilation")
        if nc not in ("yes", "no"):
            raise AdmissionError("artifact.needs_compilation")
        needs = nc == "yes"
    artifact = Artifact(package=package, version=version, kind=kind, sha256=structure.sha256, size=structure.size,
                        description_sha256=hashlib.sha256(structure.description_bytes).hexdigest(), platform=platform,
                        built_r_series=built_series, needs_compilation=needs,
                        native_libraries=structure.native_libraries if kind == "windows_binary" else None)
    return Inspected(artifact=artifact, description=description)


def select_candidates(inspected) -> dict:
    """At most ONE artifact per (package, version, kind); different bytes for the same identity are refused, never chosen between."""
    chosen = {}
    for item in inspected:
        a = item.artifact
        key = (a.package, a.version, a.kind)
        if key in chosen and chosen[key].artifact.sha256 != a.sha256:
            raise AdmissionError("artifact.conflicting_candidates:" + a.package)
        chosen.setdefault(key, item)
    return chosen
