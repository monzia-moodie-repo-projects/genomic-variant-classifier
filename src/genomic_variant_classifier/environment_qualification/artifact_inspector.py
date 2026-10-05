"""The pre-install gate: requested identity -> acquired archive -> internal DESCRIPTION -> approved SHA-256 -> admit (owner ruling 2026-10-06).

Package substitution must be stopped BEFORE installation (measured: renv 1.2.3 retrieve.R L1163-1166 overwrites a requested version with the
retrieved one). For every archive this module, WITHOUT extracting it:
  * requires exactly one top-level directory named after the package (no absolute paths, no ".." components) and one regular DESCRIPTION;
  * identifies the kind from the CONTENTS, never the filename: a Windows binary holds <pkg>/Meta/package.rds (measured in CRAN's
    DANDELION_0_1_0.zip); a source archive must not;
  * parses DESCRIPTION strictly (Debian Control File form; duplicate fields refused; UTF-8) and requires Package and Version to equal the
    intended identity -- a file named S4Arrays_1.12.0.zip whose DESCRIPTION says 1.12.1 is refused;
  * requires a source archive to carry NO Built field and a binary to carry one in the measured form "R x.y.z; <platform>; <date>; windows"
    (platform EMPTY for packages without compiled code, e.g. "R 4.7.0; ; ...; windows"), recording the built R series, platform and
    NeedsCompilation;
  * records the archive SHA-256 + size and the DESCRIPTION SHA-256, and refuses a mismatch with an approved digest.
select_candidates() refuses conflicting candidates for one identity; check_dependencies() checks Depends/Imports/LinkingTo against the
COMPLETE planned set with R's version ordering. It performs no download and no installation.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import logging
import re
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from genomic_variant_classifier.environment_qualification.install_plan import Artifact
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError, sha256_file

logger = logging.getLogger(__name__)

__all__ = ["BASE_PACKAGES", "Inspected", "parse_dcf", "inspect_archive", "select_candidates", "r_version_key", "check_dependencies"]

#: Packages that ship as part of R itself (priority "base"); they are runtime inventory, not planned artifacts.
BASE_PACKAGES = frozenset({"base", "compiler", "datasets", "graphics", "grDevices", "grid", "methods", "parallel", "splines",
                           "stats", "stats4", "tcltk", "tools", "utils"})
_MAX_DESCRIPTION_BYTES = 1 << 20
_BUILT = re.compile(r"R ([0-9]+)\.([0-9]+)\.[0-9]+; ([^;]*); ([^;]+); (windows|unix)")
_DEP = re.compile(r"^([A-Za-z][A-Za-z0-9.]*)\s*(?:\(\s*(>=|<=|==|>|<)\s*([0-9][0-9.\-]*)\s*\))?$")


@dataclass(frozen=True)
class Inspected:
    artifact: Artifact
    description: dict = field(hash=False, compare=False)


def parse_dcf(text: str) -> dict:
    """One Debian Control File record (R's DESCRIPTION). Continuation lines start with whitespace; duplicate fields are refused."""
    fields, key = {}, None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line[0] in " \t":
            if key is None:
                raise AdmissionError("artifact.description_malformed")
            fields[key] += " " + line.strip()
            continue
        if ":" not in line:
            raise AdmissionError("artifact.description_malformed")
        key, value = line.split(":", 1)
        if not key or key in fields:
            raise AdmissionError("artifact.description_duplicate_field:" + key)
        fields[key] = value.strip()
    return fields


def _safe_members(names, package):
    prefix = package + "/"
    for name in names:
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts or "\\" in name:
            raise AdmissionError("artifact.unsafe_member")
        if not (name == prefix or name.startswith(prefix)):
            raise AdmissionError("artifact.layout")


def inspect_archive(path, *, package: str, version: str, kind: str, approved_sha256: str | None = None) -> Inspected:
    """Inspect one acquired archive against the INTENDED identity. The filename is never consulted."""
    path = Path(path)
    if kind == "source":
        try:
            with tarfile.open(path, mode="r:gz") as tf:
                members = tf.getmembers()
                _safe_members([m.name for m in members], package)
                names = [m.name for m in members]
                descs = [m for m in members if m.name == package + "/DESCRIPTION"]
                if len(descs) != 1 or not descs[0].isfile():
                    raise AdmissionError("artifact.description_missing")
                if descs[0].size > _MAX_DESCRIPTION_BYTES:
                    raise AdmissionError("artifact.description_too_large")
                raw = tf.extractfile(descs[0]).read()
        except (tarfile.TarError, OSError, EOFError):
            raise AdmissionError("artifact.unreadable")
    elif kind == "windows_binary":
        try:
            with zipfile.ZipFile(path) as zf:
                infos = zf.infolist()
                names = [i.filename for i in infos]
                _safe_members(names, package)
                descs = [i for i in infos if i.filename == package + "/DESCRIPTION"]
                if len(descs) != 1:
                    raise AdmissionError("artifact.description_missing")
                if descs[0].file_size > _MAX_DESCRIPTION_BYTES:
                    raise AdmissionError("artifact.description_too_large")
                raw = zf.read(descs[0])
        except (zipfile.BadZipFile, OSError, EOFError):
            raise AdmissionError("artifact.unreadable")
    else:
        raise AdmissionError("artifact.kind")
    is_installed_layout = (package + "/Meta/package.rds") in names
    if kind == "windows_binary" and not is_installed_layout:
        raise AdmissionError("artifact.binary_layout")
    if kind == "source" and is_installed_layout:
        raise AdmissionError("artifact.source_is_installed")
    try:
        description = parse_dcf(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise AdmissionError("artifact.description_encoding")
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
    digest = sha256_file(path)
    if approved_sha256 is not None and digest != approved_sha256:
        raise AdmissionError("artifact.digest_mismatch")
    artifact = Artifact(package=package, version=version, kind=kind, sha256=digest, size=path.stat().st_size,
                        description_sha256=hashlib.sha256(raw).hexdigest(), platform=platform, built_r_series=built_series,
                        needs_compilation=needs)
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


def r_version_key(version: str) -> tuple:
    """R's package_version ordering: components split on '.' and '-', compared numerically."""
    if not re.fullmatch(r"[0-9]+([.\-][0-9]+)*", version):
        raise AdmissionError("artifact.version_syntax:" + version)
    return tuple(int(p) for p in re.split(r"[.\-]", version))


def _satisfies(have: str, op: str, need: str) -> bool:
    h, n = r_version_key(have), r_version_key(need)
    return {">=": h >= n, "<=": h <= n, "==": h == n, ">": h > n, "<": h < n}[op]


def check_dependencies(descriptions: dict, planned: dict, runtime_version: str) -> None:
    """Every Depends/Imports/LinkingTo entry of every planned package must be satisfied by the COMPLETE planned set (or by R itself)."""
    problems = []
    for package, description in sorted(descriptions.items()):
        for fieldname in ("Depends", "Imports", "LinkingTo"):
            for entry in filter(None, (e.strip() for e in description.get(fieldname, "").split(","))):
                match = _DEP.fullmatch(entry)
                if match is None:
                    problems.append("{}:{}:unparsed:{}".format(package, fieldname, entry))
                    continue
                name, op, need = match.groups()
                if name == "R":
                    if op and not _satisfies(runtime_version, op, need):
                        problems.append("{}:R {} {} (runtime {})".format(package, op, need, runtime_version))
                    continue
                if name in BASE_PACKAGES:
                    continue
                if name not in planned:
                    problems.append("{}:{}:{} not planned".format(package, fieldname, name))
                elif op and not _satisfies(planned[name], op, need):
                    problems.append("{}:{}:{} {} {} (planned {})".format(package, fieldname, name, op, need, planned[name]))
    if problems:
        raise AdmissionError("artifact.dependencies:" + "; ".join(problems))
