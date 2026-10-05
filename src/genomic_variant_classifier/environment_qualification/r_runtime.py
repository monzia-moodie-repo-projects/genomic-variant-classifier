"""The R runtime checkpoint: a clean probe of an EXPLICIT executable, and the runtime-only lockfile admission.

Owner ruling 2026-10-05b (option B): qualify plain-release R 4.6.1 explicitly. The ONLY permitted lockfile difference at this
checkpoint is /R/Version "4.6.0" -> "4.6.1"; every package record, renv 1.2.3, Bioconductor 3.23, the repositories and their order
and all other metadata stay semantically identical. An incompatibility that needs any further change STOPS the checkpoint -- it
is recorded, never silently turned into a package upgrade. Adding DANDELION is a separate, later checkpoint.

Admitting the lockfile delta declares the proposed target; it is NOT qualification. Qualification also needs the restore,
loaded-namespace and required-test evidence (required_tests.py), bound to this exact candidate by receipt.py.

THE PROBE never relies on PATH and never lets startup processing reach its output: the explicit executable, --vanilla, a neutral
temporary working directory, a child environment without any R_* or RENV_* variable, separate output streams, and a strict
four-line shape. (Measured 2026-10-05: a probe run from the repository folder captured renv's "out-of-sync" message into the
version string.) --vanilla controls R's startup files only; it does not neutralise code sourced later -- renv 1.2.3 load.R
L344-347 reads renv-root, user and project .Renviron files -- so ACTIVATION is a separate, explicit, controlled step.

The launcher digest identifies the launcher only, never the whole R installation.

Author: Monzia Moodie (composed from the owner's reference implementation, ruling 2026-10-05b)
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["AdmissionError", "RUNTIME_BASELINE", "RUNTIME_TARGET", "BIOCONDUCTOR", "RENV", "strict_json", "canonical",
           "validate_lock", "admit_runtime_change", "probe_r"]

#: The ONE declared runtime transition (owner ruling 2026-10-05b). Other code derives its runtime from the authoritative lockfile.
RUNTIME_BASELINE = "4.6.0"
RUNTIME_TARGET = "4.6.1"
BIOCONDUCTOR = "3.23"
RENV = "1.2.3"


class AdmissionError(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise AdmissionError(reason)


def strict_json(text):
    """Duplicate keys and non-finite constants (NaN, Infinity, -Infinity) are refused."""
    def object_pairs(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "duplicate_json_key:" + key)
            obj[key] = value
        return obj

    def invalid_constant(value):
        raise AdmissionError("nonfinite_json:" + value)

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def canonical(obj):
    """Canonical serialisation -- distinguishes true from 1 and refuses non-finite numbers."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def validate_lock(lock):
    require(type(lock) is dict, "lock_object_required")
    for section in ("R", "Bioconductor", "Packages"):
        require(type(lock.get(section)) is dict, "lock_section:" + section)
    require(type(lock["R"].get("Version")) is str, "r_version_type")
    require(type(lock["Bioconductor"].get("Version")) is str, "bioconductor_version_type")
    require(bool(lock["Packages"]), "empty_packages")
    for name, record in lock["Packages"].items():
        require(type(record) is dict, "package_record:" + name)
        require(record.get("Package") == name, "package_name:" + name)
        require(type(record.get("Version")) is str and bool(record["Version"]), "package_version:" + name)
    canonical(lock)


def admit_runtime_change(before, after, observed_version):
    """Admit a candidate lockfile whose ONLY difference from `before` is /R/Version = the observed runtime."""
    validate_lock(before)
    validate_lock(after)
    require(before["R"]["Version"] == RUNTIME_BASELINE, "unexpected_baseline")
    require(observed_version == RUNTIME_TARGET, "unexpected_observed_runtime")
    require(before["Bioconductor"]["Version"] == BIOCONDUCTOR, "unexpected_bioconductor")
    require(before["Packages"].get("renv", {}).get("Version") == RENV, "unexpected_renv")
    expected = copy.deepcopy(before)
    expected["R"]["Version"] = observed_version
    require(canonical(after) == canonical(expected), "change_outside_R.Version")
    return {"schema": "gvc.runtime-lock-delta/1", "before_r": RUNTIME_BASELINE, "after_r": observed_version,
            "package_count": len(before["Packages"]), "packages_unchanged": True, "qualification_complete": False}


_PROBE_KEYS = ("GVC_R_VERSION", "GVC_R_STATUS", "GVC_R_PLATFORM", "GVC_R_HOME")


def probe_r(executable, expected=RUNTIME_TARGET):
    """Identify the R behind an EXPLICIT Rscript path, isolated from startup files, the caller's directory and R/renv variables."""
    exe = Path(executable).resolve(strict=True)
    require(exe.is_file(), "rscript_not_file")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith(("R_", "RENV_"))}
    env["R_DEFAULT_PACKAGES"] = "NULL"
    expression = ('cat("GVC_R_VERSION=", as.character(getRversion()), "\\n", '
                  '"GVC_R_STATUS=", R.version$status, "\\n", '
                  '"GVC_R_PLATFORM=", R.version$platform, "\\n", '
                  '"GVC_R_HOME=", normalizePath(R.home(), winslash="/"), "\\n", sep="")')
    with tempfile.TemporaryDirectory(prefix="gvc-r-probe-") as neutral:
        run = subprocess.run([str(exe), "--vanilla", "--slave", "-e", expression], cwd=neutral, env=env,
                             capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=60, check=False)
    require(run.returncode == 0, "r_probe_exit:" + str(run.returncode))
    require(not run.stderr.strip(), "r_probe_stderr")
    lines = run.stdout.splitlines()
    require(len(lines) == len(_PROBE_KEYS), "r_probe_shape")
    values = {}
    for key, line in zip(_PROBE_KEYS, lines):
        require(line.startswith(key + "="), "r_probe_field:" + key)
        values[key] = line[len(key) + 1:]
    require(values["GVC_R_VERSION"] == expected, "r_probe_version")
    require(values["GVC_R_STATUS"] == "", "r_not_plain_release")
    require(bool(values["GVC_R_PLATFORM"]), "r_probe_platform")
    require(bool(values["GVC_R_HOME"]), "r_probe_home")
    return {"executable": str(exe), "launcher_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
            "version": values["GVC_R_VERSION"], "release_status": values["GVC_R_STATUS"],
            "platform": values["GVC_R_PLATFORM"], "r_home": values["GVC_R_HOME"]}
