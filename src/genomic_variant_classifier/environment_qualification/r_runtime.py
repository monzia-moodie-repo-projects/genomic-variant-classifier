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
           "validate_lock", "admit_runtime_change", "sha256_file", "run_r_file", "probe_r"]

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
_PROBE_PROGRAM = ('cat(\n'
                  '  "GVC_R_VERSION=", as.character(getRversion()), "\\n",\n'
                  '  "GVC_R_STATUS=", R.version$status, "\\n",\n'
                  '  "GVC_R_PLATFORM=", R.version$platform, "\\n",\n'
                  '  "GVC_R_HOME=", normalizePath(R.home(), winslash = "/"), "\\n",\n'
                  '  sep = ""\n'
                  ')\n')


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_r_file(executable, program: str, evidence_dir, *, child_env: dict, timeout_seconds: int = 120) -> dict:
    """Run R code from a FILE (never through -e on a command line) and keep the complete evidence.

    The owner's reference (ruling 2026-10-06, generation 027757ee L227-314), refined: the evidence folder and record are created
    BEFORE the executable is resolved, so a missing executable is recorded as start_failed with process.json (measured 2026-10-06:
    the reference's resolve(strict=True) raised before any evidence existed). Raw stdout/stderr are written BEFORE the exit status is
    judged. This function RETURNS the record (with the raw streams); the CALLER decides refusal with its own reason code.
    Short probes only: it does not terminate a process TREE, so long builds need a separate supervisor."""
    evidence_dir = Path(evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    script = evidence_dir / "probe.R"
    script.write_bytes(program.encode("utf-8"))
    record = {"schema_version": 1, "executable": str(executable), "executable_sha256": None,
              "script_sha256": sha256_file(script), "status": "not_started", "returncode": None}
    stdout, stderr = b"", b""
    try:
        exe = Path(executable).resolve(strict=True)
        record["executable"] = str(exe)
        record["executable_sha256"] = sha256_file(exe)
        with tempfile.TemporaryDirectory(prefix="gvc-r-probe-") as cwd:
            result = subprocess.run([str(exe), "--vanilla", str(script)], cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds, check=False, shell=False)
        stdout, stderr = result.stdout, result.stderr
        record["returncode"] = result.returncode
        record["status"] = "exited_zero" if result.returncode == 0 else "exited_nonzero"
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
        record["status"] = "timeout"
    except OSError as exc:                      # includes FileNotFoundError from resolve(strict=True)
        record["status"] = "start_failed"
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
    finally:
        (evidence_dir / "stdout.bin").write_bytes(stdout)
        (evidence_dir / "stderr.bin").write_bytes(stderr)
        record["stdout_sha256"] = sha256_file(evidence_dir / "stdout.bin")
        record["stderr_sha256"] = sha256_file(evidence_dir / "stderr.bin")
        (evidence_dir / "process.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["stdout"], record["stderr"] = stdout, stderr
    return record


def probe_r(executable, evidence_dir, expected=RUNTIME_TARGET):
    """Identify the R behind an EXPLICIT Rscript path from a FILE program, isolated from startup files, the caller's directory and
    R/renv variables. Every outcome leaves evidence in `evidence_dir` (process.json, probe.R, stdout.bin, stderr.bin)."""
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith(("R_", "RENV_"))}
    env["R_DEFAULT_PACKAGES"] = "NULL"
    run = run_r_file(executable, _PROBE_PROGRAM, evidence_dir, child_env=env, timeout_seconds=60)
    require(run["status"] != "start_failed", "r_probe_start_failed")
    require(run["status"] != "timeout", "r_probe_timeout")
    require(run["returncode"] == 0, "r_probe_exit:" + str(run["returncode"]))
    try:
        out, err = run["stdout"].decode("utf-8"), run["stderr"].decode("utf-8")
    except UnicodeDecodeError:
        raise AdmissionError("r_probe_encoding")
    require(not err.strip(), "r_probe_stderr")
    lines = out.splitlines()
    require(len(lines) == len(_PROBE_KEYS), "r_probe_shape")
    values = {}
    for key, line in zip(_PROBE_KEYS, lines):
        require(line.startswith(key + "="), "r_probe_field:" + key)
        values[key] = line[len(key) + 1:]
    require(values["GVC_R_VERSION"] == expected, "r_probe_version")
    require(values["GVC_R_STATUS"] == "", "r_not_plain_release")
    require(bool(values["GVC_R_PLATFORM"]), "r_probe_platform")
    require(bool(values["GVC_R_HOME"]), "r_probe_home")
    return {"executable": run["executable"], "launcher_sha256": run["executable_sha256"],
            "version": values["GVC_R_VERSION"], "release_status": values["GVC_R_STATUS"],
            "platform": values["GVC_R_PLATFORM"], "r_home": values["GVC_R_HOME"]}
