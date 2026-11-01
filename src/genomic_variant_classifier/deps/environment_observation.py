"""Observe one Python environment's installed distributions without changing it -- the evidence an environment contract judges.

Run with the interpreter being measured, isolated and without bytecode:
    PYTHON -I -B -m genomic_variant_classifier.deps.environment_observation NEW_OUTPUT.json

Standard library only (no third-party package and no machine-learning package is imported), so it measures the environment instead of
disturbing it. Writes exactly one NEW file (refuses an existing one). Runs `pip --version`, `pip check` and `pip freeze --all` with that
interpreter, with pip's network version check disabled.
Exit 0: the observation was written (a failing `pip check` is a FINDING recorded in it, not a failure of the observation).
Exit 2: refused (wrong flags, existing output, bad arguments). Exit 3: the observation could not be completed.

Provenance: the same program (SHA-256 abce53268558...) observed the owner's .venv312 on 2026-10-31 (block E1, observation a4fc5829...)
and the locked Linux environment on 2026-11-01; the schema is unchanged, so those observations remain comparable.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import logging
import os
import platform
import re
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = "gvc.environment-observation/1"
PIP_TIMEOUT_SECONDS = 600


def canonical_name(name: str) -> str:
    """PEP 503 normalisation (the rule packaging.utils.canonicalize_name applies)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def marker_environment() -> dict:
    """The PEP 508 marker variables, computed as packaging.markers.default_environment computes them."""
    iv = sys.implementation.version
    implementation_version = f"{iv.major}.{iv.minor}.{iv.micro}"
    if iv.releaselevel != "final":
        implementation_version += iv.releaselevel[0] + str(iv.serial)
    return {
        "implementation_name": sys.implementation.name,
        "implementation_version": implementation_version,
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_full_version": platform.python_version(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


def distribution_row(dist: md.Distribution) -> dict:
    meta = dist.metadata
    name = meta["Name"]
    version = meta["Version"]
    row: dict = {
        "name": name,
        "canonical_name": canonical_name(name) if isinstance(name, str) and name else None,
        "version": version,
        "requires_python": meta.get("Requires-Python"),
        "requires_dist": meta.get_all("Requires-Dist") or [],
        "provides_extra": meta.get_all("Provides-Extra") or [],
        "metadata_path": str(getattr(dist, "_path", "")) or None,
        "site_location": str(dist.locate_file("")),
        "installer": None,
        "direct_url": None,
        "record_present": dist.read_text("RECORD") is not None,
        "problems": [],
    }
    if not isinstance(name, str) or not name:
        row["problems"].append("metadata has no Name")
    if not isinstance(version, str) or not version:
        row["problems"].append("metadata has no Version")
    installer = dist.read_text("INSTALLER")
    if installer is not None:
        row["installer"] = installer.strip()
    raw_url = dist.read_text("direct_url.json")
    if raw_url is not None:
        try:
            row["direct_url"] = json.loads(raw_url)
        except ValueError as exc:
            row["direct_url"] = {"unparseable": raw_url}
            row["problems"].append(f"direct_url.json is not JSON: {exc}")
    return row


def run_pip(*arguments: str) -> dict:
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-I", "-B", "-m", "pip", *arguments]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=PIP_TIMEOUT_SECONDS, env=env)
    return {"arguments": list(arguments), "exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr}


def observe() -> dict:
    rows = [distribution_row(d) for d in md.distributions()]
    rows.sort(key=lambda r: ((r["canonical_name"] or ""), r["metadata_path"] or ""))
    seen: dict = {}
    for r in rows:
        if r["canonical_name"]:
            seen.setdefault(r["canonical_name"], []).append(r["metadata_path"])
    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    exe = Path(sys.executable)
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    return {
        "schema": SCHEMA,
        "author": "Monzia Moodie",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Installed-distribution metadata and pip's own reports for ONE interpreter; not artifact integrity, "
                 "not lock conformance, not model behaviour.",
        "interpreter": {
            "executable": str(exe),
            "executable_resolved": str(exe.resolve()),
            "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "is_virtual_environment": sys.prefix != sys.base_prefix,
            "pyvenv_cfg": cfg.read_text(encoding="utf-8") if cfg.is_file() else None,
            "sys_path": list(sys.path),
            "soabi": sysconfig.get_config_var("SOABI"),
            "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
            "flags": {"isolated": sys.flags.isolated, "dont_write_bytecode": sys.dont_write_bytecode},
            "platform": platform.platform(),
        },
        "marker_environment": marker_environment(),
        "distributions": rows,
        "summary": {
            "distributions": len(rows),
            "distinct_canonical_names": len(seen),
            "duplicated_names": duplicates,
            "rows_with_problems": sum(1 for r in rows if r["problems"]),
            "editable_or_url_installs": sorted(r["canonical_name"] or "?" for r in rows if r["direct_url"] is not None),
        },
        "pip_version": run_pip("--version"),
        "pip_check": run_pip("check"),
        "pip_freeze_all": run_pip("freeze", "--all"),
    }


def main(argv: list) -> int:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        print("REFUSED: run this interpreter with -I -B", file=sys.stderr)
        return 2
    if len(argv) != 1:
        print("REFUSED: exactly one argument, the NEW output file", file=sys.stderr)
        return 2
    out = Path(argv[0])
    if not out.is_absolute():
        print("REFUSED: the output path must be absolute", file=sys.stderr)
        return 2
    if out.exists():
        print(f"REFUSED: {out} already exists", file=sys.stderr)
        return 2
    try:
        result = observe()
    except Exception as exc:  # noqa: BLE001 -- reported loudly with its type, never swallowed
        print(f"FAILED: the observation could not be completed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    text = json.dumps(result, indent=1, ensure_ascii=True) + "\n"
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    s = result["summary"]
    print(f"observation: {out}")
    print(f"sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}  bytes {len(text.encode('utf-8'))}")
    print(f"distributions {s['distributions']}  distinct names {s['distinct_canonical_names']}  "
          f"duplicated {len(s['duplicated_names'])}  with metadata problems {s['rows_with_problems']}  "
          f"editable/url {len(s['editable_or_url_installs'])}")
    print(f"pip --version exit {result['pip_version']['exit_code']}: {result['pip_version']['stdout'].strip()}")
    print(f"pip check exit {result['pip_check']['exit_code']}  pip freeze --all exit {result['pip_freeze_all']['exit_code']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
