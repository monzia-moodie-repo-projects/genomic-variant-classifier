#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_layer/literature_scout_agent.py
=======================================
LiteratureScoutAgent — monitors external library releases and database
version changes that require action in the training pipeline.

Watch targets (Phase 3 initial set):
  1. pykan          — watch for memory/OOM fixes → KAN re-enablement trigger
  2. ClinVar        — watch for schema changes in variant_summary.txt header
  3. gnomAD         — watch for v4.2+ constraint metrics column changes
  4. AlphaMissense  — watch for new hg38 TSV releases
  5. torch-geometric — watch for version bumps matching system torch
  6. Python          — running interpreter vs latest patch / series / EOL
  7. ALL packages    — pip list --outdated; flags major bumps (e.g. pandas 2->3)
  8. PyG companions  — torch_scatter/torch_sparse ABI health vs installed torch

SharedState keys written:
  literature_scout.last_run          ISO timestamp of last check
  literature_scout.pykan_installed   currently installed pykan version
  literature_scout.pykan_latest      latest PyPI pykan version
  literature_scout.pykan_alert       True if newer version available
  literature_scout.pykan_changelog   recent changelog snippet if alert
  literature_scout.clinvar_header_hash   MD5 of first 50 header lines
  literature_scout.gnomad_latest_tag     latest gnomAD release tag seen
  literature_scout.alphamissense_etag    ETag of AlphaMissense download URL
  literature_scout.alerts            list of actionable alert strings

Conventions:
  - No logging.basicConfig() at module level (Issue L)
  - from __future__ import annotations (Issue N)
  - All external I/O wrapped in try/except with graceful degradation
  - Runs headlessly; results surfaced via SharedState only
"""

from __future__ import annotations

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.paths.runtime_paths import resolve_runtime_paths
from genomic_variant_classifier.state.json_state_store import JsonStateStore

import gzip
import hashlib
import json
import logging
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SharedState interface (mirrors DataFreshnessAgent pattern)
# ---------------------------------------------------------------------------
#: The literature-scout store. A FLAT key-value change-detection log --
#: NOT the orchestrator SharedState, which is structured and lives elsewhere.
#: Two files named agent_state.json held these unrelated schemas, and reading
#: the wrong one previously SUCCEEDED and returned a dict that meant something
#: else. The envelope now makes that a loud failure.
LITERATURE_SCOUT_SCHEMA = "gvc.literature-scout-state"

#: Retained for provenance: the path this store used before STATE-STORE-1.
#: It was CWD-RELATIVE, so the destination depended on where the process was
#: launched -- and two divergent copies exist as a result, at data/ and at
#: src/.../agent_layer/data/. STATE-FILE-DUPLICATES-1 reconciles them.
_LEGACY_STATE_PATH = Path("data/agent_state.json")

_store_override: JsonStateStore | None = None


def set_state_store(store: JsonStateStore | None) -> None:
    """Inject a store, for hermetic tests. Pass None to restore the default.

    The two pre-existing tests for this agent stub _run_watch_targets and pass
    dry_run=True, so NEITHER reaches the store at all. Without injection a new
    test could only drive it by writing to the real repository.
    """
    global _store_override
    _store_override = store


def _state_store() -> JsonStateStore:
    """The store to use: injected, else anchored to RuntimePaths.

    Anchored rather than cwd-relative. The previous Path("data/agent_state.json")
    resolved against the working directory, which is how the same logical store
    came to exist at two depths with divergent contents.
    """
    if _store_override is not None:
        return _store_override
    return JsonStateStore(
        path=resolve_runtime_paths().literature_scout_state,
        schema=LITERATURE_SCOUT_SCHEMA,
    )


def _get(key: str, default: Any = None) -> Any:
    """One value. Corruption RAISES rather than reading as absent.

    The previous _load_state swallowed JSONDecodeError into {}, so a truncated
    file reported "no history" and the next _set_many persisted that emptiness
    over the original -- destroying exactly the ClinVar and AlphaMissense
    baselines this agent exists to keep.
    """
    return _state_store().load(allow_legacy=True).values.get(key, default)


def _set_many(updates: dict[str, Any]) -> None:
    """Merge and persist ATOMICALLY.

    The previous _save_state wrote with write_text directly, so an interrupted
    write left partial JSON -- which the previous _load_state then read as an
    empty store. Those two defects compounded.
    """
    _state_store().update(updates)

# ---------------------------------------------------------------------------
# Watch target 1: pykan — KAN re-enablement trigger
# ---------------------------------------------------------------------------
PYPI_PYKAN_URL = "https://pypi.org/pypi/pykan/json"

def _check_pykan() -> dict[str, Any]:
    """
    Compare installed pykan version against latest on PyPI.
    Returns a dict of updates to merge into SharedState.
    """
    updates: dict[str, Any] = {}

    # Installed version
    installed = ""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "pykan"],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                installed = line.split(":", 1)[1].strip()
                break
    except Exception as exc:
        logger.debug("Could not determine installed pykan version: %s", exc)

    updates["literature_scout.pykan_installed"] = installed or "not_installed"

    # Latest on PyPI
    latest = ""
    changelog = ""
    try:
        with urllib.request.urlopen(PYPI_PYKAN_URL, timeout=15) as resp:
            data = json.loads(resp.read())
        latest = data.get("info", {}).get("version", "")
        # Grab description snippet for changelog context (first 500 chars)
        desc = data.get("info", {}).get("description", "")
        changelog = desc[:500] if desc else ""
    except Exception as exc:
        logger.debug("Could not fetch pykan PyPI info: %s", exc)

    updates["literature_scout.pykan_latest"] = latest or "unknown"

    alert = bool(latest and installed and latest != installed)
    updates["literature_scout.pykan_alert"] = alert
    updates["literature_scout.pykan_changelog"] = changelog

    if alert:
        logger.info(
            "pykan update available: installed=%s latest=%s -- "
            "evaluate for KAN re-enablement (check for OOM/memory fixes)",
            installed, latest,
        )

    return updates

# ---------------------------------------------------------------------------
# Watch target 2: ClinVar header schema
# ---------------------------------------------------------------------------
CLINVAR_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

def _check_clinvar_schema() -> dict[str, Any]:
    """
    Fetch only the header line of variant_summary.txt.gz and hash it.
    Alert if the hash changes from the previously stored value.
    """
    updates: dict[str, Any] = {}
    previous_hash = _get("literature_scout.clinvar_header_hash", "")

    try:
        req = urllib.request.Request(
            CLINVAR_SUMMARY_URL,
            headers={"Range": "bytes=0-8191"},  # first 8KB covers headers
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        # Decompress partial gzip — may fail on truncated stream; catch below
        try:
            text = gzip.decompress(raw).decode("utf-8", errors="replace")
        except Exception:
            text = ""

        header_lines = "\n".join(text.splitlines()[:5])
        new_hash = hashlib.md5(header_lines.encode()).hexdigest()
        updates["literature_scout.clinvar_header_hash"] = new_hash

        if previous_hash and new_hash != previous_hash:
            msg = (
                "ClinVar variant_summary.txt header changed -- "
                "verify column order in DataFreshnessAgent patch script"
            )
            logger.warning(msg)
            updates["literature_scout.clinvar_schema_alert"] = msg
        else:
            updates["literature_scout.clinvar_schema_alert"] = ""

    except Exception as exc:
        logger.debug("ClinVar schema check failed: %s", exc)
        updates["literature_scout.clinvar_schema_alert"] = f"check_failed: {exc}"

    return updates

# ---------------------------------------------------------------------------
# Watch target 3: AlphaMissense ETag
# ---------------------------------------------------------------------------
AM_DOWNLOAD_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_hg38.tsv.gz"
)

def _check_alphamissense() -> dict[str, Any]:
    """HEAD request to check ETag — changes when a new version is released."""
    updates: dict[str, Any] = {}
    previous_etag = _get("literature_scout.alphamissense_etag", "")

    try:
        req = urllib.request.Request(AM_DOWNLOAD_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            etag = resp.headers.get("ETag", "")
            last_modified = resp.headers.get("Last-Modified", "")

        updates["literature_scout.alphamissense_etag"] = etag
        updates["literature_scout.alphamissense_last_modified"] = last_modified

        if previous_etag and etag and etag != previous_etag:
            msg = (
                f"AlphaMissense hg38 TSV updated (ETag changed). "
                f"Last-Modified: {last_modified}. "
                f"Re-download and reindex before next run."
            )
            logger.warning(msg)
            updates["literature_scout.alphamissense_alert"] = msg
        else:
            updates["literature_scout.alphamissense_alert"] = ""

    except Exception as exc:
        logger.debug("AlphaMissense check failed: %s", exc)
        updates["literature_scout.alphamissense_alert"] = f"check_failed: {exc}"

    return updates

# ---------------------------------------------------------------------------
# Watch target 4: torch-geometric version vs system torch
# ---------------------------------------------------------------------------
PYPI_PYG_URL = "https://pypi.org/pypi/torch-geometric/json"

def _check_torch_geometric() -> dict[str, Any]:
    """Check if a newer torch-geometric is available on PyPI."""
    updates: dict[str, Any] = {}

    installed = ""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "torch-geometric"],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                installed = line.split(":", 1)[1].strip()
                break
    except Exception as exc:
        logger.debug("Could not determine installed torch-geometric: %s", exc)

    updates["literature_scout.pyg_installed"] = installed or "not_installed"

    latest = ""
    try:
        with urllib.request.urlopen(PYPI_PYG_URL, timeout=15) as resp:
            data = json.loads(resp.read())
        latest = data.get("info", {}).get("version", "")
    except Exception as exc:
        logger.debug("Could not fetch torch-geometric PyPI info: %s", exc)

    updates["literature_scout.pyg_latest"] = latest or "unknown"

    if latest and installed and latest != installed:
        updates["literature_scout.pyg_alert"] = (
            f"torch-geometric update: installed={installed} latest={latest}"
        )
    else:
        updates["literature_scout.pyg_alert"] = ""

    return updates

# ---------------------------------------------------------------------------
# Watch target 5: running Python vs latest patch / series / EOL
# ---------------------------------------------------------------------------
PYTHON_EOL_URL = "https://endoflife.date/api/python.json"

def _check_python() -> dict[str, Any]:
    """Compare the running interpreter against the latest patch in its series and
    the newest stable series (endoflife.date). Network-optional; degrades to just
    the running version on any failure."""
    updates: dict[str, Any] = {}
    running = platform.python_version()
    updates["literature_scout.python_running"] = running
    series = ".".join(running.split(".")[:2])
    try:
        with urllib.request.urlopen(PYTHON_EOL_URL, timeout=15) as resp:
            cycles = json.loads(resp.read())
        latest_series = cycles[0].get("cycle", "") if cycles else ""
        latest_patch = next(
            (c.get("latest", "") for c in cycles if c.get("cycle") == series), ""
        )
        eol = next((c.get("eol") for c in cycles if c.get("cycle") == series), None)
        updates["literature_scout.python_latest_series"] = latest_series
        updates["literature_scout.python_latest_patch"] = latest_patch
        updates["literature_scout.python_eol"] = eol
        parts = []
        if latest_patch and latest_patch != running:
            parts.append(f"patch {running} -> {latest_patch} in {series}")
        if latest_series and latest_series != series:
            parts.append(f"newer series {latest_series} available (running {series})")
        today = datetime.now(timezone.utc).date().isoformat()
        if eol is True or (isinstance(eol, str) and eol < today):
            parts.append(f"series {series} is EOL ({eol})")
        updates["literature_scout.python_alert"] = "; ".join(parts)
    except Exception as exc:
        logger.debug("Python version check failed: %s", exc)
        updates["literature_scout.python_alert"] = ""
        updates["literature_scout.python_check_error"] = str(exc)
    return updates

# ---------------------------------------------------------------------------
# Watch target 6: ALL installed packages vs PyPI (pip list --outdated)
# ---------------------------------------------------------------------------
def _is_major_bump(current: str, latest: str) -> bool:
    """True iff latest's major version exceeds current's (e.g. pandas 2.x -> 3.x)."""
    try:
        return int(str(latest).split(".")[0]) > int(str(current).split(".")[0])
    except (ValueError, IndexError, AttributeError):
        return False

def _check_dependencies() -> dict[str, Any]:
    """Scan every installed package against PyPI via 'pip list --outdated'.
    Surfaces the full outdated set plus the major-version-bump subset (the
    migration-sensitive ones). Subprocess/network-bound; graceful with a generous
    timeout. Distinct from InfrastructureDriftAgent (installed-vs-recorded drift);
    this is installed-vs-latest-upstream."""
    updates: dict[str, Any] = {}
    outdated: list = []
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0 and result.stdout.strip():
            outdated = json.loads(result.stdout)
    except Exception as exc:
        logger.debug("pip list --outdated failed: %s", exc)
        updates["literature_scout.deps_check_error"] = str(exc)

    rows = [
        {"name": p.get("name", ""), "installed": p.get("version", ""),
         "latest": p.get("latest_version", "")}
        for p in outdated
    ]
    major = [
        f"{r['name']} {r['installed']} -> {r['latest']}"
        for r in rows if _is_major_bump(r["installed"], r["latest"])
    ]
    updates["literature_scout.deps_outdated_count"] = len(rows)
    updates["literature_scout.deps_outdated"] = rows
    updates["literature_scout.deps_major_bumps"] = major
    return updates

# ---------------------------------------------------------------------------
# Watch target 7: PyG companion ABI health (torch_scatter/torch_sparse vs torch)
# ---------------------------------------------------------------------------
_PYG_COMPANIONS = ("torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv")

def _installed_version(dist: str):
    """Installed version via importlib.metadata; None if absent."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(dist)
        except PackageNotFoundError:
            return None
    except Exception:
        return None

def _try_import(module_name: str):
    """Import *module_name*; return (ok, error). Catches OSError/SystemError too --
    a CUDA/CPU or torch-version ABI mismatch raises those (WinError 127 /
    0xc0000139), not ImportError, when the compiled .pyd fails to load."""
    import importlib
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

def _check_pyg_abi() -> dict[str, Any]:
    """Detect torch_scatter/torch_sparse built against a different torch than the
    installed one -- the failure that silently breaks GNN test collection. Absent
    companions are FINE (modern PyG uses native scatter)."""
    updates: dict[str, Any] = {}
    torch_ver = _installed_version("torch")
    updates["literature_scout.torch_version"] = torch_ver or "not_installed"
    companions: dict[str, str] = {}
    alert = ""
    for comp in _PYG_COMPANIONS:
        ver = _installed_version(comp)
        if ver is None:
            companions[comp] = "absent"
            continue
        ok, err = _try_import(comp)
        if ok:
            companions[comp] = f"ok ({ver})"
        else:
            companions[comp] = f"BROKEN ({ver}): {err[:90]}"
            alert = (
                f"{comp} {ver} fails to load against torch {torch_ver} "
                "(build/ABI mismatch). Uninstall it (PyG falls back to native "
                "scatter) or reinstall a build matching the installed torch."
            )
    updates["literature_scout.pyg_companions"] = companions
    updates["literature_scout.pyg_abi_alert"] = alert
    return updates

# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------
def run(*, dry_run: bool = False) -> dict[str, Any]:
    """
    Run all watch targets and persist results to SharedState.

    Args:
        dry_run: If True, print results but do not write to agent_state.json.

    Returns:
        Dict of all updates that were (or would be) written to SharedState.
    """
    logger.info("LiteratureScoutAgent starting ...")
    all_updates: dict[str, Any] = {
        "literature_scout.last_run": datetime.now(timezone.utc).isoformat(),
    }

    # Collect alerts
    alerts: list[str] = []

    # pykan
    pykan_updates = _check_pykan()
    all_updates.update(pykan_updates)
    if pykan_updates.get("literature_scout.pykan_alert"):
        installed = pykan_updates.get("literature_scout.pykan_installed", "?")
        latest = pykan_updates.get("literature_scout.pykan_latest", "?")
        alerts.append(
            f"[KAN] pykan {latest} available (installed: {installed}). "
            f"Review changelog for OOM/memory fixes before re-enabling KAN."
        )

    # ClinVar schema
    clinvar_updates = _check_clinvar_schema()
    all_updates.update(clinvar_updates)
    if clinvar_updates.get("literature_scout.clinvar_schema_alert"):
        alerts.append(
            f"[ClinVar] {clinvar_updates['literature_scout.clinvar_schema_alert']}"
        )

    # AlphaMissense
    am_updates = _check_alphamissense()
    all_updates.update(am_updates)
    if am_updates.get("literature_scout.alphamissense_alert"):
        alerts.append(
            f"[AlphaMissense] {am_updates['literature_scout.alphamissense_alert']}"
        )

    # torch-geometric
    pyg_updates = _check_torch_geometric()
    all_updates.update(pyg_updates)
    if pyg_updates.get("literature_scout.pyg_alert"):
        alerts.append(f"[PyG] {pyg_updates['literature_scout.pyg_alert']}")

    # Python version
    py_updates = _check_python()
    all_updates.update(py_updates)
    if py_updates.get("literature_scout.python_alert"):
        alerts.append(f"[Python] {py_updates['literature_scout.python_alert']}")

    # ALL installed packages vs PyPI
    dep_updates = _check_dependencies()
    all_updates.update(dep_updates)
    _major = dep_updates.get("literature_scout.deps_major_bumps", [])
    for _mb in _major:
        alerts.append(f"[deps:major] {_mb}")
    _n_out = dep_updates.get("literature_scout.deps_outdated_count", 0)
    if _n_out and not _major:
        alerts.append(f"[deps] {_n_out} package(s) have newer releases (no major bumps)")

    # PyG companion ABI health
    abi_updates = _check_pyg_abi()
    all_updates.update(abi_updates)
    if abi_updates.get("literature_scout.pyg_abi_alert"):
        alerts.append(f"[PyG-ABI] {abi_updates['literature_scout.pyg_abi_alert']}")

    all_updates["literature_scout.alerts"] = alerts

    if alerts:
        logger.info("LiteratureScoutAgent: %d alert(s):", len(alerts))
        for a in alerts:
            logger.info("  * %s", a)
    else:
        logger.info("LiteratureScoutAgent: no alerts.")

    if not dry_run:
        _set_many(all_updates)
        logger.info("LiteratureScoutAgent: state written to %s",
                    _state_store().path)
    else:
        logger.info("LiteratureScoutAgent: dry_run=True, state not written.")

    return all_updates



_run_watch_targets = run  # module-level watch-target orchestrator (aliased before the method below)


class VersionMonitorAgent(BaseAgent):
    """Upstream-release monitor: pykan / ClinVar / AlphaMissense / torch-geometric.

    Distinct from InfrastructureDriftAgent (which diffs *installed* package versions):
    this watches for *new upstream releases*. BaseAgent adapter over the module-level
    watch-target functions; surfaces a summary into the 'version_monitor' section.
    """

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)
        updates = _run_watch_targets(dry_run=dry_run)
        alerts = updates.get("literature_scout.alerts", [])
        result = {
            "status": "ok",
            "n_alerts": len(alerts),
            "alerts": alerts,
            "pykan_installed": updates.get("literature_scout.pykan_installed"),
            "pykan_latest": updates.get("literature_scout.pykan_latest"),
            "pykan_alert": updates.get("literature_scout.pykan_alert", False),
            "python_running": updates.get("literature_scout.python_running"),
            "python_alert": updates.get("literature_scout.python_alert", ""),
            "deps_outdated_count": updates.get("literature_scout.deps_outdated_count", 0),
            "deps_major_bumps": updates.get("literature_scout.deps_major_bumps", []),
            "pyg_abi_alert": updates.get("literature_scout.pyg_abi_alert", ""),
            "last_run": updates.get("literature_scout.last_run"),
            "checked_at": self._now_iso(),
            "dry_run": dry_run,
        }
        self._update_section("version_monitor", result)
        self._log_finish(result)
        return result


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description="VersionMonitorAgent")
    p.add_argument("--dry-run", action="store_true",
                   help="Print results without writing to agent_state.json")
    args = p.parse_args()
    results = run(dry_run=args.dry_run)
    print(json.dumps(
        {k: v for k, v in results.items() if "changelog" not in k},
        indent=2,
    ))
