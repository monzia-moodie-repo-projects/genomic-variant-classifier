"""Tests-wide pytest conftest for tests/unit/.

Path-aware SpliceAI isolation: blocks BaseConnector cache I/O ONLY when the
cache target resolves under the production data/raw/cache/ directory.
tmp_path-scoped FetchConfigs (the normal test pattern) are unaffected and
exercise the real cache load/save flow.

History:
- 2026-04-XX: class-scoped fixture in TestAnnotationPipeline (test_core.py)
  patched DEFAULT_SPLICEAI_PATH and BaseConnector._load_cache, but NOT
  _save_cache. fetch()'s unconditional _save_cache(cache_key, result) at
  database_connectors.py L217 wrote the stub-zero result to
  data/raw/cache/spliceai_scores_snv.parquet (430 MB). The fixture also
  did not cover TestSpliceAIConnector (a sibling class), leaving 73 of
  the SpliceAI tests with no isolation.
- 2026-05-10 first attempt: module-scoped autouse with UNCONDITIONAL
  _load_cache and _save_cache no-ops. Fixed the leak but broke
  TestSpliceAIConnector::test_parquet_cache_used_on_second_call, which
  uses a tmp_path cache_dir and legitimately exercises cache write->read.
- 2026-05-10 final: path-aware patches. Block load/save only when
  cache_path resolves under data/raw/cache/.

See memory #5 / #9 and INCIDENT_2026-04-29_gcp-billing-deletion.md.

Also here (2026-11-05, TESTS-REACH-THE-NETWORK-1): the offline_freshness_sources fixture, requested by the tests that run the
data-freshness agents for the orchestrator's sake; and the program_starts fixture, which records and blocks every program a test
starts, in the form the platform itself reports it (PROGRAM-START-RECORD-PLATFORM-FORM-1).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_PROD_CACHE_DIR = Path("data/raw/cache")


def _is_prod_cache_path(cache_path: Path) -> bool:
    """True iff cache_path resolves inside the production data/raw/cache/."""
    try:
        cache_path.resolve().relative_to(_PROD_CACHE_DIR.resolve())
        return True
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _isolate_spliceai(monkeypatch, tmp_path):
    """Path-aware SpliceAI isolation for every test in tests/unit/.

    Three patches:
      - DEFAULT_SPLICEAI_PATH -> a nonexistent path under tmp_path. Tests
        constructing SpliceAIConnector() with no vcf_path and no explicit
        DEFAULT_SPLICEAI_PATH override see no real index and take the
        stub-zero code path.
      - BaseConnector._load_cache -> return None ONLY when cache_path
        resolves under data/raw/cache/. tmp_path caches load normally.
      - BaseConnector._save_cache -> no-op ONLY when cache_path resolves
        under data/raw/cache/. tmp_path caches save normally. THIS is the
        actual leak fix; the prior class-scoped fixture only nulled
        _load_cache, leaving _save_cache to write zeros to the 430 MB
        prod cache file at the end of fetch().

    Tests that genuinely need to exercise prod-cache I/O can override these
    via in-test monkeypatch.
    """
    from genomic_variant_classifier.data import spliceai as _spliceai_mod
    from genomic_variant_classifier.data.database_connectors import BaseConnector

    monkeypatch.setattr(
        _spliceai_mod, "DEFAULT_SPLICEAI_PATH", tmp_path / "nonexistent.parquet"
    )

    _orig_load_cache = BaseConnector._load_cache
    _orig_save_cache = BaseConnector._save_cache

    def _safe_load_cache(self, key):
        if _is_prod_cache_path(self._cache_path(key)):
            return None
        return _orig_load_cache(self, key)

    def _safe_save_cache(self, key, df):
        if _is_prod_cache_path(self._cache_path(key)):
            return None
        return _orig_save_cache(self, key, df)

    monkeypatch.setattr(BaseConnector, "_load_cache", _safe_load_cache)
    monkeypatch.setattr(BaseConnector, "_save_cache", _safe_save_cache)


# ---------------------------------------------------------------------------
# The upstream data sources, offline (added 2026-11-05, TESTS-REACH-THE-NETWORK-1).
# ---------------------------------------------------------------------------
# MEASURED 2026-11-05: three tests that run the "data_freshness" pipeline for the ORCHESTRATOR's behaviour (agent isolation, the
# dry run's telemetry, the artifact gate) made DataFreshnessAgent poll ClinVar's FTP server, gnomAD and AlphaMissense on
# storage.googleapis.com and LOVD -- in a dry run -- and the agent swallowed each failure. A test that requests this fixture gets
# the four polls replaced by "no change detected" (what the agent returns when a poll fails), and the registry monitor's probe
# replaced by one that reports every source unreachable; nothing is fetched. It returns the list of polls made, in order, so a
# test can show the agent ran. The repository root's conftest.py refuses and fails any network attempt.
@pytest.fixture
def offline_freshness_sources(monkeypatch):
    from genomic_variant_classifier.agent_layer.agents import database_freshness_detector as detector
    from genomic_variant_classifier.agent_layer.agents.data_freshness_agent import DataFreshnessAgent

    polled = []

    def no_change(source):
        def poll(self, prev):
            polled.append(source)
            return False
        return poll

    for source in ("clinvar", "gnomad", "lovd", "alphamissense"):
        monkeypatch.setattr(DataFreshnessAgent, "_poll_" + source, no_change(source))
    monkeypatch.setattr(detector, "_default_probe", lambda source: (None, "offline under test (no probe is made)"))
    return polled


# ---------------------------------------------------------------------------
# Program starts, recorded and blocked (added 2026-11-05, PROGRAM-START-RECORD-PLATFORM-FORM-1).
# ---------------------------------------------------------------------------
# MEASURED 2026-11-05 on the owner's Windows machine (set "ah" V1, targeted stage): the first version of this recorder lived in
# tests/unit/test_inference_backend_trace.py and read the command of a "subprocess.Popen" audit event as a list of arguments.
# That is what CPython reports on POSIX; on Windows its _execute_child joins the list into ONE command line
# (subprocess.list2cmdline) and passes executable=None before raising the event (Lib/subprocess.py, 3.11 to 3.13 alike). The test
# failed there; it had been verified only on Linux. The first version also called sys.addaudithook at the test module's import,
# so its own import check, which executes the module a second time, left a second hook in the process for the rest of the session.
#
# This recorder is installed ONCE, when this conftest is imported (pytest imports a conftest once per session; audit hooks cannot
# be removed). It does nothing until a test requests program_starts. While that test runs, every event below is recorded and, by
# default, blocked by raising RuntimeError inside the call, so the program never runs:
#   subprocess.Popen, os.system, os.exec, os.spawn, os.posix_spawn, os.startfile, os.startfile/2, os.fork, os.forkpty -- every
#   Python-level way of starting a program -- and sys.addaudithook (a blocked addition is dropped by CPython, so no hook leaks).
# A subprocess.Popen record is {"event", "executable", "command", "cwd"} exactly as the platform raised it (paths decoded with
# os.fsdecode); program_starts.subprocess_start(argv, cwd) is the record that platform raises for subprocess.run(argv, cwd=cwd)
# with its defaults, and tests/unit/test_program_start_recorder.py compares it with a REAL start, on whichever machine runs it.
# NOT COVERED, stated: _posixsubprocess.fork_exec called directly (MEASURED 2026-11-05, CPython 3.11, 3.12 and 3.13 on Linux: it
# raises no audit event; multiprocessing's spawn and forkserver start methods use it) and _winapi.CreateProcess called directly
# (multiprocessing's spawn on Windows); and anything done by a program that is not this Python process.
_PROGRAM_START_EVENTS = frozenset({"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.startfile",
                                   "os.startfile/2", "os.fork", "os.forkpty", "sys.addaudithook"})


class ProgramStarts:
    """What program_starts yields: the records of the requesting test, and the platform's own form of one start."""

    def __init__(self) -> None:
        self.recording = False
        self.block = True
        self.seen: list = []

    def _record(self, event: str, args: tuple) -> None:
        if not self.recording or event not in _PROGRAM_START_EVENTS:
            return
        if event == "subprocess.Popen":
            executable, command, cwd = args[0], args[1], args[2]
            if not isinstance(command, (str, bytes)):
                command = [os.fsdecode(a) for a in command]
            self.seen.append({"event": event, "executable": None if executable is None else os.fsdecode(executable),
                              "command": command if isinstance(command, list) else os.fsdecode(command),
                              "cwd": None if cwd is None else os.fsdecode(cwd)})
        else:
            self.seen.append({"event": event, "arguments": args})
        if self.block:
            raise RuntimeError("program start blocked by the test (recorded): " + event)

    @staticmethod
    def subprocess_start(argv, cwd=None) -> dict:
        """The record of subprocess.run(argv, cwd=cwd) on this platform, shell, executable and close_fds at their defaults."""
        argv = [os.fsdecode(a) for a in argv]
        cwd = None if cwd is None else os.fsdecode(cwd)
        if os.name == "nt":
            return {"event": "subprocess.Popen", "executable": None, "command": subprocess.list2cmdline(argv), "cwd": cwd}
        return {"event": "subprocess.Popen", "executable": argv[0], "command": argv, "cwd": cwd}


_PROGRAM_STARTS = ProgramStarts()
sys.addaudithook(_PROGRAM_STARTS._record)


@pytest.fixture
def program_starts():
    """Records, and by default blocks, every program the requesting test starts (set .block = False to let them run)."""
    _PROGRAM_STARTS.seen = []
    _PROGRAM_STARTS.block = True
    _PROGRAM_STARTS.recording = True
    try:
        yield _PROGRAM_STARTS
    finally:
        _PROGRAM_STARTS.recording = False
        _PROGRAM_STARTS.block = True


# ---------------------------------------------------------------------------
# The API's optional data tables, absent (added 2026-11-06, DATA-GATED-TESTS-1).
# ---------------------------------------------------------------------------
# MEASURED 2026-11-06 by the data census on the owner's machine: the five tests that run the API's real startup (the lifespan) --
# test_runtime_attribution's registered_production_client and its negative twin, and test_containment_serving's quarantine test --
# READ data/processed/dbsnp_index.parquet from his checkout. api/main.py takes GENE_SUMMARY_PATH and DBSNP_INDEX_PATH from the
# environment with defaults RELATIVE TO THE CURRENT FOLDER (data/processed/...); both files are git-ignored, so a checkout loads
# them and a clone does not, and the startup leaves the loaded tables (_GENE_SUMMARY, _DBSNP_INDEX, _GNOMAD_INDEX) in the module
# for every later test. The outcomes happened to agree; the code paths did not. A test that requests this fixture gets all three
# tables pointed at paths that do not exist (gnomAD's off, as when GNOMAD_INDEX_PATH is unset) and the three module tables put back
# afterwards. tests/unit/test_local_data_contract.py requires every test or fixture that runs the startup to request it.
@pytest.fixture
def api_without_local_data(tmp_path, monkeypatch):
    import genomic_variant_classifier.api.main as api_main

    absent = tmp_path / "api_local_data_absent"
    monkeypatch.setattr(api_main, "GENE_SUMMARY_PATH", absent / "gene_summary.parquet")
    monkeypatch.setattr(api_main, "DBSNP_INDEX_PATH", absent / "dbsnp_index.parquet")
    monkeypatch.setattr(api_main, "GNOMAD_INDEX_PATH", None)
    for table in ("_GENE_SUMMARY", "_DBSNP_INDEX", "_GNOMAD_INDEX"):
        monkeypatch.setattr(api_main, table, getattr(api_main, table))      # restored at teardown, whatever the startup loaded
    return api_main
