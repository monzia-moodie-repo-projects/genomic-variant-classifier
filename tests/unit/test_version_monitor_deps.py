"""Tests for the VersionMonitorAgent dependency/Python/ABI watch targets.

All network and subprocess I/O is monkeypatched -- these tests never touch PyPI,
endoflife.date, pip, or the import system for real. They verify parsing, alert
construction, graceful degradation, run() aggregation, and that the BaseAgent
wrapper surfaces the new fields. Author: Monzia Moodie
"""
from __future__ import annotations

import json

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents import version_monitor_agent as vm


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _Resp:
    """Minimal context-manager stand-in for urllib.request.urlopen()."""
    def __init__(self, payload): self._payload = payload
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return json.dumps(self._payload).encode("utf-8")


class _Proc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode, self.stderr = stdout, returncode, ""


# --------------------------------------------------------------------------- #
# _is_major_bump
# --------------------------------------------------------------------------- #
def test_is_major_bump_pandas_2_to_3():
    assert vm._is_major_bump("2.3.3", "3.0.2") is True

def test_is_major_bump_minor_is_not_major():
    assert vm._is_major_bump("2.11.0+cpu", "2.12.0") is False

def test_is_major_bump_garbage_is_false():
    assert vm._is_major_bump("not-a-version", "x") is False


# --------------------------------------------------------------------------- #
# _check_dependencies
# --------------------------------------------------------------------------- #
def test_check_dependencies_parses_and_flags_major(monkeypatch):
    payload = [
        {"name": "pandas", "version": "2.3.3", "latest_version": "3.0.2"},
        {"name": "numpy",  "version": "2.4.4", "latest_version": "2.4.6"},
    ]
    monkeypatch.setattr(vm.subprocess, "run",
                        lambda *a, **k: _Proc(stdout=json.dumps(payload)))
    u = vm._check_dependencies()
    assert u["literature_scout.deps_outdated_count"] == 2
    assert u["literature_scout.deps_major_bumps"] == ["pandas 2.3.3 -> 3.0.2"]
    assert {"name": "numpy", "installed": "2.4.4", "latest": "2.4.6"} in \
        u["literature_scout.deps_outdated"]

def test_check_dependencies_graceful_when_pip_raises(monkeypatch):
    def boom(*a, **k): raise OSError("pip unavailable")
    monkeypatch.setattr(vm.subprocess, "run", boom)
    u = vm._check_dependencies()
    assert u["literature_scout.deps_outdated_count"] == 0
    assert "literature_scout.deps_check_error" in u

def test_check_dependencies_empty_output(monkeypatch):
    monkeypatch.setattr(vm.subprocess, "run", lambda *a, **k: _Proc(stdout=""))
    assert vm._check_dependencies()["literature_scout.deps_outdated_count"] == 0


# --------------------------------------------------------------------------- #
# _check_python
# --------------------------------------------------------------------------- #
_CYCLES = [
    {"cycle": "3.13", "latest": "3.13.2", "eol": "2029-10-31"},
    {"cycle": "3.12", "latest": "3.12.11", "eol": "2028-10-31"},
    {"cycle": "3.9",  "latest": "3.9.20", "eol": "2025-10-31"},
]

def test_check_python_alerts_on_newer_patch_and_series(monkeypatch):
    monkeypatch.setattr(vm.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(_CYCLES))
    monkeypatch.setattr(vm.platform, "python_version", lambda: "3.12.10", raising=False)
    u = vm._check_python()
    assert u["literature_scout.python_running"] == "3.12.10"
    a = u["literature_scout.python_alert"]
    assert "3.12.11" in a and "3.13" in a

def test_check_python_graceful_offline(monkeypatch):
    def boom(*a, **k): raise OSError("no network")
    monkeypatch.setattr(vm.urllib.request, "urlopen", boom)
    u = vm._check_python()
    assert u["literature_scout.python_alert"] == ""
    assert u["literature_scout.python_running"]   # still reports running version


# --------------------------------------------------------------------------- #
# _check_pyg_abi
# --------------------------------------------------------------------------- #
def test_check_pyg_abi_detects_broken_companion(monkeypatch):
    monkeypatch.setattr(vm, "_installed_version",
                        lambda d: {"torch": "2.11.0+cpu",
                                   "torch_scatter": "2.1.2+pt25cu124"}.get(d))
    monkeypatch.setattr(vm, "_try_import",
                        lambda m: (False, "OSError: WinError 127"))
    u = vm._check_pyg_abi()
    assert u["literature_scout.pyg_abi_alert"]
    assert "BROKEN" in u["literature_scout.pyg_companions"]["torch_scatter"]
    assert u["literature_scout.pyg_companions"]["torch_sparse"] == "absent"

def test_check_pyg_abi_absent_is_clean(monkeypatch):
    monkeypatch.setattr(vm, "_installed_version",
                        lambda d: "2.11.0+cpu" if d == "torch" else None)
    u = vm._check_pyg_abi()
    assert u["literature_scout.pyg_abi_alert"] == ""
    assert all(v == "absent" for k, v in u["literature_scout.pyg_companions"].items())

def test_check_pyg_abi_present_and_loads(monkeypatch):
    monkeypatch.setattr(vm, "_installed_version",
                        lambda d: {"torch": "2.11.0+cpu",
                                   "torch_scatter": "2.1.2"}.get(d))
    monkeypatch.setattr(vm, "_try_import", lambda m: (True, ""))
    u = vm._check_pyg_abi()
    assert u["literature_scout.pyg_abi_alert"] == ""
    assert u["literature_scout.pyg_companions"]["torch_scatter"].startswith("ok")


# --------------------------------------------------------------------------- #
# run() aggregation + dry-run safety
# --------------------------------------------------------------------------- #
def test_run_aggregates_new_alerts(monkeypatch, tmp_path):
    # silence the four pre-existing checks
    for fn in ("_check_pykan", "_check_clinvar_schema",
               "_check_alphamissense", "_check_torch_geometric",
               # ADDED 2026-09-14 with the gnomAD watch target. Targets are
               # silenced BY NAME, so every new one must appear here or it
               # executes for real. MEASURED: without this entry the test
               # still PASSES -- its assertions use `any` -- while making a
               # live HTTP request, which this file's docstring forbids.
               "_check_gnomad_release"):
        monkeypatch.setattr(vm, fn, lambda: {})
    monkeypatch.setattr(vm, "_check_python",
                        lambda: {"literature_scout.python_alert": "patch 3.12.10 -> 3.12.11 in 3.12",
                                 "literature_scout.python_running": "3.12.10"})
    monkeypatch.setattr(vm, "_check_dependencies",
                        lambda: {"literature_scout.deps_outdated_count": 1,
                                 "literature_scout.deps_major_bumps": ["pandas 2.3.3 -> 3.0.2"]})
    monkeypatch.setattr(vm, "_check_pyg_abi",
                        lambda: {"literature_scout.pyg_abi_alert": "torch_scatter ABI mismatch"})
    # dry_run must NOT write state
    monkeypatch.setattr(vm, "_set_many",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("wrote state in dry_run")))
    out = vm.run(dry_run=True)
    alerts = out["literature_scout.alerts"]
    assert any(a.startswith("[Python]") for a in alerts)
    assert any(a.startswith("[deps:major]") for a in alerts)
    assert any(a.startswith("[PyG-ABI]") for a in alerts)


def test_wrapper_surfaces_new_fields(monkeypatch):
    monkeypatch.setattr(vm, "_run_watch_targets", lambda *, dry_run=False: {
        "literature_scout.last_run": "t",
        "literature_scout.alerts": [],
        "literature_scout.python_running": "3.12.10",
        "literature_scout.deps_outdated_count": 3,
        "literature_scout.pyg_abi_alert": "",
    })
    agent = vm.VersionMonitorAgent(SharedState())
    r = agent.run(dry_run=True)
    assert r["python_running"] == "3.12.10"
    assert r["deps_outdated_count"] == 3
    assert r["pyg_abi_alert"] == ""
    sec = agent._get_section("version_monitor")
    assert sec["deps_outdated_count"] == 3


# --------------------------------------------------------------------------- #
# _check_gnomad_release -- added 2026-09-14
#
# The module docstring promised "3. gnomAD -- watch for v4.2+ constraint
# metrics column changes" since the file was written, and run() dispatched
# seven targets, none of them gnomAD. MEASURED: gnomAD v4.1.1 was released
# 2026-03-30; VersionMonitorAgent RAN on 2026-06-20 and recorded status "ok".
#
# These use the MEASURED prefix list from the live endpoint, so a passing test
# reflects the bucket's real shape rather than an invented one.
# --------------------------------------------------------------------------- #

#: MEASURED 2026-09-14 from
#: storage.googleapis.com/storage/v1/b/gcp-public-data--gnomad/o
#: ?prefix=release/&delimiter=/&fields=prefixes
_GNOMAD_PREFIXES_2026_09_14 = [
    "release/2.1.1/", "release/2.1/", "release/3.0.1/", "release/3.0/",
    "release/3.1.1/", "release/3.1.2/", "release/3.1.3/", "release/3.1/",
    "release/4.0/", "release/4.1.1/", "release/4.1/", "release/v4.0/",
]


def test_parse_release_version_reads_the_measured_prefixes():
    assert vm._parse_release_version("release/4.1.1/") == (4, 1, 1)
    assert vm._parse_release_version("release/4.1/") == (4, 1)
    assert vm._parse_release_version("release/2.1.1/") == (2, 1, 1)


def test_parse_release_version_strips_the_v_prefix():
    """The bucket holds BOTH release/4.0/ and release/v4.0/, measured
    2026-09-14. A strict numeric parse raises on the second."""
    assert vm._parse_release_version("release/v4.0/") == (4, 0)
    assert vm._parse_release_version("release/4.0/") == (4, 0)


def test_parse_release_version_refuses_rather_than_guessing():
    for bad in ("release/", "release/latest/", "release/4.x/", "release/beta/"):
        assert vm._parse_release_version(bad) is None


def test_version_ordering_is_numeric_not_lexicographic():
    """As STRINGS "4.10" < "4.9" is True, which would call 4.10 the older
    release. This bucket will eventually have a 4.10."""
    assert "4.10" < "4.9"
    assert vm._parse_release_version("release/4.10/") > \
        vm._parse_release_version("release/4.9/")
    assert vm._parse_release_version("release/4.1.1/") > \
        vm._parse_release_version("release/4.1/")


def test_check_gnomad_release_alerts_on_the_measured_listing(monkeypatch):
    """THE ALERT THAT WAS NEVER RAISED. Against the real prefix list and the
    pinned 4.1, a newer release exists and must be reported."""
    monkeypatch.setattr(vm.urllib.request, "urlopen",
                        lambda *a, **k: _Resp({"prefixes": _GNOMAD_PREFIXES_2026_09_14}))
    out = vm._check_gnomad_release()
    assert out["literature_scout.gnomad_latest_tag"] == "4.1.1"
    assert "4.1.1" in out["literature_scout.gnomad_alert"]
    assert out["literature_scout.gnomad_unrecognised"] == []


def test_check_gnomad_release_is_silent_when_the_pin_is_current(monkeypatch):
    """A watch that always alerts is not a watch."""
    monkeypatch.setattr(vm.urllib.request, "urlopen",
                        lambda *a, **k: _Resp({"prefixes": ["release/4.0/", "release/4.1/"]}))
    assert vm._check_gnomad_release()["literature_scout.gnomad_alert"] == ""


def test_check_gnomad_release_reports_failure_not_silence(monkeypatch):
    """A FAILED CHECK IS NOT A CLEAN CHECK. An unreachable endpoint must never
    read as 'no new release' -- the defect class this agent already embodies."""
    def boom(*a, **k):
        raise OSError("simulated network failure")
    monkeypatch.setattr(vm.urllib.request, "urlopen", boom)
    alert = vm._check_gnomad_release()["literature_scout.gnomad_alert"]
    assert alert.startswith("check_failed:")


def test_check_gnomad_release_reports_an_unparseable_listing(monkeypatch):
    """If every prefix stops parsing, the format changed. Saying nothing would
    be indistinguishable from 'no new release'."""
    monkeypatch.setattr(vm.urllib.request, "urlopen",
                        lambda *a, **k: _Resp({"prefixes": ["release/latest/", "release/beta/"]}))
    out = vm._check_gnomad_release()
    assert "NONE parsed" in out["literature_scout.gnomad_alert"]


def test_run_dispatches_the_gnomad_target(monkeypatch):
    """run() dispatches BY HAND-WRITTEN SEQUENCE, with no registry, which is
    how a documented target went unimplemented for months. This asserts the
    dispatch exists rather than trusting the docstring."""
    for fn in ("_check_pykan", "_check_clinvar_schema", "_check_alphamissense",
               "_check_torch_geometric", "_check_python", "_check_dependencies",
               "_check_pyg_abi"):
        monkeypatch.setattr(vm, fn, lambda: {})
    monkeypatch.setattr(vm, "_check_gnomad_release",
                        lambda: {"literature_scout.gnomad_alert": "v9.9 available"})
    monkeypatch.setattr(vm, "_set_many", lambda *_a, **_k: None)
    alerts = vm.run(dry_run=True)["literature_scout.alerts"]
    assert any(a.startswith("[gnomAD]") for a in alerts)


def test_status_is_degraded_when_a_check_fails(monkeypatch):
    """status was a LITERAL "ok", set whether or not any target succeeded, and
    check_agents_active.py reported that constant back as the agent's health."""
    monkeypatch.setattr(vm, "_run_watch_targets", lambda *, dry_run=False: {
        "literature_scout.alerts": ["[gnomAD] check_failed: unreachable"],
    })
    agent = vm.VersionMonitorAgent(SharedState())
    r = agent.run(dry_run=True)
    assert r["status"] == "degraded"
    assert r["n_checks_failed"] == 1


def test_status_is_ok_when_no_check_failed(monkeypatch):
    monkeypatch.setattr(vm, "_run_watch_targets", lambda *, dry_run=False: {
        "literature_scout.alerts": ["[gnomAD] 4.1.1 available"],
    })
    r = vm.VersionMonitorAgent(SharedState()).run(dry_run=True)
    assert r["status"] == "ok"
    assert r["n_checks_failed"] == 0
