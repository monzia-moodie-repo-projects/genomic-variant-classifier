"""The literature-scout store, driven through the agent that owns it.

LITERATURE-STATE-CWD-RELATIVE-1
===============================
version_monitor_agent.py:58 read `_STATE_PATH = Path("data/agent_state.json")`
-- a RELATIVE path, resolved against the process working directory, both read
(lines 61-63) and written (69-70). The agent is live: registered in the
orchestrator at line 166 and scheduled in the `version_monitor` and
`adaptation` pipelines.

Two divergent copies exist as a result:

    data/agent_state.json                        2026-06-13, 25 keys
    src/.../agent_layer/data/agent_state.json    2026-06-19, 25 keys

Same key set; five values differ; every difference is the nested copy being a
LATER observation. That is STATE-FILE-DUPLICATES-1, reconciled separately.

WHY THESE TESTS EXIST
The two pre-existing test files for this agent -- test_version_monitor_agent.py
and test_version_monitor_deps.py -- both stub `_run_watch_targets` and pass
`dry_run=True`. Line 495 reads `if not dry_run: _set_many(...)`, so NEITHER
ever reaches the store. Replacing it would have been invisible to the whole
suite.

These drive it directly, through an INJECTED store, so nothing touches the real
repository file.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

from genomic_variant_classifier.agent_layer.agents import version_monitor_agent as vm
from genomic_variant_classifier.state.json_state_store import (
    JsonStateStore, PayloadShape, StateCorruptionError,
)


@pytest.fixture
def store(tmp_path):
    """An injected store. Restored afterwards, so no test leaks into another."""
    s = JsonStateStore(path=tmp_path / "literature_scout.json",
                       schema=vm.LITERATURE_SCOUT_SCHEMA)
    vm.set_state_store(s)
    try:
        yield s
    finally:
        vm.set_state_store(None)


# ---- the path is no longer ambient --------------------------------------
def test_the_default_store_is_anchored_not_cwd_relative():
    """THE DEFECT. Path("data/agent_state.json") resolved against wherever the
    process happened to start, which is how one logical store came to exist at
    two depths with divergent contents."""
    vm.set_state_store(None)
    s = vm._state_store()
    assert s.path.is_absolute(), s.path
    assert "literature_scout" in str(s.path)


def test_the_default_store_carries_the_literature_scout_schema():
    """NOT the orchestrator SharedState. Two files named agent_state.json held
    unrelated schemas, and reading the wrong one previously SUCCEEDED."""
    vm.set_state_store(None)
    assert vm._state_store().schema == "gvc.literature-scout-state"
    assert vm.LITERATURE_SCOUT_SCHEMA != "gvc.orchestrator-state"


def test_the_legacy_path_is_recorded_but_not_used():
    """Kept for provenance so a reader can find the two divergent copies."""
    assert str(vm._LEGACY_STATE_PATH) in ("data/agent_state.json",
                                          "data\\agent_state.json")
    vm.set_state_store(None)
    assert vm._state_store().path != vm._LEGACY_STATE_PATH


# ---- injection, which the existing tests could not do -------------------
def test_an_injected_store_is_used_instead_of_the_default(store, tmp_path):
    assert vm._state_store() is store
    assert vm._state_store().path.parent == tmp_path


def test_injection_is_restored_by_the_fixture():
    """set_state_store(None) must restore the anchored default, or one test
    would silently configure the next."""
    vm.set_state_store(None)
    default = vm._state_store()
    s = JsonStateStore(path=Path(tempfile.mkdtemp()) / "x.json",
                       schema=vm.LITERATURE_SCOUT_SCHEMA)
    vm.set_state_store(s)
    assert vm._state_store() is s
    vm.set_state_store(None)
    assert vm._state_store().path == default.path


# ---- the two operations the agent actually uses -------------------------
def test_get_reads_what_set_many_wrote(store):
    """The round trip through the real accessors, at the two keys the agent
    reads: lines 156 and 202."""
    vm._set_many({
        "literature_scout.clinvar_header_hash": "abc123",
        "literature_scout.alphamissense_etag": "W/\"etag\"",
    })
    assert vm._get("literature_scout.clinvar_header_hash", "") == "abc123"
    assert vm._get("literature_scout.alphamissense_etag", "") == 'W/"etag"'


def test_get_returns_the_default_for_an_absent_key(store):
    assert vm._get("literature_scout.never_written", "fallback") == "fallback"


def test_set_many_MERGES_rather_than_replacing(store):
    """The original _set_many loaded, updated and saved. Replacing wholesale
    would discard every baseline not present in this run."""
    vm._set_many({"a": 1, "b": 2})
    vm._set_many({"b": 20, "c": 3})
    assert vm._get("a") == 1
    assert vm._get("b") == 20
    assert vm._get("c") == 3


def test_the_written_payload_carries_an_envelope(store):
    vm._set_many({"literature_scout.last_run": "2026-08-15T00:00:00Z"})
    raw = json.loads(io.open(store.path, encoding="utf-8").read())
    assert raw["schema"] == "gvc.literature-scout-state"
    assert raw["generation"] >= 1
    assert "literature_scout.last_run" in raw["values"]


def test_generation_advances_across_writes(store):
    vm._set_many({"n": 1})
    first = store.load().generation
    vm._set_many({"n": 2})
    assert store.load().generation == first + 1


# ---- corruption is no longer silent -------------------------------------
def test_a_corrupt_store_RAISES_instead_of_reading_as_empty(store):
    """THE COMPOUNDING DEFECT. The previous _load_state swallowed
    JSONDecodeError into {}, and the previous _save_state then wrote that
    emptiness over the original -- destroying the ClinVar header hash and
    AlphaMissense entity tag this agent exists to track."""
    vm._set_many({"literature_scout.clinvar_header_hash": "precious"})
    io.open(store.path, "w", encoding="utf-8").write('{"schema": "gvc.lit')
    with pytest.raises(StateCorruptionError):
        vm._get("literature_scout.clinvar_header_hash", "")
    with pytest.raises(StateCorruptionError):
        vm._set_many({"anything": 1})


def test_a_legacy_bare_payload_is_readable_and_upgraded(store):
    """MEASURED: the real data/agent_state.json is a bare 25-key dict with no
    envelope. The agent must read it, and the first write must preserve every
    key while adding identity."""
    io.open(store.path, "w", encoding="utf-8").write(json.dumps({
        "literature_scout.clinvar_header_hash": "old",
        "literature_scout.deps_outdated_count": 99,
    }))
    assert store.load(allow_legacy=True).shape is PayloadShape.LEGACY_BARE
    assert vm._get("literature_scout.clinvar_header_hash", "") == "old"
    vm._set_many({"literature_scout.deps_outdated_count": 110})
    after = store.load()
    assert after.shape is PayloadShape.ENVELOPED
    assert after.values["literature_scout.clinvar_header_hash"] == "old"
    assert after.values["literature_scout.deps_outdated_count"] == 110


# ---- the dead function is gone ------------------------------------------
def test_the_dead_set_helper_was_removed():
    """_set was defined at lines 77-80 and called NOWHERE -- verified by an
    abstract-syntax-tree call census across src, scripts and tests."""
    assert not hasattr(vm, "_set"), "_set survived; it had zero callers"


def test_the_old_private_helpers_are_gone():
    """_load_state and _save_state are replaced by the store, not wrapped."""
    for name in ("_load_state", "_save_state", "_STATE_PATH"):
        assert not hasattr(vm, name), name


# ---- the agent still works end to end -----------------------------------
def test_run_persists_through_the_store_when_not_dry_run(store, monkeypatch):
    """The path the two pre-existing tests never take.

    They stub _run_watch_targets AND pass dry_run=True, so line 495's
    `if not dry_run: _set_many(...)` never fires. This drives it.
    """
    monkeypatch.setattr(vm, "_check_pykan", lambda: {})
    monkeypatch.setattr(vm, "_check_clinvar_schema", lambda: {})
    monkeypatch.setattr(vm, "_check_alphamissense", lambda: {})
    monkeypatch.setattr(vm, "_check_torch_geometric", lambda: {})
    monkeypatch.setattr(vm, "_check_python", lambda: {})
    monkeypatch.setattr(vm, "_check_dependencies", lambda: {})
    monkeypatch.setattr(vm, "_check_pyg_abi", lambda: {})
    out = vm.run(dry_run=False)
    assert "literature_scout.last_run" in out
    assert store.load().values["literature_scout.last_run"] == \
        out["literature_scout.last_run"]


def test_run_with_dry_run_writes_NOTHING(store, monkeypatch):
    for n in ("_check_pykan", "_check_clinvar_schema", "_check_alphamissense",
              "_check_torch_geometric", "_check_python", "_check_dependencies",
              "_check_pyg_abi"):
        monkeypatch.setattr(vm, n, lambda: {})
    vm.run(dry_run=True)
    assert not store.path.exists(), "dry_run wrote to the store"
