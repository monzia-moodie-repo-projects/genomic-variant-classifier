"""Tests for the atomic, identified, fail-closed JSON state store.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import pytest

from genomic_variant_classifier.state.json_state_store import (
    GENERATION_KEY, SCHEMA_KEY, SCHEMA_VERSION_KEY, VALUES_KEY, JsonStateStore,
    LoadedState, PayloadShape, StateCorruptionError, StateSchemaMismatch,
    StateStoreError,
)

SCHEMA = "gvc.literature-scout-state"


def _store(tmp: Path = None, schema: str = SCHEMA) -> JsonStateStore:
    d = tmp or Path(tempfile.mkdtemp())
    return JsonStateStore(path=d / "state.json", schema=schema)


# ---- the defect this module exists for ---------------------------------
def test_a_corrupt_store_RAISES_rather_than_reporting_empty():
    """THE DEFECT. version_monitor_agent.py:64-65 swallows JSONDecodeError into
    {}, and the next _set_many writes that emptiness over the original. A
    truncated file becomes a silently reset change-detection baseline."""
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    io.open(s.path, "w", encoding="utf-8").write('{"schema": "x", "val')
    with pytest.raises(StateCorruptionError) as exc:
        s.load(allow_legacy=True)
    assert "DAMAGE as ABSENCE" in str(exc.value)


def test_an_ABSENT_store_is_empty_and_that_is_not_an_error():
    """Absent and damaged are different answers."""
    s = _store()
    got = s.load()
    assert got.values == {}
    assert got.shape is PayloadShape.ABSENT


def test_a_non_object_payload_RAISES():
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    io.open(s.path, "w", encoding="utf-8").write("[1, 2, 3]")
    with pytest.raises(StateCorruptionError):
        s.load(allow_legacy=True)


# ---- schema identity ----------------------------------------------------
def test_a_payload_from_a_DIFFERENT_store_RAISES():
    """Two files named agent_state.json held unrelated schemas -- a flat
    literature-scout log and the orchestrator's structured state. Reading the
    wrong one previously SUCCEEDED and returned a dict that meant something
    else."""
    s = _store()
    s.save({"a": 1})
    other = JsonStateStore(path=s.path, schema="gvc.orchestrator-state")
    with pytest.raises(StateSchemaMismatch) as exc:
        other.load()
    assert "unrelated schemas" in str(exc.value)


def test_the_envelope_carries_schema_version_and_generation():
    s = _store()
    s.save({"a": 1})
    raw = json.loads(io.open(s.path, encoding="utf-8").read())
    assert raw[SCHEMA_KEY] == SCHEMA
    assert raw[SCHEMA_VERSION_KEY] == 1
    assert raw[GENERATION_KEY] == 1
    assert raw[VALUES_KEY] == {"a": 1}


def test_generation_advances_monotonically():
    """Stronger ordering than file modification time. Reconciling two copies of
    the literature-scout store required comparing five values by hand because
    neither carried a logical clock."""
    s = _store()
    assert s.save({"n": 1}) == 1
    assert s.update({"n": 2}) == 2
    assert s.update({"n": 3}) == 3
    assert s.load().generation == 3


# ---- legacy payloads are readable, deliberately -------------------------
def test_a_LEGACY_bare_dict_is_readable_when_asked_for():
    """MEASURED: data/agent_state.json is a bare flat dict of 25 keys with no
    envelope. A store that refused it could not read the data it exists to
    migrate."""
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    io.open(s.path, "w", encoding="utf-8").write(
        json.dumps({"literature_scout.last_run": "2026-06-13T19:14:10Z"}))
    got = s.load(allow_legacy=True)
    assert got.shape is PayloadShape.LEGACY_BARE
    assert got.values["literature_scout.last_run"].startswith("2026-06-13")
    assert got.generation == 0


def test_a_LEGACY_payload_is_REFUSED_unless_asked_for():
    """Reading legacy data must be a deliberate act, not a default."""
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    io.open(s.path, "w", encoding="utf-8").write('{"a": 1}')
    with pytest.raises(StateSchemaMismatch) as exc:
        s.load()
    assert "allow_legacy=True" in str(exc.value)


def test_require_enveloped_rejects_a_legacy_load():
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    io.open(s.path, "w", encoding="utf-8").write('{"a": 1}')
    got = s.load(allow_legacy=True)
    with pytest.raises(StateSchemaMismatch):
        got.require_enveloped()


def test_require_enveloped_accepts_an_enveloped_load():
    s = _store()
    s.save({"a": 1})
    assert s.load().require_enveloped().values == {"a": 1}


# ---- atomicity ----------------------------------------------------------
def test_the_temporary_file_is_in_the_SAME_directory_as_the_target():
    """os.replace is an atomic rename only within one filesystem. A temporary
    file in the system temp directory could be on another device, making the
    'atomic' write a copy."""
    s = _store()
    seen = {}
    import genomic_variant_classifier.state.json_state_store as M
    real = M.tempfile.mkstemp

    def spy(*a, **k):
        seen["dir"] = k.get("dir")
        return real(*a, **k)

    M.tempfile.mkstemp = spy
    try:
        s.save({"a": 1})
    finally:
        M.tempfile.mkstemp = real
    assert seen["dir"] == str(s.path.parent)


def test_a_failed_write_leaves_NO_temporary_file_behind():
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)

    class _Unserialisable:
        pass

    import genomic_variant_classifier.state.json_state_store as M
    real_dump = M.json.dump

    def boom(*a, **k):
        raise OSError("disk full")

    M.json.dump = boom
    try:
        with pytest.raises(OSError):
            s.save({"a": 1})
    finally:
        M.json.dump = real_dump
    leftovers = [p.name for p in s.path.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers


def test_a_failed_write_leaves_the_ORIGINAL_intact():
    s = _store()
    s.save({"keep": "this"})
    before = io.open(s.path, "rb").read()
    import genomic_variant_classifier.state.json_state_store as M
    real_dump = M.json.dump
    M.json.dump = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    try:
        with pytest.raises(OSError):
            s.save({"clobber": "that"})
    finally:
        M.json.dump = real_dump
    assert io.open(s.path, "rb").read() == before


def test_fsync_is_called_before_the_rename():
    """SharedState omits fsync, so os.replace is atomic against other PROCESSES
    while the bytes may still sit in the operating system cache at power loss.
    For change-detection baselines that is worth the cost."""
    order = []
    import genomic_variant_classifier.state.json_state_store as M
    real_fsync, real_replace = M.os.fsync, M.os.replace
    M.os.fsync = lambda fd: order.append("fsync")
    M.os.replace = lambda a, b: (order.append("replace"), real_replace(a, b))[1]
    try:
        _store().save({"a": 1})
    finally:
        M.os.fsync, M.os.replace = real_fsync, real_replace
    assert order == ["fsync", "replace"], order


# ---- the two operations the agent actually needs ------------------------
def test_get_returns_a_value_or_the_default():
    """Measured: version_monitor_agent calls _get at lines 156 and 202 only."""
    s = _store()
    s.save({"literature_scout.clinvar_header_hash": "abc"})
    assert s.get("literature_scout.clinvar_header_hash", "") == "abc"
    assert s.get("literature_scout.absent", "fallback") == "fallback"


def test_update_MERGES_rather_than_replacing():
    """_set_many loads, updates and saves. Replacing wholesale would discard
    every baseline not present in this run's updates."""
    s = _store()
    s.save({"a": 1, "b": 2})
    s.update({"b": 20, "c": 3})
    assert s.load().values == {"a": 1, "b": 20, "c": 3}


def test_update_works_against_a_LEGACY_store_without_losing_keys():
    """The migration path: the store on disk is bare, and the first update must
    preserve all of it while adding the envelope."""
    s = _store()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    io.open(s.path, "w", encoding="utf-8").write(json.dumps({"old": 1, "keep": 2}))
    s.update({"new": 3})
    got = s.load()
    assert got.shape is PayloadShape.ENVELOPED
    assert got.values == {"old": 1, "keep": 2, "new": 3}


def test_non_dict_input_is_REFUSED():
    s = _store()
    with pytest.raises(StateStoreError):
        s.save(["not", "a", "dict"])
    with pytest.raises(StateStoreError):
        s.update("neither")


# ---- immutability and provenance ----------------------------------------
def test_the_store_and_its_loads_are_immutable():
    import dataclasses
    s = _store()
    s.save({"a": 1})
    got = s.load()
    for obj, field, value in ((s, "schema", "x"), (got, "generation", 99)):
        try:
            setattr(obj, field, value)
        except dataclasses.FrozenInstanceError:
            continue
        raise AssertionError("{} was mutable".format(type(obj).__name__))


def test_load_returns_values_INDEPENDENT_of_the_parsed_envelope():
    """What dict() actually guarantees.

    An earlier version mutated `got.values` and then re-read the FILE, which
    passes whether or not a copy is made -- json.loads builds a fresh object
    per call, so every load is already independent of every other. That test
    proved nothing, and sabotage confirmed it: replacing dict(values) with
    values went undetected.

    The real property is that LoadedState.values is not the SAME object the
    envelope dict holds, so a caller's edit cannot reach through into a
    structure the store still refers to.
    """
    s = _store()
    s.save({"a": 1})
    import json as _json
    raw = _json.loads(io.open(s.path, encoding="utf-8").read())
    got = s.load()
    assert got.values == raw[VALUES_KEY]
    assert got.values is not raw[VALUES_KEY]


def test_two_loads_do_not_share_a_values_object():
    """Measured: json.loads yields a fresh object per call, so this holds by
    construction. Asserted anyway, because a future caching layer would break
    it silently and this is where that would surface."""
    s = _store()
    s.save({"a": 1})
    first, second = s.load(), s.load()
    assert first.values == second.values
    assert first.values is not second.values
    first.values["a"] = 999
    assert second.values["a"] == 1


def test_the_path_is_SUPPLIED_not_computed():
    """OUTPUT-ROOT-CONFLATION-1 as a constructor argument: this module never
    derives a location from repository layout."""
    import ast
    import genomic_variant_classifier.state.json_state_store as M
    src = io.open(M.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            b = getattr(n, "body", None)
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(getattr(b[0], "value", None), ast.Constant)):
                docs.add(id(b[0].value))
    live = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]
    import re
    drive = re.compile(r"^[A-Za-z]:[\\/]")
    bad = [x for x in live if (x.startswith("/") and len(x) > 8) or drive.match(x)]
    assert not bad, bad
    assert "PROJECT_ROOT" not in src.replace("OUTPUT-ROOT-CONFLATION-1", "")
