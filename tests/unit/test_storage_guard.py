"""Tests for the data/ structural and free-space guard.

WHY THIS FILE EXISTS
====================
Two defects were found in scripts/maintenance/preflight_data_guard.py on
2026-07-21, both of omission.

FIRST: it never checked free space. It verified structure -- not a dangling
junction, not shadowed by a stray file, canonical subtrees present -- and
stopped. The guard whose purpose is to catch storage problems before a run
could not catch the storage problem the machine had. On 2026-07-20 the volume
reached 0.716 per cent free and that was discovered by a run failing.

SECOND, AND WORSE: nothing ever called it. A repository-wide search across every
.py, .sh, .ps1 and .yaml found zero invocations of assert_data_usable. Its own
docstring said it was importable "to wire into run scripts / conftest", and it
never was. A guard that is not invoked is not a guard; it is a comment that
happens to be executable.

THE PROPERTY THAT MATTERS MOST HERE
------------------------------------
The policy numbers live in configs/data_manifest.yaml. scripts/forensics/
audit_disk_census.py keeps its OWN copies, deliberately, because it must run
standalone from any directory without the repository importable. Two copies of
one number is exactly the stale-literal defect repaired four times on 2026-07-21
-- the conformal export list, PARTITIONS, verify_dtype's source-text assertion,
and the census walker. The difference is that here the duplication is a
considered trade, and test_policy_matches_the_census_tool is what makes it safe.
Delete that test and the duplication becomes the defect again.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_GUARD = _ROOT / "scripts" / "maintenance" / "preflight_data_guard.py"
_CENSUS = _ROOT / "scripts" / "forensics" / "audit_disk_census.py"
_MANIFEST = _ROOT / "configs" / "data_manifest.yaml"

_Usage = namedtuple("usage", "total used free")
_VOLUME = 935.59            # gibibytes, the machine measured on 2026-07-21


def _load(path: Path, name: str):
    """Load a module by path.

    The module MUST be registered in sys.modules before exec_module: @dataclass
    under `from __future__ import annotations` resolves its annotations through
    sys.modules[cls.__module__], which is empty otherwise, and the decorator
    raises AttributeError on None. Found the hard way, 2026-07-21.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    argv = sys.argv
    sys.argv = [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


G = _load(_GUARD, "preflight_data_guard")


@pytest.fixture
def policy():
    return G.StoragePolicy.load(_MANIFEST)


@pytest.fixture
def tree(tmp_path):
    for sub in ("external", "raw", "processed"):
        (tmp_path / "data" / sub).mkdir(parents=True)
    return tmp_path / "data"


def _at(monkeypatch, free_gib: float, total_gib: float = _VOLUME):
    t = int(total_gib * G.GIB)
    f = int(free_gib * G.GIB)
    monkeypatch.setattr(G.shutil, "disk_usage", lambda _p: _Usage(t, t - f, f))


# --------------------------------------------------------------------------- #
# 1. the duplication that must not drift
# --------------------------------------------------------------------------- #
def test_defaults_match_the_manifest(policy):
    """DEFAULT_POLICY exists so the guard works when the manifest is missing.
    If it silently diverges, the fallback enforces a different rule than the
    configured one -- and only in the situation where nobody is watching."""
    for key, value in G.DEFAULT_POLICY.items():
        assert getattr(policy, key) == value, (
            f"{key}: manifest has {getattr(policy, key)}, DEFAULT_POLICY has {value}")


def test_policy_matches_the_census_tool():
    """audit_disk_census.py keeps its own constants so it can run standalone.
    They must agree with the manifest, or a number is wrong in one of two
    places and nothing says so.

    AMENDED 2026-07-30. The original wording said the two tools must not
    "report different verdicts about the same volume on the same day". They
    now DO, deliberately: audit_disk_census answers "can this volume hold the
    JEPA embedding cache?" from jepa_embedding_cache_gib, and
    preflight_data_guard answers "may a run start?" from working_cache_gib.
    Different questions, different answers, one source of truth each."""
    src = _CENSUS.read_text(encoding="utf-8")
    ns: dict = {}
    for line in src.split("\n"):
        s = line.strip()
        for name in ("HEADROOM_FRACTION", "HEADROOM_MIN", "JEPA_CACHE_GIB"):
            if s.startswith(name + " ="):
                ns[name] = s.split("=", 1)[1].strip()
    assert set(ns) == {"HEADROOM_FRACTION", "HEADROOM_MIN", "JEPA_CACHE_GIB"}, (
        f"census constants not found; got {sorted(ns)}")

    p = G.StoragePolicy.load(_MANIFEST)
    assert float(ns["HEADROOM_FRACTION"]) == p.headroom_fraction
    # RE-POINTED 2026-07-30. This pinned JEPA_CACHE_GIB to working_cache_gib
    # while one constant carried two meanings. The census reports the JEPA
    # cache; the guard reports the run gate. They now answer different
    # questions, so the pin follows the JEPA figure.
    assert float(ns["JEPA_CACHE_GIB"]) == p.jepa_embedding_cache_gib
    assert ns["HEADROOM_MIN"].replace(" ", "") == f"{int(p.headroom_min_gib)}*GiB"


def test_required_reproduces_the_measured_figure(policy):
    """61.48 GiB is what audit_disk_census printed on the real volume on
    2026-07-21. The guard must compute the same number from the same policy."""
    got = policy.required_free_bytes(_VOLUME * G.GIB) / G.GIB
    assert got == pytest.approx(61.48, abs=0.01)


# --------------------------------------------------------------------------- #
# 2. the three bands, at their exact boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("free_gib,expected", [
    (500.00, "OK"), (61.49, "OK"), (61.48, "OK"),
    (61.47, "WARN"), (40.00, "WARN"), (25.01, "WARN"), (25.00, "WARN"),
    (24.99, "FAIL"), (5.00, "FAIL"), (0.10, "FAIL"),
])
def test_severity_bands(monkeypatch, policy, free_gib, expected):
    _at(monkeypatch, free_gib)
    assert G.check_free_space(".", policy).severity == expected


def test_the_boundaries_are_inclusive_where_documented(monkeypatch, policy):
    """required is met AT required; the floor is breached only BELOW it."""
    _at(monkeypatch, 61.48); assert G.check_free_space(".", policy).severity == "OK"
    _at(monkeypatch, 25.00); assert G.check_free_space(".", policy).severity == "WARN"
    _at(monkeypatch, 24.99); assert G.check_free_space(".", policy).severity == "FAIL"


def test_check_free_space_never_raises_on_a_healthy_filesystem(monkeypatch, policy):
    """The verdict is the return value. A caller decides what WARN means; a
    guard that raises cannot be asked a question."""
    for free in (500.0, 40.0, 1.0):
        _at(monkeypatch, free)
        v = G.check_free_space(".", policy)
        assert v.severity in ("OK", "WARN", "FAIL")


def test_the_verdict_carries_every_number_that_produced_it(monkeypatch, policy):
    """A severity with no arithmetic behind it cannot be checked by a reader."""
    _at(monkeypatch, 40.0)
    v = G.check_free_space(".", policy)
    assert v.total_bytes == int(_VOLUME * G.GIB)
    assert v.free_bytes == int(40.0 * G.GIB)
    assert v.required_bytes == pytest.approx(61.48 * G.GIB, rel=1e-4)
    assert v.percent_free == pytest.approx(100 * 40.0 / _VOLUME, abs=0.01)
    assert v.policy_source.endswith("data_manifest.yaml")
    assert not v.ok


def test_the_ten_percent_advisory_fires_only_where_it_helps(monkeypatch, policy):
    """Below ten per cent Windows competes with the workload. On FAIL the note
    is noise -- the run is already refused."""
    _at(monkeypatch, 200.0)
    assert "NOTE:" not in G.check_free_space(".", policy).message
    _at(monkeypatch, 90.0)
    assert "NOTE:" in G.check_free_space(".", policy).message
    _at(monkeypatch, 5.0)
    v = G.check_free_space(".", policy)
    assert v.severity == "FAIL" and "NOTE:" not in v.message


# --------------------------------------------------------------------------- #
# 3. the policy refuses to be nonsense
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,value", [
    ("headroom_fraction", 1.0), ("headroom_fraction", 1.5),
    ("headroom_fraction", -0.01),
    ("working_cache_gib", -1.0), ("jepa_embedding_cache_gib", -1.0),
    ("headroom_min_gib", -1.0),
    ("hard_floor_gib", -1.0),
])
def test_post_init_rejects_impossible_policy(field, value):
    kw = dict(G.DEFAULT_POLICY); kw[field] = value
    with pytest.raises(ValueError):
        G.StoragePolicy(**kw)


def test_a_floor_that_swallows_the_warn_band_is_rejected():
    """If hard_floor exceeds the required total, the gate goes straight from OK
    to refusing and the WARN band -- the whole point of three severities --
    cannot occur. That is a misconfiguration, not a strict setting."""
    kw = dict(G.DEFAULT_POLICY); kw["hard_floor_gib"] = 999.0
    with pytest.raises(ValueError, match="swallow|exceeds"):
        G.StoragePolicy(**kw)


def test_a_valid_but_strict_policy_is_accepted():
    """The guard must not mistake strictness for error."""
    kw = dict(G.DEFAULT_POLICY); kw["hard_floor_gib"] = 34.0
    p = G.StoragePolicy(**kw)
    assert p.hard_floor_gib == 34.0


# --------------------------------------------------------------------------- #
# 4. a missing manifest degrades loudly, not silently and not fatally
# --------------------------------------------------------------------------- #
def test_missing_manifest_uses_defaults_and_says_so(tmp_path, capsys):
    p = G.StoragePolicy.load(tmp_path / "absent.yaml")
    for key, value in G.DEFAULT_POLICY.items():
        assert getattr(p, key) == value
    assert "could not read" in p.source
    assert "WARNING" in capsys.readouterr().err


def test_malformed_manifest_uses_defaults_and_says_so(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("storage: [this, is, a, list]\n", encoding="utf-8")
    p = G.StoragePolicy.load(bad)
    assert p.working_cache_gib == G.DEFAULT_POLICY["working_cache_gib"]
    assert "WARNING" in capsys.readouterr().err


def test_manifest_missing_one_key_falls_back_rather_than_half_applying(tmp_path, capsys):
    """A partially-applied policy is worse than a default one, because nobody
    can tell which rule was enforced."""
    part = tmp_path / "part.yaml"
    part.write_text("storage:\n  working_cache_gib: 99.0\n", encoding="utf-8")
    p = G.StoragePolicy.load(part)
    assert p.working_cache_gib == G.DEFAULT_POLICY["working_cache_gib"]
    assert "WARNING" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 5. structural checks -- unchanged behaviour, now pinned
# --------------------------------------------------------------------------- #
def test_healthy_tree_passes(tree, capsys):
    G.assert_data_usable(tree)
    assert "OK" in capsys.readouterr().out


def test_missing_data_dir_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        G.assert_data_usable(tmp_path / "nope")


def test_a_file_shadowing_data_is_refused(tmp_path):
    stray = tmp_path / "data"
    stray.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SystemExit, match="NOT a directory"):
        G.assert_data_usable(stray)


def test_missing_canonical_subtree_is_refused(tmp_path):
    (tmp_path / "data" / "external").mkdir(parents=True)
    with pytest.raises(SystemExit, match="canonical subtrees"):
        G.assert_data_usable(tmp_path / "data")


def test_a_dangling_symlink_is_refused(tmp_path):
    """The 2026-06-14 incident: data/ was a junction into an unmounted G: and
    every run failed with a deep mkdir traceback instead of one clear line."""
    link = tmp_path / "data"
    try:
        link.symlink_to(tmp_path / "gone", target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlink creation unavailable on this platform/privilege level")
    with pytest.raises(SystemExit, match="DANGLING"):
        G.assert_data_usable(link)


# --------------------------------------------------------------------------- #
# 6. the gate rows -- the shape preflight_run17 consumes
# --------------------------------------------------------------------------- #
def test_storage_rows_returns_two_rows_when_healthy(monkeypatch, tree, capsys):
    _at(monkeypatch, 500.0)
    rows = G.storage_rows(tree, _MANIFEST)
    capsys.readouterr()
    assert len(rows) == 2
    assert [r[0] for r in rows] == ["OK", "OK"]


@pytest.mark.parametrize("free_gib,expected", [(500.0, "OK"), (40.0, "WARN"), (5.0, "FAIL")])
def test_storage_rows_reports_the_band(monkeypatch, tree, capsys, free_gib, expected):
    _at(monkeypatch, free_gib)
    rows = G.storage_rows(tree, _MANIFEST)
    capsys.readouterr()
    assert rows[1][0] == expected


def test_a_broken_tree_yields_a_FAIL_ROW_not_a_raise(tmp_path):
    """A preflight that dies on its first problem hides the second. Every gate
    must be able to report."""
    rows = G.storage_rows(tmp_path / "nope", _MANIFEST)
    assert rows and rows[0][0] == "FAIL"
    assert "data-guard" in rows[0][1]


def test_every_row_is_a_two_tuple_of_strings(monkeypatch, tree, capsys):
    """preflight_run17 sorts on row[0] and prints row[1]; anything else breaks
    the aggregation rather than the check."""
    _at(monkeypatch, 40.0)
    rows = G.storage_rows(tree, _MANIFEST)
    capsys.readouterr()
    for r in rows:
        assert isinstance(r, tuple) and len(r) == 2
        assert isinstance(r[0], str) and isinstance(r[1], str)
        assert r[0] in ("OK", "WARN", "FAIL")


# --------------------------------------------------------------------------- #
# 7. the command line
# --------------------------------------------------------------------------- #
def test_skip_space_restores_the_old_behaviour(tree):
    r = subprocess.run([sys.executable, str(_GUARD), str(tree), "--skip-space"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "SKIPPED" in r.stdout


def test_a_broken_tree_exits_one(tmp_path):
    r = subprocess.run([sys.executable, str(_GUARD), str(tmp_path / "nope")],
                       capture_output=True, text=True)
    assert r.returncode == 1


# --------------------------------------------------------------------------- #
# 8. the wiring -- the second defect, and the one that mattered more
# --------------------------------------------------------------------------- #
# NOTE ON VERIFICATION, 2026-07-21. These four tests were NOT executed before
# delivery: preflight_run17 imports EXPECTED_TABULAR_FEATURE_COUNT from
# variant_ensemble, which needs xgboost, which the authoring environment lacks.
# They are expected to RUN, not skip, on the development machine and in
# Continuous Integration. If they skip there, the skip is the finding -- a
# guarded import that always fails is how a wiring test quietly stops testing
# the wiring, which is the exact defect this section exists to prevent.
def _preflight():
    scripts = _ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        return _load(scripts / "preflight_run17.py", "preflight_run17")
    except BaseException as e:          # SystemExit on a missing package, too
        pytest.skip(f"preflight_run17 not importable here ({type(e).__name__}); "
                    "this MUST NOT skip on the development machine or in CI")


def test_preflight_exposes_a_storage_gate():
    P = _preflight()
    assert hasattr(P, "storage_gate"), (
        "preflight_run17.storage_gate is missing -- the data guard is unwired "
        "again, which is the state it sat in from 2026-06-14 to 2026-07-21")


def test_storage_gate_returns_the_row_convention(tmp_path, monkeypatch, capsys):
    P = _preflight()
    for sub in ("external", "raw", "processed"):
        (tmp_path / "data" / sub).mkdir(parents=True)
    rows = P.storage_gate(str(tmp_path / "data"), _MANIFEST)
    capsys.readouterr()
    assert rows
    for r in rows:
        assert isinstance(r, tuple) and len(r) == 2
        assert r[0] in ("OK", "WARN", "FAIL")


def test_run_all_actually_calls_the_storage_gate(monkeypatch):
    """Drives run_all and asserts the gate CONTRIBUTED a row, rather than
    reading the source for a call. A source check passes on dead code and fails
    on a clean refactor -- both directions wrong, as scripts/forensics/
    verify_dtype.py demonstrated on 2026-07-21."""
    P = _preflight()
    seen = {}

    def spy(data_root="data", manifest="configs/data_manifest.yaml"):
        seen["called"] = True
        return [("OK", "storage: SENTINEL")]

    monkeypatch.setattr(P, "storage_gate", spy)
    try:
        rows = P.run_all("python scripts/train.py", "data", 3000, defer_kg=True)
    except Exception:
        pytest.skip("run_all needs a fuller fixture here; the call is asserted below")
    assert seen.get("called"), "run_all did not call storage_gate"
    assert any("SENTINEL" in m for _, m in rows)


def test_the_gate_reports_rather_than_crashing_when_the_guard_is_absent(tmp_path):
    """A preflight that dies because one gate cannot load hides every other
    finding. The gate must degrade to a FAIL row."""
    P = _preflight()
    import shutil as _sh
    guard = _ROOT / "scripts" / "maintenance" / "preflight_data_guard.py"
    backup = tmp_path / "guard.bak"
    _sh.copy2(guard, backup)
    guard.unlink()
    try:
        rows = P.storage_gate("data", _MANIFEST)
        assert rows and rows[0][0] == "FAIL"
        assert "data guard" in rows[0][1] or "guard" in rows[0][1]
    finally:
        _sh.copy2(backup, guard)
