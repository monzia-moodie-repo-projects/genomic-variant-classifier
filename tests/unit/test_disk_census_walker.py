"""Tests for the disk-census walker in scripts/forensics/audit_disk_census.py.

WHY THIS FILE EXISTS
====================
The census tool has now reported a materially wrong number for the repository's
data/ directory three times running:

    v1  2026-07-20   161.38 GiB / 15,260 files
    v2  2026-07-20   every subdirectory 0.0 MiB -- the time budget had expired
                     and zeros were printed as if they were measurements
    v3  2026-07-21     3.21 GiB -- against a true 98.75 GiB measured the same
                     hour by an independent walk; short by 95.54 GiB

Each fix was correct as far as it went, and each shipped without a test. A tool
whose entire job is measurement, and which has been wrong three times, has to be
held to the same standard as the code it measures.

THE DEFECT THESE TESTS PIN (defect 5)
--------------------------------------
Walker maintains visited sets keyed on (st_dev, st_ino) so that a CENSUS -- where
the sum over many roots must equal one volume -- never counts an overlapping root
twice. That is correct and necessary: without it, the legacy junction
C:\\Documents and Settings makes the census report 5.94x the volume size.

data_breakdown() then reused the same Walker that had already walked the whole
volume. Every subdirectory of data/ was in _seen_dirs, and size_of() skips any
child already there -- while never checking the root it is handed. So each
subtotal counted ONLY the files sitting loose in that directory:

    directory            true      in subdirs   loose only   v3 reported
    data/external       75.18 GiB   75.18 GiB     0.00 GiB     0.003 GiB
    data/processed       3.50 GiB    0.54 GiB     2.96 GiB     2.950 GiB
    data/raw            19.80 GiB   19.80 GiB     0.00 GiB     0.000 GiB
    data/_drift_check    0.26 GiB    0.00 GiB     0.26 GiB     0.264 GiB

Four for four. The bug was not the de-duplication; it was applying CENSUS
semantics to a STANDALONE MEASUREMENT.

WHAT WOULD NOT HAVE CAUGHT IT
------------------------------
Asserting that the census total matches the volume: it did. Asserting the walker
terminates: it did. Asserting no exceptions: there were none. Only measuring a
directory whose true size is known, AFTER walking its parent, exposes it -- which
is why every test below that matters does exactly that.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_FORENSICS = Path(__file__).resolve().parents[2] / "scripts" / "forensics"


def _load():
    spec = importlib.util.spec_from_file_location(
        "audit_disk_census", _FORENSICS / "audit_disk_census.py")
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["audit_disk_census"]          # module must not parse pytest's argv
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


C = _load()


@pytest.fixture
def tree(tmp_path):
    """A tree whose every subtotal is unambiguous.

        data/external/eve/        5 files x   1,000 =  5,000
        data/external/finngen/    3 files x  20,000 = 60,000
        data/external/*.bin       2 files x       7 =     14   <- loose
        data/raw/cache/           4 files x   2,500 = 10,000
        ------------------------------------------------------
        data/external total                          65,014
        data/raw total                               10,000
        whole tree                                   75,014
    """
    for d in ("data/external/eve", "data/external/finngen", "data/raw/cache"):
        (tmp_path / d).mkdir(parents=True)
    for i in range(5):
        (tmp_path / f"data/external/eve/e{i}.bin").write_bytes(b"x" * 1000)
    for i in range(3):
        (tmp_path / f"data/external/finngen/f{i}.bin").write_bytes(b"x" * 20000)
    for i in range(2):
        (tmp_path / f"data/external/loose{i}.bin").write_bytes(b"x" * 7)
    for i in range(4):
        (tmp_path / f"data/raw/cache/c{i}.bin").write_bytes(b"x" * 2500)
    return tmp_path


# --------------------------------------------------------------------------- #
# 1. the defect itself
# --------------------------------------------------------------------------- #
def test_shared_mode_after_a_parent_walk_reports_only_loose_files(tree):
    """THE DEFECT, PINNED AS A CHARACTERISATION TEST.

    This is not a bug being asserted as correct -- it documents exactly what
    census semantics do when misapplied, so the distinction between the two
    modes cannot quietly collapse back into one.
    """
    w = C.Walker()
    whole, _, _ = w.size_of(tree)
    assert whole == 75_014

    ext, ext_n, _ = w.size_of(tree / "data/external")
    raw, raw_n, _ = w.size_of(tree / "data/raw")
    assert ext == 14 and ext_n == 2, "expected loose files only under census semantics"
    assert raw == 0 and raw_n == 0, "raw/ holds nothing loose; census semantics give zero"


def test_independent_mode_measures_the_true_size_after_a_parent_walk(tree):
    """THE FIX. Same sequence, one keyword, correct answers."""
    w = C.Walker()
    assert w.size_of(tree)[0] == 75_014

    ext, ext_n, ok_e = w.size_of(tree / "data/external", independent=True)
    raw, raw_n, ok_r = w.size_of(tree / "data/raw", independent=True)
    assert ok_e and ok_r
    assert ext == 65_014, f"data/external measured {ext}, expected 65,014"
    assert ext_n == 10
    assert raw == 10_000, f"data/raw measured {raw}, expected 10,000"
    assert raw_n == 4


def test_independent_mode_is_unaffected_by_walk_order(tree):
    """A measurement that depends on what was walked earlier is not a measurement."""
    fresh = C.Walker().size_of(tree / "data/external", independent=True)[0]
    used = C.Walker()
    used.size_of(tree)
    after = used.size_of(tree / "data/external", independent=True)[0]
    assert fresh == after == 65_014


# --------------------------------------------------------------------------- #
# 2. census semantics must survive the fix
# --------------------------------------------------------------------------- #
def test_census_mode_still_refuses_to_double_count_an_overlapping_root(tree):
    """Without shared de-duplication the census reports multiples of the volume;
    the 2026-07-20 run reported 5.94x. The fix must not weaken that."""
    w = C.Walker()
    first = w.size_of(tree / "data")[0]
    second = w.size_of(tree / "data/external")[0]
    assert first == 75_014
    assert second == 14, "the subtree was already counted; only loose files remain"
    assert first + second < 2 * 75_014


def test_independent_mode_does_not_pollute_the_shared_visited_set(tree):
    """An independent measurement must leave the census exactly as it found it,
    or the volume total silently changes depending on what was inspected."""
    w = C.Walker()
    w.size_of(tree / "data/external")          # census: marks eve/ and finngen/
    seen_before = set(w._seen_dirs)
    w.size_of(tree / "data/raw", independent=True)
    assert w._seen_dirs == seen_before, "independent mode mutated the census state"


def test_independent_mode_does_not_mutate_hardlink_savings(tree):
    """hardlink_savings describes the census. An independent measurement that
    re-walks an already-counted subtree must not inflate it."""
    w = C.Walker()
    w.size_of(tree)
    before = w.hardlink_savings
    w.size_of(tree / "data/external", independent=True)
    assert w.hardlink_savings == before


# --------------------------------------------------------------------------- #
# 3. cycle safety holds in BOTH modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("independent", [False, True])
def test_a_self_referencing_link_still_terminates(tmp_path, independent):
    """The whole reason the visited set exists. Scoping it per call must not
    reintroduce the infinite recursion that made v2 report 5,593.90 GiB on a
    935.59 GiB volume."""
    root = tmp_path / "top"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "f.bin").write_bytes(b"x" * 100)
    try:
        os.symlink(root, root / "sub" / "loop", target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlink creation unavailable on this platform/privilege level")
    size, count, ok = C.Walker().size_of(root, independent=independent)
    assert ok
    assert size == 100 and count == 1


# --------------------------------------------------------------------------- #
# 4. the call site -- the actual bug was here, not in the walker
# --------------------------------------------------------------------------- #
def test_data_breakdown_measures_independently(tmp_path, monkeypatch, capsys):
    """data_breakdown() previously shared the census walker. It must not.

    This drives the real function rather than inspecting its source, so a
    refactor that preserves the behaviour keeps passing and one that loses it
    fails -- the distinction that mattered in scripts/forensics/verify_dtype.py.
    """
    for d in ("data/external/eve", "data/raw/cache"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "data/external/eve/e.bin").write_bytes(b"x" * 50_000)
    (tmp_path / "data/raw/cache/c.bin").write_bytes(b"x" * 30_000)

    monkeypatch.setattr(C, "REPO", str(tmp_path))
    w = C.Walker()
    w.size_of(tmp_path)                        # consume the tree, as main() does
    rows = C.data_breakdown(w, 10)
    capsys.readouterr()

    by = {Path(r["path"]).name: r["size"] for r in rows}
    assert by.get("external") == 50_000, (
        f"data/external measured {by.get('external')}, expected 50,000. "
        "data_breakdown is sharing the census walker again (defect 5).")
    assert by.get("raw") == 30_000


def test_data_breakdown_subtotal_matches_a_direct_measurement(tmp_path, monkeypatch, capsys):
    """The single assertion that would have caught the 95.54 GiB shortfall."""
    (tmp_path / "data/external/finngen").mkdir(parents=True)
    (tmp_path / "data/external/finngen/big.bin").write_bytes(b"x" * 90_000)
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/p.bin").write_bytes(b"x" * 10_000)

    monkeypatch.setattr(C, "REPO", str(tmp_path))
    w = C.Walker()
    w.size_of(tmp_path)
    rows = C.data_breakdown(w, 10)
    capsys.readouterr()

    subtotal = sum(r["size"] for r in rows if r["size"])
    direct = C.Walker().size_of(tmp_path / "data")[0]
    assert subtotal == direct == 100_000


def test_the_report_no_longer_invites_a_false_reclaim_conclusion(tmp_path, monkeypatch, capsys):
    """v3 printed 'Compare the figure above against 161.38 GiB' beneath its own
    wrong number, inviting the reader to conclude 158 GiB had been reclaimed.
    Nothing had. The section must now carry its own error history instead."""
    (tmp_path / "data/external").mkdir(parents=True)
    (tmp_path / "data/external/f.bin").write_bytes(b"x" * 1000)
    monkeypatch.setattr(C, "REPO", str(tmp_path))
    C.data_breakdown(C.Walker(), 10)
    out = capsys.readouterr().out
    assert "Compare the figure above against 161.38 GiB" not in out
    assert "wrong three times" in out
    assert "98.75" in out


# --------------------------------------------------------------------------- #
# 5. the deadline contract (defect 4) must not have regressed
# --------------------------------------------------------------------------- #
def test_an_expired_deadline_reports_incomplete_rather_than_zero(tree):
    """v2 printed zeros as measurements once its budget expired. `complete`
    exists so a caller can tell a non-measurement from a small one."""
    w = C.Walker(deadline=0.0)                 # already past
    size, count, complete = w.size_of(tree)
    assert complete is False
    assert w.timed_out is True


def test_a_generous_deadline_completes(tree):
    import time
    w = C.Walker(deadline=time.monotonic() + 60)
    size, _, complete = w.size_of(tree)
    assert complete is True and size == 75_014
