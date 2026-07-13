"""Test bootstrap (tests/conftest.py).

Makes the repo-local `scripts/` directory importable during tests regardless of CWD or
environment, so tests can `import clean_cohort` (and other script-level modules) the same way
locally and in CI.

Root cause this fixes: CI runs `pytest tests/unit/ -x -q` from the repo root, where `scripts/` is
not a package and not on sys.path; `from clean_cohort import run_clean` raised ModuleNotFoundError
on the GitHub runner (CI run #233) even though it imported in the local dev environment. Walking
ancestors (rather than hardcoding depth) keeps this correct if the tests tree is ever moved.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_here = Path(__file__).resolve()
for _anc in (_here.parent, *_here.parents):
    _scripts = _anc / "scripts"
    if _scripts.is_dir() and (_scripts / "clean_cohort.py").is_file():
        for _p in (str(_scripts), str(_anc)):  # scripts dir + repo root
            if _p not in sys.path:
                sys.path.insert(0, _p)
        break


# ===========================================================================================
# THE SUITE-SIZE RATCHET (roadmap 6.14, built 2026-07-13)
# ===========================================================================================
# The G1 pre-flight pytest floor rotted FIVE TIMES IN TWO DAYS -- 1485 -> 1805 -> 1842 ->
# 1850 -> 1853 -- every time directly beneath an all-capitals comment ordering the next person
# to raise it, written by the person who then failed to raise it. At 1485 against a suite of
# 1,815, THREE HUNDRED AND THIRTY tests could have silently vanished and the gate would still
# have said PASS.
#
# A COMMENT DOES NOT ENFORCE ITSELF. No volume of emphasis will make it. So this replaces the
# comment with a gate: `tests/EXPECTED_SUITE_SIZE` holds ONE number, and under the explicit
# `--assert-suite-size` flag the suite ABORTS if the collected count disagrees with it -- in
# EITHER direction. Adding a test therefore turns the suite red until the number is bumped.
# Forgetting is no longer possible, because forgetting FAILS.
#
# This is the same fail-loud pattern the project already uses successfully for features:
# TABULAR_FEATURES is guarded by EXPECTED_TABULAR_FEATURE_COUNT, and adding a feature without
# bumping the count is a hard error. Tests now get the same protection.
#
# Why COLLECTED and not PASSED: the collected count is environment-independent; the
# passed/skipped split is not (Windows 1863p/7s vs Linux CI 1856p/13s/1xf -- both 1870
# collected). Asserting `passed` would need two numbers and would re-create the very
# divergence this exists to kill.
# ===========================================================================================

_SUITE_SIZE_FILE = _here.parent / "EXPECTED_SUITE_SIZE"


def _read_expected_suite_size() -> int:
    """Parse tests/EXPECTED_SUITE_SIZE -> int. A malformed ratchet is a DEAD ratchet."""
    if not _SUITE_SIZE_FILE.is_file():
        raise pytest.UsageError(
            f"--assert-suite-size was requested but {_SUITE_SIZE_FILE} does not exist.\n"
            f"This file is the single source of truth for the suite size (roadmap 6.14). "
            f"Without it the ratchet cannot run, and a missing guard must NEVER degrade to a "
            f"silent pass."
        )

    numbers = [
        line.strip()
        for line in _SUITE_SIZE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(numbers) != 1 or not numbers[0].isdigit() or int(numbers[0]) <= 0:
        raise pytest.UsageError(
            f"{_SUITE_SIZE_FILE} is MALFORMED. It must contain exactly one bare positive "
            f"integer (comments starting with '#' and blank lines are ignored).\n"
            f"Found these non-comment lines: {numbers!r}\n"
            f"A ratchet that cannot be parsed is a ratchet that does not guard."
        )
    return int(numbers[0])


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--assert-suite-size",
        action="store_true",
        default=False,
        help=(
            "Fail if the number of COLLECTED tests differs from tests/EXPECTED_SUITE_SIZE. "
            "Passed by the G1 pre-flight gate and by Continuous Integration. Off by default, "
            "so running a subset locally (e.g. `pytest tests/unit/test_foo.py`) is unaffected."
        ),
    )


def pytest_collection_modifyitems(session, config, items) -> None:
    """THE RATCHET. Collected != expected -> abort, in either direction."""
    if not config.getoption("--assert-suite-size"):
        return

    expected = _read_expected_suite_size()
    collected = len(items)
    if collected == expected:
        return

    delta = collected - expected
    if delta > 0:
        diagnosis = (
            f"{delta} MORE test(s) than expected.\n"
            f"  You ADDED tests and did not bump the ratchet. This is the intended failure:\n"
            f"  the number cannot go stale, because adding a test turns the suite RED until\n"
            f"  you update it. Set the value to {collected} IN THE SAME COMMIT as the tests."
        )
    else:
        diagnosis = (
            f"{-delta} FEWER test(s) than expected.\n"
            f"  *** TESTS HAVE VANISHED. *** They were deleted, renamed out of discovery, lost\n"
            f"  to a collection error, or silently skipped at module import (an importorskip on\n"
            f"  a MISSING DEPENDENCY collapses N tests into ONE skip entry -- that is exactly how\n"
            f"  the entire graph-neural-network branch went untested for 508 Continuous\n"
            f"  Integration runs; see roadmap 6.17).\n"
            f"  DO NOT 'fix' this by lowering the number. Find out what stopped running."
        )

    raise pytest.UsageError(
        f"\nSUITE-SIZE RATCHET FAILED (roadmap 6.14)\n"
        f"  expected (tests/EXPECTED_SUITE_SIZE): {expected}\n"
        f"  actually collected:                   {collected}\n"
        f"  {diagnosis}\n"
    )


# ---------------------------------------------------------------------------
# sys.path leak guard (added 2026-07-11 -- cluster E)
# ---------------------------------------------------------------------------
# A test must not permanently mutate sys.path. Global-state pollution of this kind
# is invisible in isolation and only detonates later, in an unrelated test.
#
# It has already cost us once. tests/test_rekey_seq_windows_v2.py wrote a COUNTERFEIT
# `genomic_variant_classifier` package (real __init__.py; only a `data` subpackage)
# into a tmp dir and published it with a bare
#
#     sys.path.insert(0, str(tmp_path / "gvc_join"))     # never removed
#
# Nothing broke in-process -- the real package was already in sys.modules and was
# never re-resolved. But five later tests launch a CHILD interpreter with
# PYTHONPATH built from sys.path. The child starts with an empty sys.modules, so it
# resolved the counterfeit first: the top-level import SUCCEEDED and every subpackage
# import failed --
#
#     ModuleNotFoundError: No module named 'genomic_variant_classifier.evaluation'
#     ModuleNotFoundError: No module named 'genomic_variant_classifier.agent_layer'
#
# Green in isolation (38 passed), red in the full suite. Five tests, unexplained for
# three days. See docs/status/REMEDIATION_2026-07-11_test-suite-red.md, cluster E.
#
# This fixture makes that failure mode IMPOSSIBLE TO INTRODUCE SILENTLY: any test that
# leaves sys.path modified now fails loudly, in the test that did it, naming the
# entries it added or removed. Tests that legitimately need a path change must use
# `monkeypatch.syspath_prepend`, which pytest reverts at teardown.
# Design notes, so this guard is neither too loose nor too brittle:
#
#  * It compares SETS, not lists. Several existing tests legitimately call
#    sys.path.insert() inside the test body with a path that tests/conftest.py has
#    ALREADY added (e.g. the scripts dir). That merely creates a duplicate entry --
#    harmless, and a strict list comparison would fail those tests for no reason.
#    Only a genuinely NEW entry, or a VANISHED one, can change what an import resolves
#    to. Those are the two conditions worth failing on.
#
#  * It RESTORES sys.path unconditionally, before raising. A leaking test must not be
#    able to cascade into the rest of the suite -- the offender fails, and only the
#    offender. Without this, the guard would report the leak and then let it corrupt
#    everything downstream anyway, which is the exact behaviour we are eliminating.
# ===========================================================================
# THE INVARIANT FOR THIS FILE (2026-07-12), and how it was learned the hard way.
# ===========================================================================
#   *** NO AUTOUSE FIXTURE HERE MAY REQUEST `monkeypatch`. ***
#
# WHY. Two fixtures below are GUARDS: they snapshot global state (sys.path; the data/raw/
# cache tree) at setup and inspect it at teardown. For a guard to be correct, it must run
# its check AFTER every other finaliser -- in particular after `monkeypatch` has undone what
# a test did with `monkeypatch.syspath_prepend` / `monkeypatch.setattr`.
#
# Fixtures tear down in REVERSE order of setup. If an AUTOUSE fixture requests `monkeypatch`,
# pytest hoists monkeypatch into the autouse group and sets it up BEFORE the guards -- so
# monkeypatch's finaliser runs AFTER theirs. The sys.path guard then inspects a sys.path that
# monkeypatch has not yet reverted, and errors a test that did nothing wrong. That is exactly
# what happened to the two tests in tests/test_rekey_seq_windows_v2.py, which legitimately
# publish a counterfeit package via `monkeypatch.syspath_prepend`.
#
# WHAT DID NOT WORK. Re-declaring the fixtures in a different order. pytest hoists an autouse
# fixture's DEPENDENCIES regardless of where that fixture is declared, so monkeypatch was
# still set up first and the guard still fired. Declaration order was never the lever, and a
# comment here previously claimed it was -- it was wrong, and this replaces it.
#
# WHAT WORKS. `_isolate_connector_caches` saves and restores the module attributes BY HAND
# and requests only `tmp_path`. With no autouse fixture depending on it, `monkeypatch` is
# instantiated only when a TEST asks for it -- i.e. after every autouse fixture -- so its
# finaliser runs FIRST, and the guards see fully-reverted global state.
#
# If you add an autouse fixture here and reach for `monkeypatch`, don't. Save and restore
# explicitly, or you will silently break the guards and blame the wrong test.
# ---------------------------------------------------------------------------
# data/ pollution guard (added 2026-07-11) -- DETECTION.
# ---------------------------------------------------------------------------
# A test must not write into the repository's data/ tree. Measured, on a fresh clone:
#
#   run 1: 1805 passed, 17 skipped
#   run 2: 1812 passed, 10 skipped     <- SAME checkout, SAME code
#
# Seven tests in test_alphafold.py skipped on run 1 ("real cached CIF not present") and
# PASSED on run 2, because run 1 DOWNLOADED AF-E7ENB7-F1-model_v4.cif from
# https://alphafold.ebi.ac.uk and wrote it into the checkout (ProteinStructurePipeline
# defaults cache_dir to a CWD-relative "data/raw/cache/alphafold"). Two ESM-2 tests were
# likewise writing esm2_cache.sqlite + esm2_scores.parquet into data/raw/cache.
#
# A suite whose result depends on whether it has been run before is not a suite. And none
# of it showed up in `git status`, because data/raw/ is gitignored -- the tool that would
# have caught it was blindfolded. cf. INCIDENT_2026-06-14_data-junction-dangling, which
# already recorded that tests "write to the REAL data/" and was never acted on.
#
# This guard makes the pollution LOUD and attributes it to the test that did it. Tests
# needing a cache must pass tmp_path; tests needing a real artifact must use a committed
# fixture under tests/fixtures/ (see tests/unit/test_alphafold.py).
#
# Scope: data/raw/cache only -- the directory the suite was actually polluting. Widen it
# if new offenders appear; do NOT widen it blindly, since some tracked data/ files are
# legitimately read (data/external, data/processed, data/reference are all committed).
_DATA_CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "cache"

# COST, and how the first version of this guard got it catastrophically wrong.
#
# v1 called _DATA_CACHE.rglob("*") and stat()ed every entry, TWICE PER TEST. That
# directory holds 36,202 files (36,074 of them AlphaFold structures -- the 8.77 GB cache
# recorded in docs/STORAGE_ACTION_LEDGER_2026-07-03.md). The arithmetic:
#
#     1,822 tests x 2 snapshots x 36,202 files = ~132,000,000 stat() calls
#
# The suite went from 7 minutes to 6 hours 45 minutes. A guard that costs 58x the runtime
# it protects will simply be deleted by the next person, and then the defect returns.
#
# v2 is O(1) per test. It scans ONLY the immediate entries of data/raw/cache (~a handful),
# never recursing, and relies on a filesystem invariant:
#
#     * a DIRECTORY's mtime changes when an entry is CREATED or REMOVED inside it,
#     * a FILE's mtime and size change when it is REWRITTEN.
#
# So a new CIF inside data/raw/cache/alphafold/ is caught by that directory's mtime,
# without ever listing its 36,074 entries. A rewrite of esm2_scores.parquet is caught by
# that file's mtime+size.
#
# Known limit, stated rather than hidden: modifying an EXISTING file nested inside a
# subdirectory (rather than creating one) may not move the parent directory's mtime and
# can escape this guard. Creation -- the behaviour that actually broke idempotency -- is
# always caught. Widen it only with the cost arithmetic above in front of you.
def _fingerprint() -> frozenset:
    if not _DATA_CACHE.is_dir():
        return frozenset()
    out = []
    try:
        with os.scandir(_DATA_CACHE) as it:
            for e in it:
                try:
                    st = e.stat()
                except OSError:
                    continue
                is_dir = e.is_dir()
                out.append((e.name, is_dir, st.st_mtime_ns, 0 if is_dir else st.st_size))
    except OSError:
        return frozenset()
    return frozenset(out)


@pytest.fixture(autouse=True)
def _no_data_dir_writes():
    before = _fingerprint()
    yield
    after = _fingerprint()
    if after == before:
        return

    # Only NOW -- on an actual violation -- do the expensive work of naming the culprit.
    changed = sorted(
        {n for n, *_ in (after - before)} | {n for n, *_ in (before - after)}
    )
    raise AssertionError(
        "This test WROTE into the repository's data/raw/cache tree. That makes the suite "
        "non-hermetic and NON-IDEMPOTENT: a later run sees the artifact and behaves "
        "differently. Measured on a clean clone before this guard existed --\n"
        "    run 1: 1805 passed, 17 skipped\n"
        "    run 2: 1812 passed, 10 skipped   (SAME checkout, SAME code)\n"
        "because run 1 downloaded an AlphaFold structure into the checkout and run 2's "
        "collection-time skipif then found it.\n"
        "None of this appears in `git status` -- data/raw/ is gitignored, so the tool that "
        "would flag it is blindfolded.\n"
        f"  ENTRIES CHANGED under data/raw/cache: {changed}\n"
        "Pass an explicit cache path under tmp_path, or use a committed fixture under "
        "tests/fixtures/ (see tests/unit/test_alphafold.py). "
        "docs/status/REMEDIATION_2026-07-11_test-suite-red.md has the full write-up."
    )


@pytest.fixture(autouse=True)
def _no_sys_path_leaks():
    before = list(sys.path)
    try:
        yield
    finally:
        after = list(sys.path)
        added = [p for p in dict.fromkeys(after) if p not in set(before)]
        removed = [p for p in dict.fromkeys(before) if p not in set(after)]
        # Restore FIRST -- containment before reporting.
        sys.path[:] = before
        if added or removed:
            raise AssertionError(
                "This test leaked a permanent change to sys.path. That corrupts every "
                "later test, and every subprocess they launch with a PYTHONPATH built "
                "from sys.path (the child starts with an empty sys.modules and will "
                "resolve the leaked path FIRST).\n"
                f"  ADDED  : {added}\n"
                f"  REMOVED: {removed}\n"
                "Use `monkeypatch.syspath_prepend(...)` rather than `sys.path.insert(...)`"
                " -- pytest reverts it at teardown. Precedent: "
                "tests/test_rekey_seq_windows_v2.py. Full write-up: "
                "docs/status/REMEDIATION_2026-07-11_test-suite-red.md (cluster E)."
            )


# ---------------------------------------------------------------------------
# Connector-cache isolation (added 2026-07-11) -- PREVENTION.
#
# DECLARED LAST, DELIBERATELY. It requests `monkeypatch`, which pulls monkeypatch's setup
# earlier and therefore its TEARDOWN later. Declaring this fixture before the two guards
# above makes monkeypatch's finalizer run AFTER theirs, so the sys.path guard sees paths
# that monkeypatch was about to revert -- and it errored two legitimately-passing tests in
# tests/test_rekey_seq_windows_v2.py. The guard was right; the ORDER was wrong. Keep it here.
# ---------------------------------------------------------------------------
# The guards above are DETECTION. This is PREVENTION. Both are kept on purpose: this fixture
# redirects the writable defaults we KNOW about; the guards catch any NEW writable path a
# future connector introduces, which this fixture cannot know about.
#
# Several connectors fall back to a CURRENT-WORKING-DIRECTORY-RELATIVE cache under
# data/raw/cache when the caller passes nothing. Any test exercising them with a default
# config therefore wrote into the repository -- and one made a LIVE NETWORK CALL to
# https://alphafold.ebi.ac.uk, downloading a structure straight into the checkout.
#
# Measured on a COLD clone, before this existed:
#     data/raw/cache/esm2_cache.sqlite                      126,976 bytes
#     data/raw/cache/esm2_scores.parquet                      4,409
#     data/raw/cache/alphafold/AF-E7ENB7-F1-model_v4.cif    101,171   <- downloaded
#     data/raw/cache/alphafold/gene_uniprot_map.json             25
#     data/raw/cache/alphafold/uniprot_features_E7ENB7.json      26
#
# and the suite was NON-IDEMPOTENT as a result:
#     run 1: 1805 passed, 17 skipped
#     run 2: 1812 passed, 10 skipped     (SAME checkout, SAME code)
# because run 1 downloaded the structure and run 2's collection-time skipif then found it.
#
# WHY NOBODY SAW IT. On a developer machine those caches are WARM, so nothing is written and
# nothing shows. Only a COLD cache -- a fresh Continuous Integration runner, or a fresh clone
# -- exposes it. And `git status` is blind: data/raw/ is gitignored.
# docs/incidents/INCIDENT_2026-06-14_data-junction-dangling.md already recorded that
# test_lovd_annotation_reaches_training_matrix.py "writes to the REAL data/". Nothing was
# done, because nothing ever FAILED. A finding in a document is a comment; a finding that
# fails a test is a gate.
#
# The production injection points all exist and are honoured -- AnnotationConfig
# .esm2_cache_path, .protein_cache_dir, .genomiclm_cache_path. This redirects only the
# DEFAULTS the connectors fall back to when a test supplies none, so no test signature and
# no production code path changes.
@pytest.fixture(autouse=True)
def _isolate_connector_caches(tmp_path):
    """Redirect the connectors' writable DEFAULT caches into tmp_path.

    DELIBERATELY DOES NOT REQUEST `monkeypatch`, and that is the whole point.

    The first version did (`monkeypatch.setattr(...)`), and it broke two passing tests in
    tests/test_rekey_seq_windows_v2.py. An AUTOUSE fixture that requests `monkeypatch` drags
    monkeypatch into the autouse group, so monkeypatch is SET UP BEFORE the guard fixtures
    -- and therefore TORN DOWN AFTER them. `_no_sys_path_leaks` then ran its check while the
    counterfeit package that test legitimately publishes via `monkeypatch.syspath_prepend`
    was STILL on sys.path, and errored a test that had done nothing wrong.

    I first tried to fix that by re-declaring the fixtures in a different order. It did not
    work: pytest hoists an autouse fixture's dependencies regardless of where the fixture is
    declared, so monkeypatch was still set up first. Declaration order was never the lever.

    Doing the save/restore by hand removes the dependency entirely. `monkeypatch` is then
    instantiated only when a TEST asks for it -- i.e. after every autouse fixture -- so its
    finalizer runs FIRST, before the guards inspect global state. The ordering hazard is
    gone at its source rather than worked around.
    """
    import genomic_variant_classifier.data.esm2 as _esm2
    from genomic_variant_classifier.pipelines import protein_pipeline as _pp

    _prev_esm2 = getattr(_esm2, "_DEFAULT_CACHE", None)
    _prev_pp = getattr(_pp, "_DEFAULT_CACHE_DIR", None)

    _esm2._DEFAULT_CACHE = tmp_path / "esm2_cache.sqlite"
    _pp._DEFAULT_CACHE_DIR = tmp_path / "alphafold_cache"
    try:
        yield
    finally:
        # Restore unconditionally, even if the test raised.
        if _prev_esm2 is not None:
            _esm2._DEFAULT_CACHE = _prev_esm2
        if _prev_pp is not None:
            _pp._DEFAULT_CACHE_DIR = _prev_pp
