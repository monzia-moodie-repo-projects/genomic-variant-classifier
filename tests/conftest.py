"""Test bootstrap for tests/ (tests/conftest.py).

What belongs to tests/ alone: `scripts/` on sys.path (below), the suite-size ratchet, and the connector-cache isolation. The
session guards that apply to EVERY suite in this repository -- sys.path leaks, writes into data/raw/cache, the workflow's
publication files, the child-pytest environment and conftest.py, temporary-file containment, no thread-parallel joblib work with
the warning-filter guard, the session-start import of the libraries that register warning filters (since 2026-11-05), and the
working-tree guard -- are in the repository root's conftest.py since 2026-11-04 (DRIFT-SUITE-UNGUARDED-1), which pytest loads for
tests/ and tests_drift/ alike.

Makes the repo-local `scripts/` directory importable during tests regardless of CWD or
environment, so tests can `import clean_cohort` (and other script-level modules) the same way
locally and in CI.

Root cause this fixes: CI runs `pytest tests/unit/ -x -q` from the repo root, where `scripts/` is
not a package and not on sys.path; `from clean_cohort import run_clean` raised ModuleNotFoundError
on the GitHub runner (CI run #233) even though it imported in the local dev environment. Walking
ancestors (rather than hardcoding depth) keeps this correct if the tests tree is ever moved.
"""
from __future__ import annotations

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
# Connector-cache isolation (added 2026-07-11) -- PREVENTION.
#
# It does NOT request `monkeypatch`. Its first version did, when it shared a file with the guards: monkeypatch's finaliser then
# ran after the sys.path guard's, and that guard errored two legitimately-passing tests in tests/test_rekey_seq_windows_v2.py.
# Since 2026-11-04 the guards are in the root conftest.py and set up first, so this no longer depends on it (THE INVARIANT
# there records the measurement); the hand-written save and restore is kept.
# ---------------------------------------------------------------------------
# The guards in the root conftest.py are DETECTION. This is PREVENTION. Both are kept on purpose: this fixture
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
