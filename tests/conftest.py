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


# ---------------------------------------------------------------------------
# GitHub workflow-command isolation (added 2026-09-27) -- PREVENTION.
# ---------------------------------------------------------------------------
# MEASURED on CI run #896 (the merge of #28): the pytest (3.11) and (3.12) job summaries carried
# "Source-monitor run verification (PREVIEW)" reports -- one saying run 36300779115 was VERIFIED and
# one saying the SAME run was NOT VERIFIED ("archive digest ... differs from GitHub's"). Neither
# verification happened: tests/unit/test_report_verifier.py called scripts/verify_monitor_run.py's
# main(), which appended to $GITHUB_STEP_SUMMARY -- and inside CI that variable is SET. A tampered-
# archive TEST published a fabricated verdict about a real run onto CI's own summary page.
#
# These five variables name the files through which a process PUBLISHES to its workflow run (job
# summary, step outputs, environment and PATH for later steps, action state). No test may publish.
# They are removed before every test and restored after it; a subprocess a test launches inherits
# os.environ, so it cannot publish either. Same discipline as _isolate_connector_caches below:
# save and restore BY HAND, never `monkeypatch` (see THE INVARIANT FOR THIS FILE).
#
# THE ONE SOURCE of this set (2026-09-29): the isolation tests parse it from this file and require ci.yml's `env -u`
# flags to equal it. GITHUB_ARTIFACTS added 2026-09-29: the runner-variable census in CI run #899 (names only) showed it,
# GitHub's variables reference documents it as the file that DECLARES workflow artifacts for the current step (one
# declaration per line), and actions/runner v2.336.0 added it. GITHUB_ARTIFACTS_LIST is a READ-ONLY metadata file --
# not a publication channel -- and is deliberately NOT removed.
_GITHUB_WORKFLOW_COMMAND_FILES = ("GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_PATH", "GITHUB_STATE",
                                  "GITHUB_ARTIFACTS")

# LAYER B (added 2026-09-29, owner ruling 2026-09-28). MEASURED: with only the per-test fixture below, a test MODULE that
# publishes at IMPORT time (during collection) still wrote -- and pytest exited 0. This conftest is imported before any
# test module is collected, so the variables are removed HERE for the whole session. LAYER C (ci.yml: `env -u ...`)
# removes them before the interpreter even starts, covering plugins that load before this file; LAYER A (the fixture
# below) still restores around every test.
for _name in _GITHUB_WORKFLOW_COMMAND_FILES:
    os.environ.pop(_name, None)


@pytest.fixture(autouse=True)
def _isolate_github_workflow_commands():
    """No test may write to the running workflow's summary, outputs, environment, PATH or state."""
    saved = {name: os.environ.pop(name, None) for name in _GITHUB_WORKFLOW_COMMAND_FILES}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------------------
# The environment of a CHILD pytest that a test starts (added 2026-11-03) -- the one source.
# ---------------------------------------------------------------------------
# Seven calls in six test files run a child pytest and read its console (measured 2026-11-03 by parsing tests/:
# test_working_tree_guard.py, test_warning_filters_are_scoped.py, test_sklearn_parallel_warning_contract.py (two),
# test_replicate_engine.py, test_review_status_tier.py, test_github_workflow_command_isolation.py); every one builds its
# environment here. What a pytest prints depends on variables it
# inherits, so a child built from os.environ as it stands passes or fails with the host. MEASURED 2026-11-03 (the second, an
# independent review): with CI=true -- which GitHub Actions sets -- pytest stops truncating its short summary (_pytest/compat.py
# running_on_ci reads CI and BUILD_NUMBER), and a count of messages doubled; with PY_COLORS=1 three child tests failed on
# colour codes. Dropped as well: the width (COLUMNS, LINES), the other colour switches (FORCE_COLOR, NO_COLOR), and
# what adds options, plugins or filters (PYTEST_ADDOPTS, PYTEST_PLUGINS, PYTHONWARNINGS). Everything else is inherited.
CHILD_PYTEST_ENVIRONMENT_DROPPED = ("CI", "BUILD_NUMBER", "COLUMNS", "LINES", "PY_COLORS", "FORCE_COLOR", "NO_COLOR",
                                    "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONWARNINGS")


@pytest.fixture
def child_pytest_environment():
    """A function returning the environment for a child pytest: os.environ without the variables above, plus the given ones."""
    def build(**extra: str) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in CHILD_PYTEST_ENVIRONMENT_DROPPED}
        env.update(extra)
        return env
    return build


# ---------------------------------------------------------------------------
# Temporary-file containment (added 2026-10-12) -- PREVENTION.
# ---------------------------------------------------------------------------
# MEASURED 2026-10-11 on the owner's machine (census of %TEMP%): 973 folders named tmp*, created by the test runs of 2026-10-09
# and 2026-10-10 and never removed -- 72 `tempfile.mkdtemp()` calls in nine test files, plus library code that calls
# mkdtemp/mkstemp itself. Each run left more; pytest's own retention never saw them, because they were not under its root.
#
# This points `tempfile` -- and TMPDIR/TEMP/TMP, which every child process inherits (git, a child pytest, a child interpreter)
# -- at a folder INSIDE pytest's base temporary directory, from configuration to the end of the run. Every mkdtemp, mkstemp,
# NamedTemporaryFile and TemporaryDirectory then lands where pytest manages the lifetime (an explicit --basetemp is emptied by
# the next run; the default root keeps the last three runs). The call sites are contained by construction, including ones not
# yet written. tests/unit/test_working_tree_guard.py proves it inside a test, from a child interpreter, and at import time.
#
# WHY A HOOK AND NOT A FIXTURE. The first version was a session fixture. MEASURED: a full run still left torchinductor_root in
# the outer temporary folder -- created while modules were imported during COLLECTION, before any fixture exists (a
# --collect-only run creates it too). pytest_configure runs before collection; trylast puts it after pytest's own tmpdir plugin
# has made the factory that the tmp_path_factory fixture itself returns (config._tmp_path_factory). If that factory is ever
# absent, the run stops here with that said, rather than running uncontained.
#
# Saved and restored BY HAND, never `monkeypatch` (THE INVARIANT FOR THIS FILE), and restored FIRST at unconfigure. A test that
# changes any of these with monkeypatch gets this value back afterwards.
_TEMPORARY_FILE_VARIABLES = ("TMPDIR", "TEMP", "TMP")
_CONTAINMENT_SAVED = pytest.StashKey()


@pytest.hookimpl(trylast=True)
def pytest_configure(config) -> None:
    import tempfile

    factory = getattr(config, "_tmp_path_factory", None)
    if factory is None:
        raise pytest.UsageError(
            "tests/conftest.py cannot contain temporary files: pytest's temporary-path factory (config._tmp_path_factory, the "
            "object the tmp_path_factory fixture returns) does not exist in this pytest. Without it every mkdtemp() in the suite "
            "would land in the system temporary folder and stay there.")
    root = factory.getbasetemp() / "tempfile"
    root.mkdir()
    config.stash[_CONTAINMENT_SAVED] = ({name: os.environ.get(name) for name in _TEMPORARY_FILE_VARIABLES}, tempfile.tempdir)
    for name in _TEMPORARY_FILE_VARIABLES:
        os.environ[name] = str(root)
    tempfile.tempdir = str(root)
    _run_joblib_sequentially(config)


# ---------------------------------------------------------------------------
# No thread-parallel scikit-learn in the session, and a guard on the warning filters (added 2026-11-03,
# SKLEARN-WARNING-FILTER-RACE-1) -- PREVENTION and DETECTION.
# ---------------------------------------------------------------------------
# THE DEFECT. On Python 3.12 the warning filters are ONE process-wide list, and `warnings.catch_warnings` saves it on entry and
# rebinds the saved list on exit. scikit-learn 1.8.0 runs every task of a thread-parallel estimator (RandomForestClassifier with
# n_jobs != 1) inside `catch_warnings(); resetwarnings()` and refills the list one entry at a time (sklearn/utils/parallel.py,
# read in full). Concurrent tasks enter and exit out of order, so the process is left bound to some worker's REBUILT copy, and an
# unlucky interleaving empties a list outright. Upstream issue scikit-learn #35089 (opened 2026-10-06) is open; its reporter
# finds every build with the global interpreter lock affected, 3.14 included, and Python's concurrency-safe warnings
# (context_aware_warnings) stay off by default on those builds through 3.15.
#
# MEASURED, 2026-11-03:
#   * inside pytest, in the sandbox (2 processors): 20 of 20 RandomForest fits (500 trees, n_jobs=-1) left the process bound to
#     a different list, and each changed an entry's meaning -- Python's default::DeprecationWarning:__main__ names its module as
#     a plain string (matched by equality) and comes back as a regular expression (matched by prefix);
#   * on the owner's machine (8 processors): 448 copies of scikit-learn's "`sklearn.utils.parallel.delayed` should be used with
#     ..." warning reached pytest from tests/unit/test_isolation_prerequisite.py (2026-11-03 06:18 UTC), and 1,000 from
#     tests/unit/test_containment_serving.py on pull request #58's validation (2026-11-01) -- through pyproject.toml's ignore
#     entry for that message. While a list is empty EVERY entry is gone, the `error::` entries included: a test that must fail
#     on an unclosed file or an undefined ranking metric would pass instead.
# A probe outside pytest (2026-11-03) counted 0 changes in 700 fits on that machine; it compared contents only, after
# resetwarnings() had removed the one entry that comes back changed, so it could not see what the sandbox measured inside pytest.
#
# PREVENTION. joblib's THREAD backend is its sequential backend for the whole session: `joblib.register_parallel_backend`
# (public) maps the name "threading" to joblib's SequentialBackend here at configuration -- before collection, as the
# temporary-file containment above is -- and the original is registered again at unconfigure. Every joblib thread request that
# NAMES its backend goes through that one name (joblib 1.5.3, joblib/parallel.py, read): an explicit backend="threading"
# (scikit-learn 1.8.0: ensemble/_hist_gradient_boosting/binning.py:240 and metrics/pairwise.py:1979), the hints prefer="threads"
# and require="sharedmem" (thirteen sites -- eleven literal, two passed through a variable: ensemble/_iforest.py:296 and
# linear_model/_logistic.py:1948 -- resolved through DEFAULT_THREAD_BACKEND = "threading"), and parallel_config(backend=
# "threading"). Not covered, by construction: a backend INSTANCE (Parallel(backend=ThreadingBackend()), or one handed to
# parallel_config), which bypasses the registry -- measured 2026-11-03: none in the package, the tests, the scripts or
# scikit-learn 1.8.0 outside its own tests -- and a thread pool that is not joblib's. The guard below catches both by their
# effect. joblib's PROCESS backends are untouched: a worker process has its own filters, and scalable_svm's bags keep running as
# they do in production.
# MEASURED 2026-11-03, a 16-tree forest at n_jobs=-1, a histogram gradient-boosting fit and pairwise_distances(n_jobs=2): with
# joblib's registry as shipped, all 40 of scikit-learn's tasks ran in worker threads and the filters were rebound and changed;
# with "threading" mapped to the sequential backend, all 40 ran in the main thread -- also under parallel_config(backend=
# "threading") -- and the filters were bound and unchanged. The FIRST version of this change set parallel_config(backend=
# "sequential") instead; the guard below then failed a test in tests/unit/test_containment_propagation.py, because an explicit
# backend="threading" overrides that setting (the histogram binning's), and the setting also made the process backends
# sequential, which nothing called for. Results: a forest's trees are seeded before they are dispatched -- measured with 500 trees
# on 3,000 rows (2 processors), identical trees and probabilities bitwise equal to the threaded run's. (Threads add the trees'
# probabilities in the order they finish, so on more processors a threaded run's last bits may vary; the sequential order is
# fixed.) Production runs keep their threads; this governs the suite.
#
# DETECTION. The warning state is taken before a test's setup -- before every fixture, module- and session-scoped ones included --
# and compared after its teardown. The test in which a change is found FAILS at teardown, naming it, when the list object pytest
# bound for it was replaced, when any entry it started with is gone (an entry changed in meaning is one gone and one new), when
# those entries are no longer in their original order (order is precedence: the first matching entry decides), when
# warnings.showwarning or the hook pytest records through (warnings._showwarnmsg_impl) was replaced, or when joblib's
# "threading" name no longer maps to the sequential backend. The state is restored first, so later tests start whole. Each of
# these is how the race shows itself; none is how a correct test behaves: a filter set inside `with warnings.catch_warnings():`
# in a test or a function-scoped fixture, pytest.warns and the recwarn fixture all restore before teardown ends. A
# catch_warnings opened in a MODULE- or SESSION-scoped fixture is found too, in the test it was opened in and in the test whose
# teardown closes it: pytest binds a fresh list for every test, so such a fixture's filters never governed the tests between.
# THE SESSION FAILS as well whenever any test was found changing the state (exit 1, its node named in this guard's section of the
# terminal summary): MEASURED (independent review, 2026-11-03), pytest reports any exception in the teardown of an xfail-marked
# test as "xfailed", which would otherwise hide the finding. An interrupt during teardown (KeyboardInterrupt, SystemExit,
# pytest.exit) is never replaced: the state is restored and recorded, and the interrupt propagates.
# ENTRIES ADDED during a test are not a failure, and not silent either. MEASURED 2026-11-03: libraries register filters when
# first imported -- torch three, scipy and numpy one each, during the setup of the first test that imported them (this file's
# connector-cache fixture imports the embedding module) -- and pytest discards them when that test ends. An added entry can
# still shadow one of pyproject.toml's for the rest of that test, so every distinct added entry is listed, with the first test
# that added it, in this guard's section of the terminal summary.
# Hook wrappers, not fixtures: no fixture runs before a module-scoped fixture is set up (THE INVARIANT FOR THIS FILE holds --
# nothing here requests monkeypatch). tests/unit/test_sklearn_parallel_warning_contract.py proves both, in this session and in
# child sessions that run a byte copy of this file.
_JOBLIB_THREAD_BACKEND = pytest.StashKey()
_WARNING_STATE_BEFORE_SETUP = pytest.StashKey()
_WARNING_GUARD_TALLY = pytest.StashKey()


def _joblib_thread_name_is_sequential() -> bool:
    from joblib.parallel import BACKENDS

    return BACKENDS["threading"] is BACKENDS["sequential"]


def _run_joblib_sequentially(config) -> None:
    import joblib
    from joblib.parallel import BACKENDS, get_active_backend

    config.stash[_JOBLIB_THREAD_BACKEND] = BACKENDS["threading"]
    config.stash[_WARNING_GUARD_TALLY] = {"checked": 0, "failed": [], "added": {}, "printed": False}
    joblib.register_parallel_backend("threading", BACKENDS["sequential"])
    resolved = [type(get_active_backend(prefer="threads", require="sharedmem")[0]), type(get_active_backend(prefer="threads")[0]),
                type(get_active_backend(require="sharedmem")[0])]
    if not _joblib_thread_name_is_sequential() or any(b is not BACKENDS["sequential"] for b in resolved):
        raise pytest.UsageError(
            "tests/conftest.py mapped joblib's 'threading' backend to its sequential one, and joblib does not resolve thread "
            "requests to it (resolved: {}). Thread-parallel scikit-learn would rebind the warning filters "
            "(SKLEARN-WARNING-FILTER-RACE-1); the suite does not run without this.".format([b.__name__ for b in resolved]))


def _warning_state() -> tuple:
    import warnings

    return warnings.filters, tuple(warnings.filters), warnings.showwarning, warnings._showwarnmsg_impl


def _describe_filter(entry) -> str:
    action, message, category, module, lineno = entry
    return "{}  message={!r}  category={}  module={!r}  line={}".format(
        action, getattr(message, "pattern", message), getattr(category, "__name__", category), module, lineno)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    item.stash[_WARNING_STATE_BEFORE_SETUP] = _warning_state()
    return (yield)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    try:
        result = yield
    except (KeyboardInterrupt, SystemExit, pytest.exit.Exception):
        _check_the_warning_state(item, report=False)        # restored and recorded; the interrupt is never replaced
        raise
    except BaseException:
        _check_the_warning_state(item)                      # a finding raised here carries the teardown's own error as context
        raise
    _check_the_warning_state(item)
    return result


def _check_the_warning_state(item, report: bool = True) -> None:
    import warnings

    before = item.stash.get(_WARNING_STATE_BEFORE_SETUP, None)
    if before is None:
        raise AssertionError("the warning-filter guard was not armed for {}: its setup hook did not run".format(item.nodeid))
    binding, entries, showwarning, showwarnmsg_impl = before
    after = _warning_state()
    tally = item.config.stash[_WARNING_GUARD_TALLY]
    tally["checked"] += 1
    found = []
    if after[0] is not binding:
        found.append("warnings.filters is a DIFFERENT list from the one bound before setup ({} entries, now {})".format(
            len(entries), len(after[1])))
    gone = [e for e in entries if e not in after[1]]
    new = [e for e in after[1] if e not in entries]
    remaining = iter(after[1])
    in_order = all(e in remaining for e in entries)          # the entries it started with, as a subsequence, in their order
    if gone:
        found.append("the filter entries changed: {} gone, {} new".format(len(gone), len(new)))
        found.extend("    gone: " + _describe_filter(e) for e in gone)
        found.extend("    new:  " + _describe_filter(e) for e in new)
    elif not in_order:
        found.append("the entries this test started with are no longer in their original order (order is precedence)")
    if after[2] is not showwarning:
        found.append("warnings.showwarning was replaced: {!r}".format(after[2]))
    if after[3] is not showwarnmsg_impl:
        found.append("warnings._showwarnmsg_impl (the hook pytest records warnings through) was replaced: {!r}".format(after[3]))
    sequential = _joblib_thread_name_is_sequential()
    if not sequential:
        found.append("joblib's 'threading' backend no longer maps to its sequential one (a register_parallel_backend call)")
    if not found:
        for entry in new:
            tally["added"].setdefault(_describe_filter(entry), item.nodeid)
        return
    tally["failed"].append(item.nodeid)
    # Restore FIRST -- containment before reporting.
    warnings.filters = binding
    binding[:] = entries
    warnings.showwarning = showwarning
    warnings._showwarnmsg_impl = showwarnmsg_impl
    if not sequential:
        import joblib
        from joblib.parallel import BACKENDS

        joblib.register_parallel_backend("threading", BACKENDS["sequential"])
    if not report:
        return
    raise AssertionError(
        "THE WARNING FILTERS AFTER THIS TEST'S TEARDOWN ARE NOT THE ONES BOUND BEFORE ITS SETUP "
        "(SKLEARN-WARNING-FILTER-RACE-1). While they differ, pyproject.toml's entries -- the `error::` ones included -- may not "
        "apply, so a warning that must fail a test may not.\n  " + "\n  ".join(found) + "\n"
        "The usual cause is thread-parallel work that enters warnings.catch_warnings() in worker threads (scikit-learn "
        "estimators with n_jobs != 1 under a threading backend, or a thread pool that is not joblib's); a catch_warnings opened "
        "in a module- or session-scoped fixture is another. Set filters inside `with warnings.catch_warnings():` within the test "
        "or a function-scoped fixture, and leave joblib's 'threading' backend as tests/conftest.py registers it. The state has "
        "been restored.")


# ---------------------------------------------------------------------------
# Working-tree guard (added 2026-10-12) -- DETECTION, for the whole session.
# ---------------------------------------------------------------------------
# MEASURED 2026-10-11: the two validation clones that had run the full suite held models/best_gat.pt -- GNNTrainer's default
# checkpoint path is relative to the current directory -- and git status showed nothing, because /models/ and *.pt are ignored.
# The ROADMAP entry of 2026-09-23 said the suite no longer wrote into the repository; nothing had measured it.
#
# A snapshot of the whole working tree is taken when the session starts and again when the TEST LOOP ends
# (repository_hygiene/workspace_snapshot.py says what is observed and what is not entered). Anything ADDED, REMOVED or CHANGED
# is named in its own section of the terminal summary and FAILS the session (exit 1), even when every test passed. A per-test
# check would name the culprit, but walks the tree twice per test; once per session costs two walks, and the names it prints
# lead to the culprit (git grep). Its cost is printed on every run, so it can never grow unseen.
#
# WHY AT THE END OF THE TEST LOOP, NOT AT SESSION FINISH. MEASURED 2026-10-31: the first Continuous Integration run of this guard
# (pull request #56) failed on both Python versions, every test passing, with ADDED .coverage and coverage.xml. pytest-cov's own
# pytest_runtestloop wrapper writes the coverage data file and every --cov-report file AFTER the loop and BEFORE sessionfinish:
# they are outputs the INVOCATION asked for, not something a test left. The last test's teardown -- every session-scoped fixture
# included -- runs inside the loop, so the end of the loop is the end of what the tests did. No file name is excused: a test that
# writes .coverage or coverage.xml itself is still found (tests/unit/test_working_tree_guard.py proves both).
#
# A file you edit yourself while the suite runs is also a difference: the summary names it, and re-running settles it.
_WORKSPACE_ROOT = _here.parent.parent
_WORKSPACE_BEFORE = pytest.StashKey()
_WORKSPACE_AFTER = pytest.StashKey()
_WORKSPACE_REPORT = pytest.StashKey()


def pytest_sessionstart(session) -> None:
    from genomic_variant_classifier.repository_hygiene.workspace_snapshot import take_snapshot

    session.config.stash[_WORKSPACE_BEFORE] = take_snapshot(_WORKSPACE_ROOT)


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtestloop(session):
    """The 'after' snapshot, taken the moment the test loop returns or raises. trylast makes this the INNERMOST wrapper, so it runs
    before any other plugin's wrapper resumes (pytest-cov's writes its files there)."""
    try:
        return (yield)
    finally:
        from genomic_variant_classifier.repository_hygiene.workspace_snapshot import take_snapshot

        session.config.stash[_WORKSPACE_AFTER] = (take_snapshot(_WORKSPACE_ROOT), bool(session.shouldstop or session.shouldfail))


def _warning_guard_lines(tally: dict) -> list:
    lines = ["joblib: 'threading' runs sequentially for the session (no worker threads); {} tests checked, {} changed the "
             "warning state".format(tally["checked"], len(tally["failed"]))]
    if tally["failed"]:
        lines.append("THE WARNING STATE CHANGED IN THESE TESTS (each restored; the session fails):")
        lines.extend("    " + nodeid for nodeid in tally["failed"])
    return lines


def pytest_sessionfinish(session, exitstatus) -> None:
    from genomic_variant_classifier.repository_hygiene.workspace_snapshot import compare, describe, take_snapshot

    tally = session.config.stash.get(_WARNING_GUARD_TALLY, None)
    if tally is not None and tally["failed"] and session.exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED      # also when pytest reported the finding as an xfail
    before = session.config.stash.get(_WORKSPACE_BEFORE, None)
    if before is None:
        session.config.stash[_WORKSPACE_REPORT] = (["NOT ARMED: tests/conftest.py was loaded after the session started, so no "
                                                    "snapshot exists. Nothing was compared."], False)
        return
    at_loop_end = session.config.stash.get(_WORKSPACE_AFTER, None)
    if at_loop_end is None:                     # the test loop never started (collection itself raised)
        after, stopped_early, when = take_snapshot(_WORKSPACE_ROOT), False, "at session finish: the test loop never ran"
    else:
        (after, stopped_early), when = at_loop_end, "at the end of the test loop"
    change = compare(before, after)
    lines = ["{}: {} entries before, {} after ({}); {:.2f} s + {:.2f} s".format(
        _WORKSPACE_ROOT, len(before.entries), len(after.entries), when, before.seconds, after.seconds)]
    if stopped_early:
        lines.append("the test loop stopped early (-x, --maxfail or an interruption): session-scoped fixtures had not been torn down "
                     "when the tree was compared")
    for label, unobserved in (("unobserved at the start", before.unobserved), ("unobserved at the end", after.unobserved)):
        if unobserved:
            lines.append("{} ({}): {}".format(label, len(unobserved), ", ".join(unobserved[:20])))
    if change:
        lines.append("THE TEST SESSION CHANGED THE WORKING TREE. A test must write under tmp_path (or a fixture's folder), never "
                     "into the checkout:")
        lines.extend(describe(change))
        if session.exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    else:
        lines.append("unchanged")
    session.config.stash[_WORKSPACE_REPORT] = (lines, bool(change))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    tally = config.stash.get(_WARNING_GUARD_TALLY, None)
    if tally is not None:
        failed = bool(tally["failed"])
        terminalreporter.section("warning-filter guard", sep="=", red=failed, green=not failed, bold=True)
        for line in _warning_guard_lines(tally):
            terminalreporter.line(line, red=failed)
        tally["printed"] = True
        if tally["added"]:
            terminalreporter.line("entries ADDED during a test (pytest discards them when the test ends), each with the first "
                                  "test that added it:")
            for entry, nodeid in sorted(tally["added"].items()):
                terminalreporter.line("  {}\n      first added in {}".format(entry, nodeid))
    report = config.stash.get(_WORKSPACE_REPORT, None)
    if report is None:
        return
    lines, failed = report
    terminalreporter.section("working-tree guard", sep="=", red=failed, green=not failed, bold=True)
    for line in lines:
        terminalreporter.line(line, red=failed)
    config.stash[_WORKSPACE_REPORT] = (lines, failed, "printed")


@pytest.hookimpl(tryfirst=True)
def pytest_unconfigure(config) -> None:
    """Registers joblib's own thread backend again and restores what temporary-file containment changed; then, if the terminal
    summary was skipped (--no-summary, -p no:terminal, some exit codes), writes the two guards' findings here."""
    original = config.stash.get(_JOBLIB_THREAD_BACKEND, None)
    if original is not None:
        import joblib

        joblib.register_parallel_backend("threading", original)
    tally = config.stash.get(_WARNING_GUARD_TALLY, None)
    if tally is not None and tally["failed"] and not tally["printed"]:
        sys.stderr.write("\n".join(["warning-filter guard:"] + _warning_guard_lines(tally)) + "\n")
    saved = config.stash.get(_CONTAINMENT_SAVED, None)
    if saved is not None:
        import tempfile

        env, tempdir = saved
        tempfile.tempdir = tempdir
        for name, value in env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    report = config.stash.get(_WORKSPACE_REPORT, None)
    if report is None or len(report) == 3 or not report[1]:
        return
    sys.stderr.write("\n".join(["working-tree guard:"] + report[0]) + "\n")
