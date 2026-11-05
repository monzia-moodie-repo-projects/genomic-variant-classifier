"""Session guards for every test suite in this repository (conftest.py at the repository root).

pytest loads a conftest.py at the root directory for every run inside the repository -- `pytest tests/` (Continuous Integration's
test job and the owner's gate) and `pytest tests_drift/` (the drift-monitor suite, which runs in its own environment) alike.
MEASURED 2026-11-04 (pytest 9.0.3): with this file at the root, both invocations load it, and tests/conftest.py is loaded for
tests/ only; a child pytest that a test starts on files OUTSIDE the repository does not load it, even with this repository's
configuration (-c) or this directory as its working directory.

WHY HERE (DRIFT-SUITE-UNGUARDED-1, 2026-11-04). Until then these guards lived in tests/conftest.py, which tests_drift/ never
loads: the drift suite ran with no warning-filter guard, no working-tree guard, its temporary files in the system temporary
folder, and only ci.yml's `env -u` between its tests and the workflow's publication files. What applies to every suite is
here; what belongs to tests/ alone -- scripts/ on sys.path, the suite-size ratchet, the connector-cache isolation -- stays in
tests/conftest.py.

The guards, in this file's order: sys.path leaks and writes into data/raw/cache (per test); the workflow's publication files
(per session and per test); the environment and the conftest.py of a child pytest; temporary-file containment; no thread-parallel
joblib work and the warning-filter guard; the libraries that register warning filters when first imported, imported at session
start, and the check around collection (since 2026-11-05); the working-tree guard (per session).

Author: Monzia Moodie
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_here = Path(__file__).resolve()


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
# WHAT WORKS. `_isolate_connector_caches` (tests/conftest.py since 2026-11-04) saves and restores the module attributes BY HAND
# and requests only `tmp_path`. With no autouse fixture depending on it, `monkeypatch` is
# instantiated only when a TEST asks for it -- i.e. after every autouse fixture -- so its
# finaliser runs FIRST, and the guards see fully-reverted global state.
#
# WHERE IT BINDS SINCE 2026-11-04 (MEASURED, independent review). pytest sets up the autouse fixtures of an OUTER conftest before
# those of an inner one, and within one conftest in order of name. The guards are in this outermost conftest, so an autouse
# fixture in tests/conftest.py or deeper is set up after them, and a `monkeypatch` it requests is torn down before them: with
# _isolate_connector_caches changed to request `monkeypatch`, tests/test_rekey_seq_windows_v2.py gave 7 passed and 2 errors in the
# old layout and 7 passed, 0 errors in this one (tests/unit/conftest.py::_isolate_spliceai requests it, harmlessly, in both).
# Inside THIS file the rule still binds: _isolate_github_workflow_commands sorts ahead of the two guards. tests/conftest.py keeps
# the rule by choice. The autouse order is now: _isolate_github_workflow_commands, _no_data_dir_writes, _no_sys_path_leaks (this
# file), then tmp_path and _isolate_connector_caches (tests/conftest.py) -- so the guards also see the connector-cache fixture's
# first import of the embedding and protein modules (measured: it leaves sys.path unchanged and writes nothing there).
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
_DATA_CACHE = _here.parent / "data" / "raw" / "cache"

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
# GitHub workflow-command isolation (added 2026-09-27) -- PREVENTION.
# ---------------------------------------------------------------------------
# MEASURED on CI run #896 (the merge of #28): the pytest (3.11) and (3.12) job summaries carried
# "Source-monitor run verification (PREVIEW)" reports -- one saying run 36300779115 was VERIFIED and
# one saying the SAME run was NOT VERIFIED ("archive digest ... differs from GitHub's"). Neither
# verification happened: tests/unit/test_report_verifier.py called scripts/verify_monitor_run.py's
# main(), which appended to $GITHUB_STEP_SUMMARY -- and inside CI that variable is SET. A tampered-
# archive TEST published a fabricated verdict about a real run onto CI's own summary page.
#
# These six variables name the files through which a process PUBLISHES to its workflow run (job
# summary, step outputs, environment and PATH for later steps, action state, artifact declarations). No test may publish.
# They are removed before every test and restored after it; a subprocess a test launches inherits
# os.environ, so it cannot publish either. Same discipline as tests/conftest.py's _isolate_connector_caches:
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


# The conftest.py of a CHILD project (added 2026-11-05) -- the one source. Four test files run child sessions whose root conftest.py
# is this file; until 2026-11-05 each wrote a byte copy. A byte copy now imports this file's list of filter-registering modules
# (torch, spaCy and the rest, about 4 s) at session start, and with no pyproject.toml of the repository's to hide them, reports the
# three warnings torch_geometric emits about its own code: neither has anything to do with the guard such a session tests. So a
# child gets this file with exactly one statement changed -- _FILTER_REGISTERING_MODULES, replaced by the names given (none by
# default) -- found by parsing; tests/unit/test_filter_registering_imports.py proves nothing else differs.
@pytest.fixture
def child_conftest():
    """A function returning this file's bytes with _FILTER_REGISTERING_MODULES replaced by the given module names."""
    def build(*names: str) -> bytes:
        import ast

        text = _here.read_text(encoding="utf-8")
        found = [node for node in ast.parse(text, filename=str(_here)).body if isinstance(node, ast.Assign)
                 and [getattr(target, "id", "") for target in node.targets] == ["_FILTER_REGISTERING_MODULES"]]
        if len(found) != 1:
            raise AssertionError("conftest.py (the repository root's) must assign _FILTER_REGISTERING_MODULES exactly once; "
                                 "found {}".format(len(found)))
        lines = text.splitlines(keepends=True)
        listing = "_FILTER_REGISTERING_MODULES = {!r}\n".format(tuple(names))
        return "".join(lines[:found[0].lineno - 1] + [listing] + lines[found[0].end_lineno:]).encode("utf-8")
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
            "conftest.py (the repository root's) cannot contain temporary files: pytest's temporary-path factory "
            "(config._tmp_path_factory, the object the tmp_path_factory fixture returns) does not exist in this pytest. Without it "
            "every mkdtemp() in the suite would land in the system temporary folder and stay there.")
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
# AN ENTRY ADDED during a test fails it too, since 2026-11-05 (WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1; from 2026-11-03 added
# entries were listed in the summary, not failed). simplefilter() and filterwarnings() put an entry at the FRONT of the list,
# ahead of pyproject.toml's entries, for the rest of the test -- an identical entry already in the list is MOVED there, the one
# shape that failed before -- so an added entry and a moved one have the same effect, and differed only by whether that entry
# happened to exist already. Libraries that register filters when first imported are imported at session start (the next
# section), so no test's first import can register anything; what is left to fail is a test that sets a filter outside
# `with warnings.catch_warnings():`, or a library missing from that section's list.
# Hook wrappers, not fixtures: no fixture runs before a module-scoped fixture is set up (THE INVARIANT FOR THIS FILE holds --
# nothing here requests monkeypatch). tests/unit/test_sklearn_parallel_warning_contract.py proves both, in this session and in
# child sessions that run this file (child_conftest, above).
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
    config.stash[_WARNING_GUARD_TALLY] = {"checked": 0, "failed": [], "printed": False}
    joblib.register_parallel_backend("threading", BACKENDS["sequential"])
    resolved = [type(get_active_backend(prefer="threads", require="sharedmem")[0]), type(get_active_backend(prefer="threads")[0]),
                type(get_active_backend(require="sharedmem")[0])]
    if not _joblib_thread_name_is_sequential() or any(b is not BACKENDS["sequential"] for b in resolved):
        raise pytest.UsageError(
            "conftest.py (the repository root's) mapped joblib's 'threading' backend to its sequential one, and joblib does not "
            "resolve thread requests to it (resolved: {}). Thread-parallel scikit-learn would rebind the warning filters "
            "(SKLEARN-WARNING-FILTER-RACE-1); the suite does not run without this.".format([b.__name__ for b in resolved]))


def _warning_state() -> tuple:
    import warnings

    return warnings.filters, tuple(warnings.filters), warnings.showwarning, warnings._showwarnmsg_impl


def _describe_filter(entry) -> str:
    action, message, category, module, lineno = entry
    return "{}  message={!r}  category={}  module={!r}  line={}".format(
        action, getattr(message, "pattern", message), getattr(category, "__name__", category), module, lineno)


def _filter_differences(binding, entries: tuple, after_binding, after: tuple, bound: str, subject: str) -> list:
    """Every difference between two warning-filter states, one line each; empty only when they are the same list object holding the
    same entries in the same order. Shared by the per-test guard and the collection check; `bound` and `subject` name the 'before'.
    Each finding says where the entry stands now: precedence is position (the first matching entry decides)."""
    found = []
    if after_binding is not binding:
        found.append("warnings.filters is a DIFFERENT list from the one bound {} ({} entries, now {})".format(
            bound, len(entries), len(after)))
    gone = [e for e in entries if e not in after]
    new = [e for e in after if e not in entries]
    if gone:
        found.append("the filter entries changed: {} gone, {} new".format(len(gone), len(new)))
        found.extend("    gone: " + _describe_filter(e) for e in gone)
        found.extend("    new:  " + _describe_filter(e) for e in new)
        return found
    now = {e: after.index(e) for e in entries}               # where each entry it started with stands now (its first occurrence)
    if new:
        found.append("{} filter entr{} registered -- by simplefilter() or filterwarnings() outside "
                     "`with warnings.catch_warnings():`, or by a library's first import:".format(
                         len(new), "y was" if len(new) == 1 else "ies were"))
        for e in new:
            i = after.index(e)
            found.append("    new:  {}  (position {} of {}; ahead of {} of the {} entries {} started with)".format(
                _describe_filter(e), i + 1, len(after), sum(1 for s in entries if now[s] > i), len(entries), subject))
    moved = []
    for j, e in enumerate(entries):
        overtaken = sum(1 for o in entries[:j] if now[o] > now[e])     # entries it used to follow and now precedes
        if overtaken:
            moved.append("    moved: {}  (now ahead of {} entr{} it used to follow)".format(
                _describe_filter(e), overtaken, "y" if overtaken == 1 else "ies"))
    if moved:
        found.append("the entries {} started with are no longer in their original order (order is precedence):".format(subject))
        found.extend(moved)
    if not found and tuple(after) != tuple(entries):
        # Nothing new, nothing gone, nothing out of order -- and still not the same: an entry appears more often (only a direct
        # list operation can do that; simplefilter() and filterwarnings() remove an identical entry first). Fail closed.
        extra = [e for e in dict.fromkeys(after) if after.count(e) > entries.count(e)]
        found.append("the entries {} started with are all there, in order, and the list still differs: duplicated".format(subject))
        found.extend("    duplicated: {}  (now {} times, {} before)".format(_describe_filter(e), after.count(e), entries.count(e))
                     for e in extra)
    return found


def _state_differences(before: tuple, after: tuple, bound: str, subject: str) -> list:
    """_filter_differences, plus the two functions warnings are shown through. `before` and `after` are _warning_state()s."""
    found = _filter_differences(before[0], before[1], after[0], after[1], bound, subject)
    if after[2] is not before[2]:
        found.append("warnings.showwarning was replaced: {!r}".format(after[2]))
    if after[3] is not before[3]:
        found.append("warnings._showwarnmsg_impl (the hook pytest records warnings through) was replaced: {!r}".format(after[3]))
    return found


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
    found = _state_differences(before, after, "before setup", "this test")
    sequential = _joblib_thread_name_is_sequential()
    if not sequential:
        found.append("joblib's 'threading' backend no longer maps to its sequential one (a register_parallel_backend call)")
    if not found:
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
        "in a module- or session-scoped fixture is another, and a filter registered during the test -- by the test outside "
        "`with warnings.catch_warnings():`, or by a library imported for the first time inside it (declare that library in "
        "_FILTER_REGISTERING_MODULES, the repository root's conftest.py) -- a third. Set filters inside "
        "`with warnings.catch_warnings():` within the test or a function-scoped fixture, and leave joblib's 'threading' backend "
        "as the repository root's conftest.py registers it. The state has been restored.")


# ---------------------------------------------------------------------------
# Libraries that register warning filters when first imported (added 2026-11-05, WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1) --
# PREVENTION, and a check of its completeness on every run.
# ---------------------------------------------------------------------------
# THE DEFECT. Thirteen libraries in the training-and-test environment register warning filters when first imported, outside a
# catch_warnings of their own (read, 2026-11-04): requests, urllib3, numpy, pandas, scipy, torch, sympy, torch_geometric,
# accelerate, mlxtend, srsly, spaCy and pyspark. mlxtend's frequent_patterns/fpcommon.py:8 calls simplefilter("always",
# DeprecationWarning) at module level; spaCy's errors.py registers numpy's two "size changed" ignore entries again. pytest collects
# inside warnings.catch_warnings() and DISCARDS what collection registered (pytest 9.0.3, _pytest/warnings.py, read): in a full
# run, where collection imports them all, no test runs under those entries. In a run of fewer files, a library those files do
# not import at module level is imported for the first time INSIDE A TEST -- by its body or a fixture -- and what it registers
# goes to the front of that test's filters, ahead of pyproject.toml's entries, for the rest of the test; an identical entry is
# moved there. So a test's filters depended on which tests ran with it.
# MEASURED 2026-11-04 (sandbox, Linux, Python 3.12.3, the training-and-test lock's environment, pytest 9.0.3):
#   * tests/unit/test_1kgp.py run alone: 46 passed, 1 error -- this guard failed the teardown of
#     test_population_cols_in_tabular_features, whose body imports models.variant_ensemble (and through it mlxtend and spaCy) for
#     the first time; the full suite passes it (0 changed).
#   * every one of the 425 files of tests/ run alone, one at a time, with Continuous Integration's options apart from coverage:
#     17 failed this guard (docs/sessions/SESSION_2026-11-05_warning-state-determinism.md has the list).
#   * a census of every filter registration from pytest's plugin loading to the end of the full collection (8,647 tests): 124
#     calls -- 61 by pytest itself (its configured entries per phase, and 29 pytest.importorskip calls), 63 by libraries and
#     Python's os module; the ones that persist come from the thirteen libraries above, imported through the names below.
#
# PREVENTION. At session start -- after the working-tree guard's first snapshot, before collection -- every module named below
# that this environment has is imported inside warnings.catch_warnings(): what it registers is discarded, as collection discards
# it, and the module stays imported, so no later import, in collection or in a test, registers anything again. Every selection
# of tests then runs each test under the filters the full run gives it. A module this environment does not have (the drift
# environment has no torch) cannot be imported by any test either: it is named as absent in this guard's summary. A module that
# is installed and fails to import stops the session with a usage error naming it, what it warned and what it changed in the
# working tree (pytest runs neither the terminal summary nor sessionfinish after session start raises). A TRADE-OFF, chosen: a test
# guarded by pytest.importorskip used to skip when its library was installed but broken; now the session stops, because a broken
# installation is not an absent dependency (the locks install every one of these).
#
# COMPLETENESS, on every run. The warning state (the filters and the two functions warnings are shown through) is compared before
# and after collection, inside the catch_warnings pytest's warnings plugin opens for it (measured: this wrapper runs innermost). If
# it differs, something collection imported changed it -- a library missing from the list below, or a test module that registers
# a filter at module level (which collection discards, and which a run without that module would not have) -- and the session
# stops with a usage error naming the entries, before any test runs; with `-p no:warnings` nothing is discarded, and the check
# stops the session all the same. A library first imported inside a test, never at collection, is caught by the per-test guard
# above. Not a difference, by design: a library that changes the state at CALL time and restores it within the same test.
_FILTER_REGISTERING_MODULES = ("requests", "numpy", "pandas", "scipy.sparse", "scipy.special", "torch", "torch_geometric",
                               "accelerate", "mlxtend.frequent_patterns", "spacy", "pyspark")


def _import_the_filter_registering_modules(config) -> None:
    import importlib
    import importlib.util
    import time
    import warnings

    start = time.perf_counter()
    imported, absent = [], []
    for name in _FILTER_REGISTERING_MODULES:
        top = name.partition(".")[0]
        if top not in sys.modules and importlib.util.find_spec(top) is None:
            absent.append(name)
            continue
        # record=True: the filters in force are the configuration's (pytest applies them for the whole configured session), so an
        # `error::` entry still raises; whatever else the import warns goes to pytest's own report, as a warning collection
        # emitted would -- never to the console unseen, never dropped.
        failure = None
        with warnings.catch_warnings(record=True) as emitted:
            try:
                importlib.import_module(name)
            except Exception as error:
                failure = error
        for message in emitted:
            config.hook.pytest_warning_recorded.call_historic(
                kwargs=dict(warning_message=message, when="config", nodeid="", location=None))
        if failure is not None:
            raise pytest.UsageError(_import_failure(config, name, failure, emitted)) from failure
        imported.append(name)
    config.stash[_WARNING_GUARD_TALLY]["imported"] = (tuple(imported), tuple(absent), time.perf_counter() - start)


def _import_failure(config, name: str, failure: Exception, emitted: list) -> str:
    """The usage error for a listed module that is installed and fails to import. pytest skips the terminal summary and
    sessionfinish when session start raises, so everything that would have been reported later is in this message: the warnings
    the import emitted, and what it changed in the working tree (compared here, against the snapshot taken just before)."""
    from genomic_variant_classifier.repository_hygiene.workspace_snapshot import compare, describe, take_snapshot

    lines = ["conftest.py (the repository root's) imports {!r} at session start, so that the warning filters it registers when "
             "first imported are discarded (WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1). It is installed and failed to import: "
             "{}: {}".format(name, type(failure).__name__, failure)]
    if isinstance(failure, Warning):
        lines.append("  That exception is a warning: a configured `error::` filter turned it into one (this repository's are in "
                     "pyproject.toml).")
    lines.extend("  warned while importing: {}: {} ({}:{})".format(w.category.__name__, w.message, w.filename, w.lineno)
                 for w in emitted)
    before = config.stash.get(_WORKSPACE_BEFORE, None)
    change = compare(before, take_snapshot(_WORKSPACE_ROOT)) if before is not None else None
    if change:
        lines.append("  THE IMPORT CHANGED THE WORKING TREE:")
        lines.extend("  " + line for line in describe(change))
    elif before is not None:
        lines.append("  The working tree is unchanged.")
    lines.append("  Tests that would have skipped such a library through pytest.importorskip now stop the session instead: an "
                 "installed library that cannot be imported is a broken environment, not an absent dependency.")
    return "\n".join(lines)


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_collection(session):
    """The completeness check. trylast: inside pytest's own wrapper, so the comparison is made within the catch_warnings pytest
    opens for collection, around everything collection imports."""
    before = _warning_state()
    result = yield
    found = _state_differences(before, _warning_state(), "when collection started", "collection")
    if found:
        raise pytest.UsageError(
            "THE WARNING STATE CHANGED WHILE TESTS WERE COLLECTED (WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1). A module imported by "
            "collection changed it; pytest's warnings plugin discards that when collection ends, but a run that does not collect "
            "that module would make the change inside the first test that imports it, ahead of pyproject.toml's entries.\n  " +
            "\n  ".join(found) + "\n"
            "If a library registered them, add the module that registers them (or one that imports it) to "
            "_FILTER_REGISTERING_MODULES in conftest.py (the repository root's). If a test module did, set the filter with "
            "pytestmark = pytest.mark.filterwarnings(...) instead.")
    return result


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
_WORKSPACE_ROOT = _here.parent
_WORKSPACE_BEFORE = pytest.StashKey()
_WORKSPACE_AFTER = pytest.StashKey()
_WORKSPACE_REPORT = pytest.StashKey()


def pytest_sessionstart(session) -> None:
    from genomic_variant_classifier.repository_hygiene.workspace_snapshot import take_snapshot

    session.config.stash[_WORKSPACE_BEFORE] = take_snapshot(_WORKSPACE_ROOT)
    _import_the_filter_registering_modules(session.config)       # after the snapshot: the working-tree guard sees what they do


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
    if "imported" in tally:
        imported, absent, seconds = tally["imported"]
        lines.append("imported at session start, what they register discarded: {} in {:.2f} s; absent in this environment: {}".format(
            ", ".join(imported) or "none", seconds, ", ".join(absent) or "none"))
    else:
        lines.append("NOT ARMED: conftest.py (the repository root's) was loaded after the session started, so the modules that "
                     "register warning filters were not imported at session start; a first import inside a test fails that test")
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
        session.config.stash[_WORKSPACE_REPORT] = (["NOT ARMED: conftest.py (the repository root's) was loaded after the session "
                                                    "started, so no snapshot exists. Nothing was compared."], False)
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
