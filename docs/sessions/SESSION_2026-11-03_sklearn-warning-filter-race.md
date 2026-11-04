# Session 2026-11-03 -- SKLEARN-WARNING-FILTER-RACE-1

Author: Monzia Moodie

On 2026-11-03 at 06:18 UTC the owner's post-merge block for the drift-profile change reported 448 copies of scikit-learn's
warning "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` ..." from
tests/unit/test_isolation_prerequisite.py, although pyproject.toml carried an ignore entry for exactly that message. Pull request
#58's validation had reported 1,000 from tests/unit/test_containment_serving.py on 2026-11-01. The same files reported none in
every other measured run. This session measured why, and closed it for the suite.

## What was wrong, measured

1. The mechanism, read in full in scikit-learn 1.8.0 (sklearn/utils/parallel.py) and CPython 3.12's warnings module. The warning
   filters are ONE process-wide list; `catch_warnings` saves the list bound on entry and rebinds it on exit. Each task of a
   thread-parallel scikit-learn call runs inside `config_context(...), catch_warnings()`, calls `resetwarnings()` and refills the
   list one entry at a time from the list `Parallel.__call__` captured -- the list OBJECT, not a copy. Tasks in concurrent threads
   enter and exit out of order, so the process ends bound to some task's rebuilt copy, and an interleaving in which a task resets
   while the captured list is bound empties it. Every later task then finds no captured filters and issues the warning, and no
   entry -- the ignore entry included, and every `error::` entry -- applies.
2. The environment was ruled out by content: the observation of the candidate environment that reported 448 and that of the one
   that reported 0 differ only in folder name, observation time and the editable install's commit (scikit-learn 1.8.0, joblib
   1.5.3, numpy 2.4.4, pytest 9.0.3 and the interpreter's digest identical).
3. A probe run on the owner's machine (8 processors, 700 forest fits at n_jobs 1, 2, 8 and -1, outside pytest) counted 0 changes.
   That probe was mine and was blind in two ways: it compared the list's CONTENTS only, never which list was bound, and it called
   resetwarnings() first, which removed the one default entry that comes back changed.
4. Inside pytest, in the development sandbox (2 processors), 20 forest fits of 500 trees at n_jobs=-1: the process was left bound
   to a different list after 20 of 20, and with pytest's 17-entry list the contents differed after 20 of 20 --
   `default::DeprecationWarning:__main__` names its module as a plain string, which matches by equality, and comes back as a
   compiled regular expression, which matches by prefix. With a 2-entry list the binding changed after 20 of 20 and the contents
   after none. No list was emptied in these 40 fits; that rarer outcome is the 448.
5. A passive census of the full suite on main (2c2ba877; Linux, Python 3.12): 237 scikit-learn Parallel calls inside tests, 95
   threaded (more than one worker) in 16 tests of 8 files, 37.7 s; 14 of those tests ended with the filters rebound and changed;
   no test without a threaded call changed them. The run's 8 failures were this sandbox's git 2.43.0 refusing --no-lazy-fetch
   (known; they pass with git 2.47.1).
6. Upstream: scikit-learn issue #35089 (opened 2026-10-06) is open; its reporter finds every build with the global interpreter
   lock affected, 3.14.6 and 3.14.7 included. Python's concurrency-safe warnings (context_aware_warnings) are off by default on
   those builds in 3.14 and, per the Python discussion of 2026-07-24, will stay so in 3.15.
7. The pyproject.toml explanation of the ignore entry (2026-07-22) -- a harmless thread-configuration note from an estimator run
   "inside a fold" -- was wrong; the entry could neither hide the warning reliably nor stop the race from disabling the other
   entries.

## The change

- tests/conftest.py, PREVENTION. `joblib.register_parallel_backend("threading", SequentialBackend)` at configuration (before
  collection, like the temporary-file containment), verified for the three thread hints, and joblib's own thread backend
  registered again at unconfigure. joblib 1.5.3 resolves every thread request that NAMES its backend through that name: an
  explicit backend="threading" (scikit-learn passes it in ensemble/_hist_gradient_boosting/binning.py:240 and
  metrics/pairwise.py:1979), the hints prefer="threads" and require="sharedmem" (thirteen sites: eleven literal, two through a
  variable, ensemble/_iforest.py:296 and linear_model/_logistic.py:1948), and parallel_config(backend="threading"). A backend
  INSTANCE bypasses the registry (none in the package, the tests, the scripts or scikit-learn 1.8.0 outside its own tests), and a
  thread pool that is not joblib's is not prevented; the guard catches both by their effect. Process backends are untouched, so
  scalable_svm's bags run as in production.
- tests/conftest.py, DETECTION. Hook wrappers take the warning state before each test's setup (before every fixture, module- and
  session-scoped included) and compare it after its teardown: the list pytest bound, its entries, their order, showwarning, the
  hook pytest records through, and the thread mapping. A difference fails the teardown of the test in which it is found, naming
  what changed, after the state is restored -- and the session, because pytest reports an exception in an xfail-marked test's
  teardown as an xfail. An interrupt during teardown (KeyboardInterrupt, SystemExit, pytest.exit) is never replaced: the state is
  restored and recorded, and the interrupt propagates. Entries ADDED during a test are not a failure -- libraries register filters
  at first import (torch three, scipy and numpy one each, measured) and pytest discards them when the test ends -- but every
  distinct one is listed, with the first test that added it, in the guard's own terminal-summary section, because an added entry
  can shadow one of pyproject.toml's. A catch_warnings opened in a module- or session-scoped fixture is found too: pytest binds a
  fresh list for every test, so such a fixture's filters never governed the tests between (the suite has none).
- tests/conftest.py, child_pytest_environment: the one source for the environment of a child pytest that a test starts, without
  the variables that change what it prints (CI and BUILD_NUMBER, COLUMNS and LINES, PY_COLORS, FORCE_COLOR and NO_COLOR,
  PYTEST_ADDOPTS, PYTEST_PLUGINS, PYTHONWARNINGS). Every child pytest in the suite uses it: seven calls in six test files, found
  by parsing tests/ (test_working_tree_guard.py, test_warning_filters_are_scoped.py, this change's contract test twice,
  test_replicate_engine.py, test_review_status_tier.py, test_github_workflow_command_isolation.py).
- pyproject.toml: `error::UserWarning:sklearn\.utils\.parallel`, selected by module like the ranking entry. scikit-learn 1.8.0's
  sklearn/utils/parallel.py warns at exactly two places (lines 29 and 144), both the scikit-learn configuration not reaching the
  workers, from either side of a delayed/Parallel mismatch. With no worker threads, what remains is a genuine defect -- or a
  caller that emptied the filters itself, which neither the package nor the suite does.
- tests/unit/test_sklearn_parallel_warning_contract.py rewritten (12 tests, replacing 3; its child sessions bring their own
  pytest.ini, so they hold on pytest 9.0.3 and 9.2.0 alike); tests/unit/test_warning_filters_are_scoped.py fails on a file it
  cannot read or parse and names it (+1 test; all 437 files under tests/ and 298 under src/ parse, so the silent skips had never
  fired) and its stale count of scoped filters is replaced by the measured one (9 in tests/, 2 in scripts/). README Quickstart
  states how the suite is run. Suite 8,633 -> 8,643 (+13 / -3 by node identity).

## How the design changed while it was measured

The first version set `parallel_config(backend="sequential")` for the session. Measured in joblib 1.5.3 it is thread-local,
applied at construction and restored on exit, and an explicit sequential backend is not forced back to threads by scikit-learn's
hints -- a forest at n_jobs=-1 then ran all 32 tasks in the main thread. Its first exercise in the full set of threaded files
failed tests/unit/test_containment_propagation.py::test_control_ordinary_scorer_failure_falls_back[predict_proba_with_uncertainty]
at teardown: the binding replaced and the `__main__` entry rebuilt, scikit-learn's signature. A probe recording every
`catch_warnings` entry off the main thread found 91 in each of the four tests that fit the stand-in model, in two of joblib's pool
threads, all from HistGradientBoostingClassifier's binning, which passes backend="threading" explicitly -- and an explicit backend
overrides parallel_config. The census had recorded those calls as threaded; I had read them as negligible instead of asking why
they were threaded at all. The setting also made joblib's process backends sequential, which nothing called for. The registry
mapping replaced it.

## The independent review, before delivery

A separate review of the commit, which read the pinned joblib 1.5.3, scikit-learn 1.8.0, pytest 9.0.3 and 9.2.0 and CPython
3.12.3 sources and ran reduced child projects, reported five defects and several overstatements. Each was measured here and
resolved in this change:

1. Under CI=true (which GitHub Actions sets) pytest stops truncating its short summary, and a count in the guard's test doubled
   (4 to 8): the pull request's pytest jobs would have failed. Children now run in child_pytest_environment, and the count is
   taken on traceback lines. Measured afterwards: all 36 tests of the three child-starting files pass with CI=true COLUMNS=60
   PY_COLORS=1 FORCE_COLOR=1, and with none of them; before, PY_COLORS=1 alone failed three pre-existing child tests.
2. On pytest 9.2.0 the children had no error entry to lose (9.0.3 adds one of its own for PytestRemovedIn9Warning; 9.2.0 does
   not): each child now brings `error::ResourceWarning` in its own pytest.ini.
3. A finding in the teardown of an xfail-marked test was reported as "xfailed" and the session passed. The session now fails
   (measured: "7 passed, 2 xfailed, 4 errors", exit 1, the xfail-marked test named among the five).
4. A KeyboardInterrupt raised during teardown was replaced by the finding and the session ran on. Measured now: exit 2, the next
   test never runs, the change is still named.
5. A leaked parallel_config holding a ThreadingBackend instance would have failed every later test: the per-test check now reads
   the registry mapping only (a threaded fit is still caught by what it does to the filters).
Also: the reverse mismatch (a joblib delayed function under scikit-learn's Parallel) was neither an error nor ignored -- the
module entry covers it; "every route", "eleven sites", the guard's advice about module-scoped fixtures and the README's "as
Continuous Integration runs them" were corrected; the scans now name a file they cannot read or parse.

## Verification

- A forest, a histogram gradient-boosting fit and pairwise distances: with joblib as shipped, all 40 of scikit-learn's tasks in
  worker threads and the filters rebound and changed; with the mapping, all 40 in the main thread, also under
  parallel_config(backend="threading"), and the filters bound and unchanged; with the original registered again, threads again.
- Results: 500 trees on 3,000 rows gave identical trees and bitwise-equal probabilities, threaded and sequential (2 processors).
- Cost (one run each, 2 processors): the 8 files that ran threaded took 90.06 s on main and 72.56 s with the change; the guard
  checked their 100 tests and found 0 changes.
- ci.yml's `pytest tests/` command on this change (Linux, Python 3.12, git 2.47.1 first on PATH): 8,611 passed, 26 skipped, 6
  xfailed, 0 failed, no warnings, coverage 84%, 956.28 s; 8,643 collected, equal to tests/EXPECTED_SUITE_SIZE; the warning-filter
  guard checked 8,643 tests and found 0 changes, and no entry was added during a test; the working-tree guard: unchanged.
  `pytest tests_drift/` in the drift environment: 19 passed, no warnings. Python 3.11 is measured by Continuous Integration on
  the pull request: its environment could not be rebuilt within this sandbox's disk allowance.
- The installer, rehearsed end to end in the sandbox on the first revision (Linux locks; 21 of 21 cases: dry-run refusals,
  -Validate with both candidate environments, the full suite inside it -- 8,608 passed, 26 skipped, 6 xfailed -- and -Promote to
  a simulated GitHub). Its first attempt stopped at the training-and-test install on the sandbox's disk allowance ("No space
  left on device"), changing nothing, as designed.

## The skips, audited (Windows against Linux)

The owner's validation of 2026-11-03 (receipt 1514ecb3, outcomes 9d3be48d) skipped 83 tests and xfailed 5; this sandbox's full run
skipped 26 and xfailed 6. Grouped by reason, every difference has a measured cause. Windows: 38 R (28 of them "Rscript not on
PATH"), 16 that need POSIX (shell stand-ins, Linux runner blocks, a FIFO), 12 symbolic links (the account lacks the privilege; the
junction tests cover the boundary), 9 pyBigWig parity (no Windows wheel, by the lock), 8 absent data or unfinished (TODO). Linux:
10 R (the DANDELION library and the fixture variables), the same 8, 4 for the Windows-targeted Run17_Postflight.ps1, 3 Windows
junctions, 1 ESM-2 offline. The sixth Linux xfail is test_esm2_not_in_stub_mode (weights unavailable offline here; it passed on
Windows).

## Open, recorded

- Production runs keep their threads, so a threaded fit there still perturbs the process's warning filters (the package sets
  none at run time; warnings may be duplicated or shown that a filter would hide). Closes with scikit-learn #35089 or with an
  interpreter whose warnings are concurrency-safe.
- DRIFT-SUITE-UNGUARDED-1 (new): tests_drift/ has no conftest, so neither this guard nor the working-tree guard reaches the drift
  suite. pyproject.toml's entries do govern it.
- WINDOWS-RSCRIPT-NOT-ON-PATH-1 (new): 28 R tests skip in the owner's validations because Rscript is not on that PATH (and 10
  more R tests skip there as they do on Linux); whether any R-dependent test runs on Windows at all is not yet established.
