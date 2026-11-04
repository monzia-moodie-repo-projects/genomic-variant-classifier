# Session 2026-11-04 -- DRIFT-SUITE-UNGUARDED-1, and the warning-filter race verified on Windows

Author: Monzia Moodie

The warning-filter-race change (pull request #63) closed SKLEARN-WARNING-FILTER-RACE-1 for tests/ and recorded what it did not
reach: tests_drift/, the drift-monitor suite that runs in its own environment, has no conftest of its own and never loads
tests/conftest.py. This session verified pull request #63 on the owner's machine and moved the session guards to where every
suite loads them.

## The warning-filter race, verified on Windows (set "ae")

- The owner ran -Validate three times on 2026-11-04. Each passed in its own folder with the checkout verified unchanged: 8,555
  passed, 83 skipped, 5 xfailed, 0 failed, no warnings, coverage 83%; the warning-filter guard 8,643 checked, 0 changed; the
  working tree unchanged; the drift tests 19 passed. The third run (07:50 UTC, 98 minutes, candidate 23d7c76a) was published.
- The first run took 240 minutes, 169 of them in one stage, installing the training-and-test environment. Measured from its
  install log: 41 of 42 package downloads ran below 1 MB/s (the slowest at 82 kB/s) and one dropped connection was retried; the
  second run's downloads ran at 18-75 MB/s. The owner's network that night, not the installer.
- Pull request #63 was merged with a merge commit (c98c5b92: parents 2c2ba877 and 23d7c76a, tree ad420dce, fetched and checked
  here); GitHub showed 7 checks passed. Their logs were not readable from the development sandbox and are not in this record.
- The post-merge block (10:15 UTC): targeted 567 passed, 3 skipped, NO warnings -- two post-merge blocks earlier reported 448 --
  the guard 570 checked, 0 changed; both candidate environments still exactly their Windows locks; drift 19 passed.
- The branch deletion: refused once by GitHub ("[remote rejected] (fatal error in commit_refs_rules)", 10:28 UTC, inside GitHub's
  incident with Pull Requests, Git Operations and Issues opened 09:49 UTC); the same block succeeded at 14:01 UTC.
- The 83 Windows skips, counted with their [N] weights from the published run's log: R 38 (28 with Rscript not on PATH, 8 with
  the pinned DANDELION package absent from R's library, 2 with the fixture variables unset), POSIX-only 15, symbolic links 13,
  pyBigWig and pybigtools 9, absent data or TODO 8. The roadmap's "28 R tests" was the first group only; it now says so.

## What was wrong, measured

tests_drift/test_performance_estimator.py (19 tests) writes under tmp_path, starts child interpreters, and uses monkeypatch and
caplog. Continuous Integration runs it as `env -u GITHUB_STEP_SUMMARY ... pytest tests_drift/ -v -rs`. With no conftest in
tests_drift/ and the guards in tests/conftest.py, none of them reached it: not the warning-filter guard or joblib's sequential
thread backend, not the working-tree guard, not temporary-file containment (its tmp_path folders and child processes used the
system temporary folder), and not layers A and B of the workflow-command isolation (only ci.yml's layer C).

## The design, measured before it was written

- A probe project with this repository's layout (pyproject.toml with testpaths = ["tests"], tests/ and tests/unit/ packages, a
  root conftest.py, tests/conftest.py, tests_drift/): `pytest`, `pytest tests` and `pytest tests/unit/test_a.py` load both
  conftests; `pytest tests_drift` and `pytest tests_drift/test_d.py` load the root one only.
- A child pytest that a test starts on files outside the repository does not load the root conftest -- with this repository's
  configuration (-c), with the repository root as its working directory, or with --rootdir set to the child's folder (the real
  child tests pass --rootdir tmp_path). So the "configuration only" child tests keep their meaning.
- A child project's own root conftest.py loads with or without an ini file. I had read pytest's rootdir rules as saying it would
  not; the measurement said otherwise, and no ini file was added.
- The drift environment imports everything the guards need: workspace_snapshot (standard library and two package modules),
  joblib and joblib.parallel's registry.

## The change

- conftest.py at the repository root: tests/conftest.py's lines 133-327 and 411-854 moved by program, as exact slices, under a
  new header. The text changed only where the new place requires it: the root is the file's own folder (_DATA_CACHE,
  _WORKSPACE_ROOT), four messages name the root conftest (the phrases tests assert on are unchanged), and two stale comments
  were corrected. One behaviour changed (see the review below): the order of the autouse fixtures.
- tests/conftest.py keeps scripts/ on sys.path, the suite-size ratchet and the connector-cache isolation. Its note "DECLARED
  LAST, DELIBERATELY. It requests `monkeypatch`" contradicted the fixture's own docstring (it deliberately does not) and is
  replaced with what happened.
- Followed the move: the child projects of test_working_tree_guard.py and test_sklearn_parallel_warning_contract.py carry a byte
  copy of the root conftest; test_github_workflow_command_isolation.py parses the root conftest for the one set of publication
  variables; comments in four test files, .gitignore, pyproject.toml, workspace_snapshot.py's docstring, ci.yml, the README
  Quickstart and the drift suite's docstring name the new place. Dated records are unchanged.
- New tests: tests_drift/test_session_guards.py (2) and tests/unit/test_session_guard_placement.py (4), described in the CHANGELOG.
  Negative control: without a root conftest the two drift tests fail ("2 failed").
- The repository root is newly on sys.path in the drift run (pytest adds a root conftest's folder). Measured: no module resolves
  from the root outside src/ except the conftest and the drift test modules, and no tracked root name, nor any of the 81
  untracked root files on the owner's checkout, equals a top-level module the drift session imports.

## The independent review, before delivery

A separate review agent read the change and the base, and measured in its own copies. No defect; three risks and four wording
findings, each resolved:
1. RISK, measured: the autouse fixtures run in a different order -- the root's first (_isolate_github_workflow_commands,
   _no_data_dir_writes, _no_sys_path_leaks), then tests/conftest.py's (tmp_path, _isolate_connector_caches), then
   tests/unit/conftest.py's (monkeypatch, _isolate_spliceai). The guards now also enclose the connector-cache fixture's first
   import of the embedding and protein modules, which leaves sys.path unchanged and writes nothing. Recorded here, in the
   CHANGELOG and in THE INVARIANT's text.
2. WORDING, measured: THE INVARIANT no longer protects tests/conftest.py by mechanism (with the connector-cache fixture
   requesting `monkeypatch`: 7 passed, 2 errors in the old layout; 7 passed in the new). Its text now says where it binds -- the
   root conftest, where the workflow-variable fixture sorts ahead of the guards -- and why.
3. WORDING: a cross-reference to "_isolate_connector_caches below" and "five variables" (six; ci.yml said five as well, before
   this change) corrected.
4. WORDING: this section was empty when reviewed.
5. WORDING: "the drift test module" -- two modules (test_performance_estimator, test_session_guards); corrected.
6. RISK, measured: with pytest's base temporary folder inside the checkout, a child session loaded the repository's root conftest
   besides its own copy and stopped with an internal error. Every child pytest whose files live in its own folder (six calls in
   five files) now passes --confcutdir for that folder; measured: it then loads its own conftest and none above it.
7. RISK: the placement test compared function names in two files only, and the two per-test guards had no behavioural test. Added:
   a run-time check that each guard hook has exactly one conftest implementation (the root's) over every registered plugin, and a
   data/raw/cache write and a sys.path leak in the drift-layout child session (measured messages: "ENTRIES CHANGED under
   data/raw/cache: ['planted.bin']" and "ADDED  : ['/planted/by/a/drift/test']").
Found sound by the reviewer: the moved text against the base; the order of monkeypatch for tests/unit/conftest.py; no hook defined
twice; pytest-cov's wrapper outside ours; the module-name collision with tests/conformal/conftest.py; packaging, the Dockerfile,
the path budget and the workflow checks; the drift job as Continuous Integration runs it; the mutations (no root conftest; guards
back in tests/conftest.py; guards in both).

## Verification

- The full suite as Continuous Integration runs it (sandbox, Linux, Python 3.12, the training-and-test candidate environment of
  the Linux lock; `env -u` for the six publication variables, --assert-suite-size, coverage), on the code before the review's
  additions (8,646 collected): 8,614 passed, 26 skipped, 6 xfailed, 0 failed, no warnings, coverage 84%; the warning-filter guard
  8,646 checked, 0 changed; the working tree unchanged; 1,409.81 s.
- After the review's additions: 8,647 collected with --assert-suite-size, +4 / -0 by node identity against c98c5b92; the 13 files
  this change touches or that check its documents, 178 passed (guards: 0 changed, tree unchanged); the drift suite in the drift
  candidate environment of the Linux lock, 21 passed (21 checked, 0 changed; tree unchanged).
- The final commit (tree cbb835ff; this record's text apart), on a clean worktree, as Continuous Integration runs it: 8,615 passed,
  26 skipped, 6 xfailed, 0 failed, no warnings, coverage 84%; the warning-filter guard 8,647 checked, 0 changed; the working tree
  unchanged; 1,304.32 s. Then `pytest tests_drift/ -v -rs` as the drift job runs it (cache enabled) in the drift candidate
  environment: 21 passed, 21 checked, 0 changed, the working tree unchanged.

## Open, recorded

- WINDOWS-RSCRIPT-NOT-ON-PATH-1 (above: 28 tests on Windows need Rscript on PATH, 10 more the pinned DANDELION package or the
  fixture variables).
- The pull request's check logs (Python 3.11 and 3.12) for #63 are not in the evidence; the owner was asked for them.
- The repository root tracks 101 files, among them 15 Install_*.ps1 installers and 3 *_survey.txt outputs from earlier work;
  recorded for the scheduled legacy-roadmap carry-forward audit, not classified or changed here.
