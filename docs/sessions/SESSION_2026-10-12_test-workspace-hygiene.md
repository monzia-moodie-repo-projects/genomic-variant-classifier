# SESSION 2026-10-12 -- test workspace hygiene: what the suite left behind, and the guard that keeps it from returning

**Author: Monzia Moodie**
**Base:** tree e72a9092 (main after the run9a-prep record's merge; measured at 8a65002d, which has that tree)

## 1. The owner-side events this unit answers (2026-10-11)

- **Two windows ran set m's ordinary file at the same time.** One produced the published candidate 8a65002d; the other produced a
  duplicate, 8e038f74, which has the same tree and was never published. The merge on GitHub was skipped, and nothing broke:
  GitHub main stayed at ffc549f4.
- **Set n's ordinary file was pasted whole into one window.** All five blocks refused, and nothing changed. The refusals exposed
  four defects in the owner blocks:
  - R1 demanded an empty untracked list, but the owner's checkout holds 81 files of his own.
  - H3 refused the clones' git-ignored test leftovers.
  - G1 printed a path in place of a blob identifier, because `git rev-parse rev:path` echoes its argument when the path is absent.
  - No block wrote its record when it refused.

  Set o corrects all four and puts one block in each file.
- **H3 named what the clones held:** `models/best_gat.pt` and `src/genomic_variant_classifier/**/__pycache__/*.pyc`. These are the
  starting point of this unit.
- **T3's census of TEMP:** 973 `tmp*` folders, all owned by Administrators, created by the test runs of 2026-10-09 and 2026-10-10.

## 2. Measured: what one full run leaves behind

- **Environment.** The development sandbox, with the owner's exact package versions, read from his validation's `environment.log`
  (248 packages), plus the Linux-only runtime packages PyTorch needs. It collects 8,507, the same as his machine.
  - FINDING: his environment is not the lockfile's. There are 25 version differences and 75 packages beyond the lock, and six lock
    packages are absent: evidently, fairlearn, gcsfs, great-expectations, nannyml and s3fs.
  - `pyfaidx` is in `requirements-dev.txt` but not in `requirements-dev.in`, so it is not in the lock.
- **Settings.** The run used the installer's full-suite settings:
  - `PYTHONPATH` set to the checkout's `src`;
  - `PYTHONDONTWRITEBYTECODE=1`;
  - its own temporary root;
  - `--basetemp`.
- **Before (8a65002d).** The run left, in the checkout:
  - `models/best_gat.pt`;
  - 14 `.pyc` files, among them the five H3 named;
  - `renv/library/R-4.3/x86_64-pc-linux-gnu`.

  In the temporary root, outside pytest's folders, it left 173 `tmp*` folders, three generated PyTorch Geometric modules and
  `torchinductor_root`. Test outcomes: 8 failed, all of them the git 2.43 `--no-lazy-fetch` tests (see the CORRECTION of
  2026-10-11).
- **Causes, by reading the code.**
  - `GNNTrainer.__init__` created `checkpoint_path.parent` at construction. Its default path, `models/best_gat.pt`, is relative to
    the current directory, and `train_gnn_pipeline` passed no path. Eight training tests in four files therefore wrote into the
    checkout.
  - `tests/unit/test_bootstrap_reconciliation.py` starts two child interpreters with a fresh environment (`{"PATH": ...,
    "PYTHONPATH": "src"}`), so `PYTHONDONTWRITEBYTECODE` does not reach them, and their working directory is the checkout.
  - `tests/unit/test_inference_backend_trace.py` runs `Rscript -e` in the checkout at collection time. The project's `.Rprofile`
    activates renv, and renv creates its library.
  - The 72 `tempfile.mkdtemp()` calls in nine test files are never cleaned up.
- **Why 2026-09-23 missed it.** That verification ran without PyTorch (see the correction record beside it).

## 3. Changed

- `models/gnn.py`:
  - the checkpoint folder is created in `fit`, immediately before `torch.save`, not at construction;
  - `train_gnn_pipeline(..., checkpoint_path="models/best_gat.pt")` passes the path through (the default is unchanged);
  - the eight training tests pass `tmp_path`.
- `tests/unit/test_bootstrap_reconciliation.py`: both child interpreters run with `-B`.
- `tests/conftest.py`, temporary-file containment:
  - at `pytest_configure` (trylast, after pytest's own temporary-path factory exists), `tempfile.tempdir`, `TMPDIR`, `TEMP` and
    `TMP` point at `basetemp/tempfile`;
  - everything is restored first at `pytest_unconfigure`;
  - it is saved and restored by hand, never with `monkeypatch`.

  A first version used a session fixture. The measured full run then still left `torchinductor_root` in the outer temporary
  folder: it was created during collection, before any fixture exists.
- `tests/conftest.py`, working-tree guard, backed by `repository_hygiene/workspace_snapshot.py` (new):
  - `pytest_sessionstart` snapshots the whole working tree;
  - the innermost `pytest_runtestloop` wrapper snapshots it again when the test loop ends (section 5 says why not at session
    finish), and `pytest_sessionfinish` compares; anything added, removed or changed is named and the session fails (exit 1), even
    when every test passed;
  - its section is printed in the terminal summary, and on standard error when no summary is printed;
  - the snapshot never follows a link (`artifact_inventory.is_redirected`) and does not enter
    `backup_artifacts.NOT_THIS_REPOSITORY`;
  - a directory it cannot list is reported, never read as empty.
- Tests: +30, so the suite goes from 8,507 to 8,537:
  - `test_workspace_snapshot.py`, 17;
  - `test_working_tree_guard.py`, 10, among them child sessions that observe the guard's exit code and message, containment at
    import time and in a collection-only run, and (section 5) the gate's coverage options;
  - `test_gnn_optim.py`, 3.
- Seeded defects, all caught:
  - the old `gnn.py`: 3 of the new tests fail;
  - the old GNN test file under the new conftest: the guard fails the session and names `models/best_gat.pt`;
  - containment removed: 4 tests fail;
  - the restoration removed: 1 fails;
  - the exit code not set: 2 fail;
  - the after-snapshot's wrapper not the innermost: 2 fail;
  - the comparison moved back to session finish: 3 fail;
  - the standard-error fallback removed: 2 fail with pytest 9.0.3, 1 with pytest 9.2.0.

## 4. Open, recorded

- **TEST-R-PROJECT-PROFILE-1.** R started from the checkout activates the project's renv, and which library `requireNamespace` sees
  depends on the working directory. `renv/` is declared not this repository's content, so the guard does not enter it. A change
  here belongs to the R environment-qualification work.
- **NOT-THIS-REPOSITORY-DECLARATIONS-1.** Three declarations of what is not this repository's content disagree:
  - `NOT_THIS_REPOSITORY` lacks `.venv`, `.venv-drift` and `.ruff_cache`;
  - the installers' protected inventory excludes them;
  - `deps/dependency_census.py` has its own `_SKIP_DIRS`.
- **OWNER-ENVIRONMENT-VERSUS-LOCK-1.** The finding in section 2: the owner's environment is not the lockfile's.
- **CI-INSTALLS-DEV-RANGES-1.** The gate's pytest job installs `requirements-api.lock`, then `requirements.txt` and
  `requirements-dev.txt`, which are ranges, not `requirements-dev.lock`. On 2026-10-31 it ran pytest 9.2.0, pluggy 1.7.0 and
  coverage 7.17.0; the lock pins pytest 9.0.3 and coverage 7.13.5. Section 5 shows a behaviour difference this caused.

## 5. The first Continuous Integration run, and the correction (2026-10-31)

- **What happened.** The first installer validated this change on the owner's machine and published it as the branch
  `test-workspace-hygiene-2026-10-12` (commit c3b112d2, this tree's first revision). Its pull request, #56, failed the gate:
  `pytest (3.11)` and `pytest (3.12)` were red after 13 and 14 minutes, so the Docker smoke test was skipped. This revision
  replaces it, on the branch `test-workspace-hygiene-2026-10-31`; #56 is not merged.
- **The gate's logs** (both jobs' raw logs, saved by the owner): 1 failed, 8,472 passed, 54 skipped, 6 xfailed on each Python
  version, and the guard reported `ADDED (2): .coverage, coverage.xml`. Two causes.
- **First cause, reproduced.** In a fresh clone of c3b112d2 the gate's options were run on one test file, with the owner's package
  versions: `-q -rs --tb=short --cov=genomic_variant_classifier --cov-report=term-missing --cov-report=xml:coverage.xml`.
  - Every test passed (16 passed, 1 skipped), yet the session exited 1.
  - The guard reported `ADDED (2): .coverage, coverage.xml`.
- **Cause.** pytest-cov's `pytest_runtestloop` wrapper (pytest-cov 7.1.0, `plugin.py`) writes the coverage data file and every
  `--cov-report` file after the test loop and before `pytest_sessionfinish`, which is where the guard compared. Those files are
  outputs the invocation asked for, not something a test left.
- **Second cause, reproduced.** The failed test was `test_without_a_terminal_summary_the_finding_still_reaches_the_console`. It
  required the finding on standard error under `--no-summary`, but the gate's log shows the finding in the summary instead.
  - The gate installs pytest 9.2.0 (see CI-INSTALLS-DEV-RANGES-1). Its `terminal.py` says "Always invoke pytest_terminal_summary so
    third-party plugins can report", even under `--no-summary`.
  - pytest 9.0.3, the lock's version, skips that hook under `--no-summary`, so the guard writes the finding to standard error at
    unconfigure instead.
  - With pytest 9.2.0, pluggy 1.7.0 and coverage 7.17.0 placed over the owner's environment, the test failed exactly as on the gate.
  - The guard was right; the test assumed one pytest version.
- **Why the validation did not see it.** The installer's full-suite stage ran the suite without the gate's coverage options, and
  with the owner's pytest, not the gate's. A validation must run every option of the gate's command that changes what runs or
  what is written; the gate's tool versions are a separate difference, recorded as CI-INSTALLS-DEV-RANGES-1. The second installer does,
  and moves the two coverage files out of its clone, recorded, before its own clone comparison.
- **Changed.** The guard takes its second snapshot when the test loop ends, in the innermost `pytest_runtestloop` wrapper. The last
  teardown, every session-scoped fixture included, has run by then, and no other plugin has resumed yet. No file name is
  excused. A loop that stops early says so. Three child-session tests cover the gate's coverage options:
  - a clean session passes, and `.coverage` and `coverage.xml` exist afterwards;
  - a test that writes `coverage.xml` and `models/best_gat.pt` is named for both, and `.coverage` is not;
  - `-x` reports the early stop.

  The `--no-summary` test now requires the finding on the console exactly once, on any pytest version. A new test runs without
  the terminal plugin (`-p no:terminal`), where the standard-error fallback is the only way the finding is seen.
- **Measured.**
  - Under the gate's coverage options, the guard tests and the snapshot tests passed (the Windows junction test skipped), exit 0,
    guard unchanged.
  - The 10 guard tests pass with the owner's pytest 9.0.3 and with the gate's pytest 9.2.0.
  - Seeded defects, all caught:
    - the wrapper not the innermost: 2 tests fail;
    - the comparison moved back to session finish: 3 fail;
    - the standard-error fallback removed: 2 fail with pytest 9.0.3, 1 with 9.2.0.
