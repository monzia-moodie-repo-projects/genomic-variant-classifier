# Session 2026-11-01 -- WARNINGS-AND-RESOURCES-1: unclosed resources fail their test; the package sets no warning filter (revision b, 2026-11-02: nannyml's usage telemetry off)

Author: Monzia Moodie

## Why now

Two measurements on 2026-11-01, during the dependency-profiles validation:

1. The owner's Windows validation of pull request #58 reported 1,000 copies of a scikit-learn message that pyproject.toml filters
   (33 warnings in every other measured run). scikit-learn 1.8.0's worker wrapper (sklearn/utils/parallel.py, read in full) runs
   `catch_warnings(); resetwarnings()` and refills the filters in every worker call, on the ONE process-wide list (Python earlier than
   3.14 has no per-context filters). Concurrent threads can therefore see the list empty: the warning fires and passes the filter.
   SKLEARN-WARNING-FILTER-RACE-1. The same mechanism makes ANY filter set inside the package govern every thread while it is active.
2. tests/unit/test_no_detritus.py failed under `-W error`: it read its own source without closing the file. Nothing in the suite
   reported unclosed resources (TEST-RESOURCEWARNING-CENSUS-1).

## Evidence, in order

1. Census, merged tree 4bcabb9f, Linux locked environment, `-W always::ResourceWarning` (2026-11-01, 19 min 17 s): 8,551 passed,
   26 skipped, 6 xfailed, 0 failed; 850 unclosed-resource warnings at 42 reported places: 779 text files opened for reading, 50 for
   writing, 17 binary readers, 1 binary writer, 3 child pipes (scripts/smoke_all_models._stream_child), 1 directory iterator
   (scripts/preflight_check_data_paths.py:51). 753 came from tests/unit/test_backup_artifacts.py:80. None came from a library
   -- in tests/; the drift tests were not part of the census (see Revision b).
2. The blanket filters: a probe plugin (outside the repository) neutralised `simplefilter("ignore")` in
   models/catboost_wrapper.py and models/kan.py, with a positive control showing a warning raised inside the blanket is recorded
   only with the probe. tests/unit/test_catboost.py and tests/unit/test_kan_actually_fits.py reported the same seven warnings either
   way (the pinned LightGBM message, shown because `-W always` overrides the pin). A forced pykan fit emitted none either way.
3. pykan 0.2.8 (kan/MultKAN.py): `auto_save=True, ckpt_path='./model'` by default; construction creates ./model and writes a
   checkpoint, and fit() appends ./model/history.txt and writes another (measured: model/0.0, model/0.1 after a three-step fit).
4. kan.py held `_fit_efficient_kan`, which uses `_EfficientKAN` -- defined nowhere -- and is never dispatched; its docstring listed
   fastkan and efficient-kan backends the module never imported. imodelsx's KAN modules are taken from efficient-kan
   (imodelsx/kan/kan_modules.py, line 1), so "imodelsx (efficient-kan based)" stays.
5. After the fixes, the whole suite with `-W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning` on the
   command line: 8,556 passed, 26 skipped, 6 xfailed, 0 failed, working tree unchanged. A planted leaking test fails under those
   filters and passes without them (both measured).
6. The final tree, its own pyproject.toml in force, the gate's command with coverage: 8,557 passed, 26 skipped, 6 xfailed, 0 failed,
   coverage 84%, working tree unchanged (Linux, locked environment, 2026-11-01).

## Open, recorded

- SILENT-FALLBACKS-1: models/catboost_wrapper.py's predict_proba retries on ANY exception with the raw array; shap_values returns
  ZEROS when CatBoost's computation fails (a warning only); models/kan.py substitutes pykan or scikit-learn's MLPClassifier when
  imodelsx cannot be imported, so a column labelled KAN can be another algorithm. Each changes behaviour and needs its own design.
- Pre-existing unused imports in the edited test files (os, shutil, RuntimePaths, LoadedState) and unused locals (manifest, strings)
  are left as they were; this change removed only the `io` imports it made unused.

## Revision b (2026-11-02): the drift job, and nannyml's telemetry

### What failed

Pull request #59 (this change, first revision, commit 52a8acb9, tree ec0be329) passed four of its six required checks on 2026-11-02
(both pytest jobs: 8,529 passed, 54 skipped, 6 xfailed, 0 failed; the lockfile check; the Docker smoke test). Both
"drift monitor (isolated env)" jobs failed: `pytest tests_drift/` gave 3 failed, 9 passed. Each of the three failures was an
ExceptionGroup of three PytestUnraisableExceptionWarning, each carrying a ResourceWarning for an unclosed `/proc/self/cgroup` opened
at nannyml/usage_logging.py:286.

### Why the validation did not see it

pyproject.toml's `filterwarnings` governs EVERY pytest run started from the repository root, including the drift job's
`pytest tests_drift/`, which runs in its own environment (requirements-drift.txt: nannyml, evidently, lightgbm 4.5.0). The census
and every validation of the first revision ran only `pytest tests/` in the training environment. "None came from a library" was
true of tests/ only. Rule adopted: a change to the test configuration is validated against every pytest invocation in ci.yml, each
in its own environment.

### Measured (2026-11-02, sandbox)

1. Two drift environments were built with exactly the 137 versions the two failed jobs installed (taken from their logs; `pip
   check` clean; no version differs). On the first revision both gave the job's result: 3 failed, 9 passed, with the same ROC AUC
   estimate (0.5336) in the alert test.
2. On main (c1e1555c), with `-W always::ResourceWarning`: 12 passed and 9 ResourceWarnings from usage_logging.py:286. The leak
   was there before this change; the default filter hid it.
3. Read in nannyml 0.13.1 (wheel cf5f4a33...): every CBPE fit and estimate is decorated by `log_usage`. Unless the environment
   holds NML_DISABLE_USAGE_LOGGING (any value), `SegmentUsageTracker._log` sends a Segment analytics event whose user id is
   `uuid.UUID(int=uuid.getnode())` -- derived from the machine's network hardware address -- with the operating system, Python
   version and runtime environment; finding the runtime environment opens /proc/self/cgroup without closing it. One CBPE fit and
   estimate made three `segment.analytics.track` calls (measured by recording them; nothing was sent).
4. Read in nannyml 0.13.1 and python-dotenv 1.2.4: `import nannyml` calls `load_dotenv()` twice (usage_logging.py and
   __init__.py). With no path, `find_dotenv()` searches upward from nannyml's installed directory (from the current directory in an
   interactive session) and loads the first `.env` it finds, without overriding existing variables. A `.venv-drift` inside the
   checkout puts the checkout's own `.env` on that path. python-dotenv honours PYTHON_DOTENV_DISABLED from 1.2.0 (read in the
   1.1.1, 1.2.0 and 1.2.1 wheels: absent, present, present).
5. evidently 0.7.6's telemetry (iterative_telemetry, DO_NOT_TRACK) is used only by its UI service and collector; the project uses
   `Report` only (every `event_logger` use read).
6. The revised tree, the gate's command with coverage, the Linux locked environment (linux-3.12 SATISFIED): 8,557 passed, 26
   skipped, 6 xfailed, 0 failed, coverage 84%, working tree unchanged -- the same counts as the first revision (tests/ does not
   import the drift module). GitHub's pytest jobs report 54 skipped where the sandbox reports 26: the 28 more are exactly the tests
   that need Rscript (25 environment-artifact inspector, 2 environment qualification, 1 inference ranking), which the runner lacks.

### The change

monitoring/performance_estimator.py `_import_nannyml()`:

- sets NML_DISABLE_USAGE_LOGGING=1 for the process when absent (a value already present is kept; presence is what nannyml reads),
  logged once at INFO. The policy is the project's, not a test's: a clinical-genomics monitor does not report its use to a vendor,
  so it applies to every caller (scripts/run_drift_monitor.py, the monthly drift workflow, notebooks);
- imports nannyml with PYTHON_DOTENV_DISABLED=1 and restores that variable exactly afterwards (absent stays absent; a value
  stays), because the project loads its own `.env` explicitly, by path (utils/helpers.py), and python-dotenv's switch would
  otherwise disable that too;
- reports, with a WARNING, a python-dotenv older than 1.2.0, which cannot honour the switch.

tests_drift/test_performance_estimator.py, seven tests: a CBPE run makes no Segment call and leaves the switch set (with the switch
removed first, so the result is the estimator's own doing); the negative control -- nannyml's logger without the switch DOES call
Segment (system-information probe and send both replaced, so nothing leaks or leaves); a value already in the switch is kept; in a
fresh interpreter whose current directory holds a planted `.env`, importing through the estimator calls `find_dotenv` zero times and
loads nothing, with the dotenv switch absent afterwards, or restored to a preset "0"; the negative control -- a plain
`import nannyml` calls it twice and loads the planted file; an old python-dotenv is reported.

Each part is proven necessary by mutation (2026-11-02, drift environment 3.12): without the telemetry switch, the three CBPE tests
and the telemetry test fail; without the dotenv switch, or without the restore, the two import tests fail; overwriting a present
value fails its test; without the warning, the old-dotenv test fails.

### Open, recorded

- DRIFT-ENVIRONMENT-NOT-LOCKED: requirements-drift.txt pins seven packages; everything else resolves on the day (the two jobs
  resolved scipy 1.18.1 on 3.12 and 1.17.1 on 3.11, pytest 9.2.0). One of the 16 install paths recorded under DEPENDENCY-PROFILES-1.
