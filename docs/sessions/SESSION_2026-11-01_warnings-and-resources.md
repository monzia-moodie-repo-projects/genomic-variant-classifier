# Session 2026-11-01 -- WARNINGS-AND-RESOURCES-1: unclosed resources fail their test; the package sets no warning filter

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
   (scripts/preflight_check_data_paths.py:51). 753 came from tests/unit/test_backup_artifacts.py:80. None came from a library.
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
