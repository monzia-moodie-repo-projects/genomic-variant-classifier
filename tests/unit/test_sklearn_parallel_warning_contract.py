"""SKLEARN-WARNING-FILTER-RACE-1: no thread-parallel scikit-learn in the test session, a guard on the warning filters, and
scikit-learn's parallel-pairing warning as an error. Rewritten 2026-11-03.

HISTORY
=======
2026-07-22: pyproject.toml gained an IGNORE entry for scikit-learn's UserWarning

    `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to
    propagate the scikit-learn configuration of the current thread to the joblib workers.

and this file pinned its premise: a harmless thread-configuration note from scikit-learn's own estimators. That premise was
wrong. 2026-11-01: 1,000 copies passed the ignore entry on pull request #58's validation; scikit-learn 1.8.0
(sklearn/utils/parallel.py, read in full) runs every task of a thread-parallel estimator inside `catch_warnings();
resetwarnings()` and refills the ONE process-wide filter list, so concurrent tasks rebind and can empty it.

MEASURED 2026-11-03 (the repository root's conftest.py carries the full account): inside pytest, 20 of 20 thread-parallel RandomForest fits
left the process bound to a different filter list with one entry changed in meaning; on the owner's machine 448 copies of the
warning reached pytest (2026-11-03 06:18 UTC). While a list is empty, pyproject.toml's `error::` entries do not apply.

WHAT IS PINNED HERE
===================
* PREVENTION: joblib's "threading" backend is its sequential one for the session -- every joblib thread request that names
  its backend goes through that name -- so a forest at n_jobs=-1, a histogram gradient-boosting fit (an explicit
  backend="threading") and pairwise distances run every task in the main thread and leave the filters exactly as they were;
  joblib's process backends are unchanged (in this session);
* DETECTION: the root conftest.py fails the teardown of any test that leaves the warning state changed -- an entry added or
  moved ahead included, since 2026-11-05 -- restores it, fails the SESSION as well (an xfail-marked test's teardown error
  is reported as an xfail), never replaces an interrupt, and passes the patterns that restore themselves (child sessions running the
  root conftest.py with its list of filter-registering modules replaced by CHILD_LISTED -- its child_conftest fixture -- in a
  scrubbed environment);
* scikit-learn's two pairing warnings are an ERROR entry in pyproject.toml, selected by their module: through that
  configuration a scikit-learn `delayed` function run by joblib's own Parallel fails, a joblib `delayed` function run by
  scikit-learn's Parallel fails, and the matching pair passes (a child session reading pyproject.toml only);
* in the package, `delayed` and `Parallel` come from joblib only, and every module using one uses both (parsed; a file that
  cannot be parsed fails the test rather than being skipped).

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import threading
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "genomic_variant_classifier"
PYPROJECT = REPO / "pyproject.toml"
CONFTEST = REPO / "conftest.py"
PAIRING_ENTRY = r"error::UserWarning:sklearn\.utils\.parallel"


# --------------------------------------------------------------------------- prevention, observed in this session
def test_the_sessions_thread_backend_is_the_sequential_one_and_the_process_backends_are_joblibs():
    import joblib
    from joblib.parallel import BACKENDS, get_active_backend

    assert BACKENDS["threading"] is BACKENDS["sequential"]
    for hints in ({"prefer": "threads"}, {"require": "sharedmem"}, {"prefer": "threads", "require": "sharedmem"}):
        backend, _ = get_active_backend(**hints)
        assert isinstance(backend, BACKENDS["sequential"]), (hints, type(backend))
    with joblib.parallel_config(backend="threading"):
        assert isinstance(get_active_backend()[0], BACKENDS["sequential"])
    # Untouched: a worker process has its own warning filters (scalable_svm's bags run on loky, as in production). Every other
    # name keeps joblib's own class; the default backend is joblib's default (loky where processes are available).
    for name, factory in BACKENDS.items():
        if name != "threading":
            assert factory.__module__ == "joblib._parallel_backends", (name, factory)
            assert (factory is BACKENDS["sequential"]) == (name == "sequential"), name
    for name, class_name in (("loky", "LokyBackend"), ("multiprocessing", "MultiprocessingBackend")):
        if name in BACKENDS:                       # absent only where joblib finds no process support
            assert BACKENDS[name].__name__ == class_name, (name, BACKENDS[name])
    assert isinstance(get_active_backend()[0], BACKENDS[joblib.parallel.DEFAULT_BACKEND])


def test_every_thread_route_runs_its_tasks_in_the_main_thread_and_leaves_the_filters_alone(monkeypatch):
    """A forest (prefer="threads" and require="sharedmem"), a histogram gradient-boosting fit (an EXPLICIT backend="threading",
    sklearn/ensemble/_hist_gradient_boosting/binning.py) and pairwise distances (backend="threading" again). Measured with joblib's
    registry as shipped (2026-11-03): all 40 tasks in worker threads, the filter list rebound and changed. Here: none, neither."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import pairwise_distances
    from sklearn.utils import parallel as sklearn_parallel

    threads = []
    call = sklearn_parallel._FuncWrapper.__call__

    def recorded(self, *args, **kwargs):
        threads.append(threading.get_ident())
        return call(self, *args, **kwargs)

    monkeypatch.setattr(sklearn_parallel._FuncWrapper, "__call__", recorded)
    X = np.random.default_rng(0).normal(size=(400, 6))
    y = (X[:, 0] > 0).astype(int)
    bound, entries = warnings.filters, tuple(warnings.filters)
    RandomForestClassifier(n_estimators=16, n_jobs=-1, random_state=0).fit(X, y).predict_proba(X)
    after_forest = len(threads)
    HistGradientBoostingClassifier(max_iter=5).fit(X, y)
    after_histogram = len(threads)
    pairwise_distances(X, n_jobs=2)
    assert after_forest == 32 and after_histogram > after_forest and len(threads) > after_histogram, threads  # every route ran
    assert set(threads) == {threading.main_thread().ident}, threads
    assert warnings.filters is bound and tuple(warnings.filters) == entries


# --------------------------------------------------------------------------- the configuration's error entry
def _filterwarnings() -> list:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]["filterwarnings"]


def test_the_pairing_warnings_are_one_error_entry_by_module_and_nothing_ignores_them():
    entries = _filterwarnings()
    assert entries.count(PAIRING_ENTRY) == 1, entries
    naming = [e for e in entries if "sklearn" in e and "parallel" in e]
    assert naming == [PAIRING_ENTRY], naming


PAIRING_TESTS = '''
import joblib
from sklearn.utils.parallel import Parallel, delayed


def test_scikit_learn_delayed_run_by_joblibs_own_parallel():
    joblib.Parallel(n_jobs=1)(delayed(abs)(-1) for _ in range(2))


def test_joblib_delayed_run_by_scikit_learns_parallel():
    Parallel(n_jobs=1)(joblib.delayed(abs)(-1) for _ in range(2))


def test_scikit_learn_delayed_run_by_scikit_learns_parallel():
    assert Parallel(n_jobs=1)(delayed(abs)(-1) for _ in range(2)) == [1, 1]
'''


def test_through_the_configuration_both_mismatches_fail_and_the_matching_pair_passes(tmp_path, child_pytest_environment):
    """Run in a child pytest that reads THIS repository's configuration (-c) but not its conftest, so the verdict is the
    configuration's alone. Both mismatches and the matching pair, so no result is vacuous."""
    (tmp_path / "test_planted.py").write_text(PAIRING_TESTS, encoding="utf-8")
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", str(PYPROJECT),
                           "--rootdir", str(tmp_path), "--confcutdir", str(tmp_path), str(tmp_path / "test_planted.py")],
                          cwd=str(tmp_path), env=child_pytest_environment(), capture_output=True, text=True, timeout=300)
    out = done.stdout + done.stderr
    assert done.returncode == 1, out
    assert "2 failed, 1 passed" in out, out
    assert "FAILED test_planted.py::test_scikit_learn_delayed_run_by_joblibs_own_parallel" in out, out
    assert "FAILED test_planted.py::test_joblib_delayed_run_by_scikit_learns_parallel" in out, out
    assert "UserWarning: `sklearn.utils.parallel.delayed` should be used with" in out, out
    assert "UserWarning: `sklearn.utils.parallel.Parallel` needs to be used in conjunction with" in out, out


# --------------------------------------------------------------------------- the guard, observed from outside
GUARD_TESTS = '''
import threading
import warnings

import joblib
import numpy as np
import pytest
from joblib.parallel import BACKENDS, get_active_backend
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

X = np.random.default_rng(0).normal(size=(120, 4))
Y = (X[:, 0] > 0).astype(int)


@pytest.fixture(scope="module")
def forest():
    # set up BEFORE the function-scoped autouse fixtures: prevention must already hold here
    assert BACKENDS["threading"] is BACKENDS["sequential"]
    return RandomForestClassifier(n_estimators=8, n_jobs=-1, random_state=0).fit(X, Y)


def test_a_module_scoped_forest_a_forest_and_an_explicit_thread_backend_in_the_test(forest):
    forest.predict_proba(X)
    RandomForestClassifier(n_estimators=8, n_jobs=-1, random_state=1).fit(X, Y).predict_proba(X)
    HistGradientBoostingClassifier(max_iter=3).fit(X, Y)        # backend="threading", explicitly, in its binning


def test_patterns_that_restore_themselves(recwarn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
    with pytest.warns(UserWarning, match="planted"):
        warnings.warn("planted", UserWarning)
    warnings.warn("recorded", UserWarning)
    assert [str(w.message) for w in recwarn] == ["recorded"]


def test_rebinds_the_list():
    warnings.filters = list(warnings.filters)


def test_empties_the_list():
    warnings.filters.clear()


def test_replaces_showwarning():
    warnings.showwarning = lambda *args, **kwargs: None


def test_registers_joblibs_thread_backend_again():
    from joblib._parallel_backends import ThreadingBackend
    joblib.register_parallel_backend("threading", ThreadingBackend)


@pytest.mark.xfail(reason="planted: an expected failure whose teardown finds the state changed", strict=True)
def test_an_xfail_marked_offender():
    warnings.filters = list(warnings.filters)
    raise AssertionError("the expected failure")


def test_after_the_offenders_the_state_is_whole():
    assert BACKENDS["threading"] is BACKENDS["sequential"]
    assert isinstance(get_active_backend(prefer="threads", require="sharedmem")[0], BACKENDS["sequential"])
    assert ("error", None, ResourceWarning, None, 0) in warnings.filters, warnings.filters    # the child's own entry is back
    assert warnings.showwarning is warnings._showwarning_orig
'''

# The child's own configuration: one error entry that exists on every pytest version (pytest 9.0.3 adds an error entry of its
# own for PytestRemovedIn9Warning; 9.2.0 does not -- measured by the independent review, 2026-11-03).
CHILD_INI = "[pytest]\nfilterwarnings =\n    error::ResourceWarning\n"

# The child's list of filter-registering modules (the root conftest.py imports them at session start): GUARD_TESTS imports
# scikit-learn at collection, and through it scipy.sparse and scipy.special, which register filters when first imported. MEASURED
# 2026-11-04: with an empty list the child stops at collection, naming their two entries (SpecialFunctionWarning; "the matrix
# subclass is not the recommended way").
CHILD_LISTED = ("scipy.sparse", "scipy.special")


def _child(tmp_path: Path, environment, conftest, tests: str, plugin: str = "") -> subprocess.CompletedProcess:
    """A small project whose root conftest.py is the real one with its list of filter-registering modules replaced by CHILD_LISTED
    (the root conftest's child_conftest fixture, the one difference); `plugin`, if given, is loaded with -p before the conftest, from a folder
    OUTSIDE the project."""
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "pytest.ini").write_text(CHILD_INI, encoding="utf-8")
    (project / "conftest.py").write_bytes(conftest(*CHILD_LISTED))
    (project / "tests" / "test_child.py").write_text(tests, encoding="utf-8")
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "child_probe.py").write_text(plugin, encoding="utf-8")
    env = environment(PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(plugins)] +
                                                        ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
                             CHILD_PROBE_OUT=str(tmp_path / "probe.txt"))
    return subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-p", "child_probe",
                           "--confcutdir", str(project)],
                          cwd=str(project), env=env, capture_output=True, text=True, timeout=600)


def _dropped_by_the_conftest() -> tuple:
    for node in ast.parse(CONFTEST.read_text(encoding="utf-8"), filename=str(CONFTEST)).body:
        if isinstance(node, ast.Assign) and [getattr(x, "id", "") for x in node.targets] == ["CHILD_PYTEST_ENVIRONMENT_DROPPED"]:
            return ast.literal_eval(node.value)
    raise AssertionError("conftest.py (the repository root's) no longer defines CHILD_PYTEST_ENVIRONMENT_DROPPED")


def test_a_child_pytest_environment_drops_what_changes_its_console_and_keeps_the_rest(child_pytest_environment, monkeypatch):
    dropped = _dropped_by_the_conftest()
    # The measured ones must stay in it: CI and BUILD_NUMBER (truncation), COLUMNS (width), the colour switches.
    assert {"CI", "BUILD_NUMBER", "COLUMNS", "PY_COLORS", "FORCE_COLOR", "NO_COLOR"} <= set(dropped), dropped
    for name in dropped:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("GVC_CHILD_ENVIRONMENT_KEPT", "kept")
    env = child_pytest_environment(GVC_CHILD_ENVIRONMENT_EXTRA="given")
    assert not [name for name in dropped if name in env], env
    assert env["GVC_CHILD_ENVIRONMENT_KEPT"] == "kept" and env["GVC_CHILD_ENVIRONMENT_EXTRA"] == "given"


DISABLED_SETTING_PLUGIN = '''
import joblib

joblib.register_parallel_backend = lambda *args, **kwargs: None     # imported BEFORE the conftest: its mapping now does nothing
'''

RESTORE_PROBE_PLUGIN = '''
import os

import pytest


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    from joblib.parallel import BACKENDS, get_active_backend
    with open(os.environ["CHILD_PROBE_OUT"], "w", encoding="utf-8") as fh:
        fh.write(BACKENDS["threading"].__name__ + " " + type(get_active_backend(prefer="threads", require="sharedmem")[0]).__name__)
'''


def test_a_session_whose_setting_has_no_effect_refuses_to_run(tmp_path, child_pytest_environment, child_conftest):
    run = _child(tmp_path, child_pytest_environment, child_conftest, "def test_never_runs():\n    pass\n", DISABLED_SETTING_PLUGIN)
    out = run.stdout + run.stderr
    assert run.returncode == 4, out                                                  # pytest.ExitCode.USAGE_ERROR
    assert "joblib does not resolve thread requests to it" in out and "the suite does not run without this" in out, out
    assert "passed" not in out, out


def test_the_session_registers_joblibs_own_thread_backend_again_when_it_ends(tmp_path, child_pytest_environment, child_conftest):
    run = _child(tmp_path, child_pytest_environment, child_conftest, "def test_runs():\n    pass\n", RESTORE_PROBE_PLUGIN)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path / "probe.txt").read_text(encoding="utf-8") == "ThreadingBackend ThreadingBackend"   # joblib's own


def test_the_guard_fails_exactly_the_offenders_and_the_session_and_restores_the_state(tmp_path, child_pytest_environment,
                                                                                      child_conftest):
    run = _child(tmp_path, child_pytest_environment, child_conftest, GUARD_TESTS)
    out = run.stdout + run.stderr
    assert run.returncode == 1, out
    # Each offender's call passes and its TEARDOWN errors; every other test passes. The xfail-marked offender is reported as
    # "xfailed" twice -- its call and its teardown (measured 2026-11-03) -- which is why the session fails as well.
    assert "7 passed, 2 xfailed, 4 errors" in out, out
    offenders = ("test_rebinds_the_list", "test_empties_the_list", "test_replaces_showwarning",
                 "test_registers_joblibs_thread_backend_again")
    for name in offenders:
        assert " ERROR at teardown of {} ".format(name) in out and "ERROR tests/test_child.py::{} ".format(name) in out, (name, out)
    assert out.count(" ERROR at teardown of ") == 4 and out.count("ERROR tests/test_child.py::") == 4, out
    # Counted on the traceback lines, which no terminal width or CI setting shortens.
    assert len(re.findall(r"(?m)^E\s+AssertionError: THE WARNING FILTERS AFTER THIS TEST'S TEARDOWN", out)) == 4, out
    assert "warnings.filters is a DIFFERENT list from the one bound before setup" in out, out
    assert "the filter entries changed: " in out and "0 new" in out, out
    assert "    gone: error  message=None  category=ResourceWarning  module=None  line=0" in out, out
    assert "warnings.showwarning was replaced" in out, out
    assert "joblib's 'threading' backend no longer maps to its sequential one" in out, out
    assert "The state has been restored." in out, out
    assert "8 tests checked, 5 changed the warning state" in out, out
    named = out[out.index("THE WARNING STATE CHANGED IN THESE TESTS"):].split("\n")[1:6]
    assert [n.strip() for n in named] == ["tests/test_child.py::" + n for n in offenders + ("test_an_xfail_marked_offender",)], named


INTERRUPT_TESTS = '''
import warnings

import pytest


@pytest.fixture
def interrupts_in_its_teardown():
    yield
    warnings.filters = list(warnings.filters)
    raise KeyboardInterrupt


def test_whose_teardown_is_interrupted(interrupts_in_its_teardown):
    pass


def test_never_reached():
    pass
'''


def test_an_interrupt_during_teardown_is_never_replaced_and_the_change_is_still_named(tmp_path, child_pytest_environment,
                                                                                      child_conftest):
    run = _child(tmp_path, child_pytest_environment, child_conftest, INTERRUPT_TESTS)
    out = run.stdout + run.stderr
    assert run.returncode == 2, out                                                  # pytest.ExitCode.INTERRUPTED
    assert "!! KeyboardInterrupt !!" in out and "1 passed" in out and "test_never_reached" not in out, out
    assert "THE WARNING FILTERS AFTER" not in out, out                                # the interrupt was not replaced
    assert "1 tests checked, 1 changed the warning state" in out, out
    assert "    tests/test_child.py::test_whose_teardown_is_interrupted" in out, out


ADDING_TESTS = '''
import warnings


def test_adds_an_entry_the_way_a_library_does_at_import():
    warnings.filterwarnings("ignore", message="planted by an import", category=UserWarning)


def test_moves_an_entry_ahead_of_the_others():
    action, _, category, _, _ = [f for f in warnings.filters if f[1] is None and f[3] is None and f[4] == 0][-1]
    assert warnings.filters.index((action, None, category, None, 0)) > 0
    warnings.filterwarnings(action, category=category)        # an identical entry is MOVED to the front, not added
'''


def test_an_added_entry_and_a_moved_one_both_fail_each_named(tmp_path, child_pytest_environment, child_conftest):
    """Until 2026-11-05 an added entry was listed and only a moved one failed; both put an entry ahead of the configuration's for
    the rest of the test (WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1)."""
    run = _child(tmp_path, child_pytest_environment, child_conftest, ADDING_TESTS)
    out = run.stdout + run.stderr
    assert run.returncode == 1, out
    assert "2 passed, 2 errors" in out, out
    assert out.count(" ERROR at teardown of ") == 2, out
    for name in ("test_adds_an_entry_the_way_a_library_does_at_import", "test_moves_an_entry_ahead_of_the_others"):
        assert " ERROR at teardown of {} ".format(name) in out, (name, out)
    assert "1 filter entry was registered" in out, out
    assert "    new:  ignore  message='planted by an import'  category=UserWarning  module=None  line=0" in out, out
    assert "no longer in their original order (order is precedence)" in out, out
    assert "    moved: " in out and " (now ahead of " in out, out
    assert "2 tests checked, 2 changed the warning state" in out and "ADDED during a test" not in out, out


# --------------------------------------------------------------------------- the package's use of joblib, parsed
SCIKIT_LEARN_PARALLEL_MODULES = ("sklearn.utils.parallel", "sklearn.utils.fixes")


def _imports(path: Path) -> list:
    """Every import in the module as (module, name): `from m import n` gives (m, n); `import m` gives (m, ""). Parsed; a file that
    cannot be parsed raises here rather than being skipped."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found += [(node.module or "", alias.name) for alias in node.names]
        elif isinstance(node, ast.Import):
            found += [(alias.name, "") for alias in node.names]
    return found


def _joblib_findings(path: Path) -> tuple:
    imports = _imports(path)
    shown = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.name
    wrong = ["{}: {}{}".format(shown, module, " " + name if name else "") for module, name in imports
             if module.startswith(SCIKIT_LEARN_PARALLEL_MODULES)]
    names = {name: module for module, name in imports if name in ("delayed", "Parallel")}
    wrong += ["{}: {} from {}".format(shown, name, module) for name, module in names.items() if module != "joblib"]
    unpaired = ["{}: {}".format(shown, sorted(names))] if names and set(names) != {"delayed", "Parallel"} else []
    return wrong, unpaired, bool(names)


def test_the_package_takes_delayed_and_parallel_from_joblib_and_uses_them_together():
    files = sorted(SRC.rglob("*.py"))
    assert len(files) > 100, len(files)                                             # the scan reached the package
    wrong, unpaired, users = [], [], []
    for path in files:
        w, u, uses = _joblib_findings(path)
        wrong += w
        unpaired += u
        if uses:
            users.append(path.relative_to(REPO).as_posix())
    assert not wrong and not unpaired, (wrong, unpaired)
    # Measured 2026-11-03: the one user. A second is a change worth seeing here, not a silent pass.
    assert users == ["src/genomic_variant_classifier/models/scalable_svm.py"], users


def test_the_scan_sees_planted_offenders_and_refuses_an_unparsable_file(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text("import sklearn.utils.fixes\nfrom sklearn.utils.parallel import delayed\nfrom joblib import Parallel\n",
                       encoding="utf-8")
    wrong, unpaired, uses = _joblib_findings(planted)
    assert wrong == ["planted.py: sklearn.utils.fixes", "planted.py: sklearn.utils.parallel delayed",
                     "planted.py: delayed from sklearn.utils.parallel"], wrong
    assert unpaired == [] and uses
    alone = tmp_path / "alone.py"
    alone.write_text("from joblib import delayed\n", encoding="utf-8")
    assert _joblib_findings(alone) == ([], ["alone.py: ['delayed']"], True)
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        _imports(broken)
