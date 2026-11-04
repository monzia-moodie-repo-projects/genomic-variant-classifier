"""Where the session guards live (DRIFT-SUITE-UNGUARDED-1, 2026-11-04): the repository root's conftest.py, for every suite.

Until 2026-11-04 the guards were in tests/conftest.py, which `pytest tests_drift/` never loads. They are in the root conftest.py
now; tests/conftest.py keeps what is tests/'s alone (scripts/ on sys.path, the suite-size ratchet, the connector-cache isolation).
Pinned here: both conftests are loaded in this session; no hook or fixture is defined in both files, and at run time every guard
hook has exactly one conftest implementation -- the root's -- in the whole session, every conftest under tests/ included (a hook
implemented twice runs twice); and, from outside, a session laid out like tests_drift/ -- a root conftest.py that is a byte copy of
the real one, test files in tests_drift/, no tests/conftest.py -- fails on a test that rebinds the warning filters, one that
writes into data/raw/cache, one that leaks a sys.path entry and one that writes into the checkout, and names each.
tests_drift/test_session_guards.py observes the guards from inside a real drift session.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT_CONFTEST = REPO / "conftest.py"
TESTS_CONFTEST = REPO / "tests" / "conftest.py"


GUARD_HOOKS = ("pytest_configure", "pytest_runtest_setup", "pytest_runtest_teardown", "pytest_sessionstart", "pytest_runtestloop",
               "pytest_sessionfinish", "pytest_terminal_summary", "pytest_unconfigure")


def _top_level_functions(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_this_session_loads_the_root_conftest_and_tests_conftest(request):
    loaded = {Path(plugin.__file__).resolve() for plugin in request.config.pluginmanager.get_plugins()
              if Path(getattr(plugin, "__file__", None) or "-").name == "conftest.py"}
    assert {ROOT_CONFTEST, TESTS_CONFTEST} <= loaded, loaded


def test_the_guards_are_in_the_root_conftest_and_nothing_is_defined_in_both():
    root, tests = _top_level_functions(ROOT_CONFTEST), _top_level_functions(TESTS_CONFTEST)
    assert not root & tests, sorted(root & tests)
    hooks = set(GUARD_HOOKS)
    fixtures = {"_no_data_dir_writes", "_no_sys_path_leaks", "_isolate_github_workflow_commands", "child_pytest_environment"}
    assert hooks | fixtures <= root, sorted((hooks | fixtures) - root)
    assert tests == {"_read_expected_suite_size", "pytest_addoption", "pytest_collection_modifyitems", "_isolate_connector_caches"}, \
        sorted(tests)


def test_every_guard_hook_has_exactly_one_conftest_implementation_and_it_is_the_roots(request):
    """At run time, over every plugin pytest registered -- tests/unit/conftest.py and tests/conformal/conftest.py included -- so a
    duplicate brought in by import or assignment, or added to another conftest, is seen too."""
    hook = request.config.pluginmanager.hook
    found = {name: [Path(impl.plugin.__file__).resolve() for impl in getattr(hook, name).get_hookimpls()
                    if Path(getattr(impl.plugin, "__file__", None) or "-").name == "conftest.py"] for name in GUARD_HOOKS}
    assert found == {name: [ROOT_CONFTEST] for name in GUARD_HOOKS}, found


DRIFT_LAYOUT_TESTS = '''
import sys
import warnings
from pathlib import Path


def test_rebinds_the_warning_filters():
    warnings.filters = list(warnings.filters)


def test_writes_into_the_checkout():
    Path("left_by_a_drift_test.txt").write_text("x", encoding="utf-8")


def test_writes_into_data_raw_cache():
    Path("data", "raw", "cache").mkdir(parents=True, exist_ok=True)
    Path("data", "raw", "cache", "planted.bin").write_bytes(b"x")


def test_leaks_a_sys_path_entry():
    sys.path.insert(0, "/planted/by/a/drift/test")


def test_passes():
    pass
'''


def test_a_session_laid_out_like_tests_drift_is_guarded_by_the_root_conftest(tmp_path, child_pytest_environment):
    project = tmp_path / "project"
    (project / "tests_drift").mkdir(parents=True)
    (project / "conftest.py").write_bytes(ROOT_CONFTEST.read_bytes())
    (project / "tests_drift" / "test_child.py").write_text(DRIFT_LAYOUT_TESTS, encoding="utf-8")
    env = child_pytest_environment(PYTHONPATH=os.pathsep.join([str(REPO / "src")] +
                                                               ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    run = subprocess.run([sys.executable, "-m", "pytest", "tests_drift", "-q", "-p", "no:cacheprovider", "--confcutdir", str(project)],
                         cwd=str(project),
                         env=env, capture_output=True, text=True, timeout=600)
    out = run.stdout + run.stderr
    assert run.returncode == 1, out
    assert "5 passed, 3 errors" in out, out
    for name in ("test_rebinds_the_warning_filters", "test_writes_into_data_raw_cache", "test_leaks_a_sys_path_entry"):
        assert " ERROR at teardown of {} ".format(name) in out, (name, out)
    assert out.count(" ERROR at teardown of ") == 3, out
    assert "5 tests checked, 1 changed the warning state" in out, out
    assert "ENTRIES CHANGED under data/raw/cache: ['planted.bin']" in out, out
    assert "ADDED  : ['/planted/by/a/drift/test']" in out, out
    assert "THE TEST SESSION CHANGED THE WORKING TREE" in out and "    left_by_a_drift_test.txt\n" in out, out
    assert "joblib: 'threading' runs sequentially for the session" in out, out
