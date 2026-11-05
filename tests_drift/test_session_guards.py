"""The repository root's conftest.py guards the drift suite too (DRIFT-SUITE-UNGUARDED-1, 2026-11-04). RUN IN THE DRIFT ENVIRONMENT.

Until 2026-11-04 the session guards lived in tests/conftest.py, which `pytest tests_drift/` never loads: this suite ran with no
warning-filter guard, no working-tree guard, its temporary files in the system temporary folder, and only ci.yml's `env -u`
between its tests and the workflow's publication files. They are in the root conftest.py now. These tests observe, inside a
drift session, that the root conftest is loaded and tests/conftest.py is not, and that its preventions are in force. What the
guards DO when a test breaks them is proven from outside, where an exit code can be seen, by
tests/unit/test_session_guard_placement.py (a child session with this directory's layout). Since 2026-11-05 the root conftest
also imports, at session start, the modules that register warning filters when first imported
(WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1); which of them this environment has is pinned here.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _publication_files() -> tuple:
    """The root conftest.py's _GITHUB_WORKFLOW_COMMAND_FILES -- the one source -- read by parsing it (a conftest is never imported)."""
    tree = ast.parse((REPO / "conftest.py").read_text(encoding="utf-8"), filename=str(REPO / "conftest.py"))
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "_GITHUB_WORKFLOW_COMMAND_FILES" for t in n.targets)]
    assert len(values) == 1, values
    return values[0]


def _conftest_files(config) -> set:
    return {Path(plugin.__file__).resolve() for plugin in config.pluginmanager.get_plugins()
            if Path(getattr(plugin, "__file__", None) or "-").name == "conftest.py"}


def test_this_session_runs_under_the_repository_root_conftest_and_not_tests_conftest(request):
    loaded = _conftest_files(request.config)
    assert REPO / "conftest.py" in loaded, loaded
    assert REPO / "tests" / "conftest.py" not in loaded, loaded


def test_the_root_conftests_preventions_are_in_force_here(tmp_path_factory):
    from joblib.parallel import BACKENDS, get_active_backend

    # No thread-parallel joblib work (SKLEARN-WARNING-FILTER-RACE-1): the thread backend is the sequential one.
    assert BACKENDS["threading"] is BACKENDS["sequential"]
    assert isinstance(get_active_backend(prefer="threads", require="sharedmem")[0], BACKENDS["sequential"])
    # Temporary files land under pytest's base temporary directory, for this process and the children it starts.
    root = tmp_path_factory.getbasetemp() / "tempfile"
    assert Path(tempfile.gettempdir()) == root
    assert [os.environ[n] for n in ("TMPDIR", "TEMP", "TMP")] == [str(root)] * 3
    # The workflow's publication files are not reachable from a test.
    names = _publication_files()
    assert len(names) == 6, names
    assert not [name for name in names if name in os.environ], names


def test_the_modules_that_register_warning_filters_were_imported_at_session_start_or_are_absent_here(request):
    """Both drift locks (requirements-drift-linux.lock and requirements-drift-windows.lock, read 2026-11-04) have requests,
    numpy, pandas and scipy and none of the other six: those can be imported by no test here, and are named as absent."""
    found = [plugin for plugin in request.config.pluginmanager.get_plugins()
             if Path(getattr(plugin, "__file__", None) or "-").resolve() == REPO / "conftest.py"]
    assert len(found) == 1, found
    imported, absent, _ = request.config.stash[found[0]._WARNING_GUARD_TALLY]["imported"]
    assert imported == ("requests", "numpy", "pandas", "scipy.sparse", "scipy.special"), imported
    assert absent == ("torch", "torch_geometric", "accelerate", "mlxtend.frequent_patterns", "spacy", "pyspark"), absent
