"""tests/conftest.py: temporary-file containment and the working-tree guard (both added 2026-10-12).

The in-process tests observe THIS session. The child-session tests run pytest on a small project whose tests/conftest.py is a byte
copy of the real one, so the guard's verdict, its exit code and its message are observed from outside -- the only place an exit
code can be seen.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFTEST = REPO / "tests" / "conftest.py"


# --------------------------------------------------------------------------- containment, observed in this session
def test_tempfile_lands_inside_pytests_base_temporary_directory(tmp_path_factory):
    root = tmp_path_factory.getbasetemp() / "tempfile"
    assert Path(tempfile.gettempdir()) == root
    assert [os.environ[n] for n in ("TMPDIR", "TEMP", "TMP")] == [str(root)] * 3
    made = Path(tempfile.mkdtemp())
    assert made.parent == root
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        assert Path(fh.name).parent == root


def test_a_child_interpreter_inherits_the_contained_root(tmp_path_factory):
    root = tmp_path_factory.getbasetemp() / "tempfile"
    child = subprocess.run([sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"], capture_output=True, text=True,
                           timeout=120)
    assert child.returncode == 0, child.stderr
    assert Path(child.stdout.strip()) == root


# --------------------------------------------------------------------------- child sessions
PROBE_PLUGIN = '''
import json, os, tempfile
SEEN = {}
def _state():
    return {"env": {n: os.environ.get(n) for n in ("TMPDIR", "TEMP", "TMP")}, "tempdir": tempfile.tempdir}
def pytest_configure(config):
    SEEN["before"] = _state()
def pytest_unconfigure(config):
    SEEN["after"] = _state()
    with open(os.environ["GUARD_PROBE_OUT"], "w", encoding="utf-8") as fh:
        json.dump(SEEN, fh)
'''

CLEAN_TESTS = '''
import os, tempfile
from pathlib import Path

IMPORTED_UNDER = tempfile.gettempdir()          # at IMPORT, during collection: before any fixture exists

def test_contained_already_at_import(tmp_path_factory):
    assert IMPORTED_UNDER == str(tmp_path_factory.getbasetemp() / "tempfile")

def test_contained(tmp_path_factory):
    root = tmp_path_factory.getbasetemp() / "tempfile"
    assert tempfile.gettempdir() == str(root) == os.environ["TEMP"] == os.environ["TMP"] == os.environ["TMPDIR"]

def test_a_monkeypatched_variable(monkeypatch):
    monkeypatch.setenv("TEMP", "elsewhere")

def test_gets_the_contained_value_back(tmp_path_factory):
    assert os.environ["TEMP"] == str(tmp_path_factory.getbasetemp() / "tempfile")

def test_writes_only_under_tmp_path(tmp_path):
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    Path(tempfile.mkdtemp(), "y.txt").write_text("y", encoding="utf-8")
'''

DIRTY_TESTS = '''
from pathlib import Path

def test_writes_a_checkpoint_into_the_current_directory():
    Path("models").mkdir()
    Path("models", "best_gat.pt").write_bytes(b"weights")
'''


def _child(tmp_path: Path, tests: str, *extra: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "conftest.py").write_bytes(CONFTEST.read_bytes())
    (project / "tests" / "test_child.py").write_text(tests, encoding="utf-8")
    plugins = tmp_path / "plugins"                                   # OUTSIDE the project: the guard watches the project
    plugins.mkdir()
    (plugins / "guard_probe.py").write_text(PROBE_PLUGIN, encoding="utf-8")
    probe = tmp_path / "probe.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(plugins)] +
                                                      ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
               GUARD_PROBE_OUT=str(probe))
    run = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-p", "guard_probe", *extra],
                         cwd=str(project), env=env, capture_output=True, text=True, timeout=600)
    return run, project, probe


def test_a_clean_session_passes_and_restores_everything(tmp_path):
    run, project, probe = _child(tmp_path, CLEAN_TESTS)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "5 passed" in run.stdout and "working-tree guard" in run.stdout and "unchanged" in run.stdout, run.stdout
    seen = json.loads(probe.read_text(encoding="utf-8"))
    assert seen["after"]["env"] == seen["before"]["env"]
    # tempfile caches its directory the first time it is asked (pytest asks, to place its base temporary directory): restored to
    # that cached value -- the directory the ORIGINAL environment names -- or to None if it was never asked.
    assert seen["after"]["tempdir"] in (None, os.path.abspath(seen["before"]["env"]["TMPDIR"])), seen


def test_a_session_that_writes_into_the_checkout_fails_and_names_what_it_left(tmp_path):
    run, project, _ = _child(tmp_path, DIRTY_TESTS)
    assert run.returncode == 1, run.stdout + run.stderr
    out = run.stdout
    assert "1 passed" in out, out                                     # every test passed; the SESSION fails
    assert "THE TEST SESSION CHANGED THE WORKING TREE" in out and "ADDED (2):" in out, out
    assert "    models\n" in out and "    models/best_gat.pt\n" in out, out
    assert (project / "models" / "best_gat.pt").is_file()             # it reports; it deletes nothing


def test_without_a_terminal_summary_the_finding_still_reaches_the_console(tmp_path):
    run, _, _ = _child(tmp_path, DIRTY_TESTS, "--no-summary")
    assert run.returncode == 1, run.stdout + run.stderr
    assert "working-tree guard:" in run.stderr and "models/best_gat.pt" in run.stderr, run.stderr


IMPORT_TIME_PROBE = '''
import os, tempfile
with open(os.environ["GUARD_IMPORT_OUT"], "w", encoding="utf-8") as fh:
    fh.write(tempfile.mkdtemp())

def test_never_runs():
    pass
'''


def test_a_collection_only_run_is_contained_at_import_time(tmp_path, monkeypatch):
    """MEASURED 2026-10-12: with containment in a session fixture, a --collect-only run (and every run's collection) still created
    torchinductor_root in the outer temporary folder. Containment now starts at configuration, before any test module is imported."""
    out = tmp_path / "made.txt"
    monkeypatch.setenv("GUARD_IMPORT_OUT", str(out))
    run, _, _ = _child(tmp_path, IMPORT_TIME_PROBE, "--collect-only")
    assert run.returncode == 0, run.stdout + run.stderr
    made = Path(out.read_text(encoding="utf-8"))
    assert made.parent.name == "tempfile" and made.parent.parent.name.startswith("pytest-"), made
    assert Path(tempfile.gettempdir()) in made.parents                  # inside THIS session's contained root as well
