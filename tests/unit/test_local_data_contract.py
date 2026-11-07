"""`local_data` is a contract (DATA-GATED-TESTS-1, 2026-11-06). Author: Monzia Moodie

tests/local_data.py says why it exists. What is asserted here:
  * the set of local-data tests is EXACTLY the measured one -- the data census of 2026-11-06 found three tests that check local
    data by design; a fourth is a decision to record, not an accident to absorb;
  * every declared path is relative to the repository root in POSIX form, git-ignored and untracked (asked of git, which answers
    for paths that do not exist), so the decorator cannot hide an ordinary missing file;
  * `pytest.mark.local_data` is never written by hand -- a bare mark would carry no skip -- and the mark is registered;
  * in a real pytest session, the decorator skips a test with the one reason where its data is absent and runs it where the data
    is present, and `-m local_data` selects exactly the decorated tests;
  * every test that runs the Application Programming Interface's startup requests api_without_local_data.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from tests.local_data import REPO_ROOT, SKIP_REASON, check_relative, local_data, repo_path

_TESTS = REPO_ROOT / "tests"

#: Measured 2026-11-06 (block DC on the owner's machine: these, and only these, check git-ignored data by design).
EXPECTED = {
    ("tests/unit/test_drift_reference_profile.py", "test_psi_exact_on_the_real_run15_matrix"):
        ("outputs/run15_rerun_report/full/splits/X_train.parquet",),
    ("tests/unit/test_eve_entry_name_resolution.py", "test_uniprot_index_has_entry_name_column"):
        ("data/external/uniprot/uniprot_human_reviewed.parquet",),
    ("tests/unit/test_eve_entry_name_resolution.py", "test_real_corpus_resolution_fraction"):
        ("data/external/uniprot/uniprot_human_reviewed.parquet", "data/external/eve/EVE_all_data/variant_files"),
}


def _is_local_data(node) -> bool:
    """`@local_data(...)`, the decorator imported from tests.local_data -- the one way to declare a local-data test."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "local_data"


def _declared() -> dict:
    """{(file, test): (paths...)} for every `@local_data(...)` on a test in tests/ -- a module-level function or a method of a
    module-level class ("Class::method") -- module-level string constants resolved. A class may not carry it (the mark would reach
    every method), and a function may carry it once. Functions nested inside a test are not tests: this module's own planted one is
    one of them."""
    found = {}
    for path in sorted(_TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        where = path.relative_to(REPO_ROOT).as_posix()
        constants = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                     and isinstance(n.value.value, str) for t in n.targets if isinstance(t, ast.Name)}
        tests = [(n.name, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            assert not any(_is_local_data(d) for d in cls.decorator_list), "{}:{} local_data on a class".format(where, cls.lineno)
            tests += [("{}::{}".format(cls.name, m.name), m) for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for name, node in tests:
            uses = [d for d in node.decorator_list if _is_local_data(d)]
            if not uses:
                continue
            assert len(uses) == 1 and not uses[0].keywords, "{}:{} local_data more than once, or with keywords".format(where, node.lineno)
            args = []
            for a in uses[0].args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    args.append(a.value)
                elif isinstance(a, ast.Name) and a.id in constants:
                    args.append(constants[a.id])
                else:
                    raise AssertionError("{}:{} local_data argument is not a string literal or a module string constant".format(
                        where, node.lineno))
            found[(where, name)] = tuple(args)
    return found


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(("git", "-C", str(REPO_ROOT)) + args, capture_output=True, text=True, timeout=120)


_HAS_GIT = (REPO_ROOT / ".git").exists() and _git("rev-parse", "--git-dir").returncode == 0


def test_the_local_data_tests_are_exactly_the_measured_three():
    assert _declared() == EXPECTED


@pytest.mark.skipif(not _HAS_GIT, reason="{} is not a git working tree; git cannot answer".format(REPO_ROOT))
def test_every_declared_path_is_git_ignored_and_untracked():
    paths = sorted({p for ps in _declared().values() for p in ps})
    for p in paths:
        assert check_relative(p) == p
    tracked = [p for p in paths if _git("ls-files", "--", p).stdout.strip()]
    assert not tracked, "declared local data that git TRACKS (every clone has it): {}".format(tracked)
    # NUL-separated BYTES (-z), as test_c2_qualification_record.py measured on 2026-10-02: in text mode Python writes "\r\n" on
    # Windows, the carriage return joins the path, and git answers for a path that does not exist (MEASURED on the owner's machine
    # 2026-11-06: this test's first form failed there, each path reported as "<path>\r"). The control is a tracked, never-ignored
    # file that git must NOT report, so an instrument that reports everything fails instead of passing.
    control = "tests/local_data.py"
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "--no-index", "-z", "--stdin"],
                         input=("\0".join(paths + [control]) + "\0").encode("utf-8"), capture_output=True, timeout=120)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    answer = [x for x in out.stdout.decode("utf-8").split("\0") if x]
    assert sorted(answer) == paths, "declared local data that git does NOT ignore: {}; reported: {}".format(
        sorted(set(paths) - set(answer)), answer)


@pytest.mark.parametrize("bad", ["", "/abs/x", "C:/x", "a\\b", "../x", "a/../b", "./a", "a//b", "a/"])
def test_a_declared_path_must_be_repository_relative_posix(bad):
    with pytest.raises(ValueError):
        check_relative(bad)


def test_repo_path_is_anchored_at_the_repository_root_not_the_current_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert repo_path("data/external/x.parquet") == REPO_ROOT / "data" / "external" / "x.parquet"
    assert (REPO_ROOT / "tests" / "local_data.py").is_file()


def test_the_mark_is_never_written_by_hand():
    """A bare `pytest.mark.local_data` would select the test under `-m local_data` and skip nothing where the data is absent."""
    by_hand = []
    for path in sorted(_TESTS.rglob("*.py")):
        if path == REPO_ROOT / "tests" / "local_data.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Attribute) and node.attr == "local_data" and isinstance(node.value, ast.Attribute) \
                    and node.value.attr == "mark":
                by_hand.append("{}:{}".format(path.relative_to(REPO_ROOT).as_posix(), node.lineno))
    assert not by_hand, by_hand


def test_the_mark_is_registered(request):
    assert [m for m in request.config.getini("markers") if m.startswith("local_data(")], request.config.getini("markers")


@pytest.mark.parametrize("paths", [(), ("../x",), ("data/x", "/abs")])
def test_the_decorator_refuses_no_path_and_a_bad_path(paths):
    with pytest.raises(ValueError):
        local_data(*paths)


def test_the_decorator_applies_the_mark_and_pytests_own_skip():
    absent, present = "data/external/never_present_local_data_probe.parquet", "tests/local_data.py"
    assert not repo_path(absent).exists() and repo_path(present).is_file()

    @local_data(absent, present)
    def planted():
        pass

    marks = {m.name: m for m in planted.pytestmark}
    assert set(marks) == {"local_data", "skipif"}
    assert marks["local_data"].args == (absent, present)
    assert marks["skipif"].args == (True,) and marks["skipif"].kwargs == {"reason": SKIP_REASON + absent}


PLANTED = '''
from tests.local_data import local_data


@local_data("data/external/never_present_local_data_probe.parquet")
def test_absent():
    pass


@local_data("tests/local_data.py")
def test_present():
    pass


def test_not_local_data():
    pass
'''


def test_in_a_real_session_the_decorator_skips_where_absent_runs_where_present_and_m_selects(tmp_path, child_pytest_environment):
    """A child pytest outside the repository (its own pytest.ini registering the mark, no conftest), importing this repository's
    tests.local_data. -B: a child must not write bytecode into the checkout (PYCACHE-FROM-ISOLATED-CHILDREN-1)."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "pytest.ini").write_text("[pytest]\nmarkers =\n    local_data(*paths): planted\n", encoding="utf-8")
    (project / "test_planted.py").write_text(PLANTED, encoding="utf-8")
    env = child_pytest_environment(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(
        [str(REPO_ROOT)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))

    def run(*options):
        done = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "--confcutdir",
                               str(project), "-W", "error::pytest.PytestUnknownMarkWarning"] + list(options),
                              cwd=str(project), env=env, capture_output=True, text=True, timeout=300)
        return done.returncode, done.stdout + done.stderr

    code, out = run()
    assert code == 0 and "2 passed, 1 skipped" in out, out
    assert (SKIP_REASON + "data/external/never_present_local_data_probe.parquet") in out, out
    assert out.count("SKIPPED [") == 1, out
    code, out = run("-m", "local_data")
    assert code == 0 and "1 passed, 1 skipped, 1 deselected" in out, out


# ---- the API's startup reads no local data in a test ------------------------------------------------------------------
def _runs_the_startup(function) -> bool:
    """`with TestClient(...)` (Starlette runs the lifespan only as a context manager) or `async with <x>.lifespan(...)`."""
    for node in ast.walk(function):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    f = call.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if name in ("TestClient", "lifespan"):
                        return True
    return False


def _startup_runners() -> dict:
    """{(file, function): parameters} for every top-level function and method in tests/ that runs the API's startup."""
    found = {}
    for path in sorted(_TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        outer = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        outer += [m for c in tree.body if isinstance(c, ast.ClassDef) for m in c.body
                  if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for function in outer:
            if _runs_the_startup(function):
                found[(path.relative_to(REPO_ROOT).as_posix(), function.name)] = [a.arg for a in function.args.args]
    return found


def test_every_test_that_runs_the_api_startup_requests_api_without_local_data():
    """MEASURED 2026-11-06: these are the places, and the startup otherwise loads the checkout's git-ignored dbSNP index."""
    runners = _startup_runners()
    assert set(runners) == {
        ("tests/unit/test_runtime_attribution.py", "registered_production_client"),
        ("tests/unit/test_runtime_attribution.py", "test_the_same_registered_production_artifact_is_refused_under_the_default_authority"),
        ("tests/unit/test_containment_serving.py", "test_the_api_server_refuses_to_serve_a_quarantined_artifact"),
    }, sorted(runners)
    missing = sorted(k for k, params in runners.items() if "api_without_local_data" not in params)
    assert not missing, "runs the API startup without api_without_local_data: {}".format(missing)


def test_api_without_local_data_points_every_table_at_an_absent_path_and_restores_the_tables(api_without_local_data, tmp_path):
    api = api_without_local_data
    assert api.GNOMAD_INDEX_PATH is None
    for path in (api.GENE_SUMMARY_PATH, api.DBSNP_INDEX_PATH):
        assert not path.exists() and tmp_path in path.parents, path
