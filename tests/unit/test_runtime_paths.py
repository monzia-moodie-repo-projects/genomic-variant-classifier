"""Tests for the runtime-path authority -- RUNTIME-PATHS-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest

from genomic_variant_classifier.paths.runtime_paths import (
    ENV_ARTIFACT_ROOT, ENV_PROJECT_ROOT, ENV_STATE_ROOT, PROJECT_NAME,
    PROJECT_SENTINELS, RuntimePathError, RuntimePaths, discover_project_root,
    looks_like_project_root, resolve_project_root, resolve_runtime_paths,
)


def _make_repo(tmp: Path, *, name: str = PROJECT_NAME,
               omit: str = None) -> Path:
    """A directory that satisfies the sentinels, unless one is omitted."""
    for sentinel in PROJECT_SENTINELS:
        if sentinel == omit:
            continue
        p = tmp / sentinel
        if sentinel.endswith(".toml"):
            p.parent.mkdir(parents=True, exist_ok=True)
            io.open(p, "w", encoding="utf-8", newline="\n").write(
                '[project]\nname = "{}"\nversion = "0.1.0"\n'.format(name))
        elif "." in Path(sentinel).name:
            p.parent.mkdir(parents=True, exist_ok=True)
            io.open(p, "w", encoding="utf-8", newline="\n").write("4885\n")
        else:
            p.mkdir(parents=True, exist_ok=True)
    return tmp


# ---- identity, not existence -------------------------------------------
def test_a_directory_with_the_sentinels_and_the_name_is_the_project():
    root = _make_repo(Path(tempfile.mkdtemp()))
    assert looks_like_project_root(root) is True


def test_a_directory_that_merely_contains_src_is_NOT_the_project():
    """`(candidate / "src").exists()` would be a comfort assertion -- any
    directory can contain src/. This is the case that motivates the
    conjunction."""
    other = Path(tempfile.mkdtemp())
    (other / "src").mkdir()
    assert looks_like_project_root(other) is False


@pytest.mark.parametrize("omit", PROJECT_SENTINELS)
def test_every_sentinel_is_load_bearing(omit):
    """Removing ANY one must break identification, or it is decoration."""
    root = _make_repo(Path(tempfile.mkdtemp()), omit=omit)
    assert looks_like_project_root(root) is False, omit


def test_a_DIFFERENT_project_with_the_same_shape_is_rejected():
    """The structural check: sentinels present, wrong declared name."""
    root = _make_repo(Path(tempfile.mkdtemp()), name="some-other-project")
    assert looks_like_project_root(root) is False


def test_an_unparseable_pyproject_is_rejected_not_crashed():
    root = _make_repo(Path(tempfile.mkdtemp()))
    io.open(root / "pyproject.toml", "w", encoding="utf-8").write("{{{ not toml")
    assert looks_like_project_root(root) is False


# ---- precedence, and the absence of a fallback --------------------------
def test_an_explicit_root_wins():
    root = _make_repo(Path(tempfile.mkdtemp()))
    other = _make_repo(Path(tempfile.mkdtemp()))
    got = resolve_project_root(explicit=root,
                               environ={ENV_PROJECT_ROOT: str(other)})
    assert got == root.resolve()


def test_the_environment_wins_over_discovery():
    root = _make_repo(Path(tempfile.mkdtemp()))
    got = resolve_project_root(environ={ENV_PROJECT_ROOT: str(root)})
    assert got == root.resolve()


def test_discovery_is_anchored_to_the_module_not_the_cwd(monkeypatch):
    """THE MODULE'S CENTRAL CLAIM, driven where the two answers DIFFER.

    An earlier version merely asserted `found is None or looks_like(found)`,
    which holds whether discovery walks from __file__ or from the working
    directory -- so sabotage S13, replacing Path(__file__) with Path("."),
    went undetected. In the sandbox both roots happen to fail identically.

    This constructs a repository, chdirs INTO it, and asserts discovery does
    NOT find it: the module lives elsewhere, so an __file__-anchored walk
    cannot reach it, while a cwd-anchored walk would find it immediately.
    """
    fake = _make_repo(Path(tempfile.mkdtemp()))
    monkeypatch.chdir(fake)
    found = discover_project_root()
    assert found != fake.resolve(), (
        "discovery found the working directory, so it is anchored to the CWD "
        "rather than to __file__ -- the defect this module exists to end")


def test_discovery_DOES_find_a_repository_above_the_given_origin():
    """The positive direction: given an origin inside a repository, the walk
    upward finds it. Without this, the test above could pass against a
    discover_project_root that always returns None."""
    root = _make_repo(Path(tempfile.mkdtemp()))
    deep = root / "src" / "genomic_variant_classifier" / "nested" / "deeper"
    deep.mkdir(parents=True, exist_ok=True)
    found = discover_project_root(origin=deep)
    assert found == root.resolve(), (found, root.resolve())


def test_an_explicit_root_that_is_NOT_the_project_RAISES():
    other = Path(tempfile.mkdtemp())
    (other / "src").mkdir()
    with pytest.raises(RuntimePathError) as exc:
        resolve_project_root(explicit=other, environ={})
    assert "not this repository" in str(exc.value)
    assert "Existence alone is not identity" in str(exc.value)


def test_a_nonexistent_explicit_root_RAISES():
    with pytest.raises(RuntimePathError) as exc:
        resolve_project_root(explicit="/definitely/not/here", environ={})
    assert "not a directory" in str(exc.value)


def test_a_wrong_environment_value_RAISES_rather_than_falling_back():
    """THE DEFECT, as a test. config.py:17 fell back to a Windows literal when
    GVC_PROJECT_ROOT was unset -- and it is set NOWHERE, so every machine
    received one author's absolute path."""
    other = Path(tempfile.mkdtemp())
    with pytest.raises(RuntimePathError) as exc:
        resolve_project_root(environ={ENV_PROJECT_ROOT: str(other)})
    assert "not this repository" in str(exc.value)


def test_there_is_no_developer_path_anywhere_in_the_module():
    """A literal absolute path in the source would reintroduce the defect."""
    import genomic_variant_classifier.paths.runtime_paths as M
    src = io.open(M.__file__, encoding="utf-8").read()
    import ast
    tree = ast.parse(src)
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    # Docstrings quote the old literal deliberately; check EXECUTABLE constants.
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            b = getattr(n, "body", None)
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(getattr(b[0], "value", None), ast.Constant)):
                docs.add(id(b[0].value))
    live = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]
    # BOTH separators. An earlier version checked only s[1:3] == ":\\", so
    # sabotage R1 -- reintroducing the fallback as "C:/Projects/..." with a
    # FORWARD slash -- went undetected. A Windows drive path is "X:" followed
    # by either separator, and a POSIX absolute path starts with "/".
    import re
    drive = re.compile(r"^[A-Za-z]:[\\/]")
    offenders = [s for s in live
                 if (s.startswith("/") and len(s) > 8) or drive.match(s)]
    assert not offenders, (
        "an absolute path literal in executable code reintroduces "
        "PROJECT-ROOT-HARDCODED-1: {}".format(offenders))


def test_resolution_RAISES_when_nothing_identifies_the_project():
    """The fallback path, driven directly.

    Sabotage R1 -- returning a developer path instead of raising -- was missed
    because no test reached this branch: discovery finds the real repository
    from __file__, so the raise is unreachable in situ. Monkeypatching
    discovery is the only way to exercise it.
    """
    import genomic_variant_classifier.paths.runtime_paths as M
    real = M.discover_project_root
    M.discover_project_root = lambda origin=None: None
    try:
        with pytest.raises(RuntimePathError) as exc:
            M.resolve_project_root(environ={})
        assert "NO developer-specific path" in str(exc.value)
        assert ENV_PROJECT_ROOT in str(exc.value)
    finally:
        M.discover_project_root = real


def test_a_pyproject_that_is_not_valid_TOML_is_REJECTED_not_raised():
    """The handler must catch TOMLDecodeError specifically.

    Sabotage R12 narrowed it to ValueError. tomllib.TOMLDecodeError subclasses
    ValueError in CPython, so that mutation happened to keep working -- but it
    is an implementation detail, not a contract, and the test now drives the
    real exception type.
    """
    import tomllib
    root = _make_repo(Path(tempfile.mkdtemp()))
    io.open(root / "pyproject.toml", "w", encoding="utf-8").write("name = [unclosed")
    with pytest.raises(tomllib.TOMLDecodeError):
        with (root / "pyproject.toml").open("rb") as fh:
            tomllib.load(fh)
    assert looks_like_project_root(root) is False


# ---- the three roots are distinct ---------------------------------------
def test_artifact_and_state_default_to_distinct_locations():
    root = _make_repo(Path(tempfile.mkdtemp()))
    paths = resolve_runtime_paths(project_root=root, environ={})
    assert paths.project_root == root.resolve()
    assert paths.artifact_root == root.resolve()
    assert paths.state_root == root.resolve() / ".gvc-state"
    assert paths.state_root != paths.project_root


def test_each_root_is_independently_overridable():
    root = _make_repo(Path(tempfile.mkdtemp()))
    art = Path(tempfile.mkdtemp())
    st = Path(tempfile.mkdtemp())
    paths = resolve_runtime_paths(
        project_root=root,
        environ={ENV_ARTIFACT_ROOT: str(art), ENV_STATE_ROOT: str(st)})
    assert paths.artifact_root == art.resolve()
    assert paths.state_root == st.resolve()


def test_artifact_and_state_roots_need_NOT_exist():
    """They are DESTINATIONS, created on first write. Requiring them would make
    a fresh clone fail before it could produce anything."""
    root = _make_repo(Path(tempfile.mkdtemp()))
    paths = resolve_runtime_paths(project_root=root, environ={})
    assert not paths.state_root.exists()
    assert paths.state_root.name == ".gvc-state"


# ---- the two state stores are NAMED for their owners --------------------
def test_the_two_state_stores_have_DIFFERENT_paths_and_names():
    """Two files called agent_state.json held unrelated schemas -- a flat
    literature-scout key-value log and the orchestrator's structured
    SharedState. Reasoning from the filename nearly merged them."""
    root = _make_repo(Path(tempfile.mkdtemp()))
    p = resolve_runtime_paths(project_root=root, environ={})
    assert p.literature_scout_state != p.orchestrator_state
    assert "literature_scout" in str(p.literature_scout_state)
    assert "orchestrator" in str(p.orchestrator_state)
    assert p.literature_scout_state.parent != p.orchestrator_state.parent


def test_reports_root_derives_from_ARTIFACT_root_not_project_root():
    """OUTPUT-ROOT-CONFLATION-1: where output goes is a deployment decision,
    not a fact about repository layout."""
    root = _make_repo(Path(tempfile.mkdtemp()))
    art = Path(tempfile.mkdtemp())
    p = resolve_runtime_paths(project_root=root, artifact_root=art, environ={})
    assert p.reports_root == art.resolve() / "reports"
    assert not str(p.reports_root).startswith(str(root.resolve()))


# ---- immutability and provenance ----------------------------------------
def test_runtime_paths_are_immutable():
    import dataclasses
    root = _make_repo(Path(tempfile.mkdtemp()))
    p = resolve_runtime_paths(project_root=root, environ={})
    try:
        p.project_root = Path("/elsewhere")
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("runtime paths were mutable")


def test_describe_records_every_root_for_provenance():
    root = _make_repo(Path(tempfile.mkdtemp()))
    d = resolve_runtime_paths(project_root=root, environ={}).describe()
    for key in ("project_root", "artifact_root", "state_root", "reports_root",
                "literature_scout_state", "orchestrator_state"):
        assert key in d and d[key], key


def test_the_environment_is_read_at_RESOLUTION_not_at_import():
    """A module-level snapshot is what made config.py's value unchangeable."""
    root = _make_repo(Path(tempfile.mkdtemp()))
    other = _make_repo(Path(tempfile.mkdtemp()))
    a = resolve_project_root(environ={ENV_PROJECT_ROOT: str(root)})
    b = resolve_project_root(environ={ENV_PROJECT_ROOT: str(other)})
    assert a != b
