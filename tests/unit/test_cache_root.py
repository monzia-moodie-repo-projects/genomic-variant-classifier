"""The transaction journal lives outside the repository.

INSTALLER-TRANSACTION-1, step 2: a fifth path domain.

WHY A FIFTH DOMAIN AND NOT state_root
MEASURED 2026-08-19: state_root defaults to <project>/.gvc-state -- a
repository subdirectory, git-ignored by the `/.gvc-state/` rule added at
a734ea1. That is CORRECT for what it holds: literature-scout and orchestrator
state belong to THIS checkout.

A transaction journal does not. It must survive an interrupted installer even
if the working tree is reset, and the governing invariant is that a successful
installer leaves NO rollback artefact in the repository. Putting the journal
under state_root would place it back inside the thing it exists to repair.

    repository identity  -> project_root
    artifact identity    -> artifact_root
    checkout state       -> state_root
    machine-scoped cache -> cache_root      <- new

WHAT THIS REPLACES
MEASURED 2026-08-19: 148 `.bak_<timestamp>` siblings had accumulated inside the
repository across eight days -- 17,640,928 bytes, invisible to `git status`
because .gitignore carries `*.bak_*`. 139 held bytes git already had; 8 were
superseded working-tree states; 1 was credential-bearing. Retired at 5447362
with a manifest recording what each was.

A NOTE ON WHAT CANNOT BE TESTED FROM WINDOWS
Passing a fake environment with XDG_STATE_HOME="/home/runner/.local/state"
selects the right BRANCH but produces "C:/home/runner/..." -- MEASURED
2026-08-19. Path flavour is baked into the platform, not into the environment.

So these tests assert RELATIONSHIPS -- outside the repository, beneath the
chosen base, writable -- which hold on both platforms. The literal POSIX form
is verified on the runner, where the tests actually execute on POSIX.

That is the sentinel lesson arriving from another direction: a property that
can only be checked in one environment must be expressed so it can be checked
in both.

Author: Monzia Moodie
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from genomic_variant_classifier.paths.runtime_paths import (
    ENV_CACHE_ROOT, PROJECT_NAME, RuntimePaths, _default_cache_root,
    resolve_runtime_paths,
)

_REPO = Path(__file__).resolve().parents[2]


def _make_repo(base: Path) -> Path:
    """A directory that satisfies the project sentinels."""
    (base / "src" / "genomic_variant_classifier").mkdir(parents=True, exist_ok=True)
    (base / "pyproject.toml").write_text(
        '[project]\nname = "{}"\nversion = "0.1.0"\n'.format(PROJECT_NAME),
        encoding="utf-8")
    return base


# ---- the field exists and is resolved -----------------------------------
def test_cache_root_is_a_field_of_RuntimePaths():
    import dataclasses
    names = [f.name for f in dataclasses.fields(RuntimePaths)]
    assert "cache_root" in names, names
    assert names == ["project_root", "artifact_root", "state_root", "cache_root"], (
        "field ORDER matters: the sole construction site is keyword-only "
        "today, but a positional call elsewhere would silently mis-assign")


def test_cache_root_is_resolved_absolute():
    root = _make_repo(Path(tempfile.mkdtemp()))
    p = resolve_runtime_paths(project_root=root, environ={})
    assert p.cache_root.is_absolute(), p.cache_root


# ---- THE POINT: it is not inside the repository -------------------------
def test_the_cache_root_is_OUTSIDE_the_project_root():
    """The invariant this domain exists to guarantee.

    A journal beneath project_root would reintroduce exactly what the 148
    retired artefacts were.
    """
    root = _make_repo(Path(tempfile.mkdtemp()))
    p = resolve_runtime_paths(project_root=root, environ={})
    assert root.resolve() not in p.cache_root.parents, (p.cache_root, root)
    assert not str(p.cache_root).startswith(str(root.resolve())), p.cache_root


def test_the_transaction_journal_is_OUTSIDE_the_project_root():
    root = _make_repo(Path(tempfile.mkdtemp()))
    p = resolve_runtime_paths(project_root=root, environ={})
    assert not str(p.transaction_journal).startswith(str(root.resolve()))
    assert p.transaction_journal == p.cache_root / "transactions"


def test_the_journal_is_NOT_under_state_root():
    """state_root is checkout state; the journal must outlive the checkout."""
    root = _make_repo(Path(tempfile.mkdtemp()))
    p = resolve_runtime_paths(project_root=root, environ={})
    assert not str(p.transaction_journal).startswith(str(p.state_root))
    assert p.cache_root != p.state_root


# ---- resolution order ---------------------------------------------------
def test_an_explicit_cache_root_wins():
    root = _make_repo(Path(tempfile.mkdtemp()))
    chosen = Path(tempfile.mkdtemp()) / "chosen"
    p = resolve_runtime_paths(project_root=root, cache_root=chosen,
                              environ={ENV_CACHE_ROOT: "/ignored"})
    assert p.cache_root == chosen.resolve(), p.cache_root


def test_the_environment_variable_is_honoured():
    root = _make_repo(Path(tempfile.mkdtemp()))
    chosen = Path(tempfile.mkdtemp()) / "from_env"
    p = resolve_runtime_paths(project_root=root,
                              environ={ENV_CACHE_ROOT: str(chosen)})
    assert p.cache_root == chosen.resolve(), p.cache_root


def test_the_cache_root_need_not_exist_yet():
    """Like artifact_root and state_root, it is a DESTINATION. Requiring it to
    exist would make a fresh clone fail before it could produce anything."""
    root = _make_repo(Path(tempfile.mkdtemp()))
    absent = Path(tempfile.mkdtemp()) / "not" / "yet"
    p = resolve_runtime_paths(project_root=root,
                              environ={ENV_CACHE_ROOT: str(absent)})
    assert not absent.exists()
    assert p.cache_root == absent.resolve()


# ---- the default, expressed so BOTH platforms can check it --------------
def test_the_default_prefers_LOCALAPPDATA_on_windows():
    base = Path(tempfile.mkdtemp())
    got = _default_cache_root({"LOCALAPPDATA": str(base)})
    if os.name == "nt":
        assert got == (base / "GenomicVariantClassifier").resolve(), got
    else:
        # On POSIX the LOCALAPPDATA branch is not taken at all.
        assert "GenomicVariantClassifier" in str(got), got


def test_the_default_uses_XDG_STATE_HOME_when_LOCALAPPDATA_is_absent():
    base = Path(tempfile.mkdtemp())
    got = _default_cache_root({"XDG_STATE_HOME": str(base)})
    assert got == (base / "GenomicVariantClassifier").resolve(), got


def test_the_default_ALWAYS_resolves_with_an_empty_environment():
    """The fallback that cannot be unset.

    MEASURED 2026-08-19: with HOME unset on Windows, Path.home() still returned
    C:/Users/monzi via USERPROFILE. On POSIX it falls back to the password
    database. A default that could return None would place a journal at the
    filesystem root.
    """
    got = _default_cache_root({})
    assert got.is_absolute(), got
    assert got.name == "GenomicVariantClassifier", got
    assert str(got) not in ("", "/", "\\"), got


def test_the_default_is_never_inside_THIS_repository():
    """Checked against the real repository, on whichever platform runs."""
    for env in ({}, {"XDG_STATE_HOME": tempfile.mkdtemp()},
                {"LOCALAPPDATA": tempfile.mkdtemp()}):
        got = _default_cache_root(env)
        assert not str(got).startswith(str(_REPO)), (env, got)


def test_the_default_carries_the_project_name():
    """So a shared cache directory is not ambiguous between projects."""
    got = _default_cache_root({"XDG_STATE_HOME": tempfile.mkdtemp()})
    assert got.name == "GenomicVariantClassifier"


# ---- provenance ---------------------------------------------------------
def test_describe_records_the_new_domain():
    root = _make_repo(Path(tempfile.mkdtemp()))
    d = resolve_runtime_paths(project_root=root, environ={}).describe()
    for key in ("cache_root", "transaction_journal"):
        assert key in d and d[key], key


def test_describe_still_records_the_original_four():
    """The existing test asserts membership rather than equality, so adding
    keys is safe -- verified 2026-08-19 by reading it. This restates the
    guarantee here so a future rewrite of that test cannot silently drop them.
    """
    root = _make_repo(Path(tempfile.mkdtemp()))
    d = resolve_runtime_paths(project_root=root, environ={}).describe()
    for key in ("project_root", "artifact_root", "state_root", "reports_root",
                "literature_scout_state", "orchestrator_state"):
        assert key in d and d[key], key


def test_the_paths_object_is_still_frozen():
    root = _make_repo(Path(tempfile.mkdtemp()))
    p = resolve_runtime_paths(project_root=root, environ={})
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.cache_root = Path("/elsewhere")
