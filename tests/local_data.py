"""Tests of LOCAL DATA: git-ignored files a checkout may hold and a clone never does (DATA-GATED-TESTS-1, 2026-11-06).

Author: Monzia Moodie

MEASURED, not assumed. The data census of 2026-11-06 ran the whole suite on the owner's machine twice on the same commit and the
same locked environment -- in a clean clone (V1, the validation of set "ai") and in his checkout -- with a measurement that recorded
every path each test looked at. Exactly four tests changed outcome, all skipped in the clone and passed in the checkout, all because
of a git-ignored file. One needed only three protein sequences, which are now a committed fixture. The other three exist to check the
local data ITSELF -- the UniProt index's columns, the EVE corpus's resolution, the population stability index on the real Run-15
matrix -- so a fixture would defeat them. Those are declared with this module's decorator:

    from tests.local_data import local_data, repo_path

    @local_data("data/external/uniprot/uniprot_human_reviewed.parquet")
    def test_...():
        index = repo_path("data/external/uniprot/uniprot_human_reviewed.parquet")

What it does. Every path is taken RELATIVE TO THE REPOSITORY ROOT, never to the current folder (two of the three read
`Path("data/...")`, which a run started from another folder silently skipped). The test is skipped -- by pytest's own `skipif`, with
ONE reason for every such test, naming the absent paths -- unless every path exists. It also carries the `local_data` mark
(registered in pyproject.toml), so `pytest -m local_data` runs exactly these tests; a run's skip reasons (-rs) and its junit report
say which ran. It is a decorator and not a conftest hook ON PURPOSE: the session guards are the repository root's conftest.py's
alone, one implementation per hook (tests/unit/test_session_guard_placement.py), and a skip needs no hook.

tests/unit/test_local_data_contract.py requires: the set of tests using it is exactly the measured three; each declared path is
git-ignored and untracked (so the decorator cannot hide an ordinary missing file); `pytest.mark.local_data` is never written by hand
(a bare mark would carry no skip).
"""
from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath

import pytest

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The one skip reason, followed by the absent paths.
SKIP_REASON = "local data absent (git-ignored; held by a checkout, never by a clone): "


def check_relative(relative: str) -> str:
    """A declared local-data path: a repository-relative POSIX path with no '..' and no root; returned unchanged."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("a local-data path is a non-empty string, got {!r}".format(relative))
    pure = PurePosixPath(relative)
    if pure.is_absolute() or "\\" in relative or ".." in pure.parts or ":" in relative or str(pure) != relative:
        raise ValueError("a local-data path is relative to the repository root, in POSIX form, with no '..': {!r}".format(relative))
    return relative


def repo_path(relative: str) -> Path:
    """The declared local-data path, anchored at the repository root."""
    return REPO_ROOT.joinpath(*PurePosixPath(check_relative(relative)).parts)


def local_data(*paths: str):
    """Decorator: the test checks git-ignored local data at these repository-relative paths, and runs only where all exist."""
    if not paths:
        raise ValueError("local_data takes one or more repository-relative paths")
    declared = tuple(check_relative(p) for p in paths)
    absent = [p for p in declared if not repo_path(p).exists()]

    # pytest requires a reason with a boolean condition even when the condition is False; that one is never shown.
    reason = SKIP_REASON + ", ".join(absent) if absent else "local data present: " + ", ".join(declared)

    def decorate(function):
        function = pytest.mark.local_data(*declared)(function)
        return pytest.mark.skipif(bool(absent), reason=reason)(function)

    return decorate
