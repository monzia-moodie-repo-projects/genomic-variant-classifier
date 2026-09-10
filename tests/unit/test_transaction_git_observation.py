"""A transaction cannot certify a repository it was unable to observe.

Author: Monzia Moodie

WHAT THIS CLOSES
================
TRANSACTION-GIT-FAILURE-FAILS-OPEN-1, recorded CONFIRMED and unaddressed in
ADR-0003 and defined in SESSION_2026-08-21_to_08-22 as "`_git` returns None on
failure and both clean-tree and head-unmoved assertions return early,
silently".

MEASURED 2026-09-08 against the payload unit T installed at
fdb5473aaf3f7a8d594da10b5f793d295dffe610
(fdc02af8c111573f48c838a113b432fb60e50396dcd37989783fb2ad70be7687), in a
repository holding one unowned untracked file:

    git AVAILABLE    -> refused, TransactionError, working tree dirty
    git UNAVAILABLE  -> CONSTRUCTED, self._head is None

One git failure at construction disabled BOTH assertions for the transaction's
whole lifetime: `_assert_tree_clean` returned early because `_git` yielded
None, and `_assert_head_unmoved` returned early because `self._head` was None.

THE POSITIVE CONTROL IS NOT OPTIONAL
====================================
`test_a_clean_repository_still_constructs_normally` exists because every other
test here is a refusal. A module that refused everything would satisfy them
all and be useless.

THE FIXTURE RUNS GIT CHECKED
============================
Following tests/unit/test_repository_transaction.py, which records
TXTEST-FIXTURE-UNCHECKED-GIT-1: a fixture that can silently manufacture a
non-git directory while testing git-dependent invariants is not evidence.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from genomic_variant_classifier.transactions.repository_transaction import (
    RepositoryTransaction, TransactionError, TransactionGitUnavailable)


def _run(repo, *args):
    """Git that cannot be reconfigured by whoever happens to run the suite."""
    environment = dict(os.environ)
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_SYSTEM"] = os.devnull
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                   env=environment, check=True)


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_SYSTEM"] = os.devnull
    subprocess.run(["git", "init", "-q", str(repo)], capture_output=True,
                   env=environment, check=True)
    _run(repo, "config", "user.email", "t@t")
    _run(repo, "config", "user.name", "t")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "v1")
    return repo


@pytest.fixture
def dirty_repository(repository):
    (repository / "UNOWNED.txt").write_text("unowned\n", encoding="utf-8")
    return repository


def _without_git(monkeypatch):
    monkeypatch.setenv("PATH", os.path.join(os.sep, "nonexistent"))


def test_git_unavailable_at_start_refuses_before_authorizing_mutation(
        dirty_repository, tmp_path, monkeypatch):
    """The measured exposure. The preimage CONSTRUCTED here."""
    _without_git(monkeypatch)
    with pytest.raises(TransactionGitUnavailable):
        RepositoryTransaction(dirty_repository, tmp_path / "journal")


def test_a_refused_construction_creates_no_journal(dirty_repository, tmp_path,
                                                   monkeypatch):
    """A refusal must leave the filesystem exactly as it found it."""
    journal = tmp_path / "journal"
    _without_git(monkeypatch)
    with pytest.raises(TransactionGitUnavailable):
        RepositoryTransaction(dirty_repository, journal)
    assert not journal.exists()


def test_a_nonzero_exit_with_empty_output_is_not_success(repository, tmp_path,
                                                         monkeypatch):
    """The previous body checked only whether a process object existed, so it
    could not tell an empty successful result from a failed one."""
    real = subprocess.run

    def failing(command, **kwargs):
        if command and command[0] == "git":
            return subprocess.CompletedProcess(command, 1, "", "simulated")
        return real(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing)
    with pytest.raises(TransactionGitUnavailable) as caught:
        RepositoryTransaction(repository, tmp_path / "journal")
    assert "exited 1" in str(caught.value)


def test_a_failed_clean_tree_query_grants_no_authorization(
        repository, tmp_path, monkeypatch):
    """`status --porcelain` failing must not satisfy the clean-tree
    invariant."""
    real = subprocess.run

    def selective(command, **kwargs):
        if command and command[0] == "git" and "status" in command:
            raise OSError("simulated status failure")
        return real(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", selective)
    with pytest.raises(TransactionGitUnavailable):
        RepositoryTransaction(repository, tmp_path / "journal")


def test_a_failed_head_query_grants_no_commitment_authorization(
        repository, tmp_path, monkeypatch):
    """Measured on the preimage: with `self._head` None, _assert_head_unmoved
    returned early and commit() proceeded."""
    transaction = RepositoryTransaction(repository, tmp_path / "journal")
    try:
        assert transaction._head is not None
        _without_git(monkeypatch)
        with pytest.raises(TransactionGitUnavailable):
            transaction._assert_head_unmoved()
    finally:
        monkeypatch.undo()
        transaction.rollback(reason="test complete")


def test_a_dirty_tree_still_refuses_with_its_original_error(dirty_repository,
                                                           tmp_path):
    """The repair must not swallow the case the module already handled."""
    with pytest.raises(TransactionError) as caught:
        RepositoryTransaction(dirty_repository, tmp_path / "journal")
    assert "uncommitted entr" in str(caught.value)
    assert not isinstance(caught.value, TransactionGitUnavailable)


def test_a_clean_repository_still_constructs_normally(repository, tmp_path):
    """POSITIVE CONTROL. Without it a module refusing everything would pass
    every other test in this file."""
    transaction = RepositoryTransaction(repository, tmp_path / "journal")
    try:
        assert transaction._head is not None
        assert len(transaction._head) == 40
    finally:
        transaction.rollback(reason="test complete")


def test_unobservable_is_a_distinct_condition_from_a_dirty_tree():
    """A caller must be able to tell 'git says the tree is dirty' from 'git
    could not be asked'. It is NOT the journal-persistence stop condition:
    persistence uncertainty and observation failure are different, and
    broadening that flag could break legitimate recovery."""
    assert issubclass(TransactionGitUnavailable, TransactionError)
    assert TransactionGitUnavailable is not TransactionError
