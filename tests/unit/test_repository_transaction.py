"""A transaction may temporarily own a rollback state. The repository never does.

INSTALLER-TRANSACTION-1 step 3.

WHY THIS EXISTS
Every installer wrote `<target>.bak_<timestamp>` beside the file it edited and
none removed it on success. MEASURED 2026-08-19, in two sweeps:

    148 artefacts, 17,640,928 bytes   matching `*.bak_*`
    107 artefacts,  2,828,345 bytes   matching `*.bak`, `*.orig`, `*.rej`

The second sweep existed because the first tool scanned ONE shape of four and
reported zero. One of the 148 was a credential-bearing `.env` backup, and the
manifest recording its retirement was itself overwritten by a later routine run.

Three defects, one root: rollback state inside the repository, under names no
single filter reliably covers, cleaned up only when somebody remembered.

THE INVARIANTS THESE TESTS ASSERT
    success       the repository keeps the changes and NOTHING else
    failure       the repository is byte-identical to how it was found
    interruption  a journal survives OUTSIDE the repository in a non-terminal
                  state, and is discoverable

WHAT IT REUSES
JsonStateStore for the manifest -- atomic write via mkstemp + fsync +
os.replace in the same directory, schema identification, a generation counter,
and load() that RAISES on damage. Measured 2026-08-19:
`save(values, *, generation=None)` returns the generation written.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from genomic_variant_classifier.paths.runtime_paths import resolve_runtime_paths
from genomic_variant_classifier.transactions.repository_transaction import (
    RepositoryTransaction, Sensitivity, TransactionError,
    TransactionIntegrityError, TransactionState, TransactionStateError,
    incomplete_transactions, is_secret_path,
)

_SECRET_BYTES = b"GITHUB_TOKEN=ghp_" + b"q" * 36 + b"\n"


@pytest.fixture
def repo_and_journal(tmp_path):
    """A real git repository and a journal directory OUTSIDE it."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_bytes(b"x = 1\n")
    (repo / ".env").write_bytes(_SECRET_BYTES)
    for cmd in (["git", "init", "-q"],
                ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"],
                ["git", "add", "-A"],
                ["git", "commit", "-qm", "v1"]):
        subprocess.run(cmd, cwd=str(repo), capture_output=True, timeout=60)
    return repo, tmp_path / "journal"


# ---- the three invariants ----------------------------------------------
def test_a_committed_transaction_leaves_no_artefact(repo_and_journal):
    """THE INVARIANT. 255 artefacts accumulated because this was never true."""
    repo, journal = repo_and_journal
    with RepositoryTransaction(repo, journal) as tx:
        tid = tx.transaction_id
        tx.patch("src/mod.py", b"x = 2\n")
        tx.create("src/new.py", b"y = 3\n")
        tx.verify(lambda root: True)
        tx.commit()
    assert (repo / "src" / "mod.py").read_bytes() == b"x = 2\n"
    assert (repo / "src" / "new.py").exists()
    assert not (journal / tid).exists(), "the journal survived a commit"
    strays = [p for p in repo.rglob("*")
              if any(s in p.name for s in (".bak", ".orig", ".rej", ".tmp"))]
    assert not strays, strays


def test_a_failed_transaction_restores_the_repository_exactly(repo_and_journal):
    repo, journal = repo_and_journal
    before = (repo / "src" / "mod.py").read_bytes()
    tid = None
    with pytest.raises(RuntimeError):
        with RepositoryTransaction(repo, journal) as tx:
            tid = tx.transaction_id
            tx.patch("src/mod.py", b"BROKEN\n")
            tx.create("src/new.py", b"y = 3\n")
            raise RuntimeError("the gate failed")
    assert (repo / "src" / "mod.py").read_bytes() == before
    assert not (repo / "src" / "new.py").exists(), "a created file survived"
    assert not (journal / tid).exists()


def test_an_interrupted_transaction_is_discoverable(repo_and_journal):
    """The case a `.bak_` sibling handled only by being noticed."""
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")
    tid = tx.transaction_id
    del tx                                    # the process dies here
    pending = incomplete_transactions(journal)
    mine = [p for p in pending if p.get("transaction_id") == tid]
    assert len(mine) == 1, pending
    assert mine[0]["state"] == TransactionState.APPLYING.value
    assert (journal / tid).exists(), "the journal must survive for recovery"


# ---- the journal cannot live in the repository -------------------------
def test_a_journal_inside_the_repository_is_REFUSED(repo_and_journal):
    """The defect this class removes, asserted directly."""
    repo, _journal = repo_and_journal
    with pytest.raises(TransactionError) as exc:
        RepositoryTransaction(repo, repo / ".gvc-journal")
    assert "inside the repository" in str(exc.value)


def test_the_repository_root_itself_is_refused_as_a_journal(repo_and_journal):
    repo, _ = repo_and_journal
    with pytest.raises(TransactionError):
        RepositoryTransaction(repo, repo)


# ---- secret targets ----------------------------------------------------
def test_a_secret_target_writes_NO_preimage_to_disk(repo_and_journal):
    """The 2026-08-15 incident began with a general-purpose text workflow
    operating near a credential. A secret preimage on disk is that again."""
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch(".env", b"GITHUB_TOKEN=replaced\n")
    on_disk = [p for p in tx.preimages.rglob("*") if p.is_file()]
    assert not on_disk, on_disk
    tx.rollback()
    assert (repo / ".env").read_bytes() == _SECRET_BYTES


def test_the_manifest_records_sensitivity_but_never_content(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch(".env", b"replaced\n")
    text = tx.manifest_path.read_text(encoding="utf-8")
    assert "ghp_" not in text
    assert "GITHUB_TOKEN" not in text
    values = tx.read_manifest()
    entry = [t for t in values["targets"] if t["relpath"] == ".env"][0]
    assert entry["sensitivity"] == Sensitivity.SECRET.value
    assert entry["pre_sha256"] and len(entry["pre_sha256"]) == 64
    tx.rollback()


def test_a_secret_target_is_still_restored_on_rollback(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch(".env", b"replaced\n")
    tx.rollback()
    assert (repo / ".env").read_bytes() == _SECRET_BYTES


@pytest.mark.parametrize("name", [
    ".env", "id_rsa", "server.pem", "api.key", "credentials.json",
    "token.txt", "secrets.yaml",
])
def test_the_secret_classifier_recognises_every_canary(name):
    """A classifier that recognises nothing produces a clean run and an on-disk
    copy of a credential. Sabotage confirmed the absence of any error."""
    assert is_secret_path(name), name


# ---- the state machine -------------------------------------------------
def test_states_are_monotonic(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    assert tx.state is TransactionState.PREPARED
    tx.patch("src/mod.py", b"A\n")
    assert tx.state is TransactionState.APPLYING
    tx.verify(lambda root: True)
    assert tx.state is TransactionState.VERIFYING
    tx.commit()
    assert tx.state is TransactionState.COMMITTED


@pytest.mark.parametrize("start,forbidden", [
    (TransactionState.COMMITTED, TransactionState.APPLYING),
    (TransactionState.ROLLED_BACK, TransactionState.APPLYING),
    (TransactionState.VERIFYING, TransactionState.PREPARED),
    (TransactionState.APPLYING, TransactionState.PREPARED),
    (TransactionState.PREPARED, TransactionState.COMMITTED),
    (TransactionState.PREPARED, TransactionState.VERIFYING),
])
def test_the_transition_TABLE_refuses_illegal_moves(repo_and_journal,
                                                    start, forbidden):
    """The state machine itself, exercised directly.

    MEASURED 2026-08-19: every other state test was satisfied by a method's OWN
    guard -- patch-after-commit raises from _capture, commit-from-prepared from
    commit, verify-in-the-wrong-state from verify. So _ALLOWED was never
    consulted, and sabotage H6 (permitting every transition) passed all 27
    tests.

    A transition table with no test is a comment.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx._state = start
    with pytest.raises(TransactionStateError) as exc:
        tx._transition(forbidden)
    assert "not a legal transition" in str(exc.value)
    tx._state = TransactionState.ROLLED_BACK          # leave it terminal


def test_the_legal_path_through_the_table_is_permitted(repo_and_journal):
    """The table must not refuse everything either."""
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    for nxt in (TransactionState.APPLYING, TransactionState.VERIFYING,
                TransactionState.COMMITTED):
        tx._transition(nxt)
        assert tx.state is nxt


def test_no_write_is_possible_after_commit(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")
    tx.commit()
    with pytest.raises(TransactionStateError):
        tx.patch("src/mod.py", b"B\n")


def test_commit_from_prepared_is_refused(repo_and_journal):
    """Nothing was applied, so there is nothing to commit."""
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    with pytest.raises(TransactionStateError):
        tx.commit()


def test_verify_requires_the_applying_state(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    with pytest.raises(TransactionStateError):
        tx.verify(lambda root: True)


def test_rollback_is_idempotent(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")
    tx.rollback()
    tx.rollback()                              # must not raise
    assert tx.state is TransactionState.ROLLED_BACK


# ---- integrity ---------------------------------------------------------
def test_a_corrupted_preimage_is_REFUSED_and_the_journal_retained(repo_and_journal):
    """Restoring from a preimage that fails its own digest would write unknown
    bytes. Stopping and keeping the journal is the safer failure."""
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")
    (tx.preimages / "src" / "mod.py").write_bytes(b"TAMPERED\n")
    with pytest.raises(TransactionIntegrityError):
        tx.rollback()
    assert tx.directory.exists(), "the journal must survive a failed rollback"
    # THE DISTINGUISHING ASSERTION. There are two digest checks: one BEFORE
    # writing the preimage to the target, one after. Only the first prevents
    # corrupted bytes from reaching the file at all.
    #
    # MEASURED 2026-08-19: removing the pre-write check left the post-write
    # check to raise, so the test still passed -- while the target had already
    # been overwritten with TAMPERED bytes. Sabotage H8 was reported as
    # undetected because nothing asserted the file's contents.
    assert (repo / "src" / "mod.py").read_bytes() == b"A\n", (
        "corrupted preimage bytes reached the target before being detected")


def test_the_POST_WRITE_digest_catches_a_lying_filesystem(repo_and_journal,
                                                          monkeypatch):
    """The second digest check, which the first cannot stand in for.

    The pre-write check proves the PREIMAGE is intact. The post-write check
    proves the bytes actually LANDED -- a truncated write, a full disk, or a
    filesystem that reports success and stores something else.

    MEASURED 2026-08-19: removing the post-write check left all 35 tests
    passing, because nothing simulated a write that silently failed. Sabotage
    H11 was reported as undetected for that reason.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")

    real_write = Path.write_bytes

    def lying_write(self, data):
        # Report success, store something else. Only the post-write digest
        # can see this.
        if self.name == "mod.py" and b"x = 1" in data:
            return real_write(self, b"SILENTLY WRONG\n")
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", lying_write)
    with pytest.raises(TransactionIntegrityError) as exc:
        tx.rollback()
    assert "after restoration" in str(exc.value), str(exc.value)
    monkeypatch.undo()


def test_the_secret_canary_guard_refuses_a_weakened_classifier(monkeypatch):
    """A guard on the guard.

    SECRET_PATTERNS emptied is caught by many tests. But removing the CALL to
    _assert_secret_detection_intact changes nothing while the patterns are
    intact -- so the guard itself needs a direct test, or it is only
    incidentally covered.

    MEASURED 2026-08-19: sabotage H12 removed the call and all 35 tests passed.
    """
    import genomic_variant_classifier.transactions.repository_transaction as rt
    monkeypatch.setattr(rt, "SECRET_PATTERNS", ())
    with pytest.raises(TransactionError) as exc:
        rt._assert_secret_detection_intact()
    assert "does not recognise" in str(exc.value)
    monkeypatch.undo()
    rt._assert_secret_detection_intact()       # intact again, must not raise


def test_constructing_a_transaction_runs_the_canary_guard(repo_and_journal,
                                                          monkeypatch):
    """And the guard must actually be WIRED into construction."""
    import genomic_variant_classifier.transactions.repository_transaction as rt
    repo, journal = repo_and_journal
    monkeypatch.setattr(rt, "SECRET_PATTERNS", ())
    with pytest.raises(TransactionError):
        rt.RepositoryTransaction(repo, journal)
    monkeypatch.undo()


def test_a_target_outside_the_repository_is_refused(repo_and_journal, tmp_path):
    """The escaping file must EXIST, or the test passes for the wrong reason.

    MEASURED 2026-08-19: with the containment guard removed, `../escape.py`
    simply did not exist, so patch() raised "does not exist; use create()" --
    the same exception TYPE, a completely different reason, and the test still
    passed. Sabotage H9 was reported as undetected because of this.

    Creating the file first means only the containment check can refuse it.
    """
    repo, journal = repo_and_journal
    outside = tmp_path / "escape.py"
    outside.write_bytes(b"outside = 1\n")
    assert outside.exists() and repo not in outside.parents
    tx = RepositoryTransaction(repo, journal)
    with pytest.raises(TransactionError) as exc:
        tx.patch("../escape.py", b"nope\n")
    assert "outside the repository" in str(exc.value), str(exc.value)
    assert outside.read_bytes() == b"outside = 1\n", "the outside file was written"
    tx.rollback()


def test_creating_outside_the_repository_is_also_refused(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    with pytest.raises(TransactionError) as exc:
        tx.create("../escape_new.py", b"nope\n")
    assert "outside the repository" in str(exc.value)
    tx.rollback()


def test_a_moved_HEAD_is_detected_at_commit(repo_and_journal):
    """Preimages describe the tree as it was at preparation."""
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")
    (repo / "src" / "other.py").write_bytes(b"o = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "v2"], cwd=str(repo), capture_output=True)
    with pytest.raises(TransactionIntegrityError) as exc:
        tx.commit()
    assert "HEAD moved" in str(exc.value)
    tx.rollback()


def test_patching_a_missing_file_is_refused(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    with pytest.raises(TransactionError):
        tx.patch("src/absent.py", b"nope\n")
    tx.rollback()


def test_creating_an_existing_file_is_refused(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    with pytest.raises(TransactionError):
        tx.create("src/mod.py", b"nope\n")
    tx.rollback()


# ---- the manifest ------------------------------------------------------
def test_the_manifest_is_enveloped_and_schema_identified(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")
    raw = json.loads(tx.manifest_path.read_text(encoding="utf-8"))
    assert raw["schema"] == "gvc.repository-transaction"
    assert raw["generation"] >= 1
    assert "values" in raw
    tx.rollback()


def test_the_journal_lives_under_the_runtime_cache_root():
    """Wiring, not just capability: the default location is the fifth path
    domain added at 05f1a72, which resolves OUTSIDE any repository."""
    paths = resolve_runtime_paths()
    assert not str(paths.transaction_journal).startswith(str(paths.project_root))
    assert paths.transaction_journal == paths.cache_root / "transactions"
