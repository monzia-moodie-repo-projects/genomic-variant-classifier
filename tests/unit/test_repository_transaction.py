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

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from genomic_variant_classifier.paths.runtime_paths import resolve_runtime_paths
from genomic_variant_classifier.transactions.repository_transaction import (
    RepositoryTransaction, Sensitivity, TransactionError,
    TransactionIntegrityError, TransactionRecoveryRequired, TransactionState,
    TransactionStateError, incomplete_transactions, is_secret_path,
    recover_transaction,
)

_SECRET_BYTES = b"GITHUB_TOKEN=ghp_" + b"q" * 36 + b"\n"


def _hermetic_git_env() -> dict:
    """Git that cannot be reconfigured by whoever happens to run the suite.

    TXTEST-FIXTURE-UNCHECKED-GIT-1, 2026-08-22. The previous fixture ran five
    git commands with NO check=True and no configuration isolation. If git were
    absent or a global setting interfered, `git init` failed silently, the
    fixture returned an ordinary directory, and `_assert_tree_clean` then met
    TRANSACTION-GIT-FAILURE-FAILS-OPEN-1 -- `_git` returns None and the
    assertion returns early. Several tests would have passed having proved
    nothing.

    A fixture that can silently manufacture a non-git directory while testing
    git-dependent invariants is not evidence.
    """
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _git_in(repo: Path, *args: str) -> None:
    """check=True. A git failure must be a FIXTURE failure, never a silent one."""
    subprocess.run(("git", "-C", str(repo), "-c", "core.excludesFile=", *args),
                   env=_hermetic_git_env(), capture_output=True, check=True,
                   timeout=60)


# ---------------------------------------------------------------------------
# TEST-LOCAL CERTIFICATION SCOPE.
#
# This is NOT the RepositoryCertificationSurface ruled by ADR-0002. That
# production abstraction remains UNIMPLEMENTED -- measured 2026-08-22, zero
# code references across src/, tests/ and scripts/.
# CERTIFICATION-SURFACE-UNIMPLEMENTED-1.
#
# This helper exists solely to make rollback topology observable here. Do not
# promote it to production scope without implementing the ADR properly.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TopologySnapshot:
    """Directories, and files as (relpath, digest) PAIRS.

    TRANSACTION-STATE-MODEL-INCOMPLETE-1: this module's tests modelled
    repository state as S = (F, J) -- selected file bytes and journal state --
    while the module contract claims S = (F, D, T, G, J). Directory topology
    was never in the model, so no test could fail on it.

    Files are pairs and not "path#digest" strings. MEASURED 2026-08-22: with
    the string encoding a PATCHED file appears in BOTH the added and the
    removed set, because its digest changed, and an assertion reading
    "files_added" as "newly existing paths" is then simply wrong. Pairs let
    the delta below answer the three separate questions a reader actually has.
    """

    directories: frozenset
    files: frozenset

    @classmethod
    def capture(cls, root: Path) -> "TopologySnapshot":
        root = Path(root).resolve()
        dirs, files = set(), set()
        for current, dirnames, filenames in os.walk(root):
            here = Path(current)
            if here == root:
                dirnames[:] = [d for d in dirnames if d != ".git"]
            for name in dirnames:
                dirs.add((here / name).relative_to(root).as_posix())
            for name in filenames:
                p = here / name
                files.add((p.relative_to(root).as_posix(),
                           hashlib.sha256(p.read_bytes()).hexdigest()))
        return cls(directories=frozenset(dirs), files=frozenset(files))


@dataclass(frozen=True)
class TopologyDelta:
    """Five directional facts. Direction lives in the SIGNATURE below.

    PROBE-DIRECTIONAL-LABEL-INVERSION-1, 2026-08-22: an earlier instrument
    computed this as a method called `after.diff(before)` whose body read
    `other - self`. Every label came out backwards, the verdict read a key
    that was always empty, and it reported "did not behave as predicted"
    while the evidence three lines above showed the defect exactly.

    `paths_modified` exists because content change is neither creation nor
    deletion, and collapsing it into either produces assertions that cannot
    be read correctly.
    """

    directories_added: tuple
    directories_removed: tuple
    paths_created: tuple
    paths_deleted: tuple
    paths_modified: tuple

    @property
    def unchanged(self) -> bool:
        return not (self.directories_added or self.directories_removed
                    or self.paths_created or self.paths_deleted
                    or self.paths_modified)


def topology_delta(before: TopologySnapshot,
                   after: TopologySnapshot) -> TopologyDelta:
    """BEFORE first, AFTER second. Identity is by DIGEST, never by mtime."""
    b, a = dict(before.files), dict(after.files)
    return TopologyDelta(
        directories_added=tuple(sorted(after.directories - before.directories)),
        directories_removed=tuple(sorted(before.directories - after.directories)),
        paths_created=tuple(sorted(set(a) - set(b))),
        paths_deleted=tuple(sorted(set(b) - set(a))),
        paths_modified=tuple(sorted(p for p in set(a) & set(b) if a[p] != b[p])),
    )


@pytest.fixture
def repo_and_journal(tmp_path):
    """A real git repository and a journal directory OUTSIDE it."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_bytes(b"x = 1\n")
    (repo / ".env").write_bytes(_SECRET_BYTES)
    subprocess.run(("git", "-c", "core.excludesFile=", "init", "--template=",
                    "-q", str(repo)), env=_hermetic_git_env(),
                   capture_output=True, check=True, timeout=60)
    _git_in(repo, "config", "user.email", "t@t")
    _git_in(repo, "config", "user.name", "t")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "v1")
    assert (repo / ".git").is_dir(), "the fixture did not produce a git repository"
    return repo, tmp_path / "journal"


# ---- the three invariants ----------------------------------------------
def test_a_committed_transaction_leaves_no_artefact(repo_and_journal):
    """THE INVARIANT. 255 artefacts accumulated because this was never true.

    TEST-CONTRACT-OVERCLAIM-1, repaired 2026-08-22 by STRENGTHENING rather than
    renaming. This test was named for a general artefact invariant and proved
    one specific filename family -- `.bak`, `.orig`, `.rej`, `.tmp` in a NAME --
    because it was born from the 255-artefact accumulation. A directory called
    `repository_records` matches none of those.

    Renaming it to `..._leaves_no_backup_shaped_artefact` would have made the
    name honest while leaving the weak predicate in place. Strengthening makes
    the ORIGINAL name true: no artefact of any kind, file or directory,
    survives a commit beyond the declared write set. Names are contracts, and
    the better repair is to honour the contract.
    """
    repo, journal = repo_and_journal
    before = TopologySnapshot.capture(repo)
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
    # TOPOLOGY. A commit may leave exactly what it declared and nothing else --
    # including no directory nobody asked for. `git status` cannot prove this:
    # git does not represent empty directories, so its strongest untracked
    # check reports nothing about them.
    delta = topology_delta(before, TopologySnapshot.capture(repo))
    assert delta.directories_added == (), delta.directories_added
    assert delta.directories_removed == (), delta.directories_removed
    assert delta.paths_created == ("src/new.py",), delta.paths_created
    assert delta.paths_deleted == (), delta.paths_deleted
    assert delta.paths_modified == ("src/mod.py",), delta.paths_modified


def test_a_failed_transaction_restores_the_repository_exactly(repo_and_journal):
    """TEST-CONTRACT-OVERCLAIM-1, repaired 2026-08-22 by strengthening.

    This test was named `..._exactly` and proved two file states plus the
    journal. It contained no directory assertion, no rglob, and no topology
    comparison, and its fixture creates src/ itself -- so the defect's
    precondition never arose here. MEASURED by falsification against the live
    module at 584c3fb: with a missing ancestor, files were restored and the
    directory survived, in-process AND through fresh-process recovery.

    `exactly` now means exactly: bytes, existence, and topology.
    """
    repo, journal = repo_and_journal
    before_bytes = (repo / "src" / "mod.py").read_bytes()
    before = TopologySnapshot.capture(repo)
    tid = None
    with pytest.raises(RuntimeError):
        with RepositoryTransaction(repo, journal) as tx:
            tid = tx.transaction_id
            tx.patch("src/mod.py", b"BROKEN\n")
            tx.create("src/new.py", b"y = 3\n")
            raise RuntimeError("the gate failed")
    assert (repo / "src" / "mod.py").read_bytes() == before_bytes
    assert not (repo / "src" / "new.py").exists(), "a created file survived"
    assert not (journal / tid).exists()
    assert topology_delta(before, TopologySnapshot.capture(repo)).unchanged, (
        "rollback restored file state but not repository topology")


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
def test_a_secret_target_is_REFUSED_by_default(repo_and_journal):
    """STEP 3B changed this contract deliberately.

    Step 3 permitted secret targets and held their preimages in memory only.
    That is security-conscious, and it creates an unavoidable consequence:

        secret changed -> process crashes -> old bytes vanished with process
        memory -> the durable journal cannot restore the secret

    One abstraction cannot promise both "no persistent secret preimage" and
    "arbitrary secret mutations are crash-recoverable" without another trusted
    store. So a generic source-tree patch transaction now REFUSES them.

    Credential provisioning is a different AUTHORITY -- environment injection,
    an operating-system credential store, a hosted secret store -- not a
    special case of a patch. That is the same principle the path domains
    follow: a lifecycle derives from the authority that owns the thing.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    with pytest.raises(TransactionError) as exc:
        tx.patch(".env", b"replaced\n")
    assert "credential-shaped" in str(exc.value)
    tx.rollback()
    assert (repo / ".env").read_bytes() == _SECRET_BYTES, "the secret was touched"


def test_creating_a_secret_path_is_also_refused(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    with pytest.raises(TransactionError):
        tx.create("config/api.key", b"nope\n")
    tx.rollback()


def test_an_opted_in_secret_target_writes_NO_preimage_to_disk(repo_and_journal):
    """The escape hatch, and its honest limitation.

    allow_secret_targets=True exists for a caller that has accepted the
    trade-off. Even then the preimage never reaches disk -- and therefore the
    target is NOT crash-recoverable, which the refusal message says plainly.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal, allow_secret_targets=True)
    tx.patch(".env", b"replaced\n")
    on_disk = [p for p in tx.preimages.rglob("*") if p.is_file()]
    assert not on_disk, on_disk
    text = tx.manifest_path.read_text(encoding="utf-8")
    assert "ghp_" not in text and "GITHUB_TOKEN" not in text
    entry = [t for t in tx.read_manifest()["targets"] if t["relpath"] == ".env"][0]
    assert entry["sensitivity"] == Sensitivity.SECRET.value
    assert entry["pre_sha256"] and len(entry["pre_sha256"]) == 64
    tx.rollback()
    assert (repo / ".env").read_bytes() == _SECRET_BYTES


def test_an_opted_in_secret_target_is_NOT_recoverable_after_a_crash(repo_and_journal):
    """The limitation, asserted rather than merely documented."""
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal, allow_secret_targets=True)
    tid = tx.transaction_id
    tx.patch(".env", b"replaced\n")
    del tx                                        # the memory-held preimage dies
    with pytest.raises(TransactionIntegrityError) as exc:
        recover_transaction(journal / tid)
    assert "NOT crash-recoverable" in str(exc.value)






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

    MEASURED 2026-08-20: an earlier version of this test patched
    Path.write_bytes and stopped working when step 3B introduced
    _write_durable, which uses open()+fsync instead. The test did not detect a
    regression -- it lost its interception point, and passed the write through
    honestly. A monkeypatch is a claim about HOW the code writes, so it must be
    re-aimed whenever that changes.
    """
    import genomic_variant_classifier.transactions.repository_transaction as rt
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")

    real = rt._write_durable

    def lying_write(path, data):
        # Report success, store something else. Only the post-write digest
        # can see this.
        if Path(path).name == "mod.py" and data == b"x = 1\n":
            return real(path, b"SILENTLY WRONG\n")
        return real(path, data)

    monkeypatch.setattr(rt, "_write_durable", lying_write)
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


# ---- STEP 3B: crash consistency ----------------------------------------
def test_the_target_is_PERSISTED_BEFORE_the_repository_is_touched(repo_and_journal):
    """WRITE-AHEAD. Step 3's DEMONSTRATED defect, asserted.

    Step 3 captured the preimage, wrote the new bytes, and only THEN persisted
    the target record. Reproduced 2026-08-20 by capturing, writing, and
    dropping the object:

        the file on disk        : b'MUTATED'
        targets in the manifest : []
        preimage files present  : ['mod.py']

    Discoverable and UNRECOVERABLE -- the preimage existed, but nothing
    recorded which file it belonged to.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx._transition(TransactionState.APPLYING)
    tx._write_ahead("src/mod.py")
    # The repository has NOT been touched yet.
    assert (repo / "src" / "mod.py").read_bytes() == b"x = 1\n"
    recorded = tx.read_manifest()["targets"]
    assert [t["relpath"] for t in recorded] == ["src/mod.py"]
    assert recorded[0]["pre_sha256"], "the digest must be durable before the write"
    assert recorded[0]["mutated"] is False
    tx.rollback()


def test_a_FAILED_rollback_is_recorded_as_recovery_required(repo_and_journal):
    """Step 3's second DEMONSTRATED defect, asserted.

    Step 3 set ROLLED_BACK before examining the failures, producing:

        file restored           : False
        journal retained        : True
        recorded state          : 'rolled_back'
        incomplete_transactions reports it: False
        a retry does anything   : False

    Unrestored, retained, invisible and unretryable -- four bad properties from
    one misordered assignment.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tid = tx.transaction_id
    tx.patch("src/mod.py", b"MUTATED\n")
    (tx.preimages / "src" / "mod.py").write_bytes(b"CORRUPTED\n")
    with pytest.raises(TransactionIntegrityError):
        tx.rollback()
    assert tx.state is TransactionState.RECOVERY_REQUIRED
    assert tx.directory.exists(), "the journal must survive"
    pending = incomplete_transactions(journal)
    assert any(p.get("transaction_id") == tid for p in pending), (
        "a failed rollback must remain DISCOVERABLE")


def test_a_failed_rollback_can_be_RETRIED(repo_and_journal):
    """RECOVERY_REQUIRED is not terminal. In step 3 it was, by accident."""
    repo, journal = repo_and_journal
    before = (repo / "src" / "mod.py").read_bytes()
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"MUTATED\n")
    (tx.preimages / "src" / "mod.py").write_bytes(b"CORRUPTED\n")
    with pytest.raises(TransactionIntegrityError):
        tx.rollback()
    (tx.preimages / "src" / "mod.py").write_bytes(before)   # operator repairs it
    tx.rollback()
    assert tx.state is TransactionState.ROLLED_BACK
    assert (repo / "src" / "mod.py").read_bytes() == before
    assert not tx.directory.exists()


def test_recovery_works_in_a_process_that_never_saw_the_transaction(repo_and_journal):
    """`del tx` proves DISCOVERY. Only a fresh process proves RECONSTRUCTION.

    Step 3's interruption test used `del tx`, which leaves Python alive with
    normal filesystem caches and no reconstruction boundary. This reads the
    manifest and preimages ALONE.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tid = tx.transaction_id
    tx.patch("src/mod.py", b"MUTATED\n")
    del tx
    assert (repo / "src" / "mod.py").read_bytes() == b"MUTATED\n"
    result = recover_transaction(journal / tid)
    assert result["action"] == "rolled_back", result
    assert (repo / "src" / "mod.py").read_bytes() == b"x = 1\n"
    assert not (journal / tid).exists()


def test_a_real_process_KILL_is_recoverable(repo_and_journal, tmp_path):
    """THE CLAIM STEP 3 COULD NOT MAKE.

    An exception-safe transaction is not necessarily a crash-safe transaction.
    All twelve of step 3's sabotage mutations were exception-driven; not one
    killed a process.

    This launches a second interpreter, kills it with SIGTERM AFTER the
    destructive write, and recovers in a THIRD process. MEASURED 2026-08-20
    across five kill points -- after preparation, after write-ahead, after
    mutation, after the mutation mark, and during the gate -- all five
    recovered with zero journals left behind.
    """
    import os
    import signal
    import sys as _sys
    repo, journal = repo_and_journal
    src_root = str(Path(__file__).resolve().parents[2] / "src")
    victim = tmp_path / "victim.py"
    victim.write_text(
        "import os, signal, sys\n"
        "sys.path.insert(0, {!r})\n".format(src_root) +
        "from genomic_variant_classifier.transactions.repository_transaction "
        "import RepositoryTransaction, TransactionState, _write_durable\n"
        "from pathlib import Path\n"
        "repo, journal = sys.argv[1], sys.argv[2]\n"
        "tx = RepositoryTransaction(repo, journal, require_clean_tree=False)\n"
        "print(tx.transaction_id, flush=True)\n"
        "tx._transition(TransactionState.APPLYING)\n"
        "tx._write_ahead('src/mod.py')\n"
        "_write_durable(Path(repo) / 'src' / 'mod.py', b'MUTATED')\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n",
        encoding="utf-8")
    out = subprocess.run([_sys.executable, "-B", str(victim), str(repo), str(journal)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode != 0, "the victim was supposed to die"
    assert (repo / "src" / "mod.py").read_bytes() == b"MUTATED"
    pending = incomplete_transactions(journal)
    assert len(pending) == 1, pending
    result = recover_transaction(pending[0]["directory"])
    assert result["action"] == "rolled_back", result
    assert (repo / "src" / "mod.py").read_bytes() == b"x = 1\n"
    assert not list(Path(journal).iterdir()), "a journal survived recovery"


def test_durable_writes_actually_call_fsync():
    """A STRUCTURAL assertion, and the reason it must be structural.

    MEASURED 2026-08-20: removing os.fsync from _write_durable left all 47
    tests passing. Sabotage J2 was genuinely undetectable behaviourally --
    without the fsync the bytes still reach the page cache, still read back
    correctly, and every assertion holds. Only a real power loss or a
    fault-injecting filesystem distinguishes the two.

    Write-ahead journaling is meaningless if the log is only in the page cache
    when the machine dies, so the guarantee matters even though no in-process
    test can observe it. Asserting the CALL is the strongest check available,
    and this comment exists so a future reader knows it is a deliberate
    substitute rather than a lazy one.
    """
    import ast
    import inspect
    import genomic_variant_classifier.transactions.repository_transaction as rt
    src = inspect.getsource(rt._write_durable)
    tree = ast.parse(src.lstrip())
    calls = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                base = getattr(f.value, "id", "")
                calls.add("{}.{}".format(base, f.attr) if base else f.attr)
    assert "os.fsync" in calls, (
        "_write_durable does not call os.fsync; a journal held only in the "
        "page cache is not a write-ahead log. calls seen: {}".format(sorted(calls)))
    assert "fh.flush" in calls or "flush" in calls, sorted(calls)


def test_the_manifest_store_also_writes_atomically():
    """The manifest is half the journal. JsonStateStore was chosen because it
    already does mkstemp + fsync + os.replace in the SAME directory -- verified
    against the real module rather than a stub, since a stub agrees with you."""
    import ast
    import inspect
    from genomic_variant_classifier.state import json_state_store as jss
    src = inspect.getsource(jss.JsonStateStore.save)
    names = set()
    for n in ast.walk(ast.parse(src.lstrip())):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            base = getattr(n.func.value, "id", "")
            names.add("{}.{}".format(base, n.func.attr) if base else n.func.attr)
    for required in ("os.fsync", "os.replace"):
        assert required in names, (required, sorted(names))


# ---- STEP 3B: preconditions --------------------------------------------
def test_a_dirty_working_tree_is_refused_at_preparation(repo_and_journal):
    """HEAD not moving says nothing about a concurrent editor. A transaction
    certifies a NAMED SET of files; step 3 had only the weaker invariant."""
    repo, journal = repo_and_journal
    (repo / "src" / "stray.py").write_bytes(b"unowned = 1\n")
    with pytest.raises(TransactionError) as exc:
        RepositoryTransaction(repo, journal)
    assert "uncommitted" in str(exc.value)


def test_an_unresolved_journal_blocks_a_new_transaction(repo_and_journal):
    """An interrupted installer must be reconciled, not stepped over."""
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.patch("src/mod.py", b"A\n")
    del tx
    with pytest.raises(TransactionRecoveryRequired) as exc:
        RepositoryTransaction(repo, journal, require_clean_tree=False)
    assert "unresolved" in str(exc.value)


def test_recovering_an_already_terminal_journal_is_a_no_op(tmp_path):
    """Recovery must be idempotent for anything already resolved."""
    import json
    d = tmp_path / "txn"
    (d / "preimages").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "schema": "gvc.repository-transaction", "schema_version": 2,
        "generation": 1, "updated_at": "x",
        "values": {"transaction_id": "abc", "state": "committed",
                   "repo_root": str(tmp_path), "targets": []}}), encoding="utf-8")
    result = recover_transaction(d)
    assert result["action"] == "none"
    assert d.exists(), "a terminal journal must not be destroyed by a probe"


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


# ---- STEP 3C: repository TOPOLOGY --------------------------------------
# TRANSACTION-STATE-MODEL-INCOMPLETE-1, 2026-08-22.
#
# Every case below FAILED against the module as it stood at 584c3fb, measured
# by falsification before any repair was written. The control -- a target under
# an existing parent -- passed throughout, which is what makes this a coverage
# gap rather than "rollback is broken".
#
#     parent already exists        topology restored
#     one missing ancestor         src/pkg survived
#     three missing ancestors      src/a, src/a/b, src/a/b/c survived
#     two targets, one new parent  src/pkg survived, ONCE
#     fresh-process recovery       src/pkg survived
#
# and `git status --porcelain=v2 --untracked-files=all` reported NOTHING
# throughout, because git does not represent empty directories.


def test_rollback_removes_one_created_ancestor(repo_and_journal):
    repo, journal = repo_and_journal
    before = TopologySnapshot.capture(repo)
    with pytest.raises(RuntimeError):
        with RepositoryTransaction(repo, journal) as tx:
            tx.create("src/pkg/mod.py", b"y = 3\n")
            raise RuntimeError("the gate failed")
    assert topology_delta(before, TopologySnapshot.capture(repo)).unchanged


def test_rollback_removes_three_created_ancestors_deepest_first(repo_and_journal):
    """`mkdir(parents=True)` orphaned all three levels, not merely the last."""
    repo, journal = repo_and_journal
    before = TopologySnapshot.capture(repo)
    with pytest.raises(RuntimeError):
        with RepositoryTransaction(repo, journal) as tx:
            tx.create("src/a/b/c/mod.py", b"y = 3\n")
            raise RuntimeError("the gate failed")
    delta = topology_delta(before, TopologySnapshot.capture(repo))
    assert delta.unchanged, delta.directories_added
    assert not (repo / "src" / "a").exists()
    assert (repo / "src").is_dir(), "the PRE-EXISTING parent must survive"


def test_two_targets_sharing_a_created_parent_own_it_once(repo_and_journal):
    """Directory creation belongs to the TRANSACTION, not to a target.

    Competing owners would make removal order undefined and could remove a
    directory the other target still needs.
    """
    repo, journal = repo_and_journal
    before = TopologySnapshot.capture(repo)
    tx = RepositoryTransaction(repo, journal)
    tx.create("src/pkg/one.py", b"a = 1\n")
    tx.create("src/pkg/two.py", b"b = 2\n")
    recorded = [i["relpath"]
                for i in tx.read_manifest()["directory_creation_intents"]]
    assert recorded == ["src/pkg"], recorded
    tx.rollback()
    assert topology_delta(before, TopologySnapshot.capture(repo)).unchanged


def test_a_preexisting_directory_is_never_removed(repo_and_journal):
    """The control case. It passed before the repair and must keep passing."""
    repo, journal = repo_and_journal
    before = TopologySnapshot.capture(repo)
    with pytest.raises(RuntimeError):
        with RepositoryTransaction(repo, journal) as tx:
            tx.create("src/new.py", b"y = 3\n")
            raise RuntimeError("the gate failed")
    delta = topology_delta(before, TopologySnapshot.capture(repo))
    assert delta.unchanged
    assert (repo / "src").is_dir()


def test_directory_intents_are_PERSISTED_BEFORE_the_mutation(repo_and_journal):
    """WRITE-AHEAD, for topology.

    Recovery metadata describing a mutation must be durable BEFORE that
    mutation becomes observable -- the same discipline
    test_the_target_is_PERSISTED_BEFORE_the_repository_is_touched already
    enforces for file bytes. A crash between mkdir and persistence would
    otherwise recreate the residue class this repair removes.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx._transition(TransactionState.APPLYING)
    intents = tx._record_directory_intents(repo / "src" / "a" / "b" / "mod.py")
    # NOTHING has been created yet.
    assert not (repo / "src" / "a").exists()
    recorded = [i["relpath"]
                for i in tx.read_manifest()["directory_creation_intents"]]
    assert recorded == ["src/a", "src/a/b"], recorded
    assert [i.relpath for i in intents] == ["src/a", "src/a/b"]
    tx.rollback()


def test_foreign_content_prevents_a_CLEAN_rollback(repo_and_journal):
    """A safe failure is still a failure.

    "Do not destroy someone else's state" and "the pre-state was restored" are
    SEPARATE predicates. The directory must not be deleted, and rollback must
    not claim success either.
    """
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.create("src/pkg/mod.py", b"y = 3\n")
    (repo / "src" / "pkg" / "foreign.txt").write_bytes(b"not ours\n")
    with pytest.raises(TransactionIntegrityError) as exc:
        tx.rollback()
    assert "foreign_content_present" in str(exc.value)
    assert tx.state is TransactionState.RECOVERY_REQUIRED
    assert (repo / "src" / "pkg" / "foreign.txt").read_bytes() == b"not ours\n"
    assert tx.directory.exists(), "the journal must survive for a retry"


def test_a_recorded_directory_replaced_by_a_file_is_refused(repo_and_journal):
    repo, journal = repo_and_journal
    tx = RepositoryTransaction(repo, journal)
    tx.create("src/pkg/mod.py", b"y = 3\n")
    (repo / "src" / "pkg" / "mod.py").unlink()
    (repo / "src" / "pkg").rmdir()
    (repo / "src" / "pkg").write_bytes(b"now a file\n")
    with pytest.raises(TransactionIntegrityError) as exc:
        tx.rollback()
    assert "directory_intent_type_changed" in str(exc.value)
    assert (repo / "src" / "pkg").read_bytes() == b"now a file\n"


def test_fresh_process_recovery_restores_topology(repo_and_journal):
    """recover_transaction() reads the manifest ALONE.

    If the directory is not recorded there, recovery cannot remove it either --
    which is why the intents are persisted rather than held in memory. This is
    the case that proves the gap was in the DURABLE model.
    """
    repo, journal = repo_and_journal
    before = TopologySnapshot.capture(repo)
    tx = RepositoryTransaction(repo, journal)
    tid = tx.transaction_id
    tx.create("src/pkg/mod.py", b"y = 3\n")
    del tx                                        # the process dies here
    result = recover_transaction(journal / tid)
    assert result["action"] == "rolled_back", result
    assert topology_delta(before, TopologySnapshot.capture(repo)).unchanged


def test_a_legacy_manifest_without_directory_intents_still_recovers(tmp_path):
    """Journals written before this repair must not become unrecoverable."""
    d = tmp_path / "txn"
    (d / "preimages").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "schema": "gvc.repository-transaction", "schema_version": 2,
        "generation": 1, "updated_at": "x",
        "values": {"transaction_id": "legacy", "state": "applying",
                   "repo_root": str(tmp_path), "targets": []}}),
        encoding="utf-8")
    result = recover_transaction(d)
    assert result["action"] == "rolled_back", result
    assert not d.exists()


def test_topology_restoration_is_idempotent(repo_and_journal):
    """RECOVERY_REQUIRED is not terminal, so a retry must complete."""
    repo, journal = repo_and_journal
    before = TopologySnapshot.capture(repo)
    tx = RepositoryTransaction(repo, journal)
    tx.create("src/pkg/mod.py", b"y = 3\n")
    (repo / "src" / "pkg" / "foreign.txt").write_bytes(b"not ours\n")
    with pytest.raises(TransactionIntegrityError):
        tx.rollback()
    (repo / "src" / "pkg" / "foreign.txt").unlink()   # the operator clears it
    tx.rollback()
    assert tx.state is TransactionState.ROLLED_BACK
    assert topology_delta(before, TopologySnapshot.capture(repo)).unchanged


def test_a_process_KILL_after_mkdir_leaves_a_recoverable_topology(
        repo_and_journal, tmp_path):
    """Crash window: intent persisted, directories created, file NOT written.

    An exception-safe transaction is not necessarily a crash-safe one. This
    kills a second interpreter after the directories exist and recovers in a
    third, which is the only way to prove the manifest alone suffices.
    """
    import signal
    import sys as _sys
    repo, journal = repo_and_journal
    before = TopologySnapshot.capture(repo)
    src_root = str(Path(__file__).resolve().parents[2] / "src")
    victim = tmp_path / "victim_topology.py"
    victim.write_text(
        "import os, signal, sys\n"
        "sys.path.insert(0, {!r})\n".format(src_root) +
        "from genomic_variant_classifier.transactions.repository_transaction "
        "import RepositoryTransaction, TransactionState, "
        "_materialize_directory_intents\n"
        "from pathlib import Path\n"
        "repo, journal = sys.argv[1], sys.argv[2]\n"
        "tx = RepositoryTransaction(repo, journal, require_clean_tree=False)\n"
        "print(tx.transaction_id, flush=True)\n"
        "tx._transition(TransactionState.APPLYING)\n"
        "tx._write_ahead('src/pkg/mod.py')\n"
        "i = tx._record_directory_intents(Path(repo) / 'src' / 'pkg' / 'mod.py')\n"
        "_materialize_directory_intents(Path(repo), i)\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n",
        encoding="utf-8")
    out = subprocess.run([_sys.executable, "-B", str(victim), str(repo),
                          str(journal)], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode != 0, "the victim was supposed to die"
    assert (repo / "src" / "pkg").is_dir(), "the directory should exist yet"
    pending = incomplete_transactions(journal)
    assert len(pending) == 1, pending
    result = recover_transaction(pending[0]["directory"])
    assert result["action"] == "rolled_back", result
    assert topology_delta(before, TopologySnapshot.capture(repo)).unchanged
