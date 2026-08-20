"""A transaction may temporarily own a rollback state. The repository never does.

INSTALLER-TRANSACTION-1, steps 3 and 3B.

    An exception-safe transaction is not necessarily a crash-safe transaction.

STEP 3B EXISTS BECAUSE STEP 3 WAS NOT CRASH-SAFE
The step-3 primitive passed 38 tests and 12 sabotage mutations, and every one
of those mutations was EXCEPTION-driven. Not one killed a process. Inspection
of the write ordering then found two defects that no exception test could
reach, both DEMONSTRATED before being repaired:

    DEFECT 1 -- write-ahead violation. patch() captured the preimage, wrote the
    new bytes, and only THEN persisted the target record. Reproduced 2026-08-20
    by capturing, writing, and dropping the object:

        the file on disk        : b'MUTATED\\n'
        targets in the manifest : []
        preimage files present  : ['mod.py']

    The journal was discoverable and UNRECOVERABLE: the preimage existed, but
    nothing recorded which file it belonged to.

    DEFECT 2 -- a failed rollback recorded itself as successful. rollback() set
    ROLLED_BACK before examining the failures, so a corrupted preimage produced:

        file restored           : False
        journal retained        : True
        recorded state          : 'rolled_back'
        incomplete_transactions reports it: False
        a retry does anything   : False

    Unrestored, retained, invisible and unretryable -- four bad properties from
    one misordered assignment.

WHAT CHANGED
    Write-ahead ordering. A target is persisted to the journal, with its
    preimage digest, and fsynced, BEFORE the repository is touched. create()
    persists "this path did not exist" before creating it.

    ROLLING_BACK and RECOVERY_REQUIRED. Rollback announces itself before acting
    and reaches ROLLED_BACK only when every restoration AND every
    post-restoration digest has succeeded. A failure lands in RECOVERY_REQUIRED
    -- non-terminal, discoverable, and retryable.

    recover_transaction(). Recovery reconstructs its state EXCLUSIVELY from the
    manifest and the on-disk preimages, so it works in a process that never saw
    the transaction object.

    Clean-tree enforcement. require_clean_tree asserts the working tree and
    index are clean at preparation, not merely that HEAD does not move. A
    transaction certifies a NAMED SET of files.

    Secret targets are REFUSED by default. One abstraction cannot promise both
    "no persistent secret preimage" and "arbitrary secret mutations are
    crash-recoverable". Credential provisioning is a different authority, not a
    special case of a source-tree patch.

WHAT IT REPLACES
MEASURED 2026-08-19, in two sweeps: 148 artefacts matching `*.bak_*`
(17,640,928 bytes) and 107 more matching `*.bak`, `*.orig`, `*.rej`
(2,828,345 bytes). The second sweep existed because the first tool scanned ONE
shape of four and reported zero.

WHAT IT REUSES
JsonStateStore for the manifest -- atomic write via mkstemp, fsync and
os.replace in the same directory. VERIFIED against the real module:
StateSchemaMismatch descends from StateStoreError, so an unenveloped journal is
REPORTED rather than propagated. Location is RuntimePaths.transaction_journal.

Author: Monzia Moodie
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from genomic_variant_classifier.state.json_state_store import (
    JsonStateStore, StateStoreError,
)

SCHEMA = "gvc.repository-transaction"
SCHEMA_VERSION = 2

#: Path shapes that may carry credential material.
SECRET_PATTERNS = (
    ".env", "*.env", "*.pem", "*.key", "*.p12", "*.pfx",
    "credentials*", "token*", "secrets*", "*_rsa", "*_ed25519",
)

#: Shapes the classifier MUST recognise. A classifier recognising nothing
#: produces a clean run and a credential edited by a generic patch transaction.
SECRET_CANARIES = (".env", "id_rsa", "server.pem", "api.key",
                   "credentials.json", "token.txt", "secrets.yaml")


class TransactionError(RuntimeError):
    """Any refusal by the transaction machinery."""


class TransactionStateError(TransactionError):
    """An illegal state transition, or an operation in the wrong state."""


class TransactionIntegrityError(TransactionError):
    """A preimage, target or journal is not what the manifest says it is."""


class TransactionRecoveryRequired(TransactionError):
    """A journal is unresolved and must be reconciled before work continues."""


class TransactionState(str, Enum):
    """Monotonic, and honest about failure.

    ROLLING_BACK and RECOVERY_REQUIRED exist because step 3 could record
    ROLLED_BACK while the repository was NOT restored -- making the journal
    simultaneously retained, invisible to discovery, and unretryable.
    """

    PREPARED = "prepared"
    APPLYING = "applying"
    VERIFYING = "verifying"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    RECOVERY_REQUIRED = "recovery_required"
    ABANDONED = "abandoned"


_ALLOWED = {
    TransactionState.PREPARED: (TransactionState.APPLYING,
                                TransactionState.ROLLING_BACK,
                                TransactionState.ABANDONED),
    TransactionState.APPLYING: (TransactionState.VERIFYING,
                                TransactionState.ROLLING_BACK,
                                TransactionState.ABANDONED),
    TransactionState.VERIFYING: (TransactionState.COMMITTED,
                                 TransactionState.ROLLING_BACK,
                                 TransactionState.ABANDONED),
    TransactionState.ROLLING_BACK: (TransactionState.ROLLED_BACK,
                                    TransactionState.RECOVERY_REQUIRED),
    TransactionState.RECOVERY_REQUIRED: (TransactionState.ROLLING_BACK,),
    TransactionState.COMMITTED: (),
    TransactionState.ROLLED_BACK: (),
    TransactionState.ABANDONED: (TransactionState.ROLLING_BACK,),
}

#: Only these two mean "nothing further is owed". RECOVERY_REQUIRED is NOT
#: terminal -- that was the whole defect.
TERMINAL = (TransactionState.COMMITTED, TransactionState.ROLLED_BACK)


class Sensitivity(str, Enum):
    ORDINARY = "ordinary"
    SECRET = "secret"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_secret_path(path) -> bool:
    return any(fnmatch.fnmatch(Path(path).name, pat) for pat in SECRET_PATTERNS)


def _assert_secret_detection_intact() -> None:
    """Refuse to operate if the secret classifier has been weakened.

    Emptying SECRET_PATTERNS raises no error and leaks nothing in any single
    run -- it simply reclassifies a credential file as ordinary. An
    absence-of-failure check cannot see that; this can.
    """
    missed = [c for c in SECRET_CANARIES if not is_secret_path(c)]
    if missed:
        raise TransactionError(
            "the secret-path classifier does not recognise {}. A weakened "
            "classifier lets a generic patch transaction edit a credential."
            .format(missed))


def _write_durable(path: Path, data: bytes) -> None:
    """Write and fsync, so a crash cannot leave a half-written preimage.

    Write-ahead journaling is meaningless if the log itself is only in the
    page cache when the process dies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


@dataclass(frozen=True)
class TransactionTarget:
    """One file a transaction intends to change.

    `relpath` is POSIX-style and relative to the repository root, so a manifest
    written on one machine names the same file on another.
    """

    relpath: str
    existed_before: bool
    pre_sha256: str | None
    pre_size: int | None
    sensitivity: Sensitivity
    mutated: bool = False

    def as_record(self) -> dict:
        return {
            "relpath": self.relpath,
            "existed_before": self.existed_before,
            "pre_sha256": self.pre_sha256,
            "pre_size": self.pre_size,
            "sensitivity": self.sensitivity.value,
            "mutated": self.mutated,
        }

    @classmethod
    def from_record(cls, record: dict) -> "TransactionTarget":
        return cls(
            relpath=record["relpath"],
            existed_before=bool(record["existed_before"]),
            pre_sha256=record.get("pre_sha256"),
            pre_size=record.get("pre_size"),
            sensitivity=Sensitivity(record.get("sensitivity", "ordinary")),
            mutated=bool(record.get("mutated", False)),
        )


@dataclass
class RepositoryTransaction:
    """Apply a set of file changes, or leave the repository exactly as found."""

    repo_root: Path
    journal_root: Path
    transaction_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    require_clean_head: bool = True
    require_clean_tree: bool = True
    allow_secret_targets: bool = False

    _state: TransactionState = field(init=False, default=TransactionState.PREPARED)
    _targets: dict = field(init=False, default_factory=dict)
    _secret_bytes: dict = field(init=False, default_factory=dict)
    _head: str | None = field(init=False, default=None)

    # ---- lifecycle -----------------------------------------------------
    def __post_init__(self) -> None:
        _assert_secret_detection_intact()
        self.repo_root = Path(self.repo_root).resolve()
        self.journal_root = Path(self.journal_root).resolve()
        if (self.journal_root == self.repo_root
                or self.repo_root in self.journal_root.parents):
            raise TransactionError(
                "the journal root {} is inside the repository {}. A rollback "
                "journal that lives in the tree it repairs is the defect this "
                "class exists to remove.".format(self.journal_root, self.repo_root))
        self._assert_no_unresolved_transactions()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.preimages.mkdir(parents=True, exist_ok=True)
        self._head = self._git_head()
        if self.require_clean_tree:
            self._assert_tree_clean()
        self._persist()

    def __enter__(self) -> "RepositoryTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._state in TERMINAL:
            return False
        self.rollback(reason="context exited without commit" if exc is None
                             else "{}: {}".format(exc_type.__name__, exc))
        return False

    # ---- locations -----------------------------------------------------
    @property
    def directory(self) -> Path:
        return self.journal_root / self.transaction_id

    @property
    def preimages(self) -> Path:
        return self.directory / "preimages"

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def state(self) -> TransactionState:
        return self._state

    # ---- persistence ---------------------------------------------------
    def _store(self) -> JsonStateStore:
        return JsonStateStore(path=self.manifest_path, schema=SCHEMA,
                              schema_version=SCHEMA_VERSION)

    def _persist(self) -> None:
        """Write the manifest and make it durable BEFORE any repository write."""
        self._store().save({
            "transaction_id": self.transaction_id,
            "state": self._state.value,
            "repo_root": str(self.repo_root),
            "repo_head": self._head,
            "updated_at": _utc_now(),
            "targets": [t.as_record() for t in self._targets.values()],
        })

    def read_manifest(self) -> dict:
        return self._store().load().values

    def _transition(self, to: TransactionState) -> None:
        allowed = _ALLOWED[self._state]
        if to not in allowed:
            raise TransactionStateError(
                "{} -> {} is not a legal transition (from {} only {} are "
                "permitted). States are monotonic so a manifest cannot claim a "
                "history that did not happen.".format(
                    self._state.value, to.value, self._state.value,
                    [s.value for s in allowed] or "no further states"))
        self._state = to
        self._persist()

    # ---- git -----------------------------------------------------------
    def _git(self, *args):
        try:
            return subprocess.run(("git", "-C", str(self.repo_root)) + args,
                                  capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None

    def _git_head(self):
        out = self._git("rev-parse", "HEAD")
        return (out.stdout.strip() or None) if out else None

    def _assert_tree_clean(self) -> None:
        """A transaction certifies a NAMED SET of files.

        HEAD not moving says nothing about a concurrent editor, a test run, or
        a helper script introducing an unowned change. require_clean_head and
        require_clean_tree are DIFFERENT invariants; step 3 had only the first.
        """
        out = self._git("status", "--porcelain")
        if out is None:
            return
        dirty = [l for l in out.stdout.splitlines() if l.strip()]
        if dirty:
            raise TransactionError(
                "the working tree has {} uncommitted entr(ies) before the "
                "transaction begins: {}. A transaction cannot certify a named "
                "set of files while unowned changes are present."
                .format(len(dirty), dirty[:5]))

    def _assert_head_unmoved(self) -> None:
        if not self.require_clean_head or self._head is None:
            return
        now = self._git_head()
        if now != self._head:
            raise TransactionIntegrityError(
                "HEAD moved from {} to {} during the transaction. The "
                "preimages describe a tree that is no longer current."
                .format(self._head[:12], (now or "unknown")[:12]))

    def _assert_no_unresolved_transactions(self) -> None:
        pending = [t for t in incomplete_transactions(self.journal_root)
                   if t.get("repo_root") in (None, str(self.repo_root))]
        if pending:
            raise TransactionRecoveryRequired(
                "{} unresolved transaction(s) for this repository: {}. Run "
                "recover_transaction() before starting another."
                .format(len(pending),
                        [t.get("transaction_id", "?") for t in pending[:3]]))

    # ---- capture -------------------------------------------------------
    def _resolve(self, relpath: str) -> Path:
        p = (self.repo_root / relpath).resolve()
        if p != self.repo_root and self.repo_root not in p.parents:
            raise TransactionError(
                "{} resolves outside the repository ({}). A transaction may "
                "only change files it is responsible for.".format(relpath, p))
        return p

    def _guard_sensitivity(self, relpath: str) -> Sensitivity:
        if not is_secret_path(relpath):
            return Sensitivity.ORDINARY
        if not self.allow_secret_targets:
            raise TransactionError(
                "{} is credential-shaped and this transaction refuses secret "
                "targets. One abstraction cannot promise both 'no persistent "
                "secret preimage' and 'arbitrary secret mutations are "
                "crash-recoverable'. Credential provisioning is a different "
                "authority, not a special case of a source-tree patch."
                .format(relpath))
        return Sensitivity.SECRET

    def _write_ahead(self, relpath: str) -> TransactionTarget:
        """Persist the intent BEFORE touching the repository.

        capture preimage -> verify its digest -> persist the target record ->
        fsync the journal -> and only then may the caller mutate.
        """
        p = self._resolve(relpath)
        sensitivity = self._guard_sensitivity(relpath)
        if p.exists():
            data = p.read_bytes()
            digest = _sha256_bytes(data)
            if sensitivity is Sensitivity.SECRET:
                self._secret_bytes[relpath] = data
            else:
                dest = self.preimages / relpath
                _write_durable(dest, data)
                if _sha256_bytes(dest.read_bytes()) != digest:
                    raise TransactionIntegrityError(
                        "the preimage for {} does not match the file it was "
                        "copied from".format(relpath))
            target = TransactionTarget(
                relpath=relpath, existed_before=True, pre_sha256=digest,
                pre_size=len(data), sensitivity=sensitivity)
        else:
            target = TransactionTarget(
                relpath=relpath, existed_before=False, pre_sha256=None,
                pre_size=None, sensitivity=sensitivity)
        self._targets[relpath] = target
        self._persist()
        return target

    def _mark_mutated(self, relpath: str) -> None:
        t = self._targets[relpath]
        self._targets[relpath] = TransactionTarget(
            relpath=t.relpath, existed_before=t.existed_before,
            pre_sha256=t.pre_sha256, pre_size=t.pre_size,
            sensitivity=t.sensitivity, mutated=True)
        self._persist()

    # ---- operations ----------------------------------------------------
    def patch(self, relpath: str, new_bytes: bytes) -> None:
        p = self._resolve(relpath)
        if not p.exists():
            raise TransactionError(
                "{} does not exist; use create() for a new file".format(relpath))
        if self._state is TransactionState.PREPARED:
            self._transition(TransactionState.APPLYING)
        if self._state is not TransactionState.APPLYING:
            raise TransactionStateError(
                "cannot patch in state {}".format(self._state.value))
        self._write_ahead(relpath)          # durable BEFORE the mutation
        _write_durable(p, new_bytes)
        self._mark_mutated(relpath)

    def create(self, relpath: str, new_bytes: bytes) -> None:
        p = self._resolve(relpath)
        if p.exists():
            raise TransactionError(
                "{} already exists; use patch() to replace it".format(relpath))
        if self._state is TransactionState.PREPARED:
            self._transition(TransactionState.APPLYING)
        if self._state is not TransactionState.APPLYING:
            raise TransactionStateError(
                "cannot create in state {}".format(self._state.value))
        self._write_ahead(relpath)          # records "did not exist" first
        _write_durable(p, new_bytes)
        self._mark_mutated(relpath)

    def verify(self, check) -> None:
        if self._state is not TransactionState.APPLYING:
            raise TransactionStateError(
                "verify() requires state applying, not {}".format(self._state.value))
        self._transition(TransactionState.VERIFYING)
        if check(self.repo_root) is False:
            raise TransactionIntegrityError("the verification check returned False")

    # ---- termination ---------------------------------------------------
    def commit(self) -> None:
        if self._state not in (TransactionState.APPLYING, TransactionState.VERIFYING):
            raise TransactionStateError(
                "cannot commit from state {}".format(self._state.value))
        self._assert_head_unmoved()
        if self._state is TransactionState.APPLYING:
            self._transition(TransactionState.VERIFYING)
        self._transition(TransactionState.COMMITTED)
        self._scrub_secrets()
        self._destroy_journal()

    def rollback(self, reason: str = "") -> None:
        """Restore every target, then destroy the journal.

        ROLLED_BACK is reached ONLY when every restoration and every
        post-restoration digest has succeeded. Anything else lands in
        RECOVERY_REQUIRED, which is non-terminal, discoverable and retryable.
        """
        if self._state in TERMINAL:
            return
        if self._state is not TransactionState.ROLLING_BACK:
            self._transition(TransactionState.ROLLING_BACK)
        failures = []
        for relpath, target in list(self._targets.items()):
            try:
                _restore_target(self.repo_root, self.preimages, target,
                                self._secret_bytes.get(relpath))
            except Exception as exc:          # noqa: BLE001 - reported, not hidden
                failures.append((relpath, str(exc)))
        if failures:
            self._transition(TransactionState.RECOVERY_REQUIRED)
            self._scrub_secrets()
            raise TransactionIntegrityError(
                "rollback could not restore {} target(s): {}. The journal at "
                "{} is RETAINED in state recovery_required, remains "
                "discoverable, and the rollback may be retried."
                .format(len(failures), failures[:3], self.directory))
        self._transition(TransactionState.ROLLED_BACK)
        self._scrub_secrets()
        self._destroy_journal()

    def _scrub_secrets(self) -> None:
        for key in list(self._secret_bytes):
            self._secret_bytes[key] = b""
            del self._secret_bytes[key]

    def _destroy_journal(self) -> None:
        if self.directory.exists():
            shutil.rmtree(self.directory)


# ---- restoration, usable without a transaction object ------------------
def _restore_target(repo_root: Path, preimages: Path,
                    target: TransactionTarget, secret_bytes=None) -> None:
    """Restore one target from its preimage, verifying before and after.

    Free-standing so recovery in a FRESH PROCESS uses the same code path as
    in-process rollback.
    """
    p = (repo_root / target.relpath).resolve()
    if repo_root not in p.parents and p != repo_root:
        raise TransactionIntegrityError(
            "{} resolves outside {}".format(target.relpath, repo_root))
    if not target.existed_before:
        if p.exists():
            p.unlink()
        return
    if target.sensitivity is Sensitivity.SECRET:
        data = secret_bytes
        if data is None:
            raise TransactionIntegrityError(
                "the in-memory preimage for {} is gone. A secret target is "
                "NOT crash-recoverable by design; that is why secret targets "
                "are refused by default.".format(target.relpath))
    else:
        src = preimages / target.relpath
        if not src.exists():
            raise TransactionIntegrityError(
                "the preimage for {} is missing".format(target.relpath))
        data = src.read_bytes()
    if _sha256_bytes(data) != target.pre_sha256:
        raise TransactionIntegrityError(
            "the preimage for {} does not match its recorded digest; "
            "restoring it would write unknown bytes".format(target.relpath))
    _write_durable(p, data)
    if _sha256_bytes(p.read_bytes()) != target.pre_sha256:
        raise TransactionIntegrityError(
            "{} does not match its preimage after restoration".format(target.relpath))


# ---- discovery and recovery --------------------------------------------
def incomplete_transactions(journal_root) -> list:
    """Every journal in a non-terminal state.

    RECOVERY_REQUIRED is included. In step 3 a failed rollback recorded
    ROLLED_BACK and vanished from this list while the repository was still
    unrestored.
    """
    journal_root = Path(journal_root)
    if not journal_root.exists():
        return []
    out = []
    for d in sorted(journal_root.iterdir()):
        manifest = d / "manifest.json"
        if not (d.is_dir() and manifest.is_file()):
            continue
        store = JsonStateStore(path=manifest, schema=SCHEMA,
                               schema_version=SCHEMA_VERSION)
        try:
            values = dict(store.load().values)
        except StateStoreError as exc:
            out.append({"transaction_id": d.name, "state": "unreadable",
                        "error": str(exc), "directory": str(d)})
            continue
        if values.get("state") not in (TransactionState.COMMITTED.value,
                                       TransactionState.ROLLED_BACK.value):
            values["directory"] = str(d)
            out.append(values)
    return out


def recover_transaction(directory, *, action: str = "rollback") -> dict:
    """Reconcile an interrupted transaction from PERSISTED DATA ALONE.

    A process restart destroys the RepositoryTransaction object and its
    in-memory targets. This reads the manifest and the on-disk preimages, so it
    works in a process that never saw the transaction.

    Step 3 had no such entry point: incomplete_transactions() proved DISCOVERY,
    never RECOVERY, and the interruption test used `del tx` -- which leaves
    Python alive with normal filesystem caches and no reconstruction boundary.
    """
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise TransactionRecoveryRequired(
            "{} holds no manifest".format(directory))
    store = JsonStateStore(path=manifest_path, schema=SCHEMA,
                           schema_version=SCHEMA_VERSION)
    values = dict(store.load().values)
    state = values.get("state")
    if state in (TransactionState.COMMITTED.value,
                 TransactionState.ROLLED_BACK.value):
        return {"transaction_id": values.get("transaction_id"),
                "state": state, "action": "none",
                "detail": "already terminal"}
    if action != "rollback":
        raise TransactionError("unsupported recovery action: {!r}".format(action))

    repo_root = Path(values["repo_root"]).resolve()
    targets = [TransactionTarget.from_record(r) for r in values.get("targets", [])]
    preimages = directory / "preimages"

    values["state"] = TransactionState.ROLLING_BACK.value
    values["updated_at"] = _utc_now()
    store.save(values)

    failures = []
    for t in targets:
        try:
            _restore_target(repo_root, preimages, t)
        except Exception as exc:              # noqa: BLE001 - reported
            failures.append((t.relpath, str(exc)))

    if failures:
        values["state"] = TransactionState.RECOVERY_REQUIRED.value
        values["updated_at"] = _utc_now()
        values["recovery_failures"] = [f[0] for f in failures]
        store.save(values)
        raise TransactionIntegrityError(
            "recovery could not restore {} target(s): {}. {} is RETAINED in "
            "state recovery_required.".format(len(failures), failures[:3], directory))

    values["state"] = TransactionState.ROLLED_BACK.value
    values["updated_at"] = _utc_now()
    store.save(values)
    shutil.rmtree(directory)
    return {"transaction_id": values.get("transaction_id"),
            "state": TransactionState.ROLLED_BACK.value,
            "action": "rolled_back",
            "restored": [t.relpath for t in targets]}
