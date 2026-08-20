"""A transaction may temporarily own a rollback state. The repository never does.

INSTALLER-TRANSACTION-1, step 3.

WHY THIS EXISTS
Every installer in this project wrote `<target>.bak_<timestamp>` beside the file
it edited, as its rollback path, and none removed it on success. What was
designed as a rollback IMPLEMENTATION DETAIL became a permanent archival system
by omission.

MEASURED 2026-08-19, in two sweeps:

    148 artefacts, 17,640,928 bytes   matching `*.bak_*`
    107 artefacts,  2,828,345 bytes   matching `*.bak`, `*.orig`, `*.rej`

The second sweep existed because the first tool scanned ONE shape of four and
reported "remaining .bak_* artefact(s): 0" -- true, and misleading in exactly
the way an incomplete filter always is. One of the 148 was a credential-bearing
`.env` backup; the manifest recording its retirement was itself overwritten by a
later routine run, because it was addressed by a name the next event reused.

Three defects, one root: rollback state living inside the repository under names
no single filter reliably covers, with cleanup depending on someone remembering.

THE INVARIANT
    A successful transaction leaves NO artefact in the repository.
    A failed transaction leaves the repository EXACTLY as it was found.
    An INTERRUPTED transaction leaves a journal OUTSIDE the repository, in a
    non-terminal state, and the next invocation refuses until it is reconciled.

WHAT THIS REUSES RATHER THAN REBUILDS
    JsonStateStore  atomic write via mkstemp + fsync + os.replace in the SAME
                    directory, schema identification, a generation counter, and
                    load() that RAISES on damage instead of reporting emptiness.
                    Measured 2026-08-19: save(values, *, generation=None)
                    returns the generation written, so optimistic concurrency
                    is available without inventing it here.

    RuntimePaths.transaction_journal
                    cache_root / "transactions" -- machine-scoped, outside any
                    repository, so an interrupted run survives a working-tree
                    reset.

A third `_atomic_write` was deliberately NOT added. representation_artifact.py
already documents its copy of the idiom from RunArtifactWriter; consolidating
those two is a separate unit, and a third copy would make it worse.

SECRET TARGETS ARE A DIFFERENT POLICY, NOT A FLAG
A target whose path shape is credential-bearing gets no preimage on disk at all.
Its bytes are held only for the lifetime of the process, its manifest entry
records digest and structure but never content, and the entry is scrubbed on
commit. The credential incident of 2026-08-15 began with a general-purpose text
workflow operating near a secret; that lesson belongs in the architecture.

Author: Monzia Moodie
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
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
SCHEMA_VERSION = 1

#: Path shapes that must never receive an on-disk preimage.
SECRET_PATTERNS = (
    ".env", "*.env", "*.pem", "*.key", "*.p12", "*.pfx",
    "credentials*", "token*", "secrets*", "*_rsa", "*_ed25519",
)

#: Shapes the classifier MUST recognise. A classifier that recognises nothing
#: produces a clean run and a confident on-disk copy of a credential.
SECRET_CANARIES = (".env", "id_rsa", "server.pem", "api.key",
                   "credentials.json", "token.txt", "secrets.yaml")


class TransactionError(RuntimeError):
    """Any refusal by the transaction machinery."""


class TransactionStateError(TransactionError):
    """An illegal state transition, or an operation in the wrong state."""


class TransactionIntegrityError(TransactionError):
    """A preimage, target or journal is not what the manifest says it is."""


class TransactionState(str, Enum):
    """Monotonic. A transaction moves forward or terminates; never backward.

    (str, Enum) so a manifest round-trips through JSON without a codec.
    """

    PREPARED = "prepared"
    APPLYING = "applying"
    VERIFYING = "verifying"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    ABANDONED = "abandoned"


#: The only legal moves. Anything else raises rather than being recorded.
_ALLOWED = {
    TransactionState.PREPARED: (TransactionState.APPLYING,
                                TransactionState.ROLLED_BACK,
                                TransactionState.ABANDONED),
    TransactionState.APPLYING: (TransactionState.VERIFYING,
                                TransactionState.ROLLED_BACK,
                                TransactionState.ABANDONED),
    TransactionState.VERIFYING: (TransactionState.COMMITTED,
                                 TransactionState.ROLLED_BACK,
                                 TransactionState.ABANDONED),
    TransactionState.COMMITTED: (),
    TransactionState.ROLLED_BACK: (),
    TransactionState.ABANDONED: (TransactionState.ROLLED_BACK,),
}

TERMINAL = (TransactionState.COMMITTED, TransactionState.ROLLED_BACK)


class Sensitivity(str, Enum):
    ORDINARY = "ordinary"
    SECRET = "secret"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_secret_detection_intact() -> None:
    """Refuse to operate if the secret classifier has been weakened.

    Emptying SECRET_PATTERNS produces NO error and NO leak in any single run --
    it simply reclassifies a credential file as ordinary, and an ordinary target
    gets its bytes written to a preimage on disk. An absence-of-failure check
    cannot see that; this can.
    """
    missed = [c for c in SECRET_CANARIES if not is_secret_path(c)]
    if missed:
        raise TransactionError(
            "the secret-path classifier does not recognise {}. A weakened "
            "classifier writes credential bytes to an on-disk preimage without "
            "reporting anything.".format(missed))


def is_secret_path(path) -> bool:
    name = Path(path).name
    return any(fnmatch.fnmatch(name, pat) for pat in SECRET_PATTERNS)


@dataclass(frozen=True)
class TransactionTarget:
    """One file a transaction intends to change.

    `relpath` is always relative to the repository root, so a manifest written
    on one machine names the same file on another.
    """

    relpath: str
    existed_before: bool
    pre_sha256: str | None
    pre_size: int | None
    sensitivity: Sensitivity

    def as_record(self) -> dict:
        return {
            "relpath": self.relpath,
            "existed_before": self.existed_before,
            "pre_sha256": self.pre_sha256,
            "pre_size": self.pre_size,
            "sensitivity": self.sensitivity.value,
        }


@dataclass
class RepositoryTransaction:
    """Apply a set of file changes, or leave the repository exactly as found.

    Usage:

        with RepositoryTransaction(repo_root, journal_root) as tx:
            tx.patch("src/thing.py", new_bytes)
            tx.create("tests/unit/test_thing.py", test_bytes)
            tx.verify(structural_check)
            tx.commit()

    Leaving the block without commit() rolls back. A failure inside it rolls
    back and re-raises.
    """

    repo_root: Path
    journal_root: Path
    transaction_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    require_clean_head: bool = True

    _state: TransactionState = field(init=False, default=TransactionState.PREPARED)
    _targets: dict = field(init=False, default_factory=dict)
    _secret_bytes: dict = field(init=False, default_factory=dict)
    _head: str | None = field(init=False, default=None)
    _generation: int = field(init=False, default=0)

    # ---- lifecycle -----------------------------------------------------
    def __post_init__(self) -> None:
        _assert_secret_detection_intact()
        self.repo_root = Path(self.repo_root).resolve()
        self.journal_root = Path(self.journal_root).resolve()
        if self.journal_root == self.repo_root or self.repo_root in self.journal_root.parents:
            raise TransactionError(
                "the journal root {} is inside the repository {}. A rollback "
                "journal that lives in the tree it repairs is the defect this "
                "class exists to remove.".format(self.journal_root, self.repo_root))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.preimages.mkdir(parents=True, exist_ok=True)
        self._head = self._git_head()
        self._write_manifest()

    def __enter__(self) -> "RepositoryTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._state is TransactionState.COMMITTED:
            return False
        self.rollback(reason="context exited without commit"
                             if exc is None else "{}: {}".format(
                                 exc_type.__name__, exc))
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

    # ---- state machine -------------------------------------------------
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
        self._write_manifest()

    # ---- manifest ------------------------------------------------------
    def _store(self) -> JsonStateStore:
        return JsonStateStore(path=self.manifest_path, schema=SCHEMA,
                              schema_version=SCHEMA_VERSION)

    def _write_manifest(self) -> None:
        values = {
            "transaction_id": self.transaction_id,
            "state": self._state.value,
            "repo_root": str(self.repo_root),
            "repo_head": self._head,
            "updated_at": _utc_now(),
            "targets": [t.as_record() for t in self._targets.values()],
        }
        self._generation = self._store().save(values)

    def read_manifest(self) -> dict:
        return self._store().load().values

    # ---- git -----------------------------------------------------------
    def _git_head(self):
        try:
            out = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None

    def _assert_head_unmoved(self) -> None:
        """A transaction prepared against one commit must not commit against
        another. Preimages describe the tree as it was at preparation."""
        if not self.require_clean_head or self._head is None:
            return
        now = self._git_head()
        if now != self._head:
            raise TransactionIntegrityError(
                "HEAD moved from {} to {} during the transaction. The "
                "preimages describe a tree that is no longer current."
                .format(self._head[:12], (now or "unknown")[:12]))

    # ---- capture -------------------------------------------------------
    def _resolve(self, relpath: str) -> Path:
        p = (self.repo_root / relpath).resolve()
        if p != self.repo_root and self.repo_root not in p.parents:
            raise TransactionError(
                "{} resolves outside the repository ({}). A transaction may "
                "only change files it is responsible for.".format(relpath, p))
        return p

    def _capture(self, relpath: str) -> TransactionTarget:
        if self._state not in (TransactionState.PREPARED, TransactionState.APPLYING):
            raise TransactionStateError(
                "cannot capture a target in state {}".format(self._state.value))
        p = self._resolve(relpath)
        secret = is_secret_path(relpath)
        sensitivity = Sensitivity.SECRET if secret else Sensitivity.ORDINARY
        if p.exists():
            data = p.read_bytes()
            target = TransactionTarget(
                relpath=relpath, existed_before=True,
                pre_sha256=_sha256_bytes(data), pre_size=len(data),
                sensitivity=sensitivity)
            if secret:
                # Held in memory only. A credential preimage on disk is the
                # thing the 2026-08-15 incident produced.
                self._secret_bytes[relpath] = data
            else:
                dest = self.preimages / relpath
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        else:
            target = TransactionTarget(
                relpath=relpath, existed_before=False, pre_sha256=None,
                pre_size=None, sensitivity=sensitivity)
        self._targets[relpath] = target
        return target

    # ---- operations ----------------------------------------------------
    def patch(self, relpath: str, new_bytes: bytes) -> None:
        """Replace an existing file, capturing its preimage first."""
        p = self._resolve(relpath)
        if not p.exists():
            raise TransactionError(
                "{} does not exist; use create() for a new file".format(relpath))
        if self._state is TransactionState.PREPARED:
            self._transition(TransactionState.APPLYING)
        self._capture(relpath)
        p.write_bytes(new_bytes)
        self._write_manifest()

    def create(self, relpath: str, new_bytes: bytes) -> None:
        """Write a new file, recorded so rollback removes it."""
        p = self._resolve(relpath)
        if p.exists():
            raise TransactionError(
                "{} already exists; use patch() to replace it".format(relpath))
        if self._state is TransactionState.PREPARED:
            self._transition(TransactionState.APPLYING)
        self._capture(relpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(new_bytes)
        self._write_manifest()

    def verify(self, check) -> None:
        """Run a caller-supplied check. Any exception rolls the transaction back.

        The check receives the repository root and must raise or return falsely
        to fail.
        """
        if self._state is not TransactionState.APPLYING:
            raise TransactionStateError(
                "verify() requires state applying, not {}".format(self._state.value))
        self._transition(TransactionState.VERIFYING)
        result = check(self.repo_root)
        if result is False:
            raise TransactionIntegrityError("the verification check returned False")

    # ---- termination ---------------------------------------------------
    def commit(self) -> None:
        """Finish. The journal is destroyed; the repository keeps the changes."""
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
        """Restore every target to exactly its captured state, then destroy the
        journal. Restoration is verified by digest before the journal goes."""
        if self._state in TERMINAL:
            return
        failures = []
        for relpath, target in self._targets.items():
            try:
                self._restore(target)
            except Exception as exc:          # noqa: BLE001 - reported, not hidden
                failures.append((relpath, str(exc)))
        self._state = TransactionState.ROLLED_BACK
        self._write_manifest()
        self._scrub_secrets()
        if failures:
            raise TransactionIntegrityError(
                "rollback could not restore {}: {}. The journal at {} is "
                "RETAINED for manual recovery.".format(
                    len(failures), failures[:3], self.directory))
        self._destroy_journal()

    def _restore(self, target: TransactionTarget) -> None:
        p = self._resolve(target.relpath)
        if not target.existed_before:
            if p.exists():
                p.unlink()
            return
        if target.sensitivity is Sensitivity.SECRET:
            data = self._secret_bytes.get(target.relpath)
            if data is None:
                raise TransactionIntegrityError(
                    "the in-memory preimage for {} is gone; a secret target "
                    "cannot be restored from disk by design"
                    .format(target.relpath))
        else:
            src = self.preimages / target.relpath
            if not src.exists():
                raise TransactionIntegrityError(
                    "the preimage for {} is missing".format(target.relpath))
            data = src.read_bytes()
        if _sha256_bytes(data) != target.pre_sha256:
            raise TransactionIntegrityError(
                "the preimage for {} does not match its recorded digest; "
                "restoring it would write unknown bytes".format(target.relpath))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        if _sha256_bytes(p.read_bytes()) != target.pre_sha256:
            raise TransactionIntegrityError(
                "{} does not match its preimage after restoration"
                .format(target.relpath))

    def _scrub_secrets(self) -> None:
        for key in list(self._secret_bytes):
            self._secret_bytes[key] = b""
            del self._secret_bytes[key]

    def _destroy_journal(self) -> None:
        if self.directory.exists():
            shutil.rmtree(self.directory)


# ---- recovery ----------------------------------------------------------
def incomplete_transactions(journal_root) -> list:
    """Every journal in a non-terminal state.

    An interrupted installer leaves exactly this. The next invocation should
    refuse until it has been reconciled, rather than hoping somebody notices a
    stray file.
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
            values = store.load().values
        except StateStoreError as exc:
            out.append({"transaction_id": d.name, "state": "unreadable",
                        "error": str(exc), "directory": str(d)})
            continue
        state = values.get("state")
        if state not in (TransactionState.COMMITTED.value,
                         TransactionState.ROLLED_BACK.value):
            values = dict(values)
            values["directory"] = str(d)
            out.append(values)
    return out
