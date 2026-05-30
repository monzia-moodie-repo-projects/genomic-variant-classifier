"""
ledger.py -- ScienceClaw artifact ledger + deterministic policy gate
====================================================================
Layered on the existing SharedState JSON substrate (artifact_ledger key,
backfilled by SharedState._migrate). Provides:

  ScienceClawLedger   Append-only, hash-chained record of agent-produced
                      artifacts. Each row stores the previous row's hash, so
                      any after-the-fact mutation breaks the chain and is
                      detected by verify_chain().

  compute_sha256      Caller-side helper. Hashes a file on disk. This is the
                      ONLY place that touches the filesystem -- the gate never
                      does, which is what makes the gate deterministic.

  evaluate            The pure policy gate. Given (ledger_entries, message,
                      computed_sha) it returns a Verdict(allow, reasons) with
                      no I/O and no clock reads, so identical inputs always
                      yield an identical Verdict. Enforces BOTH dimensions:
                        INTEGRITY     -- if the message payload references an
                                         artifact (artifact_id + artifact_sha256
                                         both present), that artifact must be in
                                         the ledger and the recorded hash must
                                         equal computed_sha.
                        AUTHORIZATION -- if the message requires approval, it
                                         must have approved is True.
                      When no artifact is referenced, the integrity dimension
                      is a strict no-op (authorization alone decides), so the
                      existing message-bus behaviour is unchanged.

All hashing of ledger rows uses hashlib.sha256 over a canonical JSON encoding
(sorted keys, no whitespace) of the hash-relevant fields, so the row hash is
stable and independent of dict ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_LEDGER_KEY = "artifact_ledger"
_CHUNK = 1 << 20  # 1 MiB streaming read


class LedgerError(RuntimeError):
    """Raised on append-only / hash-chain violations."""


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """Result of a gate evaluation. Immutable so it is safe to compare/reuse."""

    allow: bool
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Caller-side hashing (the only filesystem touch)
# ---------------------------------------------------------------------------


def compute_sha256(path: str) -> str:
    """Return the hex SHA-256 of the file at path. Raises if it cannot be read."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Canonical row hashing
# ---------------------------------------------------------------------------


def _row_hash(index: int, artifact_id: str, producer: str, sha256: str,
              uri: str, created_at: str, parent_ids: list[str],
              prev_hash: str | None) -> str:
    """Deterministic hash of the hash-relevant fields of one ledger row."""
    canonical = json.dumps(
        {
            "index": index,
            "artifact_id": artifact_id,
            "producer": producer,
            "sha256": sha256,
            "uri": uri,
            "created_at": created_at,
            "parent_ids": list(parent_ids or []),
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ScienceClawLedger
# ---------------------------------------------------------------------------


class ScienceClawLedger:
    """Append-only, hash-chained artifact ledger over SharedState."""

    def __init__(self, shared_state) -> None:
        self._state = shared_state

    # -- read -----------------------------------------------------------

    def entries(self) -> list[dict]:
        """Return a copy of all ledger rows in append order."""
        state = self._state.load()
        return list(state.get(_LEDGER_KEY, []))

    def lookup(self, artifact_id: str) -> dict | None:
        """Return the most recent row for artifact_id, or None if absent."""
        match = None
        for row in self.entries():
            if row.get("artifact_id") == artifact_id:
                match = row
        return match

    # -- write (append-only) -------------------------------------------

    def record(self, artifact_id: str, producer: str, sha256: str, uri: str,
               parent_ids: list[str] | None = None,
               created_at: str | None = None) -> dict:
        """Append a new artifact row. Raises LedgerError on duplicate id."""
        state = self._state.load()
        rows = state.setdefault(_LEDGER_KEY, [])

        for row in rows:
            if row.get("artifact_id") == artifact_id:
                raise LedgerError(
                    f"artifact_id '{artifact_id}' already in ledger "
                    f"(append-only; ids are unique)"
                )

        index = len(rows)
        prev_hash = rows[-1]["row_hash"] if rows else None
        ts = created_at or datetime.now(timezone.utc).isoformat()
        parents = list(parent_ids or [])

        rh = _row_hash(index, artifact_id, producer, sha256, uri, ts, parents, prev_hash)
        entry = {
            "index": index,
            "artifact_id": artifact_id,
            "producer": producer,
            "sha256": sha256,
            "uri": uri,
            "created_at": ts,
            "parent_ids": parents,
            "prev_hash": prev_hash,
            "row_hash": rh,
        }
        rows.append(entry)
        self._state.save(state)
        return dict(entry)

    # -- integrity ------------------------------------------------------

    def verify_chain(self) -> bool:
        """Recompute the whole chain; raise LedgerError on any mismatch."""
        prev = None
        for i, row in enumerate(self.entries()):
            if row.get("index") != i:
                raise LedgerError(f"row {i}: index field {row.get('index')} != {i}")
            if row.get("prev_hash") != prev:
                raise LedgerError(f"row {i}: prev_hash chain broken")
            expect = _row_hash(
                i,
                row.get("artifact_id"),
                row.get("producer"),
                row.get("sha256"),
                row.get("uri"),
                row.get("created_at"),
                row.get("parent_ids", []),
                row.get("prev_hash"),
            )
            if expect != row.get("row_hash"):
                raise LedgerError(f"row {i}: row_hash mismatch (tampered)")
            prev = row.get("row_hash")
        return True


# ---------------------------------------------------------------------------
# The pure policy gate
# ---------------------------------------------------------------------------


def _artifact_ref(payload: dict) -> tuple[str, str] | None:
    """Return (artifact_id, artifact_sha256) iff BOTH keys are present, else None."""
    if not isinstance(payload, dict):
        return None
    aid = payload.get("artifact_id")
    ash = payload.get("artifact_sha256")
    if aid is not None and ash is not None:
        return str(aid), str(ash)
    return None


def evaluate(ledger_entries: list[dict], message: dict,
             computed_sha: str | None) -> Verdict:
    """
    Pure allow/deny decision. No I/O, no clock. Same inputs -> same Verdict.

    Enforces BOTH integrity (when the message references an artifact) AND
    authorization (when the message requires approval). Both must pass.
    """
    reasons: list[str] = []
    payload = message.get("payload", {}) if isinstance(message, dict) else {}

    # --- INTEGRITY (only when a full artifact reference is present) ---
    ref = _artifact_ref(payload)
    if ref is not None:
        artifact_id, declared_sha = ref
        row = None
        for r in ledger_entries:
            if r.get("artifact_id") == artifact_id:
                row = r  # last match wins (latest record)
        if row is None:
            reasons.append(
                f"integrity: artifact '{artifact_id}' missing from ledger"
            )
        else:
            recorded = row.get("sha256")
            if computed_sha is None:
                reasons.append(
                    f"integrity: no computed hash supplied for '{artifact_id}'"
                )
            elif recorded != computed_sha:
                reasons.append(
                    f"integrity: hash mismatch for '{artifact_id}' "
                    f"(ledger != on-disk)"
                )
            elif declared_sha != recorded:
                reasons.append(
                    f"integrity: message-declared hash != ledger hash "
                    f"for '{artifact_id}'"
                )

    # --- AUTHORIZATION ---
    if message.get("requires_approval"):
        if message.get("approved") is not True:
            reasons.append("authorization: message requires approval and is not approved")

    return Verdict(allow=(len(reasons) == 0), reasons=reasons)