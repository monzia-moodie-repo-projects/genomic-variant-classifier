"""Commit the finding, then deliver it. Never the reverse.

Author: Monzia Moodie

WHY THIS EXISTS
===============
The ruling's durability table requires two behaviours this project's
monitoring layer does not have:

    lose persistence before acknowledgement -> NO successful completion
                                               acknowledgement
    commit the event, then lose delivery    -> the pending delivery SURVIVES
                                               and can be retried

MEASURED 2026-09-14, the current behaviour: alerts are appended to a list,
logged at INFO, and written into a state file by the same call that reports
success. Nothing distinguishes "the finding was recorded" from "someone was
told", so a delivery that never happened leaves no trace that it is owed.

THE ORDER IS THE WHOLE DESIGN
=============================
    1. commit the finding durably, with its own identity
    2. only then attempt delivery
    3. record the delivery ACCEPTANCE separately, keyed to that identity

Committing after delivering would lose the finding when the process dies
between them. Committing and delivering in one step cannot represent
"recorded but not yet delivered", which is exactly the state a retry needs.

WHAT "DURABLE" MEANS HERE, AND WHAT IT DOES NOT
===============================================
A write, flush, fsync, close and reopen in a FRESH process establishes
process recovery. It does NOT establish behaviour under power loss, a
filesystem that lies about fsync, or every crash point. This module claims
process recovery and says so.

ATTEMPTS AND SUCCESSES ARE DIFFERENT FACTS
==========================================
A failed attempt must never refresh the timestamp of the previous successful
observation. They are stored under separate keys, and `last_success` is
written by exactly one method.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class StoreError(RuntimeError):
    """The store could not do what was asked. Never silently swallowed."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id      TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    outcome         TEXT
);
-- attempts is declared FIRST: a foreign key cannot reference a table that
-- does not yet exist.
CREATE TABLE IF NOT EXISTS findings (
    event_id        TEXT PRIMARY KEY,
    committed_at    TEXT NOT NULL,
    subject         TEXT NOT NULL,
    attempt_id      TEXT NOT NULL,
    record_json     TEXT NOT NULL,
    delivered_at    TEXT,
    delivery_ref    TEXT,
    -- MEASURED 2026-09-15: attempt_id was a plain TEXT column with no
    -- constraint, so a finding could name an attempt that NEVER BEGAN and
    -- pending_deliveries returned it as though it were provenanced. The
    -- module docstring claims attempts and successes are separate facts;
    -- nothing bound a finding to a real one.
    FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)
);
"""


@dataclass(frozen=True)
class CommittedFinding:
    event_id: str
    attempt_id: str
    subject: str
    committed_at: str


class FindingStore:
    """A durable finding store with commit and delivery as separate facts."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(str(self.path), isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        # SQLite disables foreign keys BY DEFAULT and the setting is
        # PER-CONNECTION, so declaring the constraint without this line
        # enforces nothing. MEASURED 2026-09-15: `PRAGMA foreign_keys`
        # reported 0, and a finding referencing a non-existent attempt was
        # stored and returned by pending_deliveries.
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- attempts ----------------------------------------------------------

    def begin_attempt(self, subject: str) -> str:
        """Record that an attempt STARTED. This is not a success."""
        attempt_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO attempts(attempt_id, subject, started_at) "
                "VALUES (?,?,?)",
                (attempt_id, subject, _now()))
        return attempt_id

    def finish_attempt(self, attempt_id: str, outcome: str) -> None:
        """Record how an attempt ENDED. Does not touch any finding."""
        if outcome not in ("complete", "incomplete", "failed"):
            raise StoreError("unknown attempt outcome: {!r}".format(outcome))
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE attempts SET finished_at=?, outcome=? WHERE attempt_id=?",
                (_now(), outcome, attempt_id))
            if cur.rowcount != 1:
                raise StoreError("no such attempt: {}".format(attempt_id))

    # -- findings ----------------------------------------------------------

    def commit_finding(self, *, attempt_id: str, subject: str,
                       record: dict) -> CommittedFinding:
        """Durably record a finding BEFORE any delivery is attempted.

        Returns its identity. Delivery is a separate call keyed to that
        identity, so a process that dies before delivering leaves a finding
        whose `delivered_at` is NULL -- which is precisely a retry queue.
        """
        if type(record) is not dict:
            raise StoreError("a finding record must be a dict")
        # MEASURED 2026-09-15: {1: "x"} was ACCEPTED, and json.dumps coerced
        # the integer key to "1". The record RECOVERED was not the record
        # COMMITTED, so the round trip was lossy and nothing said so.
        bad = sorted(repr(k) for k in record if type(k) is not str)
        if bad:
            raise StoreError(
                "a finding record must have string keys; got {}".format(bad))
        try:
            blob = json.dumps(record, sort_keys=True, ensure_ascii=True,
                              allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise StoreError("finding is not serialisable: {}".format(exc)) from exc
        event_id = uuid.uuid4().hex
        committed_at = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO findings(event_id, committed_at, subject, "
                    "attempt_id, record_json) VALUES (?,?,?,?,?)",
                    (event_id, committed_at, subject, attempt_id, blob))
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                raise StoreError(
                    "no such attempt {!r}: a finding must name an attempt "
                    "that began".format(attempt_id)) from exc
            # fsync the directory entry too: a committed row whose containing
            # directory was never synced can vanish on some filesystems.
            _fsync_dir(self.path.parent)
        return CommittedFinding(event_id, attempt_id, subject, committed_at)

    def record_delivery(self, event_id: str, delivery_ref: str) -> None:
        """Record that delivery was ACCEPTED by the channel.

        Called only after the channel confirms. A finding whose delivery was
        attempted and refused keeps delivered_at NULL and remains pending.
        """
        if type(delivery_ref) is not str or not delivery_ref:
            raise StoreError("a delivery reference is required")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE findings SET delivered_at=?, delivery_ref=? "
                "WHERE event_id=? AND delivered_at IS NULL",
                (_now(), delivery_ref, event_id))
            if cur.rowcount != 1:
                raise StoreError(
                    "no undelivered finding with event_id {}".format(event_id))

    def pending_deliveries(self) -> list:
        """Findings committed but not yet accepted by a delivery channel."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, subject, attempt_id, committed_at, record_json "
                "FROM findings WHERE delivered_at IS NULL "
                "ORDER BY committed_at, event_id").fetchall()
        return [{"event_id": r[0], "subject": r[1], "attempt_id": r[2],
                 "committed_at": r[3], "record": json.loads(r[4])} for r in rows]

    def get_finding(self, event_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT event_id, subject, attempt_id, committed_at, "
                "record_json, delivered_at, delivery_ref FROM findings "
                "WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            raise StoreError("no such finding: {}".format(event_id))
        return {"event_id": row[0], "subject": row[1], "attempt_id": row[2],
                "committed_at": row[3], "record": json.loads(row[4]),
                "delivered_at": row[5], "delivery_ref": row[6]}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fsync_dir(path: Path) -> None:
    """Sync a directory entry. Best effort: not every platform supports it."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
