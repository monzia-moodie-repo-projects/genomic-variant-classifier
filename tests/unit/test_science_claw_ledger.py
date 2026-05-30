#!/usr/bin/env python3
"""
test_science_claw_ledger.py -- ScienceClaw artifact ledger + policy gate
========================================================================
Failing-first test suite for Task 3.

Covers, in one place, the two dimensions the gate enforces TOGETHER:

  INTEGRITY    -- an artifact a message references must be present in the
                  append-only ledger AND its caller-computed SHA-256 must
                  match the hash recorded in the ledger.
  AUTHORIZATION -- a message that requires approval must have been approved
                  (approved is True) before it is actionable.

Design contracts asserted here (locked with the user):
  * The gate `evaluate(...)` is PURE: the on-disk SHA-256 is computed by the
    CALLER and passed in as data. The gate performs no file I/O and reads no
    clock, so identical inputs always yield an identical Verdict.
  * Artifacts are referenced via payload keys `artifact_id` + `artifact_sha256`.
    When those keys are ABSENT, the integrity dimension is a strict no-op, so
    the 34 existing message-bus tests keep passing by construction.
  * A new canonical subject `ARTIFACT_PUBLISHED` exists in ALL_SUBJECTS and is a
    member of APPROVAL_REQUIRED_SUBJECTS (requires approval by default).
  * The ledger is APPEND-ONLY and hash-chained: each entry stores the hash of
    the previous entry; mutating or duplicating a prior entry breaks the chain
    and is detected (raises LedgerError).

These tests import the not-yet-written module:
    genomic_variant_classifier.agent_layer.science_claw
so on first run they MUST fail at import (failing-first). pytest.importorskip
is intentionally NOT used: a missing module is a real failure at this stage.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# --- Imports under test (module does not exist yet -> failing-first) --------
from genomic_variant_classifier.agent_layer.science_claw import (
    ScienceClawLedger,
    evaluate,
    Verdict,
    compute_sha256,
    LedgerError,
)
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer import message_bus as mb


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_state():
    """A SharedState backed by a temp file that does not yet exist on disk."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)  # SharedState creates it on first save
    try:
        yield SharedState(path)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@pytest.fixture()
def temp_artifact():
    """Write a small file and return (path, its true sha256)."""
    fd, path = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as f:
        f.write(b"genomic-variant-classifier ScienceClaw artifact bytes v1")
    try:
        yield path, compute_sha256(path)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _msg(subject, payload, requires_approval, approved):
    """Build a message envelope matching the bus schema (minus bus-only keys)."""
    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "from_agent": "ProducerAgent",
        "to_agent": "ConsumerAgent",
        "subject": subject,
        "payload": dict(payload),
        "timestamp": "2026-05-30T00:00:00+00:00",
        "priority": mb.PRIORITY_NORMAL,
        "read": False,
        "requires_approval": requires_approval,
        "approved": approved,
    }


# ===========================================================================
# Group A -- new subject constant wiring
# ===========================================================================


def test_artifact_published_subject_exists():
    assert hasattr(mb, "ARTIFACT_PUBLISHED")
    assert mb.ARTIFACT_PUBLISHED in mb.ALL_SUBJECTS


def test_artifact_published_requires_approval_by_default():
    # User decision: ARTIFACT_PUBLISHED defaults to requires_approval=True.
    assert mb.ARTIFACT_PUBLISHED in mb.APPROVAL_REQUIRED_SUBJECTS


def test_unknown_subject_still_raises(temp_state):
    bus = mb.MessageBus(temp_state)
    with pytest.raises(ValueError):
        bus.send("A", "B", "DEFINITELY_NOT_A_SUBJECT")


def test_artifact_published_is_sendable(temp_state):
    bus = mb.MessageBus(temp_state)
    msg_id = bus.send("A", "B", mb.ARTIFACT_PUBLISHED, {"artifact_id": "x"})
    assert isinstance(msg_id, str) and len(msg_id) == 36
    inbox = bus.get_inbox("B")
    assert len(inbox) == 1
    # requires_approval defaulted True -> not yet approved
    assert inbox[0]["requires_approval"] is True
    assert inbox[0]["approved"] is None


# ===========================================================================
# Group B -- ledger: append-only + hash chain
# ===========================================================================


def test_ledger_append_and_lookup(temp_state, temp_artifact):
    path, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    entry = led.record(
        artifact_id="model_v1",
        producer="TrainingLifecycleAgent",
        sha256=true_sha,
        uri=path,
    )
    assert entry["artifact_id"] == "model_v1"
    assert entry["sha256"] == true_sha
    assert entry["index"] == 0
    assert led.lookup("model_v1")["sha256"] == true_sha


def test_ledger_persists_via_shared_state(temp_state, temp_artifact):
    _, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("a1", "ProducerAgent", true_sha, "u1")
    # New ledger instance over the same state must see the entry.
    led2 = ScienceClawLedger(temp_state)
    assert led2.lookup("a1") is not None
    # And it lives under the artifact_ledger key in state.
    assert "artifact_ledger" in temp_state.load()


def test_ledger_is_append_only_chain(temp_state):
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", "h1", "u1")
    led.record("a2", "P", "h2", "u2")
    entries = led.entries()
    assert [e["index"] for e in entries] == [0, 1]
    # Each entry chains the previous entry's row-hash.
    assert entries[0]["prev_hash"] is None
    assert entries[1]["prev_hash"] == entries[0]["row_hash"]


def test_ledger_detects_tamper(temp_state):
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", "h1", "u1")
    led.record("a2", "P", "h2", "u2")
    # Tamper with a prior entry directly in state, behind the ledger's back.
    state = temp_state.load()
    state["artifact_ledger"][0]["sha256"] = "TAMPERED"
    temp_state.save(state)
    # verify_chain must detect the broken hash chain.
    with pytest.raises(LedgerError):
        ScienceClawLedger(temp_state).verify_chain()


def test_ledger_rejects_duplicate_artifact_id(temp_state):
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", "h1", "u1")
    with pytest.raises(LedgerError):
        led.record("a1", "P", "h1-again", "u1b")


# ===========================================================================
# Group C -- compute_sha256 helper (caller-side, deterministic)
# ===========================================================================


def test_compute_sha256_matches_known(temp_artifact):
    path, true_sha = temp_artifact
    # 64 hex chars, stable across repeated calls.
    assert len(true_sha) == 64
    assert compute_sha256(path) == true_sha


def test_compute_sha256_missing_file_raises():
    with pytest.raises((FileNotFoundError, OSError)):
        compute_sha256("/no/such/file/exists.bin")


# ===========================================================================
# Group D -- the pure gate: determinism
# ===========================================================================


def test_gate_is_deterministic(temp_state, temp_artifact):
    _, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", true_sha, "u1")
    entries = led.entries()
    msg = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=True,
    )
    v1 = evaluate(entries, msg, computed_sha=true_sha)
    v2 = evaluate(entries, msg, computed_sha=true_sha)
    assert isinstance(v1, Verdict)
    assert v1.allow == v2.allow
    assert v1.reasons == v2.reasons


# ===========================================================================
# Group E -- integrity dimension
# ===========================================================================


def test_gate_allows_on_matching_hash(temp_state, temp_artifact):
    _, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", true_sha, "u1")
    msg = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=True,  # authorization satisfied so we isolate integrity
    )
    v = evaluate(led.entries(), msg, computed_sha=true_sha)
    assert v.allow is True
    assert v.reasons == []


def test_gate_denies_on_hash_mismatch(temp_state, temp_artifact):
    _, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", true_sha, "u1")
    msg = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=True,
    )
    v = evaluate(led.entries(), msg, computed_sha="0" * 64)  # on-disk != recorded
    assert v.allow is False
    assert any("integrity" in r.lower() or "hash" in r.lower() for r in v.reasons)


def test_gate_denies_on_missing_artifact(temp_state):
    led = ScienceClawLedger(temp_state)  # empty ledger
    msg = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "ghost", "artifact_sha256": "a" * 64},
        requires_approval=True,
        approved=True,
    )
    v = evaluate(led.entries(), msg, computed_sha="a" * 64)
    assert v.allow is False
    assert any("ledger" in r.lower() or "missing" in r.lower() for r in v.reasons)


# ===========================================================================
# Group F -- authorization dimension
# ===========================================================================


def test_gate_denies_when_approval_required_and_not_approved(temp_state, temp_artifact):
    _, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", true_sha, "u1")
    msg = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=None,  # pending -> authorization fails
    )
    v = evaluate(led.entries(), msg, computed_sha=true_sha)
    assert v.allow is False
    assert any("approv" in r.lower() for r in v.reasons)


def test_gate_allows_when_approved(temp_state, temp_artifact):
    _, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", true_sha, "u1")
    msg = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=True,
    )
    v = evaluate(led.entries(), msg, computed_sha=true_sha)
    assert v.allow is True


def test_gate_denies_when_rejected(temp_state, temp_artifact):
    _, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", true_sha, "u1")
    msg = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=False,  # explicitly rejected
    )
    v = evaluate(led.entries(), msg, computed_sha=true_sha)
    assert v.allow is False


# ===========================================================================
# Group G -- combined integrity AND authorization
# ===========================================================================


def test_gate_both_must_pass(temp_state, temp_artifact):
    _, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("a1", "P", true_sha, "u1")
    entries = led.entries()

    # integrity fails (bad sha) + authorization ok -> deny
    bad_integrity = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=True,
    )
    assert evaluate(entries, bad_integrity, computed_sha="f" * 64).allow is False

    # integrity ok + authorization fails (pending) -> deny
    bad_auth = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=None,
    )
    assert evaluate(entries, bad_auth, computed_sha=true_sha).allow is False

    # both ok -> allow
    good = _msg(
        mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "a1", "artifact_sha256": true_sha},
        requires_approval=True,
        approved=True,
    )
    assert evaluate(entries, good, computed_sha=true_sha).allow is True


# ===========================================================================
# Group H -- no-op safety: messages without artifact keys
# ===========================================================================


def test_gate_noop_integrity_when_no_artifact_keys(temp_state):
    """A message with no artifact_id/artifact_sha256 skips integrity entirely:
    authorization alone decides. Mirrors the existing bus actionable contract."""
    led = ScienceClawLedger(temp_state)  # empty ledger; must NOT matter
    # No approval required + no artifact keys -> allow (integrity is a no-op).
    msg_ok = _msg(
        mb.FEATURE_CANDIDATE_ADDED,
        {"candidate_name": "SplicePAS"},
        requires_approval=False,
        approved=None,
    )
    assert evaluate(led.entries(), msg_ok, computed_sha=None).allow is True

    # Approval required + not approved + no artifact keys -> deny on auth only.
    msg_pending = _msg(
        mb.DATA_UPDATED,
        {"source": "gnomAD"},
        requires_approval=True,
        approved=None,
    )
    v = evaluate(led.entries(), msg_pending, computed_sha=None)
    assert v.allow is False
    assert any("approv" in r.lower() for r in v.reasons)


def test_gate_partial_artifact_keys_is_treated_as_no_artifact(temp_state):
    """Only one of the two keys present -> not a valid artifact reference;
    integrity stays a no-op (authorization decides)."""
    led = ScienceClawLedger(temp_state)
    msg = _msg(
        mb.FEATURE_INSTABILITY,
        {"artifact_id": "a1"},  # missing artifact_sha256
        requires_approval=False,
        approved=None,
    )
    assert evaluate(led.entries(), msg, computed_sha=None).allow is True