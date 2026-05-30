#!/usr/bin/env python3
"""
test_science_claw_orchestrator_gate.py -- Task 3 orchestrator hot-path wiring
=============================================================================
Failing-first tests for Orchestrator.enforce_artifact_gate(...).

Fixturing decision (locked with the user): REAL temp SharedState + REAL
MessageBus + REAL ScienceClawLedger + a REAL temp artifact file on disk. No
unittest.mock patch() of any import path -- this deliberately avoids the stale
"agents.data_freshness_agent" patch-target problem that causes the 2
pre-existing Group-4 errors in test_message_bus.py, and it exercises the true
production path.

Behavior under test (locked with the user):
  * enforce_artifact_gate(agent_names) runs BEFORE the agent loop.
  * For each agent's actionable, artifact-referencing message (payload carries
    BOTH artifact_id and artifact_sha256), it loads the ledger, computes the
    on-disk SHA-256 from the LEDGER ROW's uri, calls evaluate(...), and on DENY:
        - rejects the message via the bus (message["approved"] becomes False), AND
        - adds a human-review item.
    DENY BLOCKS: a denied artifact message must not remain actionable this run.
  * Messages with no artifact reference are untouched (no-op).
  * run_pipeline invokes enforce_artifact_gate (guardrail test).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer import message_bus as mb
from genomic_variant_classifier.agent_layer.science_claw import (
    ScienceClawLedger,
    compute_sha256,
)
from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Real fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_state():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)
    try:
        yield SharedState(path)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@pytest.fixture()
def temp_artifact():
    fd, path = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as f:
        f.write(b"ScienceClaw orchestrator-gate real artifact payload v1")
    try:
        yield path, compute_sha256(path)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


# A real agent name the orchestrator knows about.
_AGENT = "TrainingLifecycleAgent"


# ---------------------------------------------------------------------------
# Method existence + wiring guardrail
# ---------------------------------------------------------------------------


def test_enforce_artifact_gate_method_exists(temp_state):
    orch = Orchestrator(temp_state, dry_run=True)
    assert hasattr(orch, "enforce_artifact_gate")
    assert callable(orch.enforce_artifact_gate)


def test_run_pipeline_invokes_gate(temp_state, monkeypatch):
    """run_pipeline must call enforce_artifact_gate before running agents."""
    orch = Orchestrator(temp_state, dry_run=True)
    called = {"n": 0, "names": None}

    def _spy(agent_names):
        called["n"] += 1
        called["names"] = list(agent_names)

    monkeypatch.setattr(orch, "enforce_artifact_gate", _spy)
    # data_freshness pipeline is a single known agent; dry_run keeps agents cheap.
    orch.run_pipeline("data_freshness")
    assert called["n"] == 1
    assert called["names"] == ["DataFreshnessAgent"]


# ---------------------------------------------------------------------------
# DENY blocks: tampered artifact (on-disk hash != ledger hash)
# ---------------------------------------------------------------------------


def test_gate_blocks_tampered_artifact(temp_state, temp_artifact):
    path, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("modelA", _AGENT, true_sha, path)

    bus = mb.MessageBus(temp_state)
    msg_id = bus.send(
        "ProducerAgent", _AGENT, mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "modelA", "artifact_sha256": true_sha},
    )
    bus.approve(msg_id)  # authorization satisfied -> isolate integrity
    assert any(m["id"] == msg_id for m in bus.get_actionable(_AGENT))

    # Tamper the file on disk so its real hash no longer matches the ledger.
    with open(path, "ab") as f:
        f.write(b"TAMPER")

    orch = Orchestrator(temp_state, dry_run=True)
    orch.enforce_artifact_gate([_AGENT])

    # DENY blocks: message rejected, no longer actionable, review item added.
    inbox = bus.get_inbox(_AGENT)
    target = [m for m in inbox if m["id"] == msg_id][0]
    assert target["approved"] is False, "tampered artifact message must be rejected"
    assert msg_id not in [m["id"] for m in bus.get_actionable(_AGENT)]
    assert len(temp_state.unresolved_review_items()) >= 1


def test_gate_blocks_missing_from_ledger(temp_state, temp_artifact):
    path, true_sha = temp_artifact
    # NOTE: do NOT record in ledger -> integrity must deny (missing).
    bus = mb.MessageBus(temp_state)
    msg_id = bus.send(
        "ProducerAgent", _AGENT, mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "ghost", "artifact_sha256": true_sha},
    )
    bus.approve(msg_id)

    orch = Orchestrator(temp_state, dry_run=True)
    orch.enforce_artifact_gate([_AGENT])

    target = [m for m in bus.get_inbox(_AGENT) if m["id"] == msg_id][0]
    assert target["approved"] is False
    assert len(temp_state.unresolved_review_items()) >= 1


# ---------------------------------------------------------------------------
# ALLOW: valid artifact (on-disk hash == ledger hash) stays actionable
# ---------------------------------------------------------------------------


def test_gate_allows_valid_artifact(temp_state, temp_artifact):
    path, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("modelOK", _AGENT, true_sha, path)

    bus = mb.MessageBus(temp_state)
    msg_id = bus.send(
        "ProducerAgent", _AGENT, mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "modelOK", "artifact_sha256": true_sha},
    )
    bus.approve(msg_id)

    orch = Orchestrator(temp_state, dry_run=True)
    orch.enforce_artifact_gate([_AGENT])

    target = [m for m in bus.get_inbox(_AGENT) if m["id"] == msg_id][0]
    assert target["approved"] is True, "valid artifact must remain approved"
    assert msg_id in [m["id"] for m in bus.get_actionable(_AGENT)]
    assert len(temp_state.unresolved_review_items()) == 0


# ---------------------------------------------------------------------------
# No-op: messages without an artifact reference are untouched
# ---------------------------------------------------------------------------


def test_gate_noop_for_non_artifact_message(temp_state):
    bus = mb.MessageBus(temp_state)
    # FEATURE_CANDIDATE_ADDED carries no artifact keys; requires_approval False.
    msg_id = bus.send(
        "LiteratureScoutAgent", _AGENT, mb.FEATURE_CANDIDATE_ADDED,
        {"candidate_name": "SplicePAS"}, requires_approval=False,
    )
    before = bus.get_inbox(_AGENT)[0]["approved"]

    orch = Orchestrator(temp_state, dry_run=True)
    orch.enforce_artifact_gate([_AGENT])

    after = [m for m in bus.get_inbox(_AGENT) if m["id"] == msg_id][0]
    assert after["approved"] == before, "non-artifact message must be untouched"
    assert len(temp_state.unresolved_review_items()) == 0


def test_gate_ignores_unapproved_artifact_message(temp_state, temp_artifact):
    """A pending (unapproved) artifact message is NOT actionable yet, so the gate
    does not reject it -- it is simply not in scope until approved."""
    path, true_sha = temp_artifact
    led = ScienceClawLedger(temp_state)
    led.record("modelPending", _AGENT, true_sha, path)

    bus = mb.MessageBus(temp_state)
    msg_id = bus.send(
        "ProducerAgent", _AGENT, mb.ARTIFACT_PUBLISHED,
        {"artifact_id": "modelPending", "artifact_sha256": true_sha},
    )
    # ARTIFACT_PUBLISHED requires approval by default; leave it pending.
    assert msg_id not in [m["id"] for m in bus.get_actionable(_AGENT)]

    orch = Orchestrator(temp_state, dry_run=True)
    orch.enforce_artifact_gate([_AGENT])

    # Still pending (None), not force-rejected, no review item.
    target = [m for m in bus.get_inbox(_AGENT) if m["id"] == msg_id][0]
    assert target["approved"] is None
    assert len(temp_state.unresolved_review_items()) == 0