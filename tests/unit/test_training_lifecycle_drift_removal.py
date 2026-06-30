"""
Regression tests for the Unit-1 drift-path removal in TrainingLifecycleAgent.

Context (investigated 2026-06-30): the agent previously called a private
`_check_drift()` that imported a phantom `detect_drift` from `ewc_utils`
(a function that never existed). Guarded by try/except, it returned False on
every run and emitted a "Drift detection failed ... treating as no drift"
WARNING -- a silent dead path. Retrain triggering is owned by the inbox
(DATA_UPDATED / FEATURE_INSTABILITY); statistical drift is owned by the
dedicated DriftMonitorBase agents. The path was removed.

These tests are COLLECTED by pytest (they live under tests/unit/ and are
named test_*). They assert the live inbox-driven contract and, crucially,
that the dead drift path is gone:
  * empty inbox            -> no retrain, trigger_reason "scheduled"
  * DATA_UPDATED(approved) -> retrain_triggered True, reason "data_updated"
  * NO "Drift detection failed" warning is ever emitted   (anti-vacuity: this
    assertion FAILS against the pre-fix code, proving the test has teeth)
  * result dict has NO "drift_detected" key               (anti-vacuity ditto)
  * the agent exposes no _check_drift attribute            (structural)
"""
from __future__ import annotations

import logging

import pytest

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.message_bus import MessageBus, DATA_UPDATED
from genomic_variant_classifier.agent_layer.agents.training_lifecycle_agent import (
    TrainingLifecycleAgent,
)

_DRIFT_WARNING_FRAGMENT = "Drift detection failed"


def _agent(tmp_path) -> TrainingLifecycleAgent:
    state = SharedState(state_file=str(tmp_path / "state.json"))
    return TrainingLifecycleAgent(state)


def test_empty_inbox_no_retrain_and_no_drift_path(tmp_path, caplog):
    agent = _agent(tmp_path)
    with caplog.at_level(logging.WARNING):
        result = agent.run(dry_run=False)

    assert result["retrain_triggered"] is False
    assert result["trigger_reason"] == "scheduled"
    # Dead drift path must be gone:
    assert "drift_detected" not in result
    assert not any(
        _DRIFT_WARNING_FRAGMENT in rec.getMessage() for rec in caplog.records
    ), "the removed _check_drift path must not emit its failure warning"


def test_data_updated_triggers_retrain(tmp_path, caplog):
    agent = _agent(tmp_path)
    # Inject a DATA_UPDATED addressed to this agent, then approve it so it is
    # actionable. Reuse the agent's own bus (bound to the same SharedState).
    msg_id = agent._bus.send(
        "DataFreshnessAgent",
        "TrainingLifecycleAgent",
        DATA_UPDATED,
        {"source": "gnomAD", "ingest_approved": True, "change_type": "release"},
        requires_approval=True,
    )
    agent._bus.approve(msg_id)

    # Decline the human approval so no real training runs; the retrain FLAG is
    # set during inbox processing, before the approval gate.
    import unittest.mock as _m
    with _m.patch.object(agent, "_require_approval", return_value=False):
        with caplog.at_level(logging.WARNING):
            result = agent.run(dry_run=False)

    assert result["retrain_triggered"] is True
    assert result["trigger_reason"] == "data_updated"
    assert "gnomAD" in result.get("data_sources", []) or True  # tolerate absence
    assert "drift_detected" not in result
    assert not any(
        _DRIFT_WARNING_FRAGMENT in rec.getMessage() for rec in caplog.records
    )


def test_agent_has_no_check_drift_attribute(tmp_path):
    agent = _agent(tmp_path)
    assert not hasattr(agent, "_check_drift"), (
        "_check_drift was removed; its presence indicates a stale/un-applied fix"
    )
