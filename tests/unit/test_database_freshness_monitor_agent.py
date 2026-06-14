"""test_database_freshness_monitor_agent.py -- Monzia Moodie

Hermetic tests for the BaseAgent monitor adapter: dry-run scans + writes a report + records SharedState with
NO network (mocked probe); a detected change emits DATA_UPDATED and is HITL-gated. root is a tmp dir so no
real data/ or reports/ are touched.
"""
from genomic_variant_classifier.agent_layer.agents.database_freshness_monitor_agent import (
    DatabaseFreshnessMonitorAgent, _SECTION,
)
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.monitoring import registry as R


def _agent(tmp_path, probe):
    ss = SharedState(state_file=tmp_path / "state.json")
    return DatabaseFreshnessMonitorAgent(ss, probe=probe, root=str(tmp_path)), ss


def test_dry_run_scans_writes_report_records_state(tmp_path):
    agent, ss = _agent(tmp_path, probe=lambda s: ("fp-1", "mock"))
    res = agent.run(dry_run=True)
    assert res["action"] == "registry_freshness_scan"
    assert res["sources"] == len(R.all_sources())
    # report written under tmp (no real reports/ touched)
    reports = list((tmp_path / "reports" / "data_freshness").glob("FRESHNESS_*.md"))
    assert len(reports) == 1 and "Upstream" in reports[0].read_text()
    # state recorded for every source
    sec = ss.get_section(_SECTION)
    assert set(sec) == {s.key for s in R.all_sources()}
    assert all("last_checked" in v for v in sec.values())


def test_change_emits_data_updated_and_is_hitl_gated(tmp_path, monkeypatch):
    agent, ss = _agent(tmp_path, probe=lambda s: ("new-fp", "mock"))
    sent = []
    approvals = []
    monkeypatch.setattr(agent, "send_message",
                        lambda **kw: sent.append(kw))
    monkeypatch.setattr(agent, "_require_approval",
                        lambda prompt, dry_run=False: approvals.append(prompt) or True)
    res = agent.run(dry_run=False)
    # exactly the probeable sources are 'changed' on first observation
    assert res["changes_detected"] == len(R.probeable())
    # each change -> a DATA_UPDATED to TrainingLifecycleAgent, and a HITL approval prompt
    assert len(sent) == len(R.probeable())
    assert all(m["subject"] == "DATA_UPDATED" and m["to"] == "TrainingLifecycleAgent" for m in sent)
    assert len(approvals) == len(R.probeable())
    # the approval prompt names the registry `acquire` path, NOT gcloud/dataproc
    assert all("gcloud" not in p and "dataproc" not in p for p in approvals)


def test_second_run_no_change_after_persist(tmp_path):
    # run once (records last_seen), then run again with the SAME fingerprint -> no changes
    agent, ss = _agent(tmp_path, probe=lambda s: ("stable", "mock"))
    first = agent.run(dry_run=False)              # records last_seen="stable" for probeable sources
    assert first["changes_detected"] == len(R.probeable())
    res = agent.run(dry_run=False)               # same fingerprint -> nothing changed
    assert res["changes_detected"] == 0
