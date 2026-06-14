"""test_agent_ops_monitor_agent.py -- Monzia Moodie
Hermetic tests for the AgentOpsMonitorAgent adapter: OK when all fresh; ATTENTION on a stale section; writes a
dashboard report + records its own 'agent_ops' heartbeat (self-monitoring). root is tmp; no real reports/ touched.
"""
from datetime import datetime, timedelta, timezone

from genomic_variant_classifier.agent_layer.agents.agent_ops_monitor_agent import AgentOpsMonitorAgent, _SECTION
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.evaluation import agent_ops_detector as D


def _now_iso(delta_hours=0):
    return (datetime.now(timezone.utc) - timedelta(hours=delta_hours)).isoformat()


def _agent(tmp_path):
    ss = SharedState(state_file=tmp_path / "state.json")
    return AgentOpsMonitorAgent(ss, root=str(tmp_path)), ss


def test_ok_when_all_fresh(tmp_path):
    agent, ss = _agent(tmp_path)
    ss.update_section("training", {"last_run": _now_iso(1)})       # fresh
    res = agent.run(dry_run=True)
    assert res["action"] == "agent_ops_scan" and res["ops_status"] == "OK"
    reports = list((tmp_path / "reports" / "agent_ops").glob("OPS_*.md"))
    assert len(reports) == 1 and "STATUS: OK" in reports[0].read_text()
    sec = ss.get_section(_SECTION)                                  # self-monitoring: own heartbeat recorded
    assert sec["ops_status"] == "OK" and "last_run" in sec


def test_attention_on_stale_section(tmp_path):
    agent, ss = _agent(tmp_path)
    ss.update_section("database_freshness", {"clinvar": {"last_checked": _now_iso(24 * 40)}})  # 40d -> stale
    res = agent.run(dry_run=True)
    assert res["ops_status"] == "ATTENTION" and res["n_stale"] >= 1


def test_attention_on_unresolved_review(tmp_path):
    agent, ss = _agent(tmp_path)
    ss.update_section("training", {"last_run": _now_iso(1)})
    ss.add_review_item("a human needs to look at this")
    res = agent.run(dry_run=True)
    assert res["ops_status"] == "ATTENTION" and res["unresolved_reviews"] == 1


def test_self_heartbeat_appears_on_second_run(tmp_path):
    agent, ss = _agent(tmp_path)
    ss.update_section("training", {"last_run": _now_iso(1)})
    agent.run(dry_run=True)                                         # records agent_ops
    res2 = agent.run(dry_run=True)                                  # now agent_ops is one of the sections scanned
    sections = {b.section for b in D.analyze(ss.load())["heartbeats"]}
    assert "agent_ops" in sections and res2["ops_status"] in ("OK", "ATTENTION")


def test_telemetry_surfaced_in_report_and_state(tmp_path):
    agent, ss = _agent(tmp_path)
    ss.update_section("training", {"last_run": _now_iso(1)})
    ss.update_section("agent_runs", {"Flaky": [{"status": "error", "duration_ms": 5},
                                               {"status": "ok", "duration_ms": 6}]})
    res = agent.run(dry_run=True)
    assert res["ops_status"] == "ATTENTION"                         # agent error -> attention
    rpt = list((tmp_path / "reports" / "agent_ops").glob("OPS_*.md"))[0].read_text()
    assert "Run telemetry" in rpt and "Flaky" in rpt
    sec = ss.get_section(_SECTION)
    assert "Flaky" in sec["agents_with_errors"]
