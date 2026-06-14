"""test_finops_advisor_agent.py -- Monzia Moodie
FinOps advisor adapter: within-budget recommendation + 'finops' section + report; no snapshot -> skipped; over
budget -> HITL review item (real run) but not in dry-run; and a guard that the agent has no spend/exec surface.
"""
import json
from pathlib import Path

from genomic_variant_classifier.agent_layer.agents.finops_advisor_agent import FinOpsAdvisorAgent, _SECTION
from genomic_variant_classifier.agent_layer.shared_state import SharedState

OFFERS = [
    {"id": 2, "dph_total": 0.40, "reliability2": 0.991, "cpu_ram": 64, "num_gpus": 1},
    {"id": 3, "dph_total": 0.40, "reliability2": 0.999, "cpu_ram": 64, "num_gpus": 1},
]


def _snapshot(tmp_path, offers=OFFERS):
    p = tmp_path / "offers.json"; p.write_text(json.dumps(offers)); return str(p)


def _agent(tmp_path, snapshot=None, budget=15.0):
    ss = SharedState(state_file=tmp_path / "state.json")
    return FinOpsAdvisorAgent(ss, snapshot_path=snapshot, est_hours=15, budget_usd=budget,
                              root=str(tmp_path)), ss


def test_within_budget_recommendation(tmp_path):
    agent, ss = _agent(tmp_path, snapshot=_snapshot(tmp_path), budget=15.0)
    res = agent.run(dry_run=True)
    assert res["action"] == "finops_recommendation" and res["verdict"] == "WITHIN_BUDGET"
    assert res["chosen_id"] == 3 and res["est_cost"] == 6.0
    rpt = list((tmp_path / "reports" / "finops").glob("FINOPS_*.md"))
    assert len(rpt) == 1 and "no spend" in rpt[0].read_text()
    sec = ss.get_section(_SECTION)
    assert sec["verdict"] == "WITHIN_BUDGET" and "--dry-run" in sec["command"]


def test_no_snapshot_is_skipped(tmp_path):
    agent, ss = _agent(tmp_path, snapshot=None)
    res = agent.run(dry_run=True)
    assert res["action"] == "skipped" and ss.get_section(_SECTION) == {}    # nothing recorded, no live call


def test_over_budget_opens_hitl_on_real_run(tmp_path):
    agent, ss = _agent(tmp_path, snapshot=_snapshot(tmp_path), budget=5.0)   # 6.0 > 5 -> over
    res = agent.run(dry_run=False)
    assert res["verdict"] == "OVER_BUDGET"
    assert len(ss.unresolved_review_items()) == 1                            # HITL raised


def test_over_budget_no_hitl_in_dry_run(tmp_path):
    agent, ss = _agent(tmp_path, snapshot=_snapshot(tmp_path), budget=5.0)
    agent.run(dry_run=True)
    assert ss.unresolved_review_items() == []                               # gate skipped in dry-run


def test_agent_has_no_spend_surface():
    import genomic_variant_classifier.agent_layer.agents.finops_advisor_agent as m
    src = Path(m.__file__).read_text()
    assert "subprocess" not in src and "vastai create" not in src and "os.system" not in src
