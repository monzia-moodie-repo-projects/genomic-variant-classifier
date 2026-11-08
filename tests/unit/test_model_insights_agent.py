"""test_model_insights_agent.py -- Monzia Moodie
Hermetic tests for the ModelInsightsAgent BaseAgent adapter over REAL completed runs (package 1, set 1b): dry-run
discovers the latest completed run, writes a report
under a tmp root, and records SharedState; no run artifacts -> 'skipped'; a serious integrity flag emits one
informational FEATURE_INSTABILITY to TrainingLifecycleAgent; dry-run emits nothing. No real outputs/ or reports/
are touched (outputs_root + root are tmp).
"""
from genomic_variant_classifier.agent_layer.agents.model_insights_agent import (
    ModelInsightsAgent, _SECTION,
)
from genomic_variant_classifier.agent_layer.shared_state import SharedState


def _write_oof(tmp_path, *, leaky=False, disjoint=True, run="run99"):
    """A REAL completed run (production writer, fitted two-member ensemble). `disjoint=False` fits without gene groups,
    so the recorded StratifiedKFold folds share genes -- the Run 15 situation the detector must flag."""
    from tests.unit._run_artifact_support import completed_run

    return completed_run(tmp_path / "outputs" / run / "full", seed=0, leaky=leaky, gene_groups=disjoint)


def _agent(tmp_path):
    ss = SharedState(state_file=tmp_path / "state.json")
    return ModelInsightsAgent(ss, outputs_root=str(tmp_path / "outputs"), root=str(tmp_path)), ss


def test_dry_run_writes_report_and_records_state(tmp_path):
    _write_oof(tmp_path, leaky=False, disjoint=True)
    agent, ss = _agent(tmp_path)
    res = agent.run(dry_run=True)
    assert res["action"] == "model_insights_scan"
    assert res["models"] == 2                                   # the two fitted members; no blend row (in-sample)
    reports = list((tmp_path / "reports" / "model_insights").glob("INSIGHTS_*.md"))
    assert len(reports) == 1
    body = reports[0].read_text()
    assert "Per-model metrics" in body and "by MCC" in body and "docs/METRICS.md" in body
    sec = ss.get_section(_SECTION)
    assert sec["ranking_by_mcc"] and sec["run_dir"].endswith("full") and isinstance(sec["metrics"], list)


def test_an_incomplete_run_is_listed_and_never_analysed(tmp_path):
    rd = _write_oof(tmp_path)
    (rd / "completion_manifest.json").unlink()
    agent, ss = _agent(tmp_path)
    res = agent.run(dry_run=True)
    assert res["action"] == "skipped" and res["reason"] == "no_completed_run"
    assert [i["run_dir"] for i in res["incomplete_runs"]] == [str(rd)]
    assert not (tmp_path / "reports").exists()


def test_no_run_artifacts_skips(tmp_path):
    agent, _ = _agent(tmp_path)                                 # no outputs/ created
    res = agent.run(dry_run=True)
    assert res["action"] == "skipped" and res["reason"] == "no_run_artifacts"


def test_serious_flag_emits_one_feature_instability(tmp_path, monkeypatch):
    _write_oof(tmp_path, leaky=True, disjoint=False)            # leakage + gene-disjoint violation
    agent, _ = _agent(tmp_path)
    sent = []
    monkeypatch.setattr(agent, "send_message", lambda **kw: sent.append(kw))
    res = agent.run(dry_run=False)
    assert res["serious_flags"] >= 1
    assert len(sent) == 1
    assert sent[0]["subject"] == "FEATURE_INSTABILITY" and sent[0]["to"] == "TrainingLifecycleAgent"
    assert sent[0]["payload"]["flags"]


def test_dry_run_does_not_emit(tmp_path, monkeypatch):
    _write_oof(tmp_path, leaky=True, disjoint=False)
    agent, _ = _agent(tmp_path)
    sent = []
    monkeypatch.setattr(agent, "send_message", lambda **kw: sent.append(kw))
    res = agent.run(dry_run=True)
    assert res["serious_flags"] >= 1 and len(sent) == 0         # dry-run never emits
