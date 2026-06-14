"""test_model_insights_agent.py -- Monzia Moodie
Hermetic tests for the ModelInsightsAgent BaseAgent adapter: dry-run discovers the latest run, writes a report
under a tmp root, and records SharedState; no run artifacts -> 'skipped'; a serious integrity flag emits one
informational FEATURE_INSTABILITY to TrainingLifecycleAgent; dry-run emits nothing. No real outputs/ or reports/
are touched (outputs_root + root are tmp).
"""
import numpy as np
import pandas as pd

from genomic_variant_classifier.agent_layer.agents.model_insights_agent import (
    ModelInsightsAgent, _SECTION,
)
from genomic_variant_classifier.agent_layer.shared_state import SharedState


def _write_oof(tmp_path, *, leaky=False, disjoint=True, run="run99"):
    rd = tmp_path / "outputs" / run / "full"
    rd.mkdir(parents=True, exist_ok=True)
    n = 300
    rng = np.random.default_rng(0)
    y = (np.arange(n) % 3 == 0).astype(int)
    if leaky:
        strong = y * 0.98 + 0.01                       # near-perfect -> AUROC >= 0.99
    else:
        strong = np.clip(0.2 * rng.standard_normal(n) + 0.7 * y + 0.15, 0, 1)
    weak = np.clip(0.4 * rng.standard_normal(n) + 0.5, 0, 1)
    folds = np.arange(n) % 4
    genes = [f"SHARED{i % 3}" for i in range(n)] if not disjoint else [f"G{f}_{i}" for i, f in enumerate(folds)]
    pd.DataFrame({"variant_id": np.arange(n), "gene_symbol": genes, "fold": folds, "label": y,
                  "strong_prob": strong, "weak_prob": weak, "ensemble_prob": strong}
                 ).to_parquet(rd / "oof_predictions.parquet")
    return rd


def _agent(tmp_path):
    ss = SharedState(state_file=tmp_path / "state.json")
    return ModelInsightsAgent(ss, outputs_root=str(tmp_path / "outputs"), root=str(tmp_path)), ss


def test_dry_run_writes_report_and_records_state(tmp_path):
    _write_oof(tmp_path, leaky=False, disjoint=True)
    agent, ss = _agent(tmp_path)
    res = agent.run(dry_run=True)
    assert res["action"] == "model_insights_scan"
    assert res["models"] >= 3                                   # strong, weak, ensemble
    reports = list((tmp_path / "reports" / "model_insights").glob("INSIGHTS_*.md"))
    assert len(reports) == 1
    body = reports[0].read_text()
    assert "Per-model metrics" in body and "by MCC" in body and "docs/METRICS.md" in body
    sec = ss.get_section(_SECTION)
    assert sec["ranking_by_mcc"] and sec["run_dir"].endswith("full") and isinstance(sec["metrics"], list)


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
