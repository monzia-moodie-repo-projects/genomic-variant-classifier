"""test_data_readiness_agent.py -- Monzia Moodie
Hermetic tests for the DataReadinessAgent BaseAgent adapter: GO when all critical assets present (no splits),
NO_GO + HITL override gate when an asset is missing, and GO_WITH_WARNINGS via the optional feature-health
dimension when an injected splits parquet has a degenerate column. critical_assets() is monkeypatched to point
at tmp files; root is tmp so no real data/ or reports/ are touched. Verify-only: never runs data-prep.
"""
import numpy as np
import pandas as pd

from genomic_variant_classifier.agent_layer.agents.data_readiness_agent import DataReadinessAgent, _SECTION
from genomic_variant_classifier.agent_layer import agents  # noqa
from genomic_variant_classifier.monitoring import registry as R
from genomic_variant_classifier.evaluation import data_readiness_detector as D
from genomic_variant_classifier.agent_layer.shared_state import SharedState


def _agent(tmp_path, splits_path=None):
    ss = SharedState(state_file=tmp_path / "state.json")
    return DataReadinessAgent(ss, root=str(tmp_path), splits_path=splits_path), ss


def _make_assets(tmp_path, present=True, names=("data/a.parquet", "data/b.parquet")):
    paths = []
    for n in names:
        p = tmp_path / n
        if present:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * 100)
        paths.append(n)
    return paths


def test_go_when_assets_present_no_splits(tmp_path, monkeypatch):
    names = _make_assets(tmp_path, present=True)
    monkeypatch.setattr(R, "critical_assets", lambda: names)
    agent, ss = _agent(tmp_path)
    res = agent.run(dry_run=True)
    assert res["verdict"] == D.GO and res["missing_assets"] == []
    reports = list((tmp_path / "reports" / "data_readiness").glob("READINESS_*.md"))
    assert len(reports) == 1 and "VERDICT: GO" in reports[0].read_text()
    sec = ss.get_section(_SECTION)
    assert sec["verdict"] == "GO" and sec["n_assets"] == 2


def test_no_go_and_hitl_gate_when_asset_missing(tmp_path, monkeypatch):
    names = _make_assets(tmp_path, present=False)          # files NOT created -> MISSING
    monkeypatch.setattr(R, "critical_assets", lambda: names)
    agent, _ = _agent(tmp_path)
    approvals = []
    monkeypatch.setattr(agent, "_require_approval", lambda prompt, dry_run=False: approvals.append(prompt) or True)
    res = agent.run(dry_run=False)
    assert res["verdict"] == D.NO_GO and len(res["missing_assets"]) == 2
    assert len(approvals) == 1 and "NO_GO" in approvals[0]   # HITL override gate opened


def test_dry_run_no_go_does_not_gate(tmp_path, monkeypatch):
    names = _make_assets(tmp_path, present=False)
    monkeypatch.setattr(R, "critical_assets", lambda: names)
    agent, _ = _agent(tmp_path)
    approvals = []
    monkeypatch.setattr(agent, "_require_approval", lambda prompt, dry_run=False: approvals.append(prompt) or True)
    res = agent.run(dry_run=True)
    assert res["verdict"] == D.NO_GO and approvals == []     # dry-run never gates


def test_feature_health_dimension_with_injected_splits(tmp_path, monkeypatch):
    names = _make_assets(tmp_path, present=True)
    monkeypatch.setattr(R, "critical_assets", lambda: names)
    # a splits parquet: meta cols + 3 good + 1 degenerate (25% < 50% block) -> GO_WITH_WARNINGS
    rng = np.random.default_rng(2)
    sp = tmp_path / "splits.parquet"
    pd.DataFrame({"variant_id": np.arange(120), "gene_symbol": ["G"] * 120, "fold": np.arange(120) % 5,
                  "label": (np.arange(120) % 2), "feat_a": rng.standard_normal(120),
                  "feat_b": rng.standard_normal(120), "feat_c": rng.integers(0, 7, 120),
                  "feat_dead": np.zeros(120)}).to_parquet(sp)
    agent, ss = _agent(tmp_path, splits_path=str(sp))
    res = agent.run(dry_run=True)
    assert res["verdict"] == D.GO_WITH_WARNINGS and res["n_degenerate"] == 1
    sec = ss.get_section(_SECTION)
    assert sec["n_feature_cols"] == 4 and sec["splits_source"].endswith("splits.parquet")  # meta cols excluded
