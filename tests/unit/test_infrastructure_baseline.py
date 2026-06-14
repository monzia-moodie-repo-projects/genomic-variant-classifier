"""test_infrastructure_baseline.py  --  Monzia Moodie

Infrastructure activation (model-free): build_infrastructure_baseline.build_baseline +
InfrastructureDriftAgent.from_baseline + InfrastructureDriftMonitorAgent.from_default_baseline.
Green on a matching environment, amber on package/dag drift, red on golden-set divergence,
awaiting_baseline when a current input or the baseline file is absent.
"""
import importlib.util
import json
import os

import pandas as pd
import pytest

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.infrastructure_drift_agent import (
    InfrastructureDriftAgent,
)
from genomic_variant_classifier.agent_layer.agents.infrastructure_drift_monitor_agent import (
    InfrastructureDriftMonitorAgent,
)

_BUILDER = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_infrastructure_baseline.py")
DAG = '{"prep": ["clinvar", "gnomad", "engineer_features"]}'


def _builder():
    spec = importlib.util.spec_from_file_location("build_infrastructure_baseline", _BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def golden():
    return pd.DataFrame({"variant_id": ["v1", "v2", "v3"],
                         "feat_a": [0.1, 0.2, 0.3], "feat_b": [1.0, 0.0, 1.0]})


@pytest.fixture
def baseline_dict(golden):
    return _builder().build_baseline(golden, DAG, packages=["pandas", "numpy", "scipy"])


def test_build_baseline_shape(baseline_dict):
    assert set(baseline_dict) >= {"pinned_packages", "expected_dag_hash", "golden_set"}
    assert len(baseline_dict["expected_dag_hash"]) == 64  # sha256 hex
    assert baseline_dict["pinned_packages"]["pandas"]  # captured a real version
    assert baseline_dict["n_golden"] == 3


def test_from_baseline_roundtrip(tmp_path, baseline_dict, golden):
    bp = tmp_path / "infra.json"; bp.write_text(json.dumps(baseline_dict), encoding="utf-8")
    det = InfrastructureDriftAgent.from_baseline(bp, output_dir=tmp_path)
    r = det.detect(dict(baseline_dict["pinned_packages"]), DAG, golden.copy())
    assert r.severity == "green" and r.golden_set_divergence == 0


def test_active_green(tmp_path, baseline_dict, golden):
    bp = tmp_path / "infra.json"; bp.write_text(json.dumps(baseline_dict), encoding="utf-8")
    state = SharedState(state_file=tmp_path / "state.json")
    agent = InfrastructureDriftMonitorAgent.from_default_baseline(
        state, current_packages=dict(baseline_dict["pinned_packages"]),
        current_dag_spec=DAG, replayed_features=golden.copy(),
        baseline_path=bp, output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "green"


def test_amber_on_package_drift(tmp_path, baseline_dict, golden):
    bp = tmp_path / "infra.json"; bp.write_text(json.dumps(baseline_dict), encoding="utf-8")
    state = SharedState(state_file=tmp_path / "state.json")
    cur = dict(baseline_dict["pinned_packages"]); cur["numpy"] = "0.0.0-fake"
    agent = InfrastructureDriftMonitorAgent.from_default_baseline(
        state, current_packages=cur, current_dag_spec=DAG, replayed_features=golden.copy(),
        baseline_path=bp, output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "amber"
    assert "numpy" in out["package_changes"]


def test_red_on_golden_divergence(tmp_path, baseline_dict, golden):
    bp = tmp_path / "infra.json"; bp.write_text(json.dumps(baseline_dict), encoding="utf-8")
    state = SharedState(state_file=tmp_path / "state.json")
    rep = golden.copy(); rep.loc[0, "feat_a"] = 0.999
    agent = InfrastructureDriftMonitorAgent.from_default_baseline(
        state, current_packages=dict(baseline_dict["pinned_packages"]),
        current_dag_spec=DAG, replayed_features=rep,
        baseline_path=bp, output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "red"
    assert out["golden_set_divergence"] == 1


def test_auto_resolves_current_packages(tmp_path, golden):
    # baseline must pin the SAME set from_default_baseline auto-resolves (the full
    # monitored_packages), as the builder does by default; pinned == live -> green.
    mon = list(InfrastructureDriftAgent.__dataclass_fields__["monitored_packages"].default)
    bdict = _builder().build_baseline(golden, DAG, packages=mon)
    bp = tmp_path / "infra.json"; bp.write_text(json.dumps(bdict), encoding="utf-8")
    state = SharedState(state_file=tmp_path / "state.json")
    agent = InfrastructureDriftMonitorAgent.from_default_baseline(
        state, current_dag_spec=DAG, replayed_features=golden.copy(),
        baseline_path=bp, output_dir=tmp_path / "rep",
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok" and out["severity"] == "green"


def test_awaiting_when_dag_spec_absent(tmp_path, baseline_dict, golden, monkeypatch):
    monkeypatch.delenv("GVC_INFRA_DAG_SPEC", raising=False)
    bp = tmp_path / "infra.json"; bp.write_text(json.dumps(baseline_dict), encoding="utf-8")
    state = SharedState(state_file=tmp_path / "state.json")
    agent = InfrastructureDriftMonitorAgent.from_default_baseline(
        state, replayed_features=golden.copy(), baseline_path=bp, output_dir=tmp_path / "rep",
    )
    assert agent.run(dry_run=True)["status"] == "awaiting_baseline"


def test_awaiting_when_baseline_absent(tmp_path):
    state = SharedState(state_file=tmp_path / "state.json")
    agent = InfrastructureDriftMonitorAgent.from_default_baseline(
        state, baseline_path=tmp_path / "nope.json"
    )
    assert agent.run(dry_run=True)["status"] == "awaiting_baseline"
