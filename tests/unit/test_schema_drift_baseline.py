"""test_schema_drift_baseline.py  --  Monzia Moodie

SchemaDriftMonitorAgent.from_default_baseline activates the agent from the canonical
schema baseline (awaiting_baseline -> active detection): green on a matching matrix,
red on an added column, graceful when the baseline file is absent, env-var matrix
resolution, and the bare constructor still awaiting (unchanged default).
"""
import json

import pandas as pd
import pytest

pytest.importorskip("pandera")

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import SchemaDriftAgent
from genomic_variant_classifier.agent_layer.agents.schema_drift_monitor_agent import (
    SchemaDriftMonitorAgent,
)


def _write_baseline(path, df):
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    path.write_text(
        json.dumps(
            {
                "expected_dtypes": dtypes,
                "expected_schema_hash": SchemaDriftAgent.hash_schema(dtypes),
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def ref_df():
    return pd.DataFrame(
        {
            "a": pd.Series([1.0, 2.0], dtype="float64"),
            "b": pd.Series([0.1, 0.2], dtype="float64"),
        }
    )


@pytest.fixture
def state(tmp_path):
    return SharedState(state_file=tmp_path / "state.json")


def test_active_green_on_matching_matrix(tmp_path, state, ref_df):
    bp = tmp_path / "schema_baseline.json"
    _write_baseline(bp, ref_df)
    mx = tmp_path / "X.parquet"
    ref_df.to_parquet(mx, index=False)
    agent = SchemaDriftMonitorAgent.from_default_baseline(
        state, matrix_path=mx, baseline_path=bp, output_dir=tmp_path / "rep"
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok"
    assert out["severity"] == "green"
    assert not out["columns_added"]


def test_active_red_on_added_column(tmp_path, state, ref_df):
    bp = tmp_path / "schema_baseline.json"
    _write_baseline(bp, ref_df)
    drifted = ref_df.copy()
    drifted["c_new"] = pd.Series([9.0, 9.0], dtype="float64")
    mx = tmp_path / "X2.parquet"
    drifted.to_parquet(mx, index=False)
    agent = SchemaDriftMonitorAgent.from_default_baseline(
        state, matrix_path=mx, baseline_path=bp, output_dir=tmp_path / "rep"
    )
    out = agent.run(dry_run=True)
    assert out["status"] == "ok"
    assert out["severity"] == "red"
    assert "c_new" in out["columns_added"]


def test_awaiting_when_baseline_absent(tmp_path, state):
    agent = SchemaDriftMonitorAgent.from_default_baseline(
        state, baseline_path=tmp_path / "nope.json"
    )
    assert agent.run(dry_run=True)["status"] == "awaiting_baseline"


def test_env_var_matrix_resolution(tmp_path, state, ref_df, monkeypatch):
    bp = tmp_path / "schema_baseline.json"
    _write_baseline(bp, ref_df)
    mx = tmp_path / "X.parquet"
    ref_df.to_parquet(mx, index=False)
    monkeypatch.setenv("GVC_SCHEMA_CURRENT_MATRIX", str(mx))
    agent = SchemaDriftMonitorAgent.from_default_baseline(
        state, baseline_path=bp, output_dir=tmp_path / "rep"
    )
    assert agent.run(dry_run=True)["status"] == "ok"


def test_bare_constructor_still_awaiting(state):
    assert SchemaDriftMonitorAgent(state).run(dry_run=True)["status"] == "awaiting_baseline"
