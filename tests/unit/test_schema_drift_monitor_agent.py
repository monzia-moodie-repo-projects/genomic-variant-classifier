"""Smoke tests for SchemaDriftMonitorAgent (BaseAgent wrapper)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
import pytest
pa = pytest.importorskip("pandera.pandas")

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import SchemaDriftAgent
from genomic_variant_classifier.agent_layer.agents.schema_drift_monitor_agent import (
    SchemaDriftMonitorAgent,
)


def test_awaiting_baseline_when_unconfigured():
    agent = SchemaDriftMonitorAgent(SharedState())
    r = agent.run(dry_run=True)
    assert r["status"] == "awaiting_baseline"
    assert agent._get_section("schema_drift")["status"] == "awaiting_baseline"


def test_ok_path_with_real_detector(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
    mp = tmp_path / "matrix.parquet"
    df.to_parquet(mp)
    expected_dtypes = {c: str(df[c].dtype) for c in df.columns}
    det = SchemaDriftAgent(
        schema=pa.DataFrameSchema({"a": pa.Column("int64"), "b": pa.Column("float64")}),
        expected_dtypes=expected_dtypes,
        expected_schema_hash=SchemaDriftAgent.hash_schema(expected_dtypes),
        output_dir=tmp_path,
    )
    agent = SchemaDriftMonitorAgent(SharedState(), detector=det, matrix_path=mp)
    r = agent.run(dry_run=True)
    assert r["status"] == "ok"
    assert r["severity"] in {"green", "red"}
