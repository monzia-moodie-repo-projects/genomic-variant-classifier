"""Activation tests: a SchemaDriftMonitorAgent driven by a built baseline reports
ok/green (or ok/red on drift) -- i.e. it is active, not awaiting_baseline.
pandera is optional in CI, so this whole module skips where it is absent.
Author: Monzia Moodie.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pa = pytest.importorskip("pandera.pandas")

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import SchemaDriftAgent
from genomic_variant_classifier.agent_layer.agents.schema_drift_monitor_agent import (
    SchemaDriftMonitorAgent,
)


def _write_baseline(tmp_path: Path, df: pd.DataFrame) -> Path:
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    payload = {
        "schema_version": 1,
        "run_label": "test",
        "n_columns": len(dtypes),
        "expected_schema_hash": SchemaDriftAgent.hash_schema(dtypes),
        "expected_dtypes": dtypes,
    }
    p = tmp_path / "schema_baseline.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_activation_ok_on_unchanged(tmp_path: Path):
    # includes an all-NaN degenerate column -- must NOT false-trip (nullable=True)
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [float("nan"), 0.5]})
    df.to_parquet(tmp_path / "matrix.parquet")
    det = SchemaDriftAgent.from_baseline(_write_baseline(tmp_path, df), output_dir=tmp_path)
    agent = SchemaDriftMonitorAgent(SharedState(), detector=det, matrix_path=tmp_path / "matrix.parquet")
    r = agent.run(dry_run=True)
    assert r["status"] == "ok"            # active, not awaiting_baseline
    assert r["severity"] == "green"
    assert r["columns_added"] == [] and r["columns_removed"] == []
    assert r["pandera_violations"] == 0


def test_activation_red_on_drift(tmp_path: Path):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.1, 0.5]})
    det = SchemaDriftAgent.from_baseline(_write_baseline(tmp_path, df), output_dir=tmp_path)
    cur = df.copy()
    cur["NEW"] = 1.0
    cur["a"] = cur["a"].astype("int64")
    cur.to_parquet(tmp_path / "current.parquet")
    agent = SchemaDriftMonitorAgent(SharedState(), detector=det, matrix_path=tmp_path / "current.parquet")
    r = agent.run(dry_run=True)
    assert r["status"] == "ok"            # ran successfully (status ok), drift in severity
    assert r["severity"] == "red"
    assert "NEW" in r["columns_added"]
    assert any(c[0] == "a" for c in r["columns_dtype_changed"])


def test_default_construction_still_awaiting_baseline():
    # the activation path must not change the default (no inputs) contract
    agent = SchemaDriftMonitorAgent(SharedState())
    assert agent.run(dry_run=True)["status"] == "awaiting_baseline"
