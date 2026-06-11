"""Exit-code contract for the schema preflight gate (scripts/run_schema_drift_check.py).

A gate is only trustworthy if its exit codes are: 0 on a matching schema, 2 on drift,
3 on a usage error. Skips where optional deps (pandera/pyarrow) are absent, matching CI.
Author: Monzia Moodie.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pandera.pandas")
pytest.importorskip("pyarrow")

from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import SchemaDriftAgent

_GATE = Path(__file__).resolve().parents[2] / "scripts" / "run_schema_drift_check.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("run_schema_drift_check", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline(tmp_path: Path, df: pd.DataFrame) -> Path:
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    p = tmp_path / "schema_baseline.json"
    p.write_text(
        json.dumps(
            {"expected_dtypes": dtypes, "expected_schema_hash": SchemaDriftAgent.hash_schema(dtypes)}
        ),
        encoding="utf-8",
    )
    return p


def test_gate_script_present():
    assert _GATE.exists(), f"gate script missing at {_GATE}"


def test_gate_green_exit_0(tmp_path: Path):
    gate = _load_gate()
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [float("nan"), 0.5]})  # incl. all-NaN-safe col
    bp = _baseline(tmp_path, df)
    mp = tmp_path / "m.parquet"
    df.to_parquet(mp)
    rc = gate.main(["--baseline", str(bp), "--matrix", str(mp), "--output-dir", str(tmp_path / "out")])
    assert rc == 0


def test_gate_drift_exit_2(tmp_path: Path):
    gate = _load_gate()
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.1, 0.5]})
    bp = _baseline(tmp_path, df)
    cur = df.copy()
    cur["NEW"] = 1.0
    cur["a"] = cur["a"].astype("int64")
    mp = tmp_path / "cur.parquet"
    cur.to_parquet(mp)
    rc = gate.main(["--baseline", str(bp), "--matrix", str(mp), "--output-dir", str(tmp_path / "out")])
    assert rc == 2


def test_gate_missing_matrix_exit_3(tmp_path: Path):
    gate = _load_gate()
    df = pd.DataFrame({"a": [1.0, 2.0]})
    bp = _baseline(tmp_path, df)
    rc = gate.main(
        ["--baseline", str(bp), "--matrix", str(tmp_path / "nope.parquet"), "--output-dir", str(tmp_path / "out")]
    )
    assert rc == 3
