"""Tests for scripts/verify_gnn_score.py (post-run GNN-score non-degenerate guard).

Guards against a silent reproduction of Run 14, where the GNN block's broad
`except Exception` swallowed a failure and the run completed with gnn_score left
at its 0.0 default. See INCIDENT_2026-04-30 (gene_symbol KeyError) and the
[GNN-TRACE] instrumentation in scripts/run_phase2_eval.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_gnn_score.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_gnn_score", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vgs = _load()


def _write(d: Path, scores) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, sc in zip(vgs.SPLIT_FILES, scores):
        if sc is None:
            continue
        pd.DataFrame(
            {"feat_a": np.arange(len(sc), dtype=float), "gnn_score": sc}
        ).to_parquet(d / name, index=False)


def test_varying_scores_pass(tmp_path):
    _write(tmp_path, ([0, 0.3, 0.7, 0, 0.9], [0.1, 0.5, 0], [0, 0.4, 0.8, 0]))
    ok, _ = vgs.verify(tmp_path)
    assert ok


def test_all_zero_fails(tmp_path):
    _write(tmp_path, ([0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0]))
    ok, msgs = vgs.verify(tmp_path)
    assert not ok
    assert all("FAIL" in m for m in msgs)


def test_missing_column_fails(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in vgs.SPLIT_FILES:
        pd.DataFrame({"feat_a": [1.0, 2.0, 3.0]}).to_parquet(tmp_path / name, index=False)
    ok, _ = vgs.verify(tmp_path)
    assert not ok


def test_missing_file_fails(tmp_path):
    _write(tmp_path, ([0, 0.5, 0.9], [0.2, 0.7], None))
    ok, _ = vgs.verify(tmp_path)
    assert not ok


def test_one_split_degenerate_fails(tmp_path):
    _write(tmp_path, ([0, 0.3, 0.7], [0.1, 0.5], [0, 0, 0, 0]))
    ok, _ = vgs.verify(tmp_path)
    assert not ok
