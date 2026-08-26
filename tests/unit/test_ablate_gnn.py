"""Unit tests for the GNN ablation harness reducer (scripts/ablate_gnn.py).

summarize() is a pure metric reducer -- it must be correct without a GPU because
its JSON rows feed the per-model ML-comparison docs and the gat-vs-gps sweep.
Guarded on torch_geometric because importing ablate_gnn pulls in models.gnn."""
import math

import pytest

pytest.importorskip("torch_geometric")
import ablate_gnn  # scripts/ is on sys.path via tests/conftest.py


def _hist():
    return [
        {"epoch": 1, "train_loss": 0.6, "val_loss": 0.5, "val_auc": 0.80},
        {"epoch": 2, "train_loss": 0.4, "val_loss": 0.45, "val_auc": 0.86},
    ]


def test_summarize_core_metrics():
    row = ablate_gnn.summarize(
        "gat_baseline", _hist(), gnn_std=0.0321,
        peak_vram_mb=1234.56, wall_s=60.0, n_rows=8000, device="cuda",
    )
    assert row["tag"] == "gat_baseline"
    assert row["epochs"] == 2
    assert row["rows"] == 8000
    assert row["device"] == "cuda"
    assert row["s_per_epoch"] == 30.0          # 60 / 2
    assert row["best_val_auc"] == 0.86         # max(0.80, 0.86)
    assert row["final_train_loss"] == 0.4
    assert row["gnn_score_std"] == 0.0321
    assert row["peak_vram_mb"] == 1234.6       # 1 dp
    assert row["all_finite"] is True


def test_summarize_empty_history_is_safe():
    row = ablate_gnn.summarize("empty", [], 0.0, 0.0, 0.0, 0, "cpu")
    assert row["epochs"] == 0
    assert row["s_per_epoch"] == 0.0           # wall 0 / max(0,1) -> no ZeroDivision
    assert math.isnan(row["best_val_auc"])     # default=nan
    assert row["final_train_loss"] is None
    assert row["all_finite"] is True           # all() of empty iterable is True


def test_summarize_flags_nonfinite():
    bad = [{"epoch": 1, "train_loss": float("nan"), "val_loss": 0.5, "val_auc": 0.8}]
    row = ablate_gnn.summarize("bad", bad, 0.0, 0.0, 10.0, 100, "cpu")
    assert row["all_finite"] is False
