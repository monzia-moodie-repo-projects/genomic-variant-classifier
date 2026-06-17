"""test_run17_audit_persplit.py -- Monzia Moodie

Regression for the Run-17 smoke-log finding: hetero_gnn_score was alive in train but the focal-only
HeteroGNNScorer left val/test at the 0.5 default under gene-disjoint splits. The OLD audit concatenated
splits, so nunique>1 (from train) masked the deadness. The PER-SPLIT audit must FAIL when any FAIL-severity
feature is dead in ANY split, while keeping data-stubs (reactome) and sparse features (lovd) as WARN.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_smoke_feature_population as audit


def _cols(n):
    rng = np.arange(n)
    return {
        "af_log10": np.linspace(-6, -1, n), "gnn_score": np.linspace(.1, .9, n),
        "hetero_gnn_score": np.linspace(.2, .8, n), "reactome_pathway_count": (rng % 7).astype(float),
        "af_1kg_afr": np.where(rng % 3 == 0, 0., .2), "af_1kg_eur": np.where(rng % 3 == 1, 0., .2),
        "af_1kg_eas": np.where(rng % 4 == 0, 0., .2), "af_1kg_sas": np.where(rng % 5 == 0, 0., .2),
        "af_1kg_amr": np.where(rng % 2 == 0, 0., .2), "cadd_phred": np.linspace(0, 35, n),
        "sift_score": np.linspace(0, 1, n), "revel_score": np.linspace(0, 1, n),
        "n_tools_pathogenic": (rng % 5).astype(float), "lovd_variant_class": (rng % 3).astype(float),
    }


def _write(d, per_split_overrides=None):
    d.mkdir(parents=True, exist_ok=True)
    for nm, n in [("X_train.parquet", 60), ("X_val.parquet", 30), ("X_test.parquet", 30)]:
        df = pd.DataFrame(_cols(n))
        if per_split_overrides:
            for col, val in per_split_overrides.get(nm, {}).items():
                df[col] = val
        df.to_parquet(d / nm)


def _run(d):
    old = sys.argv
    sys.argv = ["a", str(d), "--run17"]
    try:
        return audit.main()
    finally:
        sys.argv = old


def test_all_populated_passes(tmp_path):
    d = tmp_path / "s"; _write(d); assert _run(d) == 0


def test_hetero_dead_in_val_test_only_FAILS(tmp_path):
    # THE smoke bug: train alive, val/test constant 0.5 -> must FAIL (was masked by concatenation)
    d = tmp_path / "s"
    _write(d, {"X_val.parquet": {"hetero_gnn_score": 0.5}, "X_test.parquet": {"hetero_gnn_score": 0.5}})
    assert _run(d) == 1


def test_reactome_stub_is_warn_not_fail(tmp_path):
    d = tmp_path / "s"
    _write(d, {k: {"reactome_pathway_count": 0.0} for k in ["X_train.parquet", "X_val.parquet", "X_test.parquet"]})
    assert _run(d) == 0


def test_lovd_sparse_is_warn_not_fail(tmp_path):
    d = tmp_path / "s"
    _write(d, {k: {"lovd_variant_class": 0.0} for k in ["X_train.parquet", "X_val.parquet", "X_test.parquet"]})
    assert _run(d) == 0


def test_reactome_severity_is_warn():
    assert audit.EXPECT_RUN17["reactome_pathway_count"][1] == "warn"


def test_audit_header_surfaces_splits_write_time(tmp_path, capsys):
    # staleness guard: the header must report when the splits were written, so a stale pre-fix
    # directory cannot be silently misread from the verdict (INCIDENT_2026-06-16 follow-up:
    # a pre-fix smoke dir was audited and FAILed correctly, but opaquely).
    d = tmp_path / "s"
    _write(d)
    _run(d)
    out = capsys.readouterr().out
    assert "splits written:" in out
