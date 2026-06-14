"""test_data_readiness_detector.py -- Monzia Moodie
Hermetic tests for the read-only pre-run readiness detector: asset presence (present/missing/empty), feature
health via the col_health library, and the GO / GO_WITH_WARNINGS / NO_GO verdict logic.
"""
import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation import data_readiness_detector as D


def test_check_assets_present_missing_empty(tmp_path):
    good = tmp_path / "a.parquet"; good.write_bytes(b"x" * 10)
    empty = tmp_path / "b.parquet"; empty.write_bytes(b"")
    statuses = {s.path: s for s in D.check_assets(["a.parquet", "b.parquet", "missing.parquet"], root=str(tmp_path))}
    assert statuses["a.parquet"].present and statuses["a.parquet"].size_bytes == 10
    assert not statuses["b.parquet"].present and "EMPTY" in statuses["b.parquet"].detail
    assert not statuses["missing.parquet"].present and statuses["missing.parquet"].detail == "MISSING"


def _df():
    n = 200
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "good_a": rng.standard_normal(n),
        "good_b": rng.integers(0, 5, n),
        "const_col": np.ones(n),                 # CONSTANT -> degenerate
        "allzero_col": np.zeros(n),              # ALL_ZERO -> degenerate
    })


def test_feature_health_summary_counts_degenerate():
    h = D.feature_health_summary(_df())
    assert h["n_cols"] == 4 and h["n_healthy"] == 2 and h["n_degenerate"] == 2
    assert "const_col" in h["degenerate"] and "allzero_col" in h["degenerate"]


def test_verdict_no_go_on_missing_asset():
    assets = [D.AssetStatus("x", False, None, "MISSING")]
    v, reasons = D.readiness_verdict(assets)
    assert v == D.NO_GO and "missing" in reasons[0]


def test_verdict_no_go_on_high_degenerate_fraction():
    assets = [D.AssetStatus("x", True, 10, "present")]
    health = {"n_cols": 10, "n_healthy": 3, "n_degenerate": 7, "degenerate": {f"c{i}": "CONSTANT" for i in range(7)}}
    v, _ = D.readiness_verdict(assets, health)              # 70% degenerate >= 50% -> NO_GO
    assert v == D.NO_GO


def test_verdict_warnings_then_go():
    assets = [D.AssetStatus("x", True, 10, "present")]
    warn = {"n_cols": 10, "n_healthy": 9, "n_degenerate": 1, "degenerate": {"c0": "ALL_ZERO"}}
    assert D.readiness_verdict(assets, warn)[0] == D.GO_WITH_WARNINGS
    ok = {"n_cols": 10, "n_healthy": 10, "n_degenerate": 0, "degenerate": {}}
    assert D.readiness_verdict(assets, ok)[0] == D.GO
    assert D.readiness_verdict(assets, None)[0] == D.GO     # no splits -> still GO (health not evaluated)


def test_analyze_bundles(tmp_path):
    import numpy as np, pandas as pd
    (tmp_path / "a.parquet").write_bytes(b"x" * 10)
    # 1 degenerate of 4 (25% < 50% block) -> GO_WITH_WARNINGS
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g0": rng.standard_normal(100), "g1": rng.standard_normal(100),
                       "g2": rng.integers(0, 5, 100), "dead": np.zeros(100)})
    res = D.analyze(["a.parquet"], root=str(tmp_path), feature_df=df)
    assert res["verdict"] == D.GO_WITH_WARNINGS and res["health"]["n_degenerate"] == 1
    assert any(a.present for a in res["assets"])
