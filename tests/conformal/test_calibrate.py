"""Tests for end-to-end conformal calibration on a substrate, incl. the non-bypassable gate."""
import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.conformal import calibrate as CAL


def _substrate(n_genes, per_gene, seed, aligned=True):
    rng = np.random.default_rng(seed)
    rows = []
    for gi in range(n_genes):
        g = f"GENE{gi}"
        for j in range(per_gene):
            y = int(rng.random() < 0.15)
            p = float(np.clip(rng.normal(0.8 if y else 0.2, 0.12), 0, 1))
            cons = rng.choice(["missense_variant", "nonsense", "splice_donor_variant"])
            rows.append({"variant_id": f"v{gi}_{j}", "gene_symbol": g, "consequence": cons,
                         "label": y, "ensemble_prob": p})
    df = pd.DataFrame(rows)
    if not aligned:
        df["ensemble_prob"] = rng.permutation(df["ensemble_prob"].values)
    return df


def test_aligned_calibration_hits_target(tmp_path):
    p = tmp_path / "aligned.parquet"
    _substrate(400, 50, 1).to_parquet(p)
    res = CAL.calibrate(p)
    assert res.coverage["marginal_ok"]
    assert res.n_cal > 0 and res.n_eval > 0
    assert abs(res.coverage["marginal_coverage"] - 0.90) < 0.03


def test_broken_join_aborts(tmp_path):
    p = tmp_path / "broken.parquet"
    _substrate(400, 50, 2, aligned=False).to_parquet(p)
    with pytest.raises(CAL.AlignmentError):
        CAL.calibrate(p)


def test_gene_disjoint_split_has_no_overlap(tmp_path):
    p = tmp_path / "gd.parquet"
    _substrate(200, 40, 3).to_parquet(p)
    cfg = CAL.CalibrationConfig()
    df = CAL.load_and_verify(p, cfg)
    genes = df["gene_symbol"].astype(str).values
    mask = CAL._gene_disjoint_mask(genes, cfg.cal_frac, cfg.seed)
    assert not (set(genes[mask]) & set(genes[~mask]))


def test_missing_column_fails_loud(tmp_path):
    p = tmp_path / "nogene.parquet"
    _substrate(50, 20, 4).drop(columns=["gene_symbol"]).to_parquet(p)
    with pytest.raises(ValueError):
        CAL.calibrate(p)


def test_per_stratum_present_when_stratum_col_exists(tmp_path):
    p = tmp_path / "strat.parquet"
    _substrate(300, 40, 5).to_parquet(p)
    res = CAL.calibrate(p)
    assert "per_stratum_coverage" in res.coverage
    assert "group_coverage_any" in res.coverage
