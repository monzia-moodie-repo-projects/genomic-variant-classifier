"""test_af_1kg_population.py  --  Monzia Moodie

Resurrection tests for af_1kg_* (INCIDENT_2026-06-13: the five super-population
columns were silently all-zero -- no writer). Exercises fill_population_af on
deliberately messy data: chr-prefix mismatch, NaN/out-of-range AFs, duplicate
keys, null alleles, absent variants, partial population coverage, empty/None
parquet.
"""
import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.thousandgenomes import ThousandGenomesConnector

COLS = ["af_1kg_afr", "af_1kg_eur", "af_1kg_eas", "af_1kg_sas", "af_1kg_amr"]


@pytest.fixture
def messy_parquet(tmp_path):
    p = tmp_path / "kg.parquet"
    pd.DataFrame({
        "variant_id": ["1:100:A:T", "2:200:C:G", "1:100:A:T", "3:300:G:A"],  # dup row1
        "allele_freq": [0.1, 0.2, 0.1, 0.3],
        "AFR_AF": [0.10, 1.7, 0.99, -0.2],   # out-of-range -> clip [0,1]
        "EUR_AF": [0.20, np.nan, 0.88, 0.05],  # NaN -> 0
        "EAS_AF": [0.30, 0.40, 0.30, 0.06],
        "AMR_AF": [0.05, 0.15, 0.05, 0.07],
        # no SAS column -> af_1kg_sas stays 0
    }).to_parquet(p)
    return p


@pytest.fixture
def cohort():
    return pd.DataFrame({
        "chrom": ["1", "chr2", "3", "X", "7"],   # mixed chr prefix
        "pos":   [100, 200, 300, 400, 500],
        "ref":   ["A", "C", "G", None, "T"],     # null allele row 4
        "alt":   ["T", "G", "A", "C", "A"],
        "allele_freq": [np.nan] * 5,
    })


def test_populates_and_is_robust(messy_parquet, cohort):
    out = ThousandGenomesConnector(messy_parquet).fill_population_af(cohort)
    for c in COLS:
        assert c in out.columns
        assert out[c].between(0, 1).all(), f"{c} out of [0,1]"
    assert (out["af_1kg_sas"] == 0).all()            # population absent in parquet
    assert out.iloc[0]["af_1kg_afr"] == 0.10         # matched, dup deduped
    assert out.iloc[1]["af_1kg_afr"] == 1.0          # 1.7 clipped
    assert out.iloc[1]["af_1kg_eur"] == 0.0          # NaN -> 0
    assert out.iloc[2]["af_1kg_afr"] == 0.0          # -0.2 clipped
    assert (out.iloc[3][COLS] == 0).all()            # null ref -> no key match
    assert (out.iloc[4][COLS] == 0).all()            # variant absent from parquet


def test_missing_parquet_yields_zero_columns_not_crash(cohort):
    out = ThousandGenomesConnector(None).fill_population_af(cohort)
    assert all(c in out.columns for c in COLS)
    assert (out[COLS].to_numpy() == 0).all()


def test_nonexistent_path_yields_zero_columns(tmp_path, cohort):
    out = ThousandGenomesConnector(tmp_path / "nope.parquet").fill_population_af(cohort)
    assert (out[COLS].to_numpy() == 0).all()


def test_parquet_without_population_columns(tmp_path, cohort):
    p = tmp_path / "globalonly.parquet"
    pd.DataFrame({"variant_id": ["1:100:A:T"], "allele_freq": [0.1]}).to_parquet(p)
    out = ThousandGenomesConnector(p).fill_population_af(cohort)
    assert (out[COLS].to_numpy() == 0).all()         # no pop cols -> all zero, no crash


def test_empty_cohort(messy_parquet):
    out = ThousandGenomesConnector(messy_parquet).fill_population_af(
        pd.DataFrame(columns=["chrom", "pos", "ref", "alt"]))
    assert all(c in out.columns for c in COLS) and len(out) == 0
