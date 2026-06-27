"""First-ever coverage for FinnGenConnector (no prior test_finngen*.py existed).

Stage 1 of the R12+R13 dual-release experiment: verifies the connector's column_prefix
parameterization. R12 (default prefix="") must be byte-identical to the historical behavior;
R13 (prefix="r13_") must emit the finngen_r13_* trio with the same numeric computation.
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.finngen import (
    FinnGenConnector,
    FINNGEN_COLUMNS,
    finngen_columns,
)

R12_COLS = ["finngen_af_fin", "finngen_af_nfsee", "finngen_enrichment"]
R13_COLS = ["finngen_r13_af_fin", "finngen_r13_af_nfsee", "finngen_r13_enrichment"]


@pytest.fixture
def tiny_tsv(tmp_path: Path) -> Path:
    """A 4-variant FinnGen TSV in the real R12/R13 schema (GENOME_AF_fin/GENOME_AF_nfe)."""
    p = tmp_path / "tiny_finngen.tsv.gz"
    rows = [
        ("1", 13668, "G", "A", 0.003492, 0.001),
        ("1", 14506, "G", "A", 0.003548, 0.002),
        ("1", 14521, "C", "T", 0.000291, 0.0005),
        ("2", 50000, "A", "G", 0.5, 0.4),
    ]
    with gzip.open(p, "wt", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["chr", "pos", "ref", "alt", "GENOME_AF_fin", "GENOME_AF_nfe", "filler1"])
        for r in rows:
            w.writerow(list(r) + ["x"])
    return p


def _cohort() -> pd.DataFrame:
    # Includes 2 matches (1:13668, 2:50000), 1 same-chrom non-match (1:99999),
    # 1 off-chrom non-match (3:12345) -> exercises default-fill.
    return pd.DataFrame({
        "chrom": ["1", "1", "1", "2", "3"],
        "pos":   [13668, 14506, 99999, 50000, 12345],
        "ref":   ["G", "G", "C", "A", "T"],
        "alt":   ["A", "A", "T", "G", "C"],
    })


def test_finngen_columns_helper():
    assert finngen_columns("") == R12_COLS
    assert finngen_columns("r13_") == R13_COLS


def test_module_constant_unchanged():
    # Backward-compat: the FINNGEN_COLUMNS constant remains the R12 trio (used by zero-fill callers).
    assert FINNGEN_COLUMNS == R12_COLS


def test_r12_default_prefix_emits_r12_columns(tiny_tsv):
    out = FinnGenConnector(tsv_path=tiny_tsv).annotate(_cohort())
    for c in R12_COLS:
        assert c in out.columns
    # R12 run must NOT create any r13_ columns
    assert not any(c in out.columns for c in R13_COLS)


def test_r13_prefix_emits_r13_columns_only(tiny_tsv):
    out = FinnGenConnector(tsv_path=tiny_tsv, column_prefix="r13_").annotate(_cohort())
    for c in R13_COLS:
        assert c in out.columns
    # R13 run must NOT also create the unprefixed R12 columns
    assert not any(c in out.columns for c in R12_COLS)


def test_r12_and_r13_compute_identical_values(tiny_tsv):
    # Same file, same cohort -> the only difference is column names, not the numbers.
    d12 = FinnGenConnector(tsv_path=tiny_tsv).annotate(_cohort())
    d13 = FinnGenConnector(tsv_path=tiny_tsv, column_prefix="r13_").annotate(_cohort())
    for c12, c13 in zip(R12_COLS, R13_COLS):
        assert np.allclose(d12[c12].values, d13[c13].values)


def test_matched_variant_has_correct_af(tiny_tsv):
    out = FinnGenConnector(tsv_path=tiny_tsv).annotate(_cohort())
    # row 0 is 1:13668 G>A -> GENOME_AF_fin 0.003492
    assert abs(out["finngen_af_fin"].iloc[0] - 0.003492) < 1e-9


def test_unmatched_variant_defaults_zero(tiny_tsv):
    out = FinnGenConnector(tsv_path=tiny_tsv).annotate(_cohort())
    # row 2 is 1:99999 (no match) -> 0.0
    assert out["finngen_af_fin"].iloc[2] == 0.0


def test_no_file_branch_zero_fills_per_prefix():
    d12 = FinnGenConnector(tsv_path=None).annotate(_cohort())
    assert (d12["finngen_af_fin"] == 0.0).all()
    assert (d12["finngen_enrichment"] == 1.0).all()
    d13 = FinnGenConnector(tsv_path=None, column_prefix="r13_").annotate(_cohort())
    assert (d13["finngen_r13_af_fin"] == 0.0).all()
    assert (d13["finngen_r13_enrichment"] == 1.0).all()


def test_enrichment_is_ratio_clipped(tiny_tsv):
    out = FinnGenConnector(tsv_path=tiny_tsv).annotate(_cohort())
    # 2:50000 -> fin 0.5 / (nfe 0.4 + 1e-9) ~= 1.25
    row = out[(out["chrom"] == "2") & (out["pos"] == 50000)]
    assert abs(row["finngen_enrichment"].iloc[0] - 1.25) < 1e-3
    # all enrichment <= 1000 (clip)
    assert (out["finngen_enrichment"] <= 1000.0).all()
