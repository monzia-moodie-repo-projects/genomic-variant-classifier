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


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """A throwaway index-cache directory, per test.

    HERMETICITY (2026-07-11). FinnGenConnector builds a full-variant index and CACHES it
    to disk. Until today `_cache_paths()` hard-coded `Path("data/raw/cache")` -- relative
    to the CURRENT WORKING DIRECTORY -- with no way for a caller to redirect it. So these
    tests, whose input TSV was correctly written to tmp_path, still wrote
    finngen_full_index.parquet / .meta.json into the REPOSITORY's data tree on every run.

    That was invisible to `git status` (data/raw/ is gitignored) and it made the suite
    non-idempotent: a later run found the cached index and took a different code path.
    The autouse guard in tests/conftest.py now fails any test that does this.

    The connector gained a `cache_dir` argument -- the injection point it never had, and
    which ESM2Connector (cache_path) and ProteinStructurePipeline (cache_dir) already had.
    Every construction below passes it. If you add a test here, pass it too.
    """
    d = tmp_path / "finngen_cache"
    d.mkdir()
    return d


def test_r12_default_prefix_emits_r12_columns(tiny_tsv, cache_dir):
    out = FinnGenConnector(tsv_path=tiny_tsv, cache_dir=cache_dir).annotate(_cohort())
    for c in R12_COLS:
        assert c in out.columns
    # R12 run must NOT create any r13_ columns
    assert not any(c in out.columns for c in R13_COLS)


def test_r13_prefix_emits_r13_columns_only(tiny_tsv, cache_dir):
    out = FinnGenConnector(
        tsv_path=tiny_tsv, column_prefix="r13_", cache_dir=cache_dir
    ).annotate(_cohort())
    for c in R13_COLS:
        assert c in out.columns
    # R13 run must NOT also create the unprefixed R12 columns
    assert not any(c in out.columns for c in R12_COLS)


def test_r12_and_r13_compute_identical_values(tiny_tsv, cache_dir):
    # Same file, same cohort -> the only difference is column names, not the numbers.
    d12 = FinnGenConnector(tsv_path=tiny_tsv, cache_dir=cache_dir).annotate(_cohort())
    d13 = FinnGenConnector(
        tsv_path=tiny_tsv, column_prefix="r13_", cache_dir=cache_dir
    ).annotate(_cohort())
    for c12, c13 in zip(R12_COLS, R13_COLS):
        assert np.allclose(d12[c12].values, d13[c13].values)


def test_matched_variant_has_correct_af(tiny_tsv, cache_dir):
    out = FinnGenConnector(tsv_path=tiny_tsv, cache_dir=cache_dir).annotate(_cohort())
    # row 0 is 1:13668 G>A -> GENOME_AF_fin 0.003492
    assert abs(out["finngen_af_fin"].iloc[0] - 0.003492) < 1e-9


def test_unmatched_variant_defaults_zero(tiny_tsv, cache_dir):
    out = FinnGenConnector(tsv_path=tiny_tsv, cache_dir=cache_dir).annotate(_cohort())
    # row 2 is 1:99999 (no match) -> 0.0
    assert out["finngen_af_fin"].iloc[2] == 0.0


def test_no_file_branch_zero_fills_per_prefix(cache_dir):
    # No TSV -> annotate() returns before the index is ever built, so nothing is cached.
    # cache_dir is passed anyway, so this test cannot regress into writing to data/raw/cache
    # if that early-return branch is ever changed.
    d12 = FinnGenConnector(tsv_path=None, cache_dir=cache_dir).annotate(_cohort())
    assert (d12["finngen_af_fin"] == 0.0).all()
    assert (d12["finngen_enrichment"] == 1.0).all()
    d13 = FinnGenConnector(
        tsv_path=None, column_prefix="r13_", cache_dir=cache_dir
    ).annotate(_cohort())
    assert (d13["finngen_r13_af_fin"] == 0.0).all()
    assert (d13["finngen_r13_enrichment"] == 1.0).all()


def test_enrichment_is_ratio_clipped(tiny_tsv, cache_dir):
    out = FinnGenConnector(tsv_path=tiny_tsv, cache_dir=cache_dir).annotate(_cohort())
    # 2:50000 -> fin 0.5 / (nfe 0.4 + 1e-9) ~= 1.25
    row = out[(out["chrom"] == "2") & (out["pos"] == 50000)]
    assert abs(row["finngen_enrichment"].iloc[0] - 1.25) < 1e-3
    # all enrichment <= 1000 (clip)
    # RESTORED 2026-07-11: this assertion was accidentally orphaned onto the test below
    # when cache_dir was threaded through -- the edit's match window stopped one line short
    # of the end of this function, so the clip check silently left the test it belongs to.
    # It is the only assertion covering the upper clip; losing it would have been a real
    # coverage regression hidden inside a green suite.
    assert (out["finngen_enrichment"] <= 1000.0).all()


def test_cache_dir_is_honoured_and_nothing_leaks_to_the_repo(tiny_tsv, cache_dir):
    """REGRESSION GUARD (2026-07-11): the index must land where the CALLER said.

    The whole class of defect found today was libraries hard-coding a working-directory-
    relative WRITE path with no override. Four instances: real_data_prep's AlphaMissense
    fallback, ProteinStructurePipeline's alphafold cache, ESM2Connector's default sqlite
    cache, and this one. If _cache_paths() ever stops honouring cache_dir, this fails here
    rather than silently polluting the repository again.
    """
    conn = FinnGenConnector(tsv_path=tiny_tsv, cache_dir=cache_dir)
    conn.annotate(_cohort())

    pq, meta = conn._cache_paths()
    assert pq.parent == cache_dir, f"index cache escaped to {pq.parent}, not {cache_dir}"
    assert meta.parent == cache_dir
    assert pq.exists(), "the index cache should have been written under cache_dir"
    # and it must NOT have been written into the repository's data tree
    assert "data" not in pq.parts or str(cache_dir) in str(pq)
