from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.alphamissense import (
    AM_DEFAULT_SCORE,
    AlphaMissenseConnector,
)

_HEADER = "#CHROM\tPOS\tREF\tALT\tgenome\tuniprot_id\ttranscript_id\tprotein_variant\tam_pathogenicity\tam_class\n"


def _row(chrom, pos, ref, alt, score, am_class="likely_pathogenic"):
    return f"{chrom}\t{pos}\t{ref}\t{alt}\thg38\tP00000\tENST0\tp.X1Y\t{score}\t{am_class}\n"


@pytest.fixture(autouse=True)
def _force_cache_miss(monkeypatch):
    # Independent of disk cache / config: always exercise the parse path.
    monkeypatch.setattr(AlphaMissenseConnector, "_load_cache", lambda self, k: None)


@pytest.fixture()
def source_tsv(tmp_path: Path) -> Path:
    p = tmp_path / "AlphaMissense_hg38.tsv.gz"
    lines = [_HEADER, _row("1", 100, "A", "G", 0.90),
             _row("2", 200, "C", "T", 0.10, "likely_benign"),
             _row("1", 100, "A", "G", 0.40)]  # duplicate key, lower score
    for i in range(5000):
        lines.append(_row("7", 1_000_000 + i, "G", "C", 0.50))
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.writelines(lines)
    return p


@pytest.fixture()
def cohort_df() -> pd.DataFrame:
    return pd.DataFrame({"chrom": ["chr1", "2", "5"], "pos": [100, 200, 999],
                         "ref": ["a", "C", "T"], "alt": ["g", "T", "A"]})


def test_annotate_scores_defaults_and_dedup(source_tsv, cohort_df):
    out = AlphaMissenseConnector(tsv_path=source_tsv).fetch(cohort_df)
    assert len(out) == len(cohort_df)
    s = out["alphamissense_score"].tolist()
    assert s[0] == pytest.approx(0.90, abs=1e-6)   # max of duplicate pair
    assert s[1] == pytest.approx(0.10, abs=1e-6)
    assert s[2] == pytest.approx(AM_DEFAULT_SCORE, abs=1e-6)  # absent -> default


def test_lookup_is_cohort_bounded_not_source_sized(source_tsv, cohort_df):
    conn = AlphaMissenseConnector(tsv_path=source_tsv)
    lookup = conn._parse_tsv(source_tsv, conn._cohort_keys(cohort_df))
    assert len(lookup) == 2
    assert set(lookup["lookup_key"]) == {"1:100:A:G", "2:200:C:T"}
    got = lookup.set_index("lookup_key")["alphamissense_score"]["1:100:A:G"]
    assert got == pytest.approx(0.90, abs=1e-6)


def test_cohort_filtered_build_is_not_cached(source_tsv, cohort_df, monkeypatch):
    conn = AlphaMissenseConnector(tsv_path=source_tsv)
    calls = []
    monkeypatch.setattr(conn, "_save_cache", lambda k, df: calls.append(k))
    conn.fetch(cohort_df)
    assert calls == []


def test_full_build_is_cached_and_complete(source_tsv, monkeypatch):
    conn = AlphaMissenseConnector(tsv_path=source_tsv)
    calls = []
    monkeypatch.setattr(conn, "_save_cache", lambda k, df: calls.append((k, len(df))))
    lookup = conn._get_lookup(cohort_keys=None)
    assert len(lookup) == 2 + 5000
    assert calls and calls[0][0] == "scores_hg38"


def test_empty_cohort_returns_default_column():
    out = AlphaMissenseConnector(tsv_path=None).fetch(
        pd.DataFrame({"chrom": [], "pos": [], "ref": [], "alt": []}))
    assert "alphamissense_score" in out.columns and len(out) == 0


def test_missing_file_returns_default_scores(tmp_path, cohort_df):
    out = AlphaMissenseConnector(tsv_path=tmp_path / "nope.tsv.gz").fetch(cohort_df)
    assert (out["alphamissense_score"] == AM_DEFAULT_SCORE).all()
