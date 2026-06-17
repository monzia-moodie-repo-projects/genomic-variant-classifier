"""tests/unit/test_build_gtex_parquet.py -- Monzia Moodie

Covers the GTEx bulk-expression ingestion path (RNA expression):
  1. summarise_gct computes gtex_max_tpm / n_tissues_expressed / tissue_specificity
     from a median-TPM GCT, and collapses duplicate gene symbols by max-per-tissue.
  2. those features are IDENTICAL to GTExConnector._summarise_expression (the
     canonical API-path semantics) -- so offline bulk == live API.
  3. annotate_gtex_expression_from_parquet round-trips the built parquet onto a
     cohort (UNKNOWN gene -> 0; non-constant).
  4. .gz GCTs read identically; a non-GCT file fails LOUD (no silent garbage).

No mocks: the real build + real connector function are exercised.
"""
from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_gtex_parquet as bgp  # noqa: E402
from genomic_variant_classifier.data.gtex import (  # noqa: E402
    GTEX_EXPR_MIN_TPM,
    GTExConnector,
    annotate_gtex_expression_from_parquet,
)


def _write_gct(path: Path) -> None:
    path.write_text(
        "#1.2\n"
        "4\t3\n"
        "Name\tDescription\tTissueA\tTissueB\tTissueC\n"
        "ENSG1\tBRCA1\t100.0\t0.0\t0.0\n"
        "ENSG2\tTP53\t10.0\t12.0\t8.0\n"
        "ENSG3\tMLH1\t0.5\t0.2\t0.0\n"
        "ENSG4a\tDUP\t5.0\t0.0\t0.0\n"
        "ENSG4b\tDUP\t0.0\t9.0\t0.0\n",
        encoding="utf-8",
    )


def _expect(vals):
    mx = max(vals)
    mn = sum(vals) / len(vals)
    ne = sum(1 for v in vals if v >= GTEX_EXPR_MIN_TPM)
    sp = round(1.0 - mn / mx, 4) if mx > 0 else 0.0
    return round(mx, 4), ne, sp


def test_summarise_gct_features(tmp_path):
    gct = tmp_path / "median.gct"
    _write_gct(gct)
    agg = bgp.summarise_gct(gct)
    got = {
        r.gene_symbol: (round(r.gtex_max_tpm, 4), int(r.gtex_n_tissues_expressed),
                        r.gtex_tissue_specificity)
        for r in agg.itertuples()
    }
    exp = {
        "BRCA1": _expect([100, 0, 0]),
        "TP53": _expect([10, 12, 8]),
        "MLH1": _expect([0.5, 0.2, 0.0]),
        "DUP": _expect([5, 9, 0]),   # max-per-tissue collapse across the two ENSG ids
    }
    assert got == exp
    assert set(agg.columns) == {
        "gene_symbol", "gtex_max_tpm", "gtex_n_tissues_expressed",
        "gtex_tissue_specificity",
    }


def test_summarise_gct_matches_canonical(tmp_path):
    gct = tmp_path / "median.gct"
    _write_gct(gct)
    agg = bgp.summarise_gct(gct)
    expr_df = pd.DataFrame(
        {"tissueSiteDetailId": ["TissueA", "TissueB", "TissueC"],
         "median": [10.0, 12.0, 8.0]}
    )
    canon = GTExConnector._summarise_expression("TP53", expr_df)
    row = agg[agg.gene_symbol == "TP53"].iloc[0]
    assert abs(canon["gtex_max_tpm"] - row.gtex_max_tpm) < 1e-9
    assert canon["gtex_n_tissues_expressed"] == row.gtex_n_tissues_expressed
    assert abs(canon["gtex_tissue_specificity"] - row.gtex_tissue_specificity) < 1e-9


def test_connector_roundtrip(tmp_path):
    gct = tmp_path / "median.gct"
    _write_gct(gct)
    out = tmp_path / "gtex.parquet"
    bgp.summarise_gct(gct).to_parquet(out, index=False)
    ann = annotate_gtex_expression_from_parquet(
        pd.DataFrame({"gene_symbol": ["BRCA1", "TP53", "MLH1", "DUP", "ZZZ"]}), out
    )
    got = dict(zip(ann.gene_symbol, ann.gtex_max_tpm))
    assert got == {"BRCA1": 100.0, "TP53": 12.0, "MLH1": 0.5, "DUP": 9.0, "ZZZ": 0.0}
    assert ann["gtex_n_tissues_expressed"].dtype.kind == "i"
    assert ann["gtex_max_tpm"].nunique() > 1


def test_missing_parquet_defaults_zero(tmp_path):
    ann = annotate_gtex_expression_from_parquet(
        pd.DataFrame({"gene_symbol": ["BRCA1"]}), tmp_path / "nope.parquet"
    )
    assert ann.loc[0, "gtex_max_tpm"] == 0.0
    assert ann.loc[0, "gtex_n_tissues_expressed"] == 0


def test_gz_identical_and_non_gct_fails_loud(tmp_path):
    gct = tmp_path / "median.gct"
    _write_gct(gct)
    gz = tmp_path / "median.gct.gz"
    with open(gct, "rb") as f, gzip.open(gz, "wb") as g:
        shutil.copyfileobj(f, g)
    assert bgp.summarise_gct(gz).equals(bgp.summarise_gct(gct))

    bad = tmp_path / "bad.txt"
    bad.write_text("not a gct\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        bgp.summarise_gct(bad)
