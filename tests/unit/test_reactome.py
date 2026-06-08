"""
tests/unit/test_reactome.py
===========================
Unit tests for ReactomeConnector (Phase D, gene-level pathway-count connector).

Coverage (connector layer only):
  1.  Stub mode (no pathway_path) — count = 0
  2.  Empty DataFrame — empty output with column present
  3.  _annotate with a small gene→count lookup (match / no-match / order)
  4.  Missing file on disk → defaults returned
  5.  fetch() round-trip via a real parquet file
  6.  Re-run safety — no _x/_y merge suffix when the column already exists

NOTE (intentional gap): the TABULAR_FEATURES-membership and engineer_features
wiring tests (the analogues of test_dbsnp.py tests 6-8) are deliberately NOT
here yet. reactome_pathway_count is not added to
genomic_variant_classifier.models.variant_ensemble.{engineer_features,
TABULAR_FEATURES} until the lockstep wiring patch lands; adding those assertions
before that change would (correctly) fail. They ship WITH the wiring patch so
both feature builders stay in lockstep.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from genomic_variant_classifier.data.reactome import ReactomeConnector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_variant_df(**overrides) -> pd.DataFrame:
    base = dict(
        variant_id=["clinvar:17:7676154:G:A"],
        chrom=["17"],
        pos=[7676154],
        ref=["G"],
        alt=["A"],
        gene_symbol=["TP53"],
        consequence=["missense_variant"],
        allele_freq=[0.0],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def _make_lookup(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Stub mode
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_zero():
    """No pathway_path → reactome_pathway_count = 0 for all variants."""
    connector = ReactomeConnector(pathway_path=None)
    result = connector.annotate_dataframe(_minimal_variant_df())
    assert "reactome_pathway_count" in result.columns
    assert result["reactome_pathway_count"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_column():
    connector = ReactomeConnector()
    empty = pd.DataFrame(columns=["gene_symbol"])
    result = connector.annotate_dataframe(empty)
    assert "reactome_pathway_count" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _annotate with a known lookup
# ---------------------------------------------------------------------------

def test_annotate_matching_and_missing_genes():
    lookup = _make_lookup([
        {"gene_symbol": "TP53", "reactome_pathway_count": 42},
        {"gene_symbol": "BRCA1", "reactome_pathway_count": 17},
    ])
    connector = ReactomeConnector()
    df = pd.DataFrame({"gene_symbol": ["BRCA1", "TP53", "UNKNOWNGENE"]})
    result = connector._annotate(df, lookup)
    # order preserved, values mapped, unknown -> 0
    assert result["reactome_pathway_count"].tolist() == [17, 42, 0]
    assert str(result["reactome_pathway_count"].dtype).startswith("int")


# ---------------------------------------------------------------------------
# 4. Missing file on disk
# ---------------------------------------------------------------------------

def test_missing_file_returns_zero(tmp_path):
    path = tmp_path / "nonexistent.parquet"
    connector = ReactomeConnector(pathway_path=path)
    with patch.object(connector, "_load_cache", return_value=None):
        result = connector.annotate_dataframe(_minimal_variant_df())
    assert result["reactome_pathway_count"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 5. fetch() round-trip via parquet
# ---------------------------------------------------------------------------

def test_fetch_round_trip(tmp_path):
    lookup = pd.DataFrame({
        "gene_symbol": ["TP53", "BRCA1", "EGFR"],
        "reactome_pathway_count": [42, 17, 8],
    })
    parquet_path = tmp_path / "reactome_gene_pathways.parquet"
    lookup.to_parquet(parquet_path, index=False)

    connector = ReactomeConnector(pathway_path=parquet_path)
    # isolate from any shared on-disk cache so the parquet is actually read
    with patch.object(connector, "_load_cache", return_value=None), \
            patch.object(connector, "_save_cache", return_value=None):
        df = pd.DataFrame({"gene_symbol": ["TP53", "EGFR", "ZZZ"]})
        result = connector.fetch(variant_df=df)
    assert result.loc[0, "reactome_pathway_count"] == 42
    assert result.loc[1, "reactome_pathway_count"] == 8
    assert result.loc[2, "reactome_pathway_count"] == 0   # no match


# ---------------------------------------------------------------------------
# 6. Re-run safety (defensive drop prevents _x/_y suffixes)
# ---------------------------------------------------------------------------

def test_rerun_does_not_create_merge_suffixes():
    lookup = _make_lookup([{"gene_symbol": "TP53", "reactome_pathway_count": 5}])
    connector = ReactomeConnector()
    df = pd.DataFrame({"gene_symbol": ["TP53"]})
    once = connector._annotate(df, lookup)
    twice = connector._annotate(once, lookup)   # column already present
    assert "reactome_pathway_count" in twice.columns
    assert not any(c.endswith(("_x", "_y")) for c in twice.columns)
    assert twice["reactome_pathway_count"].iloc[0] == 5
