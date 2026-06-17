"""tests/unit/test_build_reactome_parquet.py -- Monzia Moodie

Covers the --gmt activation path of scripts/build_reactome_parquet.py:
  1. build_from_gmt counts DISTINCT pathways per gene from a GMT, and writes the
     exact (gene_symbol, reactome_pathway_count) schema ReactomeConnector reads.
  2. the built parquet round-trips through the REAL ReactomeConnector and yields
     the expected per-gene counts (UNKNOWN genes -> 0).
  3. a gene listed twice within one pathway is counted once (distinct dedup).

No mocks: the real build_from_gmt and the real ReactomeConnector are exercised,
so the feature-activation contract is verified rather than assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_reactome_parquet as brp  # noqa: E402
from genomic_variant_classifier.data.database_connectors import FetchConfig  # noqa: E402
from genomic_variant_classifier.data.reactome import ReactomeConnector  # noqa: E402


def _write_gmt(path: Path) -> None:
    # GMT line: set_name <TAB> description <TAB> gene1 <TAB> gene2 ...
    # BRCA1 in 3 pathways, TP53 in 2, MLH1 in 1.
    lines = [
        "Pathway A\tR-HSA-1\tBRCA1\tTP53\tMLH1",
        "Pathway B\tR-HSA-2\tBRCA1\tTP53",
        "Pathway C\tR-HSA-3\tBRCA1",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_from_gmt_counts_distinct_pathways(tmp_path):
    gmt = tmp_path / "ReactomePathways.gmt"
    _write_gmt(gmt)
    out = tmp_path / "reactome_gene_pathways.parquet"
    agg = brp.build_from_gmt(gmt, out)

    counts = dict(zip(agg["gene_symbol"], agg["reactome_pathway_count"]))
    assert counts == {"BRCA1": 3, "TP53": 2, "MLH1": 1}
    assert out.exists()
    # schema is EXACTLY what ReactomeConnector reads
    cols = set(pd.read_parquet(out).columns)
    assert {"gene_symbol", "reactome_pathway_count"} <= cols
    assert str(agg["reactome_pathway_count"].dtype).startswith("int")


def test_built_parquet_read_by_connector(tmp_path):
    gmt = tmp_path / "ReactomePathways.gmt"
    _write_gmt(gmt)
    out = tmp_path / "reactome_gene_pathways.parquet"
    brp.build_from_gmt(gmt, out)

    # isolate the connector cache to tmp_path so the real data/raw/cache is untouched
    cfg = FetchConfig(cache_dir=tmp_path / "cache")
    conn = ReactomeConnector(pathway_path=out, config=cfg)
    df = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3", "v4"],
            "gene_symbol": ["BRCA1", "TP53", "MLH1", "UNKNOWN_GENE"],
        }
    )
    ann = conn.annotate_dataframe(df)
    got = dict(zip(ann["gene_symbol"], ann["reactome_pathway_count"]))
    assert got == {"BRCA1": 3, "TP53": 2, "MLH1": 1, "UNKNOWN_GENE": 0}
    # the feature is genuinely non-constant (would survive the run17 audit)
    assert ann["reactome_pathway_count"].nunique() > 1


def test_duplicate_gene_within_pathway_counted_once(tmp_path):
    gmt = tmp_path / "dup.gmt"
    # BRCA1 appears twice in Pathway A -> Pathway A must count once for BRCA1.
    gmt.write_text(
        "Pathway A\tR-HSA-1\tBRCA1\tBRCA1\tTP53\n"
        "Pathway B\tR-HSA-2\tBRCA1\n",
        encoding="utf-8",
    )
    out = tmp_path / "o.parquet"
    agg = brp.build_from_gmt(gmt, out)
    counts = dict(zip(agg["gene_symbol"], agg["reactome_pathway_count"]))
    assert counts["BRCA1"] == 2  # 2 distinct pathways, not 3
    assert counts["TP53"] == 1
