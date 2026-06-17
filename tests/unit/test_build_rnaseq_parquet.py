"""tests/unit/test_build_rnaseq_parquet.py -- Monzia Moodie

RNA-seq ingestion (both input forms + optional DE), connector round-trip.
No variant_ensemble import -> runs without xgboost.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_rnaseq_parquet as brp  # noqa: E402
from genomic_variant_classifier.data.rnaseq import (  # noqa: E402
    annotate_rnaseq_from_parquet, RNASEQ_FEATURES,
)


def test_gene_mode_summary(tmp_path):
    m = tmp_path / "g.tsv"
    m.write_text("gene_symbol\tS1\tS2\tS3\tS4\n"
                 "BRCA1\t100\t120\t80\t110\n"
                 "TP53\t0\t0\t50\t0\n"
                 "MLH1\t0.2\t0.5\t0.1\t0.0\n", encoding="utf-8")
    ids, samp = brp._split_id_samples(brp._read_table(m), None)
    gm = samp.copy(); gm.insert(0, "gene_symbol", ids.values); gm = gm.groupby("gene_symbol").sum()
    agg = brp.summarise_matrix(gm, None, None, 1.0)
    r = {x.gene_symbol: x for x in agg.itertuples()}
    assert r["BRCA1"].rnaseq_detection_rate == 1.0
    assert r["TP53"].rnaseq_detection_rate == 0.25
    assert r["MLH1"].rnaseq_detection_rate == 0.0
    assert r["TP53"].rnaseq_log2_cv > r["BRCA1"].rnaseq_log2_cv
    assert (agg["rnaseq_log2fc"] == 0).all() and (agg["rnaseq_de_neglog10p"] == 0).all()


def test_tx_collapse_version_stripped(tmp_path):
    m = tmp_path / "tx.tsv"
    m.write_text("transcript_id\tS1\tS2\nENST1.3\t40\t60\nENST2.1\t60\t40\nENST3\t10\t20\n", encoding="utf-8")
    g = tmp_path / "t2g.tsv"
    g.write_text("transcript_id\tgene_symbol\nENST1\tBRCA1\nENST2\tBRCA1\nENST3\tTP53\n", encoding="utf-8")
    ids, samp = brp._split_id_samples(brp._read_table(m), None)
    gm = brp._collapse_tx_to_gene(ids, samp, g)
    assert list(gm.loc["BRCA1"]) == [100.0, 100.0]
    assert list(gm.loc["TP53"]) == [10.0, 20.0]


def test_de_when_meta_present(tmp_path):
    m = tmp_path / "de.tsv"
    m.write_text("gene_symbol\tc1\tc2\tc3\tk1\tk2\tk3\n"
                 "UP\t1\t2\t1\t100\t120\t110\n"
                 "FLAT\t50\t52\t48\t51\t49\t50\n", encoding="utf-8")
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample_id\tgroup\nc1\tcontrol\nc2\tcontrol\nc3\tcontrol\n"
                    "k1\tcase\nk2\tcase\nk3\tcase\n", encoding="utf-8")
    ids, samp = brp._split_id_samples(brp._read_table(m), None)
    gm = samp.copy(); gm.insert(0, "gene_symbol", ids.values); gm = gm.groupby("gene_symbol").sum()
    case, control = brp._read_sample_meta(meta, list(gm.columns))
    agg = brp.summarise_matrix(gm, case, control, 1.0)
    r = {x.gene_symbol: x for x in agg.itertuples()}
    assert r["UP"].rnaseq_log2fc > 3
    assert r["UP"].rnaseq_de_neglog10p > r["FLAT"].rnaseq_de_neglog10p
    assert abs(r["FLAT"].rnaseq_log2fc) < 0.2


def test_connector_roundtrip_and_stub(tmp_path):
    m = tmp_path / "g.tsv"
    m.write_text("gene_symbol\tS1\tS2\nBRCA1\t10\t20\nTP53\t1\t2\n", encoding="utf-8")
    ids, samp = brp._split_id_samples(brp._read_table(m), None)
    gm = samp.copy(); gm.insert(0, "gene_symbol", ids.values); gm = gm.groupby("gene_symbol").sum()
    out = tmp_path / "r.parquet"; brp.summarise_matrix(gm, None, None, 1.0).to_parquet(out, index=False)
    ann = annotate_rnaseq_from_parquet(pd.DataFrame({"gene_symbol": ["BRCA1", "ZZZ"]}), out)
    assert set(RNASEQ_FEATURES).issubset(ann.columns)
    assert ann.loc[ann.gene_symbol == "ZZZ", "rnaseq_mean_log_tpm"].iloc[0] == 0.0
    # stub on missing path
    ann2 = annotate_rnaseq_from_parquet(pd.DataFrame({"gene_symbol": ["BRCA1"]}), tmp_path / "nope.parquet")
    assert (ann2[RNASEQ_FEATURES].iloc[0] == 0).all()


def test_loud_failures(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("sample_id\tgroup\nc1\tA\nc2\tB\nc3\tC\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        brp._read_sample_meta(bad, ["c1", "c2", "c3"])
    few = tmp_path / "few.tsv"
    few.write_text("sample_id\tgroup\nc1\tcase\nk1\tcontrol\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        brp._read_sample_meta(few, ["c1", "k1"])
