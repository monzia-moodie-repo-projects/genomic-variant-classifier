"""test_build_gnomad_ymt_af.py -- Monzia Moodie

Validates the pure (non-network) logic of build_gnomad_ymt_af against payloads shaped exactly like the
gnomAD GraphQL responses confirmed by probing. v2 adds the throttle/dirty-gene fixes: clean_y_genes
(split ';'-joined multi-gene + drop free-text), aliased batch query build, and batch-payload parse
(null aliases / per-gene 'not found' skipped, never silently lost).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_gnomad_ymt_af as B


# ---- norm_key / AF parsing -------------------------------------------------------------------------
def test_norm_key_y_mt_and_chr_prefix():
    assert B.norm_key("Y-12904862-A-G") == "Y:12904862:A:G"
    assert B.norm_key("M-8602-T-C") == "MT:8602:T:C"          # M -> MT to match cohort
    assert B.norm_key("chrY-100-A-G") == "Y:100:A:G"           # strip chr


def test_parse_mt_af_computes_af_hom_and_skips_zero_an():
    payload = {"data": {"region": {"mitochondrial_variants": [
        {"variant_id": "M-8602-T-C", "an": 56000, "ac_hom": 56, "ac_het": 3},   # 56/56000 = 0.001
        {"variant_id": "M-3308-T-C", "an": 56000, "ac_hom": 0, "ac_het": 9},    # homoplasmic 0
        {"variant_id": "M-9-G-A", "an": 0, "ac_hom": 1},                        # an=0 -> skipped
    ]}}}
    af = B.parse_mt_af(payload)
    assert af["MT:8602:T:C"] == pytest.approx(0.001)
    assert af["MT:3308:T:C"] == 0.0
    assert "MT:9:G:A" not in af


# ---- v2: dirty gene-symbol cleaning ----------------------------------------------------------------
def test_clean_y_genes_splits_multigene_and_drops_freetext():
    raw = [
        "DDX3Y",                                       # clean single
        "AKAP17A;ASMT;ASMTL;P2RY8",                    # semicolon multi-gene -> 4 symbols
        "DDX3Y;LOC108004538;USP9Y;UTY",                # overlaps DDX3Y (dedup) + adds 3
        "-",                                           # dash -> dropped
        "nan",                                         # nan -> dropped
        "covers 10 genes, none of which curated to show dosage sensitivity",  # free-text -> dropped
        "subset of 103 genes: SRY",                    # free-text (has spaces) -> dropped entirely
    ]
    genes = B.clean_y_genes(raw)
    assert genes == sorted(["DDX3Y", "AKAP17A", "ASMT", "ASMTL", "P2RY8", "LOC108004538", "USP9Y", "UTY"])
    # the real clinical Y genes survive; no free-text token leaks through
    assert all(" " not in g for g in genes)
    assert "SRY" not in genes  # only present in free-text here; would be captured elsewhere if clean


def test_clean_y_genes_keeps_hyphenated_real_symbol_but_drops_bare_dash():
    # 'ZFY-AS1' is a real (hyphenated) symbol and must survive; a bare '-' must not.
    genes = B.clean_y_genes(["ZFY;ZFY-AS1", "-"])
    assert genes == ["ZFY", "ZFY-AS1"]


# ---- v2: aliased batch query + parse ----------------------------------------------------------------
def test_build_y_batch_query_aliases_and_embeds_symbols():
    q = B.build_y_batch_query(["DDX3Y", "USP9Y"])
    assert q.startswith("query($ds: DatasetId!) {")
    assert 'a0: gene(gene_symbol: "DDX3Y"' in q
    assert 'a1: gene(gene_symbol: "USP9Y"' in q
    assert "variants(dataset: $ds)" in q


def test_parse_y_batch_skips_null_alias_and_picks_exome_then_genome():
    payload = {"data": {
        "a0": {"variants": [
            {"variant_id": "Y-100-A-G", "exome": {"af": 0.002}, "genome": {"af": 0.5}},  # exome wins
            {"variant_id": "Y-200-C-T", "exome": None, "genome": {"af": 0.01}},          # genome fallback
            {"variant_id": "Y-300-G-A", "exome": {"af": None}, "genome": None},          # null -> skip
        ]},
        "a1": None,  # gene-not-found alias -> skipped, NOT an error
    }}
    af = B.parse_y_batch(payload)
    assert af == {"Y:100:A:G": 0.002, "Y:200:C:T": 0.01}


# ---- frame build / merge / cohort extraction --------------------------------------------------------
def test_build_ymt_frame_only_cohort_keys_and_gnomad_prefix():
    cohort = {"Y:100:A:G", "MT:8602:T:C"}
    y_af = {"Y:100:A:G": 0.002, "Y:999:T:C": 0.3}    # 999 not in cohort -> dropped
    mt_af = {"MT:8602:T:C": 0.001, "MT:5:X:Y": 0.4}   # 5 not in cohort -> dropped
    df = B.build_ymt_frame(cohort, y_af, mt_af)
    assert set(df["variant_id"]) == {"gnomad:Y:100:A:G", "gnomad:MT:8602:T:C"}
    assert list(df.columns) == ["variant_id", "allele_freq"]


def test_merge_into_gnomad_dedups(tmp_path):
    base = pd.DataFrame({"variant_id": ["gnomad:1:5:A:G", "gnomad:2:9:C:T"], "allele_freq": [0.1, 0.2]})
    bp = tmp_path / "base.parquet"; base.to_parquet(bp)
    ymt = pd.DataFrame({"variant_id": ["gnomad:Y:100:A:G", "gnomad:MT:8602:T:C"], "allele_freq": [0.002, 0.001]})
    nb, ny, nc = B.merge_into_gnomad(ymt, str(bp), str(tmp_path / "merged.parquet"))
    assert (nb, ny, nc) == (2, 2, 4)
    out = pd.read_parquet(tmp_path / "merged.parquet")
    assert "gnomad:Y:100:A:G" in set(out["variant_id"]) and len(out) == 4


def test_cohort_ymt_extracts_keys_and_raw_y_symbols(tmp_path):
    c = pd.DataFrame({
        "variant_id": ["clinvar:1:5:A:G", "clinvar:Y:100:A:G", "clinvar:Y:200:C:T", "clinvar:MT:8602:T:C"],
        "gene_symbol": ["BRCA1", "DDX3Y;USP9Y", "covers 10 genes, none curated", "MT-CO1"],
    })
    cp = tmp_path / "c.parquet"; c.to_parquet(cp)
    keys, y_raw = B.cohort_ymt(str(cp))
    assert keys == {"Y:100:A:G", "Y:200:C:T", "MT:8602:T:C"}        # autosome excluded
    # cohort_ymt returns RAW Y gene strings; cleaning happens downstream in clean_y_genes
    assert "DDX3Y;USP9Y" in y_raw
    assert B.clean_y_genes(y_raw) == ["DDX3Y", "USP9Y"]            # free-text dropped, multi-gene split
