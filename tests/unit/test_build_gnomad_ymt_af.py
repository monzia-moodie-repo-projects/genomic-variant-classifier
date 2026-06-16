"""test_build_gnomad_ymt_af.py -- Monzia Moodie

Validates the pure (non-network) logic of build_gnomad_ymt_af against payloads shaped exactly like the
gnomAD GraphQL responses confirmed by probing: Y 'Y-pos-ref-alt' with exome/genome af; MT 'M-pos-ref-alt'
with an/ac_hom (af_hom computed = ac_hom/an).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_gnomad_ymt_af as B


def test_norm_key_y_mt_and_chr_prefix():
    assert B.norm_key("Y-12904862-A-G") == "Y:12904862:A:G"
    assert B.norm_key("M-8602-T-C") == "MT:8602:T:C"          # M -> MT to match cohort
    assert B.norm_key("chrY-100-A-G") == "Y:100:A:G"           # strip chr


def test_parse_y_af_prefers_exome_then_genome_skips_null():
    payload = {"data": {"gene": {"variants": [
        {"variant_id": "Y-100-A-G", "exome": {"af": 0.002}, "genome": {"af": 0.5}},   # exome wins
        {"variant_id": "Y-200-C-T", "exome": None, "genome": {"af": 0.01}},           # genome fallback
        {"variant_id": "Y-300-G-A", "exome": {"af": None}, "genome": None},           # null -> skipped
    ]}}}
    af = B.parse_y_af(payload)
    assert af == {"Y:100:A:G": 0.002, "Y:200:C:T": 0.01}


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


def test_build_ymt_frame_only_cohort_keys_and_gnomad_prefix():
    cohort = {"Y:100:A:G", "MT:8602:T:C"}          # MT:5:X:Y NOT in cohort
    y_af = {"Y:100:A:G": 0.002, "Y:999:T:C": 0.3}   # 999 not in cohort -> dropped
    mt_af = {"MT:8602:T:C": 0.001, "MT:5:X:Y": 0.4}  # 5 not in cohort -> dropped
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


def test_cohort_ymt_extracts_keys_and_y_genes(tmp_path):
    c = pd.DataFrame({
        "variant_id": ["clinvar:1:5:A:G", "clinvar:Y:100:A:G", "clinvar:Y:200:C:T", "clinvar:MT:8602:T:C"],
        "gene_symbol": ["BRCA1", "DDX3Y", "USP9Y", "MT-CO1"],
    })
    cp = tmp_path / "c.parquet"; c.to_parquet(cp)
    keys, y_genes = B.cohort_ymt(str(cp))
    assert keys == {"Y:100:A:G", "Y:200:C:T", "MT:8602:T:C"}   # autosome excluded
    assert y_genes == ["DDX3Y", "USP9Y"]                        # MT gene NOT in Y-gene list
