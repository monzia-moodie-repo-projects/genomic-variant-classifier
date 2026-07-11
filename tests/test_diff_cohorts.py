"""Test battery for diff_cohorts (two-mode parquet diff). Mirrors the sandbox validation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diff_cohorts as D  # noqa: E402
from diffcore import column_equal_series, transition_matrix  # noqa: E402


def _base(n=1000):
    return pd.DataFrame({
        "variant_id": [f"clinvar:1:{i}:A:G" for i in range(n)],
        "chrom": ["1"] * n, "pos": list(range(n)), "ref": ["A"] * n, "alt": ["G"] * n,
        "gene_symbol": ["BRCA1"] * n, "clinical_sig": ["x"] * n,
        "protein_change": [None] * n,
        "metadata": [{"k": i} for i in range(n)],
        "pathogenicity": (["pathogenic"] * 300 + ["benign"] * 300 + ["uncertain"] * 400),
    })


def test_normalized_allele_equivalence():
    sa = pd.Series(["A", None, "na", ".", "-", "", "ACGT", np.nan])
    sb = pd.Series(["A", "NA", ".", "none", "null", "-", "ACGT", None])
    eq = column_equal_series(sa, sb, allele=True)
    assert bool(np.all(eq)), "empty representations must compare equal"
    sc = pd.Series(["A", "ACGT"]); sd = pd.Series(["G", "ACGT"])
    assert list(column_equal_series(sc, sd, allele=True)) == [False, True]


def test_dict_metadata_elementwise():
    m1 = pd.Series([{"a": 1, "b": [1, 2]}, {"x": 1}], dtype=object)
    m2 = pd.Series([{"a": 1, "b": [1, 2]}, {"x": 2}], dtype=object)
    assert list(column_equal_series(m1, m2)) == [True, False]


def test_transition_matrix_reconciles():
    old = pd.Series(["pathogenic"] * 3 + ["benign"] * 2)
    new = pd.Series(["uncertain", "pathogenic", "uncertain", "benign", "benign"])
    cls = ["pathogenic", "likely_pathogenic", "uncertain", "likely_benign", "benign"]
    tm = transition_matrix(old, new, cls)
    assert tm.loc["pathogenic", "uncertain"] == 2
    assert tm.loc["pathogenic", "pathogenic"] == 1
    assert tm.loc["benign", "benign"] == 2
    assert tm.values.sum() == 5


def test_labelfix_clean(tmp_path):
    base = _base()
    raw = base.copy()
    fix = base.copy()
    fix.loc[list(range(120)), "pathogenicity"] = "uncertain"
    raw.to_parquet(tmp_path / "raw.parquet", index=False)
    fix.to_parquet(tmp_path / "fix.parquet", index=False)
    s = D.run_diff(tmp_path / "raw.parquet", tmp_path / "fix.parquet", "labelfix", tmp_path, "lf")
    assert s["added"] == 0 and s["removed"] == 0
    assert s["reclassified"] == 120
    assert s["problems"] == []


def test_labelfix_redflag_caught(tmp_path):
    base = _base()
    raw = base.copy()
    bad = base.copy()
    bad.loc[list(range(120)), "pathogenicity"] = "uncertain"
    bad.loc[500, "pathogenicity"] = "pathogenic"     # benign(row500)->pathogenic: illegal
    bad.loc[700, "clinical_sig"] = "TAMPERED"          # non-path column changed: illegal
    raw.to_parquet(tmp_path / "raw.parquet", index=False)
    bad.to_parquet(tmp_path / "bad.parquet", index=False)
    s = D.run_diff(tmp_path / "raw.parquet", tmp_path / "bad.parquet", "labelfix", tmp_path, "bad")
    assert any("besides pathogenic->uncertain" in p for p in s["problems"])
    assert any("clinical_sig" in p for p in s["problems"])


def test_snapshot_setdiff_recovery(tmp_path):
    base = _base()
    A = base.iloc[:800].copy()
    B = base.iloc[200:].copy().reset_index(drop=True)
    shared_ids = list(set(A["variant_id"]) & set(B["variant_id"]))[:50]
    B.loc[B["variant_id"].isin(shared_ids), "pathogenicity"] = "likely_pathogenic"
    A.to_parquet(tmp_path / "A.parquet", index=False)
    B.to_parquet(tmp_path / "B.parquet", index=False)
    s = D.run_diff(tmp_path / "A.parquet", tmp_path / "B.parquet", "snapshot", tmp_path, "snap")
    assert s["removed"] == 200 and s["added"] == 200 and s["shared"] == 600
    assert s["reclassified"] == 50
    assert s["added"] + s["shared"] == len(B)
    assert s["removed"] + s["shared"] == len(A)


def test_determinism(tmp_path):
    base = _base()
    A = base.iloc[:800].copy()
    B = base.iloc[200:].copy().reset_index(drop=True)
    A.to_parquet(tmp_path / "A.parquet", index=False)
    B.to_parquet(tmp_path / "B.parquet", index=False)
    s1 = D.run_diff(tmp_path / "A.parquet", tmp_path / "B.parquet", "snapshot", tmp_path, "d1")
    s2 = D.run_diff(tmp_path / "A.parquet", tmp_path / "B.parquet", "snapshot", tmp_path, "d2")
    for k in ("added", "removed", "shared", "reclassified", "value_diff_counts", "coord_changed"):
        assert s1[k] == s2[k]


# ---- rebuild tests: composite-key alignment, na:na anomaly guard, dup-policy, cross-check ----
def _base_src(n=1000):
    return pd.DataFrame({
        "variant_id": [f"v{i}" for i in range(n)],
        "chrom": ["1"] * n, "pos": list(range(n)), "ref": ["A"] * n, "alt": ["G"] * n,
        "gene_symbol": ["BRCA1"] * n, "clinical_sig": ["x"] * n, "protein_change": [None] * n,
        "metadata": [{"k": i} for i in range(n)], "source_id": [f"s{i}" for i in range(n)],
        "pathogenicity": (["pathogenic"] * 300 + ["benign"] * 300 + ["uncertain"] * 400),
    })


def test_labelfix_nana_dup_transitions_counted(tmp_path):
    """A duplicated na:na variant_id whose rows transition must be counted on the FULL frame."""
    raw = _base_src()
    dup = pd.DataFrame({
        "variant_id": ["vDUP"] * 3, "chrom": ["X"] * 3, "pos": [0, 0, 0],
        "ref": ["na"] * 3, "alt": ["na"] * 3, "gene_symbol": ["DMD"] * 3,
        "clinical_sig": ["p", "p2", "p3"], "protein_change": [None] * 3,
        "metadata": [{"k": 0}] * 3, "source_id": ["sA", "sB", "sC"],
        "pathogenicity": ["pathogenic"] * 3,
    })
    raw = pd.concat([raw, dup], ignore_index=True)
    fix = raw.copy()
    fix.loc[fix["variant_id"] == "vDUP", "pathogenicity"] = "uncertain"
    fix.loc[list(range(117)), "pathogenicity"] = "uncertain"
    raw.to_parquet(tmp_path / "raw.parquet", index=False)
    fix.to_parquet(tmp_path / "fix.parquet", index=False)
    s = D.run_diff(tmp_path / "raw.parquet", tmp_path / "fix.parquet", "labelfix", tmp_path, "t",
                   dup_policy="report")
    assert s["reclassified"] == 120       # 3 dup + 117 unique, full frame (not collapsed to 118)
    assert s["problems"] == []


def test_labelfix_clean_allele_dup_flagged(tmp_path):
    raw = _base_src()
    cleandup = pd.DataFrame({
        "variant_id": ["v5"], "chrom": ["1"], "pos": [5], "ref": ["A"], "alt": ["T"],
        "gene_symbol": ["BRCA1"], "clinical_sig": ["x"], "protein_change": [None],
        "metadata": [{"k": 5}], "source_id": ["sX"], "pathogenicity": ["pathogenic"],
    })
    raw = pd.concat([raw, cleandup], ignore_index=True)
    fix = raw.copy()
    raw.to_parquet(tmp_path / "raw.parquet", index=False)
    fix.to_parquet(tmp_path / "fix.parquet", index=False)
    s = D.run_diff(tmp_path / "raw.parquet", tmp_path / "fix.parquet", "labelfix", tmp_path, "t",
                   dup_policy="report")
    assert any("CLEAN alleles" in p for p in s["problems"])


def test_dup_policy_strict_vs_report(tmp_path):
    raw = _base_src()
    nd = pd.DataFrame({
        "variant_id": ["vX", "vX"], "chrom": ["1", "1"], "pos": [0, 0],
        "ref": ["na", "na"], "alt": ["na", "na"], "gene_symbol": ["G", "G"],
        "clinical_sig": ["a", "b"], "protein_change": [None, None], "metadata": [{}, {}],
        "source_id": ["sDUP", "sDUP"], "pathogenicity": ["pathogenic", "pathogenic"],
    })
    raw = pd.concat([raw, nd], ignore_index=True)
    fix = raw.copy()
    raw.to_parquet(tmp_path / "raw.parquet", index=False)
    fix.to_parquet(tmp_path / "fix.parquet", index=False)
    s_strict = D.run_diff(tmp_path / "raw.parquet", tmp_path / "fix.parquet", "labelfix",
                          tmp_path, "s", dup_policy="strict")
    s_report = D.run_diff(tmp_path / "raw.parquet", tmp_path / "fix.parquet", "labelfix",
                          tmp_path, "r", dup_policy="report")
    assert any("strict dup-policy" in p for p in s_strict["problems"])
    assert not any("strict dup-policy" in p for p in s_report["problems"])
