"""
tests/test_cohort_v3_rebuild.py  (2026-07-09)
Covers the shared allele_classify predicates and the v2->v3 rebuild.
Run: python -m pytest tests/test_cohort_v3_rebuild.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# prefer the installed src module; fall back to scripts copy for CI before install
try:
    ac = _load("allele_classify", "src/genomic_variant_classifier/data/allele_classify.py")
except FileNotFoundError:
    ac = _load("allele_classify", "scripts/allele_classify.py")


def _s(vals):
    return pd.Series(vals, dtype="object")


def test_padded_deletion_rejects_empty_and_nan_alleles():
    ref = _s(["ACTT", "AG",  "A",  "ACTT", None,  "ACTT", "AA",   "ACGT", "ACGT"])
    alt = _s(["A",    "A",   "G",  None,   "A",   "",     "C",    "na",   "."])
    got = ac.is_padded_deletion(ref, alt).tolist()
    #        pdel   pdel   snv   NaNalt  eref  ealt   delins  na-tok  dot
    assert got == [True, True, False, False, False, False, False, False, False]


def test_padded_deletion_still_catches_real_deletions():
    ref = _s(["GCTGCTGGACCTGCC", "TGAG", "GTGCC"])
    alt = _s(["G", "T", "G"])
    assert ac.is_padded_deletion(ref, alt).all()


def test_is_allele_less_matches_na_na_across_null_forms():
    ref = _s([None, "na", "NA", "", "ACTT", "A",  ".",  None])
    alt = _s([None, "na", "",   "na", None,  "G",  ".",  "A"])
    got = ac.is_allele_less(ref, alt).tolist()
    #        NN    nana  NA/'' ''/na  ref-only snv  ././  alt-only
    assert got == [True, True, True, True, False, False, True, False]


def test_empty_allele_tokens():
    s = _s(["", "na", "NaN", "none", ".", "A", "ACGT", None])
    assert ac.is_empty_allele(s).tolist() == [True, True, True, True, True, False, False, True]


# ---- rebuild ----
rb = _load("rebuild_cohort_v3", "scripts/rebuild_cohort_v3.py")


def _cohort():
    return pd.DataFrame({
        "variant_id": ["clinvar:7:100:ACTT:A", "clinvar:1:200:A:G",
                       "clinvar:2:300:na:na", "clinvar:3:400:na:na",
                       "clinvar:4:500:GTG:G"],
        "chrom": ["7", "1", "2", "3", "4"],
        "pos": [100, 200, 300, 400, 500],
        "ref": ["ACTT", "A", None, "na", "GTG"],
        "alt": ["A", "G", None, "na", "G"],
        "pathogenicity": ["pathogenic", "benign", "pathogenic", "benign", "pathogenic"],
    })


def test_rebuild_without_exclusion_drops_nothing(tmp_path):
    """Safe default: no --exclude-ids => no rows removed, allele-less rows KEPT."""
    df = _cohort()
    inp = tmp_path / "v2.parquet"; df.to_parquet(inp, index=False)
    out = tmp_path / "v3.parquet"
    rc = rb.main(["--in", str(inp), "--out", str(out),
                  "--quarantine", str(tmp_path / "q.parquet"),
                  "--skip-genome", "--apply"])
    assert rc == 0
    v3 = pd.read_parquet(out)
    assert len(v3) == len(df)                       # nothing dropped
    assert ac.is_allele_less(v3["ref"], v3["alt"]).sum() == 2  # na:na rows still present


def test_rebuild_excludes_only_verified_ids(tmp_path):
    """With a verified exclusion list, ONLY those ids are removed + quarantined."""
    df = _cohort()
    inp = tmp_path / "v2.parquet"; df.to_parquet(inp, index=False)
    out = tmp_path / "v3.parquet"
    quar = tmp_path / "quar.parquet"
    # exclude only ONE of the two na:na rows (the verified one)
    excl = tmp_path / "exclude.txt"
    excl.write_text("clinvar:2:300:na:na\n")
    rc = rb.main(["--in", str(inp), "--out", str(out), "--quarantine", str(quar),
                  "--exclude-ids", str(excl), "--skip-genome", "--apply"])
    assert rc == 0
    v3 = pd.read_parquet(out); q = pd.read_parquet(quar)
    assert len(v3) == 4 and len(q) == 1
    assert set(q["variant_id"]) == {"clinvar:2:300:na:na"}
    # the OTHER na:na row (not on the list) is still kept -- never dropped on assumption
    assert "clinvar:3:400:na:na" in set(v3["variant_id"])


def test_rebuild_rejects_non_alleleless_exclusion(tmp_path):
    """An exclusion id that is NOT allele-less must abort (guards against bad lists)."""
    df = _cohort()
    inp = tmp_path / "v2.parquet"; df.to_parquet(inp, index=False)
    excl = tmp_path / "exclude.txt"
    excl.write_text("clinvar:7:100:ACTT:A\n")   # a real padded deletion, NOT allele-less
    rc = rb.main(["--in", str(inp), "--out", str(tmp_path / "v3.parquet"),
                  "--quarantine", str(tmp_path / "q.parquet"),
                  "--exclude-ids", str(excl), "--skip-genome", "--apply"])
    assert rc == 7


def test_rebuild_refuses_overwrite(tmp_path):
    df = _cohort()
    inp = tmp_path / "v2.parquet"; df.to_parquet(inp, index=False)
    out = tmp_path / "v3.parquet"; out.write_text("exists")
    rc = rb.main(["--in", str(inp), "--out", str(out),
                  "--quarantine", str(tmp_path / "q.parquet"),
                  "--skip-genome", "--apply"])
    assert rc == 5
