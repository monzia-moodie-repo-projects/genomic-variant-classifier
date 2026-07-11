"""
tests/test_resolve_alleleless_ncbi.py  (2026-07-09)
Offline only -- no network. Run: python -m pytest tests/test_resolve_alleleless_ncbi.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nc = _load("resolve_alleleless_ncbi", "scripts/resolve_alleleless_ncbi.py")


def test_parse_spdi():
    assert nc.parse_spdi("NC_000001.11:12344:A:G") == ("NC_000001.11", 12345, "A", "G")
    assert nc.parse_spdi("bad") is None


def test_acc_to_chrom():
    assert nc._acc_to_chrom("NC_000001.11") == "1"
    assert nc._acc_to_chrom("NC_000023.11") == "X"
    assert nc._acc_to_chrom("NC_012920.1") == "MT"


def test_parse_record_spdi():
    rec = {"variation_set": [{"canonical_spdi": "NC_000010.11:124404663:AT:A",
                              "variation_loc": []}]}
    got = nc.parse_esummary_record(rec)
    assert got["chrom"] == "10" and got["pos"] == 124404664
    assert got["ref"] == "AT" and got["alt"] == "A"


def test_parse_record_no_simple_allele():
    rec = {"variation_set": [{"canonical_spdi": "", "variation_loc":
           [{"assembly_name": "GRCh38", "chr": "1", "start": None}]}]}
    got = nc.parse_esummary_record(rec)
    assert got.get("no_simple_allele") is True


def _fasta(path: Path):
    # chrom '10' pos 124404664..665 = 'AT'; chrom '2' pos 200 = 'A' (NCBI says ref 'G' -> mismatch)
    s10 = list("N" * 124404700)
    s10[124404663] = "A"; s10[124404664] = "T"
    s2 = list("N" * 300); s2[199] = "A"      # genome 'A' vs NCBI ref 'G' -> mismatch (gok False)
    path.write_text(">10\n" + "".join(s10) + "\n>2\n" + "".join(s2) + "\n")


def _recovery_full(path: Path):
    pd.DataFrame({
        "variant_id": ["clinvar:10:124404664:na:na",   # RESOLVED, genome ok
                       "clinvar:2:200:na:na",            # RESOLVED but genome mismatch
                       "clinvar:3:300:na:na",            # CNV -> allele-less
                       "clinvar:4:400:na:na"],           # not found
        "verdict": ["STALE_MISS_TRY_NCBI"] * 4,
        "cohort_varid": ["174", "222", "333", "444"],
    }).to_csv(path, sep="\t", index=False)


def _fixture(path: Path):
    json.dump({
        "174": {"variation_set": [{"canonical_spdi": "NC_000010.11:124404663:AT:A",
                                   "variation_loc": []}]},
        "222": {"variation_set": [{"canonical_spdi": "NC_000002.11:199:G:T",  # genome has N
                                   "variation_loc": []}]},
        "333": {"variation_set": [{"canonical_spdi": "",
                                   "variation_loc": [{"assembly_name": "GRCh38",
                                                      "chr": "3", "start": None}]}]},
        # 444 intentionally absent -> NOT_FOUND
    }, open(path, "w"))


def test_full_offline_resolution(tmp_path):
    rf = tmp_path / "recovery_full.tsv"; _recovery_full(rf)
    fx = tmp_path / "fixture.json"; _fixture(fx)
    fa = tmp_path / "g.fa"; _fasta(fa)
    rc = nc.main([
        "--recovery-full", str(rf), "--fasta", str(fa),
        "--cache", str(tmp_path / "cache.json"),
        "--out", str(tmp_path / "resolved.tsv"),
        "--summary", str(tmp_path / "summary.json"),
        "--_fetch_fixture", str(fx), "--rate", "0",
    ])
    assert rc == 0
    out = pd.read_csv(tmp_path / "resolved.tsv", sep="\t")
    by = dict(zip(out["variant_id"], out["ncbi_verdict"]))
    assert by["clinvar:10:124404664:na:na"] == "RESOLVED_HAS_ALLELE"
    assert by["clinvar:2:200:na:na"] == "RESOLVED_GENOME_MISMATCH"
    assert by["clinvar:3:300:na:na"] == "CONFIRMED_ALLELELESS_NCBI"
    assert by["clinvar:4:400:na:na"] == "NOT_FOUND"

    # the genome-ok resolved row carries the correct allele
    row = out[out["variant_id"] == "clinvar:10:124404664:na:na"].iloc[0]
    assert row["ref"] == "AT" and row["alt"] == "A" and bool(row["genome_ok"])

    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["resolved_with_allele_genome_ok"] == 1


def test_cache_resume(tmp_path):
    """A pre-populated cache entry is used and not re-fetched (idempotent/resumable)."""
    rf = tmp_path / "recovery_full.tsv"; _recovery_full(rf)
    fa = tmp_path / "g.fa"; _fasta(fa)
    cache = tmp_path / "cache.json"
    json.dump({"174": {"verdict": "CONFIRMED_ALLELELESS_NCBI", "spdi": ""}},
              open(cache, "w"))
    # empty fixture -> if it tried to fetch 174 it would become NOT_FOUND; cache must win
    fx = tmp_path / "fx.json"; json.dump({}, open(fx, "w"))
    nc.main(["--recovery-full", str(rf), "--fasta", str(fa), "--cache", str(cache),
             "--out", str(tmp_path / "o.tsv"), "--summary", str(tmp_path / "s.json"),
             "--_fetch_fixture", str(fx), "--rate", "0"])
    out = pd.read_csv(tmp_path / "o.tsv", sep="\t")
    row = out[out["variant_id"] == "clinvar:10:124404664:na:na"].iloc[0]
    assert row["ncbi_verdict"] == "CONFIRMED_ALLELELESS_NCBI"   # from cache, not refetched
