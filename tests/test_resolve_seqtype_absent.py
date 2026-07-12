"""
tests/test_resolve_seqtype_absent.py  (2026-07-09)
Run: python -m pytest tests/test_resolve_seqtype_absent.py -v
"""

from __future__ import annotations

import gzip
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


try:
    _load("allele_classify", "src/genomic_variant_classifier/data/allele_classify.py")
except FileNotFoundError:
    _load("allele_classify", "scripts/allele_classify.py")
rs = _load("resolve_seqtype_absent", "scripts/resolve_seqtype_absent.py")


def _fasta(path: Path):
    # chrom '1': pos 120..121 = 'CG' (widened-probe target, 20 away from cohort pos 100)
    # chrom '3': irrelevant (absent-from-vcf case)
    seq = list("N" * 300)
    seq[119] = "C"; seq[120] = "G"           # pos120=C,121=G -> ref 'CG'
    path.write_text(">1\n" + "".join(seq) + "\n")


def _cohort():
    return pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na",   # recoverable only via widened probe (VCF at 120)
                       "clinvar:3:300:na:na",    # CNV, absent from VCF -> CONFIRMED_ALLELELESS
                       "clinvar:5:500:na:na"],   # SNV-typed, absent everywhere -> STILL_UNRESOLVED
        "chrom": ["1", "3", "5"],
        "pos": [100, 300, 500],
        "ref": [None, None, None],
        "alt": [None, None, None],
        "pathogenicity": ["pathogenic", "benign", "pathogenic"],
    })


def _raw_vcf(path: Path):
    lines = [
        "##fileformat=VCFv4.1",
        "##fileDate=2024-01-01",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        "1\t120\t111\tCG\tC\t.\t.\t.",   # 20 bases from cohort pos 100 -> needs --win>=20
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _vs(path: Path):
    hdr = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome", "Start", "Stop"]
    rows = [
        ["111", "Deletion", "GENEA", "GRCh38", "1", "100", "101"],
        ["333", "copy number loss", "GENEB", "GRCh38", "3", "300", "9000"],
        ["555", "single nucleotide variant", "GENEC", "GRCh38", "5", "500", "500"],
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def _structural(path: Path):
    pd.DataFrame({"variant_id": ["clinvar:3:300:na:na"]}).to_parquet(path, index=False)


def test_widened_probe_and_buckets(tmp_path):
    coh = tmp_path / "v2.parquet"; _cohort().to_parquet(coh, index=False)
    raw = tmp_path / "raw.vcf.gz"; _raw_vcf(raw)
    vs = tmp_path / "vs.txt.gz"; _vs(vs)
    fa = tmp_path / "g.fa"; _fasta(fa)
    st = tmp_path / "structural.parquet"; _structural(st)

    rc = rs.main(["--cohort", str(coh), "--raw-vcf", str(raw), "--variant-summary", str(vs),
                  "--fasta", str(fa), "--structural", str(st), "--assembly", "GRCh38",
                  "--win", "25", "--outdir", str(tmp_path / "out")])
    assert rc == 0
    disp = pd.read_csv(tmp_path / "out" / "alleleless_final_disposition.tsv", sep="\t")
    by = dict(zip(disp["variant_id"], disp["bucket"]))
    assert by["clinvar:1:100:na:na"] == "RECOVER"                 # found via widened probe
    assert by["clinvar:3:300:na:na"] == "CONFIRMED_ALLELELESS"    # CNV, absent
    assert by["clinvar:5:500:na:na"] == "STILL_UNRESOLVED"        # SNV-typed, absent everywhere

    # recovered allele is the genome-verified CG>C
    rec = disp[disp["variant_id"] == "clinvar:1:100:na:na"].iloc[0]
    assert rec["ref"] == "CG" and rec["alt"] == "C" and bool(rec["genome_verified"])

    summ = json.loads((tmp_path / "out" / "alleleless_final_disposition_summary.json").read_text())
    assert summ["raw_vcf_filedate"] == "2024-01-01"
    assert summ["also_in_structural_parquet"] == 1
    assert summ["by_bucket"]["RECOVER"] == 1


def test_narrow_win_misses_far_variant(tmp_path):
    """With --win 5 the variant 20 bases away is NOT found -> STILL_UNRESOLVED, proving the
    window actually governs the probe (no accidental global match)."""
    coh = tmp_path / "v2.parquet"; _cohort().to_parquet(coh, index=False)
    raw = tmp_path / "raw.vcf.gz"; _raw_vcf(raw)
    vs = tmp_path / "vs.txt.gz"; _vs(vs)
    fa = tmp_path / "g.fa"; _fasta(fa)
    rs.main(["--cohort", str(coh), "--raw-vcf", str(raw), "--variant-summary", str(vs),
             "--fasta", str(fa), "--structural", str(tmp_path / "none.parquet"),
             "--win", "5", "--outdir", str(tmp_path / "out")])
    disp = pd.read_csv(tmp_path / "out" / "alleleless_final_disposition.tsv", sep="\t")
    by = dict(zip(disp["variant_id"], disp["bucket"]))
    assert by["clinvar:1:100:na:na"] == "STILL_UNRESOLVED"   # 20 > win 5, not found
