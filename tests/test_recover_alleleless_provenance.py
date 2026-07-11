"""
tests/test_recover_alleleless_provenance.py  (2026-07-09)
Run: python -m pytest tests/test_recover_alleleless_provenance.py -v
"""

from __future__ import annotations

import gzip
import importlib.util
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
rp = _load("recover_alleleless_provenance", "scripts/recover_alleleless_provenance.py")


def _cohort():
    # three na:na rows + one normal SNV
    return pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:2:200:na:na",
                       "clinvar:3:300:na:na", "clinvar:4:400:A:G"],
        "chrom": ["1", "2", "3", "4"],
        "pos": [100, 200, 300, 400],
        "ref": [None, None, None, "A"],
        "alt": [None, None, None, "G"],
        "pathogenicity": ["pathogenic", "benign", "pathogenic", "benign"],
    })


def _write_vcf(path: Path):
    # row at 1:100 IS in the VCF (recoverable); 2:200 and 3:300 are ABSENT
    lines = [
        "##fileformat=VCFv4.1",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        "1\t100\t12345\tACGT\tA\t.\t.\t.",
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_variant_summary(path: Path):
    # 2:200 is a copy number loss (out of scope); 3:300 is a Deletion (sequence-like)
    hdr = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome",
           "Start", "Stop", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    rows = [
        ["555", "copy number loss", "GENEB", "GRCh38", "2", "200", "9000", "na", "na"],
        ["666", "Deletion", "GENEC", "GRCh38", "3", "300", "305", "na", "na"],
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def test_recovery_verdicts(tmp_path):
    coh = tmp_path / "v2.parquet"; _cohort().to_parquet(coh, index=False)
    vcf = tmp_path / "clinvar.vcf.gz"; _write_vcf(vcf)
    vs = tmp_path / "variant_summary.txt.gz"; _write_variant_summary(vs)

    rc = rp.main([
        "--cohort", str(coh), "--clinvar-vcf", str(vcf), "--variant-summary", str(vs),
        "--assembly", "GRCh38",
        "--out-verdict", str(tmp_path / "verdict.tsv"),
        "--out-summary", str(tmp_path / "summary.json"),
        "--out-recoverable", str(tmp_path / "recoverable.tsv"),
    ])
    assert rc == 0
    v = pd.read_csv(tmp_path / "verdict.tsv", sep="\t")
    by = dict(zip(v["variant_id"], v["verdict"]))
    assert by["clinvar:1:100:na:na"] == "RECOVERABLE_FROM_VCF"        # present in VCF
    assert by["clinvar:2:200:na:na"] == "LEGITIMATELY_ALLELELESS"     # absent + CNV type
    assert by["clinvar:3:300:na:na"] == "NEEDS_REVIEW"                # absent + sequence type

    rec = pd.read_csv(tmp_path / "recoverable.tsv", sep="\t")
    assert len(rec) == 1
    assert rec.iloc[0]["ref"] == "ACGT" and rec.iloc[0]["alt"] == "A"
    assert str(rec.iloc[0]["variation_id"]) == "12345"


def test_all_absent_message(tmp_path, capsys):
    coh = tmp_path / "v2.parquet"; _cohort().to_parquet(coh, index=False)
    vcf = tmp_path / "clinvar.vcf.gz"
    with gzip.open(vcf, "wt") as fh:
        fh.write("##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    rp.main(["--cohort", str(coh), "--clinvar-vcf", str(vcf),
             "--out-verdict", str(tmp_path / "v.tsv"),
             "--out-summary", str(tmp_path / "s.json"),
             "--out-recoverable", str(tmp_path / "r.tsv")])
    out = capsys.readouterr().out
    assert "NONE of the na:na rows are in the ClinVar VCF" in out
