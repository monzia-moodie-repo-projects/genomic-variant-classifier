"""
tests/test_diagnose_and_recover_alleleless.py  (2026-07-09)
Run: python -m pytest tests/test_diagnose_and_recover_alleleless.py -v
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
dr = _load("diagnose_and_recover_alleleless", "scripts/diagnose_and_recover_alleleless.py")


def _fasta(path: Path):
    # chrom '1': 1-based pos 100 -> index 99.
    # pos 100..103 = 'ACGT' (for the 1:100 ACGT>A recovery).
    # pos 151..152 = 'CG'   (for the shifted 1:150 -> VCF 151 CG>C recovery).
    seq = list("N" * 200)
    seq[99] = "A"; seq[100] = "C"; seq[101] = "G"; seq[102] = "T"   # pos100=A,101=C,102=G,103=T
    seq[150] = "C"; seq[151] = "G"                                   # pos151=C,152=G -> 'CG'
    # chrom '2': pos 200 = 'T' (so ref 'GGGG' mismatches -> quarantine)
    seq2 = list("N" * 300); seq2[199] = "T"
    path.write_text(">1\n" + "".join(seq) + "\n>2\n" + "".join(seq2) + "\n")


def _cohort():
    return pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na",     # recoverable, ref matches genome (A...)
                       "clinvar:1:150:na:na",      # padded-deletion shift: VCF row at 151
                       "clinvar:2:200:na:na",      # in VCF but ref does NOT match genome
                       "clinvar:3:300:na:na"],     # absent from VCF, CNV type
        "chrom": ["1", "1", "2", "3"],
        "pos": [100, 150, 200, 300],
        "ref": [None, None, None, None],
        "alt": [None, None, None, None],
        "pathogenicity": ["pathogenic", "pathogenic", "benign", "pathogenic"],
    })


def _vcf(path: Path):
    lines = [
        "##fileformat=VCFv4.1",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        "1\t100\t111\tACGT\tA\t.\t.\t.",      # matches genome at pos100 (ACGT) -> recover
        "1\t151\t222\tCG\tC\t.\t.\t.",        # padded-deletion shift; cohort pos 150 -> VCF 151
        "2\t200\t333\tGGGG\tG\t.\t.\t.",      # genome at pos200 is 'T', ref 'GGGG' mismatch -> quarantine
        "1\t100\t444\tA\t.\t.\t.\t.",         # present-but-null ALT '.' -> inspect
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _vs(path: Path):
    hdr = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome", "Start", "Stop"]
    rows = [["999", "copy number loss", "GENED", "GRCh38", "3", "300", "9000"]]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def test_recover_verify_quarantine_and_shift(tmp_path):
    coh = tmp_path / "v2.parquet"; _cohort().to_parquet(coh, index=False)
    vcf = tmp_path / "clinvar.vcf.gz"; _vcf(vcf)
    vs = tmp_path / "variant_summary.txt.gz"; _vs(vs)
    fa = tmp_path / "g.fa"; _fasta(fa)

    rc = dr.main([
        "--cohort", str(coh), "--clinvar-vcf", str(vcf), "--variant-summary", str(vs),
        "--fasta", str(fa), "--assembly", "GRCh38", "--outdir", str(tmp_path / "out"),
    ])
    assert rc == 0
    o = tmp_path / "out"
    recovered = pd.read_csv(o / "alleleless_recovered_verified.tsv", sep="\t")
    quar = pd.read_csv(o / "alleleless_recovery_quarantine.tsv", sep="\t")

    got = dict(zip(recovered["variant_id"], zip(recovered["ref"], recovered["alt"])))
    # 1:100 recovered with genome-verified ref ACGT>A
    assert got["clinvar:1:100:na:na"] == ("ACGT", "A")
    # 1:150 recovered from the SHIFTED VCF row at 151 (CG>C)
    assert got["clinvar:1:150:na:na"] == ("CG", "C")
    # 2:200 ref mismatches genome -> quarantined, NOT recovered
    assert "clinvar:2:200:na:na" in set(quar["variant_id"])
    assert "clinvar:2:200:na:na" not in set(recovered["variant_id"])

    # present-but-null inspect captured the '.' ALT row
    null = pd.read_csv(o / "alleleless_null_allele_inspect.tsv", sep="\t")
    assert "clinvar:1:100:na:na" in set(null["variant_id"])

    # reclassification tagged the CNV row out-of-scope
    rc2 = pd.read_csv(o / "alleleless_needsreview_reclassified.tsv", sep="\t")
    by = dict(zip(rc2["variant_id"], rc2["verdict2"]))
    assert by["clinvar:3:300:na:na"] == "LEGITIMATELY_ALLELELESS"


def test_zero_quarantine_does_not_crash(tmp_path):
    """Regression: when ALL recoverable rows genome-verify (0 quarantined), the empty
    quarantine DataFrame must still carry columns and not raise KeyError. This is the
    exact case the real 2026-07-09 run hit (2,377 recovered, 0 quarantined)."""
    # cohort where every recoverable row's VCF ref matches the genome
    coh = pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na"],
        "chrom": ["1"], "pos": [100], "ref": [None], "alt": [None],
        "pathogenicity": ["pathogenic"],
    })
    cohp = tmp_path / "v2.parquet"; coh.to_parquet(cohp, index=False)
    vcf = tmp_path / "clinvar.vcf.gz"
    with gzip.open(vcf, "wt") as fh:
        fh.write("##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write("1\t100\t111\tACGT\tA\t.\t.\t.\n")   # matches genome -> recovered, 0 quarantine
    fa = tmp_path / "g.fa"; _fasta(fa)
    rc = dr.main(["--cohort", str(cohp), "--clinvar-vcf", str(vcf), "--fasta", str(fa),
                  "--outdir", str(tmp_path / "out")])
    assert rc == 0
    o = tmp_path / "out"
    recovered = pd.read_csv(o / "alleleless_recovered_verified.tsv", sep="\t")
    quar = pd.read_csv(o / "alleleless_recovery_quarantine.tsv", sep="\t")
    assert len(recovered) == 1 and len(quar) == 0
    assert "variant_id" in quar.columns          # empty frame still has the column


def test_pos_shift_recorded(tmp_path):
    coh = tmp_path / "v2.parquet"; _cohort().to_parquet(coh, index=False)
    vcf = tmp_path / "clinvar.vcf.gz"; _vcf(vcf)
    fa = tmp_path / "g.fa"; _fasta(fa)
    dr.main(["--cohort", str(coh), "--clinvar-vcf", str(vcf), "--fasta", str(fa),
             "--outdir", str(tmp_path / "out")])
    import json
    d = json.loads((tmp_path / "out" / "alleleless_patch_miss_diagnosis.json").read_text())
    assert d["of_recovered_pos_shifted"] >= 1     # the 1:150 -> 151 shift was recorded
