"""
tests/test_harden_recovery_identity.py  (2026-07-09)
Run: python -m pytest tests/test_harden_recovery_identity.py -v
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


hr = _load("harden_recovery_identity", "scripts/harden_recovery_identity.py")


def _cohort(path):
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na",   # pos+1 recovery -> exempt
                       "clinvar:2:200:na:na",    # wide, IDENTITY-1 match -> accept
                       "clinvar:3:300:na:na",    # wide, no match -> rebucket
                       "clinvar:4:400:na:na"],   # wide, IDENTITY-2 (gene+clnsig) -> accept
        "gene_symbol": ["GENEA", "GENEB", "GENEC", "GENED"],
        "pathogenicity": ["pathogenic", "pathogenic", "pathogenic", "benign"],
    }).to_parquet(path, index=False)


def _disposition(path):
    # vcf_pos - pos gives the offset; row1 offset 1 (exempt), others offset >=20 (wide)
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:2:200:na:na",
                       "clinvar:3:300:na:na", "clinvar:4:400:na:na"],
        "chrom": ["1", "2", "3", "4"],
        "pos": [100, 200, 300, 400],
        "type": ["deletion", "single nucleotide variant", "single nucleotide variant",
                 "single nucleotide variant"],
        "bucket": ["RECOVER", "RECOVER", "RECOVER", "RECOVER"],
        "ref": ["CG", "A", "T", "G"],
        "alt": ["C", "T", "C", "A"],
        "variation_id": ["11", "22", "999", "44"],   # recovered VCF IDs
        "vcf_pos": [101, 220, 320, 420],              # offsets: 1, 20, 20, 20
        "source": ["raw", "raw", "raw", "raw"],
        "genome_verified": [True, True, True, True],
    }).to_csv(path, sep="\t", index=False)


def _vs(path):
    hdr = ["VariationID", "GeneSymbol", "Assembly", "Chromosome", "Start"]
    rows = [
        ["22", "GENEB", "GRCh38", "2", "200"],   # cohort 2:200 VariationID = 22 (matches VCF ID 22)
        ["888", "GENEC", "GRCh38", "3", "300"],  # cohort 3:300 VariationID = 888 (VCF ID was 999 -> mismatch)
        ["44", "GENED", "GRCh38", "4", "400"],   # cohort 4:400 VariationID = 44 (matches) -- but test id2 too
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def _vcf(path):
    lines = [
        "##fileformat=VCFv4.1",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        "2\t220\t22\tA\tT\t.\t.\tGENEINFO=GENEB:111;CLNSIG=Pathogenic",
        "3\t320\t999\tT\tC\t.\t.\tGENEINFO=OTHERGENE:222;CLNSIG=Benign",  # gene disagrees -> id2 fail
        "4\t420\t44\tG\tA\t.\t.\tGENEINFO=GENED:333;CLNSIG=Benign",       # id1 will match anyway
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def test_identity_hardening(tmp_path):
    coh = tmp_path / "v2.parquet"; _cohort(coh)
    disp = tmp_path / "disp.tsv"; _disposition(disp)
    vs = tmp_path / "vs.txt.gz"; _vs(vs)
    vcf = tmp_path / "raw.vcf.gz"; _vcf(vcf)

    rc = hr.main(["--disposition", str(disp), "--cohort", str(coh), "--raw-vcf", str(vcf),
                  "--variant-summary", str(vs), "--assembly", "GRCh38",
                  "--outdir", str(tmp_path / "out")])
    assert rc == 0
    h = pd.read_csv(tmp_path / "out" / "alleleless_disposition_hardened.tsv", sep="\t")
    by = dict(zip(h["variant_id"], h["bucket_hardened"]))

    assert by["clinvar:1:100:na:na"] == "RECOVER"            # pos+1 exempt
    assert by["clinvar:2:200:na:na"] == "RECOVER"            # IDENTITY-1 match (22==22)
    assert by["clinvar:3:300:na:na"] == "STILL_UNRESOLVED"   # id 999 != 888, gene disagrees
    assert by["clinvar:4:400:na:na"] == "RECOVER"            # IDENTITY-1 match (44==44)

    # rebucketed row lost its alleles
    row3 = h[h["variant_id"] == "clinvar:3:300:na:na"].iloc[0]
    assert pd.isna(row3["ref"]) and pd.isna(row3["alt"])

    summ = json.loads((tmp_path / "out" / "alleleless_disposition_hardened_summary.json").read_text())
    assert summ["pos_or_pos1_exempt"] == 1
    assert summ["wide_window_checked"] == 3
    assert summ["wide_window_rebucketed"] == 1
    assert summ["recover_after_hardening"] == 3


def test_identity2_fallback_accepts(tmp_path):
    """A wide row failing IDENTITY-1 (no varid match) but passing IDENTITY-2 (gene+clnsig
    agree) must be accepted."""
    coh = tmp_path / "v2.parquet"
    pd.DataFrame({"variant_id": ["clinvar:5:500:na:na"], "gene_symbol": ["GENEE"],
                  "pathogenicity": ["pathogenic"]}).to_parquet(coh, index=False)
    disp = tmp_path / "disp.tsv"
    pd.DataFrame({"variant_id": ["clinvar:5:500:na:na"], "chrom": ["5"], "pos": [500],
                  "type": ["single nucleotide variant"], "bucket": ["RECOVER"],
                  "ref": ["A"], "alt": ["G"], "variation_id": ["77"], "vcf_pos": [520],
                  "source": ["raw"], "genome_verified": [True]}).to_csv(disp, sep="\t", index=False)
    vs = tmp_path / "vs.txt.gz"
    with gzip.open(vs, "wt") as fh:
        fh.write("VariationID\tGeneSymbol\tAssembly\tChromosome\tStart\n")
        fh.write("999\tGENEE\tGRCh38\t5\t500\n")   # cohort varid 999 != recovered 77 -> id1 fail
    vcf = tmp_path / "raw.vcf.gz"
    with gzip.open(vcf, "wt") as fh:
        fh.write("##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write("5\t520\t77\tA\tG\t.\t.\tGENEINFO=GENEE:444;CLNSIG=Pathogenic\n")  # gene+sig agree
    rc = hr.main(["--disposition", str(disp), "--cohort", str(coh), "--raw-vcf", str(vcf),
                  "--variant-summary", str(vs), "--assembly", "GRCh38",
                  "--outdir", str(tmp_path / "out")])
    assert rc == 0
    h = pd.read_csv(tmp_path / "out" / "alleleless_disposition_hardened.tsv", sep="\t")
    assert h.iloc[0]["bucket_hardened"] == "RECOVER"   # accepted via IDENTITY-2
    audit = pd.read_csv(tmp_path / "out" / "alleleless_recovery_identity_audit.tsv", sep="\t")
    assert bool(audit.iloc[0]["identity2_gene_clnsig"]) is True
    assert bool(audit.iloc[0]["identity1_varid"]) is False
