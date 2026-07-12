"""
tests/test_recover_identity_first.py  (2026-07-09)
Run: python -m pytest tests/test_recover_identity_first.py -v
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
ri = _load("recover_identity_first", "scripts/recover_identity_first.py")


def _fasta(path: Path):
    # chrom '1': pos 100 = 'A' (for varid 111 recovery ref check A>G at 100)
    # chrom '2': pos 200..202 = 'CAG' (for repeat varid 222 ref 'CAG' genome check)
    s1 = list("N" * 300); s1[99] = "A"
    s2 = list("N" * 300); s2[199] = "C"; s2[200] = "A"; s2[201] = "G"
    path.write_text(">1\n" + "".join(s1) + "\n>2\n" + "".join(s2) + "\n")


def _cohort(path):
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na",   # non-repeat, varid in FRESH only -> RECOVER_BY_ID_FRESH
                       "clinvar:2:200:na:na",    # repeat, varid in raw -> REPEAT_RECOVER_BY_ID
                       "clinvar:3:300:na:na",    # repeat, varid nowhere -> REPEAT_ALLELELESS
                       "clinvar:4:400:na:na"],   # non-repeat, varid nowhere -> STALE_MISS_TRY_NCBI
        "chrom": ["1", "2", "3", "4"], "pos": [100, 200, 300, 400],
        "ref": [None, None, None, None], "alt": [None, None, None, None],
        "gene_symbol": ["GENEA", "GENEB", "GENEC", "GENED"],
        "pathogenicity": ["pathogenic", "pathogenic", "benign", "pathogenic"],
    }).to_parquet(path, index=False)


def _disposition(path):
    # positional probe attached WRONG varids (simulating the neighbor mis-attach)
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:2:200:na:na"],
        "bucket": ["RECOVER", "RECOVER"],
        "variation_id": [99999.0, 88888.0],   # both wrong vs true 111 / 222
        "ref": ["X", "X"], "alt": ["Y", "Y"], "vcf_pos": [120, 220],
    }).to_csv(path, sep="\t", index=False)


def _vs(path):
    hdr = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome", "Start"]
    rows = [
        ["111", "single nucleotide variant", "GENEA", "GRCh38", "1", "100"],
        ["222", "Microsatellite", "GENEB", "GRCh38", "2", "200"],
        ["333", "Microsatellite", "GENEC", "GRCh38", "3", "300"],
        ["444", "Deletion", "GENED", "GRCh38", "4", "400"],
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def _raw_vcf(path):
    # raw has varid 222 (the repeat) only
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("##fileDate=2026-03-15\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write("2\t200\t222\tCAG\tC\t.\t.\t.\n")     # repeat allele, ref C matches genome at 200


def _fresh_vcf(path):
    # fresh has varid 111 (the SNV) that raw lacked
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("##fileDate=2026-07-09\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write("1\t100\t111\tA\tG\t.\t.\t.\n")        # ref A matches genome at 100


def test_identity_first_full(tmp_path):
    coh = tmp_path / "v2.parquet"; _cohort(coh)
    disp = tmp_path / "disp.tsv"; _disposition(disp)
    vs = tmp_path / "vs.txt.gz"; _vs(vs)
    raw = tmp_path / "raw.vcf.gz"; _raw_vcf(raw)
    fresh = tmp_path / "fresh.vcf.gz"; _fresh_vcf(fresh)
    fa = tmp_path / "g.fa"; _fasta(fa)

    rc = ri.main(["--disposition", str(disp), "--cohort", str(coh), "--raw-vcf", str(raw),
                  "--fresh-vcf", str(fresh), "--variant-summary", str(vs), "--fasta", str(fa),
                  "--assembly", "GRCh38", "--outdir", str(tmp_path / "out")])
    assert rc == 0
    res = pd.read_csv(tmp_path / "out" / "alleleless_identity_recovery_full.tsv", sep="\t")
    by = dict(zip(res["variant_id"], res["verdict"]))
    assert by["clinvar:1:100:na:na"] == "RECOVER_BY_ID_FRESH"
    assert by["clinvar:2:200:na:na"] == "REPEAT_RECOVER_BY_ID"
    assert by["clinvar:3:300:na:na"] == "REPEAT_ALLELELESS"
    assert by["clinvar:4:400:na:na"] == "STALE_MISS_TRY_NCBI"

    # recovered alleles are the TRUE by-ID ones, not the probe's X/Y
    rec = pd.read_csv(tmp_path / "out" / "alleleless_recovered_by_id.tsv", sep="\t")
    g = dict(zip(rec["variant_id"], zip(rec["rec_ref"], rec["rec_alt"])))
    assert g["clinvar:1:100:na:na"] == ("A", "G")
    assert g["clinvar:2:200:na:na"] == ("CAG", "C")

    # probe-wrong audit: both scored probe rows attached the wrong varid
    summ = json.loads((tmp_path / "out" / "alleleless_identity_recovery_summary.json").read_text())
    assert summ["positional_probe_was_wrong"] == 2
    assert summ["positional_probe_correct"] == 0


def test_genome_mismatch_quarantined(tmp_path):
    """A varid found by ID whose ref does NOT match the genome must be quarantined, not
    recovered."""
    coh = tmp_path / "v2.parquet"
    pd.DataFrame({"variant_id": ["clinvar:1:100:na:na"], "chrom": ["1"], "pos": [100],
                  "ref": [None], "alt": [None], "gene_symbol": ["GENEA"],
                  "pathogenicity": ["pathogenic"]}).to_parquet(coh, index=False)
    vs = tmp_path / "vs.txt.gz"
    with gzip.open(vs, "wt") as fh:
        fh.write("VariationID\tType\tGeneSymbol\tAssembly\tChromosome\tStart\n")
        fh.write("111\tsingle nucleotide variant\tGENEA\tGRCh38\t1\t100\n")
    raw = tmp_path / "raw.vcf.gz"
    with gzip.open(raw, "wt") as fh:
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write("1\t100\t111\tT\tG\t.\t.\t.\n")   # ref T but genome at 100 is A -> mismatch
    fa = tmp_path / "g.fa"; _fasta(fa)
    ri.main(["--cohort", str(coh), "--raw-vcf", str(raw), "--variant-summary", str(vs),
             "--fasta", str(fa), "--assembly", "GRCh38", "--outdir", str(tmp_path / "out"),
             "--disposition", str(tmp_path / "nope.tsv")])
    res = pd.read_csv(tmp_path / "out" / "alleleless_identity_recovery_full.tsv", sep="\t")
    assert res.iloc[0]["verdict"] == "RECOVER_BY_ID_GENOME_MISMATCH"
