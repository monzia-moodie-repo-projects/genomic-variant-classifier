"""Tests for classify_alleleless_by_type.py -- type-aware allele-less disposition."""
import gzip
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import classify_alleleless_by_type as cl  # noqa: E402


def _vcf(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for c, p, v, r, alt in records:
            f.write(f"{c}\t{p}\t{v}\t{r}\t{alt}\t.\t.\t.\n")


def _vs(path, records):
    pd.DataFrame(records, columns=["VariationID", "Type", "Assembly"]).to_csv(
        path, sep="\t", index=False, compression="gzip")


def _fasta(path):
    s1 = list("N" * 1200); s1[99] = "A"        # 1:100 = A
    s2 = list("N" * 1200); s2[199] = "C"        # 2:200 = C
    path.write_text(">1\n" + "".join(s1) + "\n>2\n" + "".join(s2) + "\n")


def _cohort(path):
    # sid 10 CNV @1:100 -> SV; sid 20 SNV not-in-vcf @1:100; sid 30 SNV in-vcf @1:100 (A>G, recover);
    # sid 40 SNV in-vcf @2:200 ref 'T' != genome 'C' -> genome mismatch;
    # sid 50 not in variant_summary @1:100 -> NO_VS; plus one normal row.
    df = pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na"] * 5 + ["clinvar:9:900:A:T"],
        "source_id": ["10", "20", "30", "40", "50", "900"],
        "chrom": ["1", "1", "1", "2", "1", "9"],
        "pos": [100, 100, 100, 200, 100, 900],
        "ref": ["na", "na", "na", "na", "na", "A"],
        "alt": ["na", "na", "na", "na", "na", "T"],
        "source_db": ["clinvar"] * 6,
    })
    df.to_parquet(path, index=False)


@pytest.fixture
def setup(tmp_path):
    coh = tmp_path / "c.parquet"; _cohort(coh)
    raw = tmp_path / "raw.vcf.gz"
    _vcf(raw, [
        ("1", 100, "30", "A", "G"),   # sid 30 recover
        ("2", 200, "40", "T", "G"),   # sid 40 ref T != genome C -> mismatch
    ])
    fresh = tmp_path / "fresh.vcf.gz"; _vcf(fresh, [])
    vs = tmp_path / "vs.txt.gz"
    _vs(vs, [
        ("10", "copy number gain", "GRCh38"),
        ("20", "single nucleotide variant", "GRCh38"),
        ("30", "single nucleotide variant", "GRCh38"),
        ("40", "single nucleotide variant", "GRCh38"),
        # 50 intentionally absent from variant_summary
    ])
    fa = tmp_path / "g.fa"; _fasta(fa)
    return coh, raw, fresh, vs, fa, tmp_path


def _run(setup):
    coh, raw, fresh, vs, fa, tmp = setup
    rc = cl.main(["--cohort", str(coh), "--raw-vcf", str(raw), "--fresh-vcf", str(fresh),
                  "--variant-summary", str(vs), "--fasta", str(fa), "--outdir", str(tmp / "o")])
    full = pd.read_csv(tmp / "o" / "alleleless_recovery_by_sid_full.tsv", sep="\t", dtype=str)
    return rc, full, tmp


def test_each_type_path(setup):
    rc, full, tmp = _run(setup)
    assert rc == 0
    v = dict(zip(full["source_id"], full["verdict"]))
    assert v["10"] == "CONFIRMED_ALLELELESS_SV"
    assert v["20"] == "CONFIRMED_ALLELELESS_SNV_NOT_IN_VCF"
    assert v["30"] == "RECOVER_BY_SID_RAW"
    assert v["40"] == "SID_GENOME_MISMATCH"
    assert v["50"] == "CONFIRMED_ALLELELESS_NO_VS"


def test_only_genuine_recover_in_recovered_file(setup):
    rc, full, tmp = _run(setup)
    recovered = pd.read_csv(tmp / "o" / "alleleless_recovered_by_sid.tsv", sep="\t", dtype=str)
    assert set(recovered["source_id"]) == {"30"}
    assert recovered.duplicated(subset=["source_id", "chrom", "pos"]).sum() == 0
    r = recovered.iloc[0]
    assert r["rec_ref"] == "A" and r["rec_alt"] == "G"


def test_reason_recorded(setup):
    rc, full, tmp = _run(setup)
    sv = full[full["source_id"] == "10"].iloc[0]
    assert "structural" in sv["reason"]
    assert sv["vs_type"] == "copy number gain"
