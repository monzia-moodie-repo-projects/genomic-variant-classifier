"""Tests for recover_by_sourceid.py -- source_id-keyed allele-less recovery."""
import gzip
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import recover_by_sourceid as rbs  # noqa: E402


def _write_vcf(path, records):
    """records: list of (chrom, pos, vid, ref, alt)."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for c, p, v, r, alt in records:
            f.write(f"{c}\t{p}\t{v}\t{r}\t{alt}\t.\t.\t.\n")


def _write_vs(path, records):
    """records: list of (vid, assembly, chrom, start)."""
    df = pd.DataFrame(records, columns=["VariationID", "Assembly", "Chromosome", "Start"])
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def _fasta(path):
    # chrom '1': pos100='A', pos101='C'; chrom '5': pos500='C'; chrom 'MT': pos961='G'
    s1 = list("N" * 1200); s1[99] = "A"; s1[100] = "C"
    s5 = list("N" * 1200); s5[499] = "C"; s5[500] = "C"
    smt = list("N" * 1200); smt[960] = "G"
    path.write_text(">1\n" + "".join(s1) + "\n>5\n" + "".join(s5) +
                    "\n>MT\n" + "".join(smt) + "\n")


def _cohort(path):
    # Six allele-less rows, all na:na, each with its OWN source_id:
    #  - row A: sid 111 at 1:100  -> VCF has real allele A>G, locus matches -> RECOVER
    #  - row B: sid 222 at 5:500  -> VCF record is itself na (CNV) -> CONFIRMED_ALLELELESS_CNV
    #  - row C: sid 333 at 1:100  -> co-located with A (collision); VCF has C>T -> RECOVER own
    #  - row D: sid 444 at 5:500  -> absent from both VCFs -> SID_NOT_IN_VCF_TRY_NCBI
    #  - row E: sid 555 at 1:100  -> VCF ref mismatches genome -> SID_GENOME_MISMATCH
    #  - row F: sid 666 at MT:961 -> VCF has G>A, but variant_summary places it elsewhere -> LOCUS_MISMATCH
    df = pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:5:500:na:na", "clinvar:1:100:na:na",
                       "clinvar:5:500:na:na", "clinvar:1:100:na:na", "clinvar:MT:961:na:na"],
        "source_id": ["111", "222", "333", "444", "555", "666"],
        "chrom": ["1", "5", "1", "5", "1", "MT"],
        "pos": [100, 500, 100, 500, 100, 961],
        "gene_symbol": ["G", "G", "G", "G", "G", "MT-RNR1"],
        "pathogenicity": ["pathogenic", "benign", "uncertain", "pathogenic", "pathogenic", "pathogenic"],
        "ref": ["na"] * 6, "alt": ["na"] * 6,
        "source_db": ["clinvar"] * 6,
    })
    df.to_parquet(path, index=False)


@pytest.fixture
def setup(tmp_path):
    coh = tmp_path / "cohort.parquet"; _cohort(coh)
    raw = tmp_path / "raw.vcf.gz"
    _write_vcf(raw, [
        ("1", 100, "111", "A", "G"),      # row A: clean recover
        ("5", 500, "222", "na", "na"),    # row B: CNV, allele-less in ClinVar too
        ("1", 100, "333", "AC", "A"),     # row C: co-located distinct variant (1bp del), own allele
        # 444 absent
        ("1", 100, "555", "T", "G"),      # row E: ref 'T' != genome 'A' at 1:100 -> mismatch
        ("MT", 961, "666", "G", "A"),     # row F: real allele but locus_ok will fail
    ])
    fresh = tmp_path / "fresh.vcf.gz"
    _write_vcf(fresh, [])  # nothing new
    vs = tmp_path / "vs.txt.gz"
    _write_vs(vs, [
        ("111", "GRCh38", "1", "100"),
        ("222", "GRCh38", "5", "500"),
        ("333", "GRCh38", "1", "100"),
        ("444", "GRCh38", "5", "500"),
        ("555", "GRCh38", "1", "100"),
        ("666", "GRCh38", "MT", "999"),   # placed at 999, NOT 961 -> locus mismatch
    ])
    fa = tmp_path / "g.fa"; _fasta(fa)
    return coh, raw, fresh, vs, fa, tmp_path


def _run(setup):
    coh, raw, fresh, vs, fa, tmp = setup
    rc = rbs.main([
        "--cohort", str(coh), "--raw-vcf", str(raw), "--fresh-vcf", str(fresh),
        "--variant-summary", str(vs), "--fasta", str(fa),
        "--old-recovered", str(tmp / "nonexistent.tsv"), "--assembly", "GRCh38",
        "--outdir", str(tmp / "out"),
    ])
    full = pd.read_csv(tmp / "out" / "alleleless_recovery_by_sid_full.tsv", sep="\t", dtype=str)
    return rc, full


def test_each_verdict_path(setup):
    rc, full = _run(setup)
    assert rc == 0
    v = dict(zip(full["source_id"], full["verdict"]))
    assert v["111"] == "RECOVER_BY_SID_RAW"
    assert v["222"] == "CONFIRMED_ALLELELESS_CNV"
    assert v["333"] == "RECOVER_BY_SID_RAW"
    assert v["444"] == "SID_NOT_IN_VCF_TRY_NCBI"
    assert v["555"] == "SID_GENOME_MISMATCH"
    assert v["666"] == "RECOVER_SID_LOCUS_MISMATCH"


def test_collision_rows_get_their_own_allele(setup):
    """Two co-located distinct source_ids (111, 333) at 1:100 each get their OWN allele."""
    rc, full = _run(setup)
    a = full[full["source_id"] == "111"].iloc[0]
    c = full[full["source_id"] == "333"].iloc[0]
    assert (a["rec_ref"], a["rec_alt"]) == ("A", "G")
    assert (c["rec_ref"], c["rec_alt"]) == ("AC", "A")
    # crucially they are DIFFERENT -- no splatter of one onto the other
    assert (a["rec_ref"], a["rec_alt"]) != (c["rec_ref"], c["rec_alt"])


def test_recovered_key_is_unique(setup):
    rc, full = _run(setup)
    recovered = pd.read_csv(
        setup[5] / "out" / "alleleless_recovered_by_sid.tsv", sep="\t", dtype=str)
    assert recovered.duplicated(subset=["source_id", "chrom", "pos"]).sum() == 0
    # only the two genuine recovers (111, 333) should be in the recovered file
    assert set(recovered["source_id"]) == {"111", "333"}


def test_cnv_excluded_with_reason(setup):
    rc, full = _run(setup)
    cnv = full[full["source_id"] == "222"].iloc[0]
    assert cnv["verdict"] == "CONFIRMED_ALLELELESS_CNV"
    assert pd.isna(cnv["rec_ref"]) or cnv["rec_ref"] in ("", "nan", "None")
