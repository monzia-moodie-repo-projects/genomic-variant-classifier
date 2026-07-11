"""Tests for rebuild_cohort_v3_by_sid.py -- triple-keyed cohort v3 rebuild."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rebuild_cohort_v3_by_sid as rb  # noqa: E402


def _fasta(path):
    # 1:100='A'; 2:200='C',2:201='C'; MT:961='G'
    s1 = list("N" * 1200); s1[99] = "A"
    s2 = list("N" * 1200); s2[199] = "C"; s2[200] = "C"
    smt = list("N" * 1200); smt[960] = "G"
    path.write_text(">1\n" + "".join(s1) + "\n>2\n" + "".join(s2) +
                    "\n>MT\n" + "".join(smt) + "\n")


def _cohort(path):
    # 6 rows total. Allele-less rows (na:na):
    #   sid 111 @1:100  variant_id shared A  -> recovered (A>G)
    #   sid 333 @1:100  variant_id shared A  -> NOT recovered (co-located distinct) -> excluded
    #   sid 222 @2:200  variant_id B         -> recovered (pos-shift to 201, C>T)
    #   sid 999 @MT:961 variant_id C         -> excluded (CONFIRMED_ALLELELESS)
    # Non-allele-less normal rows:
    #   sid 500 @1:100 (real allele already)  variant_id N1
    #   sid 600 @7:700 (real allele already)  variant_id N2
    df = pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:1:100:na:na", "clinvar:2:200:na:na",
                       "clinvar:MT:961:na:na", "clinvar:1:100:A:G_x", "clinvar:7:700:C:T"],
        "source_id": ["111", "333", "222", "999", "500", "600"],
        "chrom": ["1", "1", "2", "MT", "1", "7"],
        "pos": [100, 100, 200, 961, 100, 700],
        "gene_symbol": ["G"] * 6,
        "pathogenicity": ["pathogenic", "uncertain", "pathogenic", "benign", "pathogenic", "benign"],
        "ref": ["na", "na", "na", "na", "A", "C"],
        "alt": ["na", "na", "na", "na", "T", "T"],
        "source_db": ["clinvar"] * 6,
    })
    df.to_parquet(path, index=False)


def _recovered(path):
    # only 111 and 222 recovered; 222 shifts pos 200->201
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:2:200:na:na"],
        "chrom": ["1", "2"], "pos": [100, 200], "source_id": ["111", "222"],
        "rec_pos": [100, 201], "rec_ref": ["A", "C"], "rec_alt": ["G", "T"],
        "rec_source": ["raw", "raw"],
        "verdict": ["RECOVER_BY_SID_RAW", "RECOVER_BY_SID_RAW"],
    }).to_csv(path, sep="\t", index=False)


def _disposition(path):
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:1:100:na:na", "clinvar:2:200:na:na",
                       "clinvar:MT:961:na:na"],
        "source_id": ["111", "333", "222", "999"],
        "chrom": ["1", "1", "2", "MT"], "pos": [100, 100, 200, 961],
        "verdict": ["RECOVER_BY_SID_RAW", "CONFIRMED_ALLELELESS_CNV",
                    "RECOVER_BY_SID_RAW", "CONFIRMED_ALLELELESS_CNV"],
    }).to_csv(path, sep="\t", index=False)


@pytest.fixture
def setup(tmp_path):
    coh = tmp_path / "v2.parquet"; _cohort(coh)
    rec = tmp_path / "rec.tsv"; _recovered(rec)
    disp = tmp_path / "disp.tsv"; _disposition(disp)
    fa = tmp_path / "g.fa"; _fasta(fa)
    return coh, rec, disp, fa, tmp_path


def test_build_merges_and_reconciles(setup):
    coh, rec, disp, fa, tmp = setup
    out = tmp / "v3.parquet"
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-sid", str(rec),
                  "--disposition", str(disp), "--fasta", str(fa),
                  "--out", str(out), "--skip-md5-check"])
    assert rc == 0
    v3 = pd.read_parquet(out)
    # 6 total - 2 excluded (333, 999) = 4 rows
    assert len(v3) == 4
    ids = set(v3["variant_id"])
    # 111 recovered -> canonical; 222 recovered at shifted pos 201; two normals unchanged
    assert "clinvar:1:100:A:G" in ids
    assert "clinvar:2:201:C:T" in ids
    assert "clinvar:1:100:A:G_x" in ids and "clinvar:7:700:C:T" in ids
    r222 = v3[v3["source_id"] == "222"].iloc[0]
    assert int(r222["pos"]) == 201 and r222["ref"] == "C" and r222["alt"] == "T"
    # reconciliation counts ROWS: 4 allele-less rows = 2 recovered + 2 excluded
    recon = __import__("json").loads((tmp / "cohort_v3_reconciliation.json").read_text())
    assert recon["allele_less_rows"] == 4
    assert recon["recovered_merged"] == 2
    assert recon["excluded_total"] == 2
    assert recon["reconciliation_ok"] is True


def test_no_splatter_on_collision(setup):
    """333 shares variant_id clinvar:1:100:na:na with recovered 111 but is NOT recovered.
    It must be EXCLUDED, never given 111's allele."""
    coh, rec, disp, fa, tmp = setup
    out = tmp / "v3.parquet"
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-sid", str(rec),
                  "--disposition", str(disp), "--fasta", str(fa),
                  "--out", str(out), "--skip-md5-check"])
    assert rc == 0
    v3 = pd.read_parquet(out)
    # source_id 333 must NOT appear in v3 (excluded), proving no splatter
    assert "333" not in set(v3["source_id"])
    # and the excluded doc records it
    excl = pd.read_csv(tmp / "cohort_v3_excluded_alleleless.tsv", sep="\t", dtype=str)
    assert "333" in set(excl["source_id"])


def test_genome_reverify_aborts(setup):
    coh, rec, disp, fa, tmp = setup
    bad = tmp / "rec_bad.tsv"
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na"], "chrom": ["1"], "pos": [100],
        "source_id": ["111"], "rec_pos": [100], "rec_ref": ["T"], "rec_alt": ["G"],  # T!=A
        "rec_source": ["raw"], "verdict": ["RECOVER_BY_SID_RAW"],
    }).to_csv(bad, sep="\t", index=False)
    out = tmp / "v3.parquet"
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-sid", str(bad),
                  "--disposition", str(disp), "--fasta", str(fa),
                  "--out", str(out), "--skip-md5-check"])
    assert rc == 4


def test_subset_guard_aborts(setup):
    coh, rec, disp, fa, tmp = setup
    stray = tmp / "rec_stray.tsv"
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na"], "chrom": ["1"], "pos": [100],
        "source_id": ["NOTREAL"], "rec_pos": [100], "rec_ref": ["A"], "rec_alt": ["G"],
        "rec_source": ["raw"], "verdict": ["RECOVER_BY_SID_RAW"],
    }).to_csv(stray, sep="\t", index=False)
    out = tmp / "v3.parquet"
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-sid", str(stray),
                  "--disposition", str(disp), "--fasta", str(fa),
                  "--out", str(out), "--skip-md5-check"])
    assert rc == 5


def test_refuse_overwrite(setup):
    coh, rec, disp, fa, tmp = setup
    out = tmp / "v3.parquet"; out.write_text("x")
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-sid", str(rec),
                  "--disposition", str(disp), "--fasta", str(fa),
                  "--out", str(out), "--skip-md5-check"])
    assert rc == 2


def test_md5_guard_aborts(setup):
    coh, rec, disp, fa, tmp = setup
    out = tmp / "v3.parquet"
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-sid", str(rec),
                  "--disposition", str(disp), "--fasta", str(fa), "--out", str(out)])
    assert rc == 3  # real MD5 != expected canonical
