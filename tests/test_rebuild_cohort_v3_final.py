"""
tests/test_rebuild_cohort_v3_final.py  (2026-07-09)
Run: python -m pytest tests/test_rebuild_cohort_v3_final.py -v
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


try:
    _load("allele_classify", "src/genomic_variant_classifier/data/allele_classify.py")
except FileNotFoundError:
    _load("allele_classify", "scripts/allele_classify.py")
rb = _load("rebuild_cohort_v3_final", "scripts/rebuild_cohort_v3_final.py")


def _fasta(path: Path):
    # chrom '1' pos 100='A'; '2' pos 200='C' AND pos 201='C'; '6' pos 600='G'; '9' pos 900='G'
    s1 = list("N" * 1000); s1[99] = "A"
    s2 = list("N" * 1000); s2[199] = "C"; s2[200] = "C"
    s6 = list("N" * 1000); s6[599] = "G"
    s9 = list("N" * 1000); s9[899] = "G"
    path.write_text(">1\n" + "".join(s1) + "\n>2\n" + "".join(s2) +
                    "\n>6\n" + "".join(s6) + "\n>9\n" + "".join(s9) + "\n")


def _cohort(path):
    # 7 rows: 2 raw-recoverable, 1 ncbi-recoverable, 2 confirmed-alleleless, 1 quarantine, 1 normal
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na",   # recover raw -> ref A alt G
                       "clinvar:2:200:na:na",    # recover raw -> ref C alt T
                       "clinvar:6:600:na:na",    # recover NCBI -> ref G alt A (chrom 6 pos 600)
                       "clinvar:3:300:na:na",    # confirmed alleleless
                       "clinvar:4:400:na:na",    # confirmed alleleless
                       "clinvar:9:900:na:na",    # quarantine (genome mismatch)
                       "clinvar:5:500:A:G"],     # normal, retained untouched
        "chrom": ["1", "2", "6", "3", "4", "9", "5"],
        "pos": [100, 200, 600, 300, 400, 900, 500],
        "ref": [None, None, None, None, None, None, "A"],
        "alt": [None, None, None, None, None, None, "G"],
        "gene_symbol": ["GA", "GB", "GN", "GC", "GD", "GE", "GF"],
        "pathogenicity": ["pathogenic", "benign", "pathogenic", "pathogenic", "pathogenic",
                          "pathogenic", "benign"],
    }).to_parquet(path, index=False)


def _recovered(path):
    # varid 1: SNV, rec_pos == cohort pos 100
    # varid 2: pos-shifted DELETION, cohort pos 200 but true rec_pos 201 (ref 'C' at 201)
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:2:200:na:na"],
        "chrom": ["1", "2"], "pos": [100, 200], "rec_pos": [100, 201],
        "cohort_varid": ["111", "222"],
        "rec_ref": ["A", "C"], "rec_alt": ["G", "T"],
        "rec_source": ["raw", "raw"], "verdict": ["RECOVER_BY_ID_RAW", "RECOVER_BY_ID_RAW"],
    }).to_csv(path, sep="\t", index=False)


def _ncbi(path):
    pd.DataFrame({
        "variant_id": ["clinvar:6:600:na:na", "clinvar:9:900:na:na"],
        "cohort_varid": ["666", "999"], "chrom": ["6", "9"], "pos": [600, 900],
        "ref": ["G", "T"], "alt": ["A", "A"],
        "spdi": ["NC_000006.11:599:G:A", "NC_000009.11:899:T:A"],
        "ncbi_verdict": ["RESOLVED_HAS_ALLELE", "RESOLVED_GENOME_MISMATCH"],
        "genome_ok": [True, False],
    }).to_csv(path, sep="\t", index=False)


def _disposition(path):
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:2:200:na:na", "clinvar:6:600:na:na",
                       "clinvar:3:300:na:na", "clinvar:4:400:na:na", "clinvar:9:900:na:na"],
        "verdict": ["RECOVER_BY_ID_RAW", "RECOVER_BY_ID_RAW", "STALE_MISS_TRY_NCBI",
                    "STALE_MISS_TRY_NCBI", "STALE_MISS_TRY_NCBI", "STALE_MISS_TRY_NCBI"],
    }).to_csv(path, sep="\t", index=False)


def _setup(tmp_path):
    coh = tmp_path / "v2.parquet"; _cohort(coh)
    rec = tmp_path / "rec.tsv"; _recovered(rec)
    nc = tmp_path / "ncbi.tsv"; _ncbi(nc)
    disp = tmp_path / "disp.tsv"; _disposition(disp)
    fa = tmp_path / "g.fa"; _fasta(fa)
    return coh, rec, nc, disp, fa


def test_v3_build_merges_and_excludes(tmp_path):
    coh, rec, nc, disp, fa = _setup(tmp_path)
    out = tmp_path / "proc" / "v3.parquet"
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-id", str(rec),
                  "--ncbi-resolved", str(nc), "--disposition", str(disp),
                  "--fasta", str(fa), "--out", str(out), "--skip-md5-check"])
    assert rc == 0
    v3 = pd.read_parquet(out)
    # 7 - 3 excluded (3,4 confirmed + 9 quarantine) = 4 rows (1,2,6 recovered + 5 normal)
    assert len(v3) == 4
    ids = set(v3["variant_id"])
    # recovered rows now carry canonical ids clinvar:chrom:rec_pos:ref:alt (placeholder replaced)
    assert ids == {"clinvar:1:100:A:G", "clinvar:2:201:C:T", "clinvar:6:600:G:A",
                   "clinvar:5:500:A:G"}
    # recovered alleles + true coordinate present (raw SNV, raw shifted-deletion, ncbi)
    r1 = v3[v3["variant_id"] == "clinvar:1:100:A:G"].iloc[0]
    assert r1["ref"] == "A" and r1["alt"] == "G" and int(r1["pos"]) == 100
    r2 = v3[v3["variant_id"] == "clinvar:2:201:C:T"].iloc[0]
    assert r2["ref"] == "C" and r2["alt"] == "T" and int(r2["pos"]) == 201   # pos shifted to rec_pos
    r6 = v3[v3["variant_id"] == "clinvar:6:600:G:A"].iloc[0]
    assert r6["ref"] == "G" and r6["alt"] == "A"   # from NCBI RESOLVED_HAS_ALLELE
    # zero na:na remain
    import allele_classify as ac
    assert int(ac.is_allele_less(v3["ref"], v3["alt"]).sum()) == 0
    # reconciliation + excluded docs
    recon = json.loads((out.parent / "cohort_v3_reconciliation.json").read_text())
    assert recon["v3_rows"] == 4 and recon["recovered_merged"] == 3 and recon["excluded_total"] == 3
    excl = pd.read_csv(out.parent / "cohort_v3_excluded_alleleless.tsv", sep="\t")
    assert set(excl["variant_id"]) == {"clinvar:3:300:na:na", "clinvar:4:400:na:na",
                                       "clinvar:9:900:na:na"}


def test_refuse_overwrite(tmp_path):
    coh, rec, nc, disp, fa = _setup(tmp_path)
    out = tmp_path / "v3.parquet"; out.write_text("exists")
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-id", str(rec),
                  "--ncbi-resolved", str(nc), "--disposition", str(disp),
                  "--fasta", str(fa), "--out", str(out), "--skip-md5-check"])
    assert rc == 2   # refuse-overwrite


def test_md5_guard_aborts(tmp_path):
    coh, rec, nc, disp, fa = _setup(tmp_path)
    out = tmp_path / "v3.parquet"
    # WITHOUT --skip-md5-check the synthetic cohort's md5 != canonical -> abort
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-id", str(rec),
                  "--ncbi-resolved", str(nc), "--disposition", str(disp),
                  "--fasta", str(fa), "--out", str(out)])
    assert rc == 3


def test_genome_reverify_aborts(tmp_path):
    """A recovery whose ref does NOT match the genome must abort the build."""
    coh, rec, nc, disp, fa = _setup(tmp_path)
    bad = tmp_path / "rec_bad.tsv"
    pd.DataFrame({
        "variant_id": ["clinvar:1:100:na:na", "clinvar:2:200:na:na"],
        "chrom": ["1", "2"], "pos": [100, 200], "rec_pos": [100, 200],
        "cohort_varid": ["111", "222"],
        "rec_ref": ["T", "C"],   # chrom1 pos100 is 'A' not 'T' -> mismatch -> abort
        "rec_alt": ["G", "T"], "rec_source": ["raw", "raw"],
        "verdict": ["RECOVER_BY_ID_RAW", "RECOVER_BY_ID_RAW"],
    }).to_csv(bad, sep="\t", index=False)
    out = tmp_path / "v3.parquet"
    rc = rb.main(["--cohort-v2", str(coh), "--recovered-by-id", str(bad),
                  "--ncbi-resolved", str(nc), "--disposition", str(disp),
                  "--fasta", str(fa), "--out", str(out), "--skip-md5-check"])
    assert rc == 4   # genome re-verification failure


def test_duplicate_collision_aborts(tmp_path):
    """A recovered variant whose canonical id collides with an existing cohort row aborts."""
    coh, rec, nc, disp, fa = _setup(tmp_path)
    # make the normal row's id exactly the canonical id varid 1 will remap to
    c = pd.read_parquet(coh)
    c.loc[c["variant_id"] == "clinvar:5:500:A:G", "variant_id"] = "clinvar:1:100:A:G"
    c.loc[c["variant_id"] == "clinvar:1:100:A:G", ["chrom", "pos", "ref", "alt"]] = ["1", 100, "A", "G"]
    coh2 = tmp_path / "v2b.parquet"; c.to_parquet(coh2, index=False)
    out = tmp_path / "v3.parquet"
    rc = rb.main(["--cohort-v2", str(coh2), "--recovered-by-id", str(rec),
                  "--ncbi-resolved", str(nc), "--disposition", str(disp),
                  "--fasta", str(fa), "--out", str(out), "--skip-md5-check"])
    assert rc == 9   # duplicate-id collision
