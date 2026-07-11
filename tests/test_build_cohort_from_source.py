"""
test_build_cohort_from_source.py  (2026-07-09)
Full battery for scripts/build_cohort_from_source.py. Verifies real correctness against a
synthetic cohort and a tiny known genome, not just that the code runs.

Coverage:
  1. quarantine completeness -- BOTH na:na (None/None) AND half-bad (real/'.') removed
  2. padded-deletion position correction (pos -= 1) is exactly right
  3. non-padded rows (SNV, insertion, delins) are NOT shifted
  4. row reconciliation: quarantined + clean == input
  5. composition invariance across the correction
  6. no duplicate variant_id post-condition
  7. G7 bad-allele post-condition fires if a bad row somehow reaches the clean set
  8. SNV control passes on a consistent genome
  9. G8 all-indel genome-consistency PASSES on correct coords and FAILS on wrong coords
     (this is the bug-catching test -- a deliberately mis-positioned padded deletion must
      be caught by the genome check, never shipped silently)
 10. determinism: same input -> identical output MD5
 11. PROVISIONAL marking when --genome absent
 12. refuse-to-overwrite (G1) without --force
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT = ROOT / "scripts" / "build_cohort_from_source.py"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_cohort_from_source as B  # noqa: E402


# ---- a tiny known genome ---------------------------------------------------------------
# chrom "1": 1-based positions. We design alleles to match this exact sequence.
#   pos: 1234567890123456789012345
GENOME1 = "ACGTACGTACGTACGTACGTACGTA"   # chrom "1"
GENOME2 = "TTTTGGGGCCCCAAAATTTTGGGGC"   # chrom "2"


def _write_fasta(tmp_path: Path) -> Path:
    fa = tmp_path / "mini.fa"
    fa.write_text(f">1\n{GENOME1}\n>2\n{GENOME2}\n")
    # pyfaidx needs an index; it builds one on open. No .fai pre-needed.
    return fa


def _base(chrom, pos1):
    """1-based base from the known genome, for building self-consistent alleles."""
    g = GENOME1 if str(chrom) == "1" else GENOME2
    return g[pos1 - 1]


def _synthetic_cohort():
    """Rows with KNOWN correct post-correction coordinates.

    Padded deletions are stored at the UNCORRECTED Start (= VCF pos + 1), so the builder
    must decrement them by 1 to land on the genome. ref begins at the CORRECTED pos.
    """
    rows = []

    # --- SNV at 1:5 (genome[5]='A'); ref must equal genome[5], never shifted ---
    rows.append(dict(variant_id="clinvar:1:5:A:C", chrom="1", pos=5,
                     ref=_base("1", 5), alt="C", kind="snv"))

    # --- padded deletion on chrom 1: true VCF pos = 9, ref = genome[9:11] = "AC", alt = "A"
    #     stored at Start = 10 (VCF+1); builder must correct 10 -> 9 so ref "AC" sits at 9.
    ref_del = GENOME1[9 - 1:9 - 1 + 2]   # genome positions 9,10 (1-based) -> "AC"
    assert ref_del[0] == GENOME1[8]
    rows.append(dict(variant_id="clinvar:1:10:%s:%s" % (ref_del, ref_del[0]),
                     chrom="1", pos=10, ref=ref_del, alt=ref_del[0], kind="pdel"))

    # --- padded insertion on chrom 2: ref = single base at pos 4, alt = ref + extra.
    #     never shifted; ref must match genome at pos 4.
    ins_ref = _base("2", 4)
    rows.append(dict(variant_id="clinvar:2:4:%s:%sGG" % (ins_ref, ins_ref),
                     chrom="2", pos=4, ref=ins_ref, alt=ins_ref + "GG", kind="pins"))

    # --- na:na (both None) -- must be quarantined ---
    rows.append(dict(variant_id="clinvar:1:12:None:None", chrom="1", pos=12,
                     ref=None, alt=None, kind="nana"))

    # --- half-bad (alt = '.') -- must be quarantined ---
    rows.append(dict(variant_id="clinvar:2:8:%s:." % _base("2", 8), chrom="2", pos=8,
                     ref=_base("2", 8), alt=".", kind="halfbad"))

    df = pd.DataFrame(rows)
    return df[["variant_id", "chrom", "pos", "ref", "alt", "kind"]]


@pytest.fixture
def cohort():
    return _synthetic_cohort()


# ---- core build (no genome) ------------------------------------------------------------

def test_quarantine_removes_both_bad_classes(cohort):
    recon = B.BuildReconciliation()
    clean, structural = B.build(cohort.drop(columns=["kind"]), recon)
    assert recon.quarantined_bad_allele == 2
    assert recon.quarantined_na_na == 1
    assert recon.quarantined_half_bad == 1
    assert recon.clean_rows == 3
    # no bad allele survives
    bad_left = B.is_empty_allele(clean["ref"]) | B.is_empty_allele(clean["alt"])
    assert int(bad_left.sum()) == 0


def test_reconciliation_holds(cohort):
    recon = B.BuildReconciliation()
    B.build(cohort.drop(columns=["kind"]), recon)
    assert recon.reconciles()
    assert recon.input_rows == recon.quarantined_bad_allele + recon.clean_rows


def test_padded_deletion_position_corrected(cohort):
    recon = B.BuildReconciliation()
    clean, _ = B.build(cohort.drop(columns=["kind"]), recon)
    pdel = clean[clean["ref"].str.len() > clean["alt"].str.len()]
    assert len(pdel) == 1
    assert int(pdel.iloc[0]["pos"]) == 9          # corrected 10 -> 9
    assert recon.padded_deletions_corrected == 1
    # variant_id rebuilt to the corrected pos
    assert pdel.iloc[0]["variant_id"] == "clinvar:1:9:AC:A"


def test_non_padded_rows_not_shifted(cohort):
    recon = B.BuildReconciliation()
    clean, _ = B.build(cohort.drop(columns=["kind"]), recon)
    snv = clean[(clean["ref"].str.len() == 1) & (clean["alt"].str.len() == 1)]
    assert int(snv.iloc[0]["pos"]) == 5           # unchanged
    ins = clean[clean["alt"].str.len() > clean["ref"].str.len()]
    assert int(ins.iloc[0]["pos"]) == 4           # unchanged


def test_composition_invariant(cohort):
    recon = B.BuildReconciliation()
    B.build(cohort.drop(columns=["kind"]), recon)
    assert recon.composition_before == recon.composition_after


def test_no_duplicate_variant_id(cohort):
    recon = B.BuildReconciliation()
    B.build(cohort.drop(columns=["kind"]), recon)
    assert recon.dup_variant_id == 0


def test_duplicate_variant_ids_are_collapsed_not_raised():
    # NEW CONTRACT (2026-07-10): the dedup collapse merges duplicate variant_id groups into a
    # single most-severe survivor BEFORE the G9 post-condition, so build() no longer raises on
    # duplicates -- it collapses them. G9 remains as a backstop and must pass (dup == 0).
    ref_del = GENOME1[8:10]  # "AC"
    dup = pd.DataFrame([
        dict(variant_id="clinvar:1:10:AC:A", chrom="1", pos=10, ref=ref_del, alt=ref_del[0]),
        dict(variant_id="clinvar:1:10:AC:A", chrom="1", pos=10, ref=ref_del, alt=ref_del[0]),
    ])
    recon = B.BuildReconciliation()
    clean, structural = B.build(dup, recon)   # must NOT raise
    assert int(clean["variant_id"].duplicated().sum()) == 0
    assert recon.dup_variant_id == 0
    assert getattr(recon, "collapsed_groups", 0) >= 1


def test_collapse_robust_to_minimal_required_columns():
    # The collapse must tolerate frames carrying ONLY REQUIRED_COLS (no source_id/pathogenicity/
    # clinical_sig/metadata) -- these are optional per REQUIRED_COLS. Must not KeyError.
    ref_del = GENOME1[8:10]
    dup = pd.DataFrame([
        dict(variant_id="clinvar:1:10:AC:A", chrom="1", pos=10, ref=ref_del, alt=ref_del[0]),
        dict(variant_id="clinvar:1:10:AC:A", chrom="1", pos=10, ref=ref_del, alt=ref_del[0]),
    ])
    out, audit = B.collapse_duplicate_variants(dup)
    assert len(out) == 1
    assert int(out["variant_id"].duplicated().sum()) == 0
    assert audit[0]["collapsed_from_n"] == 2
    assert audit[0]["classification_conflict"] is False


# ---- genome-backed guards --------------------------------------------------------------

pyfaidx = pytest.importorskip("pyfaidx", reason="genome checks need pyfaidx")


def test_snv_control_and_indel_check_pass_on_correct_coords(cohort, tmp_path):
    fa = _write_fasta(tmp_path)
    recon = B.BuildReconciliation()
    clean, _ = B.build(cohort.drop(columns=["kind"]), recon)
    B.reference_and_indel_check(clean, fa, recon, max_mismatch_rate=0.001)
    assert recon.reference_check.startswith("PASSED")
    assert "match at pos-1" in recon.snv_control
    assert recon.indel_postcondition.startswith("PASSED")


def test_indel_check_CATCHES_wrong_position(cohort, tmp_path):
    """BUG-CATCHING TEST: a padded deletion left at the WRONG (uncorrected) position must be
    caught by the genome check. We bypass the correction to simulate the exact defect the
    old clean.parquet had (pos = Start = VCF+1), and assert the guard FAILS loudly."""
    fa = _write_fasta(tmp_path)
    recon = B.BuildReconciliation()
    clean, _ = B.build(cohort.drop(columns=["kind"]), recon)
    # SABOTAGE: move the padded deletion back to the uncorrected +1 position.
    mask = clean["ref"].str.len() > clean["alt"].str.len()
    clean.loc[mask, "pos"] = clean.loc[mask, "pos"] + 1   # 9 -> 10, now ref "AC" no longer at pos
    with pytest.raises(ValueError, match="INDEL GENOME-CONSISTENCY"):
        B.reference_and_indel_check(clean, fa, recon, max_mismatch_rate=0.0)


def test_snv_control_catches_wrong_build(tmp_path):
    """If EVERY SNV is off (wrong genome), the SNV control must fail, not the deletion path."""
    fa = _write_fasta(tmp_path)
    # SNV whose ref deliberately does NOT match genome at pos (wrong build signature)
    wrong = pd.DataFrame([
        dict(variant_id="clinvar:1:5:Z:C", chrom="1", pos=5, ref="Z", alt="C")
        for _ in range(50)
    ])
    recon = B.BuildReconciliation()
    with pytest.raises(ValueError, match="SNV CONTROL FAILED"):
        B.reference_and_indel_check(wrong, fa, recon)


# ---- end-to-end via subprocess (determinism, provisional, refuse-overwrite) ------------

def _run(args, cwd):
    return subprocess.run([sys.executable, str(SCRIPT)] + args,
                          cwd=str(cwd), capture_output=True, text=True)


def test_provisional_without_genome(cohort, tmp_path):
    inp = tmp_path / "raw.parquet"
    cohort.drop(columns=["kind"]).to_parquet(inp, index=False)
    out = tmp_path / "cohort.parquet"
    r = _run(["--apply", "--input", str(inp), "--output", str(out)], ROOT)
    assert r.returncode == 0, r.stderr
    assert out.with_name(out.stem + "_reconciliation.json").exists(), \
        "per-build reconciliation JSON must be written"
    recon = json.loads((out.with_name("cohort_build_reconciliation.json")).read_text())
    assert recon["reference_check"] == "SKIPPED_NO_GENOME"
    assert any("PROVISIONAL" in n for n in recon["notes"])


def test_determinism_same_md5(cohort, tmp_path):
    inp = tmp_path / "raw.parquet"
    cohort.drop(columns=["kind"]).to_parquet(inp, index=False)
    out1 = tmp_path / "a.parquet"
    out2 = tmp_path / "b.parquet"
    r1 = _run(["--apply", "--input", str(inp), "--output", str(out1)], ROOT)
    r2 = _run(["--apply", "--input", str(inp), "--output", str(out2)], ROOT)
    assert r1.returncode == 0 and r2.returncode == 0
    assert B._md5(out1) == B._md5(out2)


def test_refuse_overwrite_without_force(cohort, tmp_path):
    inp = tmp_path / "raw.parquet"
    cohort.drop(columns=["kind"]).to_parquet(inp, index=False)
    out = tmp_path / "cohort.parquet"
    r1 = _run(["--apply", "--input", str(inp), "--output", str(out)], ROOT)
    assert r1.returncode == 0
    r2 = _run(["--apply", "--input", str(inp), "--output", str(out)], ROOT)
    assert r2.returncode == 5           # G1 refuse-overwrite
    r3 = _run(["--apply", "--force", "--input", str(inp), "--output", str(out)], ROOT)
    assert r3.returncode == 0           # --force succeeds


def test_audit_writes_nothing_to_output_dir(cohort, tmp_path):
    """An --audit dry-run must not write the cohort OR any diagnostic side-file into the
    output's own directory (which in production is the data tree). The mismatch file, if
    any, belongs in outputs/. Here we run WITHOUT --genome (no mismatch file is produced),
    and assert the output directory stays empty except the input we placed."""
    datadir = tmp_path / "processed"
    datadir.mkdir()
    inp = datadir / "raw.parquet"
    cohort.drop(columns=["kind"]).to_parquet(inp, index=False)
    out = datadir / "cohort.parquet"
    r = _run(["--audit", "--input", str(inp), "--output", str(out)], ROOT)
    assert r.returncode == 0, r.stderr
    # only the input parquet should remain in the data dir; no cohort, no side-file, no json
    remaining = sorted(p.name for p in datadir.iterdir())
    assert remaining == ["raw.parquet"], f"audit left files in data dir: {remaining}"


# ---- ref_genome_consistent 3-state flag -------------------------------------------------

def test_flag_true_for_genome_matching_rows(cohort, tmp_path):
    fa = _write_fasta(tmp_path)
    recon = B.BuildReconciliation()
    clean, _ = B.build(cohort.drop(columns=["kind"]), recon)
    flag = B.annotate_genome_consistency(clean, fa)
    # every synthetic clean row was built to match the mini genome -> all True
    assert flag.notna().all()
    assert bool((flag == True).all()), f"expected all True, got {flag.tolist()}"  # noqa: E712


def test_flag_false_for_genome_inconsistent_row_not_dropped(cohort, tmp_path):
    """A row whose ref does not match the genome at pos must be FLAGGED False and KEPT,
    never dropped or moved. This is the core disposition rule for the 13 real mismatches."""
    fa = _write_fasta(tmp_path)
    df = cohort.drop(columns=["kind"]).copy()
    # add a deletion whose ref deliberately does NOT match the genome at its pos
    bad = pd.DataFrame([dict(variant_id="clinvar:1:7:ZZ:Z", chrom="1", pos=7, ref="ZZ", alt="Z")])
    df = pd.concat([df, bad], ignore_index=True)
    recon = B.BuildReconciliation()
    clean, _ = B.build(df, recon)
    # the bad row survives into clean (it is not a bad-ALLELE row; it is a genome mismatch)
    assert (clean["variant_id"] == "clinvar:1:6:ZZ:Z").any() or (clean["ref"] == "ZZ").any()
    flag = B.annotate_genome_consistency(clean, fa)
    # exactly the ZZ row should be False; the genuine rows True
    zz = clean["ref"] == "ZZ"
    assert bool((flag[zz.values] == False).all())  # noqa: E712
    assert int((flag == False).sum()) == int(zz.sum())


def test_flag_na_for_absent_contig(cohort, tmp_path):
    fa = _write_fasta(tmp_path)
    df = cohort.drop(columns=["kind"]).copy()
    # a row on a contig the mini genome does not have -> flag must be <NA>, not False
    off = pd.DataFrame([dict(variant_id="clinvar:99:3:A:C", chrom="99", pos=3, ref="A", alt="C")])
    df = pd.concat([df, off], ignore_index=True)
    recon = B.BuildReconciliation()
    clean, _ = B.build(df, recon)
    flag = B.annotate_genome_consistency(clean, fa)
    off_mask = clean["chrom"].astype(str) == "99"
    assert flag[off_mask.values].isna().all(), "absent contig must be <NA>, never False"


def test_column_na_when_no_genome_end_to_end(cohort, tmp_path):
    inp = tmp_path / "raw.parquet"
    cohort.drop(columns=["kind"]).to_parquet(inp, index=False)
    out = tmp_path / "cohort.parquet"
    r = _run(["--apply", "--input", str(inp), "--output", str(out)], ROOT)
    assert r.returncode == 0, r.stderr
    written = pd.read_parquet(out)
    assert "ref_genome_consistent" in written.columns
    assert written["ref_genome_consistent"].isna().all(), "no-genome build must flag all <NA>"
    assert out.with_name(out.stem + "_reconciliation.json").exists(), \
        "per-build reconciliation JSON must be written"
    recon = json.loads((out.with_name("cohort_build_reconciliation.json")).read_text())
    assert recon["genome_unchecked"] == len(written)
