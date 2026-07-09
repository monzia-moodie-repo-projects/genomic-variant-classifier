"""
tests/test_build_cohort_v2.py  (2026-07-08)
=========================================
Tests for scripts/build_cohort_v2.py.

The central test reproduces the exact 30/30-at-pos-1 signature the probe found: padded
deletions must move by one, every other representation must not. The reference-guard tests
build a tiny synthetic genome and assert the guard PASSES on correct coordinates and HARD
FAILS on wrong ones -- a guard that cannot fail is not a guard.

Run:  python -m pytest tests/test_build_cohort_v2.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_cohort_v2", Path(__file__).resolve().parents[1] / "scripts" / "build_cohort_v2.py"
)
b = importlib.util.module_from_spec(_SPEC)
sys.modules["build_cohort_v2"] = b
_SPEC.loader.exec_module(b)


def _cohort() -> pd.DataFrame:
    """One of each representation, with a known-correct VCF position baked in.

    For padded deletions the cohort `pos` is Start == PositionVCF + 1, so the CORRECT
    (VCF) position is pos - 1. For everything else the cohort pos already equals the VCF
    position.
    """
    rows = [
        # variant_id (as v1 wrote it, with the WRONG pos for padded dels), chrom, pos, ref, alt
        ("clinvar:7:4787730:GCTGCTGGACCTGCC:G", "7", 4787730, "GCTGCTGGACCTGCC", "G"),  # padded del
        ("clinvar:2:19931375:AG:A",             "2", 19931375, "AG", "A"),               # padded del
        ("clinvar:17:43076593:ACTT:A",          "17", 43076593, "ACTT", "A"),            # padded del
        ("clinvar:1:100:A:G",                   "1", 100, "A", "G"),                     # SNV
        ("clinvar:1:200:G:GAAG",                "1", 200, "G", "GAAG"),                  # padded ins
        ("clinvar:1:300:AA:C",                  "1", 300, "AA", "C"),                    # delins (shrinks!)
        ("clinvar:1:400:GGAT:TTTTT",            "1", 400, "GGAT", "TTTTT"),              # MNV/other
    ]
    return pd.DataFrame(rows, columns=["variant_id", "chrom", "pos", "ref", "alt"])


# ---------------------------------------------------------------------------
# 1. The correction is exact
# ---------------------------------------------------------------------------
def test_only_padded_deletions_shift():
    df = _cohort()
    out, recon = b.correct_coordinates(df)
    orig = dict(zip(df["variant_id"], df["pos"]))
    by_alleles = {(r, a): (po, pn) for r, a, po, pn
                  in zip(out["ref"], out["alt"], df["pos"], out["pos"])}
    # padded deletions: -1
    assert by_alleles[("GCTGCTGGACCTGCC", "G")][1] == 4787729
    assert by_alleles[("AG", "A")][1] == 19931374
    assert by_alleles[("ACTT", "A")][1] == 43076592
    # everything else: unchanged
    assert by_alleles[("A", "G")][1] == 100          # SNV
    assert by_alleles[("G", "GAAG")][1] == 200        # insertion
    assert by_alleles[("AA", "C")][1] == 300          # delins -- NOT shifted
    assert by_alleles[("GGAT", "TTTTT")][1] == 400    # MNV
    assert recon.n_padded_deletions_corrected == 3
    assert recon.n_unchanged == 4
    assert recon.identity_holds()


def test_delins_that_shrinks_is_not_shifted():
    """AA>C has len(alt) < len(ref) but is NOT a padded deletion (C is not a prefix of AA)."""
    df = pd.DataFrame({"variant_id": ["clinvar:1:300:AA:C"], "chrom": ["1"],
                       "pos": [300], "ref": ["AA"], "alt": ["C"]})
    assert not b.is_padded_deletion(df["ref"], df["alt"]).iloc[0]
    out, recon = b.correct_coordinates(df)
    assert out["pos"].iloc[0] == 300
    assert recon.n_padded_deletions_corrected == 0


def test_variant_id_is_rebuilt_from_corrected_pos():
    df = _cohort()
    out, _ = b.correct_coordinates(df)
    vid = dict(zip(out["ref"] + ">" + out["alt"], out["variant_id"]))
    assert vid["GCTGCTGGACCTGCC>G"] == "clinvar:7:4787729:GCTGCTGGACCTGCC:G"
    assert vid["A>G"] == "clinvar:1:100:A:G"          # SNV variant_id unchanged


def test_composition_is_invariant():
    df = _cohort()
    _, recon = b.correct_coordinates(df)
    assert recon.composition_before == recon.composition_after


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Required columns missing"):
        b.correct_coordinates(_cohort().drop(columns=["ref"]))


# ---------------------------------------------------------------------------
# 2. The reference-consistency guard PASSES on correct coords, FAILS on wrong
# ---------------------------------------------------------------------------
def _write_genome(tmp_path: Path) -> Path:
    """A tiny 2-contig FASTA. Requires pyfaidx (or pysam) to index."""
    fa = tmp_path / "mini.fa"
    # chr7: put GCTGCTGGACCTGCC starting at 1-based position 4787729 is impractical in a tiny
    # file, so we use small coordinates and rewrite the cohort to match.
    seq1 = "N" * 9 + "GCTGCTGGACCTGCC" + "N" * 10   # ref starts at 1-based pos 10
    fa.write_text(f">1\n{seq1}\n>2\nNNNNNNNNNAGNNNN\n")
    return fa


@pytest.mark.skipif(
    importlib.util.find_spec("pyfaidx") is None and importlib.util.find_spec("pysam") is None,
    reason="needs pyfaidx or pysam",
)
def test_reference_guard_passes_on_correct_coordinates(tmp_path):
    genome = _write_genome(tmp_path)
    # corrected pos must be 10 (1-based) so genome[9:24] == the ref
    df = pd.DataFrame({"variant_id": ["clinvar:1:11:GCTGCTGGACCTGCC:G"], "chrom": ["1"],
                       "pos": [11], "ref": ["GCTGCTGGACCTGCC"], "alt": ["G"]})  # cohort pos 11 -> corrected 10
    out, recon = b.correct_coordinates(df)
    assert out["pos"].iloc[0] == 10
    b.reference_check(out, genome, recon)
    assert recon.reference_check.startswith("PASSED")
    assert recon.reference_mismatches == 0


@pytest.mark.skipif(
    importlib.util.find_spec("pyfaidx") is None and importlib.util.find_spec("pysam") is None,
    reason="needs pyfaidx or pysam",
)
def test_reference_guard_hard_fails_on_wrong_coordinates(tmp_path):
    genome = _write_genome(tmp_path)
    # cohort pos 99 -> corrected 98, where the genome is N's, not the ref -> must raise
    df = pd.DataFrame({"variant_id": ["clinvar:1:99:GCTGCTGGACCTGCC:G"], "chrom": ["1"],
                       "pos": [99], "ref": ["GCTGCTGGACCTGCC"], "alt": ["G"]})
    out, recon = b.correct_coordinates(df)
    with pytest.raises(ValueError, match="REFERENCE-CONSISTENCY GUARD FAILED"):
        b.reference_check(out, genome, recon)
    assert recon.reference_mismatches >= 1


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def test_variant_class_splits_delins_from_padded():
    vc = b.variant_class(
        pd.Series(["GCTG", "AA", "G", "A", "A"]),
        pd.Series(["G", "C", "GAAG", "CG", "T"]),
    )
    assert list(vc) == ["padded_deletion", "delins_del", "padded_insertion", "delins_ins", "SNV"]


def test_schema_fingerprint_order_independent():
    assert b.schema_fingerprint(["b", "a"]) == b.schema_fingerprint(["a", "b"])
