"""Unit tests for seq_windows (delta-CNN window extraction).

Assertions are property-based (centering + splice invariants), not a re-derivation
of the function's own slicing, so they genuinely pin behavior. A final test opens
a real tiny FASTA via pyfaidx and confirms the adapter agrees with the dict path.
"""

from __future__ import annotations

import textwrap

import pytest

from genomic_variant_classifier.data import seq_windows as sw

# Deterministic 120-bp reference. Index i (0-based) -> known base.
REF_SEQ = ("ACGT" * 30)  # 120 bp, repeating ACGT
REF = {"c1": REF_SEQ, "c2": "ACGT" * 5}  # c2 is short (20 bp) for edge tests
W = 101
H = W // 2  # 50


def test_window_lengths_always_101():
    rw, aw = sw.build_delta_windows(REF, "c1", 60, "G", "A")
    assert len(rw) == W and len(aw) == W


def test_ref_window_centers_variant_base():
    pos = 60  # 1-based
    rw = sw.extract_ref_window(REF, "c1", pos)
    # index HALF holds reference[pos-1]; neighbors hold pos-2 and pos
    assert rw[H] == REF_SEQ[pos - 1]
    assert rw[H - 1] == REF_SEQ[pos - 2]
    assert rw[H + 1] == REF_SEQ[pos]


def test_snv_alt_differs_by_exactly_one_base():
    pos = 60
    ref_base = REF_SEQ[pos - 1]
    alt_base = "A" if ref_base != "A" else "C"
    rw, aw = sw.build_delta_windows(REF, "c1", pos, ref_base, alt_base)
    assert aw[:H] == rw[:H]          # upstream identical
    assert aw[H] == alt_base         # center swapped
    assert aw[H + 1:] == rw[H + 1:]  # downstream identical
    assert sum(a != b for a, b in zip(rw, aw)) == 1


def test_insertion_pushes_downstream_right():
    pos = 60
    ref_base = REF_SEQ[pos - 1]
    alt = ref_base + "TTT"  # insertion of TTT after the ref base context
    rw, aw = sw.build_delta_windows(REF, "c1", pos, ref_base, alt)
    assert aw[:H] == rw[:H]              # upstream identical
    assert aw[H:H + len(alt)] == alt     # alt occupies center onward
    assert len(aw) == W


def test_deletion_pulls_downstream_left():
    pos = 60
    # delete 3 ref bases starting at pos, keep first as anchor
    ref_allele = REF_SEQ[pos - 1: pos - 1 + 3]
    alt = ref_allele[0]
    rw, aw = sw.build_delta_windows(REF, "c1", pos, ref_allele, alt)
    assert aw[:H] == rw[:H]
    assert aw[H] == alt[0]
    # base after the anchor should be the reference base 3 positions downstream
    assert aw[H + 1] == REF_SEQ[pos - 1 + 3]
    assert len(aw) == W


def test_edge_near_contig_start_pads_left():
    pos = 3  # near start: left_pad = 50 - (pos-1) = 48
    rw = sw.extract_ref_window(REF, "c1", pos)
    assert rw[:48] == sw.PAD_CHAR * 48
    assert rw[48] == REF_SEQ[0]
    assert rw[H] == REF_SEQ[pos - 1]
    assert len(rw) == W


def test_short_contig_pads_right():
    # c2 is 20 bp; a centered 101-window is mostly pad
    rw = sw.extract_ref_window(REF, "c2", 10)
    assert len(rw) == W
    assert rw[H] == REF["c2"][9]


def test_missing_contig_returns_polyA_zero_delta():
    rw, aw = sw.build_delta_windows(REF, "Un", 100, "G", "A")
    assert rw == sw.PAD_CHAR * W
    assert aw == sw.PAD_CHAR * W
    assert rw == aw  # delta exactly zero


def test_ref_matches_true_and_false():
    pos = 60
    assert sw.ref_matches(REF, "c1", pos, REF_SEQ[pos - 1]) is True
    wrong = "A" if REF_SEQ[pos - 1] != "A" else "C"
    assert sw.ref_matches(REF, "c1", pos, wrong) is False
    assert sw.ref_matches(REF, "Un", pos, "G") is None


def test_multibase_ref_match():
    pos = 60
    triplet = REF_SEQ[pos - 1: pos - 1 + 3]
    assert sw.ref_matches(REF, "c1", pos, triplet) is True


def test_pyfaidx_adapter_agrees_with_dict(tmp_path):
    pytest.importorskip("pyfaidx")
    fa = tmp_path / "tiny.fa"
    body = textwrap.fill(REF_SEQ, width=60)
    fa.write_text(">c1\n" + body + "\n")
    ref_fa = sw.open_reference(fa)
    pos = 60
    rw_dict = sw.extract_ref_window(REF, "c1", pos)
    rw_fa = sw.extract_ref_window(ref_fa, "c1", pos)
    assert rw_fa == rw_dict


import random as _r

_rng = _r.Random(12345)
RREF = {"r1": "".join(_rng.choice("ACGT") for _ in range(400))}  # non-periodic


def test_find_anchor_exact_single_base():
    pos = 60
    assert sw.find_anchor(RREF, "r1", pos, RREF["r1"][pos - 1]) == pos


def test_find_anchor_deletion_resolves_minus_one():
    # 8-base ref (unique in random DNA) that actually sits at pos-1
    pos = 60
    ref = RREF["r1"][(pos - 1) - 1: (pos - 1) - 1 + 8]
    assert sw.find_anchor(RREF, "r1", pos, ref) == pos - 1


def test_find_anchor_single_base_mismatch_returns_none():
    pos = 60
    wrong = "A" if RREF["r1"][pos - 1] != "A" else "C"
    assert sw.find_anchor(RREF, "r1", pos, wrong) is None  # no ambiguous search for L==1


def test_find_anchor_absent_multibase_returns_none():
    # an 8-mer taken from far away (~pos 300) is not within +/-3 of pos 60
    pos = 60
    far = RREF["r1"][300: 308]
    assert sw.find_anchor(RREF, "r1", pos, far, max_shift=3) is None


def test_find_anchor_respects_max_shift():
    pos = 60
    ref = RREF["r1"][(pos - 1) - 4: (pos - 1) - 4 + 8]  # sits at delta -4
    assert sw.find_anchor(RREF, "r1", pos, ref, max_shift=3) is None
    assert sw.find_anchor(RREF, "r1", pos, ref, max_shift=4) == pos - 4
