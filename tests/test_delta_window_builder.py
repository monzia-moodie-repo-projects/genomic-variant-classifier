"""Tests for the core delta-window builder (sequence CNN input construction)."""
import random

from genomic_variant_classifier.data.delta_window_builder import build_window, POLY

W = 101
HALF = 50


def _genome(n=200, seed=0):
    rng = random.Random(seed)
    return {"1": "".join(rng.choice("ACGT") for _ in range(n))}


def _fetch_for(genome):
    def fetch(contig, start0, length):
        if contig not in genome or start0 < 0:
            return None
        return genome[contig][start0:start0 + length]
    return fetch


def test_snv_center_matches_and_window_equals_genome():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    pos = 100; ref = seq[99]; alt = "A" if ref != "A" else "C"
    r = build_window(fetch, "1", pos, ref, alt, W)
    assert r.ok
    assert len(r.ref_window) == W and len(r.alt_window) == W
    assert r.ref_window == seq[49:150]
    assert r.ref_window[HALF] == ref and r.alt_window[HALF] == alt
    assert r.alt_window == seq[49:99] + alt + seq[100:150]


def test_ref_mismatch_falls_back_with_reason():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    wrong = "A" if seq[99] != "A" else "G"
    r = build_window(fetch, "1", 100, wrong, "T", W)
    assert not r.ok and "ref_mismatch" in r.reason and r.ref_window == POLY * W


def test_contig_start_edge_left_pads():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    r = build_window(fetch, "1", 3, seq[2], "A" if seq[2] != "A" else "C", W)
    assert r.ok and len(r.ref_window) == W and r.ref_window.startswith("N")
    assert r.ref_window[HALF] == seq[2]


def test_contig_end_edge_right_pads():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    r = build_window(fetch, "1", 199, seq[198], "A" if seq[198] != "A" else "C", W)
    assert r.ok and len(r.ref_window) == W and r.ref_window.endswith("N")


def test_deletion_maintains_length():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    r = build_window(fetch, "1", 100, seq[99:101], seq[99], W)
    assert r.ok and len(r.ref_window) == W and len(r.alt_window) == W
    assert r.ref_window[HALF:HALF + 2] == seq[99:101]


def test_insertion_maintains_length():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    r = build_window(fetch, "1", 100, seq[99], seq[99] + "GG", W)
    assert r.ok and len(r.ref_window) == W and len(r.alt_window) == W


def test_non_acgt_and_fetch_fail_and_bad_pos_flagged():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    assert build_window(fetch, "1", 100, "-", "T", W).reason == "non_acgt_allele"
    assert build_window(fetch, "ZZ", 100, seq[99], "T", W).reason == "fetch_failed"
    assert build_window(fetch, "1", "x", seq[99], "T", W).reason == "bad_pos"


def test_all_windows_exactly_101():
    g = _genome(300, 1); fetch = _fetch_for(g); seq = g["1"]
    for pos in range(1, 301, 7):
        ref = seq[pos - 1]; alt = "A" if ref != "A" else "C"
        r = build_window(fetch, "1", pos, ref, alt, W)
        assert len(r.ref_window) == W and len(r.alt_window) == W


def test_insertion_off0_builds():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    r = build_window(fetch, "1", 100, seq[99], seq[99] + "GG", W)
    assert r.ok and len(r.ref_window) == W and len(r.alt_window) == W


def test_deletion_off_minus_1_builds():
    # cohort convention: for a deletion, ref starts one base LEFT of (pos-1), i.e. at pos-2.
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    ref_del = seq[98:101]  # 0-based 98,99,100 == pos-2 for pos=100
    alt_del = seq[98]
    r = build_window(fetch, "1", 100, ref_del, alt_del, W)
    assert r.ok and len(r.ref_window) == W and len(r.alt_window) == W


def test_deletion_homopolymer_both_offsets_match():
    genome = {"2": "C" * 40 + "AAAAAA" + "C" * 154}

    def fetch(contig, start0, length):
        if contig not in genome or start0 < 0:
            return None
        return genome[contig][start0:start0 + length]
    r = build_window(fetch, "2", 42, "AAAA", "A", W)
    assert r.ok and len(r.ref_window) == W and len(r.alt_window) == W


def test_mnv_off0_builds():
    g = _genome(); fetch = _fetch_for(g); seq = g["1"]
    ref_mnv = seq[99:101]
    alt_mnv = "".join("A" if b != "A" else "T" for b in ref_mnv)
    r = build_window(fetch, "1", 100, ref_mnv, alt_mnv, W)
    assert r.ok and len(r.ref_window) == W and len(r.alt_window) == W


def test_indel_no_offset_matches_falls_back():
    g = _genome(); fetch = _fetch_for(g)
    # a ref that (almost surely) matches neither offset in this fixed-seed genome
    r = build_window(fetch, "1", 100, "TTTTTTTT", "T", W)
    # either it validly matched by luck (ok) or fell back with a reason -- never silent/garbage
    assert r.ok or ("ref_mismatch" in r.reason)
    if not r.ok:
        assert r.ref_window == POLY * W
