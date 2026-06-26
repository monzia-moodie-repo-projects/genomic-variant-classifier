"""Tests for the EVE coordinate-triple aa_change fix (eve.py _annotate).

Proves: (1) the variant-side key builds from wt_aa/protein_pos/mut_aa identically
to the lookup side's wt_aa+position+mt_aa key; (2) the HGVSp fallback still works
for protein_change-only cohorts; (3) coordinate-triple wins where present, HGVSp
fills the rest; (4) end-to-end, a coordinate-only cohort (protein_change null) now
gets non-default EVE scores -- the whole-genome Run-17 situation; (5) a cohort with
neither key correctly stays at the default.
"""
from __future__ import annotations

import pandas as pd
import pytest

from genomic_variant_classifier.data.eve import EVEConnector, DEFAULT_SCORE


def _conn():
    return EVEConnector.__new__(EVEConnector)  # bypass __init__; _annotate needs no files


@pytest.fixture
def lookup():
    # As EVEConnector builds it: gene_symbol + aa_change + eve_score.
    return pd.DataFrame({
        "gene_symbol": ["YWHAG", "YWHAG", "TP53"],
        "aa_change":   ["N430S", "Q283P", "R175H"],
        "eve_score":   [0.81, 0.62, 0.93],
    })


def test_coordinate_triple_cohort_scores(lookup):
    """protein_change NULL, wt_aa/protein_pos/mut_aa populated -> EVE matches."""
    cohort = pd.DataFrame({
        "gene_symbol":    ["YWHAG", "YWHAG", "YWHAG", "FOXRED1"],
        "protein_change": [None, None, None, None],
        "wt_aa":          ["N", "Q", "A", None],
        "protein_pos":    [430, 283, 875, pd.NA],
        "mut_aa":         ["S", "P", "T", None],
    })
    out = _conn()._annotate(cohort, lookup)
    assert out.loc[0, "eve_score"] == pytest.approx(0.81)   # N430S
    assert out.loc[1, "eve_score"] == pytest.approx(0.62)   # Q283P
    assert out.loc[2, "eve_score"] == pytest.approx(DEFAULT_SCORE)  # A875T not in lookup
    assert out.loc[3, "eve_score"] == pytest.approx(DEFAULT_SCORE)  # no triple
    assert int((out["eve_score"] != DEFAULT_SCORE).sum()) == 2


def test_hgvsp_fallback_still_works(lookup):
    """protein_change populated, no triple cols -> HGVSp parse must still match."""
    cohort = pd.DataFrame({
        "gene_symbol":    ["TP53"],
        "protein_change": ["p.Arg175His"],
    })
    out = _conn()._annotate(cohort, lookup)
    assert out.loc[0, "eve_score"] == pytest.approx(0.93)


def test_triple_wins_hgvsp_fills_rest(lookup):
    """Mixed cohort: triple where present, HGVSp where not."""
    cohort = pd.DataFrame({
        "gene_symbol":    ["YWHAG", "TP53"],
        "protein_change": [None, "p.Arg175His"],
        "wt_aa":          ["N", None],
        "protein_pos":    [430, pd.NA],
        "mut_aa":         ["S", None],
    })
    out = _conn()._annotate(cohort, lookup)
    assert out.loc[0, "eve_score"] == pytest.approx(0.81)  # N430S from triple
    assert out.loc[1, "eve_score"] == pytest.approx(0.93)  # R175H from HGVSp


def test_no_key_cohort_stays_default(lookup):
    """Neither triple nor protein_change -> default everywhere (no spurious matches)."""
    cohort = pd.DataFrame({
        "gene_symbol":    ["YWHAG"],
        "protein_change": [None],
    })
    out = _conn()._annotate(cohort, lookup)
    assert int((out["eve_score"] != DEFAULT_SCORE).sum()) == 0


def test_triple_key_matches_lookup_format(lookup):
    """The triple key must equal the lookup's wt_aa+position+mt_aa exactly."""
    # YWHAG N430S is in the lookup; a cohort row with that triple must match.
    cohort = pd.DataFrame({
        "gene_symbol":    ["YWHAG"],
        "protein_change": [None],
        "wt_aa":          ["N"],
        "protein_pos":    [430],
        "mut_aa":         ["S"],
    })
    out = _conn()._annotate(cohort, lookup)
    assert out.loc[0, "eve_score"] == pytest.approx(0.81)


def test_float_protein_pos_coerced(lookup):
    """protein_pos may arrive as float (430.0) -> int(430) -> 'N430S', not 'N430.0S'."""
    cohort = pd.DataFrame({
        "gene_symbol":    ["YWHAG"],
        "protein_change": [None],
        "wt_aa":          ["N"],
        "protein_pos":    [430.0],
        "mut_aa":         ["S"],
    })
    out = _conn()._annotate(cohort, lookup)
    assert out.loc[0, "eve_score"] == pytest.approx(0.81)


def test_whitespace_in_aa_stripped(lookup):
    """wt_aa/mut_aa with stray whitespace must still build a clean key."""
    cohort = pd.DataFrame({
        "gene_symbol":    ["YWHAG"],
        "protein_change": [None],
        "wt_aa":          [" N "],
        "protein_pos":    [430],
        "mut_aa":         [" S "],
    })
    out = _conn()._annotate(cohort, lookup)
    assert out.loc[0, "eve_score"] == pytest.approx(0.81)
