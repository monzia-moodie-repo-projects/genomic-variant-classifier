"""Unit tests for the shared gene-symbol normalization helper.

Guards the multi-gene ';' split and -- critically -- the rule that '-' is NEVER
a split delimiter (HLA-A, NKX2-1, readthrough fusions are single symbols).
"""

from __future__ import annotations

import pytest

from genomic_variant_classifier.data.gene_symbols import (
    gene_symbol_candidates,
    normalize_gene_symbol,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  brca1 ", "BRCA1"),
        ("BRCA1", "BRCA1"),
        ("MYH11;NDE1", "MYH11;NDE1"),  # normalize does NOT split
        (None, ""),
        ("nan", ""),
        ("None", ""),
        ("<NA>", ""),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize_gene_symbol(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BRCA1", ["BRCA1"]),
        ("  myh11;nde1 ", ["MYH11;NDE1", "MYH11", "NDE1"]),
        ("ECE2;EEF1AKMT4-ECE2", ["ECE2;EEF1AKMT4-ECE2", "ECE2", "EEF1AKMT4-ECE2"]),
        ("CRIPAK;LOC126806945;UVSSA",
         ["CRIPAK;LOC126806945;UVSSA", "CRIPAK", "LOC126806945", "UVSSA"]),
        ("A;A", ["A;A", "A"]),            # de-duplicated
        ("GENE;;X", ["GENE;;X", "GENE", "X"]),  # empty middle component dropped
        ("", []),
        (None, []),
        ("nan", []),
    ],
)
def test_candidates(raw, expected):
    assert gene_symbol_candidates(raw) == expected


@pytest.mark.parametrize("raw", ["HLA-A", "HLA-DRB1", "NKX2-1", "JMJD7-PLA2G4B",
                                 "ATP5MF-PTCD1", "ARPC4-TTLL3", "X-Y-Z"])
def test_hyphen_is_never_a_split_delimiter(raw):
    """Hyphenated symbols must yield exactly one candidate (themselves)."""
    assert gene_symbol_candidates(raw) == [normalize_gene_symbol(raw)]


def test_full_symbol_is_always_first():
    for raw in ["MYH11;NDE1", "ECE2;EEF1AKMT4-ECE2", "BRCA1"]:
        assert gene_symbol_candidates(raw)[0] == normalize_gene_symbol(raw)
