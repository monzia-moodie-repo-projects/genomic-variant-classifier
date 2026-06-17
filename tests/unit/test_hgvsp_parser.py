"""Tests for genomic_variant_classifier.data.hgvsp_parser."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from genomic_variant_classifier.data.hgvsp_parser import (
    add_protein_columns,
    parse_am_protein_variant,
    parse_hgvsp,
)

NONE3 = (None, None, None)


@pytest.mark.parametrize(
    "s, expected",
    [
        # --- valid missense ---
        ("p.Asp1692Asn", (1692, "D", "N")),
        ("p.Arg1699Gln", (1699, "R", "Q")),
        ("p.(Arg1699Gln)", (1699, "R", "Q")),                 # parenthesised
        ("NP_009225.1:p.Asp1692Asn", (1692, "D", "N")),        # accession prefix
        ("p.D1692N", (1692, "D", "N")),                        # 1-letter
        ("p.Met1Val", (1, "M", "V")),                          # start-codon missense
        ("  p.Cys61Gly  ", (61, "C", "G")),                    # whitespace
        # --- rejected: synonymous ---
        ("p.Asp1692Asp", NONE3),                               # wt == mut (3-letter)
        ("p.D1692D", NONE3),                                   # wt == mut (1-letter)
        ("p.Arg1699=", NONE3),                                 # explicit synonymous
        # --- rejected: nonsense / stop ---
        ("p.Arg1699Ter", NONE3),
        ("p.Arg1699*", NONE3),
        # --- rejected: frameshift / indel / ext / start-loss ---
        ("p.Arg1699GlnfsTer12", NONE3),
        ("p.Lys23_Val25del", NONE3),
        ("p.Ala767_Val769dup", NONE3),
        ("p.Met1?", NONE3),
        ("p.Ter807GlnextTer?", NONE3),
        # --- rejected: not a protein change / malformed ---
        ("c.5074G>A", NONE3),
        ("p.Xaa12Tyr", NONE3),                                 # unknown wt code
        ("p.Asp12Xyz", NONE3),                                 # unknown mut code
        ("p.Ala0Val", NONE3),                                  # invalid position < 1
        ("", NONE3),
        (".", NONE3),
        ("nan", NONE3),
        (None, NONE3),
        (float("nan"), NONE3),
    ],
)
def test_parse_hgvsp(s, expected):
    assert parse_hgvsp(s) == expected


def test_add_protein_columns_basic():
    df = pd.DataFrame(
        {"protein_change": ["p.Asp1692Asn", "p.Arg1699Ter", None, "p.Cys61Gly"]}
    )
    out = add_protein_columns(df.copy())
    wt, mut = list(out["wt_aa"]), list(out["mut_aa"])
    assert wt[0] == "D" and wt[3] == "C" and pd.isna(wt[1]) and pd.isna(wt[2])
    assert mut[0] == "N" and mut[3] == "G" and pd.isna(mut[1]) and pd.isna(mut[2])
    pos = out["protein_pos"]
    assert str(pos.dtype) == "Int64"
    assert pos.iloc[0] == 1692 and pos.iloc[3] == 61
    assert pd.isna(pos.iloc[1]) and pd.isna(pos.iloc[2])


def test_add_protein_columns_missing_source():
    df = pd.DataFrame({"gene_symbol": ["BRCA1", "TP53"]})
    out = add_protein_columns(df.copy())
    assert {"protein_pos", "wt_aa", "mut_aa"}.issubset(out.columns)
    assert out["wt_aa"].isna().all() and out["protein_pos"].isna().all()


def test_missense_fraction_is_nonzero_on_realistic_sample():
    # Guards against a regex regression that would re-zero everything.
    sample = ["p.Asp1692Asn", "p.Arg1699Gln", "p.Gly12Val", "p.Arg1699Ter", None]
    parsed = [parse_hgvsp(s) for s in sample]
    n_missense = sum(1 for p in parsed if p[0] is not None)
    assert n_missense == 3


@pytest.mark.parametrize(
    "s, expected",
    [
        ("V123M", (123, "V", "M")),
        ("p.V123M", (123, "V", "M")),
        ("Q9Y6K9:V123M", (123, "V", "M")),         # uniprot-prefixed
        ("M1V", (1, "M", "V")),
        ("  A2G  ", (2, "A", "G")),
        ("V123V", NONE3),                          # synonymous
        ("V0M", NONE3),                            # invalid position
        ("X12Y", NONE3),                           # non-standard residue
        ("p.Asp1692Asn", NONE3),                   # 3-letter is NOT AM format
        ("c.5074G>A", NONE3),
        ("", NONE3),
        (None, NONE3),
        (float("nan"), NONE3),
    ],
)
def test_parse_am_protein_variant(s, expected):
    assert parse_am_protein_variant(s) == expected


def test_fill_protein_columns_from_hgvsp_na_only_and_preserves():
    import pandas as pd
    from genomic_variant_classifier.data.hgvsp_parser import fill_protein_columns_from_hgvsp
    df = pd.DataFrame({
        "is_missense": [1, 1, 1, 1, 0],
        "protein_change": ["p.Asp1692Asn", "p.Arg1699Gln", "p.Arg175Ter", None, "p.Gly12Val"],
        "protein_pos": pd.array([100, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Int64"),
        "wt_aa": ["Z", None, None, None, None],
        "mut_aa": ["Z", None, None, None, None],
    })
    out, n = fill_protein_columns_from_hgvsp(df.copy())
    assert n == 1                                       # only the AM-missing missense row fills
    assert out.loc[0, "protein_pos"] == 100 and out.loc[0, "wt_aa"] == "Z"   # AM row preserved
    assert out.loc[1, "protein_pos"] == 1699 and out.loc[1, "wt_aa"] == "R" and out.loc[1, "mut_aa"] == "Q"
    assert pd.isna(out.loc[2, "protein_pos"])           # nonsense rejected
    assert pd.isna(out.loc[3, "protein_pos"])           # None source
    assert pd.isna(out.loc[4, "protein_pos"])           # non-missense skipped


def test_fill_protein_columns_from_hgvsp_missing_source_col():
    import pandas as pd
    from genomic_variant_classifier.data.hgvsp_parser import fill_protein_columns_from_hgvsp
    df = pd.DataFrame({"is_missense": [1, 1]})
    out, n = fill_protein_columns_from_hgvsp(df.copy())
    assert n == 0 and "protein_pos" in out.columns and out["protein_pos"].isna().all()


def test_fill_protein_columns_from_hgvsp_no_is_missense_col():
    import pandas as pd
    from genomic_variant_classifier.data.hgvsp_parser import fill_protein_columns_from_hgvsp
    df = pd.DataFrame({
        "protein_change": ["p.Asp1692Asn", "p.Asp1692Asp"],   # 2nd synonymous
        "protein_pos": pd.array([pd.NA, pd.NA], dtype="Int64"),
    })
    out, n = fill_protein_columns_from_hgvsp(df.copy())
    assert n == 1 and out.loc[0, "protein_pos"] == 1692 and pd.isna(out.loc[1, "protein_pos"])
