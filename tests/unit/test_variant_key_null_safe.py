"""Version-stability tests for the variant-key construction.

The asserted key strings are byte-identical under pandas 2.x and 3.x (verified
against both), so these tests are themselves the regression guard for the
eventual pandas 3.0 migration: bare astype(str) would NaN-poison the null-allele
key on pandas 3.x and fail test_null_alleles_use_sentinel_not_nan.
Author: Monzia Moodie
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from genomic_variant_classifier.data.real_data_prep import _variant_key


def _frame():
    return pd.DataFrame({
        "chrom": ["17", "17", "X"],
        "pos":   [7577121, 43044295, 100],
        "ref":   ["G", None, "C"],
        "alt":   ["A", np.nan, "T"],
    })


def test_non_null_rows():
    k = _variant_key(_frame())
    assert k.iloc[0] == "17:7577121:G:A"
    assert k.iloc[2] == "X:100:C:T"


def test_null_alleles_use_sentinel_not_nan():
    k = _variant_key(_frame())
    assert k.iloc[1] == "17:43044295::"   # NOT NaN, NOT '...:None:None'
    assert k.notna().all()                # no NaN-poisoned keys on any pandas


def test_none_and_nan_unify():
    df = pd.DataFrame({"chrom": ["1", "1"], "pos": [1, 1],
                       "ref": [None, np.nan], "alt": [None, np.nan]})
    k = _variant_key(df)
    assert k.iloc[0] == k.iloc[1] == "1:1::"


def test_custom_missing_token():
    df = pd.DataFrame({"chrom": ["1"], "pos": [1], "ref": [None], "alt": ["A"]})
    assert _variant_key(df, missing=".").iloc[0] == "1:1:.:A"
