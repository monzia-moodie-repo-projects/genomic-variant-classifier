"""Regression test: _assert_clean_cohort must tolerate inputs without a variant_id column.

Commit 1720c0a's guard ran df["variant_id"].duplicated() inside _load_and_label, but raw ClinVar /
tiny fixtures only carry chrom/pos/ref/alt there (variant_id is built later), so it KeyError'd and
broke the LOVD tests. The guard now derives the duplicate-identity key from the locus when
variant_id is absent. These cases lock that behaviour.
"""
from __future__ import annotations

import pandas as pd
import pytest

from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline

_guard = DataPrepPipeline._assert_clean_cohort  # staticmethod -> plain callable


def test_passes_without_variant_id_when_locus_unique():
    df = pd.DataFrame(
        [
            {"chrom": "17", "pos": 7675234, "ref": "G", "alt": "T"},
            {"chrom": "17", "pos": 7675088, "ref": "C", "alt": "A"},
        ]
    )
    _guard(df, "fixture-no-variant-id")  # must NOT raise (this was the regression)


def test_raises_on_duplicate_locus_without_variant_id():
    df = pd.DataFrame(
        [
            {"chrom": "17", "pos": 7675234, "ref": "G", "alt": "T"},
            {"chrom": "17", "pos": 7675234, "ref": "G", "alt": "T"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate variant identity"):
        _guard(df, "fixture-dup-locus")


def test_raises_on_null_allele():
    df = pd.DataFrame([{"chrom": "17", "pos": 7675234, "ref": "G", "alt": None}])
    with pytest.raises(ValueError, match="null/empty"):
        _guard(df, "fixture-null-allele")


def test_still_catches_duplicate_variant_id_when_present():
    df = pd.DataFrame(
        [
            {"chrom": "17", "pos": 1, "ref": "G", "alt": "T", "variant_id": "17:1:G:T"},
            {"chrom": "17", "pos": 1, "ref": "G", "alt": "T", "variant_id": "17:1:G:T"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate variant identity"):
        _guard(df, "fixture-dup-variant-id")
