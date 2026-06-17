"""
tests/unit/test_reactome_wiring.py
==================================
Lockstep wiring tests for reactome_pathway_count (the engineer_features /
TABULAR_FEATURES analogues of test_dbsnp.py tests 6-8). These pass only AFTER
both wiring patchers are applied; they ship with the wiring step so the two
feature builders are verified to stay column-for-column identical.
"""
from __future__ import annotations

import pandas as pd

from genomic_variant_classifier.models.variant_ensemble import (
    TABULAR_FEATURES,
    engineer_features,
)


def _engineer_df(**overrides) -> pd.DataFrame:
    base = dict(
        gene_symbol=["TP53"],
        consequence=["missense_variant"],
        allele_freq=[0.001],
        ref=["G"],
        alt=["T"],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def test_reactome_pathway_count_in_tabular_features():
    assert "reactome_pathway_count" in TABULAR_FEATURES


def test_engineer_features_reactome_default_zero_when_absent():
    df = _engineer_df()
    assert "reactome_pathway_count" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "reactome_pathway_count"] == 0
    assert not feats["reactome_pathway_count"].isnull().any()


def test_engineer_features_reactome_real_value_passes_through():
    df = _engineer_df(reactome_pathway_count=42)
    feats = engineer_features(df)
    assert feats.loc[0, "reactome_pathway_count"] == 42


def test_reactome_is_last_feature_and_columns_match_tabular():
    # Both builders append reactome last; lock the contract: engineer_features
    # output equals TABULAR_FEATURES exactly (set AND order).
    # reactome was last until the rnaseq family (Phase D) was appended after it
    assert "reactome_pathway_count" in TABULAR_FEATURES
    assert TABULAR_FEATURES[-1] == "rnaseq_de_neglog10p"
    feats = engineer_features(_engineer_df())
    assert list(feats.columns) == TABULAR_FEATURES
