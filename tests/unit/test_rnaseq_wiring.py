"""tests/unit/test_rnaseq_wiring.py -- Monzia Moodie

Lockstep wiring for the rnaseq_* feature family (the engineer_features /
TABULAR_FEATURES analogue of test_reactome_wiring). Passes only after the
wiring lands; both feature builders must stay column-for-column identical.
"""
from __future__ import annotations
import pandas as pd
from genomic_variant_classifier.models.variant_ensemble import (
    TABULAR_FEATURES, engineer_features,
)

RNASEQ = ["rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv",
          "rnaseq_log2fc", "rnaseq_de_neglog10p"]


def _df(**ov):
    base = dict(gene_symbol=["TP53"], consequence=["missense_variant"],
                allele_freq=[0.001], ref=["G"], alt=["T"])
    base.update({k: [v] for k, v in ov.items()})
    return pd.DataFrame(base)


def test_rnaseq_family_in_tabular_features():
    for c in RNASEQ:
        assert c in TABULAR_FEATURES, c
    assert TABULAR_FEATURES[-5:] == RNASEQ  # appended as the last family


def test_engineer_features_default_zero_when_absent():
    feats = engineer_features(_df())
    for c in RNASEQ:
        assert feats.loc[0, c] == 0.0
        assert not feats[c].isnull().any()


def test_engineer_features_real_values_pass_through():
    feats = engineer_features(_df(rnaseq_mean_log_tpm=3.5, rnaseq_de_neglog10p=4.2))
    assert feats.loc[0, "rnaseq_mean_log_tpm"] == 3.5
    assert feats.loc[0, "rnaseq_de_neglog10p"] == 4.2


def test_columns_match_tabular_exactly():
    feats = engineer_features(_df())
    assert list(feats.columns) == TABULAR_FEATURES
