"""Feature-count contract: the single deliberate-bump tripwire.

EXPECTED_TABULAR_FEATURE_COUNT (in variant_ensemble.py) is the one count literal a
human edits when adding/removing a tabular feature. These tests enforce that the
actual TABULAR_FEATURES list and the derived INFERENCE_FEATURE_COLUMNS both agree
with it. If you edit the feature list but forget to bump the constant (or vice
versa), exactly one of these fails with a pointed message -- replacing the scatter
of hardcoded `== <N>` guards that previously broke on every feature addition.
"""
from __future__ import annotations

from genomic_variant_classifier.models.variant_ensemble import (
    TABULAR_FEATURES,
    EXPECTED_TABULAR_FEATURE_COUNT,
)
from genomic_variant_classifier.api.pipeline import INFERENCE_FEATURE_COLUMNS


def test_tabular_features_length_matches_constant():
    assert len(TABULAR_FEATURES) == EXPECTED_TABULAR_FEATURE_COUNT, (
        f"TABULAR_FEATURES has {len(TABULAR_FEATURES)} entries but "
        f"EXPECTED_TABULAR_FEATURE_COUNT={EXPECTED_TABULAR_FEATURE_COUNT}. "
        f"If you added/removed a feature, bump the constant in variant_ensemble.py; "
        f"if you did not, you have unexpected feature drift."
    )


def test_inference_feature_columns_length_matches_constant():
    assert len(INFERENCE_FEATURE_COLUMNS) == EXPECTED_TABULAR_FEATURE_COUNT, (
        f"INFERENCE_FEATURE_COLUMNS has {len(INFERENCE_FEATURE_COLUMNS)} entries; "
        f"expected {EXPECTED_TABULAR_FEATURE_COUNT}. It is derived from TABULAR_FEATURES "
        f"in api/pipeline.py -- check that derivation was not changed to subset/filter."
    )


def test_inference_columns_track_tabular_features_exactly():
    # The derivation is `list(TABULAR_FEATURES)`, so the columns and order must match.
    assert INFERENCE_FEATURE_COLUMNS == list(TABULAR_FEATURES), (
        "INFERENCE_FEATURE_COLUMNS must equal list(TABULAR_FEATURES) in both content "
        "and order."
    )


def test_tabular_features_are_unique():
    dupes = [c for c in TABULAR_FEATURES if TABULAR_FEATURES.count(c) > 1]
    assert not dupes, f"duplicate feature name(s) in TABULAR_FEATURES: {sorted(set(dupes))}"
