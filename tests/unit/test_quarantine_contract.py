"""The active feature contract excludes the quarantined structural features (2026-09-22).

Basis: quarantine_policy.py; docs/CONTAINMENT_2026-07-24.md section 4; the September 2026 ruling
(Option A: 95 -> 91). variant_ensemble.EXPECTED_TABULAR_FEATURE_COUNT's comment names this file:
it fails if any quarantined name returns to the contract, the source map, the serving columns, or
the feature builder's output -- including when the INPUT still carries the historical columns.
"""
from __future__ import annotations

import pandas as pd

from genomic_variant_classifier import quarantine_policy
from genomic_variant_classifier.api.pipeline import INFERENCE_FEATURE_COLUMNS
from genomic_variant_classifier.models import variant_ensemble as ve

Q = set(quarantine_policy.QUARANTINED_FEATURES)


def _frame(**extra):
    base = {"chrom": ["1", "X"], "pos": [100, 200], "ref": ["A", "G"], "alt": ["G", "T"],
            "consequence": ["missense_variant", "synonymous_variant"]}
    base.update(extra)
    return pd.DataFrame(base)


def test_policy_is_the_single_definition_used_by_the_contract():
    assert ve.QUARANTINED_FEATURES is quarantine_policy.QUARANTINED_FEATURES


def test_contract_excludes_every_quarantined_feature():
    assert not (set(ve.TABULAR_FEATURES) & Q)


def test_count_guard_matches_the_contract():
    assert len(ve.TABULAR_FEATURES) == ve.EXPECTED_TABULAR_FEATURE_COUNT
    assert len(set(ve.TABULAR_FEATURES)) == len(ve.TABULAR_FEATURES)


def test_quarantined_features_are_not_phase_2_placeholders():
    assert not (set(ve.PHASE_2_FEATURES) & Q)


def test_source_map_excludes_every_quarantined_feature():
    assert not (set(ve.FEATURE_SOURCE) & Q)


def test_serving_columns_exclude_every_quarantined_feature():
    assert not (set(INFERENCE_FEATURE_COLUMNS) & Q)
    assert list(INFERENCE_FEATURE_COLUMNS) == list(ve.TABULAR_FEATURES)


def test_feature_builder_output_is_exactly_the_contract():
    out = ve.engineer_features(_frame())
    assert list(out.columns) == list(ve.TABULAR_FEATURES)


def test_historical_structural_columns_in_the_input_never_pass_through():
    # Cached cohorts still carry the four columns with their old, wrong values.
    carried = {name: [90.0, 10.0] for name in quarantine_policy.QUARANTINED_FEATURES}
    out = ve.engineer_features(_frame(**carried))
    assert not (set(out.columns) & Q)
    assert list(out.columns) == list(ve.TABULAR_FEATURES)
