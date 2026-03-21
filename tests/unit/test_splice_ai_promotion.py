"""
tests/unit/test_splice_ai_promotion.py
=======================================
Regression tests confirming that splice_ai_score has been promoted from
PHASE_2_FEATURES to TABULAR_FEATURES in variant_ensemble.py.

The two most load-bearing tests are:
  - test_splice_ai_default_zero_when_absent  (catches wrong default, e.g. NaN or 0.5)
  - test_splice_ai_score_position_in_functional_scores  (catches wrong section placement,
    which would affect feature-importance interpretation)
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.variant_ensemble import (
    TABULAR_FEATURES,
    PHASE_2_FEATURES,
    engineer_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MODULE_SRC = (
    Path(__file__).resolve().parents[2] / "src" / "models" / "variant_ensemble.py"
).read_text(encoding="utf-8")


def _minimal_df(**overrides) -> pd.DataFrame:
    """Minimal canonical-schema DataFrame (one row) for engineer_features."""
    base = dict(
        chrom=["17"],
        pos=[43071077],
        ref=["G"],
        alt=["T"],
        gene_symbol=["BRCA1"],
        consequence=["missense_variant"],
        allele_freq=[0.001],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# 1. Membership
# ---------------------------------------------------------------------------
def test_splice_ai_in_tabular_features():
    assert "splice_ai_score" in TABULAR_FEATURES


def test_splice_ai_not_in_phase_2_features():
    assert "splice_ai_score" not in PHASE_2_FEATURES


# ---------------------------------------------------------------------------
# 2. Position — must sit in the functional-scores block, after revel_score
# ---------------------------------------------------------------------------
def test_splice_ai_score_position_in_functional_scores():
    """
    splice_ai_score must appear after revel_score and before in_coding_region
    (the first coding-context feature), confirming it lives in the
    functional-scores section and not elsewhere in the list.
    """
    idx_splice = TABULAR_FEATURES.index("splice_ai_score")
    idx_revel  = TABULAR_FEATURES.index("revel_score")
    idx_coding = TABULAR_FEATURES.index("in_coding_region")
    assert idx_revel < idx_splice < idx_coding, (
        f"splice_ai_score at position {idx_splice} is not between "
        f"revel_score ({idx_revel}) and in_coding_region ({idx_coding})"
    )


# ---------------------------------------------------------------------------
# 3. Default value — engineer_features must emit 0.0 when column absent
# ---------------------------------------------------------------------------
def test_splice_ai_default_zero_when_absent():
    """Default must be 0.0 (no splice impact), not NaN or 0.5 or any other value."""
    df = _minimal_df()
    assert "splice_ai_score" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "splice_ai_score"] == pytest.approx(0.0), (
        "Expected default splice_ai_score=0.0 but got "
        f"{feats.loc[0, 'splice_ai_score']}"
    )
    assert not feats["splice_ai_score"].isnull().any()


# ---------------------------------------------------------------------------
# 4. Real score flows through when column is present
# ---------------------------------------------------------------------------
def test_splice_ai_real_score_used_when_present():
    df = _minimal_df(splice_ai_score=0.87)
    feats = engineer_features(df)
    assert feats.loc[0, "splice_ai_score"] == pytest.approx(0.87)


def test_splice_ai_zero_score_not_replaced_by_default():
    """An explicit 0.0 in the input must pass through, not be treated as missing."""
    df = _minimal_df(splice_ai_score=0.0)
    feats = engineer_features(df)
    assert feats.loc[0, "splice_ai_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. engineer_features output shape / no NaNs
# ---------------------------------------------------------------------------
def test_tabular_features_length():
    """TABULAR_FEATURES must have exactly 20 entries after promotion."""
    assert len(TABULAR_FEATURES) == 20, (
        f"Expected 20 TABULAR_FEATURES, got {len(TABULAR_FEATURES)}: {TABULAR_FEATURES}"
    )


def test_engineer_features_columns_match_tabular_features():
    df = _minimal_df()
    feats = engineer_features(df)
    assert list(feats.columns) == TABULAR_FEATURES


def test_engineer_features_no_nans():
    df = _minimal_df()
    feats = engineer_features(df)
    assert not feats.isnull().any().any(), (
        f"Unexpected NaNs in engineer_features output:\n{feats.isnull().sum()}"
    )


# ---------------------------------------------------------------------------
# 6. Issue-compliance: no module-level logging.basicConfig (Issue L)
# ---------------------------------------------------------------------------
def test_no_basicconfig_in_module():
    """variant_ensemble.py must not call logging.basicConfig at module level."""
    tree = ast.parse(MODULE_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "basicConfig":
                pytest.fail(
                    "logging.basicConfig() found in variant_ensemble.py — violates Issue L"
                )
