"""
tests/unit/test_splice_ai_promotion.py
=======================================
Regression tests documenting the PROMOTION of splice_ai_score (and new Phase 4
features) into TABULAR_FEATURES, and verifying the new 64-feature contract.

Key invariants tested:
  1.  splice_ai_score IS in TABULAR_FEATURES (promoted from PHASE_2_FEATURES)
  2.  PHASE_2_FEATURES is empty (all features promoted)
  3.  TABULAR_FEATURES has exactly 64 entries (56 tabular + 4 RNA + 4 protein)
  4.  engineer_features() output columns match TABULAR_FEATURES exactly
  5.  engineer_features() produces no NaN values
  6.  All 9 new Phase 4 features are present in TABULAR_FEATURES
  7.  gnn_score (Phase 5) is present in TABULAR_FEATURES
  8.  No module-level logging.basicConfig (Issue L compliance)

New features promoted in Phase 4:
  splice_ai_score, eve_score                              — functional scores
  codon_position, dbsnp_af                               — coding context
  omim_n_diseases, omim_is_autosomal_dominant            — gene-disease
  clingen_validity_score                                 — gene validity
  hgmd_is_disease_mutation, hgmd_n_reports               — HGMD
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.models.variant_ensemble import (
    TABULAR_FEATURES,
    EXPECTED_TABULAR_FEATURE_COUNT,
    PHASE_2_FEATURES,
    engineer_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MODULE_SRC = (
    Path(__file__).resolve().parents[2] / "src" / "genomic_variant_classifier" / "models" / "variant_ensemble.py"
).read_text(encoding="utf-8")

# Phase 4 originally added NINE features. Two of them -- hgmd_is_disease_mutation and
# hgmd_n_reports -- were REMOVED from the feature contract on 2026-07-13 (no HGMD licence, and
# variant-level label leakage against a ClinVar-Pathogenic target). See tests/unit/test_hgmd.py
# for the full reasoning. Seven remain.
NEW_PHASE4_FEATURES = [
    "splice_ai_score",
    "eve_score",
    "codon_position",
    "dbsnp_af",
    "omim_n_diseases",
    "omim_is_autosomal_dominant",
    "clingen_validity_score",
]

#: Removed from the contract 2026-07-13. Asserted ABSENT below -- see
#: test_hgmd_features_are_no_longer_emitted.
REMOVED_HGMD_FEATURES = [
    "hgmd_is_disease_mutation",
    "hgmd_n_reports",
]


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
# 1. Membership: splice_ai_score promoted to TABULAR_FEATURES
# ---------------------------------------------------------------------------

def test_splice_ai_in_tabular_features():
    """splice_ai_score must be in TABULAR_FEATURES (promoted from PHASE_2_FEATURES)."""
    assert "splice_ai_score" in TABULAR_FEATURES, (
        "splice_ai_score should have been promoted to TABULAR_FEATURES in Phase 4"
    )


# ---------------------------------------------------------------------------
# 2. PHASE_2_FEATURES is now empty
# ---------------------------------------------------------------------------

def test_splice_ai_not_in_phase_2_features():
    """splice_ai_score must NOT be in PHASE_2_FEATURES (it was promoted)."""
    assert "splice_ai_score" not in PHASE_2_FEATURES


def test_phase_2_features_is_empty():
    """PHASE_2_FEATURES must be empty — all features have been promoted."""
    assert PHASE_2_FEATURES == [], (
        f"Expected PHASE_2_FEATURES == [] but got: {PHASE_2_FEATURES}"
    )


# ---------------------------------------------------------------------------
# 3. 69-feature contract
# ---------------------------------------------------------------------------

def test_tabular_features_length():
    """TABULAR_FEATURES length must equal EXPECTED_TABULAR_FEATURE_COUNT (single source of truth)."""
    assert len(TABULAR_FEATURES) == EXPECTED_TABULAR_FEATURE_COUNT, (
        f"Expected {EXPECTED_TABULAR_FEATURE_COUNT} TABULAR_FEATURES, got {len(TABULAR_FEATURES)}: {TABULAR_FEATURES}"
    )


# ---------------------------------------------------------------------------
# 4. engineer_features() output columns match TABULAR_FEATURES exactly
# ---------------------------------------------------------------------------

def test_engineer_features_columns_match_tabular_features():
    """engineer_features() must produce columns in exactly the TABULAR_FEATURES order."""
    df = _minimal_df()
    feats = engineer_features(df)
    assert list(feats.columns) == TABULAR_FEATURES, (
        f"Column mismatch.\n"
        f"Extra in output: {set(feats.columns) - set(TABULAR_FEATURES)}\n"
        f"Missing from output: {set(TABULAR_FEATURES) - set(feats.columns)}"
    )


# ---------------------------------------------------------------------------
# 5. No NaN values in engineer_features() output
# ---------------------------------------------------------------------------

def test_engineer_features_no_nans():
    """engineer_features() must not emit any NaN values."""
    df = _minimal_df()
    feats = engineer_features(df)
    assert not feats.isnull().any().any(), (
        f"Unexpected NaNs in engineer_features output:\n{feats.isnull().sum()}"
    )


# ---------------------------------------------------------------------------
# 6. All 9 new Phase 4 features present in TABULAR_FEATURES
# ---------------------------------------------------------------------------

def test_new_features_in_tabular_features():
    """All 7 surviving Phase 4 features must be present in TABULAR_FEATURES."""
    missing = [f for f in NEW_PHASE4_FEATURES if f not in TABULAR_FEATURES]
    assert not missing, (
        f"New Phase 4 features missing from TABULAR_FEATURES: {missing}"
    )


def test_hgmd_features_are_no_longer_in_the_contract():
    """The two Phase 4 features that were REMOVED must stay removed.

    hgmd_is_disease_mutation is, at the variant level, a near-copy of the ClinVar-Pathogenic
    training label. It was constant zero for the life of the project (no licence, connector
    never wired), so it contributed nothing -- but if a licence arrived and someone simply
    restored these two lines, the model would be handed an answer key and would collapse on
    novel variants of uncertain significance, which have no HGMD entry.
    """
    present = [f for f in REMOVED_HGMD_FEATURES if f in TABULAR_FEATURES]
    assert not present, (
        f"HGMD features are back in TABULAR_FEATURES: {present}. Removed 2026-07-13. If the "
        f"licence has been obtained, reintroduce the signal GENE-LEVEL and LEAVE-ONE-OUT "
        f"(n_hgmd_dm_in_gene, excluding the variant being scored) -- never as a variant-level "
        f"flag. See tests/unit/test_hgmd.py."
    )


# ---------------------------------------------------------------------------
# 7. gnn_score (Phase 5) present in TABULAR_FEATURES
# ---------------------------------------------------------------------------

def test_gnn_score_in_tabular_features():
    """gnn_score must be present in TABULAR_FEATURES."""
    assert "gnn_score" in TABULAR_FEATURES


def test_gnn_score_default_half_when_absent():
    """gnn_score default must be 0.5 (ambiguous / GNN not run)."""
    df = _minimal_df()
    assert "gnn_score" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "gnn_score"] == pytest.approx(0.5)
    assert not feats["gnn_score"].isnull().any()


def test_gnn_score_clipped_to_unit_interval():
    """engineer_features() must clip gnn_score to [0, 1]."""
    import pandas as pd
    df = _minimal_df()
    df["gnn_score"] = 1.5   # out-of-range
    feats = engineer_features(df)
    assert feats.loc[0, "gnn_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Default values for new features when absent from input
# ---------------------------------------------------------------------------

def test_splice_ai_default_zero_when_absent():
    """splice_ai_score default must be 0.0 (no splice disruption)."""
    df = _minimal_df()
    assert "splice_ai_score" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "splice_ai_score"] == pytest.approx(0.0)
    assert not feats["splice_ai_score"].isnull().any()


def test_eve_score_default_half_when_absent():
    """eve_score default must be 0.5 (not covered / ambiguous)."""
    df = _minimal_df()
    assert "eve_score" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "eve_score"] == pytest.approx(0.5)


def test_codon_position_default_zero_when_absent():
    """codon_position default must be 0 (non-coding / not derived)."""
    df = _minimal_df()
    assert "codon_position" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "codon_position"] == 0


def test_dbsnp_af_default_zero_when_absent():
    """dbsnp_af default must be 0.0."""
    df = _minimal_df()
    assert "dbsnp_af" not in df.columns
    feats = engineer_features(df)
    assert feats.loc[0, "dbsnp_af"] == pytest.approx(0.0)


def test_omim_n_diseases_default_zero_when_absent():
    df = _minimal_df()
    feats = engineer_features(df)
    assert feats.loc[0, "omim_n_diseases"] == 0


def test_clingen_validity_score_default_zero_when_absent():
    df = _minimal_df()
    feats = engineer_features(df)
    assert feats.loc[0, "clingen_validity_score"] == 0


def test_hgmd_features_are_no_longer_emitted():
    """These two tests previously asserted `hgmd_* == 0` when the source column was absent.

    That assertion was the DEFECT, written down as a requirement. `df.get("hgmd_...", 0)`
    fabricated a column of zeros whenever the annotation was missing -- which it ALWAYS was,
    because no HGMD licence was ever obtained and the connector was never wired into the
    pipeline. So the test passed, every run, for the life of the project, while certifying
    that a feature carrying zero information was behaving correctly.

    Both features were removed from the contract on 2026-07-13. A feature that cannot be
    computed must be ABSENT from the matrix -- not fabricated as zeros, trained on, and
    counted in the feature roster.
    """
    feats = engineer_features(_minimal_df())
    assert "hgmd_is_disease_mutation" not in feats.columns
    assert "hgmd_n_reports" not in feats.columns


# ---------------------------------------------------------------------------
# Real values pass through unchanged
# ---------------------------------------------------------------------------

def test_splice_ai_real_score_passes_through():
    """Explicit splice_ai_score in input must survive engineer_features unchanged."""
    df = _minimal_df(splice_ai_score=0.87)
    feats = engineer_features(df)
    assert feats.loc[0, "splice_ai_score"] == pytest.approx(0.87)


def test_eve_score_real_value_passes_through():
    df = _minimal_df(eve_score=0.92)
    feats = engineer_features(df)
    assert feats.loc[0, "eve_score"] == pytest.approx(0.92)


def test_codon_position_real_value_passes_through():
    df = _minimal_df(codon_position=2)
    feats = engineer_features(df)
    assert feats.loc[0, "codon_position"] == 2


# ---------------------------------------------------------------------------
# 7. Issue L compliance: no module-level logging.basicConfig
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
