"""
tests/unit/test_hgmd.py
========================
Unit tests for HGMDConnector and its wiring into engineer_features.

Coverage:
  1.  Stub mode (no hgmd_path, expected default for users without license)
  2.  Empty DataFrame — empty output with columns present
  3.  _parse_hgmd with a small tab-separated temp file
  4.  _annotate with known lookup
  5.  fetch() round-trip
  6.  TABULAR_FEATURES membership — hgmd_is_disease_mutation, hgmd_n_reports
  7.  engineer_features default (missing columns → 0)
  8.  engineer_features real values pass through
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.hgmd import HGMDConnector, DISEASE_MUTATION_CLASSES
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSV_CONTENT = """\
CHROM\tPOS\tREF\tALT\tCLASS
17\t43071077\tG\tT\tDM
17\t43071077\tG\tT\tDM
13\t32936732\tA\tC\tDM?
1\t925952\tG\tA\tDP
7\t117548628\tC\tT\tFP
"""


def _write_tsv(tmp_path: Path, content: str | None = None) -> Path:
    content = content if content is not None else _TSV_CONTENT
    path = tmp_path / "hgmd.txt"
    path.write_text(content, encoding="utf-8")
    return path


def _minimal_variant_df(**overrides) -> pd.DataFrame:
    base = dict(
        variant_id=["clinvar:17:43071077:G:T"],
        chrom=["17"],
        pos=[43071077],
        ref=["G"],
        alt=["T"],
        gene_symbol=["BRCA1"],
        consequence=["missense_variant"],
        allele_freq=[0.0],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def _engineer_df(**overrides) -> pd.DataFrame:
    base = dict(
        gene_symbol=["BRCA1"],
        consequence=["missense_variant"],
        allele_freq=[0.001],
        ref=["G"],
        alt=["T"],
    )
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# 1. Stub mode (no hgmd_path)
# ---------------------------------------------------------------------------

def test_stub_mode_no_path_returns_defaults():
    """No hgmd_path → hgmd_is_disease_mutation=0, hgmd_n_reports=0."""
    connector = HGMDConnector(hgmd_path=None)
    df = _minimal_variant_df()
    result = connector.annotate_dataframe(df)
    assert "hgmd_is_disease_mutation" in result.columns
    assert "hgmd_n_reports" in result.columns
    assert result["hgmd_is_disease_mutation"].iloc[0] == 0
    assert result["hgmd_n_reports"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 2. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_empty_with_columns():
    connector = HGMDConnector()
    empty = pd.DataFrame(columns=["chrom", "pos", "ref", "alt"])
    result = connector.annotate_dataframe(empty)
    assert "hgmd_is_disease_mutation" in result.columns
    assert "hgmd_n_reports" in result.columns
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _parse_hgmd with temp file
# ---------------------------------------------------------------------------

def test_parse_hgmd_tab_file(tmp_path):
    path = _write_tsv(tmp_path)
    connector = HGMDConnector(hgmd_path=path)
    lookup = connector._parse_hgmd(path)

    assert "lookup_key" in lookup.columns
    assert "hgmd_is_disease_mutation" in lookup.columns
    assert "hgmd_n_reports" in lookup.columns

    # 17:43071077:G:T appears twice (both DM) → n_reports=2, is_dm=1
    row_17 = lookup[lookup["lookup_key"] == "17:43071077:G:T"]
    assert len(row_17) == 1
    assert row_17["hgmd_is_disease_mutation"].iloc[0] == 1
    assert row_17["hgmd_n_reports"].iloc[0] == 2

    # 13:32936732:A:C → DM? → is_dm=1
    row_13 = lookup[lookup["lookup_key"] == "13:32936732:A:C"]
    assert row_13["hgmd_is_disease_mutation"].iloc[0] == 1

    # 1:925952:G:A → DP → is_dm=0
    row_1 = lookup[lookup["lookup_key"] == "1:925952:G:A"]
    assert row_1["hgmd_is_disease_mutation"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 4. _annotate with known lookup
# ---------------------------------------------------------------------------

def test_annotate_matching_variant():
    lookup = pd.DataFrame({
        "lookup_key":              ["17:43071077:G:T"],
        "hgmd_is_disease_mutation": [1],
        "hgmd_n_reports":           [3],
    })
    connector = HGMDConnector()
    df = _minimal_variant_df()
    result = connector._annotate(df, lookup)
    assert result["hgmd_is_disease_mutation"].iloc[0] == 1
    assert result["hgmd_n_reports"].iloc[0] == 3


def test_annotate_no_match_returns_zero():
    lookup = pd.DataFrame({
        "lookup_key":              ["1:111111:A:C"],
        "hgmd_is_disease_mutation": [1],
        "hgmd_n_reports":           [1],
    })
    connector = HGMDConnector()
    df = _minimal_variant_df()   # 17:43071077:G:T — no match
    result = connector._annotate(df, lookup)
    assert result["hgmd_is_disease_mutation"].iloc[0] == 0
    assert result["hgmd_n_reports"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 5. fetch() round-trip
# ---------------------------------------------------------------------------

def test_fetch_round_trip(tmp_path):
    path = _write_tsv(tmp_path)
    connector = HGMDConnector(hgmd_path=path)
    df = _minimal_variant_df(chrom="17", pos=43071077, ref="G", alt="T")
    result = connector.fetch(variant_df=df)
    assert result["hgmd_is_disease_mutation"].iloc[0] == 1
    assert result["hgmd_n_reports"].iloc[0] == 2


# ===========================================================================
# 6-8. THE FEATURE CONTRACT: HGMD IS NOT IN IT. (rewritten 2026-07-13)
# ===========================================================================
#
# These three tests previously asserted the OPPOSITE: that hgmd_is_disease_mutation and
# hgmd_n_reports were members of TABULAR_FEATURES, and that engineer_features emitted them
# (defaulting to 0 when absent, passing real values through when present).
#
# On 2026-07-13 both features were REMOVED from the feature contract
# (EXPECTED_TABULAR_FEATURE_COUNT 97 -> 95). Two independent reasons, either sufficient:
#
#   1. NO ACCESS. HGMD Professional is a paid QIAGEN licence that is not held
#      (docs/ROADMAP.md: "HGMD | hgmd_* (2) | PAID, blocked"). The connector below is fully
#      implemented and fully tested -- but it was never WIRED: no --hgmd-path flag reaches
#      the training pipeline and no data file exists. So `df.get("hgmd_...", 0)` in
#      engineer_features supplied a column of zeros, and both features were CONSTANT ZERO
#      across all 1,038,974 variants of Run 15, contributing exactly nothing while occupying
#      two slots in the contract.
#
#      (They were 2 of THIRTY-SIX dead features in Run 15 -- 46% of the feature space. The
#      published AUROC of 0.998 came from the 38 that were real. See roadmap 6.21.)
#
#   2. LABEL LEAKAGE -- and this reason SURVIVES the licence arriving.
#      HGMD "DM" means *disease-causing mutation*. The training label is ClinVar Pathogenic
#      (real_data_prep.py:512). These are the same quantity under two vendors' names, and
#      HGMD-DM overlaps ClinVar-Pathogenic heavily. As a VARIANT-LEVEL feature it is an
#      answer key: the gene-aware split cannot help, because the leak lives inside every fold
#      at the variant level.
#
#      The deployment failure is the damning part. A novel variant of uncertain significance
#      -- precisely what this classifier exists to score -- has no HGMD entry, so
#      hgmd_is_disease_mutation = 0, and the model reads "not a disease mutation" and leans
#      benign. It would post a superb AUROC on a test set of catalogued variants and
#      systematically under-call the variants that matter.
#
# The CONNECTOR (src/genomic_variant_classifier/data/hgmd.py) is deliberately KEPT, and every
# test of it above is deliberately KEPT PASSING. It is dormant, not deleted. If the licence is
# obtained, the parsing work is done -- but the feature must be reintroduced GENE-LEVEL and
# LEAVE-ONE-OUT (e.g. n_hgmd_dm_in_gene, counting HGMD-DM variants in the gene while EXCLUDING
# the variant being scored), mirroring the existing n_pathogenic_in_gene. Same biological
# signal, no answer key.
# ===========================================================================

def test_hgmd_is_NOT_in_the_feature_contract():
    """Pinned. A two-line deletion is exactly what a well-meaning merge restores."""
    assert "hgmd_is_disease_mutation" not in TABULAR_FEATURES, (
        "hgmd_is_disease_mutation is back in TABULAR_FEATURES. It is a near-copy of the "
        "ClinVar-Pathogenic training label. Reintroducing it as a VARIANT-LEVEL feature "
        "hands the model an answer key and wrecks it on novel variants of uncertain "
        "significance, which have no HGMD entry and would therefore read as 'not a disease "
        "mutation'. If the licence has been obtained, wire it GENE-LEVEL and LEAVE-ONE-OUT."
    )
    assert "hgmd_n_reports" not in TABULAR_FEATURES


def test_engineer_features_does_not_emit_hgmd_columns():
    """The df.get(..., 0) default is gone -- and must not come back.

    That pattern is what silently zeroed these two columns for the entire life of the
    project without anyone noticing. A feature that cannot be computed must be ABSENT, not
    fabricated as zeros and trained on.
    """
    feats = engineer_features(_engineer_df())
    assert "hgmd_is_disease_mutation" not in feats.columns
    assert "hgmd_n_reports" not in feats.columns


def test_engineer_features_ignores_hgmd_columns_even_when_supplied():
    """Even if upstream hands us real HGMD values, they must NOT reach the feature matrix.

    This is the leakage guard with teeth. Someone with an HGMD licence could plausibly join
    the columns onto the input frame and expect them to flow through. They must not -- not
    until the gene-level leave-one-out design is built.
    """
    df = _engineer_df(hgmd_is_disease_mutation=1, hgmd_n_reports=5)
    feats = engineer_features(df)
    assert "hgmd_is_disease_mutation" not in feats.columns, (
        "engineer_features passed a variant-level HGMD disease-mutation flag straight into "
        "the feature matrix. That is the training label wearing a different vendor's badge."
    )
    assert "hgmd_n_reports" not in feats.columns


# ---------------------------------------------------------------------------
# DISEASE_MUTATION_CLASSES constant sanity
# ---------------------------------------------------------------------------

def test_disease_mutation_classes_correct():
    assert "DM" in DISEASE_MUTATION_CLASSES
    assert "DM?" in DISEASE_MUTATION_CLASSES
    # DP, DFP, FP, R are NOT disease mutations
    assert "DP"  not in DISEASE_MUTATION_CLASSES
    assert "FP"  not in DISEASE_MUTATION_CLASSES
