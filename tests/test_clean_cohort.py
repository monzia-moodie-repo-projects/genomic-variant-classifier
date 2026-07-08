"""
tests/test_clean_cohort.py  (2026-07-08)
=========================================
Unit tests for scripts/clean_cohort.py v2.

These are NEGATIVE tests as much as positive ones: each guard is fed the precise
input that SHOULD trip it, and the test asserts that it DOES. A guard that no
longer rejects bad input is a broken guard, even if it still passes good input.
See docs/doctrine/ORCHESTRATOR_CANARY_SPEC.md §3 (assertion liveness).

Run:  python -m pytest tests/test_clean_cohort.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "clean_cohort", Path(__file__).resolve().parents[1] / "scripts" / "clean_cohort.py"
)
cc = importlib.util.module_from_spec(_SPEC)
sys.modules["clean_cohort"] = cc
_SPEC.loader.exec_module(cc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _base(n_extra_cols: bool = False) -> pd.DataFrame:
    """Minimal valid cohort: 1 SNV, 1 deletion, 1 insertion, 1 structural."""
    df = pd.DataFrame(
        {
            "variant_id": ["v_snv", "v_del", "v_ins", "v_bad"],
            "ref": ["A", "GCTG", "G", None],
            "alt": ["G", "G", "GTTT", "A"],
            "pathogenicity": ["pathogenic", "likely_pathogenic", "likely_benign", "benign"],
            "metadata": [
                {"review_status": "criteria provided, single submitter", "rs_id": 1},
                {"review_status": "reviewed by expert panel", "rs_id": 2},
                {"review_status": "-", "rs_id": 3},
                {"review_status": "no assertion criteria provided", "rs_id": 4},
            ],
        }
    )
    if n_extra_cols:
        df["ReviewStatus"] = ["criteria provided, single submitter", "", "", ""]
    return df


# ---------------------------------------------------------------------------
# 1. PRE-CONDITION: unresolvable review column must RAISE (the 2026-07-08 defect)
# ---------------------------------------------------------------------------
def test_no_review_column_raises():
    df = _base().drop(columns=["metadata"])
    with pytest.raises(ValueError, match="PRE-CONDITION FAILED"):
        cc.run_clean(df)


def test_no_review_column_allowed_explicitly():
    df = _base().drop(columns=["metadata"])
    clean, _, _, recon = cc.run_clean(df, allow_no_review=True)
    assert recon.review_col.startswith("(none")
    assert any("allow_no_review" in n for n in recon.notes)
    assert len(clean) == 3  # v_bad quarantined as structural


# ---------------------------------------------------------------------------
# 2. Nested struct resolution -- the field _detect_column could never see
# ---------------------------------------------------------------------------
def test_resolves_nested_metadata_review_status():
    _, resolved = cc.resolve_review_series(_base(), None)
    assert resolved == "metadata.review_status"


def test_dotted_review_col_argument():
    s, resolved = cc.resolve_review_series(_base(), "metadata.review_status")
    assert resolved == "metadata.review_status"
    assert s.iloc[1] == "reviewed by expert panel"


def test_top_level_review_column_wins():
    df = _base()
    df["review_status"] = ["practice guideline"] * 4
    _, resolved = cc.resolve_review_series(df, None)
    assert resolved == "review_status"


def test_bad_dotted_path_raises():
    with pytest.raises(ValueError, match="no column"):
        cc.resolve_review_series(_base(), "nosuch.field")


# ---------------------------------------------------------------------------
# 3. Tier + label normalisation (the underscore bugs, and the missing-token rule)
# ---------------------------------------------------------------------------
def test_tier_underscore_form_matches():
    s = pd.Series(["criteria_provided,_single_submitter"])
    assert cc._review_tier(s, 1).iloc[0] == 3


def test_tier_missing_tokens_fall_through_to_default():
    s = pd.Series(["-", "", "NA", None])
    assert list(cc._review_tier(s, 4)) == [cc.TIER_UNMATCHED] * 4


def test_label_underscore_forms_now_map():
    s = pd.Series(["pathogenic", "likely_pathogenic", "benign", "likely_benign", "uncertain"])
    assert list(cc._normalize_label(s)) == [1, 1, 0, 0, -1]


def test_dead_tier_key_both_spellings_present():
    assert cc._review_tier(pd.Series(["no classification for the single variant"]), 1).iloc[0] == 6
    assert cc._review_tier(pd.Series(["no classification for the individual variant"]), 1).iloc[0] == 6


# ---------------------------------------------------------------------------
# 4. Conflict detection now SEES likely_* (previously silently -1 => never a conflict)
# ---------------------------------------------------------------------------
def test_conflict_between_pathogenic_and_likely_benign_is_detected():
    df = pd.DataFrame(
        {
            "variant_id": ["dup", "dup"],
            "ref": ["A", "A"],
            "alt": ["G", "G"],
            "pathogenicity": ["pathogenic", "likely_benign"],
            "metadata": [
                {"review_status": "criteria provided, single submitter"},
                {"review_status": "criteria provided, single submitter"},
            ],
        }
    )
    _, _, conflicts, recon = cc.run_clean(df)
    assert recon.n_conflict_rows == 2, "tied at best tier => irreducible conflict"
    assert len(conflicts) == 2
    assert recon.identity_holds()


def test_conflict_resolved_by_better_review_tier():
    df = pd.DataFrame(
        {
            "variant_id": ["dup", "dup"],
            "ref": ["A", "A"],
            "alt": ["G", "G"],
            "pathogenicity": ["pathogenic", "likely_benign"],
            "metadata": [
                {"review_status": "reviewed by expert panel"},            # tier 1
                {"review_status": "no assertion criteria provided"},      # tier 5
            ],
        }
    )
    clean, _, _, recon = cc.run_clean(df)
    assert recon.n_conflict_resolved_dropped == 1
    assert len(clean) == 1
    assert clean.iloc[0]["pathogenicity"] == "pathogenic"
    assert recon.identity_holds()


# ---------------------------------------------------------------------------
# 5. Row post-conditions still fire
# ---------------------------------------------------------------------------
def test_structural_rows_quarantined_and_identity_holds():
    clean, structural, conflicts, recon = cc.run_clean(_base())
    assert recon.n_structural == 1
    assert recon.n_clean == 3
    assert recon.identity_holds()
    assert not clean["variant_id"].duplicated().any()


def test_missing_key_column_raises():
    with pytest.raises(ValueError, match="Required key columns missing"):
        cc.run_clean(_base().drop(columns=["ref"]))


def test_missing_label_column_raises():
    with pytest.raises(ValueError, match="Could not auto-detect a label column"):
        cc.run_clean(_base().drop(columns=["pathogenicity"]))


# ---------------------------------------------------------------------------
# 6. Schema + composition post-conditions (guards on populations, not just rows)
# ---------------------------------------------------------------------------
def test_output_columns_equal_source_columns():
    df = _base()
    clean, _, _, recon = cc.run_clean(df)
    assert list(clean.columns) == list(df.columns)
    assert recon.clean_columns == list(df.columns)
    assert len(recon.schema_fingerprint) == 16


def test_schema_fingerprint_is_order_independent():
    a = cc.schema_fingerprint(["b", "a", "c"])
    b = cc.schema_fingerprint(["c", "b", "a"])
    assert a == b


def test_composition_counts_variant_classes():
    _, _, _, recon = cc.run_clean(_base())
    assert recon.composition == {"SNV": 1, "deletion": 1, "insertion": 1}


def test_variant_class_classifier():
    vc = cc.variant_class(pd.Series(["A", "GCTG", "G", "AT"]), pd.Series(["G", "G", "GTTT", "GC"]))
    assert list(vc) == ["SNV", "deletion", "insertion", "MNV/other"]


# ---------------------------------------------------------------------------
# 7. THE GUARD THAT WOULD HAVE PREVENTED INCIDENT_2026-07-08
# ---------------------------------------------------------------------------
def test_schema_regression_guard_blocks_dropping_reviewstatus(tmp_path):
    out = tmp_path / "clinvar_grch38_clean.parquet"
    _base(n_extra_cols=True).to_parquet(out, index=False)   # 18-col augmented file exists
    incoming = list(_base().columns)                        # 17-col, no ReviewStatus
    with pytest.raises(ValueError, match="SCHEMA-REGRESSION GUARD"):
        cc.assert_no_schema_regression(incoming, out, allow=False)


def test_schema_regression_guard_names_the_dropped_column(tmp_path):
    out = tmp_path / "c.parquet"
    _base(n_extra_cols=True).to_parquet(out, index=False)
    with pytest.raises(ValueError, match="ReviewStatus"):
        cc.assert_no_schema_regression(list(_base().columns), out, allow=False)


def test_schema_regression_guard_allows_explicit_override(tmp_path):
    out = tmp_path / "c.parquet"
    _base(n_extra_cols=True).to_parquet(out, index=False)
    dropped = cc.assert_no_schema_regression(list(_base().columns), out, allow=True)
    assert dropped == ["ReviewStatus"]


def test_schema_regression_guard_noop_when_no_existing_file(tmp_path):
    assert cc.assert_no_schema_regression(["a", "b"], tmp_path / "absent.parquet") == []


def test_schema_regression_guard_allows_adding_columns(tmp_path):
    out = tmp_path / "c.parquet"
    _base().to_parquet(out, index=False)
    assert cc.assert_no_schema_regression(list(_base(n_extra_cols=True).columns), out) == []


# ---------------------------------------------------------------------------
# 8. Post-write verification reads the FILE, not the intent
# ---------------------------------------------------------------------------
def test_verify_written_schema_passes(tmp_path):
    out = tmp_path / "c.parquet"
    df = _base()
    df.to_parquet(out, index=False)
    cc.verify_written_schema(out, df.columns)


def test_verify_written_schema_fails_on_mismatch(tmp_path):
    out = tmp_path / "c.parquet"
    _base().to_parquet(out, index=False)
    with pytest.raises(ValueError, match="POST-WRITE VERIFY FAILED"):
        cc.verify_written_schema(out, ["totally", "different"])
