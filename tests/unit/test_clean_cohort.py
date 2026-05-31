"""
test_clean_cohort.py - synthetic validation of Phase-0 de-leak logic.

Covers, on a tiny in-memory cohort (no 4.4M-row file needed):
  * null/bad ref or alt -> structural quarantine
  * agreeing duplicate variant_id -> collapse to one (best review)
  * resolvable conflict (path vs benign at different review tiers) -> keep best tier
  * irreducible conflict (path vs benign tie at best tier) -> quarantine to conflicts
  * full row reconciliation identity
  * clean cohort has no null/bad key and no duplicate variant_id

Run: python -m pytest tests/unit/test_clean_cohort.py -v
"""

from __future__ import annotations

import pandas as pd

from clean_cohort import run_clean  # resolved via conftest/sys.path or installed scripts dir


def _frame() -> pd.DataFrame:
    rows = [
        # 2 clean singletons
        {"variant_id": "s1", "ref": "A", "alt": "G", "label": 1, "review_status": "criteria provided, single submitter"},
        {"variant_id": "s2", "ref": "C", "alt": "T", "label": 0, "review_status": "criteria provided, single submitter"},
        # structural: null ref, and bad-token alt
        {"variant_id": "x1", "ref": None, "alt": "G", "label": 1, "review_status": "no assertion criteria provided"},
        {"variant_id": "x2", "ref": "A", "alt": ".", "label": 0, "review_status": "no assertion criteria provided"},
        # agreeing duplicate (same label) -> collapse to 1, drop 1
        {"variant_id": "d1", "ref": "A", "alt": "C", "label": 1, "review_status": "criteria provided, single submitter"},
        {"variant_id": "d1", "ref": "A", "alt": "C", "label": 1, "review_status": "no assertion criteria provided"},
        # resolvable conflict: path at expert panel (tier1) beats benign at single submitter (tier3)
        {"variant_id": "c1", "ref": "T", "alt": "A", "label": 1, "review_status": "reviewed by expert panel"},
        {"variant_id": "c1", "ref": "T", "alt": "A", "label": 0, "review_status": "criteria provided, single submitter"},
        # irreducible conflict: path and benign tie at the SAME best tier
        {"variant_id": "c2", "ref": "G", "alt": "C", "label": 1, "review_status": "criteria provided, single submitter"},
        {"variant_id": "c2", "ref": "G", "alt": "C", "label": 0, "review_status": "criteria provided, single submitter"},
    ]
    return pd.DataFrame(rows)


def test_reconciliation_and_clean_properties():
    df = _frame()
    clean, structural, conflicts, recon = run_clean(df)

    # Reconciliation must be exact.
    assert recon.identity_holds(), recon.as_dict()
    assert recon.n_source == 10
    assert recon.n_structural == 2                 # x1, x2
    assert recon.n_exact_dup_dropped == 1          # one d1 row dropped
    assert recon.n_conflict_resolved_dropped == 1  # benign c1 dropped, path kept
    assert recon.n_conflict_rows == 2              # both c2 rows quarantined
    assert recon.n_clean == 4                       # s1, s2, d1(kept), c1(path kept)

    # Clean integrity.
    assert not clean["variant_id"].duplicated().any()
    assert clean["ref"].notna().all() and clean["alt"].notna().all()
    assert set(clean["variant_id"]) == {"s1", "s2", "d1", "c1"}

    # Resolvable conflict kept the pathogenic (expert-panel) record.
    assert clean.loc[clean["variant_id"] == "c1", "label"].iloc[0] == 1

    # Quarantine contents.
    assert set(structural["variant_id"]) == {"x1", "x2"}
    assert set(conflicts["variant_id"]) == {"c2"}


def test_fails_loud_on_missing_label():
    df = _frame().drop(columns=["label"])
    try:
        run_clean(df)
    except ValueError as e:
        assert "label column" in str(e)
    else:
        raise AssertionError("Expected ValueError when label column is absent")
