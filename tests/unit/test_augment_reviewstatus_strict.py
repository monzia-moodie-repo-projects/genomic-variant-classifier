"""Step 1b acceptance tests for the augment_reviewstatus strict-resolver rewire.

scripts/augment_reviewstatus.py no longer defines its own REVIEW_STATUS_TIER map or
substring _tier_of. It consumes the single canonical resolver in
src/genomic_variant_classifier/data/review_status.py, and validates the whole input
vocabulary in one aggregate preflight (Option C, 2026-07-24): every distinct raw
CLNREVSTAT value is resolved once; any the map does not recognise are collected and
reported together in a single raise; no permissive path and no fallback tier is used.

These tests exercise _build_strict_tier_lookup, the pure function that carries that
behavior. The full main() is an integration run against the ClinVar VCF and cohort,
which live outside the repository, so it is not unit-tested here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.review_status import (
    UnmatchedReviewStatusError,
)

# Load the script module by path (scripts/ is not an importable package).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "augment_reviewstatus.py"
_spec = importlib.util.spec_from_file_location("_augment_reviewstatus", _SCRIPT)
assert _spec is not None and _spec.loader is not None
augment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(augment)


def test_no_local_review_status_tier_map_remains() -> None:
    """The whole point of the rewire: the local map is gone, the import is present."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "REVIEW_STATUS_TIER = {" not in source, "local tier map must be removed"
    assert "from genomic_variant_classifier.data.review_status import" in source


def test_known_statuses_resolve_to_unified_tiers() -> None:
    """conflicting -> 3 under the unified exact map (was tier 5 under legacy substring)."""
    s = pd.Series(
        ["criteria provided, single submitter"] * 3
        + ["criteria provided, conflicting classifications"] * 2
        + ["no assertion criteria provided"] * 1
    )
    lookup = augment._build_strict_tier_lookup(s)
    assert lookup["criteria provided, single submitter"] == 3
    assert lookup["criteria provided, conflicting classifications"] == 3
    assert lookup["no assertion criteria provided"] == 4


def test_missing_token_resolves_and_is_not_unmatched() -> None:
    """An empty status is a recognised missing token (tier 5), never an unmatched error."""
    s = pd.Series(["criteria provided, single submitter", "", ""])
    lookup = augment._build_strict_tier_lookup(s)
    assert lookup[""] == 5  # TIER_MISSING
    assert lookup["criteria provided, single submitter"] == 3


def test_unknown_vocabulary_raises_with_complete_inventory() -> None:
    """All unknown values are reported in one pass, not just the first encountered."""
    s = pd.Series(
        ["criteria provided, single submitter"] * 2
        + ["future term one"] * 2
        + ["future term two"] * 1
    )
    with pytest.raises(UnmatchedReviewStatusError) as exc_info:
        augment._build_strict_tier_lookup(s)
    msg = str(exc_info.value)
    assert "future term one" in msg
    assert "future term two" in msg


def test_raw_spellings_that_normalize_together_are_one_key_with_raw_counts() -> None:
    """Multiple raw spellings normalise to one unmatched term; raw counts are retained."""
    s = pd.Series(
        ["future term one"] * 2
        + ["Future_Term_One"] * 1  # normalises to the same key
    )
    with pytest.raises(UnmatchedReviewStatusError) as exc_info:
        augment._build_strict_tier_lookup(s)
    msg = str(exc_info.value)
    # one normalized key totalling three rows, both raw forms reported
    assert "'future term one': 3 row(s)" in msg
    assert "Future_Term_One" in msg


def test_no_permissive_resolver_path_is_used() -> None:
    """The rewire must never call allow_unmatched=True or reference TIER_UNMATCHED."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "allow_unmatched=True" not in source
    assert "TIER_UNMATCHED" not in source


def test_validated_lookup_assigns_a_tier_to_every_known_row() -> None:
    """After a clean preflight, mapping the series leaves no unassigned rows."""
    s = pd.Series(
        ["practice guideline", "reviewed by expert panel",
         "criteria provided, multiple submitters, no conflicts",
         "criteria provided, single submitter", "no assertion criteria provided",
         "no classification provided", ""]
    )
    lookup = augment._build_strict_tier_lookup(s)
    tiers = s.map(lookup)
    assert tiers.notna().all()
    assert tiers.tolist() == [1, 1, 2, 3, 4, 5, 5]
