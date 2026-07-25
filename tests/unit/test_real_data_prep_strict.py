"""Step 1b.2 acceptance tests for the real_data_prep strict-resolver rewire.

src/genomic_variant_classifier/data/real_data_prep.py no longer defines its own
REVIEW_STATUS_TIER map or the inline substring .map(lambda ...) that applied it. It
consumes the single canonical resolver in
src/genomic_variant_classifier/data/review_status.py via
_build_strict_review_tier_lookup, which validates the whole review-status
vocabulary in one aggregate strict preflight (Option C): every distinct value is
resolved once; any the map does not recognise are collected and reported together
in a single raise; no permissive path and no fallback tier is used.

The label-filter measurement (docs/measurements/LABEL_FILTER_MEASUREMENT_2026-07-24
.txt, commit f4dcd7b) established by row identity that this rewire produces a
byte-identical training cohort: on the 1,686,333 label-filtered rows the tier
distribution is identical under the legacy substring map and the unified resolver,
and the kept-row symmetric difference is 0 at every threshold. The regression test
below pins that agreement on the review statuses that actually appear on labeled
rows, so a future change to the resolver that would move a training row turns the
suite red.
"""
from __future__ import annotations

import pandas as pd
import pytest

from genomic_variant_classifier.data.real_data_prep import (
    _build_strict_review_tier_lookup,
)
from genomic_variant_classifier.data.review_status import (
    UnmatchedReviewStatusError,
)

# The legacy substring map real_data_prep used before the rewire, transcribed for
# the regression comparison only. The rewire removed it from the module.
_LEGACY_MAP = {
    "practice guideline": 1, "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "no assertion criteria provided": 4, "no classification provided": 5,
    "no classification for the individual variant": 5,
}


def _legacy_tier(value: object) -> int:
    s = str(value).lower()
    return next((v for k, v in _LEGACY_MAP.items() if k in s), 5)


# The review statuses that appear on labeled rows in the real cohort (measurement
# 2026-07-24): the explicit vocabulary plus the empty missing token. Notably it
# does NOT include "criteria provided, conflicting classifications" -- zero
# labeled variants carry it -- which is why the rewire moves no training rows.
_LABELED_STATUSES = [
    "practice guideline",
    "reviewed by expert panel",
    "criteria provided, multiple submitters, no conflicts",
    "criteria provided, single submitter",
    "no assertion criteria provided",
    "no classification provided",
    "no classification for the individual variant",
    "",
]


def test_no_local_review_status_tier_map_remains() -> None:
    """The local map and inline substring lambda are gone; the resolver is imported."""
    from genomic_variant_classifier.data import real_data_prep
    source = real_data_prep.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "REVIEW_STATUS_TIER: dict[str, int] = {" not in text
    assert "REVIEW_STATUS_TIER.items()" not in text
    assert "from genomic_variant_classifier.data.review_status import" in text


def test_known_statuses_resolve_to_unified_tiers() -> None:
    s = pd.Series(
        ["criteria provided, single submitter",
         "criteria provided, multiple submitters, no conflicts",
         "no assertion criteria provided"]
    )
    lookup = _build_strict_review_tier_lookup(s)
    assert lookup["criteria provided, single submitter"] == 3
    assert lookup["criteria provided, multiple submitters, no conflicts"] == 2
    assert lookup["no assertion criteria provided"] == 4


def test_missing_token_resolves_and_is_not_unmatched() -> None:
    s = pd.Series(["criteria provided, single submitter", "", ""])
    lookup = _build_strict_review_tier_lookup(s)
    assert lookup[""] == 5  # TIER_MISSING
    assert lookup["criteria provided, single submitter"] == 3


def test_agrees_with_legacy_map_on_every_labeled_status() -> None:
    """The rewire moves no training rows: strict tiers equal legacy tiers on the
    statuses that appear on labeled rows (measurement 2026-07-24, sym_diff=0)."""
    s = pd.Series(_LABELED_STATUSES)
    lookup = _build_strict_review_tier_lookup(s)
    for status in _LABELED_STATUSES:
        assert lookup[status] == _legacy_tier(status), (
            f"tier disagreement on {status!r}: "
            f"strict={lookup[status]} legacy={_legacy_tier(status)}"
        )


def test_unknown_vocabulary_raises_with_complete_inventory() -> None:
    s = pd.Series(
        ["criteria provided, single submitter"] * 2
        + ["future term one"] * 2
        + ["future term two"] * 1
    )
    with pytest.raises(UnmatchedReviewStatusError) as exc_info:
        _build_strict_review_tier_lookup(s)
    msg = str(exc_info.value)
    assert "future term one" in msg
    assert "future term two" in msg


def test_raw_spellings_that_normalize_together_are_one_key_with_raw_counts() -> None:
    s = pd.Series(["future term one"] * 2 + ["Future_Term_One"] * 1)
    with pytest.raises(UnmatchedReviewStatusError) as exc_info:
        _build_strict_review_tier_lookup(s)
    msg = str(exc_info.value)
    assert "'future term one': 3 row(s)" in msg
    assert "Future_Term_One" in msg


def test_no_permissive_resolver_path_is_used() -> None:
    from genomic_variant_classifier.data import real_data_prep
    with open(real_data_prep.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert "allow_unmatched=True" not in text
    assert "TIER_UNMATCHED" not in text


def test_validated_lookup_assigns_a_tier_to_every_known_row() -> None:
    s = pd.Series(_LABELED_STATUSES)
    lookup = _build_strict_review_tier_lookup(s)
    tiers = s.map(lookup)
    assert tiers.notna().all()
