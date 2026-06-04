"""Tests for the review-tier filter + its fail-loud guard.

Regression coverage for INCIDENT_2026-06-03 (min-review-tier silent no-op): the
filter must FIRE when ReviewStatus is present, and must RAISE (not silently skip)
when ReviewStatus is absent but a real tier (<5) is requested. Exercises
DataPrepPipeline._load_and_label directly (mirrors test_cohort_guard's pattern of
testing the method in isolation).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.real_data_prep import (
    AnnotationConfig,
    DataPrepConfig,
    DataPrepPipeline,
)


def _cohort(tmp_path: Path, with_review: bool) -> Path:
    cols = {
        "variant_id": [f"v{i}" for i in range(6)],
        "chrom": ["1", "1", "2", "X", "17", "3"],
        "pos": [100, 200, 300, 400, 500, 600],
        "ref": ["A", "C", "G", "T", "A", "C"],
        "alt": ["G", "T", "A", "C", "G", "T"],
        "clinical_sig": [
            "Pathogenic", "Benign", "Likely pathogenic",
            "Benign", "Pathogenic", "Benign",
        ],
    }
    if with_review:
        cols["ReviewStatus"] = [
            "reviewed by expert panel",                              # tier 1
            "criteria provided, single submitter",                  # tier 3
            "criteria provided, multiple submitters, no conflicts",  # tier 2
            "no assertion criteria provided",                       # tier 4 -> drop
            "",                                                     # tier 5 -> drop
            "criteria provided, single submitter",                  # tier 3
        ]
    p = tmp_path / ("cohort_rev.parquet" if with_review else "cohort_norev.parquet")
    pd.DataFrame(cols).to_parquet(p, index=False)
    return p


def _pipeline(tmp_path: Path, tier: int) -> DataPrepPipeline:
    return DataPrepPipeline(
        config=DataPrepConfig(min_review_tier=tier, output_dir=tmp_path / "splits"),
        annotation_config=AnnotationConfig(),
    )


def test_filter_fires_when_reviewstatus_present(tmp_path):
    p = _cohort(tmp_path, with_review=True)
    out = _pipeline(tmp_path, 3)._load_and_label(str(p))
    # 6 labeled; tiers [1,3,2,4,5,3]; keep <=3 -> drop tier-4 + tier-5 -> 4 rows
    assert len(out) == 4
    assert "review_tier" not in out.columns  # dropped: must not leak as a feature


def test_raises_when_reviewstatus_absent_and_tier_requested(tmp_path):
    p = _cohort(tmp_path, with_review=False)
    with pytest.raises(ValueError, match="ReviewStatus"):
        _pipeline(tmp_path, 3)._load_and_label(str(p))


def test_no_raise_when_tier_filter_disabled(tmp_path):
    p = _cohort(tmp_path, with_review=False)
    out = _pipeline(tmp_path, 5)._load_and_label(str(p))  # 5 = disabled
    assert len(out) == 6  # all labeled kept; no filter, no raise
