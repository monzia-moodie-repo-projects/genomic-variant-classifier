"""Unit tests for the B1 cohort guard (_assert_clean_cohort).

The guard is a @staticmethod, so it is testable without constructing the pipeline.
See docs/incidents/INCIDENT_2026-05-31_null-key-leak.md.
"""
from __future__ import annotations

import pandas as pd
import pytest

from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline


def _clean() -> pd.DataFrame:
    return pd.DataFrame(
        {"ref": ["A", "G", "GGAT"], "alt": ["T", "C", "G"], "variant_id": ["a", "b", "c"]}
    )


def test_clean_cohort_passes() -> None:
    DataPrepPipeline._assert_clean_cohort(_clean(), "clean")  # must not raise


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"ref": [None], "alt": ["T"], "variant_id": ["a"]}),
        pd.DataFrame({"ref": ["A"], "alt": [None], "variant_id": ["a"]}),
        pd.DataFrame({"ref": ["."], "alt": ["T"], "variant_id": ["a"]}),
        pd.DataFrame({"ref": ["A"], "alt": [""], "variant_id": ["a"]}),
    ],
)
def test_null_or_bad_allele_raises(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="null/empty ref or alt"):
        DataPrepPipeline._assert_clean_cohort(df, "bad-allele")


def test_duplicate_variant_id_raises() -> None:
    df = pd.DataFrame({"ref": ["A", "A"], "alt": ["T", "T"], "variant_id": ["dup", "dup"]})
    with pytest.raises(ValueError, match="duplicate variant_id"):
        DataPrepPipeline._assert_clean_cohort(df, "dup")
