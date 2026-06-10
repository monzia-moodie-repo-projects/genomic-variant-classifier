"""Unit tests for the step-10b protein-coord coverage gate (v2, conditional).

Locks two contracts:
  1. STUB MODE (no AlphaMissense source) must NEVER raise -- the connector degrades
     and warns, ESM-2 stubs to 0.0. This is the regression that v1 broke.
  2. SOURCE PRESENT + near-zero coverage MUST raise before training (the Run 15
     silent-zero: a stale index covering 3,451 of ~2.49M missense).

Pure-function tests; no connectors, no pipeline, no I/O beyond tmp_path stubs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.real_data_prep import (
    _assert_protein_coord_coverage,
    _protein_coord_source_present,
)


def _frame(n_missense: int, n_with_coords: int, n_nonmissense: int = 0) -> pd.DataFrame:
    is_mm = [1] * n_missense + [0] * n_nonmissense
    pp = [1] * n_with_coords + [pd.NA] * (n_missense - n_with_coords) + [pd.NA] * n_nonmissense
    return pd.DataFrame(
        {
            "is_missense": pd.array(is_mm, dtype="int64"),
            "protein_pos": pd.array(pp, dtype="Int64"),
        }
    )


# -- coverage helper -------------------------------------------------------

def test_healthy_coverage_passes_and_returns_fraction():
    assert _assert_protein_coord_coverage(_frame(1000, 968), 0.50) == pytest.approx(0.968)


def test_stale_index_aborts():
    with pytest.raises(ValueError, match="Protein-coord coverage"):
        _assert_protein_coord_coverage(_frame(1_900_000, 3451), 0.50)


def test_exactly_at_threshold_passes():
    assert _assert_protein_coord_coverage(_frame(1000, 500), 0.50) == pytest.approx(0.50)


def test_just_below_threshold_aborts():
    with pytest.raises(ValueError):
        _assert_protein_coord_coverage(_frame(1000, 499), 0.50)


def test_no_missense_is_noop():
    assert _assert_protein_coord_coverage(_frame(0, 0, n_nonmissense=500), 0.80) == 1.0


def test_nonmissense_rows_do_not_dilute_ratio():
    assert _assert_protein_coord_coverage(_frame(1000, 1000, 9000), 0.99) == pytest.approx(1.0)


def test_missing_columns_default_safely():
    assert _assert_protein_coord_coverage(pd.DataFrame({"x": np.arange(10)}), 0.80) == 1.0


# -- source-present guard (decides whether the gate runs at all) -----------

def test_source_present_when_cache_exists(tmp_path):
    cache = tmp_path / "alphamissense_protein_index.parquet"
    cache.write_text("x")
    assert _protein_coord_source_present(cache, None) is True


def test_source_present_when_am_tsv_exists(tmp_path):
    am = tmp_path / "AlphaMissense_hg38.tsv.gz"
    am.write_text("x")
    assert _protein_coord_source_present(tmp_path / "nope.parquet", am) is True


def test_stub_when_neither_present(tmp_path):
    assert _protein_coord_source_present(tmp_path / "nope.parquet", None) is False


def test_stub_when_am_path_set_but_missing(tmp_path):
    assert _protein_coord_source_present(tmp_path / "nope.parquet", tmp_path / "missing_am.tsv.gz") is False


# -- the exact decision the gate makes (guard + assertion together) --------

def test_gate_fires_only_when_source_present_and_low_coverage(tmp_path):
    cache = tmp_path / "alphamissense_protein_index.parquet"
    cache.write_text("x")
    df = _frame(1000, 2)
    assert _protein_coord_source_present(cache, None) is True
    with pytest.raises(ValueError):
        _assert_protein_coord_coverage(df, 0.50)


def test_gate_skipped_in_stub_mode_even_at_zero_coverage(tmp_path):
    df = _frame(1000, 0)
    assert _protein_coord_source_present(tmp_path / "nope.parquet", None) is False
