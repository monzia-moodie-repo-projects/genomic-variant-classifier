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
#
# CONTRACT CHANGE, 2026-07-11 (TRIAGE_2026-07-08_test-suite-red, cluster A).
#
# `_protein_coord_source_present` now takes ONE argument -- the DECLARED
# AlphaMissense path -- and returns True iff the caller explicitly declared a
# source AND that declared path exists.
#
# The old two-argument form also returned True when a *coord cache file* merely
# existed on disk. That made the coverage gate arm itself against 2-row unit
# fixtures on any machine that happened to have the cache (or the 613 MB
# AlphaMissense TSV, which the caller silently substituted as a hard-coded
# default when nothing was declared). Twelve tests were green on a clean box and
# red on a populated one -- a suite whose verdict is a function of untracked
# filesystem state.
#
# The old `test_source_present_when_cache_exists` ASSERTED that behaviour, i.e.
# it locked in the defect. It is replaced below by its inverse, which is the
# regression guard for cluster A.


def test_declared_am_tsv_that_exists_is_a_source(tmp_path):
    """A source the caller DECLARED, which exists -> the gate arms. Production path."""
    am = tmp_path / "AlphaMissense_hg38.tsv.gz"
    am.write_text("x")
    assert _protein_coord_source_present(am) is True


def test_nothing_declared_is_stub_mode(tmp_path):
    """Declared nothing -> stub mode. Must never raise. The unit-test path."""
    assert _protein_coord_source_present(None) is False


def test_declared_am_path_that_is_missing_is_stub_mode(tmp_path):
    """Declared a source that is not on this box -> stub mode, not a hard failure."""
    assert _protein_coord_source_present(tmp_path / "missing_am.tsv.gz") is False


def test_stale_coord_cache_on_disk_is_NOT_a_source(tmp_path):
    """REGRESSION GUARD for cluster A (the inverse of the old, defect-asserting test).

    A coord cache sitting on disk -- built by a PREVIOUS run against a PREVIOUS
    cohort -- is NOT a source wired into THIS run. Source presence is a property of
    the DECLARED CONFIGURATION, never of the filesystem. If this ever returns True
    again, the suite's verdict once more depends on which files happen to be on the
    developer's disk, and the 12 cluster-A failures return.
    """
    cache = tmp_path / "alphamissense_protein_index.parquet"
    cache.write_text("x")                      # the cache EXISTS ...
    assert _protein_coord_source_present(None) is False   # ... and is still NOT a source


# -- the exact decision the gate makes (guard + assertion together) --------

def test_gate_fires_only_when_source_declared_and_low_coverage(tmp_path):
    """Source DECLARED + near-zero coverage -> raise before training (the Run-15 silent zero)."""
    am = tmp_path / "AlphaMissense_hg38.tsv.gz"
    am.write_text("x")
    df = _frame(1000, 2)
    assert _protein_coord_source_present(am) is True
    with pytest.raises(ValueError):
        _assert_protein_coord_coverage(df, 0.50)


def test_gate_skipped_in_stub_mode_even_at_zero_coverage(tmp_path):
    """Nothing declared + ZERO coverage -> gate is skipped, nothing raises."""
    df = _frame(1000, 0)
    assert _protein_coord_source_present(None) is False
    # and the coverage assertion is simply never reached in stub mode.


def test_stub_mode_holds_even_when_the_am_tsv_exists_but_was_not_declared(tmp_path):
    """The precise cluster-A mechanism, locked.

    The 613 MB AlphaMissense TSV is present on this box. Before 2026-07-11 the
    caller substituted its hard-coded path whenever nothing was declared, so the
    gate armed and raised on unit fixtures. Declaring nothing must remain stub mode
    REGARDLESS of what exists on disk.
    """
    am = tmp_path / "AlphaMissense_hg38.tsv.gz"
    am.write_text("x")                          # the TSV EXISTS on this box ...
    assert _protein_coord_source_present(None) is False   # ... but nothing was declared
