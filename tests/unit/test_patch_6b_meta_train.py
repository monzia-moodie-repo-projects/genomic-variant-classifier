"""Regression test for Patch 6b (INCIDENT_2026-04-30).

Asserts that DataPrepPipeline._save_splits persists meta_train.parquet when
meta_train is provided, and that gene_symbol survives the parquet roundtrip.

This prevents regression of the gene_symbol KeyError that caused Run 9 GNN
training to silently fail. See:
    docs/incidents/INCIDENT_2026-04-30_gnn-gene-symbol-keyerror.md
    scripts/run_phase2_eval.py (Patch 6b L292-L317 reads meta_train.parquet)

Author: PM11a (2026-05-27)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline


@pytest.fixture
def tiny_splits(tmp_path: Path):
    """Build minimal X/y/meta fixtures with gene_symbol column."""
    rng = np.random.default_rng(seed=42)
    n_train, n_val, n_test = 50, 10, 20

    def _make_x(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
        })

    def _make_y(n: int) -> pd.Series:
        return pd.Series(rng.integers(0, 2, n), name="label")

    def _make_meta(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "gene_symbol": rng.choice(["BRCA1", "BRCA2", "TP53"], n),
            "variant_id": [f"v_{i}" for i in range(n)],
        })

    return {
        "X_train": _make_x(n_train),
        "X_val": _make_x(n_val),
        "X_test": _make_x(n_test),
        "y_train": _make_y(n_train),
        "y_val": _make_y(n_val),
        "y_test": _make_y(n_test),
        "meta_train": _make_meta(n_train),
        "meta_val": _make_meta(n_val),
        "meta_test": _make_meta(n_test),
    }


def _make_pipeline(outdir: Path) -> DataPrepPipeline:
    """Construct a DataPrepPipeline with output_dir override.

    Uses object.__new__ to bypass DataPrepConfig field requirements that may
    drift over time. This test only exercises _save_splits, which only reads
    self.config.output_dir.
    """
    pipeline = object.__new__(DataPrepPipeline)

    class _CfgStub:
        pass

    cfg = _CfgStub()
    cfg.output_dir = outdir
    pipeline.config = cfg
    return pipeline


def test_save_splits_writes_meta_train_parquet(tiny_splits, tmp_path):
    """REGRESSION: Patch 6b writes meta_train.parquet (INCIDENT_2026-04-30)."""
    pipeline = _make_pipeline(tmp_path)
    pipeline._save_splits(
        X_train=tiny_splits["X_train"],
        X_val=tiny_splits["X_val"],
        X_test=tiny_splits["X_test"],
        y_train=tiny_splits["y_train"],
        y_val=tiny_splits["y_val"],
        y_test=tiny_splits["y_test"],
        meta_val=tiny_splits["meta_val"],
        meta_test=tiny_splits["meta_test"],
        meta_train=tiny_splits["meta_train"],
    )
    assert (tmp_path / "meta_train.parquet").exists(), (
        "meta_train.parquet must be written when meta_train is provided. "
        "See INCIDENT_2026-04-30 / Patch 6b."
    )


def test_save_splits_meta_train_preserves_gene_symbol(tiny_splits, tmp_path):
    """REGRESSION: gene_symbol survives the parquet roundtrip.

    scripts/run_phase2_eval.py L298-L302 reads meta_train.parquet and merges
    gene_symbol into gnn_df. If gene_symbol is dropped at write time, GNN
    training crashes with KeyError downstream.
    """
    pipeline = _make_pipeline(tmp_path)
    pipeline._save_splits(
        X_train=tiny_splits["X_train"],
        X_val=tiny_splits["X_val"],
        X_test=tiny_splits["X_test"],
        y_train=tiny_splits["y_train"],
        y_val=tiny_splits["y_val"],
        y_test=tiny_splits["y_test"],
        meta_val=tiny_splits["meta_val"],
        meta_test=tiny_splits["meta_test"],
        meta_train=tiny_splits["meta_train"],
    )
    reloaded = pd.read_parquet(tmp_path / "meta_train.parquet")
    assert "gene_symbol" in reloaded.columns, (
        "meta_train.parquet must contain gene_symbol column for "
        "run_phase2_eval.py GNN integration to source from."
    )
    expected_genes = set(tiny_splits["meta_train"]["gene_symbol"].unique())
    actual_genes = set(reloaded["gene_symbol"].unique())
    assert expected_genes == actual_genes, (
        f"gene_symbol values changed through roundtrip: "
        f"expected {expected_genes}, got {actual_genes}"
    )


def test_save_splits_meta_train_optional_when_none(tiny_splits, tmp_path):
    """meta_train=None must not write meta_train.parquet but must not raise.

    Backward-compat: pre-Patch-6b runs and non-GNN runs may pass None.
    meta_val/meta_test are still written unconditionally.
    """
    pipeline = _make_pipeline(tmp_path)
    pipeline._save_splits(
        X_train=tiny_splits["X_train"],
        X_val=tiny_splits["X_val"],
        X_test=tiny_splits["X_test"],
        y_train=tiny_splits["y_train"],
        y_val=tiny_splits["y_val"],
        y_test=tiny_splits["y_test"],
        meta_val=tiny_splits["meta_val"],
        meta_test=tiny_splits["meta_test"],
        meta_train=None,
    )
    assert not (tmp_path / "meta_train.parquet").exists(), (
        "When meta_train is None, meta_train.parquet must NOT be written."
    )
    assert (tmp_path / "meta_val.parquet").exists(), (
        "meta_val.parquet must always be written."
    )
    assert (tmp_path / "meta_test.parquet").exists(), (
        "meta_test.parquet must always be written."
    )
