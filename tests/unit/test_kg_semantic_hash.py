"""Regression tests for scripts/kg_semantic_hash.py -- the 1000G AF semantic-hash guard.
Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kg_semantic_hash as kg  # noqa: E402


def _frame(n=400, seed=0):
    r = np.random.default_rng(seed)
    vid = [f"{r.integers(1, 23)}:{r.integers(1, 10_000_000)}:A:G" for _ in range(n)]
    return pd.DataFrame({
        "variant_id": vid, "allele_freq": r.random(n),
        "AFR_AF": r.random(n), "EUR_AF": r.random(n), "EAS_AF": r.random(n),
        "SAS_AF": r.random(n), "AMR_AF": r.random(n),
    })


def test_hash_is_row_and_column_order_invariant():
    df = _frame()
    h = kg.semantic_hash(df)
    shuffled = df.sample(frac=1.0, random_state=9)[
        ["AMR_AF", "variant_id", "EUR_AF", "allele_freq", "AFR_AF", "SAS_AF", "EAS_AF"]]
    assert kg.semantic_hash(shuffled) == h


def test_hash_canonicalizes_across_schemas():
    df = _frame()
    sp = pd.DataFrame([s.split(":") for s in df["variant_id"]], columns=["chrom", "pos", "ref", "alt"])
    alt = pd.concat([sp, df[["allele_freq", "AFR_AF", "AMR_AF", "EAS_AF", "EUR_AF", "SAS_AF"]]
                     .reset_index(drop=True)], axis=1).rename(columns={
        "allele_freq": "af", "AFR_AF": "af_afr", "AMR_AF": "af_amr",
        "EAS_AF": "af_eas", "EUR_AF": "af_eur", "SAS_AF": "af_sas"})
    assert kg.semantic_hash(alt) == kg.semantic_hash(df)


def test_value_change_changes_hash():
    df = _frame()
    h = kg.semantic_hash(df)
    df2 = df.copy(); df2.loc[0, "AFR_AF"] = df2.loc[0, "AFR_AF"] + 0.01
    assert kg.semantic_hash(df2) != h


def test_parquet_roundtrip_stable(tmp_path):
    df = _frame()
    p = tmp_path / "kg.parquet"; df.to_parquet(p, index=False)
    assert kg.semantic_hash(p) == kg.semantic_hash(df)


def test_write_if_changed_skips_identical_and_writes_on_change(tmp_path, capsys):
    df = _frame()
    p = tmp_path / "kg.parquet"
    assert kg.write_parquet_if_changed(df, p) is True           # first write
    mtime1 = p.stat().st_mtime_ns
    assert kg.write_parquet_if_changed(df, p) is False          # identical -> skip
    assert "not rewriting parquet" in capsys.readouterr().out
    assert p.stat().st_mtime_ns == mtime1                       # genuinely untouched
    df2 = df.copy(); df2.loc[0, "EUR_AF"] = df2.loc[0, "EUR_AF"] + 0.02
    assert kg.write_parquet_if_changed(df2, p) is True          # changed -> write


def test_missing_af_column_fails_loudly():
    with pytest.raises(kg.KGSchemaError) as e:
        kg.semantic_hash(_frame().drop(columns=["SAS_AF"]))
    assert "af_sas" in str(e.value) and "available columns" in str(e.value)


def test_missing_key_fails_loudly():
    with pytest.raises(kg.KGSchemaError) as e:
        kg.semantic_hash(_frame().drop(columns=["variant_id"]))
    assert "variant key" in str(e.value)


def test_nan_is_deterministic():
    df = _frame(); df.loc[3, "EAS_AF"] = np.nan
    assert kg.semantic_hash(df) == kg.semantic_hash(df.copy())
