"""dtype-family awareness for the schema-drift gate.

Proves the gate treats a string column's pandas-version spelling (object on <=2.x,
string/str on 3.0) as ONE family -- so the pandas-3.0 migration does not read as
drift -- while still catching genuine retyping. Requires pandera + pyarrow (already
project deps). Author: Monzia Moodie
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("pandera")
pytest.importorskip("pyarrow")

from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import (  # noqa: E402
    SchemaDriftAgent,
    _dtype_family,
)


def _baseline(tmp_path, expected_dtypes):
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({
        "schema_version": 1,
        "expected_dtypes": expected_dtypes,
        "expected_schema_hash": SchemaDriftAgent.hash_schema(expected_dtypes),
    }), encoding="utf-8")
    return p


def test_dtype_family_collapses_string_only():
    for s in ("object", "string", "str", "string[python]", "string[pyarrow]"):
        assert _dtype_family(s) == "string"
    for n in ("float64", "int64", "bool", "category", "Int64"):
        assert _dtype_family(n) == n          # identity for everything else


def test_object_string_hash_equal_but_float_int_differ():
    # object <-> string must hash identically; float64 <-> int64 must not.
    assert SchemaDriftAgent.hash_schema({"c": "object"}) == \
        SchemaDriftAgent.hash_schema({"c": "string"})
    assert SchemaDriftAgent.hash_schema({"c": "float64"}) != \
        SchemaDriftAgent.hash_schema({"c": "int64"})


def test_object_baseline_vs_string_observed_is_green(tmp_path):
    bp = _baseline(tmp_path, {"chrom": "object", "af": "float64"})
    det = SchemaDriftAgent.from_baseline(bp, output_dir=tmp_path / "out")
    df = pd.DataFrame({"chrom": pd.array(["17", "X"], dtype="string"),
                       "af": [0.1, 0.2]})
    r = det.detect(df)
    assert r.severity == "green"
    assert r.columns_dtype_changed == ()
    assert r.observed_schema_hash == r.expected_schema_hash


def test_real_retype_float_to_int_is_red(tmp_path):
    bp = _baseline(tmp_path, {"af": "float64", "cadd": "float64"})
    det = SchemaDriftAgent.from_baseline(bp, output_dir=tmp_path / "out")
    r = det.detect(pd.DataFrame({"af": [1, 2, 3], "cadd": [4, 5, 6]}))  # int64
    assert r.severity == "red"
    assert r.columns_dtype_changed


def test_string_family_vs_numeric_is_red(tmp_path):
    bp = _baseline(tmp_path, {"x": "object"})
    det = SchemaDriftAgent.from_baseline(bp, output_dir=tmp_path / "out")
    r = det.detect(pd.DataFrame({"x": [1, 2, 3]}))  # int64, not string
    assert r.severity == "red"
