"""test_feature_health.py  --  Monzia Moodie

The shared feature-health module (single source of truth for degeneracy verdicts,
used by the split-health audit and FeatureCoverageSentinelAgent).
"""
import numpy as np
import pandas as pd

from genomic_variant_classifier.data.feature_health import (
    DEFAULT_NEAR_CONSTANT_FRAC, col_health, is_degenerate, verdict,
)


def test_healthy_numeric():
    h = col_health(pd.Series([0.1, 0.2, 0.3, 0.0, 0.7, 0.9]))
    assert h["degenerate"] == "" and verdict(h) == "healthy" and not is_degenerate(h)


def test_all_zero_is_constant_and_all_zero():
    h = col_health(pd.Series([0.0, 0, 0, 0]))
    assert "ALL_ZERO" in h["degenerate"] and "CONSTANT" in h["degenerate"] and is_degenerate(h)


def test_constant():
    assert verdict(col_health(pd.Series([5.0] * 8))) == "CONSTANT"


def test_near_constant_threshold():
    s = pd.Series([1.0] * 999 + [2.0])  # top_frac = 0.999
    assert "NEAR_CONSTANT" in col_health(s, 0.999)["degenerate"]
    assert col_health(s, 0.9995)["degenerate"] == ""  # raise bar -> healthy


def test_all_null_and_empty():
    assert verdict(col_health(pd.Series([np.nan] * 5))) == "ALL_NULL"
    assert verdict(col_health(pd.Series([], dtype=float))) == "ALL_NULL"


def test_unhashable_cells_do_not_raise():
    h = col_health(pd.Series([[1, 2], [3, 4], [1, 2], [5]], dtype=object))
    assert h["unhashable"] is True and verdict(h) == "healthy"


def test_string_constant():
    assert verdict(col_health(pd.Series(["x"] * 4))) == "CONSTANT"


def test_default_threshold_constant():
    assert DEFAULT_NEAR_CONSTANT_FRAC == 0.999
