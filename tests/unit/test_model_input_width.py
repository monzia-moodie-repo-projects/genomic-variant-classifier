"""Unit tests for model_input_width (run-report per-model input width).

The mock cases prove the unwrapping logic with no heavy deps; the final case
exercises the real isotonic-calibrator unwrap with a fitted TabularNN.
Author: Monzia Moodie
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.model_introspect import model_input_width


class _Obj:
    pass


def _masked_nn():
    o = _Obj()
    o.feature_mask_ = np.array([True, True, False, False, True])
    o.n_features_in_ = 5
    return o


def test_masked_neural_uses_mask_sum():
    assert model_input_width(_masked_nn()) == 3


def test_calibrator_wrap_descends_to_mask():
    cal = _Obj(); cal.n_features_in_ = 81; cal._base = _masked_nn()  # outer says 81
    assert model_input_width(cal) == 3                               # inner mask wins


def test_mc_dropout_wrap():
    mc = _Obj(); mc.estimator_ = _masked_nn()
    assert model_input_width(mc) == 3


def test_deep_ensemble_members():
    de = _Obj(); de.members_ = [_masked_nn(), _masked_nn()]
    assert model_input_width(de) == 3


def test_plain_tree_uses_n_features_in():
    t = _Obj(); t.n_features_in_ = 81
    assert model_input_width(t) == 81


def test_calibrated_tree_no_mask():
    t = _Obj(); t.n_features_in_ = 81
    ct = _Obj(); ct._base = t
    assert model_input_width(ct) == 81


def test_no_attrs_returns_none():
    assert model_input_width(_Obj()) is None
    assert model_input_width(None) is None


def test_cycle_safe():
    c = _Obj(); c.estimator_ = c; c.n_features_in_ = 7
    assert model_input_width(c) == 7


def test_all_kept_mask_is_full_width():
    o = _Obj(); o.feature_mask_ = np.ones(81, dtype=bool); o.n_features_in_ = 81
    assert model_input_width(o) == 81


def test_real_tabularnn_through_isotonic_calibrator():
    pytest.importorskip("torch")
    from genomic_variant_classifier.models.variant_ensemble import (
        TabularNNClassifier,
        _IsotonicCalibrator,
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)); X[:, 2] = 0.0; X[:, 3] = 0.0  # 2 constant -> mask keeps 4
    y = (X[:, 0] > 0).astype(int)
    tnn = TabularNNClassifier(epochs=3, random_state=0).fit(X, y)
    assert model_input_width(tnn) == int(tnn.feature_mask_.sum()) == 4
    cal = _IsotonicCalibrator(tnn)            # the real calibration wrapper
    assert model_input_width(cal) == 4        # must descend through _base
