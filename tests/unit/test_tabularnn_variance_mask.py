"""Unit tests for TabularNNClassifier's fit-time variance mask.

The mask drops columns that are exactly constant in the training fold from the
neural input (mirroring the existing self.scaler_ lifecycle), and is inherited by
the mc_dropout / deep_ensemble wrappers because both wrap a TabularNNClassifier.
The 81-column feature schema and TABULAR_FEATURES are unchanged -- this is a
model-internal optimization only.

Author: Monzia Moodie
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.models.variant_ensemble import TabularNNClassifier

torch = pytest.importorskip("torch", reason="TabularNN training requires torch")


def _xy(n=200, seed=0, constant_cols=(2, 3, 4)):
    """6-column matrix; `constant_cols` are exactly constant (var==0)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(float)
    for c in constant_cols:
        X[:, c] = 0.0
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _first_linear_in_features(model):
    for layer in model:
        if layer.__class__.__name__ == "Linear":
            return layer.in_features
    raise AssertionError("no Linear layer found")


# --- helper logic (no training) -------------------------------------------------

def test_apply_feature_mask_passthrough_select_and_mismatch():
    m = TabularNNClassifier()
    X = np.arange(12, dtype=float).reshape(4, 3)
    # absent mask -> pass-through (backward-compat with pre-mask pickles)
    assert np.array_equal(m._apply_feature_mask(X), X)
    # present mask -> column select
    m.feature_mask_ = np.array([True, False, True])
    m.n_features_in_ = 3
    assert np.array_equal(m._apply_feature_mask(X), X[:, [0, 2]])
    # wrong width -> fail loud, never silently misalign
    with pytest.raises(ValueError):
        m._apply_feature_mask(X[:, :2])


# --- fit-time mask --------------------------------------------------------------

def test_fit_drops_exact_constant_columns():
    X, y = _xy()
    m = TabularNNClassifier(epochs=3, random_state=0).fit(X, y)
    assert m.n_features_in_ == 6
    assert m.feature_mask_.tolist() == [True, True, False, False, False, True]
    # the net's first layer sizes to the kept count, not the full width
    assert _first_linear_in_features(m.model_) == int(m.feature_mask_.sum()) == 3


def test_fit_keeps_near_constant_column():
    X, y = _xy(constant_cols=(2, 3))
    X[:, 4] = 0.0
    X[0, 4] = 1.0  # variance > 0 -> must be kept (e.g. is_mitochondrial-like)
    m = TabularNNClassifier(epochs=3, random_state=0).fit(X, y)
    assert bool(m.feature_mask_[4]) is True


def test_fit_all_constant_fallback_keeps_all():
    X = np.full((60, 4), 3.0)
    y = (np.arange(60) % 2).astype(int)
    m = TabularNNClassifier(epochs=3, random_state=0).fit(X, y)
    assert m.feature_mask_.all()  # degenerate -> keep all, never 0-width
    assert _first_linear_in_features(m.model_) == 4


# --- predict path ---------------------------------------------------------------

def test_predict_accepts_full_width_and_rejects_wrong_width():
    X, y = _xy()
    m = TabularNNClassifier(epochs=3, random_state=0).fit(X, y)
    proba = m.predict_proba(X)              # full 6-col input; mask applied internally
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        m.predict_proba(X[:, :5])           # 5 != 6 -> raise


def test_persistence_roundtrip_preserves_mask_and_predictions(tmp_path):
    import joblib
    X, y = _xy()
    m = TabularNNClassifier(epochs=3, random_state=0).fit(X, y)
    before = m.predict_proba(X)
    p = tmp_path / "tnn.joblib"
    joblib.dump(m, p)
    m2 = joblib.load(p)
    assert m2.feature_mask_.tolist() == m.feature_mask_.tolist()
    assert m2.n_features_in_ == m.n_features_in_
    np.testing.assert_allclose(before, m2.predict_proba(X), rtol=1e-5, atol=1e-6)


# --- wrapper inheritance --------------------------------------------------------

def test_mc_dropout_wrapper_inherits_mask():
    from genomic_variant_classifier.models.mc_dropout import MCDropoutWrapper
    X, y = _xy()
    mc = MCDropoutWrapper(
        base_estimator=TabularNNClassifier(epochs=3, random_state=0),
        n_passes=4,
        random_state=0,
    ).fit(X, y)
    assert mc.estimator_.feature_mask_.tolist() == [True, True, False, False, False, True]
    assert mc.predict_proba(X).shape == (len(X), 2)


def test_deep_ensemble_members_inherit_mask():
    from genomic_variant_classifier.models.mc_dropout import DeepEnsembleWrapper
    X, y = _xy()
    de = DeepEnsembleWrapper(
        base_estimator=TabularNNClassifier(epochs=3, random_state=0),
        n_members=2,
        random_state=0,
    ).fit(X, y)
    assert len(de.members_) == 2
    for member in de.members_:
        assert member.feature_mask_.tolist() == [True, True, False, False, False, True]
    assert de.predict_proba(X).shape == (len(X), 2)
