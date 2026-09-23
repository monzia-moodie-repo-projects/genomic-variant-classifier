"""Containment refusals must escape the broad error handlers on scientific paths (2026-09-22).

Each test plants a refusal INSIDE the handled region and requires it to propagate. Each is paired
with a CONTROL raising an ordinary error, proving the test really reaches the handler -- without
it, "the refusal escaped" could simply mean execution never got that far.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from genomic_variant_classifier.api.pipeline import InferencePipeline
from genomic_variant_classifier.containment import ContainmentError, QuarantineError
from genomic_variant_classifier.models.variant_ensemble import (
    EnsembleConfig, VariantEnsemble, engineer_features,
)


class _Raising(BaseEstimator, ClassifierMixin):
    def __init__(self, exc_type=ContainmentError):
        self.exc_type = exc_type

    def fit(self, X, y):
        raise self.exc_type("planted inside a base model")

    def predict(self, X):  # pragma: no cover - never reached; required by scikit-learn's
        raise AssertionError("unreachable")  # parameter validation, which runs before fit

    def predict_proba(self, X):  # pragma: no cover - never reached
        raise AssertionError("unreachable")


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    return X, y


def _ensemble(tmp_path, bad):
    ens = VariantEnsemble(EnsembleConfig(model_dir=tmp_path, allow_base_model_dropout=True))
    ens.base_estimators = {"logistic_regression": LogisticRegression(max_iter=500), "bad": bad}
    return ens


# ---------------------------------------------------------------------------
# VariantEnsemble.fit: the per-model out-of-fold handler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("refusal", [ContainmentError, QuarantineError])
def test_refusal_inside_a_base_model_escapes_even_with_dropout_allowed(tmp_path, refusal):
    X, y = _data()
    with pytest.raises(refusal):
        _ensemble(tmp_path, _Raising(refusal)).fit(X, None, y)


def test_control_ordinary_failure_is_dropped_under_dropout(tmp_path):
    # Proves the test above reaches the handler: an ordinary error there is DROPPED, not raised.
    X, y = _data()
    ens = _ensemble(tmp_path, _Raising(ValueError)).fit(X, None, y)
    assert "bad" in ens.dropped_models_
    # Dropped for the PLANTED error -- not for some earlier failure that never reached fit.
    assert ens.dropped_models_["bad"].startswith("ValueError: planted inside a base model")
    assert "bad" not in ens.trained_models_


# ---------------------------------------------------------------------------
# InferencePipeline: both graph-network scorer fallbacks
# ---------------------------------------------------------------------------
class _Scorer:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def score(self, gene):
        raise self.exc_type("planted inside the graph-network scorer")


def _pipeline(exc_type):
    rows = pd.DataFrame({"chrom": ["1", "2", "X", "3"], "pos": [100, 200, 300, 400],
                         "ref": ["A", "C", "G", "T"], "alt": ["G", "T", "A", "C"],
                         "consequence": ["missense_variant", "synonymous_variant",
                                         "stop_gained", "intron_variant"],
                         "gene_symbol": ["BRCA1", "TP53", "DMD", "CFTR"],
                         "allele_freq": [0.0, 0.2, 0.0, 0.01]})
    X = engineer_features(rows)
    # engineer_features leaves genuinely missing constraint values as NaN (honest missingness), so the
    # stand-in base model must accept NaN natively; filling it here would hide that property.
    base = HistGradientBoostingClassifier(max_iter=5, min_samples_leaf=1).fit(X.values, [1, 0, 1, 0])
    meta = LogisticRegression().fit(base.predict_proba(X.values)[:, 1:], [1, 0, 1, 0])
    return InferencePipeline(trained_models={"logistic_regression": base}, meta_learner=meta,
                             gnn_scorer=_Scorer(exc_type)), rows


@pytest.mark.parametrize("method", ["predict_proba", "predict_proba_with_uncertainty"])
def test_refusal_inside_the_gnn_scorer_escapes(method):
    pipe, rows = _pipeline(ContainmentError)
    with pytest.raises(ContainmentError):
        getattr(pipe, method)(rows)


@pytest.mark.parametrize("method", ["predict_proba", "predict_proba_with_uncertainty"])
def test_control_ordinary_scorer_failure_falls_back(method):
    # Proves the test above reaches the handler: an ordinary scorer error still falls back.
    pipe, rows = _pipeline(RuntimeError)
    out = getattr(pipe, method)(rows)
    proba = out if method == "predict_proba" else out["proba"]
    assert len(proba) == len(rows)
