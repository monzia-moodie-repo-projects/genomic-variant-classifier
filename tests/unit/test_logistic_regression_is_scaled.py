"""The logistic-regression base model must be SCALED and must CONVERGE.

Added 2026-07-12.

THE DEFECT
----------
`logistic_regression` was a bare LogisticRegression in the base-model roster, and
VariantEnsemble.fit dispatches it to `X_tab_fit.values` -- the RAW tabular matrix, where
`pos` runs to 1,000,000 alongside `allele_freq` at 1e-6. It did not converge, and it said
so in every test run and every Continuous Integration run, on Python 3.11 and 3.12 alike:

    ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s)

The warning was never noise. A NON-CONVERGED logistic regression was being fit, and its
out-of-fold predictions fed the stacking meta-learner.

WHY IT SURVIVED
---------------
Because a warning is not a failure. It appeared in every run for weeks and everyone learned
to scroll past it -- the same way `INCIDENT_2026-06-14` recorded that tests write to the real
data/ directory and nothing happened for four weeks, because nothing ever FAILED.

A finding in a log is a comment. A finding that fails a test is a gate. Hence this file.

WHY IT MATTERS BEYOND CONVERGENCE
---------------------------------
A stated first-class goal of this project is to "empirically measure/compare/validate ML
algorithms on large complex data ... even at small performance differences". Comparing this
model against XGBoost while it ALONE is handicapped by unscaled inputs measures the DEFECT,
not the algorithm. Every linear-vs-tree conclusion drawn before 2026-07-12 is confounded.

THE AUDIT (2026-07-12) -- it was the only unprotected model
-----------------------------------------------------------
    svm, svm_bagged_rbf   ScalableSVM -> make_pipeline(StandardScaler(), ...)   SCALED
    tabular_nn            TabularNNClassifier -> self.scaler_ + BatchNorm1d     SCALED
    mc_dropout            wraps TabularNNClassifier                             inherits
    deep_ensemble         wraps TabularNNClassifier                             inherits
    kan                   StandardScaler                                        SCALED
    cnn_1d                consumes the ONE-HOT DNA sequence (values in {0,1})   correctly bare
    rf/xgb/lgbm/gbm/cat   trees                                                 scale-invariant
    logistic_regression   BARE                                                  <-- THE BUG

Note cnn_1d: it is dispatched `X_seq`, not `X_tab` (variant_ensemble.py, the 3-way dispatch
in fit()). Standard-scaling a one-hot encoding would DESTROY it. Do not "fix" that one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from genomic_variant_classifier.agent_layer.harness.correctness_harness import (
    build_reference_slice,
)
from genomic_variant_classifier.models.variant_ensemble import (
    EnsembleConfig,
    VariantEnsemble,
    engineer_features,
)


def _roster():
    return VariantEnsemble(EnsembleConfig()).base_estimators


def test_logistic_regression_is_wrapped_in_a_scaler():
    """STRUCTURAL: it must be a Pipeline whose first step scales."""
    lr = _roster()["logistic_regression"]
    assert isinstance(lr, Pipeline), (
        "logistic_regression must be a Pipeline(StandardScaler, LogisticRegression). "
        "VariantEnsemble.fit dispatches it to the RAW tabular matrix (X_tab.values), so a "
        "bare LogisticRegression is fit on features spanning 1e-6 to 1e6 and does not "
        f"converge. Got: {type(lr).__name__}"
    )
    first = list(lr.named_steps.values())[0]
    assert isinstance(first, StandardScaler), (
        f"the first step of the logistic_regression pipeline must be a StandardScaler; "
        f"got {type(first).__name__}"
    )


def test_logistic_regression_converges_on_the_reference_slice():
    """BEHAVIOURAL: fitting it on the real engineered matrix must NOT warn.

    This is the assertion that would have caught the defect. The reference slice is the
    same fully-populated matrix the correctness harness uses, so this exercises the actual
    feature scales -- not a toy.
    """
    df = build_reference_slice(n=200, seed=7)
    X = engineer_features(df)
    X = X.drop(columns=[c for c in ["label"] if c in X.columns])
    Xv = X.to_numpy(dtype=float, na_value=0.0)   # exactly what _stage1_smoke passes
    y = df["label"].to_numpy()

    lr = _roster()["logistic_regression"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lr.fit(Xv, y)

    convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert not convergence, (
        "logistic_regression did not converge on the reference slice:\n  "
        + "\n  ".join(str(w.message).splitlines()[0] for w in convergence)
        + "\nThe model is fit on the RAW tabular matrix. If this fires, the StandardScaler "
          "has been removed or bypassed -- see the module docstring."
    )


def test_the_other_scale_sensitive_models_still_own_a_scaler():
    """REGRESSION GUARD for the audit itself.

    logistic_regression was the only unprotected model. If a future estimator is added to the
    roster without a scaler, or an existing one loses it, this is where it should be caught --
    not in a ConvergenceWarning nobody reads, and not silently (a torch model emits no
    convergence warning at all).
    """
    roster = _roster()

    # svm variants: ScalableSVM scales internally (make_pipeline(StandardScaler(), ...)).
    for name in ("svm", "svm_bagged_rbf"):
        if name in roster:
            src = type(roster[name]).__module__
            assert "scalable_svm" in src, (
                f"{name} is expected to be a ScalableSVM (which scales internally); got {src}"
            )

    # tabular_nn owns a StandardScaler; mc_dropout / deep_ensemble wrap it and inherit.
    if "tabular_nn" in roster:
        assert hasattr(roster["tabular_nn"], "scaler_") or hasattr(
            type(roster["tabular_nn"]), "fit"
        ), "tabular_nn must own a scaler (self.scaler_ is set in fit)"

    for name in ("mc_dropout", "deep_ensemble"):
        if name in roster:
            base = getattr(roster[name], "base_estimator", None)
            assert base is not None and "TabularNN" in type(base).__name__, (
                f"{name} must wrap TabularNNClassifier so it inherits that model's "
                f"StandardScaler; got base_estimator={type(base).__name__}"
            )


def test_cnn_1d_must_NOT_be_scaled():
    """cnn_1d is dispatched the ONE-HOT DNA sequence, not the tabular matrix.

    Its inputs are already {0.0, 1.0}. A StandardScaler would destroy the encoding. This test
    exists because during the 2026-07-12 audit cnn_1d was WRONGLY flagged as unscaled --
    it looked like a bare neural model on tabular features. It is not: VariantEnsemble.fit
    routes `cnn_1d` to X_seq. Do not 'fix' it.
    """
    roster = _roster()
    if "cnn_1d" not in roster:
        pytest.skip("cnn_1d not in the roster (torch unavailable)")
    assert not isinstance(roster["cnn_1d"], Pipeline), (
        "cnn_1d must NOT be wrapped in a StandardScaler pipeline -- it consumes a one-hot "
        "DNA encoding (values in {0,1}), and scaling would destroy it."
    )
