"""A base model may NEVER vanish from the ensemble in silence.

Added 2026-07-13.

THE DEFECT
----------
`VariantEnsemble.fit` wrapped each base model's out-of-fold (OOF) step in a bare
`except Exception`. On any failure it logged one line, set that model's OOF column to a
constant 0.5, and `continue`d -- which also skipped the `model.fit(...)` call immediately
below. The model was therefore:

    * never fitted;
    * never checkpointed;
    * absent from trained_models_, from oof_model_names_, from the blend, and from every
      downstream comparison artifact.

A 13-model ensemble became a 12-model ensemble, the twelve survivors reported entirely
normal metrics, and the run LOOKED healthy.

BE PRECISE ABOUT THE BLAST RADIUS
---------------------------------
The constant 0.5 column was NOT fed to the stacking meta-learner -- the `valid_cols` filter
in fit() drops the columns of any model missing from trained_models_. That part was always
correct, and was twice mis-stated during this investigation before the code was read
properly. The harm is the SILENT ERASURE, not meta-learner poisoning.

WHY IT MATTERS
--------------
A first-class goal of this project is to measure and compare the performance of every
machine-learning algorithm in the roster. A silently dropped algorithm does not appear in
the report as a FAILURE. It appears as an algorithm that was never a candidate --
indistinguishable from one that was never configured. The comparison is then quietly wrong
in a way no reader can detect from the artifacts.

HOW IT WAS FOUND (the part worth remembering)
---------------------------------------------
Running the suite under `-W error::UserWarning` turned a *spurious* library warning into an
exception, and `test_per_model_checkpoints_written` failed because `models/lightgbm.joblib`
was missing. The warning itself is benign: LightGBM 4.6.0 populates `feature_names_in_` with
synthetic names ('Column_0', 'Column_1', ...) even when fitted on a bare numpy ndarray, so
scikit-learn's `_check_feature_names` warns at predict time. Verified directly:

    >>> m = lgb.LGBMClassifier().fit(numpy_array, y)   # no names supplied
    >>> m.feature_names_in_
    array(['Column_0', 'Column_1', 'Column_2', 'Column_3', 'Column_4'], dtype=object)

There is no column-order hazard: VariantEnsemble dispatches lightgbm the raw ndarray in BOTH
fit and predict. The warning is library noise.

That is the whole point. **Noise was sufficient to delete a model from a paid training run.**
The `except Exception` was broad enough to swallow an out-of-memory error, a transient data
fault, or nothing at all. Hence: fail loud.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from genomic_variant_classifier.models.variant_ensemble import (
    EnsembleConfig,
    VariantEnsemble,
)


class _ExplodingModel(ClassifierMixin, BaseEstimator):
    """A VALID scikit-learn classifier whose predict_proba raises inside cross-validation.

    It must be a *valid* estimator, or the test is a lie. The first version of this fixture
    was a bare duck-typed object, and `cross_val_predict` rejected it up front with
    `InvalidParameterError: The 'estimator' parameter ... must be an object implementing
    'fit' and 'predict'`. The ensemble did raise -- but on the WRONG exception. The saboteur
    was thrown out at the door and never got to sabotage anything, so the test proved only
    that scikit-learn validates its arguments.

    Hence: subclass BaseEstimator/ClassifierMixin (for get_params/set_params/clone and the
    classifier tag), implement `predict` (which the validator demands), and put the failure
    exactly where a real base model's failure lands -- in `predict_proba`, during the
    out-of-fold pass, AFTER a successful fit.
    """

    def __init__(self, boom: str = "synthetic OOF failure") -> None:
        self.boom = boom

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        raise RuntimeError(self.boom)


def _tiny_inputs(n: int = 80, seed: int = 0):
    """The three positional arguments VariantEnsemble.fit actually takes.

    fit(X_tab: DataFrame, X_seq: DataFrame | None, y: Series, ...). X_seq feeds
    cnn_1d and nothing else. cnn_1d is not in this test's roster, so this
    returns None -- and since Part 3 (ff97c34) None is a legal value rather than
    a hole to be filled.

    It used to return `pd.Series(["ACGT" * 8] * n)`, and this docstring said the
    signature "still requires it, so it is supplied and ignored". That value was
    fabricated AND the wrong shape: the annotation claimed Series, production has
    always passed a 2-column [fasta_seq_ref, fasta_seq_alt] DataFrame, and on a
    Series `oh_alt - oh_ref` is identically zero (roadmap 6.28).

    All seven callers use `ens.fit(*_tiny_inputs())`, so this one return
    statement migrates every one of them.

    Labels are made linearly separable-ish so the healthy model is never the
    thing that fails.
    """
    rng = np.random.default_rng(seed)
    f0 = rng.standard_normal(n)
    f1 = rng.standard_normal(n)
    y = (f0 + 0.4 * rng.standard_normal(n) > 0).astype(int)
    X_tab = pd.DataFrame({"f0": f0, "f1": f1})
    return X_tab, None, pd.Series(y)


def _ensemble_with_one_exploding_model(tmp_path, **cfg_kw) -> VariantEnsemble:
    """Build a real VariantEnsemble, then replace the roster with 2 toys, one of which fails.

    We swap the roster rather than sabotage a real model so the test stays fast and stays
    honest about WHAT it is testing: the failure-handling CONTRACT in fit(), not the
    behaviour of any particular estimator.
    """
    ens = VariantEnsemble(EnsembleConfig(model_dir=tmp_path, n_folds=3, **cfg_kw))

    from sklearn.linear_model import LogisticRegression

    ens.base_estimators = {
        "healthy": LogisticRegression(max_iter=200),
        "doomed": _ExplodingModel(),
    }
    return ens


def test_the_saboteur_is_a_valid_estimator_that_fails_where_intended():
    """META-TEST: guard the fixture, not the code.

    If `_ExplodingModel` is not a valid scikit-learn estimator, `cross_val_predict` rejects
    it during parameter validation and raises InvalidParameterError BEFORE the out-of-fold
    pass begins. The ensemble would still raise -- so the dropout tests would still go green
    -- but they would be testing scikit-learn's argument validation instead of a base model
    failing mid-training. That is exactly what happened on the first run of this file.

    A test whose fixture fails for the wrong reason is worse than no test: it is a gate that
    reports PASS while guarding nothing. So assert the saboteur fits cleanly, predicts
    cleanly, and detonates ONLY in predict_proba -- the place a real base model dies.
    """
    from sklearn.base import is_classifier
    from sklearn.utils.validation import check_is_fitted

    X, _, y = _tiny_inputs(n=20)
    m = _ExplodingModel()

    assert is_classifier(m), (
        "the saboteur must carry scikit-learn's classifier tag, or cross_val_predict will "
        "reject it before it can fail the way a real base model fails"
    )
    m.fit(X.values, y.values)          # must NOT raise
    check_is_fitted(m)
    assert m.predict(X.values).shape == (20,)   # must NOT raise

    with pytest.raises(RuntimeError, match="synthetic OOF failure"):
        m.predict_proba(X.values)      # the detonation, in the right place


def test_a_failing_base_model_raises_by_default(tmp_path):
    """THE GATE. Default config: an OOF failure must be a HARD STOP, not a shrug."""
    ens = _ensemble_with_one_exploding_model(tmp_path)

    assert ens.config.allow_base_model_dropout is False, (
        "the default must be to FAIL LOUD; a silently incomplete ensemble must never be "
        "the path of least resistance"
    )

    with pytest.raises(RuntimeError) as exc:
        ens.fit(*_tiny_inputs())

    msg = str(exc.value)
    assert "doomed" in msg, "the error must name the model that failed"
    assert "synthetic OOF failure" in msg, (
        "the error must carry the UNDERLYING cause, not just the fact of failure -- "
        "otherwise the operator has to go hunting through the log for it"
    )
    assert exc.value.__cause__ is not None, (
        "the original exception must be chained (`raise ... from exc`) so the full "
        "traceback of the real failure survives"
    )


def test_the_healthy_model_is_not_what_failed(tmp_path):
    """Sanity: the harness itself is sound -- the roster trains fine without the saboteur."""
    ens = _ensemble_with_one_exploding_model(tmp_path)
    ens.base_estimators.pop("doomed")
    ens.fit(*_tiny_inputs())
    assert "healthy" in ens.trained_models_
    assert ens.dropped_models_ == {}, "a clean run must record ZERO dropped models"


def test_opt_in_dropout_is_permitted_but_recorded(tmp_path, caplog):
    """The escape hatch exists -- but it is LOUD, and the incompleteness is persisted."""
    ens = _ensemble_with_one_exploding_model(tmp_path, allow_base_model_dropout=True)

    with caplog.at_level(logging.ERROR):
        ens.fit(*_tiny_inputs())

    # The run completed -- but the ensemble is a model short, and SAYS SO.
    assert "healthy" in ens.trained_models_
    assert "doomed" not in ens.trained_models_
    assert "doomed" in ens.dropped_models_, (
        "a dropped model MUST be recorded on the fitted object; if it is only logged, the "
        "fact is lost the moment the log scrolls, and every downstream artifact silently "
        "misrepresents the ensemble as complete"
    )
    assert "synthetic OOF failure" in ens.dropped_models_["doomed"], (
        "record the CAUSE, not merely the name"
    )

    text = caplog.text
    assert "ENSEMBLE IS INCOMPLETE" in text, (
        "an incomplete ensemble must announce itself unmissably at ERROR level"
    )
    assert "doomed" in text


def test_dropped_model_is_absent_from_the_comparison_artifacts(tmp_path):
    """The reason this matters: a dropped model corrupts the algorithm comparison.

    `oof_model_names_` is what downstream reporting reads. A dropped model is missing from
    it -- so in the report it is indistinguishable from an algorithm that was never
    configured at all. That is precisely why the default must raise.
    """
    ens = _ensemble_with_one_exploding_model(tmp_path, allow_base_model_dropout=True)
    ens.fit(*_tiny_inputs())

    assert ens.oof_model_names_ == ["healthy"]
    assert ens.oof_predictions_.shape[1] == 1, (
        "the OOF matrix must carry one column per SURVIVING model"
    )
    # The record of the loss lives here, and nowhere else on the object:
    assert set(ens.dropped_models_) == {"doomed"}


def test_completeness_is_recorded_on_a_clean_run(tmp_path):
    """`the ensemble was complete` must be a RECORDED FACT, not an assumption.

    Until 2026-07-13, "the roster has N models" and "the run trained N models" were different
    statements and nothing checked the second. The Kolmogorov-Arnold Network had been failing
    inside imodelsx 1.0.13 -- and being silently swallowed -- in every Continuous Integration
    run since May, and no artifact anywhere recorded that a model was missing.
    """
    ens = _ensemble_with_one_exploding_model(tmp_path)
    ens.base_estimators.pop("doomed")
    ens.fit(*_tiny_inputs())

    c = ens.ensemble_completeness_
    assert c["complete"] is True
    assert c["roster"] == ["healthy"] == c["trained"]
    assert c["dropped"] == {}
    assert c["n_roster"] == c["n_trained"] == 1


def test_completeness_records_the_loss_when_a_model_is_dropped(tmp_path):
    """And when the ensemble IS short, the artifact must say so -- with the cause."""
    ens = _ensemble_with_one_exploding_model(tmp_path, allow_base_model_dropout=True)
    ens.fit(*_tiny_inputs())

    c = ens.ensemble_completeness_
    assert c["complete"] is False, (
        "a 2-model roster that trained 1 model must NOT report itself complete"
    )
    assert c["n_roster"] == 2 and c["n_trained"] == 1
    assert c["trained"] == ["healthy"]
    assert "doomed" in c["dropped"]
    assert "synthetic OOF failure" in c["dropped"]["doomed"], "record the CAUSE"


def test_total_failure_raises_a_comprehensible_error(tmp_path):
    """If EVERY model dies, say so plainly -- do not hand (n, 0) to the meta-learner."""
    ens = _ensemble_with_one_exploding_model(tmp_path, allow_base_model_dropout=True)
    ens.base_estimators = {
        "doomed_a": _ExplodingModel("first failure"),
        "doomed_b": _ExplodingModel("second failure"),
    }

    with pytest.raises(RuntimeError, match="EVERY base model failed"):
        ens.fit(*_tiny_inputs())
