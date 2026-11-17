"""Stage 1D-1c.1C: the projected member, the measured capability census and identity preservation (ruling aa999b96;
audit R3, R8). MEMBER-WRAPPER-DRAW-SCHEDULE-1, KAN-NONFINITE-MISSING-INPUT-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from genomic_variant_classifier.evaluation.model_introspect import model_input_width
from genomic_variant_classifier.models.combiner import (
    member_draw_schedule,
    member_execution_sha256,
    prediction_stage_sha256,
)
from genomic_variant_classifier.models.feature_semantics import AF_FAMILY, FeatureSemantics as FS, derive_family
from genomic_variant_classifier.models.member import (
    ESTIMATOR_CAPABILITIES,
    EstimatorCapability as EC,
    MemberRendering,
    ProjectedMember,
    ProjectionProtocol as PP,
    is_projected,
    member_projection_mode,
    resolve_underlying_member,
    wrap_member,
)
from genomic_variant_classifier.models.model_preprocessing import MemberProjection, ProjectionMode as PM
from genomic_variant_classifier.models.variant_ensemble import EnsembleConfig, VariantEnsemble, _IsotonicCalibrator
from genomic_variant_classifier.prediction_contract import ContractError, canonical_digest

V2 = FS.V2_OBSERVED_AF


def _nan_matrix(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    X.loc[rng.random(n) < 0.3, "a"] = np.nan
    y = ((X["b"] + 0.5 * rng.normal(size=n)) > 0).astype(int).to_numpy()
    return X, y


def _canonical(n: int = 300, seed: int = 1):
    rng = np.random.default_rng(seed)
    af = np.where(rng.random(n) < 0.3, np.nan, rng.beta(0.3, 30, n))
    fam = derive_family(AF_FAMILY, af)
    X = pd.DataFrame({m: fam[m] for m in AF_FAMILY.members})
    X["b"] = rng.normal(size=n)
    X["c"] = rng.normal(size=n)
    y = ((X["b"] + 0.3 * rng.normal(size=n)) > 0).astype(int).to_numpy()
    return X, y


def _dropped_nan_column(est) -> bool:
    for cand in (est, getattr(est, "estimator_", None), *(getattr(est, "members_", None) or [])):
        mask = getattr(cand, "feature_mask_", None)
        if mask is not None:
            return not bool(np.asarray(mask)[0])
    return False


def test_the_capability_table_covers_exactly_the_live_roster():
    roster = set(VariantEnsemble(EnsembleConfig(n_jobs=1)).base_estimators)
    assert roster == set(ESTIMATOR_CAPABILITIES), (roster ^ set(ESTIMATOR_CAPABILITIES))


def test_every_registered_capability_is_re_measured_on_the_live_estimator():
    """The census is measured, never assumed: NATIVE members fit and predict finite probabilities with NaN; every
    FINITE_REQUIRED member raises, returns non-finite probabilities, or drops the NaN-bearing column (NEURALNAN-1)."""
    X, y = _nan_matrix()
    ens = VariantEnsemble(EnsembleConfig(n_jobs=1))
    measured = {}
    for name, est in ens.base_estimators.items():
        if ESTIMATOR_CAPABILITIES[name] is EC.SEQUENCE_WINDOWS:
            continue
        est = clone(est)
        Xin = X if name == "catboost" else X.to_numpy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(Xin, y)
                p = np.asarray(est.predict_proba(Xin), dtype=float)
        except Exception:
            measured[name] = EC.FINITE_REQUIRED
            continue
        measured[name] = (EC.NATIVE_NAN if np.isfinite(p).all() and not _dropped_nan_column(est)
                          else EC.FINITE_REQUIRED)
    expected = {n: c for n, c in ESTIMATOR_CAPABILITIES.items() if c is not EC.SEQUENCE_WINDOWS}
    assert measured == expected


def test_kan_with_nan_returns_non_finite_probabilities_and_the_projection_repairs_it():
    """KAN-NONFINITE-MISSING-INPUT-1 (measured): the fit completes, the probabilities are not finite."""
    X, y = _nan_matrix(seed=5)
    kan = VariantEnsemble(EnsembleConfig(n_jobs=1)).base_estimators["kan"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = clone(kan).fit(X.to_numpy(), y).predict_proba(X.to_numpy())
    assert not np.isfinite(np.asarray(raw, dtype=float)).all()
    Xc, yc = _canonical(seed=5)
    member = wrap_member("kan", kan, protocol=PP.CAPABILITY_AWARE, semantics=V2, feature_names=tuple(Xc.columns))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = clone(member).fit(Xc, yc).predict_proba(Xc)
    assert np.isfinite(np.asarray(p, dtype=float)).all()


@pytest.mark.parametrize("protocol,expected", [
    (PP.CAPABILITY_AWARE, {"random_forest": PM.NATIVE_NAN, "catboost": PM.NATIVE_NAN, "svm": PM.FOLD_MEDIAN,
                           "kan": PM.FOLD_MEDIAN, "mc_dropout": PM.FOLD_MEDIAN, "cnn_1d": None}),
    (PP.COMMON_FOLD_MEDIAN, {"random_forest": PM.FOLD_MEDIAN, "svm": PM.FOLD_MEDIAN, "cnn_1d": None}),
    (PP.MARGINAL_RESAMPLING, {"xgboost": PM.MARGINAL_RESAMPLING_DRAW, "tabular_nn": PM.MARGINAL_RESAMPLING_DRAW,
                               "cnn_1d": None}),
    (PP.LEGACY_UNPROJECTED, {"random_forest": None, "svm": None, "cnn_1d": None}),
])
def test_the_protocols_assign_modes_from_the_measured_capabilities(protocol, expected):
    assert {n: member_projection_mode(n, protocol) for n in expected} == expected


def test_an_unmeasured_member_is_refused():
    with pytest.raises(ContractError, match="no measured"):
        member_projection_mode("new_model", PP.CAPABILITY_AWARE)


def test_wrap_member_rules():
    est = LogisticRegression()
    assert wrap_member("cnn_1d", est, protocol=PP.CAPABILITY_AWARE, semantics=V2, feature_names=("x",)) is est
    assert wrap_member("svm", est, protocol=PP.LEGACY_UNPROJECTED, semantics=FS.V1_LEGACY, feature_names=("x",)) is est
    with pytest.raises(ContractError, match="epoch-2"):
        wrap_member("svm", est, protocol=PP.CAPABILITY_AWARE, semantics=FS.V1_LEGACY, feature_names=("x",))
    cb = wrap_member("catboost", est, protocol=PP.CAPABILITY_AWARE, semantics=V2, feature_names=("x",))
    assert type(cb) is ProjectedMember and cb.rendering == MemberRendering.NAMED_FRAME.value


def _fit_pair(name: str, est, seed: int = 1):
    Xc, yc = _canonical(seed=seed)
    member = wrap_member(name, est, protocol=PP.COMMON_FOLD_MEDIAN, semantics=V2, feature_names=tuple(Xc.columns))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = clone(member).fit(Xc, yc)
        bare = clone(est).fit(fitted.projection_.transform(Xc).to_numpy(), yc)
    return fitted, bare


@pytest.mark.parametrize("name", ["mc_dropout", "deep_ensemble"])
def test_stochastic_draw_schedules_survive_the_adapter(name):
    est = VariantEnsemble(EnsembleConfig(n_jobs=1)).base_estimators[name]
    if name == "mc_dropout":
        est.set_params(n_passes=3)
    else:
        est.set_params(n_members=2)
    est.base_estimator.set_params(epochs=1)
    fitted, bare = _fit_pair(name, est)
    wrapped_schedule = member_draw_schedule(name, fitted)
    assert wrapped_schedule == member_draw_schedule(name, bare)
    assert wrapped_schedule["mechanism"] != "deterministic_single_evaluation"
    assert resolve_underlying_member(fitted) is fitted.estimator_


def test_legacy_identity_digests_are_byte_identical_and_projected_ones_bind_the_projection():
    X, y = _nan_matrix()
    lr = LogisticRegression().fit(X.fillna(0).to_numpy(), y)
    calibrated = _IsotonicCalibrator(LogisticRegression().fit(X.fillna(0).to_numpy(), y)).fit(X.fillna(0).to_numpy(), y)
    for name, model, stage in (("logistic_regression", lr, "raw"),
                               ("random_forest", calibrated, "isotonic_calibrated_on_calibration_fold")):
        served = f"{type(model).__module__}.{type(model).__qualname__}"
        assert prediction_stage_sha256(name, model) == canonical_digest(
            "gvc.member-prediction-stage.v1", {"member": name, "served_class": served, "stage": stage})
        assert member_execution_sha256("a.v1", name, model, "ndarray") == canonical_digest(
            "gvc.member-execution.v1", {"adapter": "a.v1", "member": name, "served_class": served, "rendering": "ndarray"})
    fitted, _ = _fit_pair("logistic_regression", LogisticRegression())
    # The execution identity binds the fitted PROJECTION (and the classes); the estimator's own parameters are bound by
    # the member's artifact digest. A projection fitted on other rows is another execution.
    other, _ = _fit_pair("logistic_regression", LogisticRegression(), seed=9)
    assert prediction_stage_sha256("logistic_regression", fitted) != prediction_stage_sha256("logistic_regression", lr)
    # Version 2, exactly: the served adapter, the class it wraps and its fitted projection's identity.
    assert prediction_stage_sha256("logistic_regression", fitted) == canonical_digest("gvc.member-prediction-stage.v2", {
        "member": "logistic_regression", "served_class": f"{type(fitted).__module__}.{type(fitted).__qualname__}",
        "stage": "raw", "underlying_class": f"{LogisticRegression.__module__}.{LogisticRegression.__qualname__}",
        "member_projection_identity": fitted.projection_.identity_})
    assert prediction_stage_sha256("logistic_regression", fitted) != prediction_stage_sha256("logistic_regression", other)
    wrapped = _IsotonicCalibrator(fitted)
    Xc, yc = _canonical()
    wrapped.fit(Xc, yc)
    assert is_projected(wrapped) and resolve_underlying_member(wrapped) is fitted.estimator_
    assert prediction_stage_sha256("logistic_regression", wrapped) != prediction_stage_sha256("logistic_regression", fitted)
    assert member_execution_sha256("a.v1", "x", fitted, "r") != member_execution_sha256("a.v1", "x", other, "r")


def test_the_input_width_is_read_through_the_adapter():
    est = VariantEnsemble(EnsembleConfig(n_jobs=1)).base_estimators["tabular_nn"]
    est.set_params(epochs=1)
    fitted, bare = _fit_pair("tabular_nn", est)
    assert model_input_width(fitted) == model_input_width(bare) == int(np.asarray(bare.feature_mask_).sum())


class _NaNOut(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), np.nan)


def test_a_non_finite_member_output_is_refused_never_passed_on():
    Xc, yc = _canonical()
    m = ProjectedMember(MemberProjection(tuple(Xc.columns), PM.FOLD_MEDIAN.value, V2.value), _NaNOut()).fit(Xc, yc)
    with pytest.raises(ContractError):
        m.predict_proba(Xc)


def test_a_projected_member_refuses_an_anonymous_array():
    Xc, yc = _canonical()
    m = ProjectedMember(MemberProjection(tuple(Xc.columns), PM.FOLD_MEDIAN.value, V2.value), LogisticRegression())
    with pytest.raises(ContractError):
        m.fit(Xc.to_numpy(), yc)
    m.fit(Xc, yc)
    with pytest.raises(ContractError):
        m.predict_proba(Xc.to_numpy())
