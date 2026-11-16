"""Stage 1D-1c.1C: the fitted member -- a frozen projection and its estimator, fitted, serialized and served together.

Owner ruling aa999b96 (2026-11-16; revised ruling sections 6-7) and AUDIT_OF_RULING_1D1c1C_2026-11-16 (refinements R1, R3,
R8). The fitted member, not a separately applied global preprocessor, is the unit that is cloned for every out-of-fold
fit, fitted on the final rows, calibrated, saved, loaded, evaluated and served.

* ``ProjectedMember`` -- ONE registered adapter type (exact ``type(...) is`` checks, never an attribute search). It
  receives the CANONICAL named matrix, fits its ``MemberProjection`` (model_preprocessing schema version 2) on exactly the
  rows it is fitted on, renders the projected matrix the way its estimator consumes it (an ndarray, or a named frame for
  CatBoost), and validates every probability matrix through ``prediction_contract.positive_column``.
* ``resolve_underlying_member`` -- what every identity probe uses (draw schedule, prediction stage, input width,
  execution binding), so an MC-dropout or deep-ensemble member keeps its draw schedule inside the adapter
  (MEMBER-WRAPPER-DRAW-SCHEDULE-1).
* ``ESTIMATOR_CAPABILITIES`` -- the measured missing-value capability of every roster member (the small-fit census of
  2026-11-16, re-measured by tests/unit/test_member_projection_1d1c1c.py on every run). A member not listed is refused.
* ``ProjectionProtocol`` -> per-member ``ProjectionMode``: P (capability-aware, the primary predictive benchmark), C
  (common fold median, the controlled comparison), B (availability-blinded draw, research only, never servable).

The composition of a calibrated member is ``_IsotonicCalibrator(ProjectedMember(projection, estimator))``: the calibrator
stays outermost, so its stage stays detectable and calibration consumes canonical rows through the frozen projection.
The sequence model ``cnn_1d`` is never wrapped.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from genomic_variant_classifier.models.feature_semantics import FeatureSemantics
from genomic_variant_classifier.models.model_preprocessing import MemberProjection, ProjectionMode
from genomic_variant_classifier.prediction_contract import ContractError, model_call, positive_column, sklearn_binary_classes

logger = logging.getLogger(__name__)

__all__ = ["ESTIMATOR_CAPABILITIES", "EstimatorCapability", "MemberRendering", "ProjectedMember", "ProjectionProtocol",
           "SEQUENCE_ONLY_MEMBERS", "is_projected", "member_projection_mode", "resolve_underlying_member", "wrap_member"]


class EstimatorCapability(str, Enum):
    NATIVE_NAN = "native_nan"                      # fits and predicts finite probabilities with NaN in its input
    FINITE_REQUIRED = "finite_required"            # raises, drops the column (NEURALNAN-1) or returns non-finite output
    SEQUENCE_WINDOWS = "sequence_windows"          # consumes sequence windows, not the tabular matrix


#: Measured 2026-11-16 (scratchpad nan_probe.py; scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0): random forest,
#: XGBoost, LightGBM and CatBoost fit and predict finite probabilities with NaN; both support vector machines, logistic
#: regression and gradient boosting raise; the three neural members silently drop a NaN-bearing column (NEURALNAN-1); KAN
#: returns non-finite probabilities (KAN-NONFINITE-MISSING-INPUT-1). Re-measured by the test suite, never assumed.
ESTIMATOR_CAPABILITIES: Mapping[str, EstimatorCapability] = MappingProxyType({
    "random_forest": EstimatorCapability.NATIVE_NAN,
    "xgboost": EstimatorCapability.NATIVE_NAN,
    "lightgbm": EstimatorCapability.NATIVE_NAN,
    "catboost": EstimatorCapability.NATIVE_NAN,
    "svm": EstimatorCapability.FINITE_REQUIRED,
    "svm_bagged_rbf": EstimatorCapability.FINITE_REQUIRED,
    "logistic_regression": EstimatorCapability.FINITE_REQUIRED,
    "gradient_boosting": EstimatorCapability.FINITE_REQUIRED,
    "tabular_nn": EstimatorCapability.FINITE_REQUIRED,
    "kan": EstimatorCapability.FINITE_REQUIRED,
    "mc_dropout": EstimatorCapability.FINITE_REQUIRED,
    "deep_ensemble": EstimatorCapability.FINITE_REQUIRED,
    "cnn_1d": EstimatorCapability.SEQUENCE_WINDOWS,
})
SEQUENCE_ONLY_MEMBERS: frozenset = frozenset(n for n, c in ESTIMATOR_CAPABILITIES.items()
                                             if c is EstimatorCapability.SEQUENCE_WINDOWS)


class ProjectionProtocol(str, Enum):
    LEGACY_UNPROJECTED = "legacy_unprojected"                      # historical epoch 1: no member projection
    CAPABILITY_AWARE = "capability_aware_p"                        # P: native NaN where qualified, fold median otherwise
    COMMON_FOLD_MEDIAN = "common_fold_median_c"                    # C: the fold median for every tabular member
    AVAILABILITY_BLINDED = "availability_blinded_b"                # B: research only; never servable


#: The protocols that wrap members, and whether an artifact fitted under each may be served.
SERVABLE_PROTOCOLS: frozenset = frozenset({ProjectionProtocol.LEGACY_UNPROJECTED, ProjectionProtocol.CAPABILITY_AWARE,
                                           ProjectionProtocol.COMMON_FOLD_MEDIAN})


def member_projection_mode(name: str, protocol: ProjectionProtocol) -> ProjectionMode | None:
    """The projection mode a roster member receives under ``protocol``; None for an unwrapped member. A member whose
    capability was never measured is refused: the roster is enumerated, never assumed."""
    if not isinstance(protocol, ProjectionProtocol):
        raise TypeError("protocol must be a ProjectionProtocol")
    capability = ESTIMATOR_CAPABILITIES.get(name)
    if capability is None:
        raise ContractError(f"roster member {name!r} has no measured missing-value capability (ESTIMATOR_CAPABILITIES)")
    if protocol is ProjectionProtocol.LEGACY_UNPROJECTED or capability is EstimatorCapability.SEQUENCE_WINDOWS:
        return None
    if protocol is ProjectionProtocol.COMMON_FOLD_MEDIAN:
        return ProjectionMode.FOLD_MEDIAN
    if protocol is ProjectionProtocol.AVAILABILITY_BLINDED:
        return ProjectionMode.AVAILABILITY_BLINDED_DRAW
    return (ProjectionMode.NATIVE_NAN if capability is EstimatorCapability.NATIVE_NAN else ProjectionMode.FOLD_MEDIAN)


class MemberRendering(str, Enum):
    NDARRAY = "ndarray"
    NAMED_FRAME = "named_frame"


#: Members that consume a NAMED frame (CatBoost resolves its categorical features by column name).
_NAMED_FRAME_MEMBERS = frozenset({"catboost"})


class ProjectedMember(ClassifierMixin, BaseEstimator):
    """One fitted member: ``projection`` fitted on this member's fitting rows, then ``estimator`` on its output."""

    def __init__(self, projection: MemberProjection, estimator: Any, rendering: str = MemberRendering.NDARRAY.value,
                 member_name: str = "") -> None:
        self.projection = projection
        self.estimator = estimator
        self.rendering = rendering
        self.member_name = member_name

    def _render(self, Z: pd.DataFrame):
        rendering = MemberRendering(self.rendering)
        return Z if rendering is MemberRendering.NAMED_FRAME else Z.to_numpy(dtype=np.float64)

    def fit(self, X, y, **fit_kwargs) -> "ProjectedMember":
        if type(self.projection) is not MemberProjection:
            raise ContractError("a projected member requires a MemberProjection (model_preprocessing schema version 2)")
        if not isinstance(X, pd.DataFrame):
            raise ContractError("a projected member is fitted on the canonical named matrix, not an anonymous array")
        projection = clone(self.projection).fit(X)
        Z = projection.transform(X)
        estimator = clone(self.estimator)
        estimator.fit(self._render(Z), y, **fit_kwargs)
        sklearn_binary_classes(estimator)          # exactly the integer classes {0, 1}, or refused
        self.projection_ = projection
        self.estimator_ = estimator
        self.classes_ = np.asarray(estimator.classes_)
        self.input_features_ = tuple(str(c) for c in X.columns)
        self.model_features_ = tuple(Z.columns)
        self.n_features_in_ = len(self.input_features_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("the projected member has not been fitted")
        if not isinstance(X, pd.DataFrame):
            raise ContractError("a projected member predicts from the canonical named matrix, not an anonymous array")
        Z = self.projection_.transform(X)
        values = self.estimator_.predict_proba(self._render(Z))
        positive_column(self.estimator_, values, len(X))   # shape, dtype, finite, [0, 1], row sums: or refused
        return values

    def predict(self, X) -> np.ndarray:
        return model_call(self.predict_proba(X)[:, list(self.classes_).index(1)])

    def describe(self) -> dict:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("the projected member has not been fitted")
        est = self.estimator_
        return {"member": self.member_name, "rendering": MemberRendering(self.rendering).value,
                "underlying_class": f"{type(est).__module__}.{type(est).__qualname__}",
                "projection": self.projection_.describe()}


def resolve_underlying_member(model: Any) -> Any:
    """The estimator a REGISTERED adapter wraps, through the isotonic calibrator; any other object unchanged."""
    from genomic_variant_classifier.models.variant_ensemble import _IsotonicCalibrator
    inner = model._base if type(model) is _IsotonicCalibrator else model
    if type(inner) is ProjectedMember:
        if not hasattr(inner, "estimator_"):
            raise RuntimeError("an unfitted projected member has no underlying fitted estimator")
        return inner.estimator_
    return model


def is_projected(model: Any) -> bool:
    """True for a ProjectedMember, bare or inside the isotonic calibrator (exact types only)."""
    from genomic_variant_classifier.models.variant_ensemble import _IsotonicCalibrator
    inner = model._base if type(model) is _IsotonicCalibrator else model
    return type(inner) is ProjectedMember


def projected_member_of(model: Any) -> "ProjectedMember | None":
    from genomic_variant_classifier.models.variant_ensemble import _IsotonicCalibrator
    inner = model._base if type(model) is _IsotonicCalibrator else model
    return inner if type(inner) is ProjectedMember else None


def wrap_member(name: str, estimator: Any, *, protocol: ProjectionProtocol, semantics: FeatureSemantics,
                feature_names: tuple) -> Any:
    """The unfitted member a protocol prescribes: a ProjectedMember around the configured estimator, or the estimator
    itself (legacy protocol, or the sequence model)."""
    mode = member_projection_mode(name, protocol)
    if mode is None:
        return estimator
    if not isinstance(semantics, FeatureSemantics) or semantics is not FeatureSemantics.V2_OBSERVED_AF:
        raise ContractError(f"a projected protocol needs the epoch-2 semantic matrix, not {semantics!r}")
    rendering = MemberRendering.NAMED_FRAME if name in _NAMED_FRAME_MEMBERS else MemberRendering.NDARRAY
    return ProjectedMember(MemberProjection(tuple(feature_names), mode.value, semantics.value), estimator,
                           rendering.value, name)
