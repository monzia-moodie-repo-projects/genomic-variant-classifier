"""Convert the SEMANTIC feature matrix into the MODEL matrix.

THE SEPARATION THIS ENFORCES
============================
`engineer_features` produces a SEMANTIC matrix: what was measured is present,
what was not measured is NaN. That is a statement about biology and annotation
availability, and it must survive to disk unaltered.

Models need numbers. Converting one to the other is a FITTED, STATISTICAL
operation -- medians come from data -- and therefore belongs in an artefact
that is fitted on the training partition only, serialised with the model, and
versioned. It does not belong inside feature engineering, and it does not
belong at each call site.

    raw annotations
          v
    engineer_features()          SEMANTIC MATRIX; NaN = genuinely unavailable
          v
    TabularModelPreprocessor     fitted on TRAINING ONLY
          |                        - validates declared missingness
          |                        - imputes only where the estimator requires it
          |                        - emits a FIXED-SCHEMA availability mask
          v
    MODEL MATRIX

WHY AN ARTEFACT AND NOT A HELPER FUNCTION
==========================================
Measured 2026-08-09: the semantic matrix reaches a model at SIX independent
surfaces -- out-of-fold training, final fit, the stacking meta-learner, the
inference pipeline in api/pipeline.py, the continual trainer, and the benchmark
harness. A contract of the form "every caller must remember to call
prepare_for_model" is not a contract; the seventh surface added later will omit
it silently, exactly as api/pipeline.py:340 already fills absent columns with
0.0 in its own way. A fitted object serialised beside the models cannot be
forgotten, because a model without its preprocessor cannot score anything.

THE INDICATOR SCHEMA IS DECLARED, NEVER DISCOVERED
===================================================
scikit-learn's SimpleImputer(add_indicator=True) emits an indicator only for
features that contained missing values DURING FIT. A column complete in
training and missing at serving therefore produces NO indicator, and the model
receives a feature vector of a shape it has never seen.

Here the indicator set comes from POLICY. Every declared feature gets its mask
column whether or not training happened to contain a missing example, filled
with zeros when nothing was missing. Schema invariance is the contract.

THE THREE-STATE ENCODING IS INJECTIVE
======================================
For the derived binary `gene_is_constrained` the semantic states are

    1  = known constrained
    0  = known not constrained
    NA = unknown, because LOEUF was unavailable

Median imputation would collapse NA into 0 or 1 depending on training
prevalence -- another biological assertion of exactly the kind DUPLICATE-1 was
about. Instead the value carries a STRUCTURAL ZERO and the mask carries the
unknown:

    (value, mask) = (0, 0) -> known false
                    (1, 0) -> known true
                    (0, 1) -> unknown

Zero is only a numeric carrier here; the pair is lossless. That is the one
place where zero-filling is defensible, and it is defensible ONLY because the
mask makes the encoding injective. `test_the_three_state_encoding_is_injective`
holds it to that.

NATIVE-MISSING ESTIMATORS ARE NOT IMPUTED
==========================================
XGBoost, LightGBM and CatBoost handle NaN natively and learn their own split
directions for it. Forcing them through median imputation for implementation
uniformity would DEGRADE them and confound the algorithm comparison that is a
stated goal of this project: a difference between two models would then partly
measure which preprocessing each received.

Both capabilities receive the SAME INFORMATION -- the value and the mask. Only
the encoding of "unavailable" differs: NaN for native estimators, an imputed
value plus its mask for the rest.

WHAT THIS MODULE DOES NOT DECIDE
=================================
Whether the availability masks HELP is an empirical question and this module
does not assume an answer. Absence of a gnomAD constraint annotation encodes
gene curation maturity, transcript type and gene size, so a classifier can
learn P(pathogenic | annotation exists) rather than P(pathogenic | constraint).
That is a shortcut, and the three-arm ablation (FULL / NO_AVAILABILITY /
AVAILABILITY_ONLY) is what distinguishes "masks add signal" from "masks ARE the
signal". The third arm is the one that matters and an on/off comparison cannot
provide it.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MASK_SUFFIX = "__was_missing"
PREPROCESSOR_SCHEMA_VERSION = 1


class MissingStrategy(str, Enum):
    PRESERVE = "preserve"                # leave NaN; only for native estimators
    MEDIAN = "median"                    # training-fitted robust location
    STRUCTURAL_ZERO = "structural_zero"  # numeric carrier; mask holds the meaning
    FORBID = "forbid"                    # missingness here is a defect, not a state


class MissingCapability(str, Enum):
    NATIVE = "native"                    # xgboost, lightgbm, catboost
    REQUIRES_NUMERIC = "requires_numeric"


class UndeclaredMissingnessError(ValueError):
    """A column carries missing values with no declared policy.

    FAIL CLOSED. The alternative -- filling silently -- is precisely the defect
    class this module exists to end: engineer_features once swept the whole
    matrix with fillna(0.0), and because nothing ever failed, a bit-identical
    duplicate feature survived undetected for months.
    """


@dataclass(frozen=True)
class MissingFeaturePolicy:
    strategy: MissingStrategy
    emit_indicator: bool
    rationale: str


DECLARED_MISSINGNESS: Mapping[str, MissingFeaturePolicy] = {
    "gene_constraint_oe": MissingFeaturePolicy(
        strategy=MissingStrategy.MEDIAN,
        emit_indicator=True,
        rationale=("gnomAD pLoF observed/expected. Absent means UNKNOWN -- not "
                   "oe=1 (perfectly tolerant) and not oe=0 (maximally "
                   "constrained). Continuous, so a training-fitted robust "
                   "location is the least assertive numeric carrier."),
    ),
    "gene_is_constrained": MissingFeaturePolicy(
        strategy=MissingStrategy.STRUCTURAL_ZERO,
        emit_indicator=True,
        rationale=("Derived three-state indicator. A median would collapse "
                   "UNKNOWN into whichever class training prevalence favours. "
                   "Zero is a numeric carrier only; the paired mask preserves "
                   "unknown as a distinct state, and the pair is injective."),
    ),
}


@dataclass(frozen=True)
class PreprocessorIdentity:
    schema_version: int
    n_input_features: int
    n_output_features: int
    policy_fingerprint: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def policy_fingerprint(policies: Mapping[str, MissingFeaturePolicy]) -> str:
    """A stable digest of the POLICY, so a serialised preprocessor cannot be
    reloaded under a different declaration without detection -- the same
    identity principle the gnomAD constraint cache now enforces."""
    import hashlib
    parts = ["{}|{}|{}".format(k, v.strategy.value, int(v.emit_indicator))
             for k, v in sorted(policies.items())]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class TabularModelPreprocessor:
    """Semantic matrix -> model matrix. Fitted on the training partition only.

    Contracts
    ---------
    * `fit` sees the TRAINING partition and nothing else.
    * Undeclared missingness FAILS CLOSED, at fit and at transform.
    * The output schema is DECLARED, so it never varies with the data.
    * Input feature order is immutable after fit.
    * A native-missing estimator keeps its NaN and still receives the masks.
    """

    def __init__(self, feature_names, policies=DECLARED_MISSINGNESS,
                 capability: MissingCapability = MissingCapability.REQUIRES_NUMERIC):
        self.feature_names = tuple(feature_names)
        self.policies = dict(policies)
        self.capability = MissingCapability(capability)
        if len(set(self.feature_names)) != len(self.feature_names):
            dupes = sorted({f for f in self.feature_names
                            if list(self.feature_names).count(f) > 1})
            raise ValueError("duplicate feature name(s): {}".format(dupes))
        undeclared = sorted(set(self.policies) - set(self.feature_names))
        if undeclared:
            raise ValueError(
                "policy declared for feature(s) absent from the contract: "
                "{}".format(undeclared))

    # -- schema -----------------------------------------------------------
    @property
    def mask_features(self) -> tuple:
        return tuple("{}{}".format(f, MASK_SUFFIX) for f in self.feature_names
                     if self.policies.get(f) is not None
                     and self.policies[f].emit_indicator)

    def output_schema(self) -> tuple:
        return tuple(self.feature_names) + self.mask_features

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_schema(), dtype=object)

    # -- fit / transform ---------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "TabularModelPreprocessor":
        Xv = self._validate(X)
        self.medians_ = {}
        for col in self.feature_names:
            missing = Xv[col].isna()
            if not bool(missing.any()):
                continue
            policy = self._policy_or_raise(col, int(missing.sum()))
            if policy.strategy is MissingStrategy.FORBID:
                raise UndeclaredMissingnessError(
                    "{!r} forbids missing values; {} present in the training "
                    "partition".format(col, int(missing.sum())))
            if policy.strategy is MissingStrategy.MEDIAN:
                observed = pd.to_numeric(Xv.loc[~missing, col],
                                         errors="coerce").dropna()
                if observed.empty:
                    raise UndeclaredMissingnessError(
                        "{!r} is entirely missing in the TRAINING partition, so "
                        "no training median exists. Imputing from any other "
                        "partition would leak.".format(col))
                self.medians_[col] = float(observed.median())

        # Medians for declared MEDIAN features that happened to be complete in
        # training, so transform can never meet an undeclared statistic.
        for col, policy in self.policies.items():
            if policy.strategy is MissingStrategy.MEDIAN and col not in self.medians_:
                observed = pd.to_numeric(Xv[col], errors="coerce").dropna()
                if not observed.empty:
                    self.medians_[col] = float(observed.median())

        self.identity_ = PreprocessorIdentity(
            schema_version=PREPROCESSOR_SCHEMA_VERSION,
            n_input_features=len(self.feature_names),
            n_output_features=len(self.output_schema()),
            policy_fingerprint=policy_fingerprint(self.policies))
        self.fitted_ = True
        logger.info("TabularModelPreprocessor fitted: %d input -> %d output "
                    "feature(s); medians for %s",
                    len(self.feature_names), len(self.output_schema()),
                    sorted(self.medians_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("TabularModelPreprocessor has not been fitted")
        Xv = self._validate(X).copy()

        # MASKS FIRST, from the DECLARED schema -- before any imputation, and
        # regardless of whether this batch or the training partition contained
        # a missing value. A data-dependent schema would hand a model a shape
        # it has never seen.
        masks = {}
        for col in self.feature_names:
            policy = self.policies.get(col)
            if policy is not None and policy.emit_indicator:
                masks[col + MASK_SUFFIX] = Xv[col].isna().to_numpy().astype(np.float32)

        for col in self.feature_names:
            missing = Xv[col].isna()
            if not bool(missing.any()):
                continue
            policy = self._policy_or_raise(col, int(missing.sum()))
            if policy.strategy is MissingStrategy.FORBID:
                raise UndeclaredMissingnessError(
                    "{!r} forbids missing values; {} present".format(
                        col, int(missing.sum())))
            if policy.strategy is MissingStrategy.PRESERVE:
                continue
            if self.capability is MissingCapability.NATIVE:
                # The estimator learns its own direction for NaN. Imputing here
                # would deprive it of that and confound model comparison.
                continue
            if policy.strategy is MissingStrategy.MEDIAN:
                if col not in self.medians_:
                    raise UndeclaredMissingnessError(
                        "{!r} needs a median but none was fitted; it was "
                        "entirely missing in training".format(col))
                Xv[col] = Xv[col].fillna(self.medians_[col])
            elif policy.strategy is MissingStrategy.STRUCTURAL_ZERO:
                Xv[col] = Xv[col].fillna(0.0)

        for name, values in masks.items():
            Xv[name] = values
        return Xv.loc[:, list(self.output_schema())]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # -- internals ---------------------------------------------------------
    def _policy_or_raise(self, col: str, n_missing: int) -> MissingFeaturePolicy:
        policy = self.policies.get(col)
        if policy is None:
            raise UndeclaredMissingnessError(
                "{!r} carries {} missing value(s) with no declared policy. "
                "Declare it in DECLARED_MISSINGNESS with a rationale, or repair "
                "the connector -- silence is what let a bit-identical duplicate "
                "feature survive undetected.".format(col, n_missing))
        return policy

    def _validate(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("model preprocessing requires a pandas DataFrame")
        absent = [c for c in self.feature_names if c not in X.columns]
        if absent:
            raise ValueError(
                "semantic feature contract missing column(s): {}".format(absent))
        return X.loc[:, list(self.feature_names)]
