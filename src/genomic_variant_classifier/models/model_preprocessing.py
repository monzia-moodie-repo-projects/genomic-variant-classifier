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

import functools as _functools
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path as _Path
from typing import Mapping

import numpy as np
import pandas as pd

from genomic_variant_classifier.code_identity import symbol_closure_digest
from genomic_variant_classifier.models.feature_semantics import (
    PROJECTABLE_FAMILIES,
    FeatureSemantics,
    derive_family,
    infer_semantics,
)
from genomic_variant_classifier.prediction_contract import canonical_digest

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


# =====================================================================================================================
# SCHEMA VERSION 2 -- THE MEMBER PROJECTION (stage 1D-1c.1C; owner ruling aa999b96; AUDIT_OF_RULING_1D1c1C_2026-11-16)
# =====================================================================================================================
#
# Version 1 above is KEPT, unchanged, for the artifacts that carry it (historical replay). It was fitted after the
# members and applied only at serving, so no member was ever fitted on its output (MEMBER-PROJECTION-SERVING-DRIFT-1).
#
# Version 2 is fitted INSIDE each fitted member (models/member.py ProjectedMember): inside every out-of-fold clone on
# that fold's training rows, and once on the final fitting rows; calibration, evaluation and serving reuse the frozen
# fitted object. Its contracts:
#
# * The input is the CANONICAL semantic matrix of an epoch (models/feature_semantics.py), named, in its fitted order.
# * A registered derivation family (today: allele frequency, epoch 2) is projected JOINTLY: one statistic of the root,
#   the dependents re-derived by the registered implementations. The family's members are renamed to their
#   model-projected names, so an imputed zero can never be read as an observed one (AF-PROJECTION-OBSERVATION-
#   CONFLATION-1). The canonical family is re-derived from its root and compared EXACTLY on every fit and transform.
# * No availability indicator is ever emitted (strict regime; ruling aa999b96 decision A).
# * Missingness outside a family is refused unless declared below (fail closed, as version 1).
# * Three modes: NATIVE_NAN (the estimator receives NaN), FOLD_MEDIAN (the fold-fitted median of the observed root),
#   AVAILABILITY_BLINDED_DRAW (a research projection: the surrogate is drawn from the fold's observed root values, keyed
#   by the row's own content, so the imputed column cannot reveal which rows were unobserved; audit R1 measured the
#   median exposing availability at AUROC 0.99983 and the draw at 0.50487). A blinded artifact is never servable.

PROJECTION_SCHEMA = "gvc.member-projection/2"
_THIS_FILE = "src/genomic_variant_classifier/models/model_preprocessing.py"
_SEMANTICS_FILE = "src/genomic_variant_classifier/models/feature_semantics.py"


class ProjectionMode(str, Enum):
    NATIVE_NAN = "native_nan"
    FOLD_MEDIAN = "fold_median"
    AVAILABILITY_BLINDED_DRAW = "availability_blinded_draw"


#: Modes whose output must be finite everywhere.
FINITE_MODES: frozenset = frozenset({ProjectionMode.FOLD_MEDIAN, ProjectionMode.AVAILABILITY_BLINDED_DRAW})


class SingleFeatureTreatment(str, Enum):
    FOLD_MEDIAN_WHEN_FINITE_REQUIRED = "fold_median_when_finite_required"
    NATIVE_ONLY_PENDING_RULING = "native_only_pending_ruling"


@dataclass(frozen=True)
class DeclaredMissingnessV2:
    treatment: SingleFeatureTreatment
    rationale: str


#: Missingness OUTSIDE a derivation family that version 2 accepts. No indicator in any case.
DECLARED_MISSINGNESS_V2: Mapping[str, DeclaredMissingnessV2] = {
    "gene_constraint_oe": DeclaredMissingnessV2(
        SingleFeatureTreatment.FOLD_MEDIAN_WHEN_FINITE_REQUIRED,
        "gnomAD pLoF observed/expected; continuous and with no registered dependent, so a fold-fitted median of the "
        "observed values is the least assertive finite carrier. A native estimator receives NaN."),
    "gene_is_constrained": DeclaredMissingnessV2(
        SingleFeatureTreatment.NATIVE_ONLY_PENDING_RULING,
        "a STRICT function of loeuf, whose model feature is filled with 1.0 (LOEUF-SENTINEL-1), so the family cannot be "
        "projected jointly; a structural zero without its mask would assert 'not constrained'. Refused for a "
        "finite-requiring member until the owner rules (AUDIT_OF_RULING_1D1c1C_2026-11-16, section 6)."),
}


def _repo_root() -> _Path:
    return _Path(__file__).resolve().parents[3]


@_functools.lru_cache(maxsize=None)
def _closure(symbol: str) -> str:
    return symbol_closure_digest(_repo_root(), symbol)


def projection_code_identity(families: tuple) -> dict:
    """Symbol-closure digests of the code a fitted projection executes: the projection, the family derivation and every
    formula its families call (``derive_family`` reaches the formulas through the family, not by name, so each is bound
    explicitly). A code change is a different projection. Computed from the source files at fit and again at every
    transform, so an image or installation without the source refuses to serve, loudly, rather than serving unbound."""
    symbols = [f"{_THIS_FILE}::MemberProjection", f"{_SEMANTICS_FILE}::derive_family",
               f"{_SEMANTICS_FILE}::DerivationFamily"]
    for family in families:
        symbols += [f"{_SEMANTICS_FILE}::{name}" for name in sorted(set(family.implementation_names().values()))]
    return {s: _closure(s) for s in symbols}


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix64(z: np.ndarray) -> np.ndarray:
    """The splitmix64 finalizer; uint64 arithmetic wraps identically on every platform."""
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def row_content_keys(values: np.ndarray, seed: int) -> np.ndarray:
    """One 64-bit key per row from the bit patterns of ``values`` (rows x columns): independent of the batch a row
    arrives in and identical at training and serving. NaN is canonicalised and -0.0 becomes +0.0 first."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("row keys need a two-dimensional array")
    x = np.where(np.isnan(x), np.nan, x + 0.0)
    bits = np.ascontiguousarray(x, dtype="<f8").view("<u8")
    h = np.full(len(x), np.uint64(seed), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(bits.shape[1]):
            h = _mix64(h ^ (bits[:, j] + _GOLDEN))
    return h


def blinded_surrogates(sorted_observed: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Draws from the empirical distribution ``sorted_observed``, one per key (uniform on [0, 1) from the top 53 bits)."""
    n = len(sorted_observed)
    u = (keys >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)
    return sorted_observed[np.minimum((u * n).astype(np.int64), n - 1)]


class MemberProjection:
    """Canonical semantic matrix -> one member's model input, fitted on that member's fitting rows only (version 2).

    scikit-learn-cloneable (``get_params``/``set_params`` over the three constructor arguments), so every out-of-fold
    clone fits its own statistics. Not a subclass of TransformerMixin: ``fit_transform`` is not offered, because a member
    fits and transforms in separate, recorded steps.
    """

    def __init__(self, feature_names: tuple, mode: str, semantics: str) -> None:
        self.feature_names = feature_names
        self.mode = mode
        self.semantics = semantics

    # -- scikit-learn parameter protocol (clone) -----------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {"feature_names": self.feature_names, "mode": self.mode, "semantics": self.semantics}

    def set_params(self, **params) -> "MemberProjection":
        unknown = set(params) - {"feature_names", "mode", "semantics"}
        if unknown:
            raise ValueError(f"unknown MemberProjection parameter(s) {sorted(unknown)}")
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        return (f"MemberProjection(mode={self.mode!r}, semantics={self.semantics!r}, "
                f"features={len(tuple(self.feature_names))})")

    # -- validation ------------------------------------------------------------------------------------------------
    def _resolved(self) -> tuple[ProjectionMode, FeatureSemantics, tuple]:
        mode = ProjectionMode(self.mode)
        semantics = FeatureSemantics(self.semantics)
        names = tuple(self.feature_names)
        if not names or any(type(n) is not str or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("feature_names must be unique non-empty str")
        return mode, semantics, names

    def _validate(self, X) -> tuple[np.ndarray, list]:
        """The canonical matrix as float64 (a fresh array), and the families present. Refuses everything else."""
        mode, semantics, names = self._resolved()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("a member projection requires a named DataFrame; an anonymous array cannot be verified")
        if tuple(str(c) for c in X.columns) != names or any(type(c) is not str for c in X.columns):
            raise ValueError("feature names or order differ from the fitted roster")
        found = infer_semantics(names)
        if found is not None and found is not semantics:
            raise ValueError(f"the columns belong to {found.value}, not the declared {semantics.value}")
        for c in names:
            if not pd.api.types.is_numeric_dtype(X[c].dtype) or pd.api.types.is_bool_dtype(X[c].dtype):
                raise TypeError(f"{c!r} is not a numeric feature (dtype {X[c].dtype})")
        values = X.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        if np.isinf(values).any():
            bad = sorted({names[j] for j in np.nonzero(np.isinf(values).any(axis=0))[0]})
            raise ValueError(f"infinite values in {bad}; refused")
        index = {n: i for i, n in enumerate(names)}
        families = []
        for family in PROJECTABLE_FAMILIES[semantics]:
            present = [m for m in family.members if m in index]
            if not present:
                continue
            if present != list(family.members):
                raise ValueError(f"partial {family.family_id} family {present}: a family is projected whole or not at all")
            expected = derive_family(family, values[:, index[family.root]])
            for m in family.members:
                if not np.array_equal(values[:, index[m]], expected[m], equal_nan=True):
                    n_bad = int((~((values[:, index[m]] == expected[m])
                                   | (np.isnan(values[:, index[m]]) & np.isnan(expected[m])))).sum())
                    raise ValueError(f"{m}: {n_bad} row(s) disagree with its registered derivation from {family.root}")
            families.append(family)
        in_family = {m for f in families for m in f.members}
        nan_cols = [names[j] for j in np.nonzero(np.isnan(values).any(axis=0))[0] if names[j] not in in_family]
        undeclared = [c for c in nan_cols if c not in DECLARED_MISSINGNESS_V2]
        if undeclared:
            raise UndeclaredMissingnessError(
                f"{undeclared} carry missing values with no declared version-2 policy; refused (fail closed)")
        if mode in FINITE_MODES:
            pending = [c for c in nan_cols
                       if DECLARED_MISSINGNESS_V2[c].treatment is SingleFeatureTreatment.NATIVE_ONLY_PENDING_RULING]
            if pending:
                raise UndeclaredMissingnessError(
                    f"{pending} carry missing values that no finite policy is ruled for "
                    f"({DECLARED_MISSINGNESS_V2[pending[0]].rationale})")
        return values, families

    # -- fit / transform -------------------------------------------------------------------------------------------
    def fit(self, X, y=None) -> "MemberProjection":
        mode, semantics, names = self._resolved()
        values, families = self._validate(X)
        index = {n: i for i, n in enumerate(names)}
        stats: dict = {}
        observed_counts: dict = {}
        self._sorted_observed_ = {}
        for family in families:
            root = values[:, index[family.root]]
            obs = root[~np.isnan(root)]
            observed_counts[family.family_id] = int(obs.size)
            if mode in FINITE_MODES and obs.size == 0:
                raise UndeclaredMissingnessError(
                    f"{family.root} is entirely unobserved in this member's fitting rows; no statistic can be fitted "
                    "(a fitted constant would be invented). The AF-free arm removes the family from the roster instead "
                    "(feature_semantics.roster_without_family).")
            if mode is ProjectionMode.FOLD_MEDIAN:
                stats[family.family_id] = {"median_binary64_hex": float(np.median(obs)).hex()}
            elif mode is ProjectionMode.AVAILABILITY_BLINDED_DRAW:
                sorted_obs = np.sort(obs)
                sorted_obs.setflags(write=False)
                self._sorted_observed_[family.family_id] = sorted_obs
                stats[family.family_id] = {
                    "observed_sorted_float64_sha256": hashlib.sha256(sorted_obs.astype("<f8").tobytes()).hexdigest(),
                    "n_observed": int(obs.size)}
        in_family = {m for f in families for m in f.members}
        if mode is ProjectionMode.AVAILABILITY_BLINDED_DRAW and families and len(names) == len(in_family):
            raise ValueError("an availability-blinded draw keys each row by its non-family features; there are none")
        single: dict = {}
        if mode in FINITE_MODES:
            for c, policy in sorted(DECLARED_MISSINGNESS_V2.items()):
                if c not in index or policy.treatment is not SingleFeatureTreatment.FOLD_MEDIAN_WHEN_FINITE_REQUIRED:
                    continue
                col = values[:, index[c]]
                obs = col[~np.isnan(col)]
                if obs.size == 0:
                    raise UndeclaredMissingnessError(
                        f"{c!r} is entirely missing in this member's fitting rows; no median exists")
                single[c] = float(np.median(obs)).hex()
        self.mode_ = mode
        self.semantics_ = semantics
        self.families_ = tuple(families)
        self.input_features_ = names
        renames = {m: f.model_name_map()[m] for f in families for m in f.members}
        self.output_features_ = tuple(renames.get(n, n) for n in names)
        self.statistics_ = stats
        self.single_feature_medians_hex_ = single
        self.n_fit_rows_ = int(len(values))
        self.n_observed_ = observed_counts
        pre = canonical_digest("gvc.member-projection-statistics.v1", {
            "schema": PROJECTION_SCHEMA, "mode": mode.value, "semantics": semantics.value, "input": list(names),
            "statistics": stats, "single_feature_medians_hex": single, "fit_rows": self.n_fit_rows_,
            "observed": observed_counts})
        self.draw_seed_ = int(pre[:16], 16)
        if mode is ProjectionMode.AVAILABILITY_BLINDED_DRAW:
            other = [index[n] for n in names if n not in in_family]
            unobserved = np.zeros(len(values), dtype=bool)
            for family in families:
                unobserved |= np.isnan(values[:, index[family.root]])
            keys = row_content_keys(values[unobserved][:, other], self.draw_seed_)
            self.blinded_distinct_key_fraction_ = (float(len(np.unique(keys)) / len(keys)) if len(keys) else None)
        self.identity_ = self._identity()
        return self

    def _identity(self) -> str:
        return canonical_digest("gvc.member-projection-identity.v1", {
            "schema": PROJECTION_SCHEMA, "mode": self.mode_.value, "semantics": self.semantics_.value,
            "input_features": list(self.input_features_), "output_features": list(self.output_features_),
            "families": [f.as_dict() for f in self.families_],
            "declared_missingness_v2": {k: v.treatment.value for k, v in sorted(DECLARED_MISSINGNESS_V2.items())},
            "statistics": self.statistics_, "single_feature_medians_hex": self.single_feature_medians_hex_,
            "fit_rows": self.n_fit_rows_, "observed": self.n_observed_, "draw_seed": self.draw_seed_,
            "code": projection_code_identity(self.families_)})

    def transform(self, X) -> pd.DataFrame:
        if not hasattr(self, "identity_"):
            raise RuntimeError("the member projection has not been fitted")
        if self._identity() != self.identity_:
            raise RuntimeError("the fitted projection's state no longer matches its recorded identity")
        values, families = self._validate(X)
        if tuple(f.family_id for f in families) != tuple(f.family_id for f in self.families_):
            raise ValueError("the families present differ from the fitted ones")
        names = self.input_features_
        index = {n: i for i, n in enumerate(names)}
        in_family = {m for f in families for m in f.members}
        out = values
        if self.mode_ is not ProjectionMode.NATIVE_NAN:
            canonical = values.copy() if self.mode_ is ProjectionMode.AVAILABILITY_BLINDED_DRAW else None
            for family in families:
                j = index[family.root]
                root = values[:, j].copy()
                missing = np.isnan(root)
                if missing.any():
                    if self.mode_ is ProjectionMode.FOLD_MEDIAN:
                        root[missing] = float.fromhex(self.statistics_[family.family_id]["median_binary64_hex"])
                    else:
                        other = [index[n] for n in names if n not in in_family]
                        keys = row_content_keys(canonical[missing][:, other], self.draw_seed_)
                        root[missing] = blinded_surrogates(self._sorted_observed_[family.family_id], keys)
                derived = derive_family(family, root)
                for m in family.members:
                    out[:, index[m]] = derived[m]
            for c, hexval in self.single_feature_medians_hex_.items():
                col = out[:, index[c]]
                col[np.isnan(col)] = float.fromhex(hexval)
            if not np.isfinite(out).all():
                bad = sorted({names[j] for j in np.nonzero(~np.isfinite(out).all(axis=0))[0]})
                raise ValueError(f"a finite-required projection left non-finite values in {bad}")
        return pd.DataFrame(out, columns=list(self.output_features_), index=X.index)

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "output_features_"):
            raise RuntimeError("the member projection has not been fitted")
        if input_features is not None and tuple(input_features) != self.input_features_:
            raise ValueError("input feature names differ from the fitted roster")
        return np.asarray(self.output_features_, dtype=object)

    def describe(self) -> dict:
        """The fitted projection as values (never a pickle): what an artifact record carries."""
        if not hasattr(self, "identity_"):
            raise RuntimeError("the member projection has not been fitted")
        return {"identity": self.identity_, "schema": PROJECTION_SCHEMA, "mode": self.mode_.value,
                "semantics": self.semantics_.value, "families": [f.family_id for f in self.families_],
                "statistics": self.statistics_, "single_feature_medians_hex": dict(self.single_feature_medians_hex_),
                "fit_rows": self.n_fit_rows_, "observed": dict(self.n_observed_),
                "output_features": list(self.output_features_)}
