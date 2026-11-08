"""
src/genomic_variant_classifier/models/mc_dropout.py
========================
Monte Carlo dropout and deep-ensemble wrappers, with schema-2 uncertainty records.

Author: Monzia Moodie

Two mechanisms produce a finite, equally weighted set of Bernoulli draws per variant:

1. MCDropoutWrapper -- a network with dropout layers run T times with dropout ACTIVE, each pass under a recorded seed.
2. DeepEnsembleWrapper -- M independently seeded clones of a base estimator.

predict_proba() returns the mean draw, computed exactly as before this module recorded uncertainty (same seed schedule,
same precision), so a fitted member's predictions are unchanged. That matters: Run 15's stored blend weights were
fitted on these members' outputs.

Schema-2 uncertainty (package 1, gate G5; models/disagreement.py) replaces the 2026-05 decomposition:
  * two exact identities, each in its own unit, never summed -- entropy in nats, variance in probability squared;
  * the retired names "epistemic" (draw variance) and "aleatoric" (mean draw entropy) are gone: for a finite draw set
    they are the between-draw spread and the expected within-draw entropy, not population quantities;
  * the mixed-unit "total" (variance + entropy) and the triage flag thresholded on it are removed (G5.1);
  * every record states the supported, requested and applied randomness, the ordered draw identities (seeds or
    member indices), whether the draws are distinct, and degeneracy -- an estimator with no stochastic forward pass
    yields an UNAVAILABLE uncertainty with a reason, never a computed-looking zero (G5.2);
  * nothing is clipped; draws outside [0, 1] are refused.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from genomic_variant_classifier.models.disagreement import draw_decomposition
from genomic_variant_classifier.prediction_contract import (
    ContractError,
    model_call,
    positive_column,
    sklearn_binary_classes,
)

logger = logging.getLogger(__name__)

__all__ = ["DeepEnsembleWrapper", "MCDropoutWrapper", "annotate_uncertainty", "uncertainty_record"]


def _require_draw_count(value: Any, name: str) -> int:
    """A mixture needs at least two draws; bool and every non-int are refused (no coercion)."""
    if type(value) is not int or value < 2:
        raise ContractError(f"{name} must be a Python int >= 2, got {value!r}")
    return value


def _require_binary_labels(y: Any) -> np.ndarray:
    classes = np.unique(np.asarray(y))
    if classes.dtype.kind not in "iu" or classes.tolist() != [0, 1]:
        raise ContractError(f"binary integer labels {{0, 1}} required, got classes {classes.tolist()!r}")
    return classes.astype(np.int64)


def uncertainty_record(stack: np.ndarray, *, mechanism: str, supported_randomness: str, requested_randomness: str,
                       applied_randomness: str, draw_identities: list, draws_distinct: bool,
                       degeneracy: str | None) -> dict[str, Any]:
    """A schema-2 record for one wrapper call. When the draws are not a valid mixture the uncertainty is UNAVAILABLE
    with a reason; it is never reported as zero."""
    record: dict[str, Any] = {
        "schema": "gvc.uncertainty.schema-2",
        "mechanism": mechanism,
        "supported_randomness": supported_randomness,
        "requested_randomness": requested_randomness,
        "applied_randomness": applied_randomness,
        "draw_identities": list(draw_identities),
        "draws_distinct": bool(draws_distinct),
        "degeneracy": degeneracy,
    }
    if degeneracy is not None:
        record["status"] = "unavailable"
        record["reason"] = degeneracy
        return record
    record["status"] = "defined"
    record["reason"] = None
    record.update(draw_decomposition(stack))
    return record


# ---------------------------------------------------------------------------
# Deep Ensemble Wrapper
# ---------------------------------------------------------------------------

class DeepEnsembleWrapper(BaseEstimator, ClassifierMixin):
    """
    Deep ensemble over M independently trained clones of *base_estimator*.

    Member k is seeded with random_state + k through the estimator's own `random_state` (or `seed`) parameter. An
    estimator exposing neither cannot be seeded: its members are identical by construction, which is RECORDED as
    degeneracy (`seed_mechanism_ == "none"`), so the member spread is unavailable rather than a misleading zero.

    Parameters
    ----------
    base_estimator : sklearn estimator with fit() and predict_proba()
    n_members : int >= 2
    random_state : int
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_members: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_members = n_members
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> "DeepEnsembleWrapper":
        n_members = _require_draw_count(self.n_members, "n_members")
        self.classes_ = _require_binary_labels(y)
        self.members_: list[BaseEstimator] = []
        params = clone(self.base_estimator).get_params()
        if "random_state" in params:
            self.seed_mechanism_ = "random_state"
        elif "seed" in params:
            self.seed_mechanism_ = "seed"
        else:
            self.seed_mechanism_ = "none"
            logger.warning(
                "DeepEnsemble: %s exposes neither random_state nor seed; its %d members are identical by "
                "construction and the member spread will be reported UNAVAILABLE.",
                type(self.base_estimator).__name__, n_members)
        self.member_seeds_: list[int | None] = []
        for k in range(n_members):
            member = clone(self.base_estimator)
            seed = None if self.seed_mechanism_ == "none" else int(self.random_state) + k
            if seed is not None:
                member.set_params(**{self.seed_mechanism_: seed})
            logger.info("DeepEnsemble: fitting member %d/%d ...", k + 1, n_members)
            member.fit(X, y, **fit_kwargs)
            self.members_.append(member)
            self.member_seeds_.append(seed)
        return self

    def predict_draws(self, X: np.ndarray) -> np.ndarray:
        """(n_members, n_samples) positive-class probabilities, each member's classes validated."""
        check_is_fitted(self, "members_")
        n = len(X)
        draws = []
        for member in self.members_:
            p, _classes, _conversion = positive_column(member, member.predict_proba(X), n)
            draws.append(p)
        return np.array(draws)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mean member probability, shape (n_samples, 2), in the members' own precision (unchanged behaviour)."""
        mean_p = self.predict_draws(X).mean(axis=0)
        return np.column_stack([1.0 - mean_p, mean_p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return model_call(self.predict_proba(X)[:, 1])

    def predict_with_uncertainty(self, X: np.ndarray) -> dict[str, Any]:
        """The producer mean plus a schema-2 record over the member draws."""
        draws = self.predict_draws(X)
        seeds = list(getattr(self, "member_seeds_", [None] * len(self.members_)))
        mechanism = getattr(self, "seed_mechanism_", "unrecorded")
        distinct = mechanism in ("random_state", "seed") and len(set(seeds)) == len(seeds)
        degeneracy = None
        if mechanism == "none":
            degeneracy = "members_not_seeded_identical_by_construction"
        elif mechanism == "unrecorded":
            degeneracy = "member_seeds_not_recorded_by_this_fitted_object"
        elif not distinct:
            degeneracy = "duplicate_member_seeds"
        record = uncertainty_record(
            draws, mechanism="fitted_ensemble",
            supported_randomness="independent_member_initialisation",
            requested_randomness=f"{len(self.members_)}_members_seeded_{mechanism}",
            applied_randomness="none" if mechanism == "none" else f"{mechanism}=random_state+k",
            draw_identities=[{"member_index": k, "seed": s} for k, s in enumerate(seeds)],
            draws_distinct=distinct, degeneracy=degeneracy)
        record["producer_mean"] = draws.mean(axis=0)
        return record


# ---------------------------------------------------------------------------
# MC Dropout Wrapper (neural networks)
# ---------------------------------------------------------------------------

class MCDropoutWrapper(BaseEstimator, ClassifierMixin):
    """
    Monte Carlo dropout over an estimator exposing ``_predict_proba_single_pass(X, seed=...)``.

    An estimator without that method has no stochastic forward pass: predict_proba() is then its deterministic
    prediction, and predict_with_uncertainty() reports the uncertainty UNAVAILABLE (never zero).

    The seed schedule is the historical one -- numpy default_rng(random_state).integers(0, 2**31, size=n_passes) --
    so the mean is unchanged for fitted members. A repeated seed is RECORDED (draws_distinct False) and makes the
    draw set unavailable as an uncertainty estimate; the mean is not altered.

    Parameters
    ----------
    base_estimator : estimator with predict_proba
    n_passes : int >= 2
    random_state : int
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_passes: int = 50,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_passes = n_passes
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> "MCDropoutWrapper":
        _require_draw_count(self.n_passes, "n_passes")
        self.classes_ = _require_binary_labels(y)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, **fit_kwargs)
        self._supports_mc = hasattr(self.estimator_, "_predict_proba_single_pass")
        if not self._supports_mc:
            logger.warning(
                "MCDropoutWrapper: %s does not expose _predict_proba_single_pass(); predictions are its "
                "deterministic output and the uncertainty is UNAVAILABLE. Consider DeepEnsembleWrapper.",
                type(self.estimator_).__name__,
            )
        return self

    def draw_seeds(self) -> list[int]:
        """The ordered pass seeds as Python ints (the historical schedule)."""
        n_passes = _require_draw_count(self.n_passes, "n_passes")
        rng = np.random.default_rng(self.random_state)
        return rng.integers(0, 2**31, size=n_passes).tolist()

    def predict_draws(self, X: np.ndarray) -> np.ndarray:
        """(n_passes, n_samples) positive-class probabilities with dropout active. Refused without MC support."""
        check_is_fitted(self, "estimator_")
        if not self._supports_mc:
            raise ContractError(f"{type(self.estimator_).__name__} has no stochastic forward pass; there are no "
                                "MC draws to return")
        classes, _conversion = sklearn_binary_classes(self.estimator_)
        n = len(X)
        passes = []
        for seed in self.draw_seeds():
            p = np.asarray(self.estimator_._predict_proba_single_pass(X, seed=int(seed)))
            if p.ndim == 1:
                if p.shape != (n,):
                    raise ContractError(f"single pass returned shape {p.shape} for {n} rows")
                passes.append(p)
            else:
                passes.append(positive_column(self.estimator_, p, n)[0])
        return np.array(passes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        if not self._supports_mc:
            positive, _c, _conv = positive_column(self.estimator_, self.estimator_.predict_proba(X), len(X))
            mean_p = positive
        else:
            mean_p = self.predict_draws(X).mean(axis=0)
        return np.column_stack([1.0 - mean_p, mean_p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return model_call(self.predict_proba(X)[:, 1])

    def predict_with_uncertainty(self, X: np.ndarray) -> dict[str, Any]:
        """The producer mean plus a schema-2 record over the MC passes (or an unavailable record)."""
        check_is_fitted(self, "estimator_")
        if not self._supports_mc:
            mean_p = self.predict_proba(X)[:, 1]
            record = uncertainty_record(
                np.empty((0, 0)), mechanism="mc_dropout", supported_randomness="none",
                requested_randomness=f"{self.n_passes}_dropout_passes", applied_randomness="none",
                draw_identities=[], draws_distinct=False,
                degeneracy="estimator_has_no_stochastic_forward_pass")
            record["producer_mean"] = mean_p
            return record
        seeds = self.draw_seeds()
        draws = self.predict_draws(X)
        distinct = len(set(seeds)) == len(seeds)
        record = uncertainty_record(
            draws, mechanism="mc_dropout", supported_randomness="dropout_masks_with_batchnorm_in_eval",
            requested_randomness=f"{len(seeds)}_dropout_passes",
            applied_randomness="torch.manual_seed(seed)_per_pass",
            draw_identities=[{"pass_index": k, "seed": s} for k, s in enumerate(seeds)],
            draws_distinct=distinct, degeneracy=None if distinct else "duplicate_pass_seeds")
        record["producer_mean"] = draws.mean(axis=0)
        return record


# ---------------------------------------------------------------------------
# Convenience: annotate a DataFrame with schema-2 columns
# ---------------------------------------------------------------------------

def annotate_uncertainty(
    wrapper: DeepEnsembleWrapper | MCDropoutWrapper,
    df: "pd.DataFrame",
    feature_cols: list[str],
) -> "pd.DataFrame":
    """
    Return a copy of *df* with schema-2 uncertainty columns, each named with its unit.

    The features are passed in their own dtype (no float32 cast). There is no "total" and no triage flag: a variance
    and an entropy cannot be added, and no qualified decision policy exists for this record.
    """
    record = wrapper.predict_with_uncertainty(df[feature_cols].to_numpy())
    out = df.copy()
    out["pathogenicity_mean"] = record["producer_mean"]
    out["uncertainty_status"] = record["status"]
    out["uncertainty_reason"] = record["reason"]
    if record["status"] == "defined":
        out["expected_draw_entropy_nats"] = record["expected_draw_entropy_nats"]
        out["mixture_predictive_entropy_nats"] = record["mixture_predictive_entropy_nats"]
        out["disagreement_nats"] = record["disagreement_nats"]
        out["disagreement_status"] = record["disagreement_status"]
        out["between_draw_spread_variance_probability_squared"] = (
            record["between_draw_spread_variance_probability_squared"])
        out["expected_within_draw_variance_probability_squared"] = (
            record["expected_within_draw_variance_probability_squared"])
    return out


try:  # avoid a hard module-level pandas dependency; only annotate_uncertainty's annotation needs it
    import pandas as pd  # noqa: F401
except ImportError:  # pragma: no cover - pandas is a core dependency of this package
    pass
