"""Split (inductive) conformal classifier -- the base engine, from scratch.

Given calibration probabilities P_cal (n_cal, K) with true labels y_cal, and a target
miscoverage alpha, we compute a threshold q_hat = the ceil((n+1)(1-alpha))/n empirical quantile
of the calibration nonconformity scores of the TRUE labels. For a test point, the prediction set
is every class whose nonconformity score is <= q_hat.

Under exchangeability of calibration and test, marginal coverage P(y_test in set) >= 1 - alpha
holds in finite samples (Vovk et al.). The (n+1) correction is essential and is unit-tested.
"""
from __future__ import annotations

import numpy as np

from .scores import (lac_scores_true, lac_scores_all,
                    aps_scores_true, aps_scores_all,
                    raps_scores_true, raps_scores_all)


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected conformal threshold.

    q_hat = the ceil((n+1)(1-alpha)) / n empirical quantile of cal_scores. If the corrected level
    exceeds 1 (happens when (n+1)(1-alpha) > n, i.e. alpha < 1/(n+1)), the set must include all
    classes -> return +inf so every score passes.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    n = len(cal_scores)
    if n == 0:
        raise ValueError("empty calibration set")
    # The conformal threshold is the k-th smallest score (1-indexed), with
    # k = ceil((n+1)(1-alpha)). We take the order statistic DIRECTLY rather than routing
    # through np.quantile, whose linear interpolation over [0,1] does NOT coincide with the
    # k-th order statistic (it uses position level*(n-1), an off-by-one relative to the
    # conformal definition). If k > n (i.e. alpha < 1/(n+1)) no finite threshold covers the
    # required level -> return +inf so every class enters the set.
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(cal_scores)[k - 1])


class SplitConformalClassifier:
    """Marginal split-conformal classifier over a probability model.

    score: 'lac' | 'aps' | 'raps'. Fit on calibration (P_cal, y_cal); predict sets on P_test.
    """

    def __init__(self, alpha: float = 0.1, score: str = "aps", randomize: bool = True,
                 seed: int = 0, k_reg: int = 1, lam: float = 0.0):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        self.alpha = alpha
        self.score = score
        self.randomize = randomize
        self.seed = seed
        self.k_reg = k_reg
        self.lam = lam
        self.q_hat_: float | None = None
        self.K_: int | None = None

    def _true_scores(self, P, y, rng):
        if self.score == "lac":
            return lac_scores_true(P, y)
        u = rng.uniform(size=len(y)) if self.randomize else None
        if self.score == "aps":
            return aps_scores_true(P, y, u=u, randomize=self.randomize)
        if self.score == "raps":
            return raps_scores_true(P, y, u=u, randomize=self.randomize,
                                    k_reg=self.k_reg, lam=self.lam)
        raise ValueError(f"unknown score {self.score!r}")

    def _all_scores(self, P, rng):
        if self.score == "lac":
            return lac_scores_all(P)
        u = rng.uniform(size=P.shape[0]) if self.randomize else None
        if self.score == "aps":
            return aps_scores_all(P, u=u, randomize=self.randomize)
        if self.score == "raps":
            return raps_scores_all(P, u=u, randomize=self.randomize,
                                   k_reg=self.k_reg, lam=self.lam)
        raise ValueError(f"unknown score {self.score!r}")

    def fit(self, P_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalClassifier":
        P_cal = np.asarray(P_cal, dtype=float)
        self.K_ = P_cal.shape[1]
        rng = np.random.default_rng(self.seed)
        cal_scores = self._true_scores(P_cal, y_cal, rng)
        self.q_hat_ = conformal_quantile(cal_scores, self.alpha)
        return self

    def predict_set(self, P_test: np.ndarray) -> np.ndarray:
        """Boolean matrix (n_test, K): True where class is in the prediction set."""
        if self.q_hat_ is None:
            raise RuntimeError("call fit() first")
        P_test = np.asarray(P_test, dtype=float)
        rng = np.random.default_rng(self.seed + 1)
        S = self._all_scores(P_test, rng)
        return S <= self.q_hat_

    def predict_p_values(self, P_test: np.ndarray) -> np.ndarray:
        """Marginal conformal p-value per class is not returned here (needs cal score ECDF);
        provided in coverage/calibrate layer. Placeholder to keep the interface explicit."""
        raise NotImplementedError("conformal p-values are computed in calibrate.py")
