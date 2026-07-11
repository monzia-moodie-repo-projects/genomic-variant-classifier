"""Grouped conformal prediction for the gene/sample-set unit (from scratch, numpy).

Variant-level conformal treats each variant as the exchangeable unit. But GenAssoc's split is
gene-DISJOINT, and a clinical question is often posed at the GENE or SAMPLE-SET level ("does this
gene-set contain a pathogenic variant?"). The exchangeable unit then becomes the GROUP, not the
variant, and calibrating per-variant would violate exchangeability for a gene-level claim.

This module provides two coherent, documented constructions:

  (1) GROUP-AGGREGATE conformal: reduce each group's member variants to a single group score via
      an aggregator (max / mean / quantile), then run split-conformal at the GROUP level. Yields a
      set-valued statement about the GROUP with group-level coverage 1 - alpha.

  (2) EXCHANGEABILITY-CORRECT per-gene calibration: when deployment is gene-disjoint, treat GENES
      as the exchangeable unit -- one aggregated score per gene -- so the coverage guarantee is
      over genes, matching how the model is actually split and used.

Both reuse the proven split-conformal quantile (finite-sample (n+1) correction) from split.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .split import conformal_quantile
from .scores import lac_scores_true, lac_scores_all, aps_scores_true, aps_scores_all


def _agg(values: np.ndarray, how: str, q: float = 0.9) -> float:
    if how == "max":
        return float(np.max(values))
    if how == "mean":
        return float(np.mean(values))
    if how == "quantile":
        return float(np.quantile(values, q))
    raise ValueError(f"unknown aggregator {how!r}")


class GroupedConformalClassifier:
    """Group-level split conformal over a probability model.

    group_agg: 'max' | 'mean' | 'quantile' -- how member-variant true-label scores reduce to one
               group calibration score. 'max' is the most conservative (a group conforms only if
               its WORST member does); 'mean' is average behaviour; 'quantile' is tunable.
    score: 'lac' | 'aps' (binary/multiclass-general).
    """

    def __init__(self, alpha: float = 0.1, score: str = "aps", group_agg: str = "max",
                 q: float = 0.9, randomize: bool = True, seed: int = 0):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        self.alpha = alpha
        self.score = score
        self.group_agg = group_agg
        self.q = q
        self.randomize = randomize
        self.seed = seed
        self.q_hat_: float | None = None
        self.K_: int | None = None
        self.n_groups_cal_: int | None = None

    def _true_scores(self, P, y, rng):
        if self.score == "lac":
            return lac_scores_true(P, y)
        u = rng.uniform(size=len(y)) if self.randomize else None
        return aps_scores_true(P, y, u=u, randomize=self.randomize)

    def _all_scores(self, P, rng):
        if self.score == "lac":
            return lac_scores_all(P)
        u = rng.uniform(size=P.shape[0]) if self.randomize else None
        return aps_scores_all(P, u=u, randomize=self.randomize)

    def fit(self, P_cal: np.ndarray, y_cal: np.ndarray, groups_cal: np.ndarray):
        """Calibrate at the GROUP level. groups_cal: array of group ids (e.g. gene_symbol),
        aligned with the calibration rows."""
        P_cal = np.asarray(P_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        groups_cal = np.asarray(groups_cal)
        self.K_ = P_cal.shape[1]
        rng = np.random.default_rng(self.seed)
        var_scores = self._true_scores(P_cal, y_cal, rng)
        # reduce to one score per group
        df = pd.DataFrame({"g": groups_cal, "s": var_scores})
        group_scores = df.groupby("g")["s"].apply(lambda v: _agg(v.values, self.group_agg, self.q))
        self.n_groups_cal_ = len(group_scores)
        self.q_hat_ = conformal_quantile(group_scores.values, self.alpha)
        return self

    def predict_group_set(self, P_test: np.ndarray, groups_test: np.ndarray) -> dict:
        """For each test GROUP, the set of classes whose AGGREGATED member score <= q_hat.
        Returns {group_id: boolean array of length K}.

        COVERAGE SEMANTICS (important): the finite-sample guarantee is that, for an exchangeable
        test group, the group's aggregated TRUE-LABEL score is <= q_hat with probability
        >= 1 - alpha (evaluate with group_true_score below). The set returned here contains every
        class whose aggregated score clears the same threshold; a class is 'in' the group set if
        the group conforms for that class. Do NOT evaluate coverage by inventing a group label
        such as max(member y) -- that is not the calibrated quantity."""
        if self.q_hat_ is None:
            raise RuntimeError("call fit() first")
        P_test = np.asarray(P_test, dtype=float)
        groups_test = np.asarray(groups_test)
        rng = np.random.default_rng(self.seed + 1)
        S_all = self._all_scores(P_test, rng)          # (n, K) per-variant class scores
        out = {}
        for g in pd.unique(groups_test):
            mask = groups_test == g
            # aggregate each class's member scores the same way as calibration
            agg_scores = np.array([_agg(S_all[mask, k], self.group_agg, self.q)
                                   for k in range(self.K_)])
            out[g] = agg_scores <= self.q_hat_
        return out

    def group_true_score(self, P_test: np.ndarray, y_test: np.ndarray,
                         groups_test: np.ndarray) -> pd.Series:
        """The aggregated TRUE-LABEL nonconformity score per group -- the quantity whose
        coverage (fraction with score <= q_hat) is guaranteed >= 1 - alpha. Use this to
        MEASURE group-level coverage correctly."""
        P_test = np.asarray(P_test, dtype=float)
        y_test = np.asarray(y_test, dtype=int)
        groups_test = np.asarray(groups_test)
        rng = np.random.default_rng(self.seed + 2)
        s = self._true_scores(P_test, y_test, rng)
        return pd.DataFrame({"g": groups_test, "s": s}).groupby("g")["s"].apply(
            lambda v: _agg(v.values, self.group_agg, self.q))

    def group_coverage(self, P_test, y_test, groups_test) -> float:
        """Empirical group-level coverage: fraction of test groups whose aggregated true-label
        score is <= q_hat. Should be >= 1 - alpha under exchangeability."""
        gs = self.group_true_score(P_test, y_test, groups_test)
        return float(np.mean(gs.values <= self.q_hat_))
