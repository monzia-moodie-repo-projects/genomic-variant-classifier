"""Mondrian (group-conditional) conformal classification, from scratch.

Standard split conformal guarantees only MARGINAL coverage P(y in set) >= 1 - alpha. In a
clinical setting that is not enough: marginal coverage can be met while the rare/important class
(pathogenic) is systematically under-covered, because the abundant class dominates the calibration
quantile. Mondrian conformal fixes this by computing a SEPARATE conformal threshold within each
group g, guaranteeing coverage >= 1 - alpha WITHIN every group.

Two grouping modes (composable):
  * class-conditional: group = the true class label. Guarantees per-class coverage. This is the
    clinically essential default (pathogenic coverage not traded away for benign).
  * stratum-conditional: group = a categorical stratum (e.g. ClinVar review status, gene-
    constraint decile, SNV/indel, molecular consequence). Guarantees coverage within each stratum.

The two can be combined (group = (class, stratum)). Each group needs enough calibration points;
groups below a floor fall back to the pooled (marginal) threshold, and this fallback is RECORDED
(never silent) so the caller can see which groups were too small for their own guarantee.
"""
from __future__ import annotations

import numpy as np

from .split import conformal_quantile
from .scores import (lac_scores_true, lac_scores_all,
                    aps_scores_true, aps_scores_all,
                    raps_scores_true, raps_scores_all)


class MondrianConformalClassifier:
    """Group-conditional split conformal.

    group_mode: 'class' | 'stratum' | 'class_stratum'.
    min_group: minimum calibration count for a group to get its own threshold; smaller groups
               fall back to the pooled threshold (recorded in fallback_groups_).
    """

    def __init__(self, alpha: float = 0.1, score: str = "aps", group_mode: str = "class",
                 min_group: int = 50, randomize: bool = True, seed: int = 0,
                 k_reg: int = 1, lam: float = 0.0):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if group_mode not in ("class", "stratum", "class_stratum"):
            raise ValueError("group_mode must be 'class'|'stratum'|'class_stratum'")
        self.alpha = alpha
        self.score = score
        self.group_mode = group_mode
        self.min_group = min_group
        self.randomize = randomize
        self.seed = seed
        self.k_reg = k_reg
        self.lam = lam
        self.q_by_group_: dict = {}
        self.q_pooled_: float | None = None
        self.fallback_groups_: list = []
        self.group_counts_: dict = {}
        self.K_: int | None = None

    def _true_scores(self, P, y, rng):
        if self.score == "lac":
            return lac_scores_true(P, y)
        u = rng.uniform(size=len(y)) if self.randomize else None
        if self.score == "aps":
            return aps_scores_true(P, y, u=u, randomize=self.randomize)
        if self.score == "raps":
            return raps_scores_true(P, y, u=u, randomize=self.randomize, k_reg=self.k_reg, lam=self.lam)
        raise ValueError(f"unknown score {self.score!r}")

    def _all_scores(self, P, rng):
        if self.score == "lac":
            return lac_scores_all(P)
        u = rng.uniform(size=P.shape[0]) if self.randomize else None
        if self.score == "aps":
            return aps_scores_all(P, u=u, randomize=self.randomize)
        if self.score == "raps":
            return raps_scores_all(P, u=u, randomize=self.randomize, k_reg=self.k_reg, lam=self.lam)
        raise ValueError(f"unknown score {self.score!r}")

    def _group_keys(self, y, strata):
        y = np.asarray(y)
        if self.group_mode == "class":
            return [("c", int(v)) for v in y]
        if self.group_mode == "stratum":
            if strata is None:
                raise ValueError("group_mode='stratum' requires strata")
            return [("s", v) for v in np.asarray(strata)]
        # class_stratum
        if strata is None:
            raise ValueError("group_mode='class_stratum' requires strata")
        return [("cs", int(a), b) for a, b in zip(y, np.asarray(strata))]

    def fit(self, P_cal, y_cal, strata_cal=None):
        P_cal = np.asarray(P_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        self.K_ = P_cal.shape[1]
        rng = np.random.default_rng(self.seed)
        s = self._true_scores(P_cal, y_cal, rng)
        self.q_pooled_ = conformal_quantile(s, self.alpha)
        keys = self._group_keys(y_cal, strata_cal)
        # bucket scores by group key
        buckets: dict = {}
        for score_val, key in zip(s, keys):
            buckets.setdefault(key, []).append(score_val)
        self.q_by_group_, self.fallback_groups_, self.group_counts_ = {}, [], {}
        for key, vals in buckets.items():
            self.group_counts_[key] = len(vals)
            if len(vals) >= self.min_group:
                self.q_by_group_[key] = conformal_quantile(np.asarray(vals), self.alpha)
            else:
                self.q_by_group_[key] = self.q_pooled_
                self.fallback_groups_.append((key, len(vals)))
        return self

    def predict_set(self, P_test, strata_test=None):
        """Boolean (n_test, K). For class-conditional grouping the threshold for candidate class k
        is that CLASS's own group threshold (this is the standard Mondrian-by-label construction:
        a class enters the set if its score is below the threshold calibrated on that class)."""
        if self.q_pooled_ is None:
            raise RuntimeError("call fit() first")
        P_test = np.asarray(P_test, dtype=float)
        n, K = P_test.shape
        rng = np.random.default_rng(self.seed + 1)
        S = self._all_scores(P_test, rng)          # (n, K)
        out = np.zeros((n, K), dtype=bool)
        if self.group_mode == "class":
            # threshold per candidate class k = q for group ('c', k)
            qk = np.array([self.q_by_group_.get(("c", k), self.q_pooled_) for k in range(K)])
            out = S <= qk[None, :]
        elif self.group_mode == "stratum":
            strata_test = np.asarray(strata_test)
            for i in range(n):
                q = self.q_by_group_.get(("s", strata_test[i]), self.q_pooled_)
                out[i] = S[i] <= q
        else:  # class_stratum: candidate class k in stratum s -> group ('cs', k, s)
            strata_test = np.asarray(strata_test)
            for i in range(n):
                st = strata_test[i]
                row_q = np.array([self.q_by_group_.get(("cs", k, st), self.q_pooled_) for k in range(K)])
                out[i] = S[i] <= row_q
        return out
