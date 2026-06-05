"""
src/genomic_variant_classifier/models/scalable_svm.py
=====================================================
RBF-SVM that stays in the model comparison at full scale (~1.7M rows).

The exact RBF SVC (CalibratedClassifierCV(SVC(kernel="rbf"))) is O(n^2)-O(n^3) and
was auto-skipped above 100k rows in run_phase2_eval.py, so every full run produced
NO SVM data point. This class replaces that dropped estimator with two genuinely
scalable strategies (both RBF, directly comparable):

mode="nystrom"  (headline / publication):
    StandardScaler -> Nystrom RBF feature map (n_components=D) -> LinearSVC,
    Platt-calibrated for probabilities. Uses ALL training rows; cost ~O(n*D).
    D is chosen ~500-2000; probe_n_components() finds where val AUROC plateaus.
    This is the genuinely-unbiased-on-1.7M option (it sees every row), so it is
    the headline SVM result.

mode="rff"  (alternative approximation):
    Same, with RBFSampler (random Fourier features) instead of Nystrom.

mode="bagged_rbf"  (secondary / exact-kernel reference):
    K exact RBF SVCs, each on a stratified subsample of size m (default m=15k),
    probabilities averaged. Exact kernel on a tractable n; bagging cuts the
    variance of a single subsample. m=15k keeps each fit in the seconds range
    (m=100k would sit on the O(n^2) wall); probe_n_bags() finds the K plateau.

    Cost-bounding (added 2026-06-05 after the Run 15 50k probe, where the held-out
    eval spent ~31 min in serial bagged predict over 451k rows):
      * HARD CAP. The number of bags actually fit is
            k = min(max(svm_n_bags, 1), svm_max_bags)
        regardless of n, so train + predict cost is bounded at any cohort size
        (it does NOT grow with n -- each bag is a fixed-size m-row exact SVC).
      * PARALLELISM. Bag fit and bag predict are independent and are run across
        cores via joblib (n_jobs). joblib memmaps large X to workers, so a 1.7M-row
        X is not copied per worker. Averaging is order-independent, so the parallel
        result is numerically identical to the serial one.

Design notes
------------
- Module-level class AND module-level bag workers (_fit_one_bag / _predict_one_bag)
  so both joblib parallelism and pickle/joblib persistence are safe (Run 10b lesson).
- from __future__ import annotations (standing rule).
- No logging.basicConfig at import (library module); a module logger is used for
  info only, matching kan.py.
- gamma "scale" is resolved to 1/n_features AFTER standardization (post-scaling
  per-feature variance ~ 1, so 1/(d*var) ~ 1/d); "auto" -> 1/n_features; a float
  is used as given. bagged_rbf passes gamma straight to SVC (which supports
  "scale"/"auto"/float natively).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

_VALID_MODES = ("nystrom", "rff", "bagged_rbf")


def _resolve_gamma(gamma, n_features: int) -> float:
    """Resolve "scale"/"auto"/float to a float for Nystrom/RBFSampler.

    After StandardScaler the per-feature variance is ~1, so sklearn's
    "scale" = 1/(n_features * X.var()) ~ 1/n_features, which is also "auto".
    """
    if isinstance(gamma, (int, float)):
        return float(gamma)
    if gamma in ("scale", "auto"):
        return 1.0 / max(n_features, 1)
    raise ValueError(f"gamma must be 'scale', 'auto', or a float; got {gamma!r}")


# ----------------------------------------------------------------------
# Module-level bag workers (picklable -> safe for joblib loky AND joblib.dump)
# ----------------------------------------------------------------------
def _make_bag_svc(C, gamma, class_weight, seed):
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight,
            probability=True, random_state=seed),
    )


def _fit_one_bag(X, y, m, C, gamma, class_weight, seed):
    """Stratified subsample of size m, fit one exact-RBF SVC. Returns the pipeline."""
    Xb, yb = resample(X, y, replace=False, n_samples=m, stratify=y, random_state=seed)
    clf = _make_bag_svc(C, gamma, class_weight, seed)
    clf.fit(Xb, yb)
    return clf


def _predict_one_bag(clf, X) -> np.ndarray:
    """Positive-class probability for one bag."""
    return clf.predict_proba(X)[:, 1]


class ScalableSVM(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible RBF-SVM that scales to the full cohort.

    Parameters
    ----------
    mode : {"nystrom", "rff", "bagged_rbf"}
    n_components : int
        Nystrom/RFF feature-map dimension D (headline modes). Capped at n_samples
        for Nystrom (which needs n_components <= n).
    gamma : "scale" | "auto" | float
    C : float
    class_weight : "balanced" | dict | None
    svm_max_subsample : int
        Per-bag stratified subsample size m (bagged_rbf). If n_samples <= m, a
        single exact SVC on all rows is fit (bagging is moot).
    svm_n_bags : int
        Requested number of bags K (bagged_rbf).
    svm_max_bags : int
        HARD CAP on the number of bags actually fit. The effective bag count is
        min(max(svm_n_bags, 1), svm_max_bags), independent of n_samples, so both
        train and predict cost are bounded at any cohort size. Default 15 reflects
        the probe_n_bags() plateau (bagging variance reduction saturates by ~10-15)
        and ~40% less work than the old fixed-25 behaviour. Raise it to restore a
        higher bag count.
    n_jobs : int
        Parallelism for bag fit and bag predict (joblib). -1 uses all cores. The
        bottleneck the cap+parallelism addresses was a serial 25-bag predict over
        451k held-out rows (~31 min in the Run 15 50k probe).
    calibrate : bool
        Platt-calibrate the headline LinearSVC for probabilities (recommended;
        LinearSVC has no predict_proba otherwise).
    calibration_cv : int
    max_iter : int
        LinearSVC iteration cap (headline modes).
    random_state : int
    """

    def __init__(
        self,
        mode: str = "nystrom",
        n_components: int = 1024,
        gamma="scale",
        C: float = 1.0,
        class_weight="balanced",
        svm_max_subsample: int = 15_000,
        svm_n_bags: int = 25,
        svm_max_bags: int = 15,
        n_jobs: int = -1,
        calibrate: bool = True,
        calibration_cv: int = 3,
        max_iter: int = 5000,
        random_state: int = 42,
    ) -> None:
        self.mode = mode
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.class_weight = class_weight
        self.svm_max_subsample = svm_max_subsample
        self.svm_n_bags = svm_n_bags
        self.svm_max_bags = svm_max_bags
        self.n_jobs = n_jobs
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.max_iter = max_iter
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _effective_n_bags(self) -> int:
        """Bag count actually fit: bounded by svm_max_bags, never grows with n."""
        return int(min(max(self.svm_n_bags, 1), max(self.svm_max_bags, 1)))

    def _build_headline(self, n_samples: int, n_features: int):
        gamma_val = _resolve_gamma(self.gamma, n_features)
        d = int(min(self.n_components, max(n_samples - 1, 1)))
        if self.mode == "nystrom":
            fmap = Nystroem(kernel="rbf", gamma=gamma_val, n_components=d,
                            random_state=self.random_state)
        else:  # rff
            fmap = RBFSampler(gamma=gamma_val, n_components=d,
                              random_state=self.random_state)
        base = make_pipeline(
            StandardScaler(),
            fmap,
            LinearSVC(C=self.C, class_weight=self.class_weight,
                      dual="auto", max_iter=self.max_iter,
                      random_state=self.random_state),
        )
        if self.calibrate:
            return CalibratedClassifierCV(base, cv=self.calibration_cv)
        return base

    def _fit_headline(self, X: np.ndarray, y: np.ndarray) -> None:
        self._estimator = self._build_headline(X.shape[0], X.shape[1])
        self._estimator.fit(X, y)
        self._bagged = None

    def _fit_bagged(self, X: np.ndarray, y: np.ndarray) -> None:
        gamma = self.gamma  # SVC accepts "scale"/"auto"/float directly
        m = int(self.svm_max_subsample)
        if X.shape[0] <= m:
            # exact SVC on all rows; bagging is moot below the cap
            clf = _make_bag_svc(self.C, gamma, self.class_weight, self.random_state)
            clf.fit(X, y)
            self._bagged = [clf]
            self._n_bags_used = 1
        else:
            k = self._effective_n_bags()
            seeds = [self.random_state + b for b in range(k)]
            self._bagged = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_one_bag)(X, y, m, self.C, gamma, self.class_weight, s)
                for s in seeds
            )
            self._n_bags_used = k
        self._estimator = None

    def fit(self, X, y) -> "ScalableSVM":
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}; got {self.mode!r}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if self.mode == "bagged_rbf":
            self._fit_bagged(X, y)
            logger.info("ScalableSVM[bagged_rbf]: %d bag(s) of <=%d rows (cap=%d, n_jobs=%d).",
                        self._n_bags_used, self.svm_max_subsample,
                        self.svm_max_bags, self.n_jobs)
        else:
            self._fit_headline(X, y)
            logger.info("ScalableSVM[%s]: D=%d on %d rows.",
                        self.mode, min(self.n_components, X.shape[0] - 1), X.shape[0])
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "classes_")
        X = np.asarray(X, dtype=float)
        if self.mode == "bagged_rbf":
            if len(self._bagged) == 1:
                p = _predict_one_bag(self._bagged[0], X)
            else:
                parts = Parallel(n_jobs=self.n_jobs)(
                    delayed(_predict_one_bag)(clf, X) for clf in self._bagged
                )
                p = np.mean(parts, axis=0)
            return np.column_stack([1.0 - p, p])
        return self._estimator.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Tuning helpers (not auto-called by fit; deterministic fit stays cheap)
    # ------------------------------------------------------------------
    @staticmethod
    def probe_n_components(
        X, y,
        candidates=(256, 512, 1024, 2048),
        mode: str = "nystrom",
        val_frac: float = 0.2,
        tol: float = 0.002,
        random_state: int = 42,
        **svm_kwargs,
    ) -> dict:
        """Pick D where validation AUROC plateaus (improvement < tol).

        Returns {"chosen": D, "curve": [(D, auroc), ...]}.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=val_frac, stratify=y, random_state=random_state)
        curve = []
        chosen = candidates[0]
        prev = -np.inf
        for d in candidates:
            clf = ScalableSVM(mode=mode, n_components=d, random_state=random_state,
                              **svm_kwargs).fit(Xtr, ytr)
            auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
            curve.append((int(d), float(auc)))
            if auc - prev < tol and prev > -np.inf:
                chosen = d  # plateau reached; this D is enough
                break
            chosen = d
            prev = auc
        return {"chosen": int(chosen), "curve": curve}

    @staticmethod
    def probe_n_bags(
        X, y,
        candidates=(1, 5, 10, 15, 25),
        svm_max_subsample: int = 15_000,
        val_frac: float = 0.2,
        tol: float = 0.002,
        random_state: int = 42,
        **svm_kwargs,
    ) -> dict:
        """Pick K where bagged-AUROC plateaus. Returns {"chosen", "curve"}."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=val_frac, stratify=y, random_state=random_state)
        curve = []
        chosen = candidates[0]
        prev = -np.inf
        for k in candidates:
            clf = ScalableSVM(mode="bagged_rbf", svm_n_bags=k, svm_max_bags=k,
                              svm_max_subsample=svm_max_subsample,
                              random_state=random_state, **svm_kwargs).fit(Xtr, ytr)
            auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
            curve.append((int(k), float(auc)))
            if auc - prev < tol and prev > -np.inf:
                chosen = k
                break
            chosen = k
            prev = auc
        return {"chosen": int(chosen), "curve": curve}
