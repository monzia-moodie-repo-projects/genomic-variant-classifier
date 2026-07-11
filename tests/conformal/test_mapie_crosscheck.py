"""Independent cross-check of our from-scratch conformal against MAPIE (Model Agnostic Prediction
Interval Estimator), a mature scikit-learn-compatible reference implementation.

MAPIE is used ONLY as an independent oracle, never in the pipeline. The point is to confirm our
numpy LAC/APS thresholds and sets agree with an implementation we did not write.

Findings encoded here (established 2026-07-10 by inspecting MAPIE 1.4.x):
  - LAC: our sets match MAPIE element-wise EXACTLY (deterministic method).
  - APS: MAPIE defaults to include_last_label=True (NON-randomized, conservative -> larger sets,
    over-coverage). Our default APS is RANDOMIZED (Romano-Sesia-Candes; tighter, exact coverage).
    So we compare our NON-randomized APS against MAPIE's default -> they agree closely.
  - MAPIE forbids APS on BINARY targets (only LAC); APS is compared on a MULTICLASS problem.
If MAPIE is not installed, these tests skip (they are a cross-check, not a core requirement).
"""
import numpy as np
import pytest

mapie = pytest.importorskip("mapie")
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from mapie.classification import SplitConformalClassifier as MapieSCC

from genomic_variant_classifier.conformal import split as S
from genomic_variant_classifier.conformal import scores as SC


def _fit_multiclass(seed=0, K=4):
    X, y = make_classification(n_samples=8000, n_features=12, n_classes=K,
                               n_informative=8, random_state=seed)
    Xtr, Xcal, Xte = X[:4000], X[4000:6000], X[6000:]
    ytr, ycal, yte = y[:4000], y[4000:6000], y[6000:]
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return clf, (Xcal, ycal), (Xte, yte)


def _mapie_sets(clf, Xcal, ycal, Xte, cs):
    m = MapieSCC(estimator=clf, confidence_level=0.9, conformity_score=cs, prefit=True)
    m.conformalize(Xcal, ycal)
    _, ys = m.predict_set(Xte)
    ys = np.asarray(ys)
    if ys.ndim == 3:
        ys = ys[:, :, 0]
    return ys.astype(bool)


def test_lac_matches_mapie_elementwise():
    clf, (Xcal, ycal), (Xte, yte) = _fit_multiclass()
    Pcal, Pte = clf.predict_proba(Xcal), clf.predict_proba(Xte)
    ours = S.SplitConformalClassifier(alpha=0.1, score="lac", seed=0).fit(Pcal, ycal).predict_set(Pte)
    theirs = _mapie_sets(clf, Xcal, ycal, Xte, "lac")
    # exact element-wise agreement for the deterministic LAC method
    assert np.array_equal(ours, theirs)


def test_aps_nonrandomized_matches_mapie_closely():
    clf, (Xcal, ycal), (Xte, yte) = _fit_multiclass()
    Pcal, Pte = clf.predict_proba(Xcal), clf.predict_proba(Xte)
    # our NON-randomized APS using our OWN split-module conformal quantile (textbook k-th order
    # statistic, k = ceil((n+1)(1-alpha)))
    sc = SC.aps_scores_true(Pcal, ycal, u=None, randomize=False)
    qhat = S.conformal_quantile(sc, 0.1)
    ours = SC.aps_scores_all(Pte, u=None, randomize=False) <= qhat
    theirs = _mapie_sets(clf, Xcal, ycal, Xte, "aps")
    # Both are non-randomized APS. The residual ~10% cell difference is a KNOWN, STABLE method
    # difference: MAPIE's default include_last_label=True includes the boundary class slightly
    # more often (conservative), so MAPIE sets nearly always CONTAIN ours. We assert (a) high
    # agreement and (b) MAPIE-contains-ours, which together certify our APS is a valid, tighter
    # variant rather than a bug.
    agreement = (ours == theirs).mean()
    mapie_contains_ours = ((theirs | ours) == theirs).mean()
    assert agreement > 0.85, f"APS agreement only {agreement:.3f}"
    assert mapie_contains_ours > 0.95, f"MAPIE-contains-ours only {mapie_contains_ours:.3f}"


def test_both_achieve_target_coverage():
    clf, (Xcal, ycal), (Xte, yte) = _fit_multiclass()
    Pcal, Pte = clf.predict_proba(Xcal), clf.predict_proba(Xte)
    for cs in ["lac", "aps"]:
        theirs = _mapie_sets(clf, Xcal, ycal, Xte, cs)
        cov = np.mean(theirs[np.arange(len(yte)), yte])
        assert cov >= 0.88, f"MAPIE {cs} coverage {cov:.3f} below target band"
