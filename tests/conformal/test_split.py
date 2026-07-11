"""Tests for split-conformal classifier: finite-sample quantile + coverage."""
import numpy as np

from genomic_variant_classifier.conformal import split as S


def _cal_test(n_cal, n_test, K, seed, imb=None):
    rng = np.random.default_rng(seed)
    def gen(n):
        if imb is None:
            y = rng.integers(0, K, n)
        else:
            y = (rng.random(n) < imb).astype(int)
        logits = rng.normal(size=(n, K))
        # make probs somewhat informative about y
        logits[np.arange(n), y] += 1.5
        P = np.exp(logits); P /= P.sum(1, keepdims=True)
        return P, y
    return gen(n_cal), gen(n_test)


def test_conformal_quantile_k_formula():
    scores = np.linspace(0, 1, 100)
    # k = ceil((n+1)(1-alpha)) = ceil(101*0.9)=91 -> the 91st order statistic (index 90)
    q = S.conformal_quantile(scores, alpha=0.1)
    assert np.isclose(q, np.sort(scores)[90])


def test_quantile_infinite_when_k_exceeds_n():
    scores = np.array([0.2, 0.4, 0.6])  # n=3; alpha=0.001 -> k=ceil(4*0.999)=4>3 -> +inf
    q = S.conformal_quantile(scores, alpha=0.001)
    assert np.isinf(q)


def test_marginal_coverage_lac():
    covs = []
    for seed in range(60):
        (Pc, yc), (Pt, yt) = _cal_test(2000, 2000, 3, seed)
        m = S.SplitConformalClassifier(alpha=0.1, score="lac", seed=seed).fit(Pc, yc)
        sets = m.predict_set(Pt)
        covs.append(np.mean(sets[np.arange(len(yt)), yt]))
    assert abs(np.mean(covs) - 0.90) < 0.01


def test_marginal_coverage_aps():
    covs = []
    for seed in range(60):
        (Pc, yc), (Pt, yt) = _cal_test(2000, 2000, 4, seed)
        m = S.SplitConformalClassifier(alpha=0.1, score="aps", seed=seed).fit(Pc, yc)
        sets = m.predict_set(Pt)
        covs.append(np.mean(sets[np.arange(len(yt)), yt]))
    assert abs(np.mean(covs) - 0.90) < 0.015


def test_determinism_with_seed():
    (Pc, yc), (Pt, yt) = _cal_test(500, 500, 3, 1)
    a = S.SplitConformalClassifier(alpha=0.1, score="aps", seed=42).fit(Pc, yc).predict_set(Pt)
    b = S.SplitConformalClassifier(alpha=0.1, score="aps", seed=42).fit(Pc, yc).predict_set(Pt)
    assert np.array_equal(a, b)
