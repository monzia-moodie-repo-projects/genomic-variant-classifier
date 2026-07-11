"""Tests for Mondrian (class/stratum-conditional) conformal."""
import numpy as np

from genomic_variant_classifier.conformal import mondrian as M


def _imbalanced(n, seed, rare=0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rare).astype(int)
    logits = rng.normal(size=(n, 2))
    logits[np.arange(n), y] += 1.2
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    return P, y


def test_mondrian_per_class_coverage_holds_on_rare_class():
    c0, c1 = [], []
    for seed in range(80):
        Pc, yc = _imbalanced(4000, seed)
        Pt, yt = _imbalanced(4000, seed + 999)
        m = M.MondrianConformalClassifier(alpha=0.1, score="lac", group_mode="class", seed=seed).fit(Pc, yc)
        sets = m.predict_set(Pt)
        cov = sets[np.arange(len(yt)), yt]
        c0.append(np.mean(cov[yt == 0])); c1.append(np.mean(cov[yt == 1]))
    # BOTH classes should be at ~0.90 -- the rare class especially
    assert abs(np.mean(c0) - 0.90) < 0.02
    assert abs(np.mean(c1) - 0.90) < 0.03


def test_mondrian_beats_marginal_on_rare_class():
    from genomic_variant_classifier.conformal import split as S
    marg_c1, mond_c1 = [], []
    for seed in range(60):
        Pc, yc = _imbalanced(4000, seed)
        Pt, yt = _imbalanced(4000, seed + 999)
        marg = S.SplitConformalClassifier(alpha=0.1, score="lac", seed=seed).fit(Pc, yc).predict_set(Pt)
        mond = M.MondrianConformalClassifier(alpha=0.1, score="lac", group_mode="class", seed=seed).fit(Pc, yc).predict_set(Pt)
        marg_c1.append(np.mean(marg[yt == 1, :][np.arange((yt == 1).sum()), yt[yt == 1]]))
        mond_c1.append(np.mean(mond[yt == 1, :][np.arange((yt == 1).sum()), yt[yt == 1]]))
    # Mondrian rare-class coverage should be closer to 0.90 than marginal's (which under-covers)
    assert np.mean(mond_c1) > np.mean(marg_c1)
