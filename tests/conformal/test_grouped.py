"""Tests for grouped (gene-set) conformal."""
import numpy as np

from genomic_variant_classifier.conformal import grouped as G


def _grouped(n_groups, per_group, seed):
    rng = np.random.default_rng(seed)
    P_list, y_list, g_list = [], [], []
    for gi in range(n_groups):
        logits = rng.normal(size=(per_group, 2))
        Pg = np.exp(logits); Pg /= Pg.sum(1, keepdims=True)
        yg = np.array([rng.choice(2, p=Pg[i]) for i in range(per_group)])
        P_list.append(Pg); y_list.append(yg); g_list.append(np.full(per_group, gi))
    return np.vstack(P_list), np.concatenate(y_list), np.concatenate(g_list)


def test_group_coverage_at_target_all_aggregators():
    for agg in ["max", "mean", "quantile"]:
        covs = []
        for seed in range(60):
            Pc, yc, gc = _grouped(400, 5, seed)
            Pt, yt, gt = _grouped(400, 5, seed + 5000)
            m = G.GroupedConformalClassifier(alpha=0.1, score="aps", group_agg=agg, seed=seed).fit(Pc, yc, gc)
            covs.append(m.group_coverage(Pt, yt, gt))
        assert abs(np.mean(covs) - 0.90) < 0.02, f"{agg}: {np.mean(covs)}"


def test_predict_group_set_determinism():
    Pc, yc, gc = _grouped(100, 5, 1)
    Pt, yt, gt = _grouped(100, 5, 2)
    a = G.GroupedConformalClassifier(seed=7).fit(Pc, yc, gc).predict_group_set(Pt, gt)
    b = G.GroupedConformalClassifier(seed=7).fit(Pc, yc, gc).predict_group_set(Pt, gt)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_unknown_aggregator_raises():
    Pc, yc, gc = _grouped(10, 5, 1)
    import pytest
    with pytest.raises(ValueError):
        G.GroupedConformalClassifier(group_agg="nonsense").fit(Pc, yc, gc)
