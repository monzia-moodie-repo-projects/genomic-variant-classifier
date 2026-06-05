"""
scripts/test_scalable_svm_cap.py
================================
Guards the 2026-06-05 svm_bagged_rbf cost-bounding patch:
  * the bag count is HARD-CAPPED (does not grow with n) -- asserted at n=1,000,000;
  * effective bags = min(svm_n_bags, svm_max_bags);
  * n<=svm_max_subsample collapses to a single exact SVC;
  * parallel bag predict == serial (numerically identical);
  * joblib round-trip is preserved (pickle-safety; Run 10b).

Fast by construction: the n=1M test uses a tiny per-bag subsample so each bag is
sub-second, exercising the CAP LOGIC (not the full-scale wall-clock).

Run:  pytest -q scripts/test_scalable_svm_cap.py
"""

from __future__ import annotations

import os
import tempfile

import joblib
import numpy as np
import pytest

from genomic_variant_classifier.models.scalable_svm import ScalableSVM


def _xy(n, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(float)
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    return X, y


def test_cap_holds_at_one_million():
    """100 bags requested at n=1M must collapse to the cap; cost stays bounded."""
    X, y = _xy(1_000_000)
    clf = ScalableSVM(
        mode="bagged_rbf", svm_n_bags=100, svm_max_bags=12,
        svm_max_subsample=400, n_jobs=1, random_state=42,
    ).fit(X, y)
    assert clf._n_bags_used == 12
    assert len(clf._bagged) == 12
    Xte, _ = _xy(1000, seed=99)
    proba = clf.predict_proba(Xte)
    assert proba.shape == (1000, 2)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_effective_bags_passthrough_below_cap():
    X, y = _xy(40_000)
    clf = ScalableSVM(
        mode="bagged_rbf", svm_n_bags=5, svm_max_bags=12,
        svm_max_subsample=15_000, n_jobs=1, random_state=42,
    ).fit(X, y)
    assert clf._n_bags_used == 5


def test_single_bag_below_subsample():
    X, y = _xy(800)
    clf = ScalableSVM(
        mode="bagged_rbf", svm_max_subsample=15_000, n_jobs=1, random_state=42,
    ).fit(X, y)
    assert clf._n_bags_used == 1


def test_parallel_equals_serial():
    X, y = _xy(40_000, seed=3)
    Xte, _ = _xy(1000, seed=99)
    kw = dict(mode="bagged_rbf", svm_n_bags=8, svm_max_bags=8,
              svm_max_subsample=4000, random_state=7)
    ps = ScalableSVM(n_jobs=1, **kw).fit(X, y).predict_proba(Xte)
    pp = ScalableSVM(n_jobs=2, **kw).fit(X, y).predict_proba(Xte)
    assert np.allclose(ps, pp, atol=1e-9)


def test_joblib_roundtrip():
    X, y = _xy(40_000, seed=3)
    Xte, _ = _xy(1000, seed=99)
    clf = ScalableSVM(mode="bagged_rbf", svm_n_bags=6, svm_max_bags=6,
                      svm_max_subsample=4000, n_jobs=1, random_state=7).fit(X, y)
    ref = clf.predict_proba(Xte)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "svm.joblib")
        joblib.dump(clf, p)
        loaded = joblib.load(p)
    assert np.allclose(loaded.predict_proba(Xte), ref, atol=1e-9)


def test_headline_nystrom_unaffected():
    X, y = _xy(40_000, seed=3)
    Xte, _ = _xy(1000, seed=99)
    clf = ScalableSVM(mode="nystrom", n_components=256, random_state=42).fit(X, y)
    assert clf.predict_proba(Xte).shape == (1000, 2)
