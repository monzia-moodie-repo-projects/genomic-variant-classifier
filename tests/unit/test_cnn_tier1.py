"""
Regression suite for the CNN Tier-1 (`cnn_1d`) re-architecture (2026-07-05).

Drop into tests/unit/. Runs against the REAL installed module (no stubs). Pins
the contract that broke in the 2026-07-04 run (probabilities compressed into a
narrow band, MCC 0.0) so it cannot silently regress: the fused delta/positional
encoding, the decompressed output band, input-dependence, single-`fasta_seq`
back-compat, the state_dict pickle round-trip, the fit-before-predict guard, and
same-seed determinism.

    pytest tests/unit/test_cnn_tier1.py -q
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # CNN branch requires torch; skip if absent

from genomic_variant_classifier.models.variant_ensemble import (  # noqa: E402
    CNN1DClassifier,
    _build_delta_channels,
    encode_sequence,
)

WINDOW, CENTRE = 101, 50
_BASES = np.array(list("ACGT"))


def _make_data(n: int, seed: int = 0):
    """Synthetic SNV variants: alt differs from ref only at the centre base;
    label is a learnable function of the centre substitution (+5% noise)."""
    rng = np.random.default_rng(seed)
    ref, alt, y = [], [], []
    for _ in range(n):
        r = rng.integers(0, 4, WINDOW)
        ac = int(rng.choice([b for b in range(4) if b != r[CENTRE]]))
        a = r.copy()
        a[CENTRE] = ac
        label = 1 if _BASES[ac] in ("T", "G") else 0
        if rng.random() < 0.05:
            label ^= 1
        ref.append("".join(_BASES[r]))
        alt.append("".join(_BASES[a]))
        y.append(label)
    return pd.DataFrame({"fasta_seq_ref": ref, "fasta_seq_alt": alt}), np.array(y)


@pytest.fixture(scope="module")
def trained():
    Xtr, ytr = _make_data(1500, seed=0)
    Xte, yte = _make_data(500, seed=1)
    clf = CNN1DClassifier(epochs=15, batch_size=128, random_state=7).fit(Xtr, ytr)
    return clf, Xte, yte


# -- encoding contract ------------------------------------------------------
def test_default_channels_are_13():
    assert CNN1DClassifier()._in_channels() == 13


def test_encode_shape_and_delta_localised():
    X, _ = _make_data(16, seed=2)
    enc = CNN1DClassifier()._encode_batch(X)
    assert enc.shape == (16, 13, WINDOW)
    delta = enc[:, 8:12, :]  # the alt-ref channels
    nz = np.where(np.abs(delta).sum(axis=(0, 1)) > 0)[0]
    assert nz.tolist() == [CENTRE], f"delta non-zero at {nz.tolist()}, expected only {CENTRE}"


def test_build_delta_channels_flags_toggle_width():
    ref = ["ACGT" * 26]  # len >= window; truncated inside
    alt = ["ACGT" * 26]
    base = _build_delta_channels(ref, alt, WINDOW, False, False, 3.0)
    assert base.shape == (1, 8, WINDOW)
    full = _build_delta_channels(ref, alt, WINDOW, True, True, 3.0)
    assert full.shape == (1, 13, WINDOW)


def test_encode_sequence_still_one_hot():
    oh = encode_sequence("ACGT", window=8)
    assert oh.shape == (8, 4)
    assert oh[0, 0] == 1.0 and oh[1, 1] == 1.0  # A, C


# -- learning + the anti-regression contract --------------------------------
def test_predict_proba_shape_and_normalisation(trained):
    clf, Xte, _ = trained
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(Xte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    p = proba[:, 1]
    assert (p >= 0).all() and (p <= 1).all()


def test_learns_signal(trained):
    from sklearn.metrics import roc_auc_score

    clf, Xte, yte = trained
    au = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    assert au > 0.75, f"AUROC {au:.4f} -- CNN not learning the delta signal"


def test_output_band_decompressed(trained):
    """The 2026-07-04 failure: p in [0.106,0.185], never crossing 0.5, MCC 0.0."""
    from sklearn.metrics import matthews_corrcoef

    clf, Xte, yte = trained
    p = clf.predict_proba(Xte)[:, 1]
    pred = clf.predict(Xte)
    assert p.min() < 0.5 < p.max(), f"band [{p.min():.3f},{p.max():.3f}] does not cross 0.5"
    assert (p.max() - p.min()) > 0.3, "output band too narrow"
    assert set(int(v) for v in np.unique(pred)) == {0, 1}, "degenerate single-class output"
    assert matthews_corrcoef(yte, pred) > 0.0, "MCC collapsed to <= 0"


def test_input_dependent(trained):
    clf, Xte, _ = trained
    assert clf.predict_proba(Xte)[:, 1].std() > 0.05


# -- back-compat + robustness ----------------------------------------------
def test_single_fasta_seq_mode(trained):
    clf, Xte, _ = trained
    Xs = pd.DataFrame({"fasta_seq": Xte["fasta_seq_alt"].values})
    enc = clf._encode_batch(Xs.iloc[:8])
    assert enc.shape == (8, 13, WINDOW)
    assert np.abs(enc[:, 8:12, :]).sum() == 0.0  # ref == alt -> zero delta
    ps = clf.predict_proba(Xs.iloc[:8])
    assert ps.shape == (8, 2) and np.allclose(ps.sum(1), 1.0, atol=1e-5)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        CNN1DClassifier().predict_proba(pd.DataFrame({"fasta_seq_ref": ["A" * 101],
                                                      "fasta_seq_alt": ["C" + "A" * 100]}))


def test_pickle_roundtrip_identical(trained):
    import joblib

    clf, Xte, _ = trained
    p = clf.predict_proba(Xte)[:, 1]
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    clf2 = joblib.load(buf)
    p2 = clf2.predict_proba(Xte)[:, 1]
    assert np.allclose(p, p2, atol=1e-5), f"max|dp|={np.abs(p - p2).max():.2e}"


def test_same_seed_determinism():
    Xtr, ytr = _make_data(800, seed=3)
    Xte, _ = _make_data(200, seed=4)
    a = CNN1DClassifier(epochs=12, batch_size=128, random_state=123).fit(Xtr, ytr)
    b = CNN1DClassifier(epochs=12, batch_size=128, random_state=123).fit(Xtr, ytr)
    pa = a.predict_proba(Xte)[:, 1]
    pb = b.predict_proba(Xte)[:, 1]
    assert np.allclose(pa, pb, atol=1e-4), f"max|dp|={np.abs(pa - pb).max():.2e}"


def test_sklearn_clone_and_params():
    from sklearn.base import clone

    clf = CNN1DClassifier(focal_gamma=1.5, dilations=(1, 2, 4))
    cl = clone(clf)
    assert cl.get_params()["focal_gamma"] == 1.5
    assert cl.get_params()["dilations"] == (1, 2, 4)
