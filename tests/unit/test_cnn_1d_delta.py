"""Unit tests for the dilated-residual delta sequence CNN (CNN1DClassifier, Tier-1)."""
import io
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import joblib
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from genomic_variant_classifier.models import variant_ensemble as M
from genomic_variant_classifier.models.variant_ensemble import (
    CNN1DClassifier, REF_WIN_COL, ALT_WIN_COL,
)

W = 101


def _ds(n, seed=0):
    rng = np.random.RandomState(seed)
    refs, alts, ys = [], [], []
    for k in range(n):
        s = "".join(rng.choice(list("ACGT")) for _ in range(W))
        y = k % 2
        refs.append(s)
        alts.append(s[:48] + "GGGGG" + s[53:] if y else s)  # signal lives in the delta
        ys.append(y)
    return pd.DataFrame({REF_WIN_COL: refs, ALT_WIN_COL: alts}), np.array(ys)


def test_registry_construct():
    clf = CNN1DClassifier(random_state=42)        # how the ensemble instantiates it
    assert clf.window == 101 and clf.embed == 128


def test_encode_batch_dispatch():
    # Tier-1 (2026-07-05) replaced the old (ref, alt) pair-encoder -- which
    # returned two (N, 4, W) arrays -- with _encode_batch, returning ONE fused
    # (N, C, W) tensor: channels 0:4 = ref one-hot, 4:8 = alt one-hot,
    # 8:12 = alt-ref delta, 12 = positional marker. Same intent as the old test:
    # delta mode -> ref != alt and a non-zero delta; a lone Series -> zero delta.
    clf = CNN1DClassifier()
    df = pd.DataFrame({REF_WIN_COL: ["A" * W], ALT_WIN_COL: ["C" + "A" * (W - 1)]})
    x = clf._encode_batch(df)
    assert x.shape == (1, clf._in_channels(), W)
    ref_oh, alt_oh, delta = x[:, 0:4, :], x[:, 4:8, :], x[:, 8:12, :]
    assert not np.array_equal(ref_oh, alt_oh)        # ref != alt in delta mode
    assert np.abs(delta).sum() > 0                   # non-zero delta present
    # 2026-07-15 (roadmap 6.28): a lone Series NO LONGER encodes as ref == alt with an
    # identically-zero delta. That path hollowed out the very architecture the model is
    # named for, and it is now a hard error. The sanctioned reference-only replacement is
    # CNN1DClassifier(single_sequence_mode=True) -- 5 honest channels, recorded in
    # get_params(), selected by construction rather than by an argument's type.
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        clf._encode_batch(pd.Series(["A" * W]))


def test_learns_delta_signal():
    Xtr, ytr = _ds(300, 0)
    Xte, yte = _ds(120, 1)
    clf = CNN1DClassifier(epochs=10, batch_size=64, learning_rate=2e-3, random_state=0).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    assert p.shape == (120,)
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(yte, p) > 0.65          # must beat the 0.5 floor


def test_state_dict_pickle_roundtrip():
    X, y = _ds(100, 2)
    clf = CNN1DClassifier(epochs=4, batch_size=64, random_state=3).fit(X, y)
    p1 = clf.predict_proba(X)[:, 1]
    buf = io.BytesIO()
    joblib.dump(clf, buf, compress=3)
    buf.seek(0)
    p2 = joblib.load(buf).predict_proba(X)[:, 1]
    assert np.allclose(p1, p2, atol=1e-5)
    assert "_model_state" in clf.__getstate__()


def test_cross_process_unpickle_rebuilds_module():
    X, y = _ds(80, 6)
    clf = CNN1DClassifier(epochs=3, batch_size=64, random_state=6).fit(X, y)
    p1 = clf.predict_proba(X)[:, 1]
    buf = io.BytesIO()
    joblib.dump(clf, buf, compress=3)
    M._CNN1DModule = None                          # simulate a fresh load process
    buf.seek(0)
    clf2 = joblib.load(buf)
    assert M._CNN1DModule is not None              # factory ran during __setstate__
    assert np.allclose(p1, clf2.predict_proba(X)[:, 1], atol=1e-5)


def test_clone_and_cross_val_predict():
    X, y = _ds(90, 4)
    assert clone(CNN1DClassifier(epochs=2)).get_params()["epochs"] == 2
    oof = cross_val_predict(
        CNN1DClassifier(epochs=2, batch_size=64, random_state=5),
        X, y, cv=3, method="predict_proba",
    )
    assert oof.shape == (90, 2)


def test_poly_a_series_fallback_now_raises():
    """The poly-A Series fallback was REMOVED 2026-07-15 (roadmap 6.28).

    It used to accept a bare Series, set ref == alt, and train without complaint on 13
    channels of which 4 were identically zero and 8 were a duplicated pair -- then report
    the result in the algorithm comparison under the name `cnn_1d`, whose entire premise
    is the delta it no longer had. Passing a Series is now a hard error.
    """
    s = pd.Series(["A" * W] * 30)
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        CNN1DClassifier(epochs=2, batch_size=64).fit(s, np.arange(30) % 2)

    # The sanctioned reference-only path still works: opt-in, 5 channels, and visible in
    # get_params() so a run artifact records which architecture actually ran.
    # Verified end to end 2026-07-18 (SINGLE_SEQ_MODE_2026-07-18.txt).
    single = CNN1DClassifier(single_sequence_mode=True, epochs=2, batch_size=64)
    df = pd.DataFrame({REF_WIN_COL: ["A" * W] * 30})
    single.fit(df, np.arange(30) % 2)
    assert single._in_channels() == 5
    assert single.predict_proba(df).shape == (30, 2)
    assert single.get_params()["single_sequence_mode"] is True
