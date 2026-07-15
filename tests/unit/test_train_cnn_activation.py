"""Tests for CNN real-sequence activation (Change 1).

Tests 1-2 are CI-safe (pandas only) and cover the novel logic: the canonical
ref/alt window builder and the meta_train.parquet -> X_train alignment invariant
that makes the signature-free fix safe. Test 3 is a model-backed end-to-end check
that the CNN consumes the 2-column delta frame; it skips without torch.
Author: Monzia Moodie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.seq_window_join import (
    attach_delta_windows,
    REF_WIN_COL,
    ALT_WIN_COL,
)

_REF = "ACGT" * 25 + "A"   # 101 bp
_ALT = "ACGT" * 25 + "C"   # 101 bp, differs at the centre-adjacent tail -> real delta


def test_attach_delta_windows_uses_meta_ref_alt():
    """REWRITTEN 2026-07-15 (roadmap 6.28). The previous version asserted a BUG.

    It read:

        w, n_unmapped = attach_delta_windows(meta)
        assert len(w) == 3 and n_unmapped == 0
        assert w[REF_WIN_COL].iloc[2] == "A" * 101          # NaN -> poly-A fill

    on a frame whose third row is None. Tier 1 of attach_delta_windows ended with a
    hardcoded `return out, 0`, so it reported ZERO unmapped while `.fillna(poly)`
    fabricated a full poly-A window for that null row. This test asserted both halves of
    that -- the false zero AND the fabrication -- and its own comment named the
    fabrication approvingly. A green test was pinning the defect in place.

    A null window is not a window. It is now counted, and marked unusable.
    """
    meta = pd.DataFrame(
        {"gene_symbol": ["G1", "G2", "G3"],
         REF_WIN_COL: [_REF, _REF, None],
         ALT_WIN_COL: [_ALT, _ALT, None]}
    )
    att = attach_delta_windows(meta)

    assert list(att.windows.columns) == [REF_WIN_COL, ALT_WIN_COL]
    assert att.n_rows == 3
    assert att.n_usable == 2
    assert att.n_unmapped == 1, "the null row was fabricated away and reported as mapped"
    assert not att.usable[2]

    # The two REAL rows are passed through byte-for-byte and carry a real delta.
    assert att.windows[REF_WIN_COL].iloc[0] == _REF
    assert att.windows[ALT_WIN_COL].iloc[0] == _ALT
    assert att.windows[REF_WIN_COL].iloc[0] != att.windows[ALT_WIN_COL].iloc[0]
    assert att.usable[0] and att.usable[1]


def test_attach_delta_windows_yields_nothing_usable_when_windows_absent():
    """Renamed from `test_attach_delta_windows_poly_a_when_absent` 2026-07-15.

    The old name and body asserted the FILLER's identity ("A"*101). The filler is now
    deliberately not part of the contract -- exactly so that no consumer can grow a
    fifth content-based poly detector. What matters is that nothing is usable.
    """
    meta = pd.DataFrame({"gene_symbol": ["G1", "G2"]})   # no ref/alt columns at all
    att = attach_delta_windows(meta)

    assert att.n_unmapped == 2
    assert att.n_usable == 0
    assert not att.usable.any()
    assert att.provenance == "none"


def test_meta_train_parquet_aligns_to_x_train(tmp_path):
    """The fix reads meta_train.parquet for train-side sequences; this asserts
    the persisted meta_train is the same length and row order as X_train, since
    both derive from df.iloc[train_idx].reset_index(drop=True)."""
    df = pd.DataFrame(
        {"key": list(range(10)),
         "feat": np.arange(10, dtype=float),
         REF_WIN_COL: [_REF] * 10,
         ALT_WIN_COL: [_ALT] * 10,
         "label": [0, 1] * 5}
    )
    train_idx = np.array([7, 2, 9, 0, 4])   # arbitrary gene-disjoint selection
    X_train = df[["feat"]].iloc[train_idx].reset_index(drop=True)
    y_train = df["label"].iloc[train_idx].reset_index(drop=True)
    meta_train = df.iloc[train_idx].reset_index(drop=True)

    path = tmp_path / "meta_train.parquet"
    meta_train.to_parquet(path, index=False)
    mt = pd.read_parquet(path)

    assert len(mt) == len(X_train) == len(y_train) == 5            # the runtime guard
    assert list(mt["key"]) == list(df["key"].iloc[train_idx])     # row order preserved
    assert list(X_train["feat"]) == list(df["feat"].iloc[train_idx])  # same ordering
    w, n_unmapped = attach_delta_windows(mt)
    assert len(w) == len(X_train) and n_unmapped == 0


def test_cnn1d_fits_on_delta_dataframe():
    """End-to-end: the CNN consumes a 2-column [ref, alt] DataFrame (delta mode),
    trains, and returns finite probabilities. Skips without torch."""
    pytest.importorskip("torch")
    from genomic_variant_classifier.models.variant_ensemble import CNN1DClassifier

    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    refs = ["".join(rng.choice(bases, 101)) for _ in range(24)]
    alts = [r[:50] + ("C" if r[50] != "C" else "G") + r[51:] for r in refs]  # 1-bp delta
    X = pd.DataFrame({REF_WIN_COL: refs, ALT_WIN_COL: alts})
    y = np.array([0, 1] * 12)

    clf = CNN1DClassifier(epochs=2, batch_size=8, filters=8, embed=16, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (24, 2)
    assert np.isfinite(proba).all()
