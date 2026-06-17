"""Regression tests for the ESM-2 LLR long-sequence windowing fix.

Before the fix, _score_llr ran a full-length forward pass per protein, so a long
protein (e.g. TTN ~34k aa) exploded the O(L^2) attention (~94 GB) -- which would
OOM the GPU too. The fix windows the LLR pass to <= _MLM_MAX_RESIDUES for long
proteins while leaving short proteins on the original one-pass-per-protein path.

Test 1 (index math) needs no model and runs in CI. Test 2 exercises the real
connector on a synthetic long protein and skips where transformers/torch are absent.
Author: Monzia Moodie.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data import esm2

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _seq(n: int) -> str:
    return (_AA * (n // len(_AA) + 1))[:n]


def test_windowed_logit_row_reads_correct_residue(monkeypatch):
    """The window must stay within the cap AND return the logit row of the
    variant's own residue (so the wt-vs-sequence cross-check stays valid), at the
    N-terminus, mid-protein, and C-terminus -- and reproduce the full-sequence
    index for short proteins (no regression)."""
    calls = []

    def fake_mlm(tok, mdl, seq, mask_token_idx=None):
        # row t (1-based, BOS at 0) encodes the residue at window position t
        calls.append({"winlen": len(seq), "mask_idx": mask_token_idx})
        mat = np.zeros((len(seq) + 2, 33), dtype=float)
        for t in range(1, len(seq) + 1):
            mat[t, 0] = ord(seq[t - 1])
        return mat

    monkeypatch.setattr(esm2, "_mlm_logit_matrix", fake_mlm)

    long_seq = _seq(6000)
    for pos1 in (1, 3000, 6000):  # N-term, mid, C-term
        calls.clear()
        row = esm2._windowed_logit_row(None, None, long_seq, pos1)
        assert calls[-1]["winlen"] <= esm2._MLM_MAX_RESIDUES  # bounded -> no OOM
        assert chr(int(row[0])) == long_seq[pos1 - 1]         # read the right residue

    # masked path masks the variant's (windowed) token index
    calls.clear()
    esm2._windowed_logit_row(None, None, long_seq, 3000, mask=True)
    assert calls[-1]["mask_idx"] is not None
    assert calls[-1]["winlen"] <= esm2._MLM_MAX_RESIDUES

    # short protein: window == full sequence, residue still correct (identical to old path)
    short = _seq(150)
    calls.clear()
    row = esm2._windowed_logit_row(None, None, short, 75)
    assert calls[-1]["winlen"] == len(short)
    assert chr(int(row[0])) == short[74]


def test_llr_long_protein_scores_finite_without_oom(monkeypatch, tmp_path):
    """End-to-end: a 5000-aa protein (> cap) must score a finite, non-zero LLR
    without OOM. Unfixed, this forward pass would blow up; fixed, it windows."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    if getattr(esm2, "_BACKEND", None) != "transformers":
        pytest.skip("transformers backend not active")

    seq = _seq(5000)
    pos1 = 2500
    wt = seq[pos1 - 1]
    mut = "A" if wt != "A" else "G"

    conn = esm2.ESM2Connector(
        model_name="esm2_t6_8M_UR50D",
        cache_path=tmp_path / "esm2_test_cache.sqlite",
        device="cpu",
    )
    # bypass UniProt lookup -- feed the synthetic sequence directly
    monkeypatch.setattr(conn, "_get_sequence", lambda gene: seq)

    df = pd.DataFrame(
        {
            "gene_symbol": ["SYNLONG"],
            "protein_pos": [pos1],
            "wt_aa": [wt],
            "mut_aa": [mut],
            "is_missense": [1],
        }
    )
    # The windowed LLR path needs the real MLM weights. CI runners are offline
    # with no local cache, so the load raises OSError. As of fix(esm2) 961f78e
    # annotate_llr fails CLOSED (esm2_llr=0.0, no raise) so the pipeline never
    # crashes -- therefore probe loadability DIRECTLY here and skip when the
    # weights are unavailable, instead of relying on annotate_llr to raise. (The
    # window index math is covered by test_windowed_logit_row_reads_correct_residue,
    # which mocks the model and needs no download.)
    try:
        esm2._load_transformers_mlm("esm2_t6_8M_UR50D", device="cpu")
    except OSError as exc:
        pytest.skip(f"ESM-2 8M not loadable offline (HF Hub unavailable): {exc}")
    out = conn.annotate_llr(df)
    val = float(out["esm2_llr"].iloc[0])
    assert math.isfinite(val) and val != 0.0          # scored via the windowed path
    assert conn._llr_n_mismatch == 0                  # the windowed residue matched wt
