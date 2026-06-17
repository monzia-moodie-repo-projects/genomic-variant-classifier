"""Regression: ESM-2 LLR must fail CLOSED when the model cannot be loaded.

When protein_pos/wt_aa/mut_aa are populated (e.g. by the step-10c HGVSp fallback),
annotate_llr finds candidates and loads facebook/<model> from HuggingFace. Offline
with no cached weights (CI), that load raises OSError. The pipeline must stub
esm2_llr to the neutral 0.0 default, NOT crash -- the step-10c activation surfaced
this latent bug (CI run #444).
"""
from __future__ import annotations

import pandas as pd

from genomic_variant_classifier.data import esm2 as esm2mod


def test_annotate_llr_fails_closed_when_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(esm2mod, "_BACKEND", "transformers", raising=False)

    def _boom(*a, **k):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load this file")

    monkeypatch.setattr(esm2mod, "_load_transformers_mlm", _boom, raising=False)

    conn = esm2mod.ESM2Connector(cache_path=tmp_path / "esm2_cache.sqlite")
    df = pd.DataFrame({
        "gene_symbol": ["BRCA1", "TP53"],
        "protein_pos": pd.array([1699, 175], dtype="Int64"),
        "wt_aa": ["R", "R"],
        "mut_aa": ["Q", "H"],
        "is_missense": [1, 1],
    })

    out = conn.annotate_llr(df)                       # must NOT raise
    assert "esm2_llr" in out.columns
    assert (out["esm2_llr"] == 0.0).all()            # stubbed to neutral default


def test_annotate_llr_no_candidates_does_not_load_model(monkeypatch, tmp_path):
    """With no missense candidates the model must never be loaded."""
    monkeypatch.setattr(esm2mod, "_BACKEND", "transformers", raising=False)

    def _must_not_load(*a, **k):
        raise AssertionError("model loaded despite zero candidates")

    monkeypatch.setattr(esm2mod, "_load_transformers_mlm", _must_not_load, raising=False)

    conn = esm2mod.ESM2Connector(cache_path=tmp_path / "esm2_cache.sqlite")
    df = pd.DataFrame({
        "gene_symbol": ["BRCA1"],
        "protein_pos": pd.array([pd.NA], dtype="Int64"),
        "wt_aa": [None], "mut_aa": [None], "is_missense": [0],
    })
    out = conn.annotate_llr(df)                       # no candidates -> no load, no raise
    assert (out["esm2_llr"] == 0.0).all()
