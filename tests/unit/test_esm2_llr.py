"""Unit tests for the ESM-2 LLR scorer (Phase 1).

Torch-free: the model loaders are monkeypatched with a fake tokenizer and
canned logit matrices, so these run in CI without downloading the 2.5 GB model.
The real-model sign/index correctness is proven separately by
scripts/probe_esm2_llr.py (CPU gate). Author: Monzia Moodie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data import esm2 as E
from genomic_variant_classifier.data.gene_symbols import gene_symbol_candidates

# AA -> token id; <unk>=3, <mask>=32 (mirrors the ESM tokenizer's single-char AA tokens)
_VOCAB = {"<unk>": 3, "A": 4, "R": 5, "H": 6, "Q": 7, "P": 8, "K": 9,
          "L": 10, "M": 11, "V": 12, "G": 13, "S": 14, "T": 15, "D": 16}
_SEQS = {"TP53": "MRPHA", "BRCA1": "AKLMV"}   # TP53: M1 R2 P3 H4 A5


class _FakeTok:
    unk_token_id = 3
    mask_token_id = 32
    def convert_tokens_to_ids(self, t):
        return _VOCAB.get(t, self.unk_token_id)


class _FakeMdl:
    device = "cpu"


def _fake_matrix(seq, mut_high_at=None):
    """WT residue id favored (+8) at each token index; everything else -5.
    mut_high_at=(pos1, mut_id) makes one mutant favored (+12) => positive LLR."""
    mat = np.full((len(seq) + 2, 33), -5.0)
    for i, aa in enumerate(seq, start=1):       # residue i -> token index i
        mat[i, _VOCAB.get(aa, 3)] = 8.0
    if mut_high_at:
        mat[mut_high_at[0], mut_high_at[1]] = 12.0
    return mat


@pytest.fixture
def conn(monkeypatch, tmp_path):
    """A REAL, fully-constructed ESM2Connector with the model backend faked.

    Construction goes through the real __init__ -- deliberately NOT
    ESM2Connector.__new__(). The __new__ bypass this fixture used until 2026-07-11
    produced a half-built object carrying only the four attributes the fixture
    happened to set, so the moment annotate_llr reached any OTHER __init__
    attribute the tests died with an AttributeError that said nothing about
    production. That is exactly what happened when the parquet score-cache landed:
    annotate_llr -> _score_cache_load -> _score_cache_path -> self.cache_path
    -> AttributeError, six tests red since the day it shipped
    (TRIAGE_2026-07-08_test-suite-red, cluster B).

    Constructing properly is safe and stays torch-free:
      * device="cpu" is passed explicitly, so _resolve_device short-circuits and
        never touches torch.cuda (esm2.py:224-228);
      * __init__ does no I/O -- the sqlite handle (_conn) is lazy;
      * the model loader is monkeypatched to the fake tokenizer/model above.

    cache_path is pinned under tmp_path, which also makes the tests HERMETIC: the
    score cache is a sibling of cache_path (esm2.py:_score_cache_path), so both the
    sqlite cache AND esm2_scores.parquet land in tmp_path. Left to the production
    default they would read from -- and _score_cache_append would WRITE fake stub
    scores into -- the repo's real data/raw/cache. Never point this fixture at the
    default cache.

    allow_network=False is explicit belt-and-braces: no test may reach UniProt.
    """
    monkeypatch.setattr(E, "_BACKEND", "transformers", raising=False)
    monkeypatch.setattr(E, "_load_transformers_mlm",
                        lambda model_name, device="cpu": (_FakeTok(), _FakeMdl()))
    c = E.ESM2Connector(
        model_name="esm2_t33_650M_UR50D",
        cache_path=tmp_path / "esm2_cache.sqlite",
        device="cpu",
        allow_network=False,
    )
    # candidate-aware sequence resolver (mirrors the Phase-0 hardened _get_sequence)
    def _get_seq(gene):
        for cand in gene_symbol_candidates(gene):
            if cand in _SEQS:
                return _SEQS[cand]
        c._missing_genes.add(str(gene).strip().upper())
        return None
    c._get_sequence = _get_seq          # instance-level override; no UniProt, no network
    return c


def test_fixture_is_fully_constructed_and_hermetic(conn, tmp_path):
    """Regression guard for TRIAGE_2026-07-08 cluster B (2026-07-11).

    Two invariants, both of which the old __new__ fixture violated:
      1. The connector is FULLY constructed -- every __init__ attribute exists, so a
         new code path in annotate_llr can never again die on a missing attribute.
      2. The fixture is HERMETIC -- the ESM-2 score cache resolves inside tmp_path,
         never the repository's real data/raw/cache. A leak there would let unit
         tests read stale real scores (flaky, order-dependent) and, via
         _score_cache_append, WRITE fake stub scores into the production cache.
    """
    for attr in ("model_name", "cache_path", "request_timeout", "device", "batch_size",
                 "uniprot_index_path", "allow_network", "_conn", "_uniprot_index",
                 "_warned_missing", "_missing_genes"):
        assert hasattr(conn, attr), f"conn fixture must be fully constructed; missing {attr!r}"

    assert conn.cache_path == tmp_path / "esm2_cache.sqlite"
    assert conn.allow_network is False
    assert conn.device == "cpu"

    score_cache = conn._score_cache_path()
    assert tmp_path in score_cache.parents, (
        f"ESM-2 score cache must resolve inside tmp_path (hermetic); got {score_cache}")
    assert "data" not in score_cache.parts, (
        f"ESM-2 score cache must NEVER resolve into the repo data dir; got {score_cache}")


def _passcount(monkeypatch, mut_high_at=None):
    counts = {"wt": 0, "masked": 0}
    def _stub(tok, mdl, seq, mask_token_idx=None):
        counts["masked" if mask_token_idx is not None else "wt"] += 1
        return _fake_matrix(seq, mut_high_at=mut_high_at)
    monkeypatch.setattr(E, "_mlm_logit_matrix", _stub)
    return counts


def test_llr_pure_helper():
    row = np.zeros(33); row[5] = 3.0; row[6] = -2.0
    assert E._llr_from_logit_row(row, 5, 6) == -5.0      # logit[mut]-logit[wt]
    assert E._llr_from_logit_row(row, 6, 5) == 5.0


def test_wt_marginal_one_pass_per_protein(conn, monkeypatch):
    counts = _passcount(monkeypatch)
    df = pd.DataFrame({"gene_symbol": ["TP53"] * 3, "protein_pos": [2, 3, 4],
                       "wt_aa": ["R", "P", "H"], "mut_aa": ["H", "K", "Q"], "is_missense": [1, 1, 1]})
    out = conn.annotate_llr(df, method="wt")
    assert counts["wt"] == 1                              # ONE pass for all 3 variants
    assert np.allclose(out["esm2_llr"], -13.0)            # mut(-5) - wt(+8)


def test_tolerated_mutant_positive(conn, monkeypatch):
    _passcount(monkeypatch, mut_high_at=(2, _VOCAB["H"]))  # favor H at pos2
    df = pd.DataFrame({"gene_symbol": ["TP53"], "protein_pos": [2],
                       "wt_aa": ["R"], "mut_aa": ["H"], "is_missense": [1]})
    out = conn.annotate_llr(df, method="wt")
    assert out.loc[0, "esm2_llr"] == 4.0                 # +12 - (+8)


def test_wt_mismatch_skipped_and_counted(conn, monkeypatch):
    _passcount(monkeypatch)
    df = pd.DataFrame({"gene_symbol": ["TP53"], "protein_pos": [2],
                       "wt_aa": ["K"], "mut_aa": ["H"], "is_missense": [1]})  # seq pos2 is R, not K
    out = conn.annotate_llr(df, method="wt")
    assert out.loc[0, "esm2_llr"] == 0.0
    assert conn._llr_n_mismatch == 1


def test_out_of_range_position_skipped(conn, monkeypatch):
    _passcount(monkeypatch)
    df = pd.DataFrame({"gene_symbol": ["TP53"], "protein_pos": [999],
                       "wt_aa": ["R"], "mut_aa": ["H"], "is_missense": [1]})
    out = conn.annotate_llr(df, method="wt")
    assert out.loc[0, "esm2_llr"] == 0.0 and conn._llr_n_mismatch == 1


def test_masked_marginal_one_pass_per_unique_position(conn, monkeypatch):
    counts = _passcount(monkeypatch)
    df = pd.DataFrame({"gene_symbol": ["TP53"] * 3, "protein_pos": [2, 2, 4],
                       "wt_aa": ["R", "R", "H"], "mut_aa": ["H", "Q", "Q"], "is_missense": [1, 1, 1]})
    conn.annotate_llr(df, method="masked")
    assert counts["masked"] == 2 and counts["wt"] == 0   # positions {2,2,4} -> 2 unique


def test_semicolon_joined_gene_resolves(conn, monkeypatch):
    _passcount(monkeypatch)
    df = pd.DataFrame({"gene_symbol": ["FOO;TP53"], "protein_pos": [2],
                       "wt_aa": ["R"], "mut_aa": ["H"], "is_missense": [1]})
    out = conn.annotate_llr(df, method="wt")
    assert out.loc[0, "esm2_llr"] == -13.0               # FOO misses, TP53 resolves


def test_stub_backend_safe_default(conn, monkeypatch):
    _passcount(monkeypatch)
    monkeypatch.setattr(E, "_BACKEND", "fair-esm", raising=False)
    df = pd.DataFrame({"gene_symbol": ["TP53"], "protein_pos": [2],
                       "wt_aa": ["R"], "mut_aa": ["H"], "is_missense": [1]})
    out = conn.annotate_llr(df, method="wt")
    assert (out["esm2_llr"] == 0.0).all()


def test_missing_columns_and_non_missense_safe(conn, monkeypatch):
    _passcount(monkeypatch)
    assert (conn.annotate_llr(pd.DataFrame({"gene_symbol": ["TP53"]}))["esm2_llr"] == 0.0).all()
    df = pd.DataFrame({"gene_symbol": ["TP53"], "protein_pos": [2],
                       "wt_aa": ["R"], "mut_aa": ["H"], "is_missense": [0]})
    assert (conn.annotate_llr(df)["esm2_llr"] == 0.0).all()
