"""
Always-on equivalence gate for the batched ESM-2 path (patch #1).

Proves _score_batched == _score_per_variant at the output-column level using a
REAL ESM tokenizer built offline from the 33-token vocab + a small random
EsmModel -- no network, no downloaded weights, no UniProt. Also checks
padding-invariance and out-of-range handling. Skips only if transformers is
absent or patch #1 has not been applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers", reason="transformers not installed; ESM-2 stub mode locally")
torch = pytest.importorskip("torch")

# ESM-2 alphabet (facebook/esm2_*), fixed order.
ESM_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R",
             "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X",
             "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]


@pytest.fixture(scope="module")
def offline_model(tmp_path_factory):
    from transformers import EsmModel, EsmConfig, EsmTokenizer
    d = tmp_path_factory.mktemp("esm2vocab")
    vf = d / "vocab.txt"
    vf.write_text("\n".join(ESM_VOCAB) + "\n")
    tok = EsmTokenizer(vocab_file=str(vf))
    torch.manual_seed(0)
    cfg = EsmConfig(vocab_size=tok.vocab_size, hidden_size=48, num_hidden_layers=2,
                    num_attention_heads=4, intermediate_size=96, max_position_embeddings=96,
                    pad_token_id=tok.pad_token_id, position_embedding_type="rotary")
    return tok, EsmModel(cfg).eval()


@pytest.fixture
def esm2_mod(offline_model, monkeypatch):
    from genomic_variant_classifier.data import esm2
    if esm2._BACKEND != "transformers":
        pytest.skip("transformers backend not active")
    if not hasattr(esm2.ESM2Connector, "_score_batched"):
        pytest.skip("batched path absent (patch #1 not applied)")
    tok, model = offline_model
    monkeypatch.setattr(esm2, "_load_transformers_model", lambda name, device="cpu": (tok, model))
    return esm2


def _synthetic(seed=11, n_genes=7, n_var=60):
    rng = np.random.default_rng(seed)
    aas = "LAGVSERTIDPKQNFYMHWC"
    genes = {f"G{i}": "".join(rng.choice(list(aas), size=int(rng.integers(30, 140))))
             for i in range(n_genes)}
    recs = []
    for _ in range(n_var):
        g = rng.choice(list(genes)); full = genes[g]
        pos = int(rng.integers(1, len(full) + 1)); wt = full[pos - 1]
        mut = rng.choice([a for a in aas if a != wt])
        recs.append({"gene_symbol": g, "protein_pos": pos, "wt_aa": wt, "mut_aa": mut, "is_missense": 1})
    recs += [
        {"gene_symbol": "G0", "protein_pos": 99999, "wt_aa": "A", "mut_aa": "V", "is_missense": 1},
        {"gene_symbol": "G1", "protein_pos": 0, "wt_aa": "A", "mut_aa": "V", "is_missense": 1},
        {"gene_symbol": "UNKNOWN_GENE", "protein_pos": 5, "wt_aa": "A", "mut_aa": "V", "is_missense": 1},
    ]
    return genes, pd.DataFrame(recs)


def test_batched_equals_per_variant(esm2_mod, monkeypatch, tmp_path):
    genes, cand = _synthetic()
    monkeypatch.setattr(esm2_mod, "_fetch_uniprot_sequence",
                        lambda gene, timeout=10: (("U_" + gene, genes[gene]) if gene in genes else None))
    n = len(cand)

    def column(scorer):
        c = esm2_mod.ESM2Connector(cache_path=tmp_path / f"{scorer}.sqlite")
        scores = getattr(c, scorer)(cand)
        out = np.zeros(n, dtype=float)
        for i, v in scores.items():
            out[i] = v
        return out

    pv = column("_score_per_variant")   # oracle: inline _compute_delta
    ba = column("_score_batched")        # batched: _make_windows + _embed_sequences
    assert np.array_equal(pv > 0, ba > 0), "nonzero masks differ between paths"
    assert np.abs(pv - ba).max() < 1e-5, "batched deltas diverge from per-variant"
    for k in (1, 2, 3):  # the three out-of-range / unknown-gene rows
        assert pv[-k] == 0.0 and ba[-k] == 0.0


def test_annotate_dataframe_uses_batched_and_emits_column(esm2_mod, monkeypatch, tmp_path):
    genes, cand = _synthetic(seed=5, n_var=20)
    monkeypatch.setattr(esm2_mod, "_fetch_uniprot_sequence",
                        lambda gene, timeout=10: (("U_" + gene, genes[gene]) if gene in genes else None))
    out = esm2_mod.ESM2Connector(cache_path=tmp_path / "a.sqlite").annotate_dataframe(cand)
    assert "esm2_delta_norm" in out.columns
    assert len(out) == len(cand)
    assert (out["esm2_delta_norm"].to_numpy() > 0).any()


def test_padding_invariance(offline_model):
    tok, model = offline_model
    seqs = ["LAGVSE", "AGV", "RTIDPKQNFY"]
    enc = tok(seqs, return_tensors="pt", add_special_tokens=True, padding=True)
    with torch.no_grad():
        hb = model(**enc).last_hidden_state
    mask = enc["attention_mask"]
    for i, s in enumerate(seqs):
        li = int(mask[i].sum())
        res_b = hb[i, 1:li - 1, :].numpy()
        single = tok(s, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            res_s = model(**single).last_hidden_state[0, 1:-1, :].numpy()
        assert res_b.shape == res_s.shape == (len(s), 48)
        assert np.abs(res_b - res_s).max() < 1e-4


def test_device_and_batch_size_defaults(esm2_mod):
    c = esm2_mod.ESM2Connector()
    assert c.device == "cpu"
    assert c.batch_size == 64
