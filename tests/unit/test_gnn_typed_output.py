"""The GNN exports its focal representation without changing what it predicts.

WHY THIS FILE EXISTS
====================
models/gnn.py computed the focal-node representation and discarded it:

    focal_embeddings = x[gene_idx]
    return self.classifier(focal_embeddings)

That single discard is why Panel R's R3 through R7 are NOT_IMPLEMENTED. The
refactor adds an opt-in `return_embeddings` keyword so the model can hand the
representation out. The one claim this commit must earn is:

    exposing the representation changes NOTHING about the prediction.

Same logits, same gradients, and the exported tensor is the EXACT one the
classifier consumed -- not a copy, not an adjacent layer.

WHAT MAKES THESE TESTS TRUSTWORTHY
----------------------------------
The load-bearing test is test_exported_embedding_is_the_exact_classifier_input.
It registers a forward-pre-hook on the classifier, captures the tensor the head
actually received, and asserts it equals output.focal_embeddings bit-for-bit
(rtol=0, atol=0). This holds WITHIN a single forward pass, so it does not depend
on dropout determinism -- it would catch a future refactor that exported an
adjacent-but-different layer, which a shape check never would.

Exact-logit invariance is tested under eval() + inference_mode, where dropout is
a no-op, so rtol=0/atol=0 is legitimate rather than flaky. The test ASSERTS eval
mode is set rather than assuming it.

Requires torch and torch_geometric, both pinned and present in Continuous
Integration; skips only on an under-provisioned local machine.
"""
from __future__ import annotations



import numpy as np
import pytest

pytest.importorskip("torch")  # local safety net only: torch is PINNED and CI
# fails the build if it is absent (ci.yml REQUIRED gate). Never skips in CI.
pytest.importorskip("torch_geometric")

import networkx as nx
import pandas as pd
import torch

from genomic_variant_classifier.models.gnn import VariantGAT
from genomic_variant_classifier.models.gnn_optim import VariantGATGPS
from genomic_variant_classifier.models.gnn_outputs import GNNOutput


# --------------------------------------------------------------------------- #
# fixtures -- the same toy-graph scaffold the existing GNN tests use
# --------------------------------------------------------------------------- #
def _toy_dataset():
    G = nx.Graph()
    genes = [f"G{i}" for i in range(14)]
    G.add_nodes_from(genes)
    rng = np.random.default_rng(3)
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            if rng.random() < 0.5:
                s = float(rng.random())
                G.add_edge(genes[i], genes[j], experimental=s, database=s, coexpression=s)
    rows = [
        {"gene_symbol": g, "variant_id": f"{g}_{v}", "f0": rng.normal(),
         "f1": rng.normal(), "acmg_label": int((k + v) % 2)}
        for k, g in enumerate(genes) for v in range(2)
    ]
    from genomic_variant_classifier.models.gnn import build_pyg_dataset
    return build_pyg_dataset(pd.DataFrame(rows), G, ["f0", "f1"])


@pytest.fixture
def ds():
    return _toy_dataset()


@pytest.fixture
def gat():
    torch.manual_seed(0)
    return VariantGAT(in_channels=2, hidden_channels=32, heads=4)


@pytest.fixture
def gps():
    torch.manual_seed(0)
    return VariantGATGPS(in_channels=2, hidden_channels=32, heads=4)


# --------------------------------------------------------------------------- #
# 1. the default path is unchanged
# --------------------------------------------------------------------------- #
def test_default_return_is_a_bare_logits_tensor(gat, ds):
    """Every existing caller does out = model(...); softmax(out). The default
    must remain a plain (n_focal, 2) tensor, or all five call sites break."""
    gat.eval()
    with torch.no_grad():
        out = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (len(ds), 2)


def test_default_is_not_a_gnn_output(gat, ds):
    gat.eval()
    with torch.no_grad():
        out = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
    assert not isinstance(out, GNNOutput)


def test_return_embeddings_true_gives_a_gnn_output(gat, ds):
    gat.eval()
    with torch.no_grad():
        out = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                  return_embeddings=True)
    assert isinstance(out, GNNOutput)
    assert out.has_embeddings
    assert out.logits.shape == (len(ds), 2)
    assert out.focal_embeddings.shape[0] == len(ds)


# --------------------------------------------------------------------------- #
# 2. exposing the representation changes nothing
# --------------------------------------------------------------------------- #
def test_returning_embeddings_does_not_change_logits(gat, ds):
    """Exact invariance, legitimate because eval() makes dropout a no-op."""
    gat.eval()
    assert not gat.training, "the invariance claim requires eval mode"
    with torch.inference_mode():
        plain = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
        typed = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                    return_embeddings=True)
    torch.testing.assert_close(plain, typed.logits, rtol=0.0, atol=0.0)


def test_exported_embedding_is_the_exact_classifier_input(gat, ds):
    """THE LOAD-BEARING TEST. The exported tensor must be the one the classifier
    consumed -- bit-for-bit -- so a later refactor cannot quietly export an
    adjacent, scientifically different layer. Holds within one forward pass, so
    it does not depend on dropout determinism."""
    captured = []

    def hook(module, args):
        captured.append(args[0].detach().clone())

    handle = gat.classifier.register_forward_pre_hook(hook)
    try:
        gat.eval()
        with torch.inference_mode():
            out = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                      return_embeddings=True)
    finally:
        handle.remove()

    assert out.has_embeddings
    assert len(captured) == 1, "classifier should be called exactly once"
    torch.testing.assert_close(out.focal_embeddings, captured[0],
                               rtol=0.0, atol=0.0)


def test_embedding_exposure_preserves_gradients(gat, ds):
    """Training-mode gradient flow is unchanged: the encoder still receives
    gradients, and requesting the embedding does not detach the graph."""
    gat.train()
    gat.zero_grad(set_to_none=True)
    out = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
              return_embeddings=True)
    loss = torch.nn.functional.cross_entropy(out.logits, ds.y)
    loss.backward()
    grads = [p.grad for p in gat.parameters() if p.requires_grad]
    assert grads, "no trainable parameters?"
    assert all(g is not None for g in grads), "a parameter received no gradient"
    assert out.focal_embeddings.requires_grad, (
        "the exported embedding must stay attached to autograd; detaching is the "
        "extraction boundary's job, not forward()'s")


# --------------------------------------------------------------------------- #
# 3. structural properties of the exported tensor
# --------------------------------------------------------------------------- #
def test_embedding_rows_match_focal_count(gat, ds):
    gat.eval()
    with torch.no_grad():
        out = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                  return_embeddings=True)
    assert out.focal_embeddings.shape[0] == ds.focal_idx.shape[0]


def test_embedding_width_matches_classifier_input(gat, ds):
    """out_channels feeds the first Linear of the classifier; the exported width
    must equal that Linear's in_features, or the 'exact input' claim is hollow."""
    first_linear = gat.classifier[0]
    gat.eval()
    with torch.no_grad():
        out = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                  return_embeddings=True)
    assert out.focal_embeddings.shape[1] == first_linear.in_features


def test_extraction_is_deterministic_in_eval(gat, ds):
    gat.eval()
    with torch.inference_mode():
        a = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                return_embeddings=True)
        b = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                return_embeddings=True)
    torch.testing.assert_close(a.focal_embeddings, b.focal_embeddings,
                               rtol=0.0, atol=0.0)


def test_exported_embedding_is_finite(gat, ds):
    gat.eval()
    with torch.no_grad():
        out = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                  return_embeddings=True)
    assert torch.isfinite(out.focal_embeddings).all()


# --------------------------------------------------------------------------- #
# 4. the GPS drop-in stays in lock-step
# --------------------------------------------------------------------------- #
def test_gps_default_return_is_a_bare_tensor(gps, ds):
    """VariantGATGPS is a documented drop-in for GNNTrainer/GNNScorer. If its
    default return diverged from VariantGAT's, the swap would silently break."""
    gps.eval()
    with torch.no_grad():
        out = gps(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (len(ds), 2)


def test_gps_return_embeddings_gives_a_gnn_output(gps, ds):
    gps.eval()
    with torch.no_grad():
        out = gps(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                  return_embeddings=True)
    assert isinstance(out, GNNOutput)
    assert out.has_embeddings
    assert out.logits.shape == (len(ds), 2)


def test_gps_exported_embedding_is_the_classifier_input(gps, ds):
    captured = []
    handle = gps.classifier.register_forward_pre_hook(
        lambda m, a: captured.append(a[0].detach().clone()))
    try:
        gps.eval()
        with torch.inference_mode():
            out = gps(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr,
                      return_embeddings=True)
    finally:
        handle.remove()
    assert len(captured) == 1
    torch.testing.assert_close(out.focal_embeddings, captured[0],
                               rtol=0.0, atol=0.0)


def test_both_models_agree_on_the_default_contract(gat, gps, ds):
    """The property that makes GPS a drop-in: identical return TYPE and shape in
    the default path. (Values differ -- different architectures -- but the
    contract a caller relies on is identical.)"""
    gat.eval(); gps.eval()
    with torch.no_grad():
        a = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
        b = gps(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
    assert type(a) is type(b) is torch.Tensor
    assert a.shape == b.shape == (len(ds), 2)


# --------------------------------------------------------------------------- #
# 5. the dataclass contract
# --------------------------------------------------------------------------- #
def test_gnn_output_is_frozen():
    o = GNNOutput(logits=torch.zeros(2, 2))
    with pytest.raises(Exception):
        o.logits = torch.ones(2, 2)


def test_gnn_output_has_embeddings_flag():
    assert not GNNOutput(logits=torch.zeros(1, 2)).has_embeddings
    assert GNNOutput(logits=torch.zeros(1, 2),
                     focal_embeddings=torch.zeros(1, 8)).has_embeddings
