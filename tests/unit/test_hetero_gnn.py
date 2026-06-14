"""test_hetero_gnn.py  --  Monzia Moodie

Heterogeneous gene-graph GNN. The torch-free builder is tested directly; the real
PyG HeteroConv forward runs where torch_geometric is installed (.venv312), on a
deliberately messy graph (self-loops, duplicate/unknown edges, isolated nodes, an
empty relation) to prove robustness for seamless integration.
"""
import numpy as np
import pytest

from genomic_variant_classifier.models.hetero_gnn import (
    GENE,
    build_hetero_gene_graph,
    sanitize_edges,
)


def test_sanitize_drops_unknown_dedups_undirected():
    gi = {"A": 0, "B": 1, "C": 2, "D": 3}
    edges = [("A", "B"), ("B", "A"), ("A", "B"), ("C", "C"),
             ("A", "Z"), None, ("A",), ("B", "C")]
    got = set(map(tuple, sanitize_edges(edges, gi).T.tolist()))
    assert (0, 1) in got and (1, 0) in got        # symmetrised
    assert (2, 2) in got                          # self-loop kept by default
    assert (1, 2) in got and (2, 1) in got
    assert not any(3 in p for p in got)           # D isolated
    assert all(0 <= a < 4 and 0 <= b < 4 for a, b in got)  # unknown Z dropped


def test_drop_self_loops_and_empty():
    gi = {"A": 0, "B": 1}
    assert not any(a == b for a, b in
                   map(tuple, sanitize_edges([("A", "A"), ("A", "B")], gi,
                                             drop_self_loops=True).T.tolist()))
    assert sanitize_edges([], gi).shape == (2, 0)
    assert sanitize_edges([("Z", "Q")], gi).shape == (2, 0)  # all unknown


def test_builder_relations_and_feature_guard():
    genes = ["A", "B", "C", "D"]
    feats = np.random.default_rng(0).normal(size=(4, 5))
    g = build_hetero_gene_graph(genes, feats, {
        "interacts_with": [("A", "B"), ("B", "C")],
        "shares_pathway": [("A", "C"), ("D", "B")],
        "shares_disease": [],
    })
    assert g.relations == ["interacts_with", "shares_pathway", "shares_disease"]
    assert g.edge_index_by_rel["shares_disease"].shape == (2, 0)
    assert g.x.dtype == np.float32 and g.x.shape == (4, 5)
    with pytest.raises(ValueError):
        build_hetero_gene_graph(genes, np.zeros((3, 5)), {"r": []})


def test_pyg_hetero_forward_runs_and_is_finite():
    pytest.importorskip("torch_geometric")
    import torch
    from genomic_variant_classifier.models.hetero_gnn import (
        build_hetero_model, to_hetero_data,
    )

    genes = [f"G{i}" for i in range(8)]
    feats = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    g = build_hetero_gene_graph(genes, feats, {
        "interacts_with": [("G0", "G1"), ("G1", "G2"), ("G3", "G4"), ("G5", "G5")],
        "shares_pathway": [("G2", "G3"), ("G6", "G7")],
        "shares_disease": [],                       # empty relation
    })
    data = to_hetero_data(g)
    model = build_hetero_model(in_dim=4, relations=g.relations, hidden=8, n_layers=2)
    model.eval()
    out = model(data)
    assert out.shape[0] == 8                        # one score per gene node
    assert torch.isfinite(out).all()                # robust to self-loop/empty/isolated


def test_pyg_all_empty_graph_is_finite():
    pytest.importorskip("torch_geometric")
    import torch
    from genomic_variant_classifier.models.hetero_gnn import (
        build_hetero_model, to_hetero_data,
    )
    genes = ["A", "B", "C"]
    feats = np.zeros((3, 4), dtype=np.float32)
    g = build_hetero_gene_graph(genes, feats,
                                {"interacts_with": [], "shares_pathway": []})
    model = build_hetero_model(in_dim=4, relations=g.relations, hidden=6)
    model.eval()
    out = model(to_hetero_data(g))                  # no edges at all
    assert out.shape[0] == 3 and torch.isfinite(out).all()
