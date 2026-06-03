"""
Regression tests for the GNN Option B refactor (INCIDENT_2026-06-01_gnn-oom).

Guards the fix that replaced per-variant full-graph replication (which allocated
~64 GB in GATConv.lin_edge under DataLoader(batch=32)) with a single shared graph
trained transductively. These tests FAIL on the pre-refactor module.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
import pytest
import torch

import pytest
pytest.importorskip("torch_geometric")  # GNN tests need PyG; skip where absent (e.g. CI)
from genomic_variant_classifier.models.gnn import (
    build_pyg_dataset,
    train_gnn_pipeline,
    GNNScorer,
    SharedFocalGraph,
)

NCOLS = [f"f{j}" for j in range(6)]


def _graph(n=80, p=0.08, seed=3):
    rng = np.random.default_rng(seed)
    g = nx.gnp_random_graph(n, p, seed=seed)
    G = nx.Graph()
    for u, v in g.edges():
        G.add_edge(f"GENE{u}", f"GENE{v}", weight=rng.uniform(0.7, 1.0),
                   experimental=rng.uniform(0, 1), database=rng.uniform(0, 1),
                   coexpression=rng.uniform(0, 1))
    return G


def _homophilous_variants(graph, n_variants, seed=0):
    rng = np.random.default_rng(seed)
    genes = list(graph.nodes)
    idx = {g: i for i, g in enumerate(genes)}
    prop = rng.uniform(0, 1, size=len(genes))
    A = nx.to_numpy_array(graph, nodelist=genes)
    deg = A.sum(1); deg[deg == 0] = 1
    for _ in range(4):
        prop = 0.5 * prop + 0.5 * (A @ prop) / deg
    prop = (prop - prop.min()) / (prop.max() - prop.min() + 1e-9)
    rows = []
    for i in range(n_variants):
        g = genes[rng.integers(len(genes))]
        p = float(prop[idx[g]])
        feats = p + rng.normal(0, 0.20, size=6)
        r = {"variant_id": f"v{i}", "gene_symbol": g, "acmg_label": int(p > 0.5)}
        for j in range(6):
            r[f"f{j}"] = float(feats[j])
        rows.append(r)
    return pd.DataFrame(rows)


def test_returns_single_shared_graph_not_list():
    graph = _graph()
    df = _homophilous_variants(graph, 500)
    ds = build_pyg_dataset(df, graph, NCOLS)
    assert isinstance(ds, SharedFocalGraph)
    assert ds.x.dim() == 2 and ds.x.shape[0] == graph.number_of_nodes()
    # focal indicator dropped -> in_channels == n_feats (not n_feats + 1)
    assert ds.x.shape[1] == len(NCOLS)
    assert len(ds) == int(df["acmg_label"].notna().sum())


def test_edge_storage_independent_of_variant_count():
    """The anti-OOM invariant: edges stored ONCE, not once per variant."""
    graph = _graph()
    ds_small = build_pyg_dataset(_homophilous_variants(graph, 500), graph, NCOLS)
    ds_large = build_pyg_dataset(_homophilous_variants(graph, 4000), graph, NCOLS)
    assert ds_small.edge_index.shape == ds_large.edge_index.shape
    assert ds_small.edge_attr.shape == ds_large.edge_attr.shape
    # exactly 2*E directed edges among in-graph genes
    assert ds_small.edge_index.shape[1] == 2 * graph.number_of_edges()
    # edge_attr is the 3-channel STRING vector, retained
    assert ds_small.edge_attr.shape[1] == 3


def test_trains_learns_and_is_finite():
    torch.manual_seed(0)
    np.random.seed(0)
    graph = _graph()
    df = _homophilous_variants(graph, 2000)
    model, trainer, hist = train_gnn_pipeline(df, NCOLS, graph=graph, epochs=120, test_split=0.25)
    assert len(hist) > 0
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in hist)
    best = max(h["val_auc"] for h in hist)
    assert best > 0.75, f"GNN failed to learn homophilous signal (best val AUC {best:.3f})"


def test_predict_proba_aligned_finite_and_gene_level():
    torch.manual_seed(0)
    graph = _graph()
    df = _homophilous_variants(graph, 2000)
    _, trainer, _ = train_gnn_pipeline(df, NCOLS, graph=graph, epochs=60, test_split=0.25)
    full = build_pyg_dataset(df, graph, NCOLS)
    proba = trainer.predict_proba(full)
    assert proba.shape[0] == len(full)
    assert np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all()
    # gene-level readout: all variants of one gene share the score
    gene_of = dict(zip(df["variant_id"], df["gene_symbol"]))
    by_gene: dict[str, set] = {}
    for v, pr in zip(full.variant_ids, proba):
        by_gene.setdefault(gene_of[v], set()).add(round(float(pr), 6))
    assert max(len(s) for s in by_gene.values()) == 1


def test_scorer_and_nonconstant_gnn_score():
    torch.manual_seed(0)
    graph = _graph()
    df = _homophilous_variants(graph, 2000)
    _, trainer, _ = train_gnn_pipeline(df, NCOLS, graph=graph, epochs=60, test_split=0.25)
    full = build_pyg_dataset(df, graph, NCOLS)
    scorer = GNNScorer.from_trainer(trainer, full, df)
    assert len(scorer.gene_scores) > 0
    assert scorer.score("NOT_A_REAL_GENE") == 0.5
    s = scorer.score_dataframe(df)
    assert s.notna().all() and s.between(0, 1).all()
    assert s.std() > 0, "gnn_score is constant -> fails the run acceptance criterion"
