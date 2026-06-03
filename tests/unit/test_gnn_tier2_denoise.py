import warnings
warnings.filterwarnings("ignore")
import pytest
pytest.importorskip("torch_geometric")  # GNN tests need PyG; skip where absent (e.g. CI)
import numpy as np
import networkx as nx
import pandas as pd
import torch
from genomic_variant_classifier.models.gnn import build_pyg_dataset, train_gnn_pipeline


def _toy():
    G = nx.Graph()
    genes = [f"GENE{i}" for i in range(12)]
    G.add_nodes_from(genes)
    rng = np.random.default_rng(1)
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            if rng.random() < 0.5:
                s = float(rng.random())
                G.add_edge(genes[i], genes[j], experimental=s, database=s, coexpression=s)
    rows = []
    for k, g in enumerate(genes):
        for v in range(2):
            rows.append({"gene_symbol": g, "variant_id": f"{g}_{v}",
                         "f0": rng.normal(), "f1": rng.normal(),
                         "acmg_label": int((k + v) % 2)})
    return G, pd.DataFrame(rows), ["f0", "f1"]


def test_denoise_none_is_identity():
    G, df, feats = _toy()
    base = build_pyg_dataset(df, G, feats)
    none = build_pyg_dataset(df, G, feats, edge_denoise="none")
    assert torch.equal(none.edge_index, base.edge_index)
    assert torch.equal(none.edge_attr, base.edge_attr)


def test_denoise_threshold_drops_low_score_edges_symmetrically():
    G, df, feats = _toy()
    base = build_pyg_dataset(df, G, feats)
    thr = build_pyg_dataset(df, G, feats, edge_denoise="threshold", edge_denoise_tau=0.6)
    assert thr.edge_index.shape[1] < base.edge_index.shape[1]
    assert thr.edge_index.shape[1] % 2 == 0
    assert torch.equal(thr.x, base.x)
    assert torch.equal(thr.y, base.y)


def test_denoise_threshold_tau_zero_keeps_all():
    G, df, feats = _toy()
    base = build_pyg_dataset(df, G, feats)
    thr0 = build_pyg_dataset(df, G, feats, edge_denoise="threshold", edge_denoise_tau=0.0)
    assert thr0.edge_index.shape[1] == base.edge_index.shape[1]


def test_denoise_invalid_mode_raises():
    G, df, feats = _toy()
    with pytest.raises(ValueError):
        build_pyg_dataset(df, G, feats, edge_denoise="bogus")


def test_pipeline_trains_through_denoise_path():
    G, df, feats = _toy()
    _m, _tr, hist = train_gnn_pipeline(df, feats, graph=G, epochs=3,
                                       edge_denoise="threshold", edge_denoise_tau=0.5)
    assert len(hist) >= 1
    assert np.isfinite(hist[-1]["val_auc"])
