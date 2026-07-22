import warnings
warnings.filterwarnings("ignore")
import pytest
pytest.importorskip("torch_geometric")  # local safety net only: PyG is PINNED
# (torch-geometric==2.7.0, requirements.txt) and CI FAILS the build if it is not
# importable (ci.yml "Assert the coverage-critical dependencies are present"). This
# skip therefore fires only on an under-provisioned LOCAL machine, NEVER in CI --
# the GNN branch is fully covered there. Do not read this as "CI may skip the GNN
# tests": that belief is how the branch went untested for 508 runs (roadmap 6.17).
import numpy as np
import networkx as nx
import pandas as pd
import torch
from genomic_variant_classifier.models.gnn import VariantGAT, train_gnn_pipeline
from genomic_variant_classifier.models.gnn_optim import VariantGATGPS


def _toy():
    G = nx.Graph(); genes = [f"G{i}" for i in range(14)]; G.add_nodes_from(genes)
    rng = np.random.default_rng(3)
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            if rng.random() < 0.5:
                s = float(rng.random()); G.add_edge(genes[i], genes[j], experimental=s, database=s, coexpression=s)
    rows = [{"gene_symbol": g, "variant_id": f"{g}_{v}", "f0": rng.normal(), "f1": rng.normal(),
             "acmg_label": int((k + v) % 2)} for k, g in enumerate(genes) for v in range(2)]
    return G, pd.DataFrame(rows), ["f0", "f1"]


def _ds(G, df, feats):
    from genomic_variant_classifier.models.gnn import build_pyg_dataset
    return build_pyg_dataset(df, G, feats)


def test_gps_forward_shape_parity_with_gat():
    G, df, feats = _toy(); ds = _ds(G, df, feats)
    gat = VariantGAT(in_channels=2, hidden_channels=32, heads=4).eval()
    gps = VariantGATGPS(in_channels=2, hidden_channels=32, heads=4).eval()
    with torch.no_grad():
        a = gat(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
        b = gps(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
    assert a.shape == b.shape == (len(ds), 2)
    assert torch.isfinite(b).all()


def test_gps_edge_gate_variant_finite():
    G, df, feats = _toy(); ds = _ds(G, df, feats)
    gps = VariantGATGPS(in_channels=2, hidden_channels=32, heads=4, gate_edges=True).eval()
    with torch.no_grad():
        out = gps(ds.x, ds.edge_index, ds.focal_idx, edge_attr=ds.edge_attr)
    assert out.shape == (len(ds), 2) and torch.isfinite(out).all()


def test_gps_zero_edge_robust():
    G, df, feats = _toy(); ds = _ds(G, df, feats)
    gps = VariantGATGPS(in_channels=2, hidden_channels=32, heads=4).eval()
    ei0 = torch.zeros((2, 0), dtype=torch.long); ea0 = torch.zeros((0, 3))
    with torch.no_grad():
        out = gps(ds.x, ei0, ds.focal_idx, edge_attr=ea0)
    assert torch.isfinite(out).all()


def test_pipeline_layer_type_gps_builds_and_trains():
    G, df, feats = _toy()
    m, _tr, hist = train_gnn_pipeline(df, feats, graph=G, epochs=2, layer_type="gps")
    assert isinstance(m, VariantGATGPS)
    assert np.isfinite(hist[-1]["val_auc"])


def test_pipeline_default_is_gat():
    G, df, feats = _toy()
    m, _tr, _h = train_gnn_pipeline(df, feats, graph=G, epochs=2)
    assert isinstance(m, VariantGAT)


def test_pipeline_invalid_layer_type_raises():
    G, df, feats = _toy()
    with pytest.raises(ValueError):
        train_gnn_pipeline(df, feats, graph=G, epochs=1, layer_type="bogus")
