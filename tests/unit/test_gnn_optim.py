"""Unit tests for Tier-1 GNN optimizations (bf16 opt-in, device-once) and gnn_optim helpers."""
from __future__ import annotations

from contextlib import nullcontext

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

import pytest
pytest.importorskip("torch_geometric")  # local safety net only: PyG is PINNED
# (torch-geometric==2.7.0, requirements.txt) and CI FAILS the build if it is not
# importable (ci.yml "Assert the coverage-critical dependencies are present"). This
# skip therefore fires only on an under-provisioned LOCAL machine, NEVER in CI --
# the GNN branch is fully covered there. Do not read this as "CI may skip the GNN
# tests": that belief is how the branch went untested for 508 runs (roadmap 6.17).
from genomic_variant_classifier.models.gnn import (
    train_gnn_pipeline, GNNTrainer, VariantGAT, build_pyg_dataset,
)
from genomic_variant_classifier.models.gnn_optim import (
    bf16_autocast, denoise_string_edges, EdgeGate,
)


def _tiny():
    rng = np.random.default_rng(0)
    n = 120
    genes = [f"G{i % 15}" for i in range(n)]
    df = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(5)})
    df["gene_symbol"] = genes
    df["variant_id"] = [f"v{i}" for i in range(n)]
    df["acmg_label"] = rng.integers(0, 2, n)
    g = nx.Graph()
    u = sorted(set(genes))
    for i in range(len(u)):
        g.add_edge(u[i], u[(i + 1) % len(u)], experimental=0.6, database=0.5, coexpression=0.7)
    return df, [f"f{j}" for j in range(5)], g


def test_precision_default_is_fp32_and_trains(tmp_path):
    df, feat, g = _tiny()
    _, t, h = train_gnn_pipeline(df, feat, graph=g, epochs=2, test_split=0.25, checkpoint_path=tmp_path / "best_gat.pt")
    assert t.precision == "fp32"
    assert all(np.isfinite(e["train_loss"]) for e in h)


def test_precision_bf16_path_trains_finite(tmp_path):
    df, feat, g = _tiny()
    _, t, h = train_gnn_pipeline(df, feat, graph=g, epochs=2, test_split=0.25, precision="bf16",
                                 checkpoint_path=tmp_path / "best_gat.pt")
    assert t.precision == "bf16"
    assert all(np.isfinite(e["train_loss"]) for e in h)


def test_device_once_cache_returns_same_object():
    df, feat, g = _tiny()
    ds = build_pyg_dataset(df, g, feat)
    tr = GNNTrainer(VariantGAT(in_channels=len(feat)), epochs=1)
    assert tr._graph_tensors(ds) is tr._graph_tensors(ds)


def test_bf16_autocast_cpu_is_noop():
    assert isinstance(bf16_autocast(torch.device("cpu")), nullcontext)


def test_denoise_modes():
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    ea = torch.tensor([[0.9, 0.8, 0.7], [0.1, 0.1, 0.1], [0.6, 0.6, 0.6]])
    assert denoise_string_edges(ei, ea, mode="none")[0].shape[1] == 3
    assert denoise_string_edges(ei, ea, mode="threshold", tau=0.5)[0].shape[1] == 2
    with pytest.raises(ValueError):
        denoise_string_edges(ei, ea, mode="bogus")


def test_edge_gate_shapes():
    out, gate = EdgeGate(3)(torch.rand(4, 3))
    assert out.shape == (4, 3) and gate.shape == (4, 1)


# --------------------------------------------------------------------------- the checkpoint path (2026-10-12)
# MEASURED 2026-10-11: two validation clones that had run the suite held models/best_gat.pt. GNNTrainer created the checkpoint's
# folder when CONSTRUCTED and train_gnn_pipeline offered no way to choose the path, so every training test wrote into the current
# directory -- the checkout. The folder is now created where the file is written, and the pipeline passes the path through.
def test_constructing_a_trainer_creates_no_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    GNNTrainer(VariantGAT(in_channels=5), epochs=1)                                         # the default, relative path
    GNNTrainer(VariantGAT(in_channels=5), epochs=1, checkpoint_path=tmp_path / "deep" / "best.pt")
    assert list(tmp_path.iterdir()) == []


def test_the_checkpoint_is_written_where_asked_and_nowhere_else(tmp_path, monkeypatch):
    work = tmp_path / "current_directory"
    work.mkdir()
    monkeypatch.chdir(work)
    target = tmp_path / "checkpoints" / "run" / "best.pt"
    df, feat, g = _tiny()
    model, t, h = train_gnn_pipeline(df, feat, graph=g, epochs=3, test_split=0.25, checkpoint_path=target)
    assert t.checkpoint_path == target
    assert max(e["val_auc"] for e in h) > 0, "precondition: fit saves only on an improvement over 0"
    assert target.is_file()
    assert set(torch.load(target, map_location="cpu")) == set(model.state_dict())
    assert list(work.iterdir()) == []


def test_the_default_checkpoint_path_is_unchanged():
    import inspect

    assert inspect.signature(GNNTrainer).parameters["checkpoint_path"].default == "models/best_gat.pt"
    assert inspect.signature(train_gnn_pipeline).parameters["checkpoint_path"].default == "models/best_gat.pt"
