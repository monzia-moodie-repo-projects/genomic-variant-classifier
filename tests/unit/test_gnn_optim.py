"""Unit tests for Tier-1 GNN optimizations (bf16 opt-in, device-once) and gnn_optim helpers."""
from __future__ import annotations

from contextlib import nullcontext

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

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


def test_precision_default_is_fp32_and_trains():
    df, feat, g = _tiny()
    _, t, h = train_gnn_pipeline(df, feat, graph=g, epochs=2, test_split=0.25)
    assert t.precision == "fp32"
    assert all(np.isfinite(e["train_loss"]) for e in h)


def test_precision_bf16_path_trains_finite():
    df, feat, g = _tiny()
    _, t, h = train_gnn_pipeline(df, feat, graph=g, epochs=2, test_split=0.25, precision="bf16")
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
