"""
hetero_gnn.py - Heterogeneous multi-relation gene graph + GNN
=============================================================
Monzia Moodie

Extends the homogeneous STRING protein-interaction GNN (gnn.py, single edge type
via GATConv) to a HETEROGENEOUS gene graph: one node type ("gene") with multiple
edge types folded in as additional relations -- STRING interacts_with, Reactome
shares_pathway, OMIM/ClinGen shares_disease, GO shares_function, etc. Each
relation gets its own message-passing weights; contributions are aggregated
across relations. This is torch_geometric.nn.HeteroConv({rel: SAGEConv}).

Design (additive, non-breaking):
  - The torch-free BUILDER (gene->index map + per-relation edge sanitization)
    is separated from the torch model so it can be validated without PyG.
  - HeteroVariantGNN emits a per-node score, mapped to variants by gene_symbol
    exactly like GNNScorer -> a hetero_gnn_score, alongside the existing
    gnn_score. The homogeneous GNN is untouched.

Robustness: edge sanitization drops unknown genes and (optionally) self-loops,
de-duplicates, and symmetrises; empty relations are carried as (2, 0) tensors and
contribute nothing rather than crashing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

logger = logging.getLogger(__name__)

GENE = "gene"


# ---------------------------------------------------------------------------
# Torch-free builder (validated without PyG)
# ---------------------------------------------------------------------------
def sanitize_edges(
    edges: Iterable[tuple],
    gene_index: Mapping[str, int],
    *,
    drop_self_loops: bool = False,
    undirected: bool = True,
) -> np.ndarray:
    """Map (gene_a, gene_b) pairs to node indices and clean them.

    Drops pairs whose endpoints are not in gene_index, optionally drops
    self-loops, de-duplicates, and (default) symmetrises to undirected.
    Returns a (2, E) int64 array; (2, 0) when nothing survives.
    """
    seen = set()
    pairs: list[tuple[int, int]] = []
    for pair in edges:
        if pair is None or len(pair) != 2:
            continue
        a, b = pair
        ia = gene_index.get(a)
        ib = gene_index.get(b)
        if ia is None or ib is None:
            continue
        if drop_self_loops and ia == ib:
            continue
        key = (ia, ib)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((ia, ib))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(pairs, dtype=np.int64).T  # (2, E)
    if undirected:
        arr = np.concatenate([arr, arr[::-1]], axis=1)
        arr = np.unique(arr, axis=1)  # dedup after symmetrising
    return arr


@dataclass
class HeteroGeneGraph:
    node_genes: list           # gene symbol per node index
    gene_index: dict           # gene -> index
    x: np.ndarray              # (n_nodes, n_feats) float32
    edge_index_by_rel: dict    # relation name -> (2, E) int64

    @property
    def relations(self) -> list:
        return list(self.edge_index_by_rel)


def build_hetero_gene_graph(
    genes: Iterable[str],
    node_features: np.ndarray,
    edge_lists_by_relation: Mapping[str, Iterable[tuple]],
    *,
    drop_self_loops: bool = False,
) -> HeteroGeneGraph:
    """Assemble the heterogeneous gene graph.

    genes                  ordered node gene symbols (defines node indices)
    node_features          (n_nodes, n_feats), row-aligned with genes
    edge_lists_by_relation {relation: iterable of (gene_a, gene_b)}
    """
    node_genes = list(genes)
    gene_index = {g: i for i, g in enumerate(node_genes)}
    x = np.asarray(node_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != len(node_genes):
        raise ValueError(
            f"node_features must be (n_genes={len(node_genes)}, n_feats); got {x.shape}"
        )
    edge_index_by_rel: dict[str, np.ndarray] = {}
    for rel, elist in edge_lists_by_relation.items():
        ei = sanitize_edges(elist, gene_index, drop_self_loops=drop_self_loops)
        edge_index_by_rel[rel] = ei
        logger.info(
            "hetero graph relation '%s': %d directed edges after sanitise.",
            rel, ei.shape[1],
        )
    return HeteroGeneGraph(node_genes, gene_index, x, edge_index_by_rel)


# ---------------------------------------------------------------------------
# PyG model (requires torch + torch_geometric; imported lazily)
# ---------------------------------------------------------------------------
def to_hetero_data(graph: HeteroGeneGraph):
    """Convert a HeteroGeneGraph into a torch_geometric HeteroData object."""
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data[GENE].x = torch.tensor(graph.x, dtype=torch.float)
    for rel, ei in graph.edge_index_by_rel.items():
        et = (GENE, rel, GENE)
        if ei.size:
            data[et].edge_index = torch.tensor(ei, dtype=torch.long)
        else:
            data[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return data


def build_hetero_model(in_dim: int, relations: list[str], hidden: int = 64,
                       n_layers: int = 2, dropout: float = 0.2):
    """Factory for HeteroVariantGNN (kept as a function so the heavy torch import
    stays lazy and this module is importable without PyG)."""
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.nn import HeteroConv, SAGEConv

    class HeteroVariantGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self._rels = [(GENE, r, GENE) for r in relations]
            self.convs = nn.ModuleList()
            self.roots = nn.ModuleList()  # explicit per-layer root transform
            for li in range(n_layers):
                d_in = in_dim if li == 0 else hidden
                self.convs.append(HeteroConv(
                    {et: SAGEConv(d_in, hidden) for et in self._rels},
                    aggr="sum",
                ))
                self.roots.append(nn.Linear(d_in, hidden))
            self.dropout = dropout
            self.head = nn.Linear(hidden, 1)

        def forward(self, data):
            x = data[GENE].x
            for conv, root in zip(self.convs, self.roots):
                # Only pass relations that actually have edges -- an empty
                # relation contributes nothing (and avoids (2,0) edge-index
                # edge cases); the root transform always applies, so isolated
                # nodes and all-empty graphs stay finite.
                eid = {
                    et: data[et].edge_index
                    for et in self._rels
                    if et in data.edge_types and data[et].edge_index.size(1) > 0
                }
                msg = root(x)
                if eid:
                    h = conv({GENE: x}, eid)
                    if GENE in h:
                        msg = msg + h[GENE]
                x = F.dropout(F.relu(msg), p=self.dropout, training=self.training)
            return self.head(x).squeeze(-1)  # (n_nodes,) logits

    return HeteroVariantGNN()
