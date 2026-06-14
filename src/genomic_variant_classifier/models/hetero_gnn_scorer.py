"""
hetero_gnn_scorer.py - train the heterogeneous KG GNN and score genes
=====================================================================
Monzia Moodie

Faithful heterogeneous sibling of gnn.py's GNNTrainer/GNNScorer. Builds ONE
shared multi-relation gene graph (STRING interacts_with + KG relations from
kg_edges), trains models/hetero_gnn.HeteroVariantGNN with a focal-node loss
(each labeled variant supervises its gene's node), scores EVERY gene node, and
exposes a gene_symbol -> score map with a 0.5 default for unseen genes -- the
same contract as GNNScorer, so the produced hetero_gnn_score plugs in exactly
where gnn_score does (as a SEPARATE feature, preserving the homogeneous vs
heterogeneous comparison).

The data-assembly core (gene-mean node features + focal/label alignment) is
torch-free and unit-tested without PyG; the training/scoring uses
torch_geometric (imported lazily).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from genomic_variant_classifier.models.hetero_gnn import (
    build_hetero_gene_graph,
    build_hetero_model,
    to_hetero_data,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Torch-free data assembly (unit-tested without PyG)
# ---------------------------------------------------------------------------
def assemble_node_features_and_focal(
    variant_df: pd.DataFrame,
    genes: Sequence[str],
    node_feature_cols: Sequence[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (node_features, focal_idx, y).

    node_features  (n_genes, n_feats) gene-level means (NaN/missing -> 0.0)
    focal_idx      (n_samples,) node index of each labeled variant's gene
    y              (n_samples,) float labels
    Labeled variants whose gene is not a node are dropped.
    """
    gene_index = {g: i for i, g in enumerate(genes)}
    n_nodes = len(genes)
    n_feats = len(node_feature_cols)
    feats = np.zeros((n_nodes, n_feats), dtype=np.float32)
    for fi, f in enumerate(node_feature_cols):
        if f not in variant_df.columns:
            continue
        grp = variant_df.groupby("gene_symbol")[f].mean()
        for g, v in grp.items():
            gi = gene_index.get(str(g))
            if gi is not None and pd.notna(v):
                feats[gi, fi] = float(v)

    labeled = variant_df[variant_df[label_col].notna()]
    focal: list[int] = []
    ys: list[float] = []
    for g, lbl in zip(labeled["gene_symbol"].astype(str), labeled[label_col]):
        gi = gene_index.get(g)
        if gi is not None:
            focal.append(gi)
            ys.append(float(lbl))
    return feats, np.asarray(focal, dtype=np.int64), np.asarray(ys, dtype=np.float32)


@dataclass
class HeteroFocalGraph:
    data: object          # torch_geometric HeteroData
    focal_idx: object     # torch.LongTensor (n_samples,)
    y: object             # torch.FloatTensor (n_samples,)
    node_genes: list      # gene symbol per node index
    relations: list       # relation names present


def build_hetero_focal_graph(
    variant_df: pd.DataFrame,
    string_edges,
    kg_edges_by_relation: Mapping[str, list],
    node_feature_cols: Sequence[str],
    *,
    label_col: str = "acmg_label",
) -> HeteroFocalGraph:
    """Assemble the shared training graph: node type 'gene', relations
    'interacts_with' (STRING) + the KG relations; gene-mean node features;
    focal nodes = labeled variants.
    """
    import torch

    genes = sorted({str(g) for g in variant_df["gene_symbol"].dropna()})
    if not genes:
        raise ValueError("variant_df has no non-null gene_symbol values.")
    feats, focal, ys = assemble_node_features_and_focal(
        variant_df, genes, node_feature_cols, label_col
    )
    edge_lists = {"interacts_with": list(string_edges)}
    for rel, edges in kg_edges_by_relation.items():
        edge_lists[rel] = list(edges)
    graph = build_hetero_gene_graph(genes, feats, edge_lists)  # drops non-cohort genes
    data = to_hetero_data(graph)
    if focal.size == 0:
        logger.warning("build_hetero_focal_graph: no labeled variant maps to a cohort gene.")
    return HeteroFocalGraph(
        data=data,
        focal_idx=torch.tensor(focal, dtype=torch.long),
        y=torch.tensor(ys, dtype=torch.float),
        node_genes=graph.node_genes,
        relations=graph.relations,
    )


# ---------------------------------------------------------------------------
# Trainer + scorer (PyG; lazy torch)
# ---------------------------------------------------------------------------
class HeteroGNNTrainer:
    def __init__(self, in_dim: int, relations: list[str], *, hidden: int = 64,
                 n_layers: int = 2, epochs: int = 100, lr: float = 1e-3,
                 weight_decay: float = 5e-4):
        import torch
        self._torch = torch
        self.model = build_hetero_model(in_dim, relations, hidden=hidden, n_layers=n_layers)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs

    def train(self, fg: HeteroFocalGraph) -> float:
        torch = self._torch
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model.train()
        last = float("nan")
        for _ep in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            out = self.model(fg.data)                 # (n_nodes,) logits
            if fg.focal_idx.numel() == 0:
                logger.warning("HeteroGNNTrainer.train: empty focal set; skipping.")
                break
            loss = loss_fn(out[fg.focal_idx], fg.y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            last = float(loss.item())
        return last

    def score_all_nodes(self, fg: HeteroFocalGraph) -> np.ndarray:
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(fg.data)
            return torch.sigmoid(out).cpu().numpy()


class HeteroGNNScorer:
    """Inference-time gene scorer: one hetero_gnn_score per gene, 0.5 default."""

    DEFAULT_SCORE = 0.5

    def __init__(self, gene_scores: dict[str, float]) -> None:
        self.gene_scores = gene_scores

    @classmethod
    def from_trained(cls, trainer: HeteroGNNTrainer, fg: HeteroFocalGraph) -> "HeteroGNNScorer":
        node_scores = trainer.score_all_nodes(fg)
        genes = fg.node_genes
        if len(genes) != len(node_scores):
            raise ValueError(
                f"node_genes ({len(genes)}) != node_scores ({len(node_scores)})"
            )
        gene_scores = {str(g): float(s) for g, s in zip(genes, node_scores)}
        vals = (np.fromiter(gene_scores.values(), dtype=float)
                if gene_scores else np.zeros(1))
        logger.info(
            "HeteroGNNScorer: %d gene scores (mean=%.3f std=%.4f).",
            len(gene_scores), float(vals.mean()), float(vals.std()),
        )
        return cls(gene_scores)

    def score(self, gene_symbol: str) -> float:
        return self.gene_scores.get(str(gene_symbol), self.DEFAULT_SCORE)

    def score_dataframe(self, df: pd.DataFrame) -> pd.Series:
        return df["gene_symbol"].astype(str).map(self.score)
