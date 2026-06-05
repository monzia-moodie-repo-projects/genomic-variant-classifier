"""
Graph Neural Network — Variant-Gene Interaction Model
=======================================================
Builds a biological protein-interaction graph from STRING DB and trains a
Graph Attention Network (GAT) to predict variant pathogenicity by propagating
gene-level features through the protein interaction network.

OPTION B refactor (2026-06-01, INCIDENT_2026-06-01_gnn-oom):
  The previous build_pyg_dataset replicated the ENTIRE STRING graph into one
  PyG Data object per labeled variant (4,873 copies). With DataLoader(batch=32)
  PyG concatenated 32 full graphs, and the edge_dim=3 GATConv lin_edge
  projection materialized a (32 * 2*E, heads*hidden) tensor ~= 64 GB -> OOM.

  Because node features are gene-level means and the focal indicator sat on the
  same node for every variant of a gene, all variants of a gene already shared
  an identical input and an identical focal embedding -- i.e. the model was
  always gene-level (GNNScorer collapses to one score per gene downstream).

  This refactor trains transductively on ONE shared graph: a single forward
  produces all node embeddings; labeled focal-gene embeddings are gathered and
  classified. Memory is O(E) once, independent of the number of variants; the
  learned signal (gene-level) and the public API are unchanged.

CHANGES FROM PHASE 1:
  - nx.read_gpickle / nx.write_gpickle removed in NetworkX 3.3+ -> stdlib pickle.
  - Module-level logging.basicConfig removed (Issue L).
  - from __future__ import annotations added (Issue N).

Dependencies:
    pip install torch torch-geometric requests networkx pandas
"""

from __future__ import annotations

import gzip
import io
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

from genomic_variant_classifier.models.gnn_optim import bf16_autocast, denoise_string_edges, VariantGATGPS

logger = logging.getLogger(__name__)

STRING_URL = (
    "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/"
    "9606.protein.links.detailed.v12.0.txt.gz"
)
STRING_NAMES_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/"
    "9606.protein.info.v12.0.txt.gz"
)


# ---------------------------------------------------------------------------
# Graph construction  (UNCHANGED from the prior module)
# ---------------------------------------------------------------------------
class StringDBGraph:
    """Builds a NetworkX protein-interaction graph from STRING DB."""

    def __init__(
        self,
        cache_dir: Path = Path("data/raw/cache"),
        combined_score_threshold: int = 700,
        local_links_path: Optional[Path] = None,
        local_info_path: Optional[Path] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = combined_score_threshold
        self.local_links_path = Path(local_links_path) if local_links_path else None
        self.local_info_path = Path(local_info_path) if local_info_path else None
        self.graph: Optional[nx.Graph] = None
        self._protein_to_gene: dict[str, str] = {}

    def _download_gz(self, url: str) -> pd.DataFrame:
        logger.info("Downloading %s...", url)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        raw = b"".join(resp.iter_content(chunk_size=1 << 20))
        with gzip.open(io.BytesIO(raw), "rt") as fh:
            return pd.read_csv(fh, sep=" ")

    def _save_graph(self, G: nx.Graph, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_graph(self, path: Path) -> nx.Graph:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def _load_protein_names(self) -> dict[str, str]:
        cache = self.cache_dir / "string_names.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        elif self.local_info_path and self.local_info_path.exists():
            logger.info("Loading STRING protein info from %s", self.local_info_path)
            import gzip as _gz

            with _gz.open(self.local_info_path, "rt") as fh:
                df = pd.read_csv(fh, sep="\t")
            df.to_parquet(cache, index=False)
        else:
            df = self._download_gz(STRING_NAMES_URL)
            df.to_parquet(cache, index=False)
        id_col = (
            "#string_protein_id"
            if "#string_protein_id" in df.columns
            else df.columns[0]
        )
        name_col = "preferred_name" if "preferred_name" in df.columns else df.columns[1]
        return dict(zip(df[id_col], df[name_col]))

    def build(self, force_refresh: bool = False) -> nx.Graph:
        cache_path = self.cache_dir / f"string_graph_{self.threshold}.pkl"
        if cache_path.exists() and not force_refresh:
            logger.info("Loading cached STRING graph from %s", cache_path)
            self.graph = self._load_graph(cache_path)
            return self.graph

        self._protein_to_gene = self._load_protein_names()

        cache_links = self.cache_dir / "string_links.parquet"
        if cache_links.exists() and not force_refresh:
            links_df = pd.read_parquet(cache_links)
        elif self.local_links_path and self.local_links_path.exists():
            logger.info("Loading STRING links from %s", self.local_links_path)
            import gzip as _gz

            with _gz.open(self.local_links_path, "rt") as fh:
                links_df = pd.read_csv(fh, sep=" ")
            links_df.to_parquet(cache_links, index=False)
        else:
            links_df = self._download_gz(STRING_URL)
            links_df.to_parquet(cache_links, index=False)

        logger.info("Raw interactions: %d", len(links_df))
        links_df = links_df[links_df["combined_score"] >= self.threshold]
        logger.info("After threshold=%d: %d edges.", self.threshold, len(links_df))

        _CHANNELS = ["experiments", "database", "coexpression"]
        for ch in _CHANNELS:
            if ch not in links_df.columns:
                links_df[ch] = 0

        G = nx.Graph()
        for _, row in links_df.iterrows():
            p1 = self._protein_to_gene.get(row["protein1"], row["protein1"])
            p2 = self._protein_to_gene.get(row["protein2"], row["protein2"])
            G.add_edge(
                p1,
                p2,
                weight=float(row["combined_score"]) / 1000.0,
                experimental=float(row["experiments"]) / 1000.0,
                database=float(row["database"]) / 1000.0,
                coexpression=float(row["coexpression"]) / 1000.0,
            )

        self.graph = G
        self._save_graph(G, cache_path)
        logger.info(
            "STRING graph: %d nodes, %d edges. Saved to %s.",
            G.number_of_nodes(), G.number_of_edges(), cache_path,
        )
        return G

    def subgraph_for_genes(self, genes: list[str], n_hops: int = 1) -> nx.Graph:
        if self.graph is None:
            raise RuntimeError("Call build() before subgraph_for_genes().")
        seed_nodes = set(genes) & set(self.graph.nodes)
        if not seed_nodes:
            return nx.Graph()
        neighbors = set(seed_nodes)
        for _ in range(n_hops):
            new_neighbors: set[str] = set()
            for node in neighbors:
                new_neighbors.update(self.graph.neighbors(node))
            neighbors |= new_neighbors
        return self.graph.subgraph(neighbors).copy()


# ---------------------------------------------------------------------------
# Shared-graph dataset (Option B): ONE graph + per-sample focal indices
# ---------------------------------------------------------------------------
@dataclass
class SharedFocalGraph:
    """
    A single protein-interaction graph plus the labeled focal-gene samples.

    Replaces the prior list[Data] (one full-graph copy per variant). Holds the
    graph once; each labeled variant contributes a (focal_idx, label, variant_id)
    triple. Memory is O(n_nodes + n_edges + n_samples), not O(n_samples * n_edges).
    """

    x: torch.Tensor           # (n_nodes, n_feats)
    edge_index: torch.Tensor  # (2, E)
    edge_attr: torch.Tensor   # (E, 3)
    focal_idx: torch.Tensor   # (n_samples,) long — node index of each sample's gene
    y: torch.Tensor           # (n_samples,) long — binary label
    variant_ids: list[str]    # (n_samples,)
    node_genes: list[str]     # (n_nodes,) gene symbol at each node index (Option C)

    def __len__(self) -> int:
        return int(self.focal_idx.numel())

    def subset(self, idx) -> "SharedFocalGraph":
        """Same graph tensors; sliced focal samples (idx = array-like of positions)."""
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return SharedFocalGraph(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            focal_idx=self.focal_idx[idx_t],
            y=self.y[idx_t],
            variant_ids=[self.variant_ids[i] for i in idx_t.tolist()],
            node_genes=self.node_genes,
        )


def build_pyg_dataset(
    variant_df: pd.DataFrame,
    graph: nx.Graph,
    node_feature_cols: list[str],
    label_col: str = "acmg_label",
    edge_denoise: str = "none",
    edge_denoise_tau: float = 0.0,
) -> SharedFocalGraph:
    """
    Build a single shared-graph dataset for transductive focal-node training.

    Node features are gene-level means of each variant feature (as before).
    Returns ONE SharedFocalGraph; each labeled variant becomes a focal sample
    pointing at its gene's node. No per-variant graph replication, no focal
    indicator column (it was identical across a gene's variants and is a no-op
    for a single shared forward).
    """
    all_genes = list(graph.nodes)
    gene_index = {g: i for i, g in enumerate(all_genes)}
    n_nodes = len(all_genes)
    n_feats = len(node_feature_cols)

    gene_features = np.zeros((n_nodes, n_feats), dtype=np.float32)
    for feat_idx, feat in enumerate(node_feature_cols):
        if feat not in variant_df.columns:
            continue
        grp = variant_df.groupby("gene_symbol")[feat].mean()
        for gene, val in grp.items():
            if gene in gene_index:
                gene_features[gene_index[gene], feat_idx] = float(val)

    # Edge tensors: built ONCE for the whole graph.
    edge_pairs: list[list[int]] = []
    edge_attrs_list: list[list[float]] = []
    for u, v, attrs in graph.edges(data=True):
        if u in gene_index and v in gene_index:
            ea = [
                float(attrs.get("experimental", attrs.get("weight", 0.4))),
                float(attrs.get("database", attrs.get("weight", 0.4))),
                float(attrs.get("coexpression", attrs.get("weight", 0.4))),
            ]
            edge_pairs.append([gene_index[u], gene_index[v]])
            edge_attrs_list.append(ea)

    if edge_pairs:
        ei = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        ea = torch.tensor(edge_attrs_list, dtype=torch.float)
        edge_index = torch.cat([ei, ei.flip(0)], dim=1)   # undirected
        edge_attr = torch.cat([ea, ea], dim=0)            # (2*n_edges, 3)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)

    edge_index, edge_attr = denoise_string_edges(
        edge_index, edge_attr, mode=edge_denoise, tau=edge_denoise_tau
    )

    x = torch.tensor(gene_features, dtype=torch.float)

    focal: list[int] = []
    labels: list[int] = []
    vids: list[str] = []
    labeled = variant_df[variant_df[label_col].notna()]
    for _, row in labeled.iterrows():
        gene = row.get("gene_symbol")
        if gene not in gene_index:
            continue
        focal.append(gene_index[gene])
        labels.append(int(row[label_col]))
        vids.append(str(row.get("variant_id", "")))

    ds = SharedFocalGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        focal_idx=torch.tensor(focal, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
        variant_ids=vids,
        node_genes=list(all_genes),
    )
    logger.info(
        "Built shared-graph dataset: %d nodes, %d directed edges, %d labeled focal samples.",
        n_nodes, edge_index.shape[1], len(ds),
    )
    return ds


# ---------------------------------------------------------------------------
# Graph Attention Network  (in_channels = n_feats; focal-node readout)
# ---------------------------------------------------------------------------
class VariantGAT(nn.Module):
    """3-layer GAT with multi-head attention; focal-node readout; 2-class head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=3)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=3)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, edge_dim=3)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        gene_idx: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        focal_embeddings = x[gene_idx]            # (n_focal, out_channels)
        return self.classifier(focal_embeddings)  # (n_focal, 2)


# ---------------------------------------------------------------------------
# Transductive trainer (full-batch over the shared graph)
# ---------------------------------------------------------------------------
class GNNTrainer:
    """Trains VariantGAT transductively: one shared graph, focal-node loss."""

    def __init__(
        self,
        model: VariantGAT,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32,  # accepted for API compatibility; full-batch by default
        device: Optional[str] = None,
        checkpoint_path: str = "models/best_gat.pt",
        precision: str = "fp32",
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.precision = precision
        self.history: list[dict] = []

    def _graph_tensors(self, ds: SharedFocalGraph):
        cache = getattr(self, "_gt_cache", None)
        if cache is None:
            cache = {}
            self._gt_cache = cache
        key = id(ds)
        if key not in cache:
            cache[key] = (
                ds.x.to(self.device),
                ds.edge_index.to(self.device),
                ds.edge_attr.to(self.device),
            )
        return cache[key]

    def train_epoch(self, ds: SharedFocalGraph) -> float:
        self.model.train()
        x, ei, ea = self._graph_tensors(ds)
        focal = ds.focal_idx.to(self.device)
        y = ds.y.to(self.device)
        self.optimizer.zero_grad()
        with bf16_autocast(self.device, enabled=(self.precision == "bf16")):
            out = self.model(x, ei, focal, edge_attr=ea)   # one forward over the whole graph
            loss = F.cross_entropy(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, ds: SharedFocalGraph) -> tuple[float, float]:
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        x, ei, ea = self._graph_tensors(ds)
        focal = ds.focal_idx.to(self.device)
        with bf16_autocast(self.device, enabled=(self.precision == "bf16")):
            out = self.model(x, ei, focal, edge_attr=ea)
        proba = F.softmax(out.float(), dim=-1)[:, 1].cpu().numpy()
        labels = ds.y.cpu().numpy()
        auc = roc_auc_score(labels, proba) if len(np.unique(labels)) > 1 else 0.0
        logits = np.stack([1 - proba, proba], axis=1)
        ce = F.cross_entropy(
            torch.tensor(logits, dtype=torch.float),
            torch.tensor(labels, dtype=torch.long),
        ).item()
        return ce, auc

    def fit(
        self,
        train_dataset: SharedFocalGraph,
        val_dataset: SharedFocalGraph,
        patience: int = 15,
    ) -> list[dict]:
        best_val_auc = 0.0
        no_improve = 0
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_dataset)
            val_loss, val_auc = self.evaluate(val_dataset)
            self.scheduler.step()
            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc}
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), self.checkpoint_path)
                no_improve = 0
            else:
                no_improve += 1
            if epoch % 10 == 0:
                logger.info(
                    "Epoch %3d | Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f",
                    epoch, train_loss, val_loss, val_auc,
                )
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d. Best Val AUC: %.4f", epoch, best_val_auc)
                break
        return self.history

    def predict_proba(self, ds: SharedFocalGraph) -> np.ndarray:
        """Per-sample P(pathogenic), aligned with ds.variant_ids order."""
        self.model.eval()
        with torch.no_grad():
            x, ei, ea = self._graph_tensors(ds)
            focal = ds.focal_idx.to(self.device)
            with bf16_autocast(self.device, enabled=(self.precision == "bf16")):
                out = self.model(x, ei, focal, edge_attr=ea)
            return F.softmax(out.float(), dim=-1)[:, 1].cpu().numpy()

    @torch.no_grad()
    def score_all_nodes(self, ds: SharedFocalGraph) -> np.ndarray:
        """P(pathogenic) for EVERY node in the shared graph (inductive).

        Forwards the trained model once over the whole graph with the focal
        index set to all node indices, so every gene in STRING gets a score
        regardless of whether it was a labeled training focal sample. This is
        what lets gene-disjoint val/test rows receive real, varying
        gnn_score values (INCIDENT_2026-06-04_gnn-score-injection-degenerate).
        """
        self.model.eval()
        x, ei, ea = self._graph_tensors(ds)
        all_idx = torch.arange(x.shape[0], device=self.device)
        with bf16_autocast(self.device, enabled=(self.precision == "bf16")):
            out = self.model(x, ei, all_idx, edge_attr=ea)
        return F.softmax(out.float(), dim=-1)[:, 1].cpu().numpy()


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def train_gnn_pipeline(
    variant_df: pd.DataFrame,
    node_feature_cols: list[str],
    string_threshold: int = 700,
    string_kwargs: Optional[dict] = None,
    test_split: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    graph: Optional[nx.Graph] = None,
    precision: str = "fp32",
    edge_denoise: str = "none",
    edge_denoise_tau: float = 0.0,
    layer_type: str = "gat",
) -> tuple[VariantGAT, GNNTrainer, list[dict]]:
    """End-to-end GNN training pipeline (transductive, shared graph)."""
    from sklearn.model_selection import train_test_split

    if graph is None:
        _kwargs = dict(string_kwargs or {})
        _kwargs.setdefault("combined_score_threshold", string_threshold)
        builder = StringDBGraph(**_kwargs)
        graph = builder.build()

    ds = build_pyg_dataset(
        variant_df, graph, node_feature_cols,
        edge_denoise=edge_denoise, edge_denoise_tau=edge_denoise_tau,
    )
    n = len(ds)
    if n < 4:
        raise ValueError(f"Too few labeled focal samples for GNN training: {n}")

    y_np = ds.y.numpy()
    stratify = y_np if len(np.unique(y_np)) > 1 else None
    train_pos, val_pos = train_test_split(
        np.arange(n), test_size=test_split, random_state=42, stratify=stratify
    )
    train_ds, val_ds = ds.subset(train_pos), ds.subset(val_pos)

    in_channels = len(node_feature_cols)  # focal indicator dropped (Option B)
    if layer_type == "gps":
        model = VariantGATGPS(in_channels=in_channels, hidden_channels=128, heads=4)
    elif layer_type == "gat":
        model = VariantGAT(in_channels=in_channels, hidden_channels=128, heads=8)
    else:
        raise ValueError(f"unknown layer_type: {layer_type!r} (expected 'gat' or 'gps')")
    trainer = GNNTrainer(model, epochs=epochs, batch_size=batch_size, precision=precision)
    history = trainer.fit(train_ds, val_ds)
    return model, trainer, history


# ---------------------------------------------------------------------------
# Inference-time gene scorer
# ---------------------------------------------------------------------------
class GNNScorer:
    """Gene-level GNN scoring: averages GNN predictions per gene at fit time."""

    DEFAULT_SCORE = 0.5

    def __init__(self, gene_scores: dict[str, float]) -> None:
        self.gene_scores = gene_scores

    @classmethod
    def from_trainer(
        cls,
        trainer: "GNNTrainer",
        dataset: SharedFocalGraph,
        variant_df: pd.DataFrame,
    ) -> "GNNScorer":
        proba = trainer.predict_proba(dataset)  # (n_samples,), aligned with variant_ids

        vid_to_gene: dict[str, str] = {}
        if "variant_id" in variant_df.columns and "gene_symbol" in variant_df.columns:
            vid_to_gene = dict(
                zip(
                    variant_df["variant_id"].astype(str),
                    variant_df["gene_symbol"].fillna("").astype(str),
                )
            )

        gene_accumulator: dict[str, list[float]] = {}
        for vid, score in zip(dataset.variant_ids, proba):
            gene = vid_to_gene.get(str(vid), "")
            if gene:
                gene_accumulator.setdefault(gene, []).append(float(score))

        gene_scores = {g: float(np.mean(s)) for g, s in gene_accumulator.items()}
        logger.info(
            "GNNScorer built for %d genes (mean score %.3f).",
            len(gene_scores),
            float(np.mean(list(gene_scores.values()))) if gene_scores else 0.5,
        )
        return cls(gene_scores)

    @classmethod
    def from_full_graph(
        cls,
        trainer: "GNNTrainer",
        dataset: SharedFocalGraph,
    ) -> "GNNScorer":
        """Inductive scorer: one score per STRING node, keyed by gene symbol.

        Replaces from_trainer's variant_id-keyed accumulation (which produced an
        empty map when gnn_df lacked a variant_id column, collapsing every
        gnn_score to the 0.5 default in Run 15). Scores ALL graph nodes, so
        gene-disjoint val/test genes also receive real values.
        """
        if not getattr(dataset, "node_genes", None):
            raise ValueError(
                "from_full_graph requires dataset.node_genes; rebuild the "
                "SharedFocalGraph with build_pyg_dataset after the Option C patch."
            )
        node_scores = trainer.score_all_nodes(dataset)
        genes = dataset.node_genes
        if len(genes) != len(node_scores):
            raise ValueError(
                f"node_genes ({len(genes)}) != node_scores ({len(node_scores)})"
            )
        gene_scores = {str(g): float(s) for g, s in zip(genes, node_scores)}
        _vals = (
            np.fromiter(gene_scores.values(), dtype=float)
            if gene_scores else np.zeros(1)
        )
        logger.info(
            "GNNScorer.from_full_graph: %d gene scores (mean=%.3f std=%.4f).",
            len(gene_scores), float(_vals.mean()), float(_vals.std()),
        )
        return cls(gene_scores)

    def score(self, gene_symbol: str) -> float:
        return self.gene_scores.get(str(gene_symbol), self.DEFAULT_SCORE)

    def score_dataframe(self, df: pd.DataFrame) -> pd.Series:
        symbols = df.get("gene_symbol", pd.Series([""] * len(df), index=df.index)).fillna("")
        return symbols.map(self.score)
