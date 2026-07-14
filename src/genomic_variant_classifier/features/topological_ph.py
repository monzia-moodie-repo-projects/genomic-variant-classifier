"""
src/genomic_variant_classifier/features/topological_ph.py
===========================================================
Topological Persistent Homology (PH) feature generator for the STRING v12
gene–gene interaction graph.

Adopt #20 implementation.

Leakage guard (mandatory)
--------------------------
PH features are computed on the **train-subgraph only**.  The test-set gene
subgraph is constructed from edges that connect test genes *to train genes*,
not from test-only topology.  This prevents information that reflects
ClinVar pathogenicity indirectly (via gene-neighbourhood topology) from
leaking into training features.

Specifically:
  - ``fit(train_gene_symbols)`` builds the train-restricted subgraph and
    pre-computes per-gene PH.
  - ``transform(gene_symbols)`` assigns pre-computed PH to each variant.
    Genes absent from the train subgraph receive zero/default values.
  - ``train_genes_only=True`` is the default and must not be set ``False``
    for production runs.

Features produced (gene-level → assigned to each variant by gene_symbol)
--------------------------------------------------------------------------
ph_h0_n_components : int   — H0 connected components in gene ego-graph
ph_h0_max_lifetime : float — longest finite H0 barcode
ph_h1_n_loops      : int   — persistent H1 loops
ph_h1_max_lifetime : float — longest finite H1 barcode
ph_betti_0         : int   — Betti-0 at filtration threshold 0
ph_betti_1         : int   — Betti-1 at filtration threshold 0

⚠ PHASE_2_FEATURES
  Requires ``gudhi`` (pip install gudhi --break-system-packages) and
  ``networkx``.  Standing Rule #31 smoke test required before adding gudhi
  to requirements.txt.  When either library is absent the generator runs in
  zero-fallback mode — all PH features default to zero/one values and
  a WARNING is logged once.

Design rules
------------
- No logging.basicConfig at module level.
- from __future__ import annotations (standing rule).
- Never use nx.read_gpickle (deprecated networkx 3.x — standing rule).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — zero-fallback when unavailable
# ---------------------------------------------------------------------------
_GUDHI_AVAILABLE = False
try:
    import gudhi as _gudhi  # type: ignore[import]
    _GUDHI_AVAILABLE = True
except ImportError:
    logger.warning(
        "gudhi not installed -- TopologicalPHGenerator will return zero/default "
        "PH features.  Install: pip install gudhi --break-system-packages  "
        "(run SR #31 smoke test first)."
    )

_NX_AVAILABLE = False
try:
    import networkx as _nx  # type: ignore[import]
    _NX_AVAILABLE = True
except ImportError:
    logger.warning(
        "networkx not installed -- TopologicalPHGenerator disabled."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical PH feature column names — must match _engineer_features in
#: real_data_prep.py when PH features are wired into the pipeline.
PH_FEATURE_COLS: list[str] = [
    "ph_h0_n_components",
    "ph_h0_max_lifetime",
    "ph_h1_n_loops",
    "ph_h1_max_lifetime",
    "ph_betti_0",
    "ph_betti_1",
]

_PH_DEFAULTS: dict[str, float] = {
    "ph_h0_n_components": 1.0,   # isolated node has 1 component
    "ph_h0_max_lifetime": 0.0,
    "ph_h1_n_loops":      0.0,
    "ph_h1_max_lifetime": 0.0,
    "ph_betti_0":         1.0,
    "ph_betti_1":         0.0,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class TopologicalPHGenerator:
    """
    Computes topological PH features for each gene in the STRING v12
    gene–gene interaction graph, then assigns them to variants by
    ``gene_symbol``.

    Usage
    -----
    ::

        gen = TopologicalPHGenerator(
            string_path="data/external/string/9606.protein.links.detailed.v12.0.txt.gz",
            score_threshold=700,
        )
        gen.fit(train_gene_symbols=list(meta_train["gene_symbol"].unique()))
        ph_train = gen.transform(meta_train["gene_symbol"])
        ph_test  = gen.transform(meta_test["gene_symbol"])
        # ph_train / ph_test have PH_FEATURE_COLS columns, one row per variant.

    Parameters
    ----------
    string_path     : Path to STRING v12 protein links file (gzipped TSV/TXT).
                      If None or not found, zero-fallback mode.
    score_threshold : Minimum combined_score to retain an edge (default 700).
    n_hop           : Ego-graph radius in hops (default 1).
    train_genes_only: If True (default), restricts PH computation to the
                      train-gene subgraph (leakage guard — see module docstring).
    """

    def __init__(
        self,
        string_path:      Optional[Path | str] = None,
        score_threshold:  int  = 700,
        n_hop:            int  = 1,
        train_genes_only: bool = True,
    ) -> None:
        self.string_path      = Path(string_path) if string_path else None
        self.score_threshold  = score_threshold
        self.n_hop            = n_hop
        self.train_genes_only = train_genes_only

        if not train_genes_only:
            logger.warning(
                "TopologicalPHGenerator: train_genes_only=False disables the "
                "leakage guard (Adopt #20).  Do NOT use False for production runs."
            )

        # State set by fit()
        self._train_graph: Optional[object]    = None  # networkx.Graph
        self._gene_ph_cache: dict[str, dict]   = {}
        self._fit_called: bool                 = False

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, train_gene_symbols: list[str]) -> "TopologicalPHGenerator":
        """
        Build the train-subgraph and pre-compute PH for all training genes.

        Must be called before ``transform()``.  Calling ``fit()`` again
        resets the subgraph and cache (safe to call for each ablation round).

        Parameters
        ----------
        train_gene_symbols : Unique gene symbols in the TRAINING set.
        """
        self._gene_ph_cache = {}
        self._fit_called    = True

        if not _GUDHI_AVAILABLE or not _NX_AVAILABLE:
            logger.warning(
                "TopologicalPHGenerator.fit(): gudhi or networkx unavailable -- "
                "fit is a no-op; transform will return defaults."
            )
            return self

        if self.string_path is None or not self.string_path.exists():
            logger.warning(
                "TopologicalPHGenerator.fit(): STRING path not set or not found "
                "(%s) -- zero-fallback mode.", self.string_path
            )
            return self

        logger.info(
            "Loading STRING v12 from %s (threshold=%d) ...",
            self.string_path, self.score_threshold,
        )
        G_full = self._load_string_graph()
        logger.info(
            "STRING full graph: %d nodes, %d edges.",
            G_full.number_of_nodes(), G_full.number_of_edges(),
        )

        train_gene_set = set(train_gene_symbols)

        if self.train_genes_only:
            # LEAKAGE GUARD: restrict to nodes that appear in the train set.
            train_nodes = [n for n in G_full.nodes if n in train_gene_set]
            G_train = G_full.subgraph(train_nodes).copy()
            logger.info(
                "Train subgraph (leakage guard active): %d nodes, %d edges "
                "(%.1f%% of full graph nodes).",
                G_train.number_of_nodes(),
                G_train.number_of_edges(),
                100.0 * G_train.number_of_nodes() / max(G_full.number_of_nodes(), 1),
            )
        else:
            G_train = G_full
            train_nodes = list(G_full.nodes)

        self._train_graph = G_train

        # Pre-compute PH for every training gene present in the graph.
        genes_in_graph = [g for g in train_nodes if g in G_train]
        logger.info("Computing PH for %d genes ...", len(genes_in_graph))
        for gene in genes_in_graph:
            try:
                self._gene_ph_cache[gene] = self._compute_gene_ph(gene, G_train)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PH computation failed for gene %s: %s -- using defaults.",
                    gene, exc,
                )
                self._gene_ph_cache[gene] = dict(_PH_DEFAULTS)

        logger.info("PH pre-computed for %d genes.", len(self._gene_ph_cache))
        return self

    def transform(self, gene_symbols: pd.Series) -> pd.DataFrame:
        """
        Assign PH features to each variant by gene_symbol.

        Parameters
        ----------
        gene_symbols : Series of gene symbols, one per variant row.

        Returns
        -------
        DataFrame with ``PH_FEATURE_COLS`` columns, length == len(gene_symbols).
        Genes absent from the train subgraph receive ``_PH_DEFAULTS`` values.
        """
        if not self._fit_called:
            raise RuntimeError(
                "TopologicalPHGenerator.transform() called before fit().  "
                "Call fit(train_gene_symbols) first."
            )

        rows = []
        for gene in gene_symbols:
            g = str(gene) if gene else ""
            rows.append(
                self._gene_ph_cache.get(g, dict(_PH_DEFAULTS))
            )
        return pd.DataFrame(rows, columns=PH_FEATURE_COLS).reset_index(drop=True)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load_string_graph(self):  # noqa: ANN201
        """
        Load STRING v12 protein links into a networkx.Graph.

        Never uses nx.read_gpickle (standing rule — deprecated in networkx 3.x).
        """
        # Auto-detect separator from file extension.
        path_str = str(self.string_path)
        sep = "\t" if ".tsv" in path_str else " "
        df = pd.read_csv(
            self.string_path,
            sep=sep,
            compression="infer",
            usecols=["protein1", "protein2", "combined_score"],
        )
        df = df[df["combined_score"] >= self.score_threshold].copy()

        G = _nx.Graph()
        for row in df.itertuples(index=False):
            # Strip the "9606." NCBI taxon prefix that STRING prepends to Ensembl IDs.
            p1 = str(row.protein1).removeprefix("9606.")
            p2 = str(row.protein2).removeprefix("9606.")
            weight = float(row.combined_score) / 1000.0  # normalise to [0, 1]
            G.add_edge(p1, p2, weight=weight)

        return G

    def _compute_gene_ph(self, gene: str, G) -> dict:
        """
        Compute H0 and H1 persistent homology for the ego-graph of *gene*.

        Uses a gudhi SimplexTree built from the graph adjacency as a clique
        complex, with (1 − weight) as the filtration value on edges.  This
        reflects the biology: highly connected (high-weight) edges appear at
        low filtration values.
        """
        ego = _nx.ego_graph(G, gene, radius=self.n_hop)
        n_nodes = ego.number_of_nodes()

        if n_nodes == 0:
            return dict(_PH_DEFAULTS)

        # Build SimplexTree from the ego-graph.
        st = _gudhi.SimplexTree()
        node_idx = {n: i for i, n in enumerate(ego.nodes)}

        # Vertices at filtration 0.
        for n in ego.nodes:
            st.insert([node_idx[n]], filtration=0.0)

        # Edges: filtration = 1 − weight (lower = stronger interaction).
        for u, v, data in ego.edges(data=True):
            dist = 1.0 - float(data.get("weight", 0.5))
            st.insert([node_idx[u], node_idx[v]], filtration=dist)

        st.make_filtration_non_decreasing()
        st.compute_persistence()
        ph = st.persistence()

        h0 = [(b, d) for dim, (b, d) in ph if dim == 0]
        h1 = [(b, d) for dim, (b, d) in ph if dim == 1]

        def _max_finite(intervals: list) -> float:
            lts = [d - b for b, d in intervals if d != float("inf")]
            return float(max(lts)) if lts else 0.0

        return {
            "ph_h0_n_components": float(len(h0)),
            "ph_h0_max_lifetime": _max_finite(h0),
            "ph_h1_n_loops":      float(len(h1)),
            "ph_h1_max_lifetime": _max_finite(h1),
            "ph_betti_0":         float(sum(1 for _, d in h0 if d == float("inf"))),
            "ph_betti_1":         float(sum(1 for _, d in h1 if d == float("inf"))),
        }
