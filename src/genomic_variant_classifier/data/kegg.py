"""
src/genomic_variant_classifier/data/kegg.py
===========================================
KEGG pathway-membership connector -- Phase 2 (2026-07-06).

Adds two GENE-LEVEL features from the Kyoto Encyclopedia of Genes and Genomes
(KEGG) human pathway maps:

    kegg_pathway_count        float  Number of distinct KEGG pathways the gene
                                     participates in (functional centrality proxy;
                                     0 = gene absent from KEGG / unknown).
    kegg_disease_pathway_flag float  1.0 if the gene is a member of at least one
                                     KEGG "Human Diseases" pathway (map id range
                                     hsa05xxx: cancers, neurodegenerative, immune,
                                     etc.), else 0.0. This curated disease-map
                                     membership is KEGG's distinctive signal, not
                                     carried by the Reactome connector.

Gene-level join: by HGNC gene_symbol (mirrors the gnomAD-constraint / Reactome
join key, not the variant-level chrom:pos:ref:alt key). These are curated
membership signals, NOT derived from the pathogenicity label.

Expected parquet columns (built once by scripts/build_kegg_parquet.py from the
KEGG REST API):
    gene_symbol               str
    kegg_pathway_count        int/float
    kegg_disease_pathway_flag int/float (0/1)

Stub mode: when kegg_path is None or the file is absent, every variant receives
the defaults (0.0) and a WARNING is logged; the pipeline continues without
raising -- identical contract to the Reactome / gnomAD-constraint connectors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

KEGG_FEATURES = ["kegg_pathway_count", "kegg_disease_pathway_flag"]
KEGG_DEFAULTS: dict[str, float] = {
    "kegg_pathway_count": 0.0,
    "kegg_disease_pathway_flag": 0.0,
}


class KEGGConnector:
    """Annotate variants with KEGG pathway count + disease-pathway membership of their gene."""

    source_name = "kegg"

    def __init__(self, kegg_path: Optional[str | Path] = None) -> None:
        self._path: Optional[Path] = Path(kegg_path) if kegg_path is not None else None
        self._lookup: Optional[pd.DataFrame] = None  # lazy, cached in-memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add kegg_pathway_count + kegg_disease_pathway_flag (gene-level). Returns a copy."""
        result = df.copy()
        for col, default in KEGG_DEFAULTS.items():
            if col not in result.columns:
                result[col] = default

        if "gene_symbol" not in result.columns:
            logger.info("KEGGConnector: gene_symbol column absent -- returning defaults.")
            return result
        if self._path is None:
            logger.warning(
                "KEGGConnector: kegg_path not set -- kegg_* default to 0. "
                "Build it with scripts/build_kegg_parquet.py."
            )
            return result
        if not self._path.exists():
            logger.warning(
                "KEGGConnector: parquet not found at '%s' -- kegg_* default to 0.",
                self._path,
            )
            return result

        lookup = self._get_lookup()
        if lookup.empty:
            return result
        return self._annotate(result, lookup)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _get_lookup(self) -> pd.DataFrame:
        if self._lookup is not None:
            return self._lookup
        try:
            lookup = pd.read_parquet(self._path, columns=["gene_symbol"] + KEGG_FEATURES)
        except Exception as exc:  # missing column / corrupt -> loud, then defaults
            logger.error("KEGGConnector: failed to read parquet '%s': %s", self._path, exc)
            self._lookup = pd.DataFrame(columns=["gene_symbol"] + KEGG_FEATURES)
            return self._lookup

        lookup = lookup.dropna(subset=["gene_symbol"])
        for c in KEGG_FEATURES:
            lookup[c] = pd.to_numeric(lookup[c], errors="coerce").fillna(0.0).clip(lower=0.0)
        lookup = lookup.drop_duplicates(subset=["gene_symbol"])
        self._lookup = lookup
        logger.info("KEGGConnector: loaded %d gene pathway rows.", len(lookup))
        return self._lookup

    def _annotate(self, variant_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
        result = variant_df
        # Defensive: drop pre-existing feature cols so a re-run can't create _x/_y suffixes.
        for c in KEGG_FEATURES:
            if c in result.columns:
                result = result.drop(columns=[c])

        gene = result.get(
            "gene_symbol", pd.Series([""] * len(result), index=result.index)
        ).astype(str)
        result = result.copy()
        result["_gene_key"] = gene

        counts = (
            lookup[["gene_symbol"] + KEGG_FEATURES]
            .rename(columns={"gene_symbol": "_gene_key"})
            .drop_duplicates(subset=["_gene_key"])
        )
        result = result.merge(counts, on="_gene_key", how="left")
        for c in KEGG_FEATURES:
            result[c] = (
                pd.to_numeric(result[c], errors="coerce")
                .fillna(KEGG_DEFAULTS[c])
                .astype(float)
                .clip(lower=0.0)
            )
        result = result.drop(columns=["_gene_key"])

        n_path = int((result["kegg_pathway_count"] > 0).sum())
        n_dis = int((result["kegg_disease_pathway_flag"] > 0).sum())
        logger.info(
            "KEGGConnector: %d / %d variants have kegg_pathway_count > 0; %d in a KEGG disease pathway.",
            n_path, len(result), n_dis,
        )
        return result
