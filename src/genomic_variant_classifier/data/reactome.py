"""
src/genomic_variant_classifier/data/reactome.py
================================================
Reactome pathway-membership connector — Phase D, Connector (new).

Adds one gene-level feature:

    reactome_pathway_count   int   Number of distinct Reactome pathways the
                                   gene participates in.  Genes embedded in many
                                   curated pathways tend to be more functionally
                                   central, and pathway membership is a curated,
                                   non-fabricated signal.  Default: 0 (gene absent
                                   from Reactome / unknown).

Gene-level join: by HGNC gene_symbol (mirrors the gnomAD-constraint join key,
not the variant-level chrom:pos:ref:alt key used by dbSNP/SpliceAI).

Expected parquet columns:
    gene_symbol             str   HGNC gene symbol
    reactome_pathway_count  int   Distinct pathway count for the gene

Stub mode:
    When pathway_path is None or the file does not exist, every variant receives
    reactome_pathway_count = 0 and a WARNING is logged.  The pipeline continues
    without raising — identical contract to DbSNPConnector.

Data source:
    Reactome bulk mapping files (https://reactome.org/download-data):
      Ensembl2Reactome_All_Levels.txt / UniProt2Reactome_All_Levels.txt /
      NCBI2Reactome_All_Levels.txt
    Build the per-gene parquet with scripts/build_reactome_parquet.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from genomic_variant_classifier.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

DEFAULT_PATHWAY_COUNT = 0


class ReactomeConnector(BaseConnector):
    """
    Annotates variants with the Reactome pathway count of their gene.

    Usage
    -----
        connector = ReactomeConnector(
            pathway_path="data/external/reactome_gene_pathways.parquet"
        )
        annotated_df = connector.annotate_dataframe(variant_df)
        # annotated_df now has a reactome_pathway_count column

    If pathway_path is None or the file is absent, stub mode applies:
    all variants receive reactome_pathway_count = 0.
    """

    source_name = "reactome"

    def __init__(
        self,
        pathway_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.pathway_path: Optional[Path] = (
            Path(pathway_path) if pathway_path is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a reactome_pathway_count column to df (gene-level).

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame; must contain gene_symbol for joining.

        Returns
        -------
        pd.DataFrame with reactome_pathway_count column added.
        """
        if df.empty:
            result = df.copy()
            result["reactome_pathway_count"] = pd.Series(dtype="int64")
            return result

        if self.pathway_path is None:
            logger.warning(
                "ReactomeConnector: pathway_path not set -- returning "
                "reactome_pathway_count=0.  Build it from Reactome bulk data with "
                "scripts/build_reactome_parquet.py."
            )
            result = df.copy()
            result["reactome_pathway_count"] = DEFAULT_PATHWAY_COUNT
            return result

        lookup = self._get_lookup()
        if lookup.empty:
            result = df.copy()
            result["reactome_pathway_count"] = DEFAULT_PATHWAY_COUNT
            return result

        return self._annotate(df, lookup)

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_lookup(self) -> pd.DataFrame:
        """Return the gene→pathway-count lookup, using the parquet cache when present."""
        cache_key = "pathway_lookup"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info(
                "ReactomeConnector: loaded %d gene pathway counts from cache.",
                len(cached),
            )
            return cached

        if not self.pathway_path.exists():
            logger.warning(
                "ReactomeConnector: parquet not found at '%s' -- returning "
                "reactome_pathway_count=0.",
                self.pathway_path,
            )
            return pd.DataFrame(columns=["gene_symbol", "reactome_pathway_count"])

        logger.info(
            "ReactomeConnector: loading Reactome parquet from %s ...",
            self.pathway_path,
        )
        try:
            lookup = pd.read_parquet(
                self.pathway_path, columns=["gene_symbol", "reactome_pathway_count"]
            )
        except Exception as exc:
            logger.error("ReactomeConnector: failed to read parquet: %s", exc)
            return pd.DataFrame(columns=["gene_symbol", "reactome_pathway_count"])

        lookup = lookup.dropna(subset=["gene_symbol"])
        lookup["reactome_pathway_count"] = (
            pd.to_numeric(lookup["reactome_pathway_count"], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(lower=0)
        )
        lookup = lookup.drop_duplicates(subset=["gene_symbol"])

        self._save_cache(cache_key, lookup)
        logger.info("ReactomeConnector: cached %d gene pathway counts.", len(lookup))
        return lookup

    def _annotate(self, variant_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
        """Left-join Reactome pathway counts onto variant_df by gene_symbol."""
        result = variant_df.copy()

        # Defensive: drop any pre-existing column so a re-run cannot create
        # reactome_pathway_count_x / _y suffixes from the merge.
        if "reactome_pathway_count" in result.columns:
            result = result.drop(columns=["reactome_pathway_count"])

        gene = result.get(
            "gene_symbol", pd.Series([""] * len(result), index=result.index)
        ).astype(str)
        result["_gene_key"] = gene

        counts = (
            lookup[["gene_symbol", "reactome_pathway_count"]]
            .rename(columns={"gene_symbol": "_gene_key"})
            .drop_duplicates(subset=["_gene_key"])
        )

        result = result.merge(counts, on="_gene_key", how="left")
        result["reactome_pathway_count"] = (
            result["reactome_pathway_count"]
            .fillna(DEFAULT_PATHWAY_COUNT)
            .astype(int)
            .clip(lower=0)
        )
        result = result.drop(columns=["_gene_key"])

        n_found = int((result["reactome_pathway_count"] > 0).sum())
        logger.info(
            "ReactomeConnector: %d / %d variants have reactome_pathway_count > 0.",
            n_found,
            len(result),
        )
        return result
