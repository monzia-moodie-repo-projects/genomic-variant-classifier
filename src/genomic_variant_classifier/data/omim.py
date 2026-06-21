"""
src/genomic_variant_classifier/data/omim.py
================
OMIM gene-disease connector — Phase 4, Connector 2.

Reads a downloaded mim2gene.txt flat file (available from
https://omim.org/downloads after free registration) and adds two gene-level
features to the variant DataFrame:

    omim_n_diseases           int   Number of OMIM phenotype entries for the gene
    omim_is_autosomal_dominant int  1 if any phenotype has autosomal dominant
                                     inheritance (requires a phenotype annotation
                                     file; defaults to 0 from mim2gene.txt alone)

mim2gene.txt column layout (tab-separated, comment lines start with '#'):
    MIM_number   MIM_type   Entrez_ID   HGNC_symbol   Ensembl_ID

MIM_type values that count as phenotype:
    "phenotype"
    "predominantly phenotypes"

Gene-level join: left-join by gene_symbol (= HGNC_symbol in mim2gene).
Missing genes → omim_n_diseases = 0, omim_is_autosomal_dominant = 0.

Stub mode:
    When mim2gene_path is None or the file does not exist, all variants receive
    the default values and a WARNING is logged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from genomic_variant_classifier.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

PHENOTYPE_TYPES = {"phenotype", "predominantly phenotypes"}
DEFAULT_N_DISEASES  = 0
DEFAULT_IS_AD       = 0


class OMIMConnector(BaseConnector):
    """
    Annotates variants with OMIM gene-disease features.

    Usage
    -----
        connector = OMIMConnector(mim2gene_path="data/external/mim2gene.txt")
        annotated_df = connector.annotate_dataframe(variant_df)

    If mim2gene_path is None or file is absent, stub mode applies:
    all variants receive omim_n_diseases=0, omim_is_autosomal_dominant=0.
    """

    source_name = "omim"

    def __init__(
        self,
        mim2gene_path: Optional[str | Path] = None,
        genemap2_path: Optional[str | Path] = None,
        api_key: Optional[str] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.mim2gene_path: Optional[Path] = (
            Path(mim2gene_path) if mim2gene_path is not None else None
        )
        self.genemap2_path: Optional[Path] = (
            Path(genemap2_path) if genemap2_path is not None else None
        )
        self.api_key = api_key   # reserved for future REST mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add omim_n_diseases and omim_is_autosomal_dominant to df.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame; must contain 'gene_symbol' for a real join.

        Returns
        -------
        pd.DataFrame with two new columns added.
        """
        if df.empty:
            result = df.copy()
            result["omim_n_diseases"]           = pd.Series(dtype=int)
            result["omim_is_autosomal_dominant"] = pd.Series(dtype=int)
            return result

        gene_table = self._get_gene_table()

        result = df.copy()
        if gene_table.empty:
            result["omim_n_diseases"]           = DEFAULT_N_DISEASES
            result["omim_is_autosomal_dominant"] = DEFAULT_IS_AD
            return result

        result = result.merge(
            gene_table,
            left_on="gene_symbol",
            right_on="gene_symbol",
            how="left",
        )
        result["omim_n_diseases"] = (
            result["omim_n_diseases"].fillna(DEFAULT_N_DISEASES).astype(int)
        )
        result["omim_is_autosomal_dominant"] = (
            result["omim_is_autosomal_dominant"].fillna(DEFAULT_IS_AD).astype(int)
        )

        n_annotated = (result["omim_n_diseases"] > 0).sum()
        logger.debug(
            "OMIMConnector: %d / %d variants annotated with omim_n_diseases > 0.",
            n_annotated, len(result),
        )
        return result

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_gene_table(self) -> pd.DataFrame:
        """Return a gene-level summary DataFrame, or empty if unavailable."""
        if self.mim2gene_path is None:
            logger.warning(
                "OMIMConnector: mim2gene_path not set — returning default values "
                "(omim_n_diseases=0, omim_is_autosomal_dominant=0).  "
                "Download mim2gene.txt from https://omim.org/downloads."
            )
            return pd.DataFrame(columns=["gene_symbol", "omim_n_diseases", "omim_is_autosomal_dominant"])

        cache_key = f"gene_table:mim2gene={self.mim2gene_path}:genemap2={self.genemap2_path}"
        cached = None
        if self.genemap2_path is None:
            cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info("OMIMConnector: loaded gene table from cache (%d genes).", len(cached))
            return cached

        if not self.mim2gene_path.exists():
            logger.warning(
                "OMIMConnector: mim2gene.txt not found at '%s' — returning default values.",
                self.mim2gene_path,
            )
            return pd.DataFrame(columns=["gene_symbol", "omim_n_diseases", "omim_is_autosomal_dominant"])

        gene_table = self._parse_mim2gene(self.mim2gene_path)
        if self.genemap2_path is not None and self.genemap2_path.exists():
            ad_table = self._parse_genemap2_autosomal_dominant(self.genemap2_path)
            if not ad_table.empty:
                gene_table = gene_table.drop(columns=["omim_is_autosomal_dominant"], errors="ignore")
                gene_table = gene_table.merge(ad_table, on="gene_symbol", how="outer")
                gene_table["omim_n_diseases"] = (
                    gene_table["omim_n_diseases"].fillna(DEFAULT_N_DISEASES).astype(int)
                )
                gene_table["omim_is_autosomal_dominant"] = (
                    gene_table["omim_is_autosomal_dominant"].fillna(DEFAULT_IS_AD).astype(int)
                )
        if not gene_table.empty:
            if self.genemap2_path is None:
                self._save_cache(cache_key, gene_table)
                logger.info("OMIMConnector: parsed and cached %d genes.", len(gene_table))
            else:
                logger.info("OMIMConnector: parsed %d genes with genemap2 cache bypass.", len(gene_table))
        return gene_table

    def _parse_mim2gene(self, path: Path) -> pd.DataFrame:
        """
        Parse mim2gene.txt into a gene-level feature DataFrame.

        Expected columns (tab-separated, comment lines skipped):
            MIM_number  MIM_type  Entrez_ID  HGNC_symbol  Ensembl_ID
        """
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue
                    mim_type   = parts[1].strip().lower()
                    hgnc_sym   = parts[3].strip()
                    if mim_type in PHENOTYPE_TYPES and hgnc_sym:
                        rows.append({"gene_symbol": hgnc_sym})
        except OSError as exc:
            logger.error("OMIMConnector: failed to read %s: %s", path, exc)
            return pd.DataFrame(columns=["gene_symbol", "omim_n_diseases", "omim_is_autosomal_dominant"])

        if not rows:
            return pd.DataFrame(columns=["gene_symbol", "omim_n_diseases", "omim_is_autosomal_dominant"])

        raw = pd.DataFrame(rows)
        # Count phenotype entries per gene → omim_n_diseases
        gene_counts = (
            raw.groupby("gene_symbol")
            .size()
            .rename("omim_n_diseases")
            .reset_index()
        )
        # omim_is_autosomal_dominant: mim2gene.txt has no inheritance field
        # → default to 0; a phenotype annotation file would be needed for real values
        gene_counts["omim_is_autosomal_dominant"] = DEFAULT_IS_AD

        logger.info(
            "OMIMConnector: parsed %d gene–phenotype entries → %d unique genes.",
            len(raw), len(gene_counts),
        )
        return gene_counts


    def _parse_genemap2_autosomal_dominant(self, path: Path) -> pd.DataFrame:
        """Parse genemap2.txt into gene-level autosomal-dominant flags."""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            logger.error("OMIMConnector: failed to read genemap2 %s: %s", path, exc)
            return pd.DataFrame(columns=["gene_symbol", "omim_is_autosomal_dominant"])

        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("# Chromosome") and "Approved Gene Symbol" in line and "Phenotypes" in line:
                header_idx = i
                break

        if header_idx is None:
            logger.warning("OMIMConnector: could not find genemap2 header in %s.", path)
            return pd.DataFrame(columns=["gene_symbol", "omim_is_autosomal_dominant"])

        header = lines[header_idx].lstrip("# ").split("\t")
        rows = []

        for line in lines[header_idx + 1:]:
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == len(header):
                rows.append(parts)

        if not rows:
            return pd.DataFrame(columns=["gene_symbol", "omim_is_autosomal_dominant"])

        raw = pd.DataFrame(rows, columns=header)
        required = {"Approved Gene Symbol", "Phenotypes"}
        if not required.issubset(raw.columns):
            logger.warning("OMIMConnector: genemap2 missing required columns in %s.", path)
            return pd.DataFrame(columns=["gene_symbol", "omim_is_autosomal_dominant"])

        x = raw[["Approved Gene Symbol", "Phenotypes"]].copy()
        x = x.rename(columns={"Approved Gene Symbol": "gene_symbol"})
        x["gene_symbol"] = x["gene_symbol"].astype(str).str.strip()
        x["Phenotypes"] = x["Phenotypes"].astype(str)
        x = x[x["gene_symbol"].str.len() > 0].copy()

        x["omim_is_autosomal_dominant"] = (
            x["Phenotypes"]
            .str.contains("Autosomal dominant", case=False, na=False)
            .astype(int)
        )

        return (
            x.groupby("gene_symbol", as_index=False)["omim_is_autosomal_dominant"]
            .max()
        )
