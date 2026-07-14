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
            result["omim_n_diseases_molecular"] = pd.Series(dtype=int)
            result["omim_is_autosomal_dominant"] = pd.Series(dtype=int)
            return result

        gene_table = self._get_gene_table()

        result = df.copy()
        if gene_table.empty:
            result["omim_n_diseases"]           = DEFAULT_N_DISEASES
            result["omim_n_diseases_molecular"] = DEFAULT_N_DISEASES
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
        result["omim_n_diseases_molecular"] = (
            result["omim_n_diseases_molecular"].fillna(DEFAULT_N_DISEASES).astype(int)
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
        """Return a gene-level summary DataFrame, or empty if unavailable.

        genemap2.txt is the PRIMARY (and sufficient) source: it carries the
        gene->phenotype relationships from which omim_n_diseases,
        omim_n_diseases_molecular and omim_is_autosomal_dominant are all derived.
        mim2gene.txt is an ID cross-reference whose own header states it is NOT a
        gene-phenotype table, so it is no longer used for the disease count.
        """
        empty_cols = ["gene_symbol", "omim_n_diseases",
                      "omim_n_diseases_molecular", "omim_is_autosomal_dominant"]

        if self.genemap2_path is None or not self.genemap2_path.exists():
            logger.warning(
                "OMIMConnector: genemap2.txt not available (path=%s) -- returning default "
                "values (omim_n_diseases=0, omim_n_diseases_molecular=0, "
                "omim_is_autosomal_dominant=0).  Download genemap2.txt from "
                "https://omim.org/downloads.",
                self.genemap2_path,
            )
            return pd.DataFrame(columns=empty_cols)

        cache_key = f"gene_table:genemap2={self.genemap2_path}"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info("OMIMConnector: loaded gene table from cache (%d genes).", len(cached))
            return cached

        gene_table = self._parse_genemap2(self.genemap2_path)
        if not gene_table.empty:
            self._save_cache(cache_key, gene_table)
            logger.info("OMIMConnector: parsed and cached %d genes from genemap2.", len(gene_table))
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
            "OMIMConnector: parsed %d gene-phenotype entries -> %d unique genes.",
            len(raw), len(gene_counts),
        )
        return gene_counts


    @staticmethod
    def _count_phenotypes(phenotypes: str) -> "tuple[int, int, int]":
        """Return (n_diseases_all, n_diseases_molecular, is_autosomal_dominant)
        for one gene's genemap2 Phenotypes string.

        - n_diseases_all:       count of ;-separated entries that are real diseases
                                (EXCLUDES [non-disease] bracketed entries: biomarkers/QTLs).
                                INCLUDES plain, {susceptibility}, and ?provisional entries.
        - n_diseases_molecular: count of entries containing the (3) mapping key
                                = molecular basis of the disorder is known (confirmed gene).
        - is_autosomal_dominant: 1 if any entry mentions "Autosomal dominant".

        Counting entries that CONTAIN "(3)" is robust to the 2/8953 entries that
        embed a stray "(N)" inside disease text (verified against live genemap2).
        """
        import re as _re
        s = str(phenotypes).strip()
        if not s:
            return 0, 0, 0
        n_all = 0
        n_mol = 0
        is_ad = 0
        for entry in s.split(";"):
            e = entry.strip()
            if not e:
                continue
            if e.startswith("["):          # [non-disease] — exclude from disease counts entirely
                continue
            n_all += 1
            if _re.search(r"\(3\)", e):
                n_mol += 1
            if "autosomal dominant" in e.lower():
                is_ad = 1
        return n_all, n_mol, is_ad

    def _parse_genemap2(self, path: Path) -> pd.DataFrame:
        """Parse genemap2.txt into gene-level OMIM features.

        Returns gene_symbol, omim_n_diseases, omim_n_diseases_molecular,
        omim_is_autosomal_dominant (one row per gene; aggregated across the gene's
        genemap2 rows). genemap2.txt is the file that actually carries
        gene->phenotype relationships (mim2gene.txt explicitly is NOT).
        """
        empty_cols = ["gene_symbol", "omim_n_diseases",
                      "omim_n_diseases_molecular", "omim_is_autosomal_dominant"]
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            logger.error("OMIMConnector: failed to read genemap2 %s: %s", path, exc)
            return pd.DataFrame(columns=empty_cols)

        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("# Chromosome") and "Approved Gene Symbol" in line and "Phenotypes" in line:
                header_idx = i
                break

        if header_idx is None:
            logger.warning("OMIMConnector: could not find genemap2 header in %s.", path)
            return pd.DataFrame(columns=empty_cols)

        header = lines[header_idx].lstrip("# ").split("\t")
        rows = []

        for line in lines[header_idx + 1:]:
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == len(header):
                rows.append(parts)

        if not rows:
            return pd.DataFrame(columns=empty_cols)

        raw = pd.DataFrame(rows, columns=header)
        required = {"Approved Gene Symbol", "Phenotypes"}
        if not required.issubset(raw.columns):
            logger.warning("OMIMConnector: genemap2 missing required columns in %s.", path)
            return pd.DataFrame(columns=empty_cols)

        x = raw[["Approved Gene Symbol", "Phenotypes"]].copy()
        x = x.rename(columns={"Approved Gene Symbol": "gene_symbol"})
        x["gene_symbol"] = x["gene_symbol"].astype(str).str.strip()
        x["Phenotypes"] = x["Phenotypes"].astype(str)
        x = x[x["gene_symbol"].str.len() > 0].copy()

        counts = x["Phenotypes"].map(self._count_phenotypes)
        x["omim_n_diseases"]            = counts.map(lambda t: t[0]).astype(int)
        x["omim_n_diseases_molecular"]  = counts.map(lambda t: t[1]).astype(int)
        x["omim_is_autosomal_dominant"] = counts.map(lambda t: t[2]).astype(int)

        # One gene may appear on multiple genemap2 rows: take max per gene so a
        # gene's disease count / AD flag reflect its richest annotation.
        agg = (
            x.groupby("gene_symbol", as_index=False)[
                ["omim_n_diseases", "omim_n_diseases_molecular", "omim_is_autosomal_dominant"]
            ].max()
        )
        logger.info(
            "OMIMConnector: parsed genemap2 -> %d genes; %d with >=1 disease, "
            "%d with >=1 molecular (3) disease, %d autosomal-dominant.",
            len(agg),
            int((agg["omim_n_diseases"] > 0).sum()),
            int((agg["omim_n_diseases_molecular"] > 0).sum()),
            int((agg["omim_is_autosomal_dominant"] > 0).sum()),
        )
        return agg
