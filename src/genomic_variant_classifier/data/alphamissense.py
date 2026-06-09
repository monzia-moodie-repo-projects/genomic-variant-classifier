"""
src/genomic_variant_classifier/data/alphamissense.py
==========================
AlphaMissense pre-computed score connector -- Phase 2, Pillar 1.

Annotates variants with AlphaMissense missense pathogenicity scores from
DeepMind's pre-computed lookup file (AlphaMissense_hg38.tsv.gz).

Default for non-missense variants or variants not in the file: 0.5 (ambiguous).

Data source (free, no account required):
  https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz
  Place at: data/external/alphamissense/AlphaMissense_hg38.tsv.gz
  Or run:   bash scripts/download_data.sh alphamissense

ANNOTATOR pattern (identical to SpliceAIConnector):
  connector = AlphaMissenseConnector(tsv_path="path/to/AlphaMissense_hg38.tsv.gz")
  annotated_df = connector.fetch(variant_df=canonical_df)

PHASE_2_PLACEHOLDER: UniProt residue cross-referencing via parse_hgvsp().
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from genomic_variant_classifier.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

AM_PATHOGENIC_THRESHOLD: float = 0.564
AM_BENIGN_THRESHOLD: float     = 0.340
AM_DEFAULT_SCORE: float        = 0.5

DEFAULT_TSV_PATH = Path("data/external/alphamissense/AlphaMissense_hg38.tsv.gz")


class AlphaMissenseConnector(BaseConnector):
    """
    Annotates variants with AlphaMissense pathogenicity scores.

    If tsv_path is None or the file does not exist, all variants receive
    alphamissense_score = 0.5 (ambiguous) and a WARNING is logged.
    """

    source_name = "alphamissense"

    def __init__(
        self,
        tsv_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.tsv_path: Optional[Path] = (
            Path(tsv_path) if tsv_path is not None else None
        )

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if variant_df.empty:
            df = variant_df.copy()
            df["alphamissense_score"] = pd.Series(dtype=float)
            return df

        if self.tsv_path is None:
            logger.debug(
                "AlphaMissenseConnector: no tsv_path provided, "
                "returning %.1f scores.", AM_DEFAULT_SCORE
            )
            df = variant_df.copy()
            df["alphamissense_score"] = AM_DEFAULT_SCORE
            return df

        # Build the cohort's lookup keys up front so the (huge) source file can
        # be filtered to just these variants DURING parsing. This bounds the
        # in-memory lookup to the cohort size (<=27k smoke / <=2.5M full) instead
        # of materialising + sorting + de-duplicating all ~71M rows -- the cause
        # of the Run-15 smoke OOM/swap hang (instance 40170927).
        cohort_keys = self._cohort_keys(variant_df)
        lookup = self._get_lookup(cohort_keys)
        if lookup.empty:
            df = variant_df.copy()
            df["alphamissense_score"] = AM_DEFAULT_SCORE
            return df

        return self._annotate(variant_df, lookup)

    @staticmethod
    def _cohort_keys(variant_df: pd.DataFrame) -> set[str]:
        """Lookup keys for the cohort, normalised IDENTICALLY to _parse_*/_annotate."""
        return set(
            variant_df["chrom"].astype(str).str.lstrip("chr") + ":" +
            variant_df["pos"].astype(str)                      + ":" +
            variant_df["ref"].astype(str).str.upper()          + ":" +
            variant_df["alt"].astype(str).str.upper()
        )

    def _get_lookup(self, cohort_keys: Optional[set[str]] = None) -> pd.DataFrame:
        cache_key = "scores_hg38"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info(
                "AlphaMissense: loaded %d scores from parquet cache.", len(cached)
            )
            return cached

        if not self.tsv_path.exists():
            logger.warning(
                "AlphaMissense TSV not found at '%s' -- setting "
                "alphamissense_score=%.1f for all variants. "
                "Download with: bash scripts/download_data.sh alphamissense",
                self.tsv_path, AM_DEFAULT_SCORE,
            )
            return pd.DataFrame(columns=["lookup_key", "alphamissense_score"])

        logger.info(
            "AlphaMissense: building lookup from %s "
            "(first run -- takes ~3 minutes for the full 71M-row file)...",
            self.tsv_path,
        )
        if str(self.tsv_path).endswith(".parquet"):
            lookup = self._parse_parquet(self.tsv_path, cohort_keys)
        else:
            lookup = self._parse_tsv(self.tsv_path, cohort_keys)
        # Only the FULL (cohort-independent) lookup may be written under the
        # cohort-independent cache key. A cohort-filtered lookup written there
        # would be silently served to a LATER run with a different cohort and
        # truncate its scores -- so we never cache it under this key.
        if not lookup.empty and cohort_keys is None:
            self._save_cache(cache_key, lookup)
            logger.info("AlphaMissense: cached %d variant scores.", len(lookup))
        elif not lookup.empty:
            logger.info(
                "AlphaMissense: cohort-filtered lookup (%d rows) -- not cached "
                "under the cohort-independent key.", len(lookup)
            )
        return lookup

    def _parse_parquet(
        self, path: Path, cohort_keys: Optional[set[str]] = None
    ) -> pd.DataFrame:
        """Read pre-built AlphaMissense parquet index and return lookup_key/alphamissense_score."""
        df = pd.read_parquet(path, columns=["chrom", "pos", "ref", "alt", "alphamissense_score"])
        df["chrom"] = df["chrom"].astype(str).str.lstrip("chr")
        df["ref"]   = df["ref"].astype(str).str.upper()
        df["alt"]   = df["alt"].astype(str).str.upper()
        df["lookup_key"] = (
            df["chrom"] + ":" +
            df["pos"].astype(str) + ":" +
            df["ref"] + ":" +
            df["alt"]
        )
        df = df[["lookup_key", "alphamissense_score"]].copy()
        df["alphamissense_score"] = df["alphamissense_score"].astype("float32")
        if cohort_keys is not None:
            df = df[df["lookup_key"].isin(cohort_keys)]
        df = (
            df
            .sort_values("alphamissense_score", ascending=False)
            .drop_duplicates(subset=["lookup_key"], keep="first")
            .reset_index(drop=True)
        )
        return df

    def _parse_tsv(
        self, path: Path, cohort_keys: Optional[set[str]] = None
    ) -> pd.DataFrame:
        _COL_NAMES = [
            "CHROM", "POS", "REF", "ALT", "genome",
            "uniprot_id", "transcript_id", "protein_variant",
            "am_pathogenicity", "am_class",
        ]
        _USECOLS = ["CHROM", "POS", "REF", "ALT", "am_pathogenicity"]
        _DTYPES = {
            "CHROM": str, "POS": "int32", "REF": str, "ALT": str,
            "am_pathogenicity": "float32",
        }
        _CHUNK_SIZE = 500_000

        chunks: list[pd.DataFrame] = []
        n_rows = 0

        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for chunk in pd.read_csv(
                f,
                sep="\t",
                comment="#",
                header=None,
                names=_COL_NAMES,
                usecols=_USECOLS,
                dtype=_DTYPES,
                chunksize=_CHUNK_SIZE,
            ):
                chunk["CHROM"] = chunk["CHROM"].str.lstrip("chr")
                chunk["REF"]   = chunk["REF"].str.upper()
                chunk["ALT"]   = chunk["ALT"].str.upper()
                chunk["lookup_key"] = (
                    chunk["CHROM"].astype(str) + ":" +
                    chunk["POS"].astype(str)   + ":" +
                    chunk["REF"]               + ":" +
                    chunk["ALT"]
                )
                sub = chunk[["lookup_key", "am_pathogenicity"]].rename(
                    columns={"am_pathogenicity": "alphamissense_score"}
                )
                if cohort_keys is not None:
                    sub = sub[sub["lookup_key"].isin(cohort_keys)]
                if not sub.empty:
                    chunks.append(sub)
                n_rows += len(chunk)
                if n_rows % 10_000_000 == 0:
                    logger.info(
                        "AlphaMissense: parsed %dM rows...", n_rows // 1_000_000
                    )

        logger.info("AlphaMissense: finished parsing -- %d rows total.", n_rows)
        if cohort_keys is not None:
            logger.info(
                "AlphaMissense: cohort filter active -- retained rows for up to "
                "%d cohort keys.", len(cohort_keys)
            )

        if not chunks:
            return pd.DataFrame(columns=["lookup_key", "alphamissense_score"])

        lookup = pd.concat(chunks, ignore_index=True)
        lookup = (
            lookup
            .sort_values("alphamissense_score", ascending=False)
            .drop_duplicates(subset=["lookup_key"], keep="first")
            .reset_index(drop=True)
        )
        logger.info(
            "AlphaMissense: %d unique loci after deduplication.", len(lookup)
        )
        return lookup

    def _annotate(
        self, variant_df: pd.DataFrame, lookup: pd.DataFrame
    ) -> pd.DataFrame:
        df = variant_df.copy()

        df["_lookup_key"] = (
            df["chrom"].astype(str).str.lstrip("chr") + ":" +
            df["pos"].astype(str)                      + ":" +
            df["ref"].astype(str).str.upper()          + ":" +
            df["alt"].astype(str).str.upper()
        )

        score_cols = lookup[["lookup_key", "alphamissense_score"]].rename(
            columns={"lookup_key": "_lookup_key"}
        )

        df = df.merge(score_cols, on="_lookup_key", how="left")
        df["alphamissense_score"] = (
            df["alphamissense_score"]
            .fillna(AM_DEFAULT_SCORE)
            .astype(float)
            .clip(0.0, 1.0)
        )
        df = df.drop(columns=["_lookup_key"])

        n_found    = (df["alphamissense_score"] != AM_DEFAULT_SCORE).sum()
        n_likely_p = (df["alphamissense_score"] >= AM_PATHOGENIC_THRESHOLD).sum()
        n_likely_b = (df["alphamissense_score"] <= AM_BENIGN_THRESHOLD).sum()
        logger.info(
            "AlphaMissense: %d / %d variants annotated "
            "(%d likely_pathogenic >= %.3f, %d likely_benign <= %.3f).",
            n_found, len(df),
            n_likely_p, AM_PATHOGENIC_THRESHOLD,
            n_likely_b, AM_BENIGN_THRESHOLD,
        )
        return df
