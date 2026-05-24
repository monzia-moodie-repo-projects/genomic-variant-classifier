"""
PrimateAI-3D Connector
========================
Lookup-join PrimateAI-3D pathogenicity scores for missense variants.

Run 11 integration: Standard scope item #6b.

PrimateAI-3D provides precomputed pathogenicity scores for all possible
human missense variants, trained on primate evolutionary constraint.
Higher scores indicate higher predicted pathogenicity.

Data source:
  https://storage.googleapis.com/dm-primateai3d/
  File: primateai3d_scores_hg38.tsv.gz (~4 GB)

Columns expected in the TSV:
  chr, pos, ref, alt, score

Usage:
    from genomic_variant_classifier.data.primateai3d import PrimateAI3DConnector
    connector = PrimateAI3DConnector(
        tsv_path="data/external/primateai3d/primateai3d_scores_hg38.tsv.gz"
    )
    df = connector.annotate(df)  # adds primateai3d_score column

PHASE_2_PLACEHOLDER:
  Parquet index build (like SpliceAI) is deferred to Run 12.
  Current implementation reads the full TSV which is slow but correct.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Output column name (must match TABULAR_FEATURES when wired)
PRIMATEAI3D_SCORE_COL = "primateai3d_score"
DEFAULT_SCORE = 0.0  # no signal for variants not in the lookup


class PrimateAI3DConnector:
    """
    Annotate variants with PrimateAI-3D pathogenicity scores.

    Strategy: precomputed lookup table, left-join by (chrom, pos, ref, alt).
    Variants not found receive DEFAULT_SCORE (0.0).
    """

    def __init__(
        self,
        tsv_path: Optional[str] = None,
        parquet_path: Optional[str] = None,
        chrom_col: str = "chr",
        pos_col: str = "pos",
        ref_col: str = "ref",
        alt_col: str = "alt",
        score_col: str = "score",
    ):
        self.tsv_path = tsv_path
        self.parquet_path = parquet_path
        self.chrom_col = chrom_col
        self.pos_col = pos_col
        self.ref_col = ref_col
        self.alt_col = alt_col
        self.score_col = score_col
        self._lookup: Optional[pd.DataFrame] = None

    def _load_lookup(self) -> pd.DataFrame:
        """Load the PrimateAI-3D score lookup table."""
        if self._lookup is not None:
            return self._lookup

        if self.parquet_path and Path(self.parquet_path).exists():
            logger.info("Loading PrimateAI-3D from parquet: %s", self.parquet_path)
            self._lookup = pd.read_parquet(self.parquet_path)
        elif self.tsv_path and Path(self.tsv_path).exists():
            logger.info("Loading PrimateAI-3D from TSV: %s", self.tsv_path)
            self._lookup = pd.read_csv(
                self.tsv_path,
                sep="\t",
                usecols=[self.chrom_col, self.pos_col, self.ref_col,
                         self.alt_col, self.score_col],
                dtype={
                    self.chrom_col: str,
                    self.pos_col: "int64",
                    self.ref_col: str,
                    self.alt_col: str,
                    self.score_col: "float32",
                },
            )
            logger.info("  Loaded %d scores", len(self._lookup))
        else:
            logger.warning(
                "PrimateAI-3D: no data file found. "
                "All variants will receive %s=%s. "
                "Download from: https://storage.googleapis.com/dm-primateai3d/",
                PRIMATEAI3D_SCORE_COL, DEFAULT_SCORE,
            )
            self._lookup = pd.DataFrame(
                columns=[self.chrom_col, self.pos_col, self.ref_col,
                         self.alt_col, self.score_col]
            )

        # Normalize chromosome names
        if self.chrom_col in self._lookup.columns:
            self._lookup[self.chrom_col] = (
                self._lookup[self.chrom_col]
                .astype(str)
                .str.replace("chr", "", regex=False)
            )

        # Rename to standard columns for joining
        self._lookup = self._lookup.rename(columns={
            self.chrom_col: "chrom",
            self.pos_col: "pos",
            self.ref_col: "ref",
            self.alt_col: "alt",
            self.score_col: PRIMATEAI3D_SCORE_COL,
        })

        # Deduplicate
        before = len(self._lookup)
        self._lookup = self._lookup.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
        if len(self._lookup) < before:
            logger.info("  Deduplicated: %d -> %d", before, len(self._lookup))

        return self._lookup

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add primateai3d_score column to the input DataFrame.

        Join keys: chrom, pos, ref, alt (all must exist in df).
        Missing variants receive DEFAULT_SCORE.

        Returns the input DataFrame with the new column added.
        """
        lookup = self._load_lookup()

        if lookup.empty:
            logger.warning(
                "PrimateAI-3D lookup is empty. "
                "Setting %s=%s for all %d variants.",
                PRIMATEAI3D_SCORE_COL, DEFAULT_SCORE, len(df),
            )
            df[PRIMATEAI3D_SCORE_COL] = DEFAULT_SCORE
            return df

        # Verify join columns exist
        required_cols = ["chrom", "pos", "ref", "alt"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(
                "PrimateAI-3D: missing join columns %s. "
                "Setting %s=%s for all variants.",
                missing, PRIMATEAI3D_SCORE_COL, DEFAULT_SCORE,
            )
            df[PRIMATEAI3D_SCORE_COL] = DEFAULT_SCORE
            return df

        # Normalize chrom in input
        df_chrom = df["chrom"].astype(str).str.replace("chr", "", regex=False)

        # Build join key
        df_key = (
            df_chrom + ":" +
            df["pos"].astype(str) + ":" +
            df["ref"].astype(str) + ":" +
            df["alt"].astype(str)
        )
        lookup_key = (
            lookup["chrom"] + ":" +
            lookup["pos"].astype(str) + ":" +
            lookup["ref"] + ":" +
            lookup["alt"]
        )

        # Create lookup dict for fast access
        score_map = dict(zip(lookup_key, lookup[PRIMATEAI3D_SCORE_COL]))

        # Map scores
        df[PRIMATEAI3D_SCORE_COL] = df_key.map(score_map).fillna(DEFAULT_SCORE).astype(np.float32)

        n_annotated = (df[PRIMATEAI3D_SCORE_COL] != DEFAULT_SCORE).sum()
        logger.info(
            "PrimateAI-3D: %d/%d variants annotated (%.1f%%)",
            n_annotated, len(df), 100 * n_annotated / max(len(df), 1),
        )

        return df

    def build_parquet_index(self, output_path: str) -> None:
        """
        Convert the raw TSV to a filtered Parquet index for faster loading.

        This is analogous to the SpliceAI parquet index build.
        Call once, then use parquet_path for subsequent runs.
        """
        if not self.tsv_path or not Path(self.tsv_path).exists():
            raise FileNotFoundError(f"TSV not found: {self.tsv_path}")

        lookup = self._load_lookup()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        lookup.to_parquet(str(output), compression="zstd", index=False)
        size_mb = output.stat().st_size / 1e6
        logger.info(
            "PrimateAI-3D parquet index: %d rows, %.1f MB -> %s",
            len(lookup), size_mb, output,
        )
