from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FinnGen R10 population AF connector
# ---------------------------------------------------------------------------
# Data source: https://r10.finngen.fi/
# File: finngen_R10_annotated_variants_v1.gz  (or current release equiv.)
# Columns used: #chrom, pos, ref, alt, af_fin, af_nfsee, rsid
#
# Feature columns produced:
#   finngen_af_fin     — Finnish population allele frequency
#   finngen_af_nfsee   — Non-Finnish SEE allele frequency (comparison anchor)
#   finngen_enrichment — af_fin / (af_nfsee + 1e-9); Finnish enrichment ratio
#
# Used as third-tier AF fallback after gnomAD and 1KGP:
#   gnomAD AF → 1KGP AF → FinnGen AF → 0.0 default
# ---------------------------------------------------------------------------

FINNGEN_COLUMNS = [
    "finngen_af_fin",
    "finngen_af_nfsee",
    "finngen_enrichment",
]


def finngen_columns(column_prefix: str = "") -> list[str]:
    """The three FinnGen output column names for a given release prefix.

    prefix=""      -> finngen_af_fin / finngen_af_nfsee / finngen_enrichment (R12)
    prefix="r13_"  -> finngen_r13_af_fin / finngen_r13_af_nfsee / finngen_r13_enrichment
    """
    return [
        f"finngen_{column_prefix}af_fin",
        f"finngen_{column_prefix}af_nfsee",
        f"finngen_{column_prefix}enrichment",
    ]

_CHROM_NORMALISE = {str(i): str(i) for i in range(1, 23)}
_CHROM_NORMALISE.update({"X": "X", "Y": "Y", "MT": "MT", "M": "MT"})


def _normalise_chrom(c: str) -> str:
    c = str(c).replace("chr", "").upper()
    return _CHROM_NORMALISE.get(c, c)


class FinnGenConnector:
    """
    Annotates a variant DataFrame with FinnGen R10 population AF columns.

    Parameters
    ----------
    tsv_path:
        Path to the FinnGen R10 annotated variants TSV (gzipped or plain).
        Download from https://r10.finngen.fi/
        Expected columns: #chrom, pos, ref, alt, af_fin, af_nfsee
    chunksize:
        Rows per chunk when reading the large TSV. Default 500_000.
    """

    def __init__(
        self,
        tsv_path: Optional[str | Path] = None,
        chunksize: int = 500_000,
        column_prefix: str = "",
    ) -> None:
        self.tsv_path = Path(tsv_path) if tsv_path else None
        self.chunksize = chunksize
        self.column_prefix = column_prefix
        # Output column names; default prefix "" reproduces the R12 names exactly.
        self._out_fin = f"finngen_{column_prefix}af_fin"
        self._out_nfsee = f"finngen_{column_prefix}af_nfsee"
        self._out_enrich = f"finngen_{column_prefix}enrichment"
        self._out_cols = [self._out_fin, self._out_nfsee, self._out_enrich]
        self._index: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add finngen_af_fin, finngen_af_nfsee, finngen_enrichment columns
        to *df* in-place and return it.

        Variants with no FinnGen match receive 0.0 / 0.0 / 1.0 defaults.
        """
        for col in self._out_cols:
            if col not in df.columns:
                df[col] = 0.0

        if self.tsv_path is None or not Path(self.tsv_path).exists():
            logger.warning(
                "FinnGenConnector: tsv_path not set or file not found (%s). "
                "All variants will receive finngen_af_fin=0.0. "
                "Download from https://r10.finngen.fi/",
                self.tsv_path,
            )
            df[self._out_enrich] = 1.0
            return df

        if self._index is None:
            self._index = self._load_full_index()

        if self._index.empty:
            df[self._out_enrich] = 1.0
            return df

        # Join on chrom / pos / ref / alt
        query_keys = df[["chrom", "pos", "ref", "alt"]].copy()
        query_keys["chrom"] = query_keys["chrom"].astype(str).map(_normalise_chrom)

        merged = query_keys.merge(
            self._index,
            on=["chrom", "pos", "ref", "alt"],
            how="left",
        )

        df[self._out_fin]   = merged["af_fin"].fillna(0.0).values
        df[self._out_nfsee] = merged["af_nfsee"].fillna(0.0).values
        df[self._out_enrich] = (
            df[self._out_fin] / (df[self._out_nfsee] + 1e-9)
        ).clip(upper=1000.0)

        n_annotated = (df[self._out_fin] > 0).sum()
        logger.info(
            "FinnGen annotation: %d / %d variants matched (%.1f%%).",
            n_annotated, len(df), 100 * n_annotated / max(len(df), 1),
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_full_index(self) -> pd.DataFrame:
        """
        Load the COMPLETE FinnGen release index (all variants), using a parquet
        cache for fast warm starts.

        The first call extracts the full ``.gz`` once (~20M rows) and writes a
        parquet cache plus a sidecar signature file; every subsequent call loads
        the parquet in seconds. The cache is invalidated automatically if the
        source ``.gz`` size or mtime changes.

        The exact (chrom, pos, ref, alt) left-join in :meth:`annotate` makes a
        bounding-box pre-filter unnecessary: a full index yields identical
        matches while being reusable across any cohort.
        """
        pq_path, meta_path = self._cache_paths()
        sig = self._source_signature()

        # 1. Warm start: a valid, matching cache -> load parquet.
        if pq_path.exists() and meta_path.exists():
            try:
                cached_sig = json.loads(meta_path.read_text())
                if (
                    cached_sig.get("size") == sig["size"]
                    and cached_sig.get("mtime_ns") == sig["mtime_ns"]
                ):
                    idx = pd.read_parquet(pq_path)
                    logger.info(
                        "FinnGen: loaded %d variants from cache %s.",
                        len(idx), pq_path,
                    )
                    return idx
                logger.info(
                    "FinnGen: cache stale (source changed) -> rebuilding %s.",
                    pq_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FinnGen: cache read failed (%s) -> rebuilding.", exc
                )

        # 2. Cold start: extract the FULL release (no bounding box) and cache it.
        logger.info(
            "FinnGen: building FULL index from %s (one-time) ...", self.tsv_path
        )
        compression = "gzip" if str(self.tsv_path).endswith(".gz") else "infer"
        chunks = []
        try:
            reader = pd.read_csv(
                self.tsv_path,
                sep="\t",
                comment=None,
                chunksize=self.chunksize,
                compression=compression,
                usecols=["chr", "pos", "ref", "alt", "GENOME_AF_fin", "GENOME_AF_nfe"],
                dtype={"chr": str, "pos": int, "ref": str, "alt": str,
                       "GENOME_AF_fin": float, "GENOME_AF_nfe": float},
            )
            for chunk in reader:
                chunk.rename(
                    columns={"chr": "chrom", "GENOME_AF_fin": "af_fin",
                             "GENOME_AF_nfe": "af_nfsee"},
                    inplace=True,
                )
                chunk["chrom"] = chunk["chrom"].map(_normalise_chrom)
                chunks.append(chunk)
        except Exception as exc:  # noqa: BLE001
            logger.error("FinnGen: failed to read TSV: %s", exc)
            return pd.DataFrame(columns=["chrom", "pos", "ref", "alt",
                                         "af_fin", "af_nfsee"])

        if not chunks:
            logger.warning("FinnGen: no rows read from TSV.")
            return pd.DataFrame(columns=["chrom", "pos", "ref", "alt",
                                         "af_fin", "af_nfsee"])

        index = pd.concat(chunks, ignore_index=True).drop_duplicates(
            subset=["chrom", "pos", "ref", "alt"]
        )

        # Write cache (best-effort; a cache-write failure is never fatal).
        try:
            pq_path.parent.mkdir(parents=True, exist_ok=True)
            index.to_parquet(pq_path, index=False)
            meta_path.write_text(json.dumps(sig))
            logger.info(
                "FinnGen: wrote full-index cache -> %s (%d variants).",
                pq_path, len(index),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinnGen: could not write cache (%s).", exc)

        return index

    def _cache_paths(self) -> tuple[Path, Path]:
        """(parquet, sidecar) paths for this release's full-index cache."""
        cache_dir = Path("data/raw/cache")
        stem = f"finngen_{self.column_prefix}full_index"
        return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.meta.json"

    def _source_signature(self) -> dict:
        """Size + mtime of the source .gz, used for cache invalidation."""
        st = Path(self.tsv_path).stat()
        return {
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "source": str(self.tsv_path),
        }