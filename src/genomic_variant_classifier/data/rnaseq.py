"""
src/genomic_variant_classifier/data/rnaseq.py
=============================================
RNA-seq gene-expression connector -- Phase D, Connector (new).

Adds five GENE-LEVEL features derived from a quantified RNA-seq expression
matrix (built offline by scripts/build_rnaseq_parquet.py from a gene x sample
or transcript x sample matrix):

    rnaseq_mean_log_tpm    float  mean log1p(TPM) across samples
    rnaseq_detection_rate  float  fraction of samples expressing the gene
    rnaseq_log2_cv         float  log2(1 + CV) of TPM across samples (dispersion)
    rnaseq_log2fc          float  log2 fold-change case/control (0 if no DE)
    rnaseq_de_neglog10p    float  -log10(p) case-vs-control test (0 if no DE)

Gene-level join: by HGNC gene_symbol (mirrors gnomAD-constraint / Reactome /
GTEx-bulk join key, not the variant-level chrom:pos:ref:alt key).

Stub mode: when rnaseq_path is None or absent, every variant receives the
defaults (all 0.0 except the int-typed columns) and a WARNING is logged; the
pipeline continues -- identical contract to ReactomeConnector / the GTEx bulk
path.

LEAKAGE NOTE: rnaseq_log2fc / rnaseq_de_neglog10p (when present in the parquet)
must have been computed on an RNA-seq cohort INDEPENDENT of the variant-label
cohort, else the differential-expression signal leaks the label.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RNASEQ_FEATURES = [
    "rnaseq_mean_log_tpm",
    "rnaseq_detection_rate",
    "rnaseq_log2_cv",
    "rnaseq_log2fc",
    "rnaseq_de_neglog10p",
]
_RNASEQ_DEFAULTS = {c: 0.0 for c in RNASEQ_FEATURES}


def _apply_defaults(result: pd.DataFrame) -> pd.DataFrame:
    for col, default in _RNASEQ_DEFAULTS.items():
        result[col] = default
    return result


def annotate_rnaseq_from_parquet(
    df: pd.DataFrame, parquet_path: "str | Path"
) -> pd.DataFrame:
    """Gene-level left-join of the bulk RNA-seq expression features.

    NaN-safe and defensive against pre-existing columns (no _x/_y suffixes on a
    re-run). When the parquet is missing/unreadable, all five features default
    to 0.0 and the pipeline continues (no silent failure -- a WARNING is logged).
    """
    result = df.copy()
    pth = Path(parquet_path) if parquet_path is not None else None
    if pth is None or not pth.exists():
        logger.warning(
            "RNASeqConnector: parquet not set/found ('%s') -- rnaseq_* features "
            "default to 0. Build it with scripts/build_rnaseq_parquet.py.",
            pth,
        )
        return _apply_defaults(result)

    try:
        lookup = pd.read_parquet(pth, columns=["gene_symbol"] + RNASEQ_FEATURES)
    except Exception as exc:  # missing column / corrupt file -> loud, then defaults
        logger.error("RNASeqConnector: failed to read parquet '%s': %s", pth, exc)
        return _apply_defaults(result)

    lookup = lookup.dropna(subset=["gene_symbol"]).drop_duplicates(
        subset=["gene_symbol"]
    )

    for col in RNASEQ_FEATURES:
        if col in result.columns:
            result = result.drop(columns=[col])

    gene = result.get(
        "gene_symbol", pd.Series([""] * len(result), index=result.index)
    ).astype(str)
    result["_gene_key"] = gene
    lk = lookup.rename(columns={"gene_symbol": "_gene_key"})
    result = result.merge(lk, on="_gene_key", how="left").drop(columns=["_gene_key"])

    for col in RNASEQ_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    n_expr = int((result["rnaseq_mean_log_tpm"] > 0).sum())
    n_de = int((result["rnaseq_de_neglog10p"] > 0).sum())
    logger.info(
        "RNASeqConnector: %d / %d variants have rnaseq expression; %d with DE signal.",
        n_expr,
        len(result),
        n_de,
    )
    return result
