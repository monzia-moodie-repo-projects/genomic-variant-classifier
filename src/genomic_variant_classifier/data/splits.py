"""
src/genomic_variant_classifier/data/splits.py
===============================================
Gene-stratified and hash-stable split utilities.

Provides
--------
gene_stratified_split
    Three-way train/val/test gene-disjoint split using a deterministic
    hash function.  Reproducible and stable as the dataset grows.

unseen_gene_holdout_split
    Two-way train/holdout gene-disjoint split.  Hash-based so that genes
    assigned to the holdout bucket remain there when new genes are added
    (Rule 6 longitudinal-comparison invariant; C3 hypothesis gate for Run 15).

split_summary
    Produces a summary DataFrame (n_variants, n_genes, prevalence) for a
    dict of named index arrays.

time_disjoint_split
    ClinVar LastEvaluated date split.
    PHASE_3_FEATURES — deferred; raises NotImplementedError until a
    ClinVar re-pull populates the last_evaluated column.

Design rules
------------
- No logging.basicConfig at module level.
- from __future__ import annotations (standing rule).
- Hash-based algorithm (hashlib.md5) ensures stability as datasets grow:
  adding new genes never moves existing genes between buckets.
- gene_stratified_split and unseen_gene_holdout_split do NOT require a
  label column — they partition by gene identity only.
- All public functions raise ValueError (not KeyError) when required
  columns are absent, for consistent error handling in callers.

Hash stability guarantee
------------------------
For a fixed (seed, holdout_frac), the set of genes in the holdout is
determined solely by hashlib.md5(f"{seed}:{gene}") — independent of the
total number of genes in the dataset.  This means:

    holdout_genes(df_v1) ⊆ holdout_genes(df_v2)

whenever df_v2 is a superset of df_v1 (same genes plus new ones).
GroupShuffleSplit does NOT provide this guarantee and is therefore not
used here.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gene_hash(gene: str, seed: int) -> float:
    """
    Deterministic float in [0, 1) for a gene symbol and seed.

    Uses MD5 of ``f"{seed}:{gene}"`` for speed and stability.  The seed
    namespace prevents accidental collisions between different experiments
    that use the same gene symbols.
    """
    raw = f"{seed}:{gene}".encode()
    return int(hashlib.md5(raw).hexdigest(), 16) / (2 ** 128)


def _check_gene_col(df: pd.DataFrame, gene_col: str) -> None:
    """Raise ValueError (not KeyError) when gene_col is absent."""
    if gene_col not in df.columns:
        raise ValueError(
            f"Required column '{gene_col}' not found in DataFrame.  "
            f"Available columns: {list(df.columns)[:10]}.  "
            "Ensure the DataFrame has a gene_symbol column."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gene_stratified_split(
    df: pd.DataFrame,
    test_frac:  float = 0.20,
    val_frac:   float = 0.10,
    seed:       int   = 42,
    gene_col:   str   = "gene_symbol",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three-way gene-disjoint train / val / test split.

    Every variant belonging to a gene lands in exactly one split.  The
    assignment is deterministic and hash-stable (see module docstring).

    Parameters
    ----------
    df        : DataFrame with a *gene_col* column.
    test_frac : Fraction of genes to assign to test (default 0.20).
    val_frac  : Fraction of genes to assign to val (default 0.10).
    seed      : Hash seed for reproducibility.
    gene_col  : Column containing gene identifiers.

    Returns
    -------
    train_idx, val_idx, test_idx : integer position arrays into *df*.
        Every row appears in exactly one array.

    Raises
    ------
    ValueError
        If *gene_col* is absent, or if test_frac + val_frac >= 1.0.
    """
    _check_gene_col(df, gene_col)

    if not (0.0 < test_frac < 1.0):
        raise ValueError(
            f"test_frac must be in (0, 1), got {test_frac}."
        )
    if not (0.0 < val_frac < 1.0):
        raise ValueError(
            f"val_frac must be in (0, 1), got {val_frac}."
        )
    if test_frac + val_frac >= 1.0:
        raise ValueError(
            f"test_frac + val_frac = {test_frac + val_frac:.3f} >= 1.0; "
            "not enough data for training."
        )

    genes = df[gene_col].fillna("unknown").unique()
    hashes = {g: _gene_hash(g, seed) for g in genes}

    # Bucket assignment: [0, test_frac) -> test
    #                    [test_frac, test_frac+val_frac) -> val
    #                    [test_frac+val_frac, 1.0) -> train
    test_boundary = test_frac
    val_boundary  = test_frac + val_frac

    gene_hash_s = df[gene_col].fillna("unknown").map(hashes)
    te_mask = gene_hash_s < test_boundary
    va_mask = (gene_hash_s >= test_boundary) & (gene_hash_s < val_boundary)
    tr_mask = gene_hash_s >= val_boundary

    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]
    te_idx = np.where(te_mask)[0]

    n_genes   = len(genes)
    n_tr_g    = df.iloc[tr_idx][gene_col].nunique()
    n_va_g    = df.iloc[va_idx][gene_col].nunique()
    n_te_g    = df.iloc[te_idx][gene_col].nunique()
    logger.info(
        "gene_stratified_split (seed=%d): "
        "train=%d rows (%d genes), val=%d rows (%d genes), "
        "test=%d rows (%d genes)  total_genes=%d",
        seed,
        len(tr_idx), n_tr_g,
        len(va_idx), n_va_g,
        len(te_idx), n_te_g,
        n_genes,
    )
    return tr_idx, va_idx, te_idx


def unseen_gene_holdout_split(
    df:           pd.DataFrame,
    holdout_frac: float = 0.20,
    seed:         int   = 42,
    gene_col:     str   = "gene_symbol",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-way gene-disjoint train / holdout split.

    Uses a hash-based assignment so that the holdout gene set is stable
    as the dataset grows (Rule 6 / C3 gate invariant — see module docstring).

    Parameters
    ----------
    df           : DataFrame with a *gene_col* column.
    holdout_frac : Fraction of genes to assign to the holdout (default 0.20).
    seed         : Hash seed for reproducibility.
    gene_col     : Column containing gene identifiers.

    Returns
    -------
    train_idx, holdout_idx : integer position arrays into *df*.

    Raises
    ------
    ValueError
        If *gene_col* is absent, or if *holdout_frac* is not in (0, 1).
    """
    _check_gene_col(df, gene_col)

    if not (0.0 < holdout_frac < 1.0):
        raise ValueError(
            f"holdout_frac must be strictly in (0, 1), got {holdout_frac}.  "
            "Use a value such as 0.20 (20% holdout)."
        )

    genes   = df[gene_col].fillna("unknown").unique()
    n_genes = len(genes)

    holdout_genes = frozenset(
        g for g in genes if _gene_hash(g, seed) < holdout_frac
    )
    train_genes = frozenset(g for g in genes if g not in holdout_genes)

    ho_mask = df[gene_col].fillna("unknown").isin(holdout_genes)
    tr_idx  = np.where(~ho_mask)[0]
    ho_idx  = np.where( ho_mask)[0]

    n_ho_genes = len(holdout_genes)
    logger.info(
        "unseen_gene_holdout_split (seed=%d, holdout_frac=%.2f): "
        "train=%d rows (%d genes), holdout=%d rows (%d genes), "
        "actual_gene_holdout_frac=%.3f",
        seed, holdout_frac,
        len(tr_idx), len(train_genes),
        len(ho_idx), n_ho_genes,
        n_ho_genes / max(n_genes, 1),
    )
    return tr_idx, ho_idx


def split_summary(
    df:     pd.DataFrame,
    splits: Dict[str, np.ndarray],
    gene_col:  str = "gene_symbol",
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Produce a summary DataFrame for a set of named index arrays.

    Parameters
    ----------
    df        : The full DataFrame from which the splits were drawn.
    splits    : Dict mapping split name to integer position array.
    gene_col  : Column for gene-count summary (optional; silently skipped
                if absent).
    label_col : Column for prevalence summary (optional; silently skipped
                if absent).

    Returns
    -------
    DataFrame with columns: ``split``, ``n_variants``
    and optionally ``n_genes``, ``prevalence``.
    """
    rows = []
    for name, idx in splits.items():
        sub  = df.iloc[idx]
        row: dict = {
            "split":      name,
            "n_variants": int(len(sub)),
        }
        if gene_col in df.columns:
            row["n_genes"] = int(sub[gene_col].nunique())
        if label_col is not None and label_col in df.columns:
            row["prevalence"] = (
                float(sub[label_col].mean()) if len(sub) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def time_disjoint_split(
    df:          pd.DataFrame,
    cutoff_year: int = 2023,
    date_col:    str = "last_evaluated",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split variants by ClinVar LastEvaluated date.

    .. note::
        **PHASE_3_FEATURES** — requires a ClinVar re-pull that populates
        *date_col*.  See RUN_15_PLAN.md item TDS-1.
    """
    raise NotImplementedError(
        "time_disjoint_split requires a ClinVar re-pull with the "
        f"'{date_col}' column populated.  "
        "See RUN_15_PLAN.md PHASE_3_FEATURES item TDS-1."
    )
