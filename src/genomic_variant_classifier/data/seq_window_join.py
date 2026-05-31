"""Attach ref/alt delta sequence windows to a split's rows, alignment-safe.

The sequence CNN needs a 2-column [fasta_seq_ref, fasta_seq_alt] frame whose
row i corresponds to feature-matrix row i. Splits are produced by
DataPrepPipeline as positionally-aligned (X_*, meta_*) pairs, so we attach
windows to `meta` and rely on key identity -- never on row order between the
cohort parquet and the split.

Resolution order:
  1. `meta` already carries fasta_seq_ref + fasta_seq_alt  -> use directly.
  2. `seq_windows_path` given -> key-join on chrom:pos:ref:alt (order-preserving).
  3. otherwise -> poly-A for every row (no signal), with full unmapped count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REF_WIN_COL = "fasta_seq_ref"
ALT_WIN_COL = "fasta_seq_alt"
_KEY_COLS = ("chrom", "pos", "ref", "alt")


def _make_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["chrom"].astype(str) + ":" + df["pos"].astype(str)
        + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)
    )


def attach_delta_windows(meta: pd.DataFrame, seq_windows_path=None, window: int = 101):
    """Return (windows_df, n_unmapped).

    windows_df is a 2-column DataFrame [fasta_seq_ref, fasta_seq_alt] whose rows
    are aligned 1:1 to `meta` (reset index). n_unmapped is the number of rows
    that fell back to poly-A because their key was absent from the window source.
    """
    poly = "A" * window
    n = len(meta)

    # 1. windows already present on the rows -> structurally aligned, no join.
    if REF_WIN_COL in meta.columns and ALT_WIN_COL in meta.columns:
        out = pd.DataFrame(
            {
                REF_WIN_COL: meta[REF_WIN_COL].fillna(poly).astype(str).to_numpy(),
                ALT_WIN_COL: meta[ALT_WIN_COL].fillna(poly).astype(str).to_numpy(),
            }
        )
        return out, 0

    # 2. key-join from a windows parquet (order-preserving via .map).
    if seq_windows_path is not None:
        seq = pd.read_parquet(
            seq_windows_path,
            columns=[*_KEY_COLS, REF_WIN_COL, ALT_WIN_COL],
        )
        seq = seq.assign(_key=_make_key(seq)).drop_duplicates("_key")  # window = f(key)
        ref_map = seq.set_index("_key")[REF_WIN_COL]
        alt_map = seq.set_index("_key")[ALT_WIN_COL]
        mkey = _make_key(meta)
        r = mkey.map(ref_map)
        a = mkey.map(alt_map)
        n_unmapped = int(r.isna().sum())
        out = pd.DataFrame(
            {
                REF_WIN_COL: r.fillna(poly).astype(str).to_numpy(),
                ALT_WIN_COL: a.fillna(poly).astype(str).to_numpy(),
            }
        )
        return out, n_unmapped

    # 3. no source -> poly-A (no sequence signal).
    out = pd.DataFrame({REF_WIN_COL: [poly] * n, ALT_WIN_COL: [poly] * n})
    return out, n
