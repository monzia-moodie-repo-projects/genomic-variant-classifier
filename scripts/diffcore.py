"""Core comparison logic for diff_cohorts (developed + tested standalone before packaging).

Handles the two subtleties the real schema demands:
  - dict-typed 'metadata' and large 'fasta_seq' columns: compared via element-wise equality that
    tolerates unhashable cells (NEVER a hash that assumes hashable scalars).
  - alleles compared via a NORMALIZED empty-token view so None / nan / 'na' / '.' / '-' / '' all
    register as EQUAL (representation-only differences never count as changes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Empty-allele tokens (mirrors the builder's is_empty_allele semantics).
_EMPTY_TOKENS = {"", "none", "nan", ".", "-", "na", "null"}


def normalize_allele(series: pd.Series) -> pd.Series:
    """Map every empty representation to a single canonical token '<EMPTY>'; otherwise the
    stripped string. So None, nan, 'na', '.', '-', '' all become '<EMPTY>' and compare EQUAL."""
    s = series.astype("object")
    out = []
    for v in s:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("<EMPTY>")
            continue
        t = str(v).strip()
        out.append("<EMPTY>" if t.lower() in _EMPTY_TOKENS else t)
    return pd.Series(out, index=series.index, dtype="object")


def cells_equal(a, b) -> bool:
    """Element equality tolerant of dicts, arrays, nan, None. Used for dict/object columns."""
    # nan == nan should be True for our purposes (unchanged)
    a_nan = isinstance(a, float) and np.isnan(a)
    b_nan = isinstance(b, float) and np.isnan(b)
    if a_nan and b_nan:
        return True
    if a_nan != b_nan:
        return False
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    if isinstance(a, dict) or isinstance(b, dict):
        return a == b
    if isinstance(a, (np.ndarray, list)) or isinstance(b, (np.ndarray, list)):
        try:
            return list(a) == list(b)
        except Exception:
            return bool(a == b)
    return bool(a == b)


def column_equal_series(sa: pd.Series, sb: pd.Series, allele: bool = False) -> pd.Series:
    """Boolean Series (aligned index): True where the two columns are EQUAL on that row.
    allele=True applies normalized-allele comparison. Object/dict columns fall back to
    element-wise cells_equal; simple columns use a vectorized compare with nan-eq."""
    if allele:
        na, nb = normalize_allele(sa), normalize_allele(sb)
        return na.values == nb.values
    # try vectorized for numeric/string; detect dict/object by sampling
    is_objecty = sa.dtype == object or sb.dtype == object
    if is_objecty:
        return pd.Series([cells_equal(x, y) for x, y in zip(sa.values, sb.values)],
                         index=sa.index).values
    # numeric/bool: nan==nan should count equal
    a = sa.values
    b = sb.values
    eq = (a == b)
    both_nan = pd.isna(a) & pd.isna(b)
    return eq | both_nan


def transition_matrix(old: pd.Series, new: pd.Series, classes: list[str]) -> pd.DataFrame:
    """(len(classes) x len(classes)) counts of old_label -> new_label over aligned rows."""
    idx = pd.Categorical(old.astype(str), categories=classes)
    col = pd.Categorical(new.astype(str), categories=classes)
    m = pd.crosstab(idx, col, dropna=False)
    # ensure full square even if a class is absent
    m = m.reindex(index=classes, columns=classes, fill_value=0)
    m.index.name = "old"
    m.columns.name = "new"
    return m
