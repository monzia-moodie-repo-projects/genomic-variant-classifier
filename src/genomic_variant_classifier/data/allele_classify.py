"""
allele_classify.py  (2026-07-09)
==========================================================================
Single canonical source of truth for allele-shape classification. Imported by the
cohort builder, the seq-window rekey, and the coverage diagnostics so every path uses
IDENTICAL logic -- no per-file re-implementations that can drift.

WHY THIS MODULE EXISTS (bug found 2026-07-09)
    The previous inline `is_padded_deletion` did `alt.astype("string").fillna("")`, turning
    a NaN/empty alt into "". Since `ref.startswith("")` is ALWAYS True, ANY row with an
    empty/NaN alt was misclassified as a padded deletion. Two diagnostics that used
    `astype(str)` (NaN -> "nan", not a prefix) were accidentally immune, producing a
    0-vs-5 disagreement that took a full investigation to resolve. The fix is an explicit
    non-empty guard, made canonical here so it cannot diverge again.

    Measured blast radius on cohort-v2 (clinvar_grch38_clean_v2_verified.parquet, 2026-07-09):
      * rows the buggy classifier wrongly shifted (pos-=1): 0  (cohort-v2 uncorrupted)
      * the "5" that tripped the rekey-verify were malformed seq-parquet rows, not cohort rows.
    So this is latent-hazard hygiene, not a data correction -- but it MUST be fixed so a
    future empty-allele row in a coordinate-shifting path can never be silently mis-shifted.
"""

from __future__ import annotations

import pandas as pd

_NULL_TOKENS = frozenset({"", "na", "nan", "none", "."})


def _norm(s: pd.Series) -> pd.Series:
    """Lower, stripped string view with NaN -> '' (for emptiness tests only)."""
    return s.astype("string").fillna("").str.strip()


def is_empty_allele(s: pd.Series) -> pd.Series:
    """True where the allele is null/empty or a null-token ('na','nan','none','.',"")."""
    n = _norm(s).str.lower()
    return n.isin(_NULL_TOKENS)


def is_allele_less(ref: pd.Series, alt: pd.Series) -> pd.Series:
    """True where BOTH ref and alt are empty/null (the na:na malformed rows)."""
    return is_empty_allele(ref) & is_empty_allele(alt)


def _startswith_elementwise(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    return pd.Series([rr.startswith(aa) for rr, aa in zip(r, a)], index=ref.index, dtype=bool)


def is_padded_deletion(ref: pd.Series, alt: pd.Series) -> pd.Series:
    """A padded deletion: a NON-EMPTY alt that is a strict prefix of a strictly longer,
    NON-EMPTY ref (e.g. ref='ACTT', alt='A'). The non-empty guards on BOTH sides are the
    fix for the 2026-07-09 bug: without them a NaN/empty alt becomes '' and matches every
    ref as a prefix.
    """
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    non_empty = (a.str.len() >= 1) & (r.str.len() >= 1)
    return non_empty & (a.str.len() < r.str.len()) & _startswith_elementwise(r, a)
