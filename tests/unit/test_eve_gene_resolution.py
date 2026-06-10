"""Regression tests for EVE gene-symbol resolution hardening (Phase 0).

EVEConnector._annotate uses no instance state, so it is exercised directly with
a dummy ``self`` -- no eve_path/CSV fixtures or network needed. Guards the
case-drift fix (variant-side gene_symbol was never upper-cased) and the
semicolon-join recovery, while confirming '-' is never split and genuine misses
fall through to the default score.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from genomic_variant_classifier.data.eve import (
    DEFAULT_SCORE,
    EVEConnector,
    _hgvsp_to_eve_key,
)


def _annotate(variant_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    # _annotate references no self.* attributes; a bare namespace is sufficient.
    return EVEConnector._annotate(SimpleNamespace(), variant_df, lookup)


def test_gene_key_resolution_paths():
    key = _hgvsp_to_eve_key("p.Arg175His")
    assert key, "aa-change parser produced no key; cannot isolate gene resolution"
    lookup = pd.DataFrame(
        {"gene_symbol": ["BRCA1", "TP53"], "aa_change": [key, key], "eve_score": [0.9, 0.8]}
    )
    vdf = pd.DataFrame(
        {
            "gene_symbol": ["BRCA1", "brca1", "ZZZ;BRCA1", "NOSUCH", "HLA-A"],
            "protein_change": ["p.Arg175His"] * 5,
        }
    )
    s = _annotate(vdf, lookup)["eve_score"].tolist()
    assert s[0] == 0.9                # exact match
    assert s[1] == 0.9                # case-drift FIXED (latent bug)
    assert s[2] == 0.9                # ';'-join resolves to BRCA1
    assert s[3] == DEFAULT_SCORE      # genuine miss -> default
    assert s[4] == DEFAULT_SCORE      # 'HLA-A' not split, not present -> default


def test_empty_lookup_yields_all_default():
    empty = pd.DataFrame({"gene_symbol": [], "aa_change": [], "eve_score": []})
    vdf = pd.DataFrame({"gene_symbol": ["BRCA1"], "protein_change": ["p.Arg175His"]})
    s = _annotate(vdf, empty)["eve_score"].tolist()
    assert s == [DEFAULT_SCORE]


def test_no_spurious_join_on_empty_gene_symbol():
    # An empty variant gene_symbol must not match an (accidentally) empty lookup key.
    key = _hgvsp_to_eve_key("p.Arg175His")
    lookup = pd.DataFrame({"gene_symbol": [""], "aa_change": [key], "eve_score": [0.95]})
    vdf = pd.DataFrame({"gene_symbol": [""], "protein_change": ["p.Arg175His"]})
    s = _annotate(vdf, lookup)["eve_score"].tolist()
    assert s == [DEFAULT_SCORE]
