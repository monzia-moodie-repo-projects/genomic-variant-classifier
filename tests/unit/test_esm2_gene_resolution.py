"""Regression tests for ESM-2 gene-symbol resolution hardening (Phase 0).

Exercises the candidate-based lookup added to ``_get_sequence`` and the
``_missing_genes`` accumulator -- all backend-independent (no transformers/torch
needed), driven through a tiny on-disk UniProt index via ``uniprot_index_path``.

Guards: semicolon-joined multi-gene symbols resolve to the first present
component; hyphenated symbols are NEVER split; case-insensitive lookups are
preserved; and definitive misses accumulate (normalized) for the aggregate log.
"""

from __future__ import annotations

import pandas as pd
import pytest

from genomic_variant_classifier.data import esm2 as E
from genomic_variant_classifier.data.esm2 import ESM2Connector


@pytest.fixture(autouse=True)
def _clear_index_cache():
    # Mirror test_esm2_uniprot_index.py: avoid cross-test index-cache bleed.
    cache = getattr(E, "_UNIPROT_INDEX_CACHE", None)
    if cache is not None:
        cache.clear()


@pytest.fixture()
def index_parquet(tmp_path):
    p = tmp_path / "uniprot_human_reviewed.parquet"
    pd.DataFrame(
        {
            "gene_symbol": ["BRCA1", "TP53", "ECE2"],
            "uniprot_id": ["P38398", "P04637", "O60344"],
            "sequence": ["MENWALK" * 4, "MEEPQSDPSV" * 3, "MQRLLL" * 5],
        }
    ).to_parquet(p)
    return p


def _conn(tmp_path, index_parquet):
    return ESM2Connector(
        cache_path=tmp_path / "c.sqlite",
        uniprot_index_path=index_parquet,
    )


def test_exact_and_case_insensitive_preserved(tmp_path, index_parquet):
    c = _conn(tmp_path, index_parquet)
    assert c.allow_network is False
    assert c._get_sequence("BRCA1").startswith("MENWALK")
    assert c._get_sequence("brca1").startswith("MENWALK")   # case-insensitive
    assert c._get_sequence("  TP53 ").startswith("MEEPQSDPSV")  # strip


def test_semicolon_join_resolves_to_first_present_component(tmp_path, index_parquet):
    c = _conn(tmp_path, index_parquet)
    # "MYH11;NDE1"-style: neither joined; the present component wins.
    assert c._get_sequence("ZZZ;BRCA1").startswith("MENWALK")
    assert c._get_sequence("ECE2;EEF1AKMT4-ECE2").startswith("MQRLLL")  # 'ECE2' wins; '-' untouched
    assert not c._missing_genes  # everything above resolved


def test_hyphen_is_never_split_and_miss_is_recorded(tmp_path, index_parquet):
    c = _conn(tmp_path, index_parquet)
    assert c._get_sequence("HLA-A") is None        # not in index, '-' not split
    assert c._get_sequence("NOSUCH") is None
    assert c._get_sequence("ZZZ;QQQ") is None       # no component present
    assert c._missing_genes == {"HLA-A", "NOSUCH", "ZZZ;QQQ"}


def test_missing_genes_starts_empty(tmp_path, index_parquet):
    c = _conn(tmp_path, index_parquet)
    assert c._missing_genes == set()
