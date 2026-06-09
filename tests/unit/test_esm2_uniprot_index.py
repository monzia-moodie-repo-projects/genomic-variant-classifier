"""
tests/unit/test_esm2_uniprot_index.py
=====================================
Regression tests for the ESM-2 connector hardening (Run-15 smoke stall,
instance 40187155): sequences served from a local UniProt index with NO
run-time REST call, fail-loud on a missing gene, and GPU device auto-detect.
"""
from __future__ import annotations

import pandas as pd
import pytest

from genomic_variant_classifier.data import esm2 as E
from genomic_variant_classifier.data.esm2 import ESM2Connector


@pytest.fixture(autouse=True)
def _clear_index_cache():
    E._UNIPROT_INDEX_CACHE.clear()


@pytest.fixture()
def index_parquet(tmp_path):
    p = tmp_path / "uniprot_human_reviewed.parquet"
    pd.DataFrame(
        {
            "gene_symbol": ["BRCA1", "TP53", "tp53"],   # dup TP53 (first wins), lowercase
            "uniprot_id": ["P38398", "P04637", "X1"],
            "sequence": ["MENWALK" * 10, "MEEPQSDPSV" * 5, "WRONGSEQ" * 5],
        }
    ).to_parquet(p)
    return p


def test_local_index_serves_sequences_with_no_network(tmp_path, index_parquet, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("live UniProt fetch must NOT happen when a local index is set")

    monkeypatch.setattr(E, "_fetch_uniprot_sequence", _boom)
    c = ESM2Connector(cache_path=tmp_path / "c.sqlite", uniprot_index_path=index_parquet)
    assert c.allow_network is False                       # index => network off by default
    assert c._get_sequence("BRCA1").startswith("MENWALK")
    assert c._get_sequence("tp53").startswith("MEEPQSDPSV")   # case-insensitive
    assert c._get_sequence("TP53").startswith("MEEPQSDPSV")   # dup -> canonical first row


def test_missing_gene_fails_loud_not_network(tmp_path, index_parquet, monkeypatch):
    monkeypatch.setattr(
        E, "_fetch_uniprot_sequence",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network allowed")),
    )
    c = ESM2Connector(cache_path=tmp_path / "c.sqlite", uniprot_index_path=index_parquet)
    assert c._get_sequence("NOSUCHGENE") is None          # 0.0 downstream, no stall


def test_legacy_network_path_when_no_index(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _fake(gene, timeout=10):
        calls["n"] += 1
        return ("P1", "MAAAA")

    monkeypatch.setattr(E, "_fetch_uniprot_sequence", _fake)
    c = ESM2Connector(cache_path=tmp_path / "c.sqlite")   # no index -> legacy behaviour
    assert c.allow_network is True
    assert c._get_sequence("G") == "MAAAA"
    assert c._get_sequence("G") == "MAAAA"                # 2nd served from sqlite cache
    assert calls["n"] == 1


def test_device_auto_detect(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "_cuda_available", lambda: False)
    assert ESM2Connector(cache_path=tmp_path / "a.sqlite").device == "cpu"
    monkeypatch.setattr(E, "_cuda_available", lambda: True)
    assert ESM2Connector(cache_path=tmp_path / "b.sqlite").device == "cuda"
    assert ESM2Connector(cache_path=tmp_path / "c.sqlite", device="cpu").device == "cpu"
