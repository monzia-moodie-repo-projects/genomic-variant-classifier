"""Wiring tests for the scripts/train.py ESM-2 flags.

Guards the path that the LLR/activation work depends on: a regen must be able to
select the 650M model and an offline UniProt index, instead of silently using the
8M default with live per-gene REST. Test 1 pins the AnnotationConfig contract the
flags rely on; test 2 proves the CLI flags parse and carry the right values.
Author: Monzia Moodie.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAIN_PY = REPO / "scripts" / "train.py"


def _load_train(monkeypatch, tmp_path):
    """Import scripts/train.py as a module without running main(). chdir keeps
    its module-level logging side effects (logs/train.log) inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("_train_under_test", TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_train_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_annotation_config_esm2_contract():
    """The four esm2 fields train.py passes must exist with the documented
    defaults, and must accept the production override values."""
    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig

    cfg = AnnotationConfig()
    assert cfg.esm2_model_name == "esm2_t6_8M_UR50D"
    assert cfg.esm2_uniprot_index_path is None
    assert cfg.esm2_cache_path is None
    assert cfg.esm2_device is None

    cfg2 = AnnotationConfig(
        esm2_model_name="esm2_t33_650M_UR50D",
        esm2_uniprot_index_path=Path("idx.parquet"),
        esm2_cache_path=Path("c.sqlite"),
        esm2_device="cuda",
    )
    assert cfg2.esm2_model_name == "esm2_t33_650M_UR50D"
    assert cfg2.esm2_uniprot_index_path == Path("idx.parquet")
    assert cfg2.esm2_cache_path == Path("c.sqlite")
    assert cfg2.esm2_device == "cuda"


def test_train_esm2_flags_parse(monkeypatch, tmp_path):
    """--esm2-* flags parse to the right namespace attrs (production + defaults)."""
    mod = _load_train(monkeypatch, tmp_path)

    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--esm2-model", "esm2_t33_650M_UR50D",
        "--esm2-uniprot-index", "data/external/uniprot/uniprot_human_reviewed.parquet",
        "--esm2-cache", "cache.sqlite",
        "--esm2-device", "cuda",
    ])
    a = mod.parse_args()
    assert a.esm2_model == "esm2_t33_650M_UR50D"
    assert a.esm2_uniprot_index == "data/external/uniprot/uniprot_human_reviewed.parquet"
    assert a.esm2_cache == "cache.sqlite"
    assert a.esm2_device == "cuda"

    monkeypatch.setattr(sys, "argv", ["train.py"])
    d = mod.parse_args()
    assert d.esm2_model == "esm2_t6_8M_UR50D"   # 8M default -> smoke tests stay fast
    assert d.esm2_uniprot_index is None
    assert d.esm2_cache is None
    assert d.esm2_device is None
