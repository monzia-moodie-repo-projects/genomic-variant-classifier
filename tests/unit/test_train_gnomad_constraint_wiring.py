"""Wiring test for the scripts/train.py --gnomad-constraint flag.

Guards the path that revives gene_constraint_oe (Run-15 #2 feature): without
gnomad_constraint_path the GnomADConstraintConnector runs in stub mode, loeuf is a
constant, and gene_constraint_oe silently deadzones. Mirrors test_train_esm2_wiring.py.
Author: Monzia Moodie.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest  # noqa: F401  (kept for parity / future fixtures)

REPO = Path(__file__).resolve().parents[2]
TRAIN_PY = REPO / "scripts" / "train.py"


def _load_train(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("_train_gc_under_test", TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_train_gc_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_annotation_config_has_gnomad_constraint_path():
    """AnnotationConfig must expose gnomad_constraint_path (default None) and accept
    a production override -- the field train.py now threads."""
    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig

    assert AnnotationConfig().gnomad_constraint_path is None
    cfg = AnnotationConfig(gnomad_constraint_path=Path("c.tsv.bgz"))
    assert cfg.gnomad_constraint_path == Path("c.tsv.bgz")


def test_train_gnomad_constraint_flag_parses(monkeypatch, tmp_path):
    """--gnomad-constraint parses to args.gnomad_constraint; default is None."""
    mod = _load_train(monkeypatch, tmp_path)

    tsv = "data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv.bgz"
    monkeypatch.setattr(sys, "argv", ["train.py", "--gnomad-constraint", tsv])
    assert mod.parse_args().gnomad_constraint == tsv

    monkeypatch.setattr(sys, "argv", ["train.py"])
    assert mod.parse_args().gnomad_constraint is None
