#!/usr/bin/env python3
"""test_reclassification_reference.py -- ReclassificationSentinel builder + monitor (Monzia Moodie).

Builder: build_reclassification_reference.build_reference extracts (variant_id, split) per split and
SKIPS missing splits (never mislabels, cf. the legacy run_drift_monitor meta_test->training bug).
Monitor: from_default_baseline active/awaiting/env. ASCII-only.
"""
import importlib.util
import os
import pandas as pd
import pytest
from pathlib import Path
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.reclassification_sentinel_monitor_agent import (
    ReclassificationSentinelMonitorAgent,
)

_BUILDER = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_reclassification_reference.py")
_spec = importlib.util.spec_from_file_location("build_reclassification_reference", _BUILDER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_reference = _mod.build_reference

_BASE = dict(gene_symbol="GENE", chrom="1", ref="A", alt="T",
             review_status="criteria provided, single submitter")


def _state(tmp_path):
    return SharedState(state_file=str(tmp_path / "st.json"))


def _make_splits(tmp_path, *, train=1000, val=2, test=1):
    sd = tmp_path / "splits"; sd.mkdir(exist_ok=True)
    if train:
        pd.DataFrame({"variant_id": [f"tr_{i}" for i in range(train)]}).to_parquet(sd / "meta_train.parquet")
    if val:
        pd.DataFrame({"variant_id": [f"v_{i}" for i in range(val)]}).to_parquet(sd / "meta_val.parquet")
    if test:
        pd.DataFrame({"variant_id": [f"t_{i}" for i in range(test)]}).to_parquet(sd / "meta_test.parquet")
    return sd


def _reference(tmp_path):
    ref, _ = build_reference(_make_splits(tmp_path))
    rp = tmp_path / "reclassification_reference.parquet"; ref.to_parquet(rp, index=False)
    return rp


def _clinvar_pair(tmp_path, n_flip, old_sig="Uncertain significance", new_sig="Likely pathogenic"):
    o, n = [], []
    for i in range(1000):
        row = dict(variant_id=f"tr_{i}", pos=1000 + i, **_BASE)
        o.append({**row, "clinical_sig": old_sig})
        n.append({**row, "clinical_sig": new_sig if i < n_flip else old_sig})
    op, npth = tmp_path / "old.parquet", tmp_path / "new.parquet"
    pd.DataFrame(o).to_parquet(op); pd.DataFrame(n).to_parquet(npth)
    return op, npth


def test_builder_per_split(tmp_path):
    ref, skipped = build_reference(_make_splits(tmp_path, train=1000, val=2, test=1))
    assert ref.groupby("split").size().to_dict() == {"train": 1000, "val": 2, "test": 1}
    assert skipped == []


def test_builder_skips_missing_without_mislabel(tmp_path):
    # only meta_test present -> train/val SKIPPED, test NOT relabeled as train (the legacy bug)
    ref, skipped = build_reference(_make_splits(tmp_path, train=0, val=0, test=1))
    counts = ref.groupby("split").size().to_dict()
    assert counts == {"test": 1} and "train" not in counts
    assert {s for s, _ in skipped} == {"train", "val"}


def test_builder_variant_id_col_absent(tmp_path):
    sd = tmp_path / "splits"; sd.mkdir()
    pd.DataFrame({"vid": ["x"]}).to_parquet(sd / "meta_train.parquet")  # wrong column
    pd.DataFrame({"variant_id": ["t_0"]}).to_parquet(sd / "meta_test.parquet")
    ref, skipped = build_reference(sd)
    assert "train" in {s for s, _ in skipped} and ref.groupby("split").size().to_dict() == {"test": 1}


def test_monitor_active_red(tmp_path):
    op, npth = _clinvar_pair(tmp_path, 12)  # 1.2% -> retrain
    m = ReclassificationSentinelMonitorAgent.from_default_baseline(
        _state(tmp_path), old_path=str(op), new_path=str(npth), reference_path=_reference(tmp_path),
        output_dir=tmp_path, new_release="2024_07")
    r = m.run(dry_run=False)
    assert r["status"] == "ok" and r["severity"] == "red" and r["urgency"] == "retrain"
    assert r["new_release"] == "2024_07" and r["flip_rate_training"] == pytest.approx(0.012)


def test_monitor_awaiting(tmp_path):
    op, npth = _clinvar_pair(tmp_path, 12); rp = _reference(tmp_path)
    # no new_path
    assert ReclassificationSentinelMonitorAgent.from_default_baseline(
        _state(tmp_path), old_path=str(op), reference_path=rp, output_dir=tmp_path
    ).run(dry_run=False)["status"] == "awaiting_baseline"
    # set-but-missing file
    assert ReclassificationSentinelMonitorAgent.from_default_baseline(
        _state(tmp_path), old_path=str(op), new_path=str(tmp_path / "nope.parquet"),
        reference_path=rp, output_dir=tmp_path).run(dry_run=False)["status"] == "awaiting_baseline"
    # missing reference -> inactive
    assert ReclassificationSentinelMonitorAgent.from_default_baseline(
        _state(tmp_path), old_path=str(op), new_path=str(npth),
        reference_path=tmp_path / "nope.parquet", output_dir=tmp_path).run(dry_run=False)["status"] == "awaiting_baseline"


def test_monitor_env_resolution(tmp_path, monkeypatch):
    op, npth = _clinvar_pair(tmp_path, 12)
    monkeypatch.setenv("GVC_RECLASS_OLD_RELEASE", str(op))
    monkeypatch.setenv("GVC_RECLASS_NEW_RELEASE", str(npth))
    monkeypatch.setenv("GVC_RECLASS_NEW_LABEL", "2024_07")
    r = ReclassificationSentinelMonitorAgent.from_default_baseline(
        _state(tmp_path), reference_path=_reference(tmp_path), output_dir=tmp_path).run(dry_run=False)
    assert r["status"] == "ok" and r["severity"] == "red" and r["new_release"] == "2024_07"
