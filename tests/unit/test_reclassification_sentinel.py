#!/usr/bin/env python3
"""test_reclassification_sentinel.py -- ReclassificationSentinel detector (Monzia Moodie).

Validates the urgency -> severity mapping over the REAL ClinVarTracker thresholds
(FLIP_RATE_MONITOR=0.005, FLIP_RATE_RETRAIN=0.010, FLIP_RATE_URGENT=0.025, WEIGHTED_IMPACT_RETRAIN=0.015)
and the from_reference parquet round-trip. ASCII-only (direction_breakdown carries a unicode arrow at
runtime, so tests assert on counts, not the arrow string).
"""
import pandas as pd
import pytest
from pathlib import Path
from genomic_variant_classifier.agent_layer.agents.reclassification_sentinel_agent import (
    ReclassificationSentinelAgent,
)

N_TRAIN = 1000
TRAIN_IDS = frozenset(f"tr_{i}" for i in range(N_TRAIN))
_BASE = dict(gene_symbol="GENE", chrom="1", ref="A", alt="T",
             review_status="criteria provided, single submitter")


def _clinvar_pair(n_flip, old_sig="Uncertain significance", new_sig="Likely pathogenic", n_new=0):
    old_rows, new_rows = [], []
    for i in range(N_TRAIN):
        row = dict(variant_id=f"tr_{i}", pos=1000 + i, **_BASE)
        old_rows.append({**row, "clinical_sig": old_sig})
        new_rows.append({**row, "clinical_sig": (new_sig if i < n_flip else old_sig)})
    for j in range(n_new):
        new_rows.append(dict(variant_id=f"new_{j}", pos=900000 + j, clinical_sig="Benign", **_BASE))
    return pd.DataFrame(old_rows), pd.DataFrame(new_rows)


def _detect(tmp_path, n_flip, old_sig="Uncertain significance", new_sig="Likely pathogenic", n_new=0):
    o, n = _clinvar_pair(n_flip, old_sig, new_sig, n_new)
    op, npth = tmp_path / "old.parquet", tmp_path / "new.parquet"
    o.to_parquet(op); n.to_parquet(npth)
    ag = ReclassificationSentinelAgent(training_ids=TRAIN_IDS, output_dir=tmp_path)
    return ag.detect(op, npth, old_release="2024_01", new_release="2024_07")


def test_no_change_green(tmp_path):
    r = _detect(tmp_path, 0)
    assert r.severity == "green" and r.urgency == "none"
    assert r.n_reclassified_total == 0 and r.flip_rate_training == 0.0


def test_below_monitor_green(tmp_path):
    r = _detect(tmp_path, 2)  # 0.002 < 0.005
    assert r.severity == "green" and r.urgency == "none" and r.flip_rate_training == pytest.approx(0.002)


def test_monitor_amber(tmp_path):
    r = _detect(tmp_path, 7)  # 0.007 in [0.005, 0.010)
    assert r.severity == "amber" and r.urgency == "monitor"
    assert sum(r.direction_breakdown.values()) == 7 and len(r.direction_breakdown) == 1


def test_retrain_red(tmp_path):
    r = _detect(tmp_path, 12)  # 0.012 >= 0.010
    assert r.severity == "red" and r.urgency == "retrain" and r.should_retrain is True


def test_urgent_red(tmp_path):
    r = _detect(tmp_path, 30)  # 0.030 >= 0.025
    assert r.severity == "red" and r.urgency == "urgent" and r.should_retrain is True


def test_weighted_impact_escalates_red(tmp_path):
    # 6 Pathogenic->Benign (weight 3.0): flip_rate 0.006 (monitor band) but weighted_impact 0.018 >= 0.015
    r = _detect(tmp_path, 6, old_sig="Pathogenic", new_sig="Benign")
    assert r.flip_rate_training == pytest.approx(0.006)
    assert r.weighted_impact == pytest.approx(0.018)
    assert r.severity == "red" and r.urgency == "retrain"


def test_n_new_variants_surfaced(tmp_path):
    r = _detect(tmp_path, 0, n_new=25)
    assert r.n_new_variants == 25 and r.severity == "green"


def test_from_reference_roundtrip(tmp_path):
    ref = pd.DataFrame({"variant_id": [f"tr_{i}" for i in range(5)] + ["v_a", "v_b"],
                        "split": ["train"] * 5 + ["val", "test"]})
    rp = tmp_path / "ref.parquet"; ref.to_parquet(rp)
    ag = ReclassificationSentinelAgent.from_reference(rp, output_dir=tmp_path)
    assert len(ag.training_ids) == 5 and len(ag.val_ids) == 1 and len(ag.test_ids) == 1
    assert "tr_0" in ag.training_ids and "v_a" in ag.val_ids and "v_b" in ag.test_ids
