"""test_finops_detector.py -- Monzia Moodie
Pure FinOps detector: selection, cost, budget verdicts, snapshot parsing, emit-only (NO vastai/subprocess), and a
PARITY check that the canonical pick_offer matches the legacy launch_run16.pick_offer so they cannot drift.
"""
import json
import sys
from pathlib import Path

from genomic_variant_classifier.evaluation import finops_detector as D

OFFERS = [
    {"id": 1, "dph_total": 0.80, "reliability2": 0.995, "cpu_ram": 128, "num_gpus": 1},
    {"id": 2, "dph_total": 0.40, "reliability2": 0.991, "cpu_ram": 64, "num_gpus": 1},   # cheapest 1-GPU
    {"id": 3, "dph_total": 0.40, "reliability2": 0.999, "cpu_ram": 64, "num_gpus": 1},   # tie price, higher rel
    {"id": 9, "dph_total": 0.20, "reliability2": 0.999, "cpu_ram": 256, "num_gpus": 2},  # 2-GPU -> excluded
]


def test_pick_offer_cheapest_single_gpu_then_reliability():
    chosen = D.pick_offer(OFFERS)
    assert chosen["id"] == 3                                  # 0.40 tie -> higher reliability wins; 2-GPU excluded


def test_pick_offer_empty_and_all_multi_gpu():
    assert D.pick_offer([]) is None
    assert D.pick_offer([{"id": 9, "dph_total": 0.2, "num_gpus": 2}]) is None


def test_estimate_cost():
    assert D.estimate_cost(15, 0.40) == 6.0
    assert D.estimate_cost(10, 0.473) == 4.73


def test_recommend_within_and_over_budget():
    within = D.recommend(OFFERS, est_hours=15, budget_usd=15)
    assert within["verdict"] == D.WITHIN_BUDGET and within["chosen_id"] == 3 and within["est_cost"] == 6.0
    over = D.recommend(OFFERS, est_hours=15, budget_usd=5)   # 6.0 > 5 -> over
    assert over["verdict"] == D.OVER_BUDGET
    none = D.recommend([{"id": 9, "dph_total": 0.2, "num_gpus": 2}])
    assert none["verdict"] == D.NO_SUITABLE_OFFER and none["command"] is None


def test_command_is_emit_only_preview_string():
    rec = D.recommend(OFFERS)
    # advisory preview only -- never a bare create/destroy, always the --dry-run preview
    assert "--dry-run" in rec["command"] and "vastai create" not in rec["command"]


def test_load_offers_snapshot_list_and_wrapped(tmp_path):
    p1 = tmp_path / "a.json"; p1.write_text(json.dumps(OFFERS))
    assert len(D.load_offers_snapshot(p1)) == 4
    p2 = tmp_path / "b.json"; p2.write_text(json.dumps({"offers": OFFERS}))
    assert len(D.load_offers_snapshot(p2)) == 4
    p3 = tmp_path / "c.json"; p3.write_text("null")
    assert D.load_offers_snapshot(p3) == []


def test_pick_offer_parity_with_launch_script():
    # the legacy copy in scripts/launch_run16.py must agree -> no silent drift
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import launch_run16
    assert launch_run16.pick_offer(OFFERS) == D.pick_offer(OFFERS)
    assert launch_run16.pick_offer([]) == D.pick_offer([])


def test_load_offers_snapshot_tolerates_bom_encodings(tmp_path):
    # PowerShell 5.1 `> offers.json` writes UTF-16-LE-with-BOM; Out-File utf8 writes a UTF-8 BOM. Both must load.
    import json as _json
    for enc, name in [("utf-16", "ps_redirect.json"), ("utf-8-sig", "outfile_utf8.json"), ("utf-8", "plain.json")]:
        p = tmp_path / name
        p.write_bytes(_json.dumps(OFFERS).encode(enc))
        loaded = D.load_offers_snapshot(p)
        assert len(loaded) == 4, f"{name} ({enc}) failed to load"
    # and the recommendation still works off a UTF-16 snapshot
    p16 = tmp_path / "u16.json"; p16.write_bytes(_json.dumps(OFFERS).encode("utf-16"))
    assert D.recommend(D.load_offers_snapshot(p16))["chosen_id"] == 3
