"""test_agent_ops_detector.py -- Monzia Moodie
Hermetic tests for the schema-agnostic ops detector: heartbeat staleness + no-timestamp, inbox backlog,
unresolved reviews, surfaced flags, and the OK/ATTENTION roll-up. now is injected for determinism.
"""
from datetime import datetime, timedelta, timezone

from genomic_variant_classifier.evaluation import agent_ops_detector as D

NOW = datetime(2026, 6, 14, 18, 0, 0, tzinfo=timezone.utc)


def _state(**over):
    base = {
        "training": {"last_run": (NOW - timedelta(hours=2)).isoformat()},          # fresh
        "database_freshness": {"clinvar": {"last_checked": (NOW - timedelta(days=40)).isoformat()}},  # stale
        "literature": {"last_run": None, "feature_candidates": []},                 # no timestamp
        "review_items": [],
        "agent_messages": {},
    }
    base.update(over)
    return base


def test_heartbeats_fresh_stale_and_missing():
    beats = {b.section: b for b in D.scan_heartbeats(_state(), NOW)}
    assert beats["training"].stale is False and beats["training"].age_hours == 2.0
    assert beats["database_freshness"].stale is True                                # 40d >= ~35d threshold
    assert beats["literature"].newest_iso is None and "no timestamp" in beats["literature"].detail
    assert "review_items" not in beats and "agent_messages" not in beats            # non-heartbeat sections skipped


def test_inbox_backlog():
    st = _state(agent_messages={
        "TrainingLifecycleAgent": [
            {"read": False, "requires_approval": True, "approved": None},
            {"read": True, "requires_approval": False, "approved": None},
        ],
        "QuietAgent": [{"read": True, "requires_approval": False, "approved": None}],   # no backlog -> omitted
    })
    inbox = {s.agent: s for s in D.scan_inbox(st)}
    assert inbox["TrainingLifecycleAgent"].unread == 1 and inbox["TrainingLifecycleAgent"].pending_approval == 1
    assert "QuietAgent" not in inbox


def test_unresolved_reviews_and_flags():
    st = _state(review_items=[{"resolved": False}, {"resolved": True}],
                data_readiness={"verdict": "NO_GO"},
                interpretability={"instability_flags": ["x", "y"]},
                model_insights={"flags": ["LEAKAGE_SUSPICION[m]"]})
    assert D.unresolved_reviews(st) == 1
    flags = D.scan_flags(st)
    assert any("NO_GO" in f for f in flags) and any("instability" in f for f in flags) and any("model_insights" in f for f in flags)


def test_analyze_ok_vs_attention():
    # all fresh, no backlog/flags -> OK
    ok = {"training": {"last_run": (NOW - timedelta(hours=1)).isoformat()}, "review_items": [], "agent_messages": {}}
    assert D.analyze(ok, now=NOW)["ops_status"] == "OK"
    # a stale section -> ATTENTION
    assert D.analyze(_state(), now=NOW)["ops_status"] == "ATTENTION"
    # a pending approval alone -> ATTENTION
    pend = {"training": {"last_run": NOW.isoformat()}, "review_items": [],
            "agent_messages": {"A": [{"read": True, "requires_approval": True, "approved": None}]}}
    assert D.analyze(pend, now=NOW)["ops_status"] == "ATTENTION"


def test_scan_run_telemetry_error_rate_and_drift():
    state = {"agent_runs": {
        "Flaky": [{"status": "ok", "duration_ms": 10}, {"status": "error", "duration_ms": 12},
                  {"status": "ok", "duration_ms": 11}, {"status": "error", "duration_ms": 13}],
        "Slowing": [{"status": "ok", "duration_ms": d} for d in [10, 10, 11, 30, 32, 31]],
    }}
    t = {x.agent: x for x in D.scan_run_telemetry(state)}
    assert t["Flaky"].n_runs == 4 and t["Flaky"].n_errors == 2 and t["Flaky"].error_rate == 0.5
    assert t["Slowing"].drift_pct is not None and t["Slowing"].drift_pct >= D.PERF_DRIFT_PCT   # +210%


def test_telemetry_flags_error_and_drift():
    state = {"agent_runs": {"Flaky": [{"status": "error", "duration_ms": 5}, {"status": "ok", "duration_ms": 5}]}}
    flags = D.telemetry_flags(D.scan_run_telemetry(state))
    assert any(f.startswith("AGENT_ERRORS[Flaky]") for f in flags)


def test_analyze_attention_on_agent_error():
    fresh = (NOW - timedelta(hours=1)).isoformat()
    state = {"training": {"last_run": fresh}, "review_items": [], "agent_messages": {},
             "agent_runs": {"X": [{"status": "error", "duration_ms": 5}]}}
    res = D.analyze(state, now=NOW)
    assert res["ops_status"] == "ATTENTION"
    assert any("AGENT_ERRORS" in f for f in res["flags"]) and res["telemetry"]
