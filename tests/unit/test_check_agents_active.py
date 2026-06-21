"""test_check_agents_active.py -- Monzia Moodie
Agent-liveness gate: AST parse of the REAL orchestrator (repo-relative, utf-8),
cross-provider activity assessment, and portable inline fixtures (no sandbox paths).
"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
import check_agents_active as C  # noqa: E402

_ORCH = _REPO / "src" / "genomic_variant_classifier" / "agent_layer" / "orchestrator.py"
REAL_ORCH = _ORCH.read_text(encoding="utf-8")   # utf-8: orchestrator has box/emoji chars

# A flat literature_scout.* store (data/agent_state.json) is NOT a SharedState -> portable inline fixture.
FLAT_LIT_STORE = {
    "literature_scout.last_run": "2026-06-13T19:14:10.740728+00:00",
    "literature_scout.pykan_installed": "0.2.8",
    "literature_scout.clinvar_header_hash": "d41d8cd98f00b204e9800998ecf8427e",
}
NOW = datetime(2026, 6, 19, tzinfo=timezone.utc)

def test_parse_real_orchestrator_consistent_and_finops_recommend_only():
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    assert len(reg) >= 21
    assert "FinOpsAdvisorAgent" in reg
    assert pipes["finops"] == ["FinOpsAdvisorAgent"]      # FinOps stays recommend-only, alone
    assert len(pipes["drift"]) == 10 and len(pipes["full"]) == 9
    # Wiring-agnostic: if ProvisioningAgent has been wired in, it must be a SEPARATE
    # agent (not merged into finops) and scheduled in >=1 pipeline (not UNSCHEDULED).
    if "ProvisioningAgent" in reg:
        assert "ProvisioningAgent" not in pipes["finops"]
        assert "ProvisioningAgent" in C.pipeline_union(pipes)

def test_every_registered_agent_is_scheduled_today():
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    assert C.pipeline_union(pipes) == reg

def test_flat_store_is_not_a_sharedstate():
    assert C.looks_like_sharedstate(FLAT_LIT_STORE) is False
    assert C.looks_like_sharedstate({"training": {"last_run": "x"}}) is True

def test_flat_store_yields_all_never_run():
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    rows, problem = C.assess(reg, pipes, FLAT_LIT_STORE, NOW, 30.0)
    assert problem is True
    assert all(r["status"] == "NEVER_RUN" for r in rows)

def test_reproduce_real_sharedstate_verdict():
    # Mirror the REAL pre-real-run SharedState: drift sections dry-run-only,
    # literature/finops wrote sections (no dry_run flag), training/interp null,
    # and NO agent_runs telemetry anywhere.
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    state = {
        "literature": {"last_run": "2026-04-09T16:27:14.738040+00:00"},  # no dry_run flag
        "finops": {"last_run": "2026-06-14T23:09:30+00:00"},             # no dry_run flag
        "adaptation": {"last_run": "2026-06-13T22:04:52+00:00", "dry_run": True},
        "concept_drift": {"checked_at": "2026-06-18T18:35:24+00:00", "dry_run": True},
        "version_monitor": {"last_run": "t", "checked_at": "2026-06-18T18:36:54+00:00", "dry_run": True},
        "training": {"last_run": None}, "interpretability": {"last_run": None},
        "agent_messages": {},
    }
    by = {r["agent"]: r for r in C.assess(reg, pipes, state, NOW, 30.0)[0]}
    # section-only, no dry_run flag -> SECTION_ONLY (NOT silently ACTIVE/STALE)
    assert by["LiteratureScoutAgent"]["status"] == "SECTION_ONLY"
    assert by["FinOpsAdvisorAgent"]["status"] == "SECTION_ONLY"
    # dry-run section -> DRY_RUN_ONLY (the true "never really ran" signal)
    assert by["AdaptationAgent"]["status"] == "DRY_RUN_ONLY"
    assert by["ConceptDriftMonitorAgent"]["status"] == "DRY_RUN_ONLY"
    # version_monitor.last_run=='t' is non-ISO -> falls back to checked_at (dry-run)
    assert by["VersionMonitorAgent"]["status"] == "DRY_RUN_ONLY"
    # null last_run / no section at all -> NEVER_RUN
    assert by["TrainingLifecycleAgent"]["status"] == "NEVER_RUN"
    assert by["ModelInsightsAgent"]["status"] == "NEVER_RUN"   # no persisted section
    assert C.has_problem(list(by.values()), strict=False) is True  # every state here is a problem

def test_active_requires_authoritative_agent_runs_not_section():
    # A recent section write alone must NOT read as ACTIVE; only agent_runs does.
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    recent = (NOW - timedelta(days=2)).isoformat()
    old = (NOW - timedelta(days=90)).isoformat()
    state = {"agent_runs": {"FinOpsAdvisorAgent": [{"ts": recent, "status": "finops_recommendation"}],
                            "DataFreshnessAgent": [{"ts": old, "status": "poll"}]},
             "literature": {"last_run": recent}}              # section only -> SECTION_ONLY
    by = {r["agent"]: r for r in C.assess(reg, pipes, state, NOW, 30.0)[0]}
    assert by["FinOpsAdvisorAgent"]["status"] == "ACTIVE"     # via agent_runs
    assert by["DataFreshnessAgent"]["status"] == "STALE"      # via agent_runs (90d>30d)
    assert by["LiteratureScoutAgent"]["status"] == "SECTION_ONLY"   # NOT ACTIVE
    assert by["TrainingLifecycleAgent"]["status"] == "NEVER_RUN"

def test_real_run_makes_all_active():
    # After a real --pipeline all run, agent_runs exists for all 21 -> all ACTIVE.
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    recent = (NOW - timedelta(hours=1)).isoformat()
    state = {"agent_runs": {a: [{"ts": recent, "status": "unknown"}] for a in reg}}
    rows, problem = C.assess(reg, pipes, state, NOW, 30.0)
    assert problem is False and all(r["status"] == "ACTIVE" for r in rows)

def test_nested_freshness_section_counts_as_section_only():
    # data_freshness has no top-level ts; the one-level-deep last_checked must be found.
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    state = {"data_freshness": {"clinvar": {"last_checked": (NOW - timedelta(days=3)).isoformat()}}}
    by = {r["agent"]: r for r in C.assess(reg, pipes, state, NOW, 30.0)[0]}
    assert by["DataFreshnessAgent"]["status"] == "SECTION_ONLY"

def test_all_active_no_problem():
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    recent = (NOW - timedelta(days=1)).isoformat()
    state = {"agent_runs": {a: [{"ts": recent, "status": "ok"}] for a in reg}}
    rows, problem = C.assess(reg, pipes, state, NOW, 30.0)
    assert problem is False and all(r["status"] == "ACTIVE" for r in rows)

def test_unscheduled_and_missing_impl_both_flagged():
    reg = {"ProvisioningAgent", "FinOpsAdvisorAgent"}
    pipes = {"finops": ["FinOpsAdvisorAgent"]}   # ProvisioningAgent registered but not scheduled
    by = {r["agent"]: r for r in C.assess(reg, pipes, {"agent_runs": {}}, NOW, 30.0)[0]}
    assert by["ProvisioningAgent"]["status"] == "UNSCHEDULED"

def test_strict_promotes_stale_to_problem():
    rows = [{"status": "STALE", "agent": "X"}]
    assert C.has_problem(rows, strict=False) is False
    assert C.has_problem(rows, strict=True) is True

def test_errored_last_run_is_not_silently_active():
    reg, pipes = C.parse_orchestrator(REAL_ORCH)
    recent = (NOW - timedelta(days=1)).isoformat()
    state = {"agent_runs": {a: [{"ts": recent, "status": "ok"}] for a in reg}}
    state["agent_runs"]["FinOpsAdvisorAgent"] = [{"ts": recent, "status": "error", "error": "boom"}]
    by = {r["agent"]: r for r in C.assess(reg, pipes, state, NOW, 30.0)[0]}
    assert by["FinOpsAdvisorAgent"]["status"] == "ERRORED"   # ran but crashed -> flagged, not ACTIVE
    assert C.has_problem(C.assess(reg, pipes, state, NOW, 30.0)[0], strict=False) is True
