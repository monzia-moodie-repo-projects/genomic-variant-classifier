#!/usr/bin/env python3
"""check_agents_active.py  --  Author: Monzia Moodie

Agent-liveness gate (standing rule: no agent may be dormant). Reports, for every
agent, whether it is WIRED (a registered class), SCHEDULED (appears in >=1
orchestrator pipeline), and ACTIVE (has recent run telemetry in the SharedState).
Exits non-zero if any agent is a problem, so it can gate preflight (before AND
after every launch) and run at session start.

Ground truth (read, not guessed):
  * Registered agents + pipelines come from the orchestrator SOURCE, parsed with
    ast -- NO import of the package or the agent modules. This means the checker
    never crashes because torch/xgboost or a single agent import is broken; a
    broken import is itself a liveness problem we want to REPORT, not die on.
  * "ACTIVE" evidence comes from the SharedState file -- which is
    src/genomic_variant_classifier/agent_layer/agent_state.json
    (_DEFAULT_STATE_FILE = Path(__file__).parent / 'agent_state.json'), NOT
    data/agent_state.json (that is a separate flat literature_scout.* version
    store and is NOT a SharedState). The primary signal is the on-demand
    'agent_runs' telemetry section written by Orchestrator._record_run_telemetry;
    a few agents also stamp a per-section last_run.

Statuses:
  ACTIVE       registered + scheduled + agent_runs telemetry within max-age window
  STALE        registered + scheduled + agent_runs telemetry OLDER than window  (warn; problem under --strict)
  ERRORED      registered + scheduled + last agent_runs status == "error"       (problem -> ran but crashed)
  DRY_RUN_ONLY registered + scheduled + only a dry-run section write, no telemetry (problem -> never really ran)
  SECTION_ONLY registered + scheduled + a section timestamp but no telemetry      (problem -> not via the authoritative path)
  NEVER_RUN    registered + scheduled + no activity at all                       (problem -> dormant)
  UNSCHEDULED  registered but in NO pipeline                                     (problem -> dormant-by-design)
  MISSING_IMPL scheduled in a pipeline but NOT registered                        (problem)

Authoritative signal is agent_runs (written by Orchestrator only on non-dry-run
executions). A section timestamp alone never reads as ACTIVE -- that is exactly
how the agents looked "dormant": they had only ever been dry-run.

Exit code: 0 if no problems, 1 otherwise (STALE counts only under --strict).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Per-section last_run fallbacks (agents that stamp their own section instead of,
# or in addition to, the orchestrator's agent_runs telemetry).
# Complete agent -> own SharedState section map (grounded in the real
# agent_state.json, not guessed). Each listed section carries a top-level or
# one-level-deep ISO timestamp (last_run / checked_at / last_checked). The three
# agents with NO persisted section (ModelInsightsAgent, DataReadinessAgent,
# AgentOpsMonitorAgent) are intentionally absent -> they rely on agent_runs only.
_AGENT_SECTION = {
    "DataFreshnessAgent": "data_freshness",
    "TrainingLifecycleAgent": "training",
    "InterpretabilityAgent": "interpretability",
    "LiteratureScoutAgent": "literature",
    "SchemaDriftMonitorAgent": "schema_drift",
    "ConceptDriftMonitorAgent": "concept_drift",
    "LabelShiftMonitorAgent": "label_shift",
    "CalibrationDriftMonitorAgent": "calibration_drift",
    "InfrastructureDriftMonitorAgent": "infrastructure_drift",
    "FairnessSubgroupMonitorAgent": "fairness_subgroup",
    "AdversarialSubmissionMonitorAgent": "adversarial_submission",
    "AnnotationPolicyMonitorAgent": "annotation_policy",
    "FeatureCoverageSentinelMonitorAgent": "feature_coverage",
    "ReclassificationSentinelMonitorAgent": "reclassification",
    "VersionMonitorAgent": "version_monitor",
    "AdaptationAgent": "adaptation",
    "FinOpsAdvisorAgent": "finops",
    "DatabaseFreshnessMonitorAgent": "database_freshness",
    "ProvisioningAgent": "provisioning",
}
# Top-level keys that prove a file is a real SharedState (from _default_state()).
_SHAREDSTATE_SECTIONS = {
    "data_freshness", "training", "interpretability", "literature",
    "review_items", "agent_messages", "artifact_ledger",
}


# --------------------------------------------------------------------------- #
# Pure parsing / assessment core (import-free; unit-tested in the sandbox)
# --------------------------------------------------------------------------- #

def parse_orchestrator(source: str) -> tuple[set[str], dict[str, list[str]]]:
    """Extract (registered_agent_names, pipeline_definitions) from orchestrator
    source via ast -- no import. Registered names are the string KEYS of the
    `self._agent_registry = { "Name": Name, ... }` dict; pipelines are the
    `PIPELINE_DEFINITIONS = { "name": [ "Agent", ... ] }` literal (the computed
    ["all"] augmentation is ignored; the union is recomputed by the caller)."""
    tree = ast.parse(source)
    registered: set[str] = set()
    pipelines: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        # PIPELINE_DEFINITIONS literal (annotated or plain assignment)
        target_names = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
        elif isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "PIPELINE_DEFINITIONS" in target_names and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.List):
                    pipelines[k.value] = [
                        e.value for e in v.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]

        # self._agent_registry = { "Name": Name, ... }
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and t.attr == "_agent_registry"
                        and isinstance(node.value, ast.Dict)):
                    for k in node.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            registered.add(k.value)

    return registered, pipelines


def pipeline_union(pipelines: dict[str, list[str]]) -> set[str]:
    return {a for agents in pipelines.values() for a in agents}


def looks_like_sharedstate(state: dict) -> bool:
    """True iff the loaded JSON has at least one _default_state() section."""
    return any(s in state for s in _SHAREDSTATE_SECTIONS)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


_HARD_FAIL = {"UNSCHEDULED", "MISSING_IMPL", "NEVER_RUN", "ERRORED",
              "DRY_RUN_ONLY", "SECTION_ONLY"}


def _agent_runs_activity(state: dict, agent: str) -> tuple[datetime | None, str | None]:
    """Authoritative signal: last (ts, status) from agent_runs telemetry, which
    Orchestrator._record_run_telemetry writes ONLY on non-dry-run executions."""
    runs = (state.get("agent_runs") or {}).get(agent) or []
    if runs:
        dt = _parse_iso(runs[-1].get("ts"))
        if dt is not None:
            return dt, runs[-1].get("status", "?")
    return None, None


def _section_activity(state: dict, section: str | None) -> tuple[datetime | None, bool | None]:
    """Secondary signal: newest ISO timestamp the agent stamped in its OWN
    section, plus that section's dry_run flag. Handles three shapes seen in the
    real SharedState: (a) top-level last_run/checked_at (drift+monitor agents;
    version_monitor.last_run=='t' is non-ISO and is skipped in favour of
    checked_at), and (b) nested source dicts with last_checked
    (data_freshness/database_freshness). Returns (dt|None, dry_run|None)."""
    if not section:
        return None, None
    sec = state.get(section)
    if not isinstance(sec, dict):
        return None, None
    dry = sec.get("dry_run")
    # (a) top-level timestamps
    cands = [_parse_iso(sec.get("last_run")), _parse_iso(sec.get("checked_at"))]
    # (b) one level deep (nested per-source dicts)
    for v in sec.values():
        if isinstance(v, dict):
            cands.append(_parse_iso(v.get("last_checked")))
            cands.append(_parse_iso(v.get("last_run")))
    cands = [c for c in cands if c is not None]
    return (max(cands) if cands else None), dry


def assess(registered: set[str], pipelines: dict[str, list[str]], state: dict,
           now: datetime, max_age_days: float) -> tuple[list[dict], bool]:
    """Return (rows, has_problem). One row per agent in registered ∪ scheduled."""
    scheduled = pipeline_union(pipelines)
    rows: list[dict] = []
    for agent in sorted(registered | scheduled):
        wired = agent in registered
        is_sched = agent in scheduled
        runs_dt, runs_status = _agent_runs_activity(state, agent)
        sec_dt, sec_dry = _section_activity(state, _AGENT_SECTION.get(agent))

        last, src, age_days = None, None, None
        if wired and not is_sched:
            row_status = "UNSCHEDULED"
        elif is_sched and not wired:
            row_status = "MISSING_IMPL"
        elif runs_dt is not None:                       # authoritative telemetry
            last, src = runs_dt, f"agent_runs[{runs_status}]"
            age_days = (now - last).total_seconds() / 86400.0
            if runs_status == "error":
                row_status = "ERRORED"                  # ran but crashed
            elif age_days > max_age_days:
                row_status = "STALE"
            else:
                row_status = "ACTIVE"
        elif sec_dt is not None:                        # wrote a section, no telemetry
            last, src = sec_dt, f"{_AGENT_SECTION.get(agent)} (section)"
            age_days = (now - last).total_seconds() / 86400.0
            # only ever dry-run, or never recorded telemetry: NOT active per the rule
            row_status = "DRY_RUN_ONLY" if sec_dry is True else "SECTION_ONLY"
        else:
            row_status = "NEVER_RUN"

        rows.append({
            "agent": agent, "wired": wired, "scheduled": is_sched,
            "pipelines": sorted(p for p, a in pipelines.items() if agent in a),
            "last_activity": last.isoformat() if last else None,
            "activity_source": src,
            "age_days": round(age_days, 2) if age_days is not None else None,
            "status": row_status,
        })
    return rows, has_problem(rows, strict=False)


def has_problem(rows: list[dict], strict: bool) -> bool:
    bad = _HARD_FAIL | ({"STALE"} if strict else set())
    return any(r["status"] in bad for r in rows)


# --------------------------------------------------------------------------- #
# Thin CLI (does the on-machine file reads; no package import required)
# --------------------------------------------------------------------------- #

def _default_paths(repo_root: Path) -> tuple[Path, Path]:
    base = repo_root / "src" / "genomic_variant_classifier" / "agent_layer"
    return base / "orchestrator.py", base / "agent_state.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agent liveness gate (no dormant agents).")
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--orchestrator", type=Path, default=None,
                    help="override path to orchestrator.py")
    ap.add_argument("--state", type=Path, default=None,
                    help="override path to the SharedState agent_state.json")
    ap.add_argument("--max-age-days", type=float, default=30.0)
    ap.add_argument("--strict", action="store_true", help="treat STALE as a failure too")
    ap.add_argument("--phase", default="", help="label, e.g. pre / post")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--no-fail", action="store_true", help="report only; always exit 0")
    args = ap.parse_args(argv)

    orch_path, state_path = _default_paths(args.repo_root)
    orch_path = args.orchestrator or orch_path
    state_path = args.state or state_path

    if not orch_path.exists():
        print(f"ERROR: orchestrator not found at {orch_path}", file=sys.stderr)
        return 2
    registered, pipelines = parse_orchestrator(orch_path.read_text(encoding="utf-8"))

    warnings: list[str] = []
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"ERROR: cannot read state {state_path}: {exc}", file=sys.stderr)
            return 2
        if state and not looks_like_sharedstate(state):
            warnings.append(
                f"{state_path} has NONE of the SharedState sections "
                f"({sorted(_SHAREDSTATE_SECTIONS)}); this is NOT the orchestrator "
                f"SharedState (likely the flat literature_scout.* store). All agents "
                f"will read as NEVER_RUN. Point --state at "
                f"src/.../agent_layer/agent_state.json.")
    else:
        state = {}
        warnings.append(f"SharedState file {state_path} does not exist -> the "
                        f"orchestrator has never persisted a real run; all agents NEVER_RUN.")

    now = datetime.now(timezone.utc)
    rows, _ = assess(registered, pipelines, state, now, args.max_age_days)
    problem = has_problem(rows, args.strict)

    if args.json:
        print(json.dumps({"phase": args.phase, "warnings": warnings,
                          "problem": problem, "rows": rows}, indent=2))
    else:
        tag = f"[{args.phase}] " if args.phase else ""
        print(f"{tag}Agent liveness  --  {len(rows)} agents  "
              f"(registered={len(registered)}, scheduled={len(pipeline_union(pipelines))})")
        for w in warnings:
            print(f"  ! WARNING: {w}")
        for r in rows:
            age = f"{r['age_days']}d" if r["age_days"] is not None else "-"
            print(f"  {r['status']:<12} {r['agent']:<38} "
                  f"sched={'Y' if r['scheduled'] else 'N'} "
                  f"last={r['last_activity'] or 'never':<32} age={age} "
                  f"src={r['activity_source'] or '-'}")
        bad = [r["agent"] for r in rows
               if r["status"] in (_HARD_FAIL | ({"STALE"} if args.strict else set()))]
        print(f"\n  {'PROBLEM' if problem else 'OK'}: "
              f"{len(bad)} dormant/problem agent(s){': ' + ', '.join(bad) if bad else ''}")

    return 0 if (args.no_fail or not problem) else 1


if __name__ == "__main__":
    raise SystemExit(main())
