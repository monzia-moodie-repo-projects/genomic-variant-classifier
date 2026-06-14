"""
agent_ops_monitor_agent.py -- Monzia Moodie

Flat agent-layer OPS monitor (BaseAgent). Loads the whole agent_state.json, runs the schema-agnostic
agent_ops_detector, writes a DOCUMENTED dashboard report, and records its own heartbeat to SharedState
('agent_ops') -- so it appears in its own pane (self-monitoring, NON-recursive: one flat monitor, no
agent-of-agent tower). Read-only and informational: it never gates, never mutates another agent's section, and
never executes anything. It surfaces staleness / inbox backlog / unresolved reviews / recorded problem flags so a
stuck or stale agent layer is visible at a glance.

Scope honesty: per-agent ERROR-RATE and run-DURATION/perf-drift are intentionally NOT reported -- agent_state.json
persists no run telemetry, so those would be fabricated. They are a documented future enhancement requiring an
orchestrator change to record run telemetry. `stale_after_hours` and `root` are injectable for hermetic tests;
the orchestrator constructs it as cls(shared_state).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.evaluation import agent_ops_detector as D

_SECTION = "agent_ops"


class AgentOpsMonitorAgent(BaseAgent):
    def __init__(self, shared_state, stale_after_hours: float = D.DEFAULT_STALE_HOURS, root: str = ".") -> None:
        super().__init__(shared_state)
        self._stale_after_hours = stale_after_hours
        self._root = root

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)

        state = self._state.load()
        analysis = D.analyze(state, stale_after_hours=self._stale_after_hours)
        report_path = self._write_report(analysis)

        n_stale = sum(1 for b in analysis["heartbeats"] if b.stale)
        n_no_hb = sum(1 for b in analysis["heartbeats"] if b.newest_iso is None)
        self._update_section(_SECTION, {
            "last_run": self._now_iso(),
            "ops_status": analysis["ops_status"],
            "n_sections": len(analysis["heartbeats"]),
            "n_stale": n_stale,
            "n_no_heartbeat": n_no_hb,
            "inbox_backlog_agents": [s.agent for s in analysis["inbox"]],
            "unresolved_reviews": analysis["unresolved_reviews"],
            "flags": analysis["flags"],
            "agents_with_errors": [t.agent for t in analysis["telemetry"] if t.n_errors],
            "perf_drift_agents": [t.agent for t in analysis["telemetry"]
                                  if t.drift_pct is not None and t.drift_pct >= D.PERF_DRIFT_PCT],
            "report": str(report_path),
        })

        result = {
            "action": "agent_ops_scan",
            "ops_status": analysis["ops_status"],
            "n_sections": len(analysis["heartbeats"]),
            "n_stale": n_stale,
            "unresolved_reviews": analysis["unresolved_reviews"],
            "flags": len(analysis["flags"]),
            "report": str(report_path),
        }
        self._log_finish(result)
        return result

    def _write_report(self, analysis: dict) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = Path(self._root) / "reports" / "agent_ops" / f"OPS_{ts}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Agent-layer ops dashboard -- {ts}", "",
            f"## STATUS: {analysis['ops_status']}", "",
            "## Heartbeats (newest timestamp per state section)", "",
            "| section | newest | age (h) | state |",
            "| --- | --- | --- | --- |",
        ]
        for b in analysis["heartbeats"]:
            newest = "--" if b.newest_iso is None else b.newest_iso[:19]
            age = "--" if b.age_hours is None else f"{b.age_hours:.1f}"
            lines.append(f"| {b.section} | {newest} | {age} | {b.detail} |")
        lines += ["", "## Inbox backlog", ""]
        if analysis["inbox"]:
            for s in analysis["inbox"]:
                lines.append(f"- {s.agent}: {s.unread} unread, {s.pending_approval} pending approval")
        else:
            lines.append("- none")
        lines += ["", f"## Unresolved review items: {analysis['unresolved_reviews']}", "",
                  "## Surfaced problem flags", ""]
        lines += [f"- {f}" for f in analysis["flags"]] if analysis["flags"] else ["- none"]
        lines += ["", "## Run telemetry (from 'agent_runs'; real runs only)", ""]
        if analysis["telemetry"]:
            lines += ["| agent | runs | errors | error-rate | median ms | drift % |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for t in analysis["telemetry"]:
                med = "--" if t.median_ms is None else f"{t.median_ms:.1f}"
                drift = "--" if t.drift_pct is None else f"{t.drift_pct:+.1f}"
                lines.append(f"| {t.agent} | {t.n_runs} | {t.n_errors} | {t.error_rate:.0%} | {med} | {drift} |")
        else:
            lines.append("- no run telemetry recorded yet (agents run via the orchestrator on real, non-dry-run executions)")
        lines.append("")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
