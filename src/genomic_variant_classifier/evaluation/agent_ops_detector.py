"""
agent_ops_detector.py -- Monzia Moodie

Pure, schema-agnostic OPS scan over the agent_state.json dict: one flat pane answering "is anything stale, stuck,
or backed up across the agent layer?". Four read-only dimensions, all derived from what SharedState actually
persists -- NOTHING is fabricated:

  1. Heartbeats   -- per state section, the newest ISO timestamp found anywhere within it + its age; flagged
                     stale only above a generous, cadence-aware threshold (sections have different cadences:
                     per-run / weekly / monthly, so staleness is advisory, not a hard alarm).
  2. Inbox backlog-- per-agent unread + pending-approval message counts (same logic as SharedState.summary()).
  3. Review backlog- count of unresolved review_items.
  4. Surfaced flags- problems an agent already recorded: a data_readiness verdict != GO, non-empty
                     instability_flags / model-insights flags, etc.

NOT covered (no data source -- documented gap, not a silent stub): per-agent ERROR-RATE and run-DURATION /
perf-drift. agent_state.json records no per-run status/duration/error telemetry; computing those would require a
separate orchestrator change to persist run telemetry (a future 'agent_runs' section). This detector never
invents those numbers. No BaseAgent / no SharedState -> unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

DEFAULT_STALE_HOURS = 24 * 35          # ~5 weeks: generous so monthly-cadence sections do not false-alarm
_NONHEARTBEAT_SECTIONS = {"review_items", "agent_messages", "artifact_ledger"}


@dataclass
class Heartbeat:
    section: str
    newest_iso: str | None
    age_hours: float | None
    stale: bool
    detail: str


@dataclass
class InboxStatus:
    agent: str
    unread: int
    pending_approval: int


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _parse_iso(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _as_utc(datetime.fromisoformat(value))
    except (ValueError, TypeError):
        return None


def _collect_timestamps(obj) -> list[datetime]:
    out: list[datetime] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_timestamps(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_timestamps(v))
    else:
        ts = _parse_iso(obj)
        if ts is not None:
            out.append(ts)
    return out


def scan_heartbeats(state: dict, now: datetime, stale_after_hours: float = DEFAULT_STALE_HOURS) -> list[Heartbeat]:
    now = _as_utc(now)
    beats: list[Heartbeat] = []
    for section, payload in state.items():
        if section in _NONHEARTBEAT_SECTIONS:
            continue
        stamps = _collect_timestamps(payload)
        if not stamps:
            beats.append(Heartbeat(section, None, None, False, "no timestamp recorded yet"))
            continue
        newest = max(stamps)
        age_h = (now - newest).total_seconds() / 3600.0
        stale = age_h >= stale_after_hours
        beats.append(Heartbeat(section, newest.isoformat(), round(age_h, 2), stale,
                               "STALE" if stale else "ok"))
    return beats


def scan_inbox(state: dict) -> list[InboxStatus]:
    out: list[InboxStatus] = []
    for agent, inbox in state.get("agent_messages", {}).items():
        unread = sum(1 for m in inbox if not m.get("read"))
        pending = sum(1 for m in inbox if m.get("requires_approval") and m.get("approved") is None)
        if unread or pending:
            out.append(InboxStatus(agent, unread, pending))
    return out


def unresolved_reviews(state: dict) -> int:
    return sum(1 for i in state.get("review_items", []) if not i.get("resolved"))


def scan_flags(state: dict) -> list[str]:
    flags: list[str] = []
    dr = state.get("data_readiness", {})
    if dr.get("verdict") and dr["verdict"] != "GO":
        flags.append(f"data_readiness verdict={dr['verdict']}")
    interp = state.get("interpretability", {})
    if interp.get("instability_flags"):
        flags.append(f"interpretability instability_flags={len(interp['instability_flags'])}")
    mi = state.get("model_insights", {})
    if mi.get("flags"):
        flags.append(f"model_insights flags={len(mi['flags'])}")
    return flags


def analyze(state: dict, now: datetime | None = None, stale_after_hours: float = DEFAULT_STALE_HOURS) -> dict:
    now = _as_utc(now or datetime.now(timezone.utc))
    beats = scan_heartbeats(state, now, stale_after_hours)
    inbox = scan_inbox(state)
    reviews = unresolved_reviews(state)
    flags = scan_flags(state)
    attention = bool(flags) or reviews > 0 \
        or any(b.stale for b in beats) \
        or any(s.pending_approval for s in inbox)
    return {
        "ops_status": "ATTENTION" if attention else "OK",
        "heartbeats": beats,
        "inbox": inbox,
        "unresolved_reviews": reviews,
        "flags": flags,
    }
