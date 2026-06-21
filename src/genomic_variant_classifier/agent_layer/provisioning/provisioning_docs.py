#!/usr/bin/env python3
"""provisioning_docs.py  --  Author: Monzia Moodie

Provisioning bookkeeping for the (separate) ProvisioningAgent: a `provisioning`
SharedState section + an append-only markdown run-doc that satisfies the standing
rule to document EVERY provisioning detail for every preflight and official run
(provider, search filters, candidate offers considered, chosen offer + why, image,
disk, $/hr, instance/contract id, SSH host/port, SCP legs+sizes, every preflight
gate result, agent-liveness result, teardown/destroy confirmation, total cost).

This module is PURE (stdlib only, no torch / no package imports) so it is fully
unit-testable in the sandbox. The ProvisioningAgent wires it to the real
SharedState via BaseAgent._get_section / _update_section.

Design notes
------------
* The `provisioning` section mirrors the real SharedState's nested-section style
  (cf. finops/adaptation): a small set of "last_*" scalars for at-a-glance status
  plus a capped `history` list of full event records. update_section() merges
  shallowly, so history is read-modify-written here (append + cap) rather than
  relying on update to append.
* new_event() validates loudly: a provisioning record with a negative price, a
  missing provider/phase, or a non-numeric cost is a bug we want to raise on, not
  silently persist (the whole point is auditable provisioning).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROVISIONING_SECTION = "provisioning"

# Lifecycle phases a provisioning event can describe.
PHASES = ("select", "provision", "teardown")
PROVIDERS = ("vast", "runpod")

_HISTORY_CAP = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_event(
    *,
    phase: str,
    provider: str,
    offer_id: str,
    gpu_name: str = "",
    num_gpus: int = 1,
    vram_gb: float = 0.0,
    price_per_hr: float = 0.0,
    budget_cap_per_hr: float | None = None,
    est_hours: float | None = None,
    within_budget: bool | None = None,
    approved: bool | None = None,
    dry_run: bool = True,
    instance_id: str | None = None,
    ssh_host: str | None = None,
    ssh_port: int | None = None,
    candidate_offers: list[dict] | None = None,
    search_filters: dict | None = None,
    image: str | None = None,
    disk_gb: float | None = None,
    scp_legs: list[dict] | None = None,
    preflight_gates: dict | None = None,
    agent_liveness: str | None = None,
    teardown_confirmed: bool | None = None,
    total_cost_usd: float | None = None,
    reason: str = "",
    doc_path: str | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Build a validated provisioning event record. Raises ValueError on bad input."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    if not offer_id:
        raise ValueError("offer_id is required (auditability)")
    if price_per_hr < 0:
        raise ValueError(f"price_per_hr must be >= 0, got {price_per_hr}")
    if budget_cap_per_hr is not None and budget_cap_per_hr < 0:
        raise ValueError(f"budget_cap_per_hr must be >= 0, got {budget_cap_per_hr}")

    est_cost = None
    if est_hours is not None:
        if est_hours < 0:
            raise ValueError(f"est_hours must be >= 0, got {est_hours}")
        est_cost = round(price_per_hr * est_hours, 4)
    if within_budget is None and budget_cap_per_hr is not None:
        within_budget = price_per_hr <= budget_cap_per_hr

    return {
        "ts": ts or _now_iso(),
        "phase": phase,
        "provider": provider,
        "offer_id": str(offer_id),
        "gpu_name": gpu_name,
        "num_gpus": int(num_gpus),
        "vram_gb": float(vram_gb),
        "price_per_hr": float(price_per_hr),
        "budget_cap_per_hr": budget_cap_per_hr,
        "est_hours": est_hours,
        "est_cost_usd": est_cost,
        "within_budget": within_budget,
        "approved": approved,
        "dry_run": bool(dry_run),
        "instance_id": instance_id,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "candidate_offers": candidate_offers or [],
        "search_filters": search_filters or {},
        "image": image,
        "disk_gb": disk_gb,
        "scp_legs": scp_legs or [],
        "preflight_gates": preflight_gates or {},
        "agent_liveness": agent_liveness,
        "teardown_confirmed": teardown_confirmed,
        "total_cost_usd": total_cost_usd,
        "reason": reason,
        "doc_path": doc_path,
    }


def record_provisioning(
    get_section: Callable[[str], dict],
    update_section: Callable[[str, dict], Any],
    event: dict,
    *,
    cap: int = _HISTORY_CAP,
) -> dict:
    """Persist a provisioning event into the `provisioning` SharedState section.

    get_section/update_section are BaseAgent._get_section/_update_section (or any
    pair with the SharedState contract: get_section(name)->dict and
    update_section(name, updates) shallow-merges). History is read-modify-written
    (append + cap newest-last). Returns the merged section dict that was written.
    """
    cur = dict(get_section(PROVISIONING_SECTION) or {})
    history = list(cur.get("history") or [])
    history.append(event)
    if len(history) > cap:
        history = history[-cap:]

    merged = {
        "last_run": event["ts"],
        "last_phase": event["phase"],
        "last_provider": event["provider"],
        "last_offer_id": event["offer_id"],
        "last_instance_id": event.get("instance_id"),
        "last_within_budget": event.get("within_budget"),
        "last_approved": event.get("approved"),
        "dry_run": event.get("dry_run", True),
        "n_events": len(history),
        "history": history,
    }
    update_section(PROVISIONING_SECTION, merged)
    return merged


def _fmt(v: Any) -> str:
    if v is None:
        return "_(n/a)_"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def render_provisioning_doc(event: dict) -> str:
    """Return the markdown body documenting one provisioning event (every detail)."""
    e = event
    lines: list[str] = []
    lines.append(f"# Provisioning record — {e['provider']} / {e['offer_id']}")
    lines.append("")
    lines.append(f"- **Timestamp (UTC):** {e['ts']}")
    lines.append(f"- **Phase:** {e['phase']}")
    lines.append(f"- **Dry-run:** {_fmt(e.get('dry_run'))}")
    lines.append(f"- **Author:** Monzia Moodie")
    lines.append("")
    lines.append("## Selection")
    lines.append(f"- **Provider:** {_fmt(e.get('provider'))}")
    lines.append(f"- **Chosen offer id:** {_fmt(e.get('offer_id'))}")
    lines.append(f"- **GPU:** {_fmt(e.get('gpu_name'))} x{_fmt(e.get('num_gpus'))} "
                 f"({_fmt(e.get('vram_gb'))} GB VRAM)")
    lines.append(f"- **Price/hr:** ${_fmt(e.get('price_per_hr'))}")
    lines.append(f"- **Budget cap/hr:** {('$' + str(e['budget_cap_per_hr'])) if e.get('budget_cap_per_hr') is not None else '_(n/a)_'}")
    lines.append(f"- **Within budget:** {_fmt(e.get('within_budget'))}")
    lines.append(f"- **Est. hours:** {_fmt(e.get('est_hours'))}  |  "
                 f"**Est. cost:** {('$' + str(e['est_cost_usd'])) if e.get('est_cost_usd') is not None else '_(n/a)_'}")
    lines.append(f"- **Why this offer:** {_fmt(e.get('reason'))}")
    sf = e.get("search_filters") or {}
    lines.append(f"- **Search filters:** {', '.join(f'{k}={v}' for k, v in sf.items()) or '_(n/a)_'}")
    lines.append("")
    lines.append("### Candidate offers considered")
    cands = e.get("candidate_offers") or []
    if cands:
        lines.append("| provider | offer_id | gpu | $/hr |")
        lines.append("|---|---|---|---|")
        for c in cands:
            lines.append(f"| {_fmt(c.get('provider'))} | {_fmt(c.get('offer_id'))} | "
                         f"{_fmt(c.get('gpu_name'))} | {_fmt(c.get('price_per_hr'))} |")
    else:
        lines.append("_(none recorded)_")
    lines.append("")
    lines.append("## Provisioning")
    lines.append(f"- **Image:** {_fmt(e.get('image'))}")
    lines.append(f"- **Disk (GB):** {_fmt(e.get('disk_gb'))}")
    lines.append(f"- **HITL approved:** {_fmt(e.get('approved'))}")
    lines.append(f"- **Instance / contract id:** {_fmt(e.get('instance_id'))}")
    lines.append(f"- **SSH host:** {_fmt(e.get('ssh_host'))}  |  **SSH port:** {_fmt(e.get('ssh_port'))}")
    lines.append("")
    lines.append("### SCP legs")
    legs = e.get("scp_legs") or []
    if legs:
        lines.append("| direction | path | size |")
        lines.append("|---|---|---|")
        for leg in legs:
            lines.append(f"| {_fmt(leg.get('direction'))} | {_fmt(leg.get('path'))} | "
                         f"{_fmt(leg.get('size'))} |")
    else:
        lines.append("_(none recorded)_")
    lines.append("")
    lines.append("## Gates")
    gates = e.get("preflight_gates") or {}
    if gates:
        for k, v in gates.items():
            lines.append(f"- **{k}:** {_fmt(v)}")
    else:
        lines.append("- _(no preflight gates recorded)_")
    lines.append(f"- **Agent liveness:** {_fmt(e.get('agent_liveness'))}")
    lines.append("")
    lines.append("## Teardown")
    lines.append(f"- **Teardown confirmed:** {_fmt(e.get('teardown_confirmed'))}")
    lines.append(f"- **Total cost (USD):** {('$' + str(e['total_cost_usd'])) if e.get('total_cost_usd') is not None else '_(n/a)_'}")
    lines.append("")
    return "\n".join(lines)


def _safe_token(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(s))[:40]


def write_provisioning_doc(repo_root: str | Path, event: dict) -> Path:
    """Write the markdown run-doc under docs/provisioning/ and return its path.

    Filename: PROVISION_<ts>_<provider>_<offer>.md (filesystem-safe tokens).
    """
    root = Path(repo_root)
    out_dir = root / "docs" / "provisioning"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_token = _safe_token(event["ts"].replace(":", "").replace("+", "Z"))
    fname = f"PROVISION_{ts_token}_{_safe_token(event['provider'])}_{_safe_token(event['offer_id'])}.md"
    path = out_dir / fname
    # BOM-free UTF-8 (PS5.1 lesson: write_bytes avoids encoding surprises).
    path.write_bytes(render_provisioning_doc(event).encode("utf-8"))
    return path
