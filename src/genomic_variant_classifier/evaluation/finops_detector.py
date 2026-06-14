"""
finops_detector.py -- Monzia Moodie

Pure, RECOMMEND-ONLY FinOps logic for GPU rental. It NEVER calls vastai, NEVER provisions, NEVER spends: given an
offers SNAPSHOT (a `vastai search offers ... --raw` JSON dump) it selects the cheapest suitable single-GPU offer,
estimates the run cost, checks it against a budget cap, and EMITS a recommended (preview) launch command string for
a human to run -- mirroring the established `preflight_gate.py --emit` HITL pattern.

`pick_offer` here is the CANONICAL copy of the selection logic that also lives (untested, legacy) in
scripts/launch_run16.py; test_finops_detector.py asserts behavioural PARITY between the two so they cannot drift
silently. No BaseAgent / no SharedState -> unit-testable.
"""
from __future__ import annotations

import json
from pathlib import Path

WITHIN_BUDGET = "WITHIN_BUDGET"
OVER_BUDGET = "OVER_BUDGET"
NO_SUITABLE_OFFER = "NO_SUITABLE_OFFER"

DEFAULT_EST_HOURS = 15.0      # a full Run is ~15-19 h (launch_run16 cost note: ~$7-9 at $0.473/hr)
DEFAULT_BUDGET_USD = 15.0     # advisory cap; injectable


def pick_offer(offers: list[dict]) -> dict | None:
    """Cheapest SINGLE-GPU offer by (price, -reliability, -cpu_ram). Canonical copy; parity-tested vs launch_run16."""
    def price(o): return float(o.get("dph_total", o.get("dph", 1e9)))
    def rel(o): return float(o.get("reliability2", o.get("reliability", 0)))
    def ram(o): return float(o.get("cpu_ram", 0))
    usable = [o for o in offers if int(o.get("num_gpus", 1)) == 1]
    if not usable:
        return None
    return sorted(usable, key=lambda o: (price(o), -rel(o), -ram(o)))[0]


def estimate_cost(hours: float, dph: float) -> float:
    return round(float(hours) * float(dph), 2)


def load_offers_snapshot(path: str | Path) -> list[dict]:
    """Parse a `vastai search offers --raw` JSON dump (a list, or {'offers': [...]}). Returns [] on anything odd.

    Read as BYTES so json auto-detects the encoding (UTF-8/16/32, with or without BOM). This matters on Windows:
    PowerShell 5.1 `... > offers.json` writes UTF-16-LE-with-BOM, and `Out-File -Encoding utf8` writes a UTF-8 BOM;
    a strict utf-8 text read would reject both (0xff / 0xef at position 0).
    """
    data = json.loads(Path(path).read_bytes())
    if isinstance(data, dict):
        data = data.get("offers", [])
    return [o for o in data if isinstance(o, dict)] if isinstance(data, list) else []


def recommend(offers: list[dict], est_hours: float = DEFAULT_EST_HOURS,
              budget_usd: float = DEFAULT_BUDGET_USD) -> dict:
    chosen = pick_offer(offers)
    if chosen is None:
        return {"verdict": NO_SUITABLE_OFFER, "chosen_id": None, "dph": None, "est_hours": est_hours,
                "est_cost": None, "budget_usd": budget_usd, "n_offers": len(offers), "command": None,
                "reason": "no single-GPU offer in snapshot"}
    dph = float(chosen.get("dph_total", chosen.get("dph", 0.0)))
    est_cost = estimate_cost(est_hours, dph)
    verdict = WITHIN_BUDGET if est_cost <= budget_usd else OVER_BUDGET
    oid = chosen.get("id", chosen.get("ask_contract_id", "?"))
    # EMIT-ONLY: an advisory PREVIEW command for a human. Never executed by this code.
    command = (f"python scripts/launch_run16.py up --dry-run   "
               f"# offer {oid} ~${dph:.3f}/hr; est ${est_cost:.2f} for {est_hours:.0f}h; "
               f"launch_run16 auto-selects cheapest at run time -- preview with --dry-run before any real launch")
    return {"verdict": verdict, "chosen_id": oid, "dph": round(dph, 4), "est_hours": est_hours,
            "est_cost": est_cost, "budget_usd": budget_usd, "n_offers": len(offers), "command": command}


def analyze(offers: list[dict], est_hours: float = DEFAULT_EST_HOURS,
            budget_usd: float = DEFAULT_BUDGET_USD) -> dict:
    return recommend(offers, est_hours, budget_usd)
