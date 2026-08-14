"""
finops_advisor_agent.py -- Monzia Moodie

RECOMMEND-ONLY FinOps advisor (BaseAgent). Reads an offers SNAPSHOT file (a `vastai search offers --raw` dump),
selects the cheapest suitable single-GPU offer, estimates the run cost, checks it against a budget cap, writes a
report, records a 'finops' SharedState section, and -- on an over-budget or no-suitable-offer recommendation --
opens a HITL review item. It NEVER calls vastai, NEVER provisions, NEVER spends, and makes NO live account calls:
no snapshot -> 'skipped'. Autonomous provisioning is a deliberate NON-GOAL (see docs/design/GPU_FINOPS_DESIGN.md);
the emitted command is an advisory --dry-run PREVIEW string for a human to run, mirroring preflight_gate.py --emit.

`snapshot_path`, `est_hours`, `budget_usd`, `root` are injectable for hermetic tests; the orchestrator constructs
it as cls(shared_state) (snapshot_path defaults to None -> skipped until a snapshot is provided).
"""
from __future__ import annotations

from pathlib import Path

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.evaluation import finops_detector as D
from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT

_SECTION = "finops"


class FinOpsAdvisorAgent(BaseAgent):
    def __init__(self, shared_state, snapshot_path: str | None = None,
                 est_hours: float = D.DEFAULT_EST_HOURS, budget_usd: float = D.DEFAULT_BUDGET_USD,
                 root: str | None = None) -> None:
        super().__init__(shared_state)
        self._snapshot_path = snapshot_path
        self._est_hours = est_hours
        self._budget_usd = budget_usd
        self._root = root if root is not None else str(PROJECT_ROOT)

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)

        path = self._snapshot_path
        if not path or not Path(path).exists():
            result = {"action": "skipped",
                      "reason": "no offers snapshot (provide a 'vastai search offers --raw' JSON dump)"}
            self._log_finish(result)
            return result
        try:
            offers = D.load_offers_snapshot(path)
        except Exception as exc:  # unreadable/corrupt snapshot -> skip, never crash, never call vastai
            result = {"action": "skipped", "reason": f"unreadable offers snapshot: {exc}"}
            self._log_finish(result)
            return result

        rec = D.recommend(offers, self._est_hours, self._budget_usd)
        report_path = self._write_report(rec)

        self._update_section(_SECTION, {
            "last_run": self._now_iso(),
            "verdict": rec["verdict"],
            "chosen_id": rec.get("chosen_id"),
            "dph": rec.get("dph"),
            "est_cost": rec.get("est_cost"),
            "budget_usd": rec.get("budget_usd"),
            "command": rec.get("command"),
            "report": str(report_path),
        })

        # HITL: an over-budget / no-offer recommendation needs a human decision. No spend either way.
        if rec["verdict"] != D.WITHIN_BUDGET and not dry_run:
            self._state.add_review_item(
                f"FinOps: {rec['verdict']} -- est ${rec.get('est_cost')} vs budget ${rec.get('budget_usd')}. "
                f"Review before launching. Recommended preview: {rec.get('command')}")

        result = {"action": "finops_recommendation", "verdict": rec["verdict"],
                  "chosen_id": rec.get("chosen_id"), "est_cost": rec.get("est_cost"),
                  "report": str(report_path)}
        self._log_finish(result)
        return result

    def _write_report(self, rec: dict) -> Path:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = Path(self._root) / "reports" / "finops" / f"FINOPS_{ts}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# FinOps recommendation -- {ts}", "",
            f"## VERDICT: {rec['verdict']}", "",
            f"- Offers in snapshot: {rec['n_offers']}",
            f"- Chosen offer id: {rec.get('chosen_id')}",
            f"- Rate: ${rec['dph']:.4f}/hr" if rec.get("dph") is not None else "- Rate: --",
            f"- Estimated run: {rec['est_hours']:.0f} h",
            f"- Estimated cost: ${rec['est_cost']:.2f}" if rec.get("est_cost") is not None else "- Estimated cost: --",
            f"- Budget cap: ${rec['budget_usd']:.2f}",
            "",
            "## Recommended (preview only -- NOT executed; no spend)", "",
            f"```\n{rec.get('command') or '(none -- no suitable offer)'}\n```", "",
            "This agent never calls vastai, never provisions, never spends. Autonomous provisioning is a "
            "deliberate non-goal (docs/design/GPU_FINOPS_DESIGN.md). Run the preview command yourself to launch.",
            "",
        ]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
