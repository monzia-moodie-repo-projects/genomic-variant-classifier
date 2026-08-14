"""
database_freshness_monitor_agent.py -- Monzia Moodie

Registry-driven data-freshness monitor (BaseAgent). Supersedes DataFreshnessAgent's 4 hardcoded polls: it
iterates the WHOLE monitoring.registry, records per-source upstream + local status to SharedState
('database_freshness'), writes a DOCUMENTED markdown report, HITL-gates re-acquisition (records the source's
declared `acquire` path -- NO gcloud/dataproc against the deleted GCP project), and emits DATA_UPDATED to
TrainingLifecycleAgent for each changed source. `probe` and `root` are injectable so the agent is hermetic
under test; the orchestrator constructs it as cls(shared_state) (defaults -> real probe, repo root).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.agent_layer.agents import database_freshness_detector as D
from genomic_variant_classifier.agent_layer.message_bus import DATA_UPDATED, PRIORITY_HIGH
from genomic_variant_classifier.monitoring import registry as R
from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT

_RECIPIENT = "TrainingLifecycleAgent"
_SECTION = "database_freshness"


class DatabaseFreshnessMonitorAgent(BaseAgent):
    def __init__(self, shared_state, probe=None, root: str | None = None) -> None:
        super().__init__(shared_state)
        self._probe = probe or D._default_probe
        self._root = root if root is not None else str(PROJECT_ROOT)

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)
        prior = self._get_section(_SECTION)
        report = D.scan(prior, root=self._root, probe=self._probe)

        updates: dict[str, dict] = {}
        for u in report["upstream"]:
            entry = dict(prior.get(u.key, {}))
            entry["last_checked"] = self._now_iso()
            entry["status"] = u.status
            if u.status == D.CHANGED and u.current is not None:
                entry["last_seen"] = u.current
            updates[u.key] = entry
        self._update_section(_SECTION, updates)

        report_path = self._write_report(report)
        changes = report["changes"]
        approved: list[str] = []
        for u in changes:
            src = R.by_key(u.key)
            if dry_run:
                self.logger.info("  [dry-run] would request approval to re-acquire %s via: %s",
                                 src.key, src.acquire)
                continue
            if self._require_approval(
                f"Re-acquire {src.name} ({u.previous} -> {u.current})? operator step: {src.acquire}",
                dry_run=False,
            ):
                approved.append(src.key)
                self.logger.info("  Re-acquisition approved for %s. Operator step: %s", src.key, src.acquire)
            # notify regardless of approval -- TrainingLifecycle decides independently
            self.send_message(
                to=_RECIPIENT, subject=DATA_UPDATED,
                payload={"source": src.key, "previous": u.previous, "current": u.current,
                         "detected_at": self._now_iso(), "acquire": src.acquire},
                priority=PRIORITY_HIGH,
            )

        result = {
            "action": "registry_freshness_scan",
            "sources": len(report["upstream"]),
            "changes_detected": len(changes),
            "approved_reacquire": approved,
            "report": str(report_path),
        }
        self._log_finish(result)
        return result

    def _write_report(self, report: dict) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = Path(self._root) / "reports" / "data_freshness" / f"FRESHNESS_{ts}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Data-source freshness report -- {ts}", "",
            f"{len(report['upstream'])} sources scanned; {len(report['changes'])} upstream change(s) detected.",
            "", "## Upstream", "",
        ]
        for u in report["upstream"]:
            lines.append(f"- **{u.key}** [{u.status}] {u.detail}")
        lines += ["", "## Local assets", ""]
        for l in report["local"]:
            sz = "" if l.size_bytes is None else f" ({l.size_bytes / 1e6:.1f} MB)"
            lines.append(f"- **{l.key}** [{l.status}]{sz} {l.detail}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
