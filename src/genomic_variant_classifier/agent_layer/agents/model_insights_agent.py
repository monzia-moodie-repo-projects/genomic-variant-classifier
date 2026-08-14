"""
model_insights_agent.py -- Monzia Moodie

Read-only per-model comparison + integrity monitor (BaseAgent). Discovers the latest run's
oof_predictions.parquet, computes per-model metrics via model_insights_detector (the SAME sklearn functions as
evaluation/evaluator.py), writes a DOCUMENTED markdown report (per-model table + MCC ranking + integrity flags
+ a pointer to docs/METRICS.md), records the summary to SharedState ('model_insights'), and -- ONLY when a
serious integrity flag fires (leakage-suspicion, gene-disjoint violation, or a degenerate OOF column) -- emits
an informational FEATURE_INSTABILITY message to TrainingLifecycleAgent so a suspect model is not promoted
unchecked.

GUARDRAIL: this agent reports diagnostics + flags only; it NEVER tunes or recommends hyperparameters, and it
deliberately ranks by MCC rather than AUROC so a leaky near-perfect AUROC is a SUSPICION, not a trophy. No run
artifacts -> returns the 'skipped' contract (mirrors InterpretabilityAgent 'not due'). `outputs_root` and `root`
are injectable so the agent is hermetic under test; the orchestrator constructs it as cls(shared_state).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.agent_layer.message_bus import FEATURE_INSTABILITY, PRIORITY_HIGH
from genomic_variant_classifier.evaluation import model_insights_detector as D
from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT

_RECIPIENT = "TrainingLifecycleAgent"
_SECTION = "model_insights"
_SERIOUS = ("LEAKAGE_SUSPICION", "GENE_DISJOINT_VIOLATION", "DEGENERATE_OOF")


def _flag_kind(flag: str) -> str:
    # "LEAKAGE_SUSPICION[strong]: ..." / "GENE_DISJOINT_VIOLATION: ..." -> the leading TOKEN
    return flag.split("[", 1)[0].split(":", 1)[0]


class ModelInsightsAgent(BaseAgent):
    def __init__(self, shared_state, outputs_root: str = "outputs",
                 root: str | None = None) -> None:
        super().__init__(shared_state)
        self._outputs_root = outputs_root
        self._root = root if root is not None else str(PROJECT_ROOT)

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)

        run_dir = D.discover_latest_run(self._outputs_root)
        if run_dir is None:
            result = {"action": "skipped", "reason": "no_run_artifacts"}
            self._log_finish(result)
            return result

        oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
        analysis = D.analyze(oof)
        report_path = self._write_report(run_dir, analysis)

        metrics = [
            {"model": m.model, "auroc": m.auroc, "auprc": m.auprc, "mcc": m.mcc,
             "brier": m.brier, "n": m.n, "n_pos": m.n_pos, "note": m.note}
            for m in analysis["metrics"]
        ]
        self._update_section(_SECTION, {
            "run_dir": str(run_dir),
            "analyzed_at": self._now_iso(),
            "ranking_by_mcc": analysis["ranking_by_mcc"],
            "flags": analysis["flags"],
            "gene_disjoint": analysis["gene_disjoint"],
            "metrics": metrics,
            "report": str(report_path),
        })

        serious = [f for f in analysis["flags"] if _flag_kind(f) in _SERIOUS]
        if serious and not dry_run:
            self.send_message(
                to=_RECIPIENT, subject=FEATURE_INSTABILITY,
                payload={"run_dir": str(run_dir), "flags": serious,
                         "report": str(report_path), "detected_at": self._now_iso()},
                priority=PRIORITY_HIGH, requires_approval=False,  # informational
            )
        elif serious:
            self.logger.info("  [dry-run] would emit FEATURE_INSTABILITY for %d integrity flag(s)", len(serious))

        result = {
            "action": "model_insights_scan",
            "run_dir": str(run_dir),
            "models": len(metrics),
            "flags": len(analysis["flags"]),
            "serious_flags": len(serious),
            "ranking_by_mcc": analysis["ranking_by_mcc"],
            "report": str(report_path),
        }
        self._log_finish(result)
        return result

    def _write_report(self, run_dir: Path, analysis: dict) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = Path(self._root) / "reports" / "model_insights" / f"INSIGHTS_{ts}.md"
        out.parent.mkdir(parents=True, exist_ok=True)

        def fmt(x):
            return "--" if x is None else f"{x:.4f}"

        lines = [
            f"# Model-insights report -- {ts}", "",
            f"Source run: `{run_dir}`", "",
            "Metric definitions: see `docs/METRICS.md` (AUROC, AUPRC, MCC, Brier).", "",
            "## Per-model metrics (computed from oof_predictions.parquet)", "",
            "| model | AUROC | AUPRC | MCC | Brier | n | n_pos |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for m in analysis["metrics"]:
            lines.append(f"| {m.model} | {fmt(m.auroc)} | {fmt(m.auprc)} | {fmt(m.mcc)} | "
                         f"{fmt(m.brier)} | {m.n} | {m.n_pos} |")
        lines += [
            "", "## Ranking (by MCC -- balanced, NOT AUROC)", "",
            ", ".join(analysis["ranking_by_mcc"]) or "(none rankable)",
            "", "## Integrity flags", "",
        ]
        if analysis["flags"]:
            lines += [f"- {f}" for f in analysis["flags"]]
        else:
            lines.append("- none")
        lines += ["", f"Gene-disjoint folds: {analysis['gene_disjoint']} -- {analysis['gene_disjoint_msg']}", ""]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
