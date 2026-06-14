"""
data_readiness_agent.py -- Monzia Moodie

Pre-run DATA-READINESS gate (BaseAgent). Aggregates read-only checks into ONE advisory GO / GO_WITH_WARNINGS /
NO_GO verdict and a documented report, records it to SharedState ('data_readiness'), and -- on NO_GO -- opens a
HITL override gate (the operator must explicitly approve proceeding). It VERIFIES the data-prep outputs; it
never runs real_data_prep.py / smoke_all_models.py / preflight_gate.py or mutates anything (no silent mutation).
It complements, and does not duplicate, scripts/preflight_gate.py (which validates the launch COMMAND): this
agent checks DATA/ENVIRONMENT readiness.

Dimensions:
  1. Critical-asset presence -- every registry.critical_assets() path must exist + be non-empty.
  2. Feature health (optional) -- if a splits parquet is discoverable, every non-meta column is scored with the
     data.feature_health.col_health library (same as audit_split_feature_health.py).

`root` and `splits_path` are injectable so the agent is hermetic under test; the orchestrator constructs it as
cls(shared_state).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.evaluation import data_readiness_detector as D
from genomic_variant_classifier.monitoring import registry as R

_SECTION = "data_readiness"
_META_COLS = {"variant_id", "gene_symbol", "fold", "label", "chrom", "pos", "ref", "alt", "consequence"}


class DataReadinessAgent(BaseAgent):
    def __init__(self, shared_state, root: str = ".", splits_path: str | None = None) -> None:
        super().__init__(shared_state)
        self._root = root
        self._splits_path = splits_path

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)

        asset_paths = R.critical_assets()
        feature_df, feature_cols, splits_src = self._load_splits()
        analysis = D.analyze(asset_paths, root=self._root,
                             feature_df=feature_df, feature_cols=feature_cols)
        report_path = self._write_report(analysis, splits_src)

        missing = [a.path for a in analysis["assets"] if not a.present]
        health = analysis["health"]
        self._update_section(_SECTION, {
            "checked_at": self._now_iso(),
            "verdict": analysis["verdict"],
            "reasons": analysis["reasons"],
            "n_assets": len(analysis["assets"]),
            "missing_assets": missing,
            "splits_source": splits_src,
            "n_feature_cols": None if health is None else health["n_cols"],
            "n_degenerate": None if health is None else health["n_degenerate"],
            "report": str(report_path),
        })

        if analysis["verdict"] == D.NO_GO and not dry_run:
            self._require_approval(
                f"Data readiness is NO_GO: {analysis['reasons']}. "
                f"Remediate (re-acquire missing assets / regenerate splits) before launch. "
                f"Approve override to proceed anyway?",
                dry_run=False,
            )
        elif analysis["verdict"] == D.NO_GO:
            self.logger.info("  [dry-run] would open a HITL override gate (verdict=NO_GO)")

        result = {
            "action": "data_readiness_gate",
            "verdict": analysis["verdict"],
            "n_assets": len(analysis["assets"]),
            "missing_assets": missing,
            "n_degenerate": None if health is None else health["n_degenerate"],
            "report": str(report_path),
        }
        self._log_finish(result)
        return result

    def _load_splits(self):
        """Return (df, feature_cols, source_path) or (None, None, None). Defensive: any failure -> skip health."""
        path = None
        if self._splits_path is not None:
            cand = Path(self._splits_path)
            if cand.is_file():
                path = cand
        else:
            globs = list(Path(self._root).glob("outputs/*/*/splits/*.parquet")) \
                + list(Path(self._root).glob("outputs/*/splits/*.parquet"))
            if globs:
                path = max(globs, key=lambda p: p.stat().st_mtime)
        if path is None:
            return None, None, None
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # corrupt / unreadable -> skip the dimension, do not crash the gate
            self.logger.info("  splits parquet unreadable (%s) -- feature-health dimension skipped", exc)
            return None, None, None
        feature_cols = [c for c in df.columns if c not in _META_COLS]
        return df, feature_cols, str(path)

    def _write_report(self, analysis: dict, splits_src: str | None) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = Path(self._root) / "reports" / "data_readiness" / f"READINESS_{ts}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Pre-run data-readiness gate -- {ts}", "",
            f"## VERDICT: {analysis['verdict']}", "",
        ]
        lines += [f"- {r}" for r in analysis["reasons"]]
        lines += ["", "## Critical assets", ""]
        for a in analysis["assets"]:
            sz = "" if a.size_bytes is None else f" ({a.size_bytes / 1e6:.1f} MB)"
            mark = "OK" if a.present else "MISSING"
            lines.append(f"- [{mark}] {a.path}{sz} -- {a.detail}")
        health = analysis["health"]
        lines += ["", "## Feature health", ""]
        if health is None:
            lines.append(f"- not evaluated (no splits parquet found{'' if splits_src is None else f' at {splits_src}'})")
        else:
            lines.append(f"- source: {splits_src}")
            lines.append(f"- {health['n_healthy']}/{health['n_cols']} healthy; {health['n_degenerate']} degenerate")
            for col, why in list(health["degenerate"].items())[:20]:
                lines.append(f"  - {col}: {why}")
        lines += ["", "Verify-only: this gate never runs data-prep; remediate via the source `acquire` paths "
                  "(registry) and regenerate splits, then re-run the gate.", ""]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
