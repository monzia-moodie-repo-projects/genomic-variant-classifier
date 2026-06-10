#!/usr/bin/env python3
"""drift_monitor_base.py -- shared BaseAgent scaffolding for drift-detector adapters.

Subclasses set `section`, and implement `_detect(dry_run)` (returns a detector
result object, or None when its detector/inputs are not configured -> the agent
reports status='awaiting_baseline') and `_summarize(result) -> dict`. The common
run()/SharedState/dry_run handling lives here so each concrete agent is a few lines.

Author: Monzia Moodie.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent


class DriftMonitorBase(BaseAgent):
    """Common run() wrapper around standalone drift detectors (not itself an agent)."""

    section: str = "drift"

    @abstractmethod
    def _detect(self, dry_run: bool) -> Optional[Any]:
        """Return a detector result, or None if inputs/baseline are not configured."""

    @abstractmethod
    def _summarize(self, result: Any) -> dict:
        """Map a detector result to a JSON-serialisable summary dict."""

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)
        result = self._detect(dry_run)
        if result is None:
            out = {
                "status": "awaiting_baseline",
                "reason": (
                    "detector and/or current inputs not configured; "
                    "agent is wired but inactive"
                ),
                "checked_at": self._now_iso(),
                "dry_run": dry_run,
            }
        else:
            out = {"status": "ok", "dry_run": dry_run}
            out.update(self._summarize(result))
        self._update_section(self.section, out)
        self._log_finish(out)
        return out
