#!/usr/bin/env python3
"""fairness_subgroup_monitor_agent.py -- BaseAgent adapter for FairnessSubgroupAgent (Monzia Moodie)."""
from __future__ import annotations
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.fairness_subgroup_agent import FairnessSubgroupAgent


class FairnessSubgroupMonitorAgent(DriftMonitorBase):
    section = "fairness_subgroup"

    def __init__(self, shared_state: SharedState, *, detector: Optional[FairnessSubgroupAgent] = None,
                 predictions=None, axes=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._predictions = predictions
        self._axes = axes

    def _detect(self, dry_run: bool):
        if self._detector is None or self._predictions is None or self._axes is None:
            return None
        return self._detector.detect(self._predictions, self._axes)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "max_eod": r.max_eod, "max_dpd_change": r.max_dpd_change,
                "high_priority_strata_flags": list(r.high_priority_strata_flags),
                "n_strata_metrics": len(r.metrics), "checked_at": r.timestamp}
