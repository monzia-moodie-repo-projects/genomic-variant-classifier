#!/usr/bin/env python3
"""calibration_drift_monitor_agent.py -- BaseAgent adapter for CalibrationDriftAgent (Monzia Moodie)."""
from __future__ import annotations
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.calibration_drift_agent import CalibrationDriftAgent


class CalibrationDriftMonitorAgent(DriftMonitorBase):
    section = "calibration_drift"

    def __init__(self, shared_state: SharedState, *, detector: Optional[CalibrationDriftAgent] = None,
                 labeled_predictions=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._labeled = labeled_predictions

    def _detect(self, dry_run: bool):
        if self._detector is None or self._labeled is None:
            return None
        return self._detector.detect(self._labeled)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "ece_top_label": r.ece_top_label,
                "mce_top_label": r.mce_top_label, "per_class_ece": r.per_class_ece,
                "delta_ece_vs_baseline": r.delta_ece_vs_baseline, "n_samples": r.n_samples,
                "bins_used": r.bins_used, "checked_at": r.timestamp}
