#!/usr/bin/env python3
"""label_shift_monitor_agent.py -- BaseAgent adapter for LabelShiftAgent (Monzia Moodie)."""
from __future__ import annotations
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.label_shift_agent import LabelShiftAgent


class LabelShiftMonitorAgent(DriftMonitorBase):
    section = "label_shift"

    def __init__(self, shared_state: SharedState, *, detector: Optional[LabelShiftAgent] = None,
                 prediction_log=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._prediction_log = prediction_log

    def _detect(self, dry_run: bool):
        if self._detector is None or self._prediction_log is None:
            return None
        return self._detector.detect(self._prediction_log)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "max_abs_class_shift": r.max_abs_class_shift,
                "chi_square_p_value": r.chi_square_p_value, "rlls_used": r.rlls_used,
                "n_predictions": r.n_predictions, "estimated_p_test": r.estimated_p_test,
                "checked_at": r.timestamp}
