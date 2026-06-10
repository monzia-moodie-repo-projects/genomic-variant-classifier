#!/usr/bin/env python3
"""concept_drift_monitor_agent.py -- BaseAgent adapter for ConceptDriftAgent (Monzia Moodie)."""
from __future__ import annotations
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.concept_drift_agent import ConceptDriftAgent


class ConceptDriftMonitorAgent(DriftMonitorBase):
    section = "concept_drift"

    def __init__(self, shared_state: SharedState, *, detector: Optional[ConceptDriftAgent] = None,
                 cbpe_estimated_auroc: Optional[float] = None, bbse_pvalue: Optional[float] = None,
                 n_samples: Optional[int] = None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._args = (cbpe_estimated_auroc, bbse_pvalue, n_samples)

    def _detect(self, dry_run: bool):
        if self._detector is None or any(a is None for a in self._args):
            return None
        return self._detector.detect(*self._args)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "cbpe_estimated_auroc": r.cbpe_estimated_auroc,
                "cbpe_baseline_auroc": r.cbpe_baseline_auroc, "cbpe_drop": r.cbpe_drop,
                "bbse_pvalue": r.bbse_pvalue, "likely_pure_concept": r.likely_pure_concept,
                "n_samples": r.n_samples, "checked_at": r.timestamp}
