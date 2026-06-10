#!/usr/bin/env python3
"""annotation_policy_monitor_agent.py -- BaseAgent adapter for AnnotationPolicyAgent (Monzia Moodie)."""
from __future__ import annotations
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.annotation_policy_agent import AnnotationPolicyAgent


class AnnotationPolicyMonitorAgent(DriftMonitorBase):
    section = "annotation_policy"

    def __init__(self, shared_state: SharedState, *, detector: Optional[AnnotationPolicyAgent] = None,
                 new_svi_pubs=None, clinvar_status_changes=None,
                 submitter_history=None, n_inference_variants=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._args = (new_svi_pubs, clinvar_status_changes, submitter_history, n_inference_variants)

    def _detect(self, dry_run: bool):
        if self._detector is None or any(a is None for a in self._args):
            return None
        return self._detector.detect(*self._args)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "new_svi_publications": list(r.new_svi_publications),
                "pct_variants_with_review_status_change": r.pct_variants_with_review_status_change,
                "submitters_with_outlier_alarm": list(r.submitters_with_outlier_alarm),
                "requires_variant_scientist_review": r.requires_variant_scientist_review,
                "checked_at": r.timestamp}
