#!/usr/bin/env python3
"""adversarial_submission_monitor_agent.py -- BaseAgent adapter for AdversarialSubmissionAgent (Monzia Moodie)."""
from __future__ import annotations
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.adversarial_submission_agent import AdversarialSubmissionAgent


class AdversarialSubmissionMonitorAgent(DriftMonitorBase):
    section = "adversarial_submission"

    def __init__(self, shared_state: SharedState, *, detector: Optional[AdversarialSubmissionAgent] = None,
                 weekly_submissions=None, submitter_baseline=None,
                 aggregate_classifications=None, submitter_metadata=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._args = (weekly_submissions, submitter_baseline, aggregate_classifications, submitter_metadata)

    def _detect(self, dry_run: bool):
        if self._detector is None or any(a is None for a in self._args):
            return None
        return self._detector.detect(*self._args)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "n_findings": len(r.findings),
                "quarantine_submitter_ids": list(r.quarantine_submitter_ids), "checked_at": r.timestamp}
