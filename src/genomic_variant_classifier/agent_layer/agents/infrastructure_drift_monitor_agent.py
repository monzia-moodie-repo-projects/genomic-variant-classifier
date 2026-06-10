#!/usr/bin/env python3
"""infrastructure_drift_monitor_agent.py -- BaseAgent adapter for InfrastructureDriftAgent (Monzia Moodie)."""
from __future__ import annotations
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.infrastructure_drift_agent import InfrastructureDriftAgent


class InfrastructureDriftMonitorAgent(DriftMonitorBase):
    section = "infrastructure_drift"

    def __init__(self, shared_state: SharedState, *, detector: Optional[InfrastructureDriftAgent] = None,
                 current_packages=None, current_dag_spec=None, replayed_features=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._args = (current_packages, current_dag_spec, replayed_features)

    def _detect(self, dry_run: bool):
        if self._detector is None or any(a is None for a in self._args):
            return None
        return self._detector.detect(*self._args)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity,
                "package_changes": {k: list(v) for k, v in r.package_changes.items()},
                "dag_hash_changed": r.dag_hash_changed,
                "golden_set_divergence": r.golden_set_divergence, "checked_at": r.timestamp}
