#!/usr/bin/env python3
"""schema_drift_monitor_agent.py -- BaseAgent adapter for SchemaDriftAgent.

Thin subclass of DriftMonitorBase; the detector logic (detect/persist) is unchanged.
Reports status='awaiting_baseline' until a detector + current feature matrix are
supplied. Author: Monzia Moodie.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import (
    SchemaDriftAgent,
    SchemaDriftResult,
)


class SchemaDriftMonitorAgent(DriftMonitorBase):
    section = "schema_drift"

    def __init__(
        self,
        shared_state: SharedState,
        *,
        detector: Optional[SchemaDriftAgent] = None,
        matrix_path: Optional[Path] = None,
    ) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._matrix_path = Path(matrix_path) if matrix_path is not None else None

    def _load_matrix(self):
        if self._matrix_path is None or not self._matrix_path.exists():
            return None
        import pandas as pd
        return pd.read_parquet(self._matrix_path)

    def _detect(self, dry_run: bool) -> Optional[SchemaDriftResult]:
        if self._detector is None:
            return None
        df = self._load_matrix()
        if df is None:
            return None
        drift = self._detector.detect(df)
        if not dry_run:
            run_id = self._get_section(self.section).get("run_id", "adhoc")
            self._detector.persist(drift, run_id)
        return drift

    def _summarize(self, drift: SchemaDriftResult) -> dict:
        return {
            "severity": drift.severity,
            "columns_added": list(drift.columns_added),
            "columns_removed": list(drift.columns_removed),
            "columns_dtype_changed": [list(t) for t in drift.columns_dtype_changed],
            "pandera_violations": len(drift.pandera_violations),
            "observed_schema_hash": drift.observed_schema_hash,
            "expected_schema_hash": drift.expected_schema_hash,
            "checked_at": drift.timestamp,
        }
