#!/usr/bin/env python3
"""schema_drift_monitor_agent.py -- BaseAgent adapter for SchemaDriftAgent.

Thin subclass of DriftMonitorBase; the detector logic (detect/persist) is unchanged.
Reports status='awaiting_baseline' until a detector + current feature matrix are
supplied. Author: Monzia Moodie.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import (
    SchemaDriftAgent,
    SchemaDriftResult,
)

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_default_baseline(
        cls,
        shared_state: SharedState,
        *,
        matrix_path: Optional[Path] = None,
        baseline_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "SchemaDriftMonitorAgent":
        """Construct an active agent from the canonical schema baseline.

        Loads the SchemaDriftAgent detector from the versioned baseline
        (data/reference/schema/schema_baseline.json by default) so the agent is no
        longer awaiting_baseline. The current feature matrix to validate is taken
        from matrix_path, else the GVC_SCHEMA_CURRENT_MATRIX env var, else left None
        (the agent reports awaiting_baseline until a matrix is supplied at run time).
        A missing baseline file yields an inactive agent (graceful) rather than raising.
        """
        bp = (
            Path(baseline_path)
            if baseline_path is not None
            else Path("data/reference/schema/schema_baseline.json")
        )
        od = (
            Path(output_dir)
            if output_dir is not None
            else Path("outputs/drift_reports/schema")
        )
        if matrix_path is None:
            _env_mx = os.getenv("GVC_SCHEMA_CURRENT_MATRIX")
            matrix_path = Path(_env_mx) if _env_mx else None
        if not bp.exists():
            logger.info(
                "SchemaDriftMonitorAgent.from_default_baseline: baseline absent (%s); "
                "returning inactive agent (awaiting_baseline).",
                bp,
            )
            return cls(shared_state)
        detector = SchemaDriftAgent.from_baseline(bp, output_dir=od)
        logger.info(
            "SchemaDriftMonitorAgent.from_default_baseline: detector loaded from %s "
            "(matrix=%s).",
            bp,
            matrix_path,
        )
        return cls(shared_state, detector=detector, matrix_path=matrix_path)

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
