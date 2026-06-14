#!/usr/bin/env python3
"""feature_coverage_sentinel_monitor_agent.py -- BaseAgent adapter for FeatureCoverageSentinelAgent (Monzia Moodie)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.feature_coverage_sentinel_agent import (
    FeatureCoverageSentinelAgent,
)

logger = logging.getLogger(__name__)


class FeatureCoverageSentinelMonitorAgent(DriftMonitorBase):
    section = "feature_coverage"

    def __init__(self, shared_state: SharedState, *,
                 detector: Optional[FeatureCoverageSentinelAgent] = None,
                 current_matrix=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._matrix = current_matrix

    def _detect(self, dry_run: bool):
        if self._detector is None or self._matrix is None:
            return None
        return self._detector.detect(self._matrix)

    def _summarize(self, r) -> dict:
        return {
            "severity": r.severity,
            "regressed": [list(t) for t in r.regressed],
            "dropped": list(r.dropped),
            "recovered": list(r.recovered),
            "still_degenerate": list(r.still_degenerate),
            "new_columns": list(r.new_columns),
            "n_regressed": len(r.regressed),
            "n_dropped": len(r.dropped),
            "checked_at": r.timestamp,
        }

    @classmethod
    def from_default_baseline(cls, shared_state: SharedState, *, current_matrix=None,
                              reference_path: Optional[Path] = None,
                              output_dir: Optional[Path] = None) -> "FeatureCoverageSentinelMonitorAgent":
        """Construct an active agent from the canonical reference
        (data/reference/feature_coverage/feature_coverage_baseline.json, produced by
        build_feature_coverage_baseline.py). current_matrix resolves arg -> GVC_FEATURE_MATRIX
        env (parquet) -> None (awaiting_baseline until supplied). Missing reference -> inactive
        agent (graceful), never raises. Routed by the orchestrator from_default_baseline hook."""
        bp = (Path(reference_path) if reference_path is not None
              else Path("data/reference/feature_coverage/feature_coverage_baseline.json"))
        od = (Path(output_dir) if output_dir is not None
              else Path("outputs/drift_reports/feature_coverage"))
        if not bp.exists():
            logger.info("FeatureCoverageSentinelMonitorAgent.from_default_baseline: reference absent "
                        "(%s); returning inactive agent (awaiting_baseline).", bp)
            return cls(shared_state)
        detector = FeatureCoverageSentinelAgent.from_reference(bp, output_dir=od)
        if current_matrix is None:
            env = os.getenv("GVC_FEATURE_MATRIX")
            if env:
                import pandas as pd
                p = Path(env)
                current_matrix = pd.read_parquet(p) if p.exists() else None
        logger.info("FeatureCoverageSentinelMonitorAgent.from_default_baseline: reference loaded from %s "
                    "(%d cols, ncf=%s); matrix=%s.", bp, len(detector.reference),
                    detector.near_constant_frac, "set" if current_matrix is not None else "None")
        return cls(shared_state, detector=detector, current_matrix=current_matrix)
