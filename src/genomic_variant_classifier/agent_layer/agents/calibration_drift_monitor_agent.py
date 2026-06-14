#!/usr/bin/env python3
"""calibration_drift_monitor_agent.py -- BaseAgent adapter for CalibrationDriftAgent (Monzia Moodie)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.calibration_drift_agent import CalibrationDriftAgent

logger = logging.getLogger(__name__)


class CalibrationDriftMonitorAgent(DriftMonitorBase):
    section = "calibration_drift"

    def __init__(self, shared_state: SharedState, *, detector: Optional[CalibrationDriftAgent] = None,
                 labeled_predictions=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._labeled = labeled_predictions

    @classmethod
    def from_default_baseline(
        cls,
        shared_state: SharedState,
        *,
        labeled_predictions=None,
        baseline_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "CalibrationDriftMonitorAgent":
        """Construct an active agent from the canonical calibration baseline
        (data/reference/calibration_drift/calibration_drift_baseline.json: classes + baseline_ece,
        computed by build_calibration_baseline.py via this detector's own detect() on reference
        predictions -- a Run-17 artifact). labeled_predictions resolves arg ->
        GVC_CALIBRATION_LABELED_PREDICTIONS env (parquet) -> None (awaiting_baseline until supplied).
        Missing baseline -> inactive (graceful), never raises."""
        bp = (Path(baseline_path) if baseline_path is not None
              else Path("data/reference/calibration_drift/calibration_drift_baseline.json"))
        od = (Path(output_dir) if output_dir is not None
              else Path("outputs/drift_reports/calibration_drift"))
        lp = labeled_predictions
        if lp is None:
            _env = os.getenv("GVC_CALIBRATION_LABELED_PREDICTIONS")
            lp = _env if _env else None
        if isinstance(lp, (str, Path)):
            import pandas as pd
            _p = Path(lp)
            lp = pd.read_parquet(_p) if _p.exists() else None
        if not bp.exists():
            logger.info("CalibrationDriftMonitorAgent.from_default_baseline: baseline absent (%s); "
                        "returning inactive agent (awaiting_baseline).", bp)
            return cls(shared_state)
        detector = CalibrationDriftAgent.from_baseline(bp, output_dir=od)
        logger.info("CalibrationDriftMonitorAgent.from_default_baseline: detector loaded from %s "
                    "(labeled_predictions=%s).", bp, "set" if lp is not None else "None")
        return cls(shared_state, detector=detector, labeled_predictions=lp)

    def _detect(self, dry_run: bool):
        if self._detector is None or self._labeled is None:
            return None
        return self._detector.detect(self._labeled)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "ece_top_label": r.ece_top_label,
                "mce_top_label": r.mce_top_label, "per_class_ece": r.per_class_ece,
                "delta_ece_vs_baseline": r.delta_ece_vs_baseline, "n_samples": r.n_samples,
                "bins_used": r.bins_used, "checked_at": r.timestamp}
