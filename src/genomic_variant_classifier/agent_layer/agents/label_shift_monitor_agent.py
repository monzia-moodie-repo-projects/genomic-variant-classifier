#!/usr/bin/env python3
"""label_shift_monitor_agent.py -- BaseAgent adapter for LabelShiftAgent (Monzia Moodie)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.label_shift_agent import LabelShiftAgent

logger = logging.getLogger(__name__)


class LabelShiftMonitorAgent(DriftMonitorBase):
    section = "label_shift"

    def __init__(self, shared_state: SharedState, *, detector: Optional[LabelShiftAgent] = None,
                 prediction_log=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._prediction_log = prediction_log

    @classmethod
    def from_default_baseline(
        cls,
        shared_state: SharedState,
        *,
        prediction_log=None,
        baseline_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "LabelShiftMonitorAgent":
        """Construct an active agent from the canonical label-shift baseline.

        Loads the LabelShiftAgent detector from
        data/reference/label_shift/label_shift_baseline.json (classes + p_train +
        reference_confusion; produced by build_label_shift_baseline.py from a model's
        validation predictions -- a Run-17 artifact). The production window is taken from
        prediction_log (a DataFrame with a 'predicted_class' column), else a parquet at
        GVC_LABEL_SHIFT_PREDICTION_LOG, else None (the agent reports awaiting_baseline
        until a prediction log is supplied). A missing baseline file returns an inactive
        agent (graceful), never raises.
        """
        bp = (
            Path(baseline_path)
            if baseline_path is not None
            else Path("data/reference/label_shift/label_shift_baseline.json")
        )
        od = (
            Path(output_dir)
            if output_dir is not None
            else Path("outputs/drift_reports/label_shift")
        )
        pl = prediction_log
        if pl is None:
            _env = os.getenv("GVC_LABEL_SHIFT_PREDICTION_LOG")
            pl = _env if _env else None
        if isinstance(pl, (str, Path)):
            import pandas as pd

            _p = Path(pl)
            pl = pd.read_parquet(_p) if _p.exists() else None
        if not bp.exists():
            logger.info(
                "LabelShiftMonitorAgent.from_default_baseline: baseline absent (%s); "
                "returning inactive agent (awaiting_baseline).",
                bp,
            )
            return cls(shared_state)
        detector = LabelShiftAgent.from_baseline(bp, output_dir=od)
        logger.info(
            "LabelShiftMonitorAgent.from_default_baseline: detector loaded from %s "
            "(prediction_log=%s).",
            bp,
            "set" if pl is not None else "None",
        )
        return cls(shared_state, detector=detector, prediction_log=pl)

    def _detect(self, dry_run: bool):
        if self._detector is None or self._prediction_log is None:
            return None
        return self._detector.detect(self._prediction_log)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "max_abs_class_shift": r.max_abs_class_shift,
                "chi_square_p_value": r.chi_square_p_value, "rlls_used": r.rlls_used,
                "n_predictions": r.n_predictions, "estimated_p_test": r.estimated_p_test,
                "checked_at": r.timestamp}
