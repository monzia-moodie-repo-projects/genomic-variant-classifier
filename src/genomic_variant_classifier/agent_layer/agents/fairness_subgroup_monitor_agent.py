#!/usr/bin/env python3
"""fairness_subgroup_monitor_agent.py -- BaseAgent adapter for FairnessSubgroupAgent (Monzia Moodie)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.fairness_subgroup_agent import FairnessSubgroupAgent

logger = logging.getLogger(__name__)


class FairnessSubgroupMonitorAgent(DriftMonitorBase):
    section = "fairness_subgroup"

    def __init__(self, shared_state: SharedState, *, detector: Optional[FairnessSubgroupAgent] = None,
                 predictions=None, axes=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._predictions = predictions
        self._axes = axes

    @classmethod
    def from_default_baseline(
        cls,
        shared_state: SharedState,
        *,
        predictions=None,
        axes=None,
        baseline_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "FairnessSubgroupMonitorAgent":
        """Construct an active agent from the canonical fairness baseline
        (data/reference/fairness_subgroup/fairness_subgroup_baseline.json: classes +
        p_train_per_stratum, from a model's reference predictions -- a Run-17 artifact). predictions
        resolves arg -> GVC_FAIRNESS_PREDICTIONS env (parquet); axes resolves arg -> GVC_FAIRNESS_AXES
        env (a JSON object axis_name -> column). Either None -> awaiting_baseline. Missing baseline ->
        inactive (graceful). NOTE: activates the EXISTING detector; the per-stratum AUROC proxy and the
        max_dpd_change=0.0 stub remain (PHASE_2_FEATURES; see fairness_subgroup_agent.py)."""
        bp = (Path(baseline_path) if baseline_path is not None
              else Path("data/reference/fairness_subgroup/fairness_subgroup_baseline.json"))
        od = (Path(output_dir) if output_dir is not None
              else Path("outputs/drift_reports/fairness_subgroup"))
        p = predictions
        if p is None:
            _env = os.getenv("GVC_FAIRNESS_PREDICTIONS")
            p = _env if _env else None
        if isinstance(p, (str, Path)):
            import pandas as pd
            _p = Path(p)
            p = pd.read_parquet(_p) if _p.exists() else None
        if axes is None:
            _env = os.getenv("GVC_FAIRNESS_AXES")
            if _env:
                import json
                axes = json.loads(_env)
        if not bp.exists():
            logger.info("FairnessSubgroupMonitorAgent.from_default_baseline: baseline absent (%s); "
                        "returning inactive agent (awaiting_baseline).", bp)
            return cls(shared_state)
        detector = FairnessSubgroupAgent.from_baseline(bp, output_dir=od)
        logger.info("FairnessSubgroupMonitorAgent.from_default_baseline: detector loaded from %s "
                    "(predictions=%s, axes=%s).", bp, "set" if p is not None else "None",
                    "set" if axes else "None")
        return cls(shared_state, detector=detector, predictions=p, axes=axes)

    def _detect(self, dry_run: bool):
        if self._detector is None or self._predictions is None or self._axes is None:
            return None
        return self._detector.detect(self._predictions, self._axes)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "max_eod": r.max_eod, "max_dpd_change": r.max_dpd_change,
                "high_priority_strata_flags": list(r.high_priority_strata_flags),
                "n_strata_metrics": len(r.metrics), "checked_at": r.timestamp}
