#!/usr/bin/env python3
"""reclassification_sentinel_monitor_agent.py -- BaseAgent adapter for ReclassificationSentinelAgent (Monzia Moodie)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.reclassification_sentinel_agent import (
    ReclassificationSentinelAgent,
)

logger = logging.getLogger(__name__)


class ReclassificationSentinelMonitorAgent(DriftMonitorBase):
    section = "reclassification"

    def __init__(self, shared_state: SharedState, *,
                 detector: Optional[ReclassificationSentinelAgent] = None,
                 old_path=None, new_path=None,
                 old_release: str = "previous", new_release: str = "current") -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._old_path = old_path
        self._new_path = new_path
        self._old_release = old_release
        self._new_release = new_release

    @classmethod
    def from_default_baseline(cls, shared_state: SharedState, *, old_path=None, new_path=None,
                              old_release: Optional[str] = None, new_release: Optional[str] = None,
                              reference_path: Optional[Path] = None,
                              output_dir: Optional[Path] = None) -> "ReclassificationSentinelMonitorAgent":
        """Construct an active agent from the canonical split-membership reference
        (data/reference/reclassification/reclassification_reference.parquet -- a compact (variant_id, split)
        parquet produced by build_reclassification_reference.py from the committed splits). The run-time
        inputs are the OLD (training-era) and NEW (current) ClinVar release parquets: old_path/new_path
        resolve arg -> GVC_RECLASS_OLD_RELEASE / GVC_RECLASS_NEW_RELEASE env -> None; a set-but-missing file
        is treated as None (awaiting_baseline). old_release/new_release labels resolve arg ->
        GVC_RECLASS_OLD_LABEL / GVC_RECLASS_NEW_LABEL -> 'previous'/'current'. Missing reference -> inactive
        (graceful), never raises. Routed by the orchestrator from_default_baseline hook."""
        rp = (Path(reference_path) if reference_path is not None
              else Path("data/reference/reclassification/reclassification_reference.parquet"))
        od = (Path(output_dir) if output_dir is not None
              else Path("outputs/drift_reports/reclassification"))
        if old_path is None:
            old_path = os.getenv("GVC_RECLASS_OLD_RELEASE") or None
        if new_path is None:
            new_path = os.getenv("GVC_RECLASS_NEW_RELEASE") or None
        if old_path and not Path(old_path).exists():
            logger.info("ReclassificationSentinelMonitorAgent: old_path set but absent (%s) -> None.", old_path)
            old_path = None
        if new_path and not Path(new_path).exists():
            logger.info("ReclassificationSentinelMonitorAgent: new_path set but absent (%s) -> None.", new_path)
            new_path = None
        if old_release is None:
            old_release = os.getenv("GVC_RECLASS_OLD_LABEL") or "previous"
        if new_release is None:
            new_release = os.getenv("GVC_RECLASS_NEW_LABEL") or "current"
        if not rp.exists():
            logger.info("ReclassificationSentinelMonitorAgent.from_default_baseline: reference absent (%s); "
                        "returning inactive agent (awaiting_baseline).", rp)
            return cls(shared_state)
        detector = ReclassificationSentinelAgent.from_reference(rp, output_dir=od)
        logger.info("ReclassificationSentinelMonitorAgent.from_default_baseline: reference loaded from %s "
                    "(|train|=%d |val|=%d |test|=%d); old=%s new=%s.", rp, len(detector.training_ids),
                    len(detector.val_ids), len(detector.test_ids),
                    "set" if old_path else "None", "set" if new_path else "None")
        return cls(shared_state, detector=detector, old_path=old_path, new_path=new_path,
                   old_release=old_release, new_release=new_release)

    def _detect(self, dry_run: bool):
        if self._detector is None or self._old_path is None or self._new_path is None:
            return None
        return self._detector.detect(self._old_path, self._new_path,
                                     old_release=self._old_release, new_release=self._new_release)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "urgency": r.urgency,
                "flip_rate_training": r.flip_rate_training, "weighted_impact": r.weighted_impact,
                "n_reclassified_training": r.n_reclassified_training,
                "n_reclassified_total": r.n_reclassified_total, "n_new_variants": r.n_new_variants,
                "should_retrain": r.should_retrain, "direction_breakdown": r.direction_breakdown,
                "old_release": r.old_release, "new_release": r.new_release, "checked_at": r.timestamp}
