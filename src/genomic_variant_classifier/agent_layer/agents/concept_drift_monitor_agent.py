#!/usr/bin/env python3
"""concept_drift_monitor_agent.py -- BaseAgent adapter for ConceptDriftAgent (Monzia Moodie)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.concept_drift_agent import ConceptDriftAgent

logger = logging.getLogger(__name__)


class ConceptDriftMonitorAgent(DriftMonitorBase):
    section = "concept_drift"

    def __init__(self, shared_state: SharedState, *, detector: Optional[ConceptDriftAgent] = None,
                 cbpe_estimated_auroc: Optional[float] = None, bbse_pvalue: Optional[float] = None,
                 n_samples: Optional[int] = None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._args = (cbpe_estimated_auroc, bbse_pvalue, n_samples)

    @classmethod
    def from_default_baseline(
        cls,
        shared_state: SharedState,
        *,
        cbpe_estimated_auroc: Optional[float] = None,
        bbse_pvalue: Optional[float] = None,
        n_samples: Optional[int] = None,
        baseline_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "ConceptDriftMonitorAgent":
        """Construct an active agent from the canonical concept-drift baseline
        (data/reference/concept_drift/concept_drift_baseline.json: cbpe_baseline_auroc +
        cbpe_baseline_sigma, from NannyML CBPE on the reference window -- a Run-17 artifact).
        The production scalars resolve arg -> env (GVC_CONCEPT_CBPE_AUROC / GVC_CONCEPT_BBSE_PVALUE /
        GVC_CONCEPT_N_SAMPLES) -> None (awaiting_baseline until supplied). Missing baseline ->
        inactive (graceful), never raises. Routed by the orchestrator from_default_baseline hook."""
        bp = (Path(baseline_path) if baseline_path is not None
              else Path("data/reference/concept_drift/concept_drift_baseline.json"))
        od = (Path(output_dir) if output_dir is not None
              else Path("outputs/drift_reports/concept_drift"))
        if cbpe_estimated_auroc is None:
            _e = os.getenv("GVC_CONCEPT_CBPE_AUROC")
            cbpe_estimated_auroc = float(_e) if _e else None
        if bbse_pvalue is None:
            _e = os.getenv("GVC_CONCEPT_BBSE_PVALUE")
            bbse_pvalue = float(_e) if _e else None
        if n_samples is None:
            _e = os.getenv("GVC_CONCEPT_N_SAMPLES")
            n_samples = int(_e) if _e else None
        if not bp.exists():
            logger.info("ConceptDriftMonitorAgent.from_default_baseline: baseline absent (%s); "
                        "returning inactive agent (awaiting_baseline).", bp)
            return cls(shared_state)
        detector = ConceptDriftAgent.from_baseline(bp, output_dir=od)
        _ready = None not in (cbpe_estimated_auroc, bbse_pvalue, n_samples)
        logger.info("ConceptDriftMonitorAgent.from_default_baseline: detector loaded from %s "
                    "(scalars=%s).", bp, "set" if _ready else "partial/None")
        return cls(shared_state, detector=detector, cbpe_estimated_auroc=cbpe_estimated_auroc,
                   bbse_pvalue=bbse_pvalue, n_samples=n_samples)

    def _detect(self, dry_run: bool):
        if self._detector is None or any(a is None for a in self._args):
            return None
        return self._detector.detect(*self._args)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "cbpe_estimated_auroc": r.cbpe_estimated_auroc,
                "cbpe_baseline_auroc": r.cbpe_baseline_auroc, "cbpe_drop": r.cbpe_drop,
                "bbse_pvalue": r.bbse_pvalue, "likely_pure_concept": r.likely_pure_concept,
                "n_samples": r.n_samples, "checked_at": r.timestamp}
