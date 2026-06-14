#!/usr/bin/env python3
"""annotation_policy_monitor_agent.py -- BaseAgent adapter for AnnotationPolicyAgent (Monzia Moodie)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.annotation_policy_agent import AnnotationPolicyAgent

logger = logging.getLogger(__name__)


class AnnotationPolicyMonitorAgent(DriftMonitorBase):
    section = "annotation_policy"

    def __init__(self, shared_state: SharedState, *, detector: Optional[AnnotationPolicyAgent] = None,
                 new_svi_pubs=None, clinvar_status_changes=None,
                 submitter_history=None, n_inference_variants=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._args = (new_svi_pubs, clinvar_status_changes, submitter_history, n_inference_variants)

    @classmethod
    def from_default_baseline(
        cls,
        shared_state: SharedState,
        *,
        new_svi_pubs=None,
        clinvar_status_changes=None,
        submitter_history=None,
        n_inference_variants=None,
        thresholds_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "AnnotationPolicyMonitorAgent":
        """Construct an active agent. The detector's reference is its literature-derived
        threshold config (not a data artifact), so the detector is always constructed; an
        optional thresholds JSON (data/reference/annotation_policy/thresholds.json) overrides
        the defaults (only recognised dataclass fields are applied). The four run-time inputs
        resolve arg -> env -> None (awaiting_baseline until all are supplied):
          new_svi_pubs            GVC_ANNOT_SVI_PUBS (comma-separated)
          clinvar_status_changes  GVC_ANNOT_STATUS_CHANGES (parquet)
          submitter_history       GVC_ANNOT_SUBMITTER_HISTORY (parquet)
          n_inference_variants    GVC_ANNOT_N_INFERENCE (int)
        """
        import pandas as pd

        od = Path(output_dir) if output_dir is not None else Path("outputs/drift_reports/annotation_policy")
        tp = (
            Path(thresholds_path)
            if thresholds_path is not None
            else Path("data/reference/annotation_policy/thresholds.json")
        )
        overrides: dict = {}
        if tp.exists():
            valid = {f for f in AnnotationPolicyAgent.__dataclass_fields__ if f not in ("output_dir", "logger")}
            overrides = {k: v for k, v in json.loads(tp.read_text(encoding="utf-8")).items() if k in valid}
            logger.info("AnnotationPolicyMonitorAgent: threshold overrides from %s: %s", tp, sorted(overrides))
        detector = AnnotationPolicyAgent(output_dir=od, **overrides)

        if new_svi_pubs is None:
            _env = os.getenv("GVC_ANNOT_SVI_PUBS")
            if _env is not None:
                new_svi_pubs = [p for p in _env.split(",") if p]
        if clinvar_status_changes is None:
            _env = os.getenv("GVC_ANNOT_STATUS_CHANGES")
            if _env:
                _p = Path(_env)
                clinvar_status_changes = pd.read_parquet(_p) if _p.exists() else None
        if submitter_history is None:
            _env = os.getenv("GVC_ANNOT_SUBMITTER_HISTORY")
            if _env:
                _p = Path(_env)
                submitter_history = pd.read_parquet(_p) if _p.exists() else None
        if n_inference_variants is None:
            _env = os.getenv("GVC_ANNOT_N_INFERENCE")
            if _env:
                try:
                    n_inference_variants = int(_env)
                except ValueError:
                    n_inference_variants = None
        logger.info(
            "AnnotationPolicyMonitorAgent.from_default_baseline: detector ready (overrides=%d); "
            "inputs svi=%s changes=%s history=%s n=%s.",
            len(overrides),
            "set" if new_svi_pubs is not None else "None",
            "set" if clinvar_status_changes is not None else "None",
            "set" if submitter_history is not None else "None",
            "set" if n_inference_variants is not None else "None",
        )
        return cls(
            shared_state,
            detector=detector,
            new_svi_pubs=new_svi_pubs,
            clinvar_status_changes=clinvar_status_changes,
            submitter_history=submitter_history,
            n_inference_variants=n_inference_variants,
        )

    def _detect(self, dry_run: bool):
        if self._detector is None or any(a is None for a in self._args):
            return None
        return self._detector.detect(*self._args)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "new_svi_publications": list(r.new_svi_publications),
                "pct_variants_with_review_status_change": r.pct_variants_with_review_status_change,
                "submitters_with_outlier_alarm": list(r.submitters_with_outlier_alarm),
                "requires_variant_scientist_review": r.requires_variant_scientist_review,
                "checked_at": r.timestamp}
