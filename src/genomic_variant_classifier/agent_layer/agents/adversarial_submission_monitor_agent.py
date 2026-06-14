#!/usr/bin/env python3
"""adversarial_submission_monitor_agent.py -- BaseAgent adapter for AdversarialSubmissionAgent (Monzia Moodie)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.adversarial_submission_agent import AdversarialSubmissionAgent

logger = logging.getLogger(__name__)


class AdversarialSubmissionMonitorAgent(DriftMonitorBase):
    section = "adversarial_submission"

    def __init__(self, shared_state: SharedState, *, detector: Optional[AdversarialSubmissionAgent] = None,
                 weekly_submissions=None, submitter_baseline=None,
                 aggregate_classifications=None, submitter_metadata=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._args = (weekly_submissions, submitter_baseline, aggregate_classifications, submitter_metadata)

    @classmethod
    def from_default_baseline(
        cls,
        shared_state: SharedState,
        *,
        weekly_submissions=None,
        submitter_baseline=None,
        aggregate_classifications=None,
        submitter_metadata=None,
        thresholds_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "AdversarialSubmissionMonitorAgent":
        """Construct an active agent. The detector's reference is its literature-derived
        threshold config (not a data artifact), so the detector is always constructed; an
        optional thresholds JSON (data/reference/adversarial_submission/thresholds.json)
        overrides the defaults (only recognised dataclass fields are applied). The four
        run-time DataFrame inputs resolve arg -> env (parquet) -> None (awaiting_baseline
        until all are supplied):
          weekly_submissions         GVC_ADV_WEEKLY_SUBMISSIONS
          submitter_baseline         GVC_ADV_SUBMITTER_BASELINE
          aggregate_classifications  GVC_ADV_AGGREGATE_CLASS
          submitter_metadata         GVC_ADV_SUBMITTER_METADATA
        """
        import pandas as pd

        od = Path(output_dir) if output_dir is not None else Path("outputs/drift_reports/adversarial_submission")
        tp = (
            Path(thresholds_path)
            if thresholds_path is not None
            else Path("data/reference/adversarial_submission/thresholds.json")
        )
        overrides: dict = {}
        if tp.exists():
            valid = {f for f in AdversarialSubmissionAgent.__dataclass_fields__ if f not in ("output_dir", "logger")}
            overrides = {k: v for k, v in json.loads(tp.read_text(encoding="utf-8")).items() if k in valid}
            logger.info("AdversarialSubmissionMonitorAgent: threshold overrides from %s: %s", tp, sorted(overrides))
        detector = AdversarialSubmissionAgent(output_dir=od, **overrides)

        resolved = {
            "weekly_submissions": weekly_submissions,
            "submitter_baseline": submitter_baseline,
            "aggregate_classifications": aggregate_classifications,
            "submitter_metadata": submitter_metadata,
        }
        env_map = {
            "weekly_submissions": "GVC_ADV_WEEKLY_SUBMISSIONS",
            "submitter_baseline": "GVC_ADV_SUBMITTER_BASELINE",
            "aggregate_classifications": "GVC_ADV_AGGREGATE_CLASS",
            "submitter_metadata": "GVC_ADV_SUBMITTER_METADATA",
        }
        for key, env_var in env_map.items():
            if resolved[key] is None:
                _e = os.getenv(env_var)
                if _e:
                    _p = Path(_e)
                    resolved[key] = pd.read_parquet(_p) if _p.exists() else None
        logger.info(
            "AdversarialSubmissionMonitorAgent.from_default_baseline: detector ready (overrides=%d); "
            "inputs %s.",
            len(overrides),
            {k: ("set" if v is not None else "None") for k, v in resolved.items()},
        )
        return cls(shared_state, detector=detector, **resolved)

    def _detect(self, dry_run: bool):
        if self._detector is None or any(a is None for a in self._args):
            return None
        return self._detector.detect(*self._args)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity, "n_findings": len(r.findings),
                "quarantine_submitter_ids": list(r.quarantine_submitter_ids), "checked_at": r.timestamp}
