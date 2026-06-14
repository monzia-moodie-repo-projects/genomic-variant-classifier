#!/usr/bin/env python3
"""infrastructure_drift_monitor_agent.py -- BaseAgent adapter for InfrastructureDriftAgent (Monzia Moodie)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.drift_monitor_base import DriftMonitorBase
from genomic_variant_classifier.agent_layer.agents.infrastructure_drift_agent import InfrastructureDriftAgent

logger = logging.getLogger(__name__)


class InfrastructureDriftMonitorAgent(DriftMonitorBase):
    section = "infrastructure_drift"

    def __init__(self, shared_state: SharedState, *, detector: Optional[InfrastructureDriftAgent] = None,
                 current_packages=None, current_dag_spec=None, replayed_features=None) -> None:
        super().__init__(shared_state)
        self._detector = detector
        self._args = (current_packages, current_dag_spec, replayed_features)

    @classmethod
    def from_default_baseline(
        cls,
        shared_state: SharedState,
        *,
        current_packages=None,
        current_dag_spec=None,
        replayed_features=None,
        baseline_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> "InfrastructureDriftMonitorAgent":
        """Construct an active agent from the canonical infrastructure baseline.

        Loads the InfrastructureDriftAgent detector from
        data/reference/infrastructure/infrastructure_baseline.json (pinned_packages +
        expected_dag_hash + golden_set; all model-free, from build_infrastructure_baseline.py).
        current_packages auto-resolves from the live env (importlib.metadata) when not
        supplied; current_dag_spec resolves arg -> GVC_INFRA_DAG_SPEC env (inline string or
        a file path); replayed_features resolves arg -> GVC_INFRA_REPLAYED_FEATURES env
        (parquet). Any current input left None keeps the agent awaiting_baseline. A missing
        baseline file returns an inactive agent (graceful), never raises.
        """
        bp = (
            Path(baseline_path)
            if baseline_path is not None
            else Path("data/reference/infrastructure/infrastructure_baseline.json")
        )
        od = (
            Path(output_dir)
            if output_dir is not None
            else Path("outputs/drift_reports/infrastructure")
        )
        if not bp.exists():
            logger.info(
                "InfrastructureDriftMonitorAgent.from_default_baseline: baseline absent "
                "(%s); returning inactive agent (awaiting_baseline).",
                bp,
            )
            return cls(shared_state)
        detector = InfrastructureDriftAgent.from_baseline(bp, output_dir=od)
        if current_packages is None:
            current_packages = InfrastructureDriftAgent.current_package_versions(
                detector.monitored_packages
            )
        if current_dag_spec is None:
            _env_dag = os.getenv("GVC_INFRA_DAG_SPEC")
            if _env_dag:
                _dp = Path(_env_dag)
                current_dag_spec = _dp.read_text(encoding="utf-8") if _dp.exists() else _env_dag
        if replayed_features is None:
            _env_rep = os.getenv("GVC_INFRA_REPLAYED_FEATURES")
            if _env_rep:
                import pandas as pd

                _rp = Path(_env_rep)
                replayed_features = pd.read_parquet(_rp) if _rp.exists() else None
        logger.info(
            "InfrastructureDriftMonitorAgent.from_default_baseline: detector loaded from %s "
            "(packages=%s, dag_spec=%s, replayed=%s).",
            bp,
            "set" if current_packages is not None else "None",
            "set" if current_dag_spec is not None else "None",
            "set" if replayed_features is not None else "None",
        )
        return cls(
            shared_state,
            detector=detector,
            current_packages=current_packages,
            current_dag_spec=current_dag_spec,
            replayed_features=replayed_features,
        )

    def _detect(self, dry_run: bool):
        if self._detector is None or any(a is None for a in self._args):
            return None
        return self._detector.detect(*self._args)

    def _summarize(self, r) -> dict:
        return {"severity": r.severity,
                "package_changes": {k: list(v) for k, v in r.package_changes.items()},
                "dag_hash_changed": r.dag_hash_changed,
                "golden_set_divergence": r.golden_set_divergence, "checked_at": r.timestamp}
