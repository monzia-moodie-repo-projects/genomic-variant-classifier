from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ReclassificationResult:
    timestamp: str
    old_release: str
    new_release: str
    n_reclassified_total: int
    n_reclassified_training: int
    flip_rate_training: float
    weighted_impact: float
    urgency: str
    severity: str
    should_retrain: bool
    n_new_variants: int
    direction_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class ReclassificationSentinelAgent:
    """Label-drift sentinel: quantifies ClinVar reclassification between two releases and maps the
    training-set flip rate / impact-weighted flip rate to a green/amber/red severity.

    Thin wrapper over monitoring.clinvar_tracker.ClinVarTracker (single source of truth for the flip
    accounting + urgency). The sentinel adds the agent-layer interface and the urgency -> severity map;
    it does not re-derive the flip thresholds. detect() runs the comparison with output_dir=None (no
    file side effects); the temporal-cohort/manifest writing stays a separate Run-17 concern.

    The reference is the committed split membership (which variant_ids are train/val/test), supplied as
    frozensets or loaded from a compact reference parquet via from_reference(); the run-time inputs are
    the OLD (training-era) and NEW (current) ClinVar release parquets.
    """

    training_ids: frozenset[str]
    val_ids: frozenset[str] = frozenset()
    test_ids: frozenset[str] = frozenset()
    output_dir: Path = Path("outputs/drift_reports/reclassification")
    logger: Optional[Logger] = field(default=None, repr=False)

    # ClinVarTracker urgency vocabulary -> drift severity vocabulary.
    _URGENCY_TO_SEVERITY = {"none": "green", "monitor": "amber", "retrain": "red", "urgent": "red"}

    @classmethod
    def from_reference(cls, reference_path, output_dir, **overrides) -> "ReclassificationSentinelAgent":
        """Load split membership from a compact reference parquet with columns (variant_id, split),
        split in {train, val, test}. Produced by build_reclassification_reference.py from the committed
        splits (a Run-15/17 artifact). Variant-id sets can be millions of rows, hence parquet not JSON.
        """
        import pandas as pd

        ref = pd.read_parquet(Path(reference_path), columns=["variant_id", "split"])
        by = {
            str(split): frozenset(sub["variant_id"].astype(str))
            for split, sub in ref.groupby("split", sort=False)
        }
        return cls(
            training_ids=by.get("train", frozenset()),
            val_ids=by.get("val", frozenset()),
            test_ids=by.get("test", frozenset()),
            output_dir=Path(output_dir),
            **overrides,
        )

    def detect(
        self,
        old_path,
        new_path,
        *,
        old_release: str = "previous",
        new_release: str = "current",
    ) -> ReclassificationResult:
        """Compare two ClinVar release parquets and map the result to a drift severity.

        Required columns in each parquet (read by ClinVarTracker): variant_id, clinical_sig,
        review_status, gene_symbol, chrom, pos, ref, alt. output_dir is forced to None here so the
        detector has no file side effects.
        """
        from genomic_variant_classifier.monitoring.clinvar_tracker import ClinVarTracker

        tracker = ClinVarTracker(set(self.training_ids), set(self.val_ids), set(self.test_ids))
        report = tracker.compare(
            old_path, new_path, output_dir=None, old_release=old_release, new_release=new_release
        )
        severity = self._URGENCY_TO_SEVERITY.get(report.urgency, "green")
        return ReclassificationResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            old_release=report.old_release,
            new_release=report.new_release,
            n_reclassified_total=int(report.n_reclassified_total),
            n_reclassified_training=int(report.n_reclassified_training),
            flip_rate_training=float(report.flip_rate_training),
            weighted_impact=float(report.weighted_impact),
            urgency=str(report.urgency),
            severity=severity,
            should_retrain=bool(report.should_retrain),
            n_new_variants=int(report.n_new_variants),
            direction_breakdown=dict(report.direction_breakdown),
        )
