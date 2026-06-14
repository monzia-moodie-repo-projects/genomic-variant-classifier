from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import pandas as pd

from genomic_variant_classifier.data.feature_health import (
    DEFAULT_NEAR_CONSTANT_FRAC,
    col_health,
    verdict,
)


@dataclass(frozen=True)
class FeatureCoverageResult:
    timestamp: str
    regressed: tuple[tuple[str, str], ...]   # (column, current_reason): healthy at reference, degenerate now
    dropped: tuple[str, ...]                  # in reference, absent from the current matrix
    recovered: tuple[str, ...]                # degenerate at reference, healthy now
    still_degenerate: tuple[str, ...]         # degenerate at reference and still degenerate
    new_columns: tuple[str, ...]              # in the current matrix, absent from the reference
    severity: str


@dataclass
class FeatureCoverageSentinelAgent:
    """Audit a current feature matrix against a reference health verdict.

    The reference is the per-column verdict produced by
    scripts/audit_split_feature_health.py (column -> 'healthy' or a degeneracy reason
    such as ALL_ZERO / CONSTANT / NEAR_CONSTANT(frac) / ALL_NULL). The sentinel's job is
    to catch SILENT feature regressions -- a column healthy at reference time that has gone
    degenerate now (the 34/78 and 38/78 dead-feature regressions seen in Run 14 / Run 10b) --
    and feature drops, before they reach training. It scores the current matrix with the SAME
    feature_health.col_health logic and the SAME near_constant_frac the reference was built
    with, so 'current health' and the reference are directly comparable.

    severity: red if any column regressed or was dropped; amber if there are new (unaudited)
    columns; green otherwise. recovered / still_degenerate are reported but do not raise
    severity (recovery is good news; a still-dead column is the known baseline state).
    """

    reference: dict[str, str]
    output_dir: Path
    near_constant_frac: float = DEFAULT_NEAR_CONSTANT_FRAC
    logger: Optional[Logger] = field(default=None, repr=False)

    @staticmethod
    def _is_healthy(v: str) -> bool:
        return (not v) or v == "healthy"

    @classmethod
    def from_reference(cls, reference_path, output_dir, **kw) -> "FeatureCoverageSentinelAgent":
        """Load a reference verdict map from JSON. Canonical form:
        {"reference": {col: verdict}, "near_constant_frac": float, ...}; a bare
        {col: verdict} dict is also accepted (near_constant_frac defaults)."""
        data = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "reference" in data:
            ref_map = data["reference"]
            ncf = float(data.get("near_constant_frac", DEFAULT_NEAR_CONSTANT_FRAC))
        else:
            ref_map = data
            ncf = DEFAULT_NEAR_CONSTANT_FRAC
        return cls(
            reference={str(k): str(v) for k, v in ref_map.items()},
            output_dir=Path(output_dir),
            near_constant_frac=ncf,
            **kw,
        )

    def detect(self, current_matrix: pd.DataFrame) -> FeatureCoverageResult:
        cur_cols = set(current_matrix.columns)
        cur_verdict = {
            c: verdict(col_health(current_matrix[c], self.near_constant_frac))
            for c in current_matrix.columns
        }
        regressed: list[tuple[str, str]] = []
        dropped: list[str] = []
        recovered: list[str] = []
        still_deg: list[str] = []
        for col, ref_v in self.reference.items():
            if col not in cur_cols:
                dropped.append(col)
                continue
            ref_healthy = self._is_healthy(ref_v)
            cur_v = cur_verdict[col]
            cur_healthy = self._is_healthy(cur_v)
            if ref_healthy and not cur_healthy:
                regressed.append((col, cur_v))
            elif (not ref_healthy) and cur_healthy:
                recovered.append(col)
            elif (not ref_healthy) and (not cur_healthy):
                still_deg.append(col)
        new_columns = sorted(cur_cols - set(self.reference))
        regressed.sort()
        dropped.sort()
        recovered.sort()
        still_deg.sort()
        if regressed or dropped:
            severity = "red"
        elif new_columns:
            severity = "amber"
        else:
            severity = "green"
        return FeatureCoverageResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            regressed=tuple(regressed),
            dropped=tuple(dropped),
            recovered=tuple(recovered),
            still_degenerate=tuple(still_deg),
            new_columns=tuple(new_columns),
            severity=severity,
        )
