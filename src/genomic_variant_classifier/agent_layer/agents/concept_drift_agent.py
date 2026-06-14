from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConceptDriftResult:
    timestamp: str
    cbpe_estimated_auroc: float
    cbpe_baseline_auroc: float
    cbpe_drop: float
    bbse_pvalue: float
    likely_pure_concept: bool
    severity: str
    n_samples: int


@dataclass
class ConceptDriftAgent:
    """Differential-diagnosis between label shift and concept drift.

    Concept drift signature: NannyML CBPE-estimated AUROC drops >= 2σ
    AND BBSE label-shift test is NOT significant. Reference:
      - Gama et al., ACM Comput. Surv. 2014, "A Survey on Concept Drift Adaptation".
      - Lipton et al., ICML 2018, BBSE.
      - NannyML CBPE: Confidence-Based Performance Estimation.
    """

    cbpe_baseline_auroc: float
    cbpe_baseline_sigma: float
    output_dir: Path
    sigma_drop_amber: float = 2.0
    auroc_drop_red: float = 0.03
    bbse_alpha: float = 0.05
    logger: Optional[Logger] = field(default=None, repr=False)

    @classmethod
    def from_baseline(cls, baseline_path, output_dir, **overrides) -> "ConceptDriftAgent":
        """Load cbpe_baseline_auroc + cbpe_baseline_sigma (+ optional thresholds) from a baseline JSON.

        The two scalars come from NannyML CBPE on the model's reference window (the estimated AUROC and
        its confidence sigma) -- a Run-17 artifact. Mirrors LabelShiftAgent.from_baseline.
        """
        data = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        kw = {k: float(data[k]) for k in ("sigma_drop_amber", "auroc_drop_red", "bbse_alpha") if k in data}
        kw.update(overrides)
        return cls(
            cbpe_baseline_auroc=float(data["cbpe_baseline_auroc"]),
            cbpe_baseline_sigma=float(data["cbpe_baseline_sigma"]),
            output_dir=Path(output_dir),
            **kw,
        )

    def detect(
        self,
        cbpe_estimated_auroc: float,
        bbse_pvalue: float,
        n_samples: int,
    ) -> ConceptDriftResult:
        drop = self.cbpe_baseline_auroc - cbpe_estimated_auroc
        sigma_drop = drop / max(self.cbpe_baseline_sigma, 1e-6)
        likely_pure_concept = (
            sigma_drop >= self.sigma_drop_amber and bbse_pvalue >= self.bbse_alpha
        )
        if drop >= self.auroc_drop_red and likely_pure_concept:
            severity = "red"
        elif sigma_drop >= self.sigma_drop_amber:
            severity = "amber"
        else:
            severity = "green"
        return ConceptDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cbpe_estimated_auroc=float(cbpe_estimated_auroc),
            cbpe_baseline_auroc=float(self.cbpe_baseline_auroc),
            cbpe_drop=float(drop),
            bbse_pvalue=float(bbse_pvalue),
            likely_pure_concept=bool(likely_pure_concept),
            severity=severity,
            n_samples=int(n_samples),
        )
