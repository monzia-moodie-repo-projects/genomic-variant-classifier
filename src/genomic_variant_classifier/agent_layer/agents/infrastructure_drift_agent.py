from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InfraDriftResult:
    timestamp: str
    package_changes: dict[str, tuple[str, str]]
    dag_hash_changed: bool
    golden_set_divergence: int
    severity: str


@dataclass
class InfrastructureDriftAgent:
    """Detects environment-induced silent failures.

    Sculley et al., "Hidden Technical Debt in Machine Learning Systems",
    NeurIPS 2015 — this class targets the "Configuration Debt" and
    "Pipeline Jungles" anti-patterns.
    """

    pinned_packages: dict[str, str]
    expected_dag_hash: str
    golden_set: pd.DataFrame
    output_dir: Path
    monitored_packages: frozenset[str] = frozenset(
        {
            "evidently",
            "nannyml",
            "alibi-detect",
            "river",
            "pandera",
            "pandas",
            "numpy",
            "pyspark",
            "lightgbm",
            "xgboost",
            "catboost",
            "torch",
            "torch-geometric",
            "scikit-learn",
            "scipy",
        }
    )
    logger: Optional[Logger] = field(default=None, repr=False)

    @staticmethod
    def _hash_dag(dag_spec: str) -> str:
        return hashlib.sha256(dag_spec.encode("utf-8")).hexdigest()

    @staticmethod
    def current_package_versions(packages) -> "dict[str, str]":
        """Installed version per package (importlib.metadata), '' if absent. Shared by
        build_infrastructure_baseline.py (pinned) and the monitor agent (current), so
        baseline capture and live capture use one code path. Model-free."""
        from importlib.metadata import PackageNotFoundError, version

        out: dict[str, str] = {}
        for pkg in packages:
            try:
                out[pkg] = version(pkg)
            except PackageNotFoundError:
                out[pkg] = ""
        return out

    @classmethod
    def from_baseline(cls, baseline_path, output_dir, **kw) -> "InfrastructureDriftAgent":
        """Reconstruct a detector from an infrastructure baseline JSON (pinned_packages +
        expected_dag_hash + golden_set records). golden_set is the feature pipeline's
        output on a fixed variant set (model-free); records preserve numeric dtypes for
        the np.isclose divergence check in detect()."""
        data = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        golden_set = pd.DataFrame(data["golden_set"])
        return cls(
            pinned_packages={str(k): str(v) for k, v in data["pinned_packages"].items()},
            expected_dag_hash=str(data["expected_dag_hash"]),
            golden_set=golden_set,
            output_dir=Path(output_dir),
            **kw,
        )

    def detect(
        self,
        current_packages: dict[str, str],
        current_dag_spec: str,
        replayed_features: pd.DataFrame,
    ) -> InfraDriftResult:
        changes: dict[str, tuple[str, str]] = {}
        for pkg in self.monitored_packages:
            old = self.pinned_packages.get(pkg, "")
            new = current_packages.get(pkg, "")
            if old != new:
                changes[pkg] = (old, new)
        dag_changed = self._hash_dag(current_dag_spec) != self.expected_dag_hash
        merged = self.golden_set.merge(
            replayed_features, on="variant_id", suffixes=("_ref", "_replay")
        )
        divergence = 0
        for col in self.golden_set.columns:
            if col == "variant_id":
                continue
            ref = merged[f"{col}_ref"].to_numpy()
            rep = merged[f"{col}_replay"].to_numpy()
            if np.issubdtype(ref.dtype, np.number):
                divergence += int(np.sum(~np.isclose(ref, rep, equal_nan=True)))
            else:
                divergence += int(np.sum(ref != rep))
        if divergence > 0:
            severity = "red"
        elif changes or dag_changed:
            severity = "amber"
        else:
            severity = "green"
        return InfraDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            package_changes=changes,
            dag_hash_changed=dag_changed,
            golden_set_divergence=divergence,
            severity=severity,
        )
