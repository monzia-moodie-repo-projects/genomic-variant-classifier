"""
data_readiness_detector.py -- Monzia Moodie

Pure, read-only PRE-RUN readiness checks aggregated into one GO / GO_WITH_WARNINGS / NO_GO verdict. VERIFIES the
data-prep OUTPUTS -- it never runs real_data_prep.py or mutates data (no silent mutation). Two dimensions:

  1. Critical-asset presence: every registry.critical_assets() path (the ACTIVE local sources whose absence
     silent-stubs a connector) must exist and be non-empty.
  2. Feature health: if a splits parquet is available, every feature column is scored with the SAME
     data.feature_health.col_health used by audit_split_feature_health.py -> healthy vs degenerate counts.

The verdict is ADVISORY: the agent surfaces the numbers and a recommended GO/NO_GO; the operator approves or
overrides via the HITL gate. Block thresholds are documented defaults, not silent policy. No BaseAgent / no
SharedState -> unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.data.feature_health import col_health, is_degenerate, verdict

GO = "GO"
GO_WITH_WARNINGS = "GO_WITH_WARNINGS"
NO_GO = "NO_GO"

DEGENERATE_FRAC_BLOCK = 0.5   # >= this fraction of feature cols degenerate -> NO_GO (splits stale/broken)


@dataclass
class AssetStatus:
    path: str
    present: bool
    size_bytes: int | None
    detail: str


def check_assets(asset_paths: list[str], root: str = ".") -> list[AssetStatus]:
    out: list[AssetStatus] = []
    for p in asset_paths:
        fp = Path(root) / p
        if not fp.exists():
            out.append(AssetStatus(p, False, None, "MISSING"))
        else:
            sz = fp.stat().st_size if fp.is_file() else None
            if sz == 0:
                out.append(AssetStatus(p, False, 0, "present but EMPTY (0 bytes)"))
            else:
                out.append(AssetStatus(p, True, sz, "present"))
    return out


def feature_health_summary(df: pd.DataFrame, feature_cols: list[str] | None = None) -> dict:
    cols = feature_cols if feature_cols is not None else list(df.columns)
    degenerate: dict[str, str] = {}
    healthy: list[str] = []
    for c in cols:
        h = col_health(df[c])
        if is_degenerate(h):
            degenerate[c] = verdict(h)
        else:
            healthy.append(c)
    return {"n_cols": len(cols), "n_healthy": len(healthy),
            "n_degenerate": len(degenerate), "degenerate": degenerate}


def readiness_verdict(assets: list[AssetStatus], health: dict | None = None,
                      degenerate_frac_block: float = DEGENERATE_FRAC_BLOCK) -> tuple[str, list[str]]:
    reasons: list[str] = []
    missing = [a for a in assets if not a.present]
    if missing:
        reasons.append(f"{len(missing)} critical asset(s) missing/empty: {[a.path for a in missing]}")
        return NO_GO, reasons
    reasons.append(f"all {len(assets)} critical assets present")
    if health is not None and health["n_cols"]:
        frac = health["n_degenerate"] / health["n_cols"]
        if frac >= degenerate_frac_block:
            reasons.append(f"{health['n_degenerate']}/{health['n_cols']} feature cols degenerate "
                           f"(>= {degenerate_frac_block:.0%}) -- splits look stale/broken; regenerate before launch")
            return NO_GO, reasons
        if health["n_degenerate"]:
            sample = list(health["degenerate"])[:5]
            reasons.append(f"{health['n_degenerate']}/{health['n_cols']} feature cols degenerate "
                           f"(e.g. {sample}) -- review before launch")
            return GO_WITH_WARNINGS, reasons
        reasons.append(f"feature health OK ({health['n_healthy']}/{health['n_cols']} healthy)")
    else:
        reasons.append("feature health not evaluated (no splits parquet found)")
    return GO, reasons


def analyze(asset_paths: list[str], root: str = ".",
            feature_df: pd.DataFrame | None = None,
            feature_cols: list[str] | None = None) -> dict:
    assets = check_assets(asset_paths, root=root)
    health = feature_health_summary(feature_df, feature_cols) if feature_df is not None else None
    v, reasons = readiness_verdict(assets, health)
    return {"assets": assets, "health": health, "verdict": v, "reasons": reasons}
