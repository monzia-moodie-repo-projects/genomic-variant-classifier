#!/usr/bin/env python3
"""build_feature_coverage_baseline.py  --  Monzia Moodie

Build the FeatureCoverageSentinel reference from the split-health audit CSV
(scripts/audit_split_feature_health.py --out ...). Replicates the audit's cross-file
aggregation EXACTLY: a column is degenerate if degenerate in ANY split file, and its verdict
is the first (sorted) degeneracy reason seen across files; otherwise 'healthy'. Writes the
canonical reference JSON consumed by FeatureCoverageSentinelAgent.from_reference.

Output: data/reference/feature_coverage/feature_coverage_baseline.json

NOTE on --near-constant-frac: pass the SAME value the audit ran with (default 0.999). It is
recorded in the reference so the sentinel scores future matrices with the identical threshold.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path("data/reference/feature_coverage/feature_coverage_baseline.json")


def build_reference(health: pd.DataFrame, near_constant_frac: float = 0.999, source: str = "") -> dict:
    """health: the audit's per-(file,column) table (must have 'column' + 'degenerate').
    Empty 'degenerate' may arrive as '' (in-memory) or NaN (re-read CSV) -- normalise first."""
    if "column" not in health.columns or "degenerate" not in health.columns:
        raise ValueError("health table must contain 'column' and 'degenerate' columns")
    h = health.copy()
    h["degenerate"] = h["degenerate"].fillna("").astype(str)
    deg = (h[h["degenerate"] != ""]
           .groupby("column")["degenerate"].agg(lambda s: sorted(set(s))[0]))
    all_cols = sorted(h["column"].unique())
    ref = {c: (deg[c] if c in deg.index else "healthy") for c in all_cols}
    n_deg = int(sum(1 for v in ref.values() if v != "healthy"))
    return {
        "reference": ref,
        "near_constant_frac": float(near_constant_frac),
        "n_healthy": len(ref) - n_deg,
        "n_degenerate": n_deg,
        "n_total": len(ref),
        "source": source,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the FeatureCoverageSentinel reference from the audit CSV.")
    ap.add_argument("--health-csv", type=Path, required=True,
                    help="audit_split_feature_health.py --out CSV (per-(file,column) health)")
    ap.add_argument("--near-constant-frac", type=float, default=0.999,
                    help="MUST match the value the audit ran with (default 0.999)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    # keep empty 'degenerate' as '' rather than NaN (read defensively; build_reference also normalises)
    health = pd.read_csv(args.health_csv)
    payload = build_reference(health, args.near_constant_frac, source=str(args.health_csv))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  healthy={payload['n_healthy']}  degenerate={payload['n_degenerate']}  total={payload['n_total']}  "
          f"near_constant_frac={payload['near_constant_frac']}")


if __name__ == "__main__":
    main()
