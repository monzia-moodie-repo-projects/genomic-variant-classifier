#!/usr/bin/env python3
"""build_fairness_baseline.py -- Monzia Moodie

Reference baseline for FairnessSubgroupMonitorAgent. p_train_per_stratum is the per-(axis, stratum)
reference predicted-class count vector -- computed EXACTLY as FairnessSubgroupAgent._stratum_metric's
`observed = [(sub.predicted_class == c).sum() for c in classes]`, so the PSI-vs-train comparison is
self-consistent. The reference predictions are a Run-17 artifact (need a trained model).

NOTE: this builds the baseline for the EXISTING detector. The detector's per-stratum AUROC is a
confidence proxy and max_dpd_change is a 0.0 stub (both PHASE_2_FEATURES); this script only captures
p_train_per_stratum and does not change those placeholders.

RUN AT Run-17.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path("data/reference/fairness_subgroup/fairness_subgroup_baseline.json")


def build_baseline(reference_predictions, axes, classes, **thresholds) -> dict:
    """axes maps axis_name -> column in reference_predictions (e.g. {'ancestry': 'gnomad_pop'})."""
    classes = tuple(classes)
    records = []
    for axis_label, col in axes.items():
        for stratum, sub in reference_predictions.groupby(col, sort=False):
            observed = [int((sub["predicted_class"] == c).sum()) for c in classes]
            records.append({"axis": axis_label, "stratum": str(stratum), "p_train": observed})
    out = {"classes": list(classes), "p_train_per_stratum": records}
    for k in ("eod_amber", "auroc_below_overall_sigma", "ece_red"):
        if thresholds.get(k) is not None:
            out[k] = float(thresholds[k])
    if thresholds.get("high_priority_strata"):
        out["high_priority_strata"] = list(thresholds["high_priority_strata"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the fairness-subgroup reference baseline.")
    ap.add_argument("--reference-predictions", type=Path, required=True,
                    help="Parquet of reference predictions (predicted_class, p_<class>, axis columns).")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--axes", required=True,
                    help='JSON object axis_name -> column, e.g. \'{"ancestry":"gnomad_pop"}\'.')
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    df = pd.read_parquet(a.reference_predictions)
    axes = json.loads(a.axes)
    payload = build_baseline(df, axes, a.classes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(payload['p_train_per_stratum'])} strata, classes={payload['classes']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
