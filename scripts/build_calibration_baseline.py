#!/usr/bin/env python3
"""build_calibration_baseline.py -- Monzia Moodie

Reference baseline for CalibrationDriftMonitorAgent. baseline_ece is the model's reference top-label
ECE, computed by calling CalibrationDriftAgent.detect() on the reference labeled predictions with
baseline_ece=0 -- the SAME code path the monitor uses, so the reference and monitored ECE can never
silently diverge. The reference predictions are a Run-17 artifact (need a trained model).

RUN AT Run-17.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.agent_layer.agents.calibration_drift_agent import (
    CalibrationDriftAgent,
)

DEFAULT_OUT = Path("data/reference/calibration_drift/calibration_drift_baseline.json")


def build_baseline(reference_labeled_predictions, classes, n_bins: int = 15, **thresholds) -> dict:
    """Compute baseline_ece on the reference predictions via the detector's own detect()."""
    classes = tuple(classes)
    tmp = CalibrationDriftAgent(
        classes=classes, baseline_ece=0.0, output_dir=Path("."), n_bins=int(n_bins)
    )
    ref_ece = float(tmp.detect(reference_labeled_predictions).ece_top_label)
    out = {"classes": list(classes), "baseline_ece": ref_ece, "n_bins": int(n_bins)}
    for k in ("ece_amber", "ece_red", "mce_red", "per_class_red"):
        if thresholds.get(k) is not None:
            out[k] = float(thresholds[k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the calibration-drift reference baseline.")
    ap.add_argument("--reference-predictions", type=Path, required=True,
                    help="Parquet of reference labeled predictions (true_class, predicted_class, p_<class>).")
    ap.add_argument("--classes", nargs="+", required=True, help="Class labels (p_<class> columns).")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    df = pd.read_parquet(a.reference_predictions)
    payload = build_baseline(df, a.classes, n_bins=a.n_bins)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out} (baseline_ece={payload['baseline_ece']:.4f}, classes={payload['classes']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
