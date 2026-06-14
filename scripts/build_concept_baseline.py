#!/usr/bin/env python3
"""build_concept_baseline.py -- Monzia Moodie

Reference baseline for ConceptDriftMonitorAgent. The two scalars (cbpe_baseline_auroc,
cbpe_baseline_sigma) come from NannyML CBPE on the model's reference window -- a Run-17 artifact.
A thin writer: pass the two CBPE scalars (+ optional thresholds), it writes the canonical JSON that
ConceptDriftAgent.from_baseline loads.

RUN AT Run-17 (needs the trained model's CBPE reference output).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUT = Path("data/reference/concept_drift/concept_drift_baseline.json")


def build_baseline(cbpe_baseline_auroc, cbpe_baseline_sigma, **thresholds) -> dict:
    out = {
        "cbpe_baseline_auroc": float(cbpe_baseline_auroc),
        "cbpe_baseline_sigma": float(cbpe_baseline_sigma),
    }
    for k in ("sigma_drop_amber", "auroc_drop_red", "bbse_alpha"):
        if thresholds.get(k) is not None:
            out[k] = float(thresholds[k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the concept-drift reference baseline.")
    ap.add_argument("--cbpe-baseline-auroc", type=float, required=True,
                    help="NannyML CBPE estimated AUROC on the reference window (Run-17).")
    ap.add_argument("--cbpe-baseline-sigma", type=float, required=True,
                    help="CBPE AUROC confidence sigma on the reference window (Run-17).")
    ap.add_argument("--sigma-drop-amber", type=float, default=None)
    ap.add_argument("--auroc-drop-red", type=float, default=None)
    ap.add_argument("--bbse-alpha", type=float, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    payload = build_baseline(a.cbpe_baseline_auroc, a.cbpe_baseline_sigma,
                             sigma_drop_amber=a.sigma_drop_amber, auroc_drop_red=a.auroc_drop_red,
                             bbse_alpha=a.bbse_alpha)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out} (auroc={payload['cbpe_baseline_auroc']}, sigma={payload['cbpe_baseline_sigma']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
