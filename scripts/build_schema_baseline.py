#!/usr/bin/env python3
"""build_schema_baseline.py -- capture the expected feature-matrix schema (ordered
column names + dtypes + canonical hash) from a reference split, writing the artifact
that SchemaDriftMonitorAgent compares incoming matrices against.

Uses SchemaDriftAgent.hash_schema so the stored hash matches the live detector exactly.
Reads the full matrix via pandas (a few seconds for the ~25 MB X_train) so dtypes match
what detect() observes. Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import SchemaDriftAgent

DEFAULT_MATRIX = Path("outputs/run15_rerun_report/full/splits/X_train.parquet")
DEFAULT_OUT = Path("data/reference/schema/schema_baseline.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build schema-baseline JSON from a reference feature matrix.")
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="reference feature-matrix parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output baseline JSON")
    ap.add_argument("--run-label", default="run15", help="provenance label for the source run")
    args = ap.parse_args()

    if not args.matrix.exists():
        print(f"ABORT: matrix not found: {args.matrix}")
        return 1

    df = pd.read_parquet(args.matrix)
    expected_dtypes = {str(c): str(df[c].dtype) for c in df.columns}  # column order preserved
    if not expected_dtypes:
        print("ABORT: matrix has no columns")
        return 1
    expected_hash = SchemaDriftAgent.hash_schema(expected_dtypes)

    payload = {
        "schema_version": 1,
        "run_label": args.run_label,
        "captured_from": str(args.matrix).replace("\\", "/"),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "n_columns": len(expected_dtypes),
        "expected_schema_hash": expected_hash,
        "expected_dtypes": expected_dtypes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # round-trip: the written file must re-hash to the same value
    reloaded = json.loads(args.out.read_text(encoding="utf-8"))
    if reloaded["expected_schema_hash"] != SchemaDriftAgent.hash_schema(reloaded["expected_dtypes"]):
        print("ABORT: hash mismatch after reload")
        return 1

    print(f"OK: wrote {args.out}")
    print(f"  columns={payload['n_columns']}  hash={expected_hash[:16]}...  run={args.run_label}")
    print(f"  source={payload['captured_from']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
