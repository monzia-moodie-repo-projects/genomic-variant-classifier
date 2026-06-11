#!/usr/bin/env python3
"""run_schema_drift_check.py -- Preflight schema gate.

Validate a feature matrix's column/dtype schema against the schema baseline
(data/reference/schema/schema_baseline.json, produced by build_schema_baseline.py).
Run this BEFORE any regen or training run to catch dropped / renamed / retyped
columns before they silently zero a feature.

Only the first batch of the parquet is read: parquet stores dtypes in its schema,
so a head-read is dtype-exact while keeping memory bounded on full-cohort matrices.

Exit codes (consistent with run_drift_monitor.py):
    0 = schema matches baseline (green)
    2 = schema drift detected (red): column added/removed, dtype changed, or pandera violation
    3 = usage / environment error (baseline or matrix missing, or pandera/pyarrow absent)

Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preflight schema gate: validate a feature matrix against the schema baseline."
    )
    p.add_argument("--baseline", type=Path,
                   default=Path("data/reference/schema/schema_baseline.json"),
                   help="Schema baseline JSON (build_schema_baseline.py output).")
    p.add_argument("--matrix", type=Path, required=True,
                   help="Feature matrix parquet to validate (e.g. a fresh-regen X_train.parquet).")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/drift_reports/schema"),
                   help="Directory the detector may write artefacts to.")
    p.add_argument("--sample-rows", type=int, default=4096,
                   help="Rows read for validation. Parquet dtypes are exact regardless; "
                        "this only bounds memory.")
    return p


def check(baseline_path: Path, matrix_path: Path, output_dir: Path, sample_rows: int = 4096):
    """Run the schema detector against a matrix; returns a SchemaDriftResult."""
    from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import SchemaDriftAgent
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    detector = SchemaDriftAgent.from_baseline(baseline_path, output_dir=output_dir)
    pf = pq.ParquetFile(str(matrix_path))
    try:
        df = next(pf.iter_batches(batch_size=max(1, sample_rows))).to_pandas()
    except StopIteration:
        df = pf.read().to_pandas()
    return detector.detect(df)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.baseline.exists():
        print(f"ABORT: baseline not found: {args.baseline}", file=sys.stderr)
        return 3
    if not args.matrix.exists():
        print(f"ABORT: matrix not found: {args.matrix}", file=sys.stderr)
        return 3
    try:
        result = check(args.baseline, args.matrix, args.output_dir, args.sample_rows)
    except ImportError as exc:
        print(f"ABORT: schema gate requires pandera and pyarrow ({exc}). "
              f"pip install pandera pyarrow", file=sys.stderr)
        return 3

    print(f"baseline : {args.baseline}  (expected_hash={result.expected_schema_hash[:16]}...)")
    print(f"matrix   : {args.matrix}  (observed_hash={result.observed_schema_hash[:16]}...)")
    print(f"severity : {result.severity}")
    if result.columns_added:
        print(f"  + added   ({len(result.columns_added)}): {', '.join(result.columns_added)}")
    if result.columns_removed:
        print(f"  - removed ({len(result.columns_removed)}): {', '.join(result.columns_removed)}")
    for col, exp, obs in result.columns_dtype_changed:
        print(f"  ~ dtype {col}: expected {exp}, observed {obs}")
    if result.pandera_violations:
        print(f"  ! pandera violations: {len(result.pandera_violations)}")
        for v in list(result.pandera_violations)[:10]:
            print(f"      {v}")

    if result.severity == "green":
        print("RESULT: schema matches baseline (green).")
        return 0
    print("RESULT: SCHEMA DRIFT detected (red) -- investigate before regen/training.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
