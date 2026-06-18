#!/usr/bin/env python3
"""
extend_schema_baseline_rnaseq.py  --  Monzia Moodie

Resolve the Run-17 schema-drift blocker. TABULAR_FEATURES is 87 columns; the matrix
always materializes the 5 rnaseq_* columns (zero-filled when no RNA-seq data). The
schema baseline (data/reference/schema/schema_baseline.json) was captured at 82 columns
from run16b-smoke + hetero_gnn_score, BEFORE the rnaseq_* features entered the matrix.
run_schema_drift_check.py exits 2 (red) on any added column, so at run time the 87-col
matrix vs the 82-baseline => 5 columns_added => DRIFT => Run 17 blocked.

This surgically adds the 5 rnaseq_* columns (float64, matching the matrix and the existing
gtex_/reactome_ float64 features) to the baseline using the project's OWN
SchemaDriftAgent.hash_schema (no hash replication). It re-validates the baseline's stored
hash BEFORE editing (aborts if already inconsistent), is idempotent, and round-trips the
written file. The hash is order-independent (sorted dtype-family pairs) and detect() is
set-based, so column position does not affect correctness.

Usage:  python scripts/extend_schema_baseline_rnaseq.py
        python scripts/extend_schema_baseline_rnaseq.py --baseline data/reference/schema/schema_baseline.json --dry-run
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

RNASEQ = ["rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv",
          "rnaseq_log2fc", "rnaseq_de_neglog10p"]
DTYPE = "float64"
ANCHOR = "reactome_pathway_count"  # insert rnaseq_* right after this for readability (order is hash-irrelevant)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=Path("data/reference/schema/schema_baseline.json"))
    ap.add_argument("--dry-run", action="store_true", help="report the change without writing")
    args = ap.parse_args(argv)

    sys.path.insert(0, "src")
    from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import SchemaDriftAgent

    if not args.baseline.exists():
        print(f"ABORT: baseline not found: {args.baseline}"); return 3
    b = json.loads(args.baseline.read_text(encoding="utf-8"))
    dt = dict(b["expected_dtypes"])

    # (1) re-validate the EXISTING baseline before touching it
    if SchemaDriftAgent.hash_schema(dt) != b["expected_schema_hash"]:
        print("ABORT: existing baseline is self-inconsistent (stored hash != recomputed). "
              "Do not extend a corrupt baseline; rebuild via build_schema_baseline.py."); return 3
    print(f"[ok] existing baseline self-consistent: n={b['n_columns']} hash={b['expected_schema_hash'][:16]}...")

    # (2) idempotency
    present = [c for c in RNASEQ if c in dt]
    if present:
        if len(present) == len(RNASEQ):
            print(f"[no-op] all {len(RNASEQ)} rnaseq_* already present; baseline already at n={b['n_columns']}.")
            return 0
        print(f"ABORT: baseline partially contains rnaseq_* {present} -- inconsistent state, inspect manually."); return 3
    if b["n_columns"] != len(dt):
        print(f"ABORT: n_columns ({b['n_columns']}) != len(expected_dtypes) ({len(dt)})."); return 3

    # (3) insert the 5 rnaseq_* as float64, after the anchor (order irrelevant to hash)
    new_dt: dict[str, str] = {}
    inserted = False
    for k, v in dt.items():
        new_dt[k] = v
        if k == ANCHOR:
            for r in RNASEQ:
                new_dt[r] = DTYPE
            inserted = True
    if not inserted:  # anchor absent -> append (still correct; just less tidy)
        for r in RNASEQ:
            new_dt[r] = DTYPE
        print(f"[warn] anchor '{ANCHOR}' not in baseline; appended rnaseq_* at end (hash unaffected).")

    new_hash = SchemaDriftAgent.hash_schema(new_dt)
    payload = dict(b)
    payload["expected_dtypes"] = new_dt
    payload["n_columns"] = len(new_dt)
    payload["expected_schema_hash"] = new_hash
    payload["captured_from"] = (str(b.get("captured_from", "")) +
                                " + 5 rnaseq_* (float64) surgically added for Run-17 RNA-seq branch")
    payload["captured_at"] = datetime.now(timezone.utc).isoformat()

    print(f"[plan] n_columns {b['n_columns']} -> {payload['n_columns']}   "
          f"hash {b['expected_schema_hash'][:12]}... -> {new_hash[:12]}...")
    print(f"[plan] added: {RNASEQ}")

    if args.dry_run:
        print("[dry-run] no file written."); return 0

    args.baseline.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # (4) round-trip
    rl = json.loads(args.baseline.read_text(encoding="utf-8"))
    if SchemaDriftAgent.hash_schema(rl["expected_dtypes"]) != rl["expected_schema_hash"]:
        print("ABORT: round-trip hash mismatch after write."); return 3
    if rl["n_columns"] != 87 or any(c not in rl["expected_dtypes"] for c in RNASEQ):
        print("ABORT: post-write validation failed."); return 3
    print(f"[PASS] baseline extended to n={rl['n_columns']}, round-trip hash OK, all 5 rnaseq_* present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
