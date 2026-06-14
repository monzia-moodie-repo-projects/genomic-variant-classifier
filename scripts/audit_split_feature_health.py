"""
scripts/audit_split_feature_health.py
=====================================
READ-ONLY staleness/health audit of an existing split set, plus a git-log of
annotation code changed since those splits were built. Answers, by measurement
rather than memory: which feature columns are degenerate (silent-zero / all-null
/ constant), which are healthy, and which annotation connectors changed since
the splits were produced.

Discovers split files by glob (does NOT assume filenames or column layout).
Run it BEFORE a regen to scope it, and AFTER to verify the regen fixed the dead
columns and broke none of the healthy ones.

A column is DEGENERATE if (among non-null values) it is: all-null, all-zero,
n_unique <= 1, or one value covers >= --near-constant-frac of non-null rows.

USAGE
-----
  python scripts/audit_split_feature_health.py
  python scripts/audit_split_feature_health.py --splits-dir outputs/run15_rerun_report/full/splits --out split_health.csv

EXIT: 0 ok | 1 splits dir / parquet not found
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.data.feature_health import col_health as _col_health

# substring tags for features we specifically want eyes on (no exact-name assumptions)
_WATCH = ["esm2", "eve", "gnn", "phylop", "gtex", "splice", "alphamiss",
          "cadd", "loeuf", "gerp", "pli", "sift", "revel", "pathogenic_in_gene"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="split feature-health + staleness audit")
    ap.add_argument("--splits-dir", default="outputs/run15_rerun_report/full/splits")
    ap.add_argument("--data-dir", default="src/genomic_variant_classifier/data")
    ap.add_argument("--near-constant-frac", type=float, default=0.999)
    ap.add_argument("--out", default=None, help="optional CSV of full per-(file,column) health")
    args = ap.parse_args(argv)

    sd = Path(args.splits_dir)
    if not sd.exists():
        print(f"splits dir not found: {sd.resolve()}  -- STOP.")
        return 1
    files = sorted(p for p in sd.rglob("*.parquet"))
    if not files:
        print(f"no *.parquet under {sd.resolve()}  -- STOP.")
        return 1

    print(f"splits dir : {sd.resolve()}")
    print(f"parquets   : {len(files)}")
    rows = []
    newest_mtime = 0.0
    for f in files:
        newest_mtime = max(newest_mtime, f.stat().st_mtime)
        try:
            df = pd.read_parquet(f)
        except Exception as e:  # read failure must be loud, not silent
            print(f"  !! FAILED to read {f.name}: {e}")
            return 1
        print(f"  {f.relative_to(sd)} : {df.shape[0]:,} rows x {df.shape[1]} cols")
        for c in df.columns:
            h = _col_health(df[c], args.near_constant_frac)
            h["file"] = f.relative_to(sd).as_posix()
            h["column"] = c
            rows.append(h)

    health = pd.DataFrame(rows)
    if args.out:
        health.to_csv(args.out, index=False)
        print(f"\nfull per-(file,column) health written: {Path(args.out).resolve()}")

    # cross-file degeneracy: a column is flagged if degenerate in ANY file it appears in
    deg = (health[health["degenerate"] != ""]
           .groupby("column")["degenerate"].agg(lambda s: sorted(set(s))[0]))
    all_cols = sorted(health["column"].unique())
    healthy = [c for c in all_cols if c not in deg.index]

    print(f"\n=== DEGENERATE columns (regen targets): {len(deg)} ===")
    if len(deg):
        for c in deg.index:
            print(f"  {c:40s} {deg[c]}")
    else:
        print("  (none -- all columns carry signal)")

    print(f"\n=== watched features present? ===")
    cols_lower = {c.lower(): c for c in all_cols}
    for tag in _WATCH:
        hits = [orig for low, orig in cols_lower.items() if tag in low]
        if not hits:
            print(f"  [{tag:18s}] ABSENT")
        for h in hits:
            state = deg[h] if h in deg.index else "healthy"
            print(f"  [{tag:18s}] {h:36s} -> {state}")

    print(f"\nhealthy columns: {len(healthy)} | degenerate: {len(deg)} | total: {len(all_cols)}")

    # ---- Part B: annotation code changed since the splits were built ----
    built = datetime.fromtimestamp(newest_mtime, tz=timezone.utc)
    print(f"\n=== code touching {args.data_dir} since splits built "
          f"({built:%Y-%m-%d %H:%M UTC}) ===")
    try:
        since = built.strftime("%Y-%m-%dT%H:%M:%S")
        res = subprocess.run(
            ["git", "log", f"--since={since}", "--oneline", "--", args.data_dir],
            capture_output=True, text=True, timeout=30)
        out = res.stdout.strip()
        if res.returncode != 0:
            print(f"  (git log unavailable: {res.stderr.strip()[:120]})")
        elif not out:
            print("  (no connector commits since splits were built)")
        else:
            for line in out.splitlines():
                print(f"  {line}")
    except Exception as e:
        print(f"  (git log skipped: {e})")

    print("\nNOTE: full re-annotation will recompute every column above; this audit "
          "is the BEFORE baseline -- rerun on the new splits to confirm the dead "
          "columns came alive and the healthy ones are unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
