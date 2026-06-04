#!/usr/bin/env python3
"""Post-run guard: verify gnn_score is non-degenerate in the persisted splits.

The GNN block in scripts/run_phase2_eval.py wraps training in `except Exception`
and continues on failure, so a run can exit 0 with gnn_score silently left at its
0.0 default (how Run 14 went GNN-blind). This turns that into a hard gate: it reads
the re-persisted X_*.parquet (Patch 6a output) and asserts gnn_score actually varies
(nunique > 1 AND std > 0). A column that is entirely the default collapses to
nunique==1 / std==0 and FAILS. Exit 0 = real GNN scores; 1 = degenerate (GNN likely
failed and was swallowed; inspect the [GNN-TRACE] / 'GNN training failed' log lines).

Unmapped genes legitimately score 0, so we do NOT require all-nonzero; we require
variation, which is present iff at least some genes received real scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SPLIT_FILES = ("X_train.parquet", "X_val.parquet", "X_test.parquet")


def verify(splits_dir: Path) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    for f in SPLIT_FILES:
        p = splits_dir / f
        if not p.exists():
            msgs.append(f"[FAIL] {f}: missing ({p})")
            ok = False
            continue
        try:
            import pyarrow.parquet as pq

            names = pq.read_schema(p).names
        except Exception as e:  # noqa: BLE001
            msgs.append(f"[FAIL] {f}: cannot read schema ({e})")
            ok = False
            continue
        if "gnn_score" not in names:
            msgs.append(f"[FAIL] {f}: no gnn_score column")
            ok = False
            continue
        s = pd.read_parquet(p, columns=["gnn_score"])["gnn_score"]
        nuniq = int(s.nunique(dropna=False))
        std = float(s.std()) if len(s) else 0.0
        nonzero = float((s != 0).mean()) if len(s) else 0.0
        degenerate = (nuniq <= 1) or (std == 0.0)
        if degenerate:
            ok = False
        msgs.append(
            f"[{'FAIL' if degenerate else 'PASS'}] {f}: rows={len(s):,} "
            f"nunique={nuniq} std={std:.6f} nonzero_frac={nonzero:.4f}"
        )
    return ok, msgs


def main(splits_dir_str: str) -> int:
    ok, msgs = verify(Path(splits_dir_str))
    print("== GNN-SCORE POST-RUN VERIFY ==")
    for m in msgs:
        print("  " + m)
    print(
        "\nVERDICT:",
        "OK (gnn_score is real)"
        if ok
        else "DEGENERATE (gnn_score did not vary; GNN likely failed and was "
        "swallowed by the except in run_phase2_eval.py; check [GNN-TRACE] and "
        "'GNN training failed' in the training log before trusting this run)",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "outputs/run15_baseline/full/splits"
    sys.exit(main(d))
