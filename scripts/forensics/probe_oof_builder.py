#!/usr/bin/env python
"""probe_oof_builder.py (2026-07-10)

Resolve the AUTHORITATIVE meaning of oof_predictions._train_row_idx for run15_baseline, so the
conformal calibration labels can be joined correctly (not guessed).

Established so far:
  - oof_predictions: 883,127 rows; _train_row_idx dense 0..883,126 (positional over the OOF subset).
  - train split on disk: 1,038,974 rows. 883,127 = exactly 85.0% of 1,038,974 (155,847 = 15% gap).
  - No 883,127-row artifact is persisted; the OOF subset's original-train mapping is NOT on disk.
  - The iloc[:883127] shortcut is UNSAFE unless the OOF rows are provably the first-N-in-order.

This probe does NOT guess. It:
  1. Re-reads oof_predictions schema + dtypes + the full column list (confirm no hidden id column,
     confirm _train_row_idx is the only linkage), and prints head/tail of _train_row_idx.
  2. Searches the source tree for the OOF-writing code: any .py under src/ or scripts/ that
     mentions 'oof_predictions', '_train_row_idx', 'out_of_fold', 'oof', or writes that parquet.
     Prints the matching file paths and the relevant code lines (context) so the mapping logic is
     visible and authoritative.
  3. Searches for how the 85/15 reduction happens: mentions of 'complete', 'dropna', 'notna',
     'train_test_split', 'StratifiedKFold', 'KFold', 'cv', 'holdout', 'early_stop', 'sample('.
  4. Reports whether an oof-aligned label/meta file is written anywhere (grep for 'oof_label',
     'oof_meta', 'meta_oof', 'y_oof').

Read-only. ASCII-clean. Prints code excerpts for human adjudication -- no automated conclusion.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(".").resolve()
OOF = Path("outputs/run15_baseline/full/oof_predictions.parquet")
SEARCH_DIRS = ["src", "scripts"]
EXCLUDE = {".venv312", ".venv", ".git", "__pycache__", "site-packages", "node_modules"}

PATTERNS = [
    r"oof_predictions", r"_train_row_idx", r"out[_-]?of[_-]?fold", r"\boof\b",
    r"train_test_split", r"StratifiedKFold", r"KFold", r"\.dropna\(", r"notna\(",
    r"complete_cases", r"early_stop", r"holdout", r"\.sample\(", r"meta_oof",
    r"oof_label", r"y_oof", r"oof_meta",
]
RX = re.compile("|".join(PATTERNS), re.IGNORECASE)


def _ascii_safe(s: str) -> str:
    """Make a string safe to print on any console (e.g. Windows cp1252) by replacing any
    non-ASCII byte. Scanned repo files may contain non-ASCII characters; echoing them raw
    would crash on a cp1252 stdout. This sanitizes only what we PRINT, not the files."""
    return s.encode("ascii", "replace").decode("ascii")


def line(c="-", n=78):
    print(c * n)


def main():
    print("=" * 78)
    print("OOF-BUILDER PROBE (authoritative _train_row_idx meaning for run15_baseline)")
    print("=" * 78)

    # 1. oof schema
    if OOF.exists():
        oof = pd.read_parquet(OOF)
        print(f"oof_predictions: {len(oof):,} rows x {len(oof.columns)} cols")
        print(_ascii_safe(f"  columns: {list(oof.columns)}"))
        idx = oof["_train_row_idx"].values
        print(f"  _train_row_idx head: {list(idx[:5])}  tail: {list(idx[-5:])}")
        print(f"  dtype: {oof['_train_row_idx'].dtype}  min {idx.min()}  max {idx.max()}")
        # is it exactly range(N)?
        import numpy as np
        is_range = bool(np.array_equal(idx, np.arange(len(oof))))
        print(f"  _train_row_idx == range(0, {len(oof)}) exactly: {is_range}")
        print("  (if True: it is a DENSE positional index over the OOF subset, carrying NO")
        print("   original-train position -> mapping to train labels is NOT in this file.)")
    else:
        print(f"oof_predictions ABSENT at {OOF}")
    line()

    # 2/3/4. search source for the builder + reduction logic
    print("Source files mentioning OOF / row-index / CV / reduction logic:")
    hits = []
    for base in SEARCH_DIRS:
        bp = ROOT / base
        if not bp.exists():
            continue
        for dp, dn, fn in os.walk(bp):
            dn[:] = [d for d in dn if d not in EXCLUDE]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = Path(dp) / f
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    print(f"  [read error] {p}: {e}")
                    continue
                matches = [(i + 1, ln.rstrip()) for i, ln in enumerate(text.splitlines())
                           if RX.search(ln)]
                if matches:
                    hits.append((p.relative_to(ROOT), matches))

    if not hits:
        print("  NONE found under src/ or scripts/. The OOF builder may live elsewhere or in a")
        print("  notebook. Report and stop -- do not guess the mapping.")
    for rel, matches in hits:
        print(_ascii_safe(f"\n  FILE: {rel}  ({len(matches)} matching line(s))"))
        for lineno, ln in matches[:40]:
            print(_ascii_safe(f"    {lineno:5d}: {ln[:160]}"))
        if len(matches) > 40:
            print(f"    ... {len(matches) - 40} more")
    line("=")

    # ---- targeted: which file WRITES oof_predictions or ASSIGNS _train_row_idx? (the authority) ----
    print("WRITER DETECTION (files that WRITE oof_predictions or ASSIGN _train_row_idx):")
    writer_rx = re.compile(
        r"(oof_predictions.{0,40}to_parquet|to_parquet.{0,40}oof|_train_row_idx\s*=|"
        r"['\"]_train_row_idx['\"]|assign\(.{0,40}_train_row_idx)",
        re.IGNORECASE)
    writers = []
    for base in SEARCH_DIRS:
        bp = ROOT / base
        if not bp.exists():
            continue
        for dp, dn, fn in os.walk(bp):
            dn[:] = [d for d in dn if d not in EXCLUDE]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = Path(dp) / f
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                wl = [(i + 1, ln.rstrip()) for i, ln in enumerate(text.splitlines())
                      if writer_rx.search(ln)]
                if wl:
                    writers.append((p.relative_to(ROOT), wl))
    if not writers:
        print("  NO file writes oof_predictions or assigns _train_row_idx under src/ or scripts/.")
        print("  The OOF writer may be in a notebook or archived script. Report -- do not guess.")
    for rel, wl in writers:
        print(_ascii_safe(f"  WRITER FILE: {rel}"))
        for lineno, ln in wl[:20]:
            print(_ascii_safe(f"    {lineno:5d}: {ln[:160]}"))
    line("=")
    print("PROBE COMPLETE -- read the matching code to determine the authoritative oof->label map.")
    print("Do NOT wire calibrate.py until the mapping is confirmed from the builder source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
