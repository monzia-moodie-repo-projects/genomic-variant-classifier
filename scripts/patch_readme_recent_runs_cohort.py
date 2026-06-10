#!/usr/bin/env python3
"""patch_readme_recent_runs_cohort.py -- fix the lone residual current-state claim.

The "publication snapshot" metrics (1,197,216-variant cohort, 0.9847, per-model
table) are a deliberately frozen reference and are LEFT untouched. Only the
non-snapshot clause "recent runs use the full 1.70 M-variant matrix" is corrected
(recent runs = Run 14/15 = ~1.49 M cohort). Count-guarded, idempotent, backup-first.
Author: Monzia Moodie."""
from __future__ import annotations

import datetime as _dt
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RM = REPO / "README.md"
OLD = "publication snapshot; recent runs use the full 1.70 M-variant matrix."
NEW = "publication snapshot; recent runs (Run 14/15) use the full ~1.49 M-variant cohort."
MARKER = "~1.49 M-variant cohort"


def main() -> int:
    if not RM.exists():
        print(f"ABORT: missing {RM}")
        return 2
    raw = RM.read_bytes().decode("utf-8")
    if MARKER in raw:
        print("  skip (already applied): recent-runs cohort 1.70M -> 1.49M")
        return 0
    n = raw.count(OLD)
    if n != 1:
        print(f"ABORT: anchor found {n}x (expected 1); nothing written")
        return 3
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(RM, f"{RM}.bak_{ts}")
    RM.write_bytes(raw.replace(OLD, NEW, 1).encode("utf-8"))
    print(f"  ok: recent-runs cohort 1.70M -> ~1.49M  (backup -> README.md.bak_{ts})")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
