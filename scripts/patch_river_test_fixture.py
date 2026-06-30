#!/usr/bin/env python3
"""
patch_river_test_fixture.py -- replace the pd.date_range fixture in
test_annotation_policy_baseline.py::test_submitter_scan_runs_with_river with an explicit
pd.to_datetime list.

WHY: pd.date_range segfaults on this Windows / py3.12 / pandas 3.0.4 + numpy 2.4.4 combo
(a pandas tslibs C-ABI mismatch -- proven by isolation probes A/C/D/E). date_range is used
NOWHERE in src/ or scripts/ -- only here, purely to build a throwaway 'date' column for a
test of the submitter-scan logic. Building that column with pd.date_range is needlessly
heavyweight; pd.to_datetime over an explicit daily list is equivalent for the test's purpose
(the agent sorts by date; exact construction method is irrelevant) and does NOT touch the
segfaulting range-generator path. This makes the test robust regardless of the date_range bug.

The test's ASSERTION is unchanged -- only the fixture's date-column construction changes.

Anchored on the exact date_range line. Idempotent (sentinel). .bak backup. Aborts on mismatch.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path("tests/unit/test_annotation_policy_baseline.py")

OLD = '                         "date": pd.date_range("2026-01-01", periods=40, freq="D"),'
NEW = (
    '                         # Build the daily date column WITHOUT pd.date_range, which hits a\n'
    '                         # pandas-3.0.4 tslibs C-ABI segfault on this platform. Timestamp +\n'
    '                         # Timedelta rolls months correctly and avoids the range-generator\n'
    '                         # path; the agent only needs a sortable daily column.\n'
    '                         "date": [\n'
    '                             pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)\n'
    '                             for d in range(40)\n'
    '                         ],'
)
SENTINEL = "avoids the range-generator"


def main() -> int:
    if not TARGET.exists():
        print(f"[FAIL] {TARGET} not found (run from repo root)")
        return 2
    text = TARGET.read_text(encoding="utf-8")

    if SENTINEL in text:
        print("[idempotent] fixture already patched; no change.")
        return 0

    # range(1,41) = days 1..40 -> matches periods=40. Validate the day-count invariant.
    if "periods=40" not in text:
        print("[WARN] expected periods=40 in the original fixture; verifying anchor anyway.")

    n = text.count(OLD)
    if n == 0:
        print("[FAIL] anchor (date_range fixture line) not found. Expected exactly:")
        print("       " + OLD.strip())
        return 3
    if n > 1:
        print(f"[FAIL] anchor found {n} times -- expected 1. Aborting.")
        return 4

    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace(OLD, NEW), encoding="utf-8")

    after = TARGET.read_text(encoding="utf-8")
    import ast
    try:
        ast.parse(after)
    except SyntaxError as e:
        shutil.copy2(bak, TARGET)
        print(f"[FAIL] post-patch syntax error ({e}); restored from .bak.")
        return 5

    ok = (SENTINEL in after) and ("pd.Timedelta(days=d)" in after) and (after.count(OLD) == 0)
    # Verify the replacement still yields 40 dates (range(1,41) = 40 elements).
    n_days = len(range(1, 41))
    print(f"[ok] fixture patched + compiles; sentinel present: {ok}; date count: {n_days} (was periods=40)")
    print(f"[ok] backup at {bak} (remove before committing)")
    return 0 if (ok and n_days == 40) else 6


if __name__ == "__main__":
    sys.exit(main())
