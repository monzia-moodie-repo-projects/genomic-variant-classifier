#!/usr/bin/env python
"""fix_ece_binning.py (2026-07-10)

Guarded single-match fix for the Expected Calibration Error (ECE) top-bin bug in
evaluator.py::_calibration_error. The final bin was half-open [0.9, 1.0), silently dropping every
prediction of exactly 1.0 (pure tree/ensemble leaves) from the ECE/MCE sum while still counting
them in the denominator, biasing ECE LOW (demonstrated: up to ~89% under-report on overconfident
leaves). Fix: close the FINAL bin to [lo, 1.0] inclusive so p==1.0 is counted. No other bin
changes; the fix is a no-op when no prediction equals exactly 1.0.

Aborts if the exact expected code is not found (count != 1), leaving the file untouched.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

CANDIDATES = [
    "src/genomic_variant_classifier/evaluation/evaluator.py",
    "src/genomic_variant_classifier/evaluator.py",
    "src/genomic_variant_classifier/models/evaluator.py",
]

OLD = """        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        n = len(y)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)"""

NEW = """        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        n = len(y)
        _n_bins_used = len(bin_edges) - 1
        for _b, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            # Close the FINAL bin to [lo, 1.0] so predictions of exactly 1.0 (pure tree/ensemble
            # leaves) are counted, not silently dropped. Half-open [lo, hi) elsewhere.
            # Fixed 2026-07-10: top-bin bug under-reported ECE (see docs/incidents).
            if _b == _n_bins_used - 1:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)"""


def main() -> int:
    target = None
    for c in CANDIDATES:
        if Path(c).exists():
            target = Path(c)
            break
    if target is None:
        print("ABORT: evaluator.py not found in known locations:")
        for c in CANDIDATES:
            print(f"  {c}")
        return 2
    text = target.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count != 1:
        print(f"ABORT: expected exactly 1 match of the target block, found {count} in {target}.")
        print("File left UNTOUCHED. The function body may differ from the dumped version; re-dump.")
        return 3
    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    # verify it compiles
    import py_compile
    try:
        py_compile.compile(str(target), doraise=True)
    except Exception as e:
        target.write_text(text, encoding="utf-8")  # roll back
        print(f"ABORT: post-edit compile failed ({e}); rolled back from backup.")
        return 4
    print(f"OK: patched {target} (backup at {backup.name}). Top bin now closed [lo, 1.0].")
    print("Re-run the evaluator tests / a calibration check to confirm ECE now counts p==1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
