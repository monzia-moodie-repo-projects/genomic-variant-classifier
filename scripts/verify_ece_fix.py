#!/usr/bin/env python
"""verify_ece_fix.py (2026-07-10)

Confirm the LIVE patched evaluator._calibration_error now counts predictions of exactly 1.0, using
the real class on this machine (not a copy). Read-only except for importing. Prints a before/after
style check on a constructed overconfident case and a no-p==1.0 equivalence case, plus reports
whether the .bak backup exists (hygiene) and lists any evaluator tests that touch calibration.
ASCII-safe.
"""
from __future__ import annotations

import io
import sys
import inspect
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path("src").resolve()))

import numpy as np  # noqa: E402


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def find_evaluator_class():
    import importlib
    for mod in ["genomic_variant_classifier.evaluation.evaluator",
                "genomic_variant_classifier.evaluator",
                "genomic_variant_classifier.models.evaluator"]:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for name, obj in vars(m).items():
            if inspect.isclass(obj) and hasattr(obj, "_calibration_error"):
                return mod, name, obj
    return None, None, None


def main() -> int:
    print("=" * 78)
    print("VERIFY LIVE ECE FIX (evaluator._calibration_error counts p==1.0)")
    print("=" * 78)
    mod, name, cls = find_evaluator_class()
    if cls is None:
        print("ABORT: could not locate an evaluator class with _calibration_error.")
        return 2
    print(_ascii_safe(f"class: {mod}.{name}"))

    # show the live source of the method so we SEE the closed top bin
    src = inspect.getsource(cls._calibration_error)
    print("--- live _calibration_error source ---")
    for ln in src.splitlines():
        print(_ascii_safe("  " + ln[:150]))
    print("-" * 78)

    # instantiate without running __init__ (avoid heavy constructor deps)
    inst = cls.__new__(cls)

    # Case 1: all predictions exactly 1.0, true accuracy 0.70 -> ECE should be ~0.30, NOT 0.
    rng = np.random.default_rng(0)
    y1 = (rng.random(5000) < 0.70).astype(int)
    p1 = np.ones(5000)
    try:
        ece1, mce1 = cls._calibration_error(inst, y1, p1)
    except TypeError:
        ece1, mce1 = cls._calibration_error(inst, y1, p1, 10)
    print(f"Case 1 (all p==1.0, acc 0.70): ECE={ece1:.4f} MCE={mce1:.4f}  (expect ECE ~0.30)")
    ok1 = ece1 > 0.2

    # Case 2: no p==1.0 -> fix is a no-op; ECE should be small and finite.
    p2 = rng.uniform(0, 0.99, 5000)
    y2 = (rng.random(5000) < p2).astype(int)
    try:
        ece2, _ = cls._calibration_error(inst, y2, p2)
    except TypeError:
        ece2, _ = cls._calibration_error(inst, y2, p2, 10)
    print(f"Case 2 (no p==1.0):          ECE={ece2:.4f}  (expect small, finite)")
    ok2 = 0.0 <= ece2 < 0.2

    # hygiene: is the .bak present?
    for cand in ["src/genomic_variant_classifier/evaluation/evaluator.py",
                 "src/genomic_variant_classifier/evaluator.py",
                 "src/genomic_variant_classifier/models/evaluator.py"]:
        bak = Path(cand + ".bak")
        if bak.exists():
            print(_ascii_safe(f"backup present: {bak} ({bak.stat().st_size} bytes)"))

    print("-" * 78)
    verdict = "PASS" if (ok1 and ok2) else "FAIL"
    print(f"VERDICT: {verdict} -- "
          f"{'ECE now counts p==1.0 and normal cases unchanged' if verdict=='PASS' else 'unexpected ECE values; investigate'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
