"""
scripts/probe_a1_boundary.py
============================
One-shot probe: import the real _decompose_uncertainty from the codebase
and exercise it with boundary inputs (probs_stack containing exactly 0.0
and 1.0). Run 14 anomaly catalogue A1 flagged a potential np.log(0) at
mc_dropout.py:87. Inspection showed line 86 already clips inputs to
[eps, 1 - eps], so the log should be finite. This probe confirms with
the real function.

If A1 were real, this script would print NaN/inf and exit non-zero.
"""
from __future__ import annotations

import sys
import numpy as np

try:
    from genomic_variant_classifier.models.mc_dropout import _decompose_uncertainty
except Exception as exc:
    print(f"IMPORT FAIL: {exc}", file=sys.stderr)
    print("Hint: ensure `pip install -e .` has been run in the venv.", file=sys.stderr)
    raise SystemExit(2)


def banner(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def check_finite(name: str, mean_p, epi, alea) -> bool:
    ok = (np.all(np.isfinite(mean_p))
          and np.all(np.isfinite(epi))
          and np.all(np.isfinite(alea)))
    print(f"  mean_prob : finite = {bool(np.all(np.isfinite(mean_p)))}, values = {mean_p}")
    print(f"  epistemic : finite = {bool(np.all(np.isfinite(epi)))}, values = {epi}")
    print(f"  aleatoric : finite = {bool(np.all(np.isfinite(alea)))}, values = {alea}")
    print(f"  -> {name}: {'PASS' if ok else 'FAIL'}")
    return ok


all_ok = True

banner("Probe 1: all-zero probs_stack (worst case lower boundary)")
probs = np.zeros((10, 3))
mean_p, epi, alea = _decompose_uncertainty(probs)
all_ok &= check_finite("zeros", mean_p, epi, alea)
expected_boundary_alea = -(1e-8) * np.log(1e-8) - (1.0 - 1e-8) * np.log(1.0 - 1e-8)
print(f"  theoretical boundary aleatoric: {expected_boundary_alea:.6e}")

banner("Probe 2: all-one probs_stack (worst case upper boundary)")
probs = np.ones((10, 3))
mean_p, epi, alea = _decompose_uncertainty(probs)
all_ok &= check_finite("ones", mean_p, epi, alea)

banner("Probe 3: mixed boundary values (0.0, 1.0, 1e-15, 0.5)")
probs = np.array([
    [0.0, 1.0, 1e-15, 0.5, 1.0 - 1e-15],
    [0.0, 1.0, 1e-15, 0.5, 1.0 - 1e-15],
    [0.0, 1.0, 1e-15, 0.5, 1.0 - 1e-15],
])
mean_p, epi, alea = _decompose_uncertainty(probs)
all_ok &= check_finite("mixed", mean_p, epi, alea)

banner("Probe 4: interior p=0.5 (sanity -- should give log(2) ~ 0.693)")
probs = np.full((20, 3), 0.5)
mean_p, epi, alea = _decompose_uncertainty(probs)
all_ok &= check_finite("interior_0.5", mean_p, epi, alea)
print(f"  theoretical entropy at p=0.5: log(2) = {np.log(2):.6f}")
print(f"  match within 1e-9? {bool(np.allclose(alea, np.log(2), atol=1e-9))}")

banner("Verdict")
if all_ok:
    print("A1 is a CONFIRMED FALSE ANOMALY.")
    print("Line 86 clip (eps=1e-8) fully protects line 87 log.")
    raise SystemExit(0)
else:
    print("A1 IS REAL: non-finite values observed. Investigate.")
    raise SystemExit(1)