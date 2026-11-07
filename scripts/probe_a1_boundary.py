"""
scripts/probe_a1_boundary.py
============================
Author: Monzia Moodie

One-shot probe of the Run 14 anomaly A1 (a possible np.log(0) in the MC-dropout entropy) against the CURRENT code.

History. The function this probe originally imported, mc_dropout._decompose_uncertainty, clipped its input to
[1e-8, 1 - 1e-8] -- which is what made A1 a false anomaly (and, in float32, caused INCIDENT_2026-06-01). Package 1,
set 1a (2026-11-07) retired it: models/disagreement.draw_decomposition computes the schema-2 quantities with an exact
entropy (0 log 0 := 0) and NO clipping, so boundary draws now give exact zeros rather than a clip-floor value. This
probe exercises the replacement with the same four inputs. Exit 0 when every value is finite and the exact
expectations hold; exit 1 otherwise; exit 2 when the package cannot be imported.
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from genomic_variant_classifier.models.disagreement import draw_decomposition
except Exception as exc:
    print(f"IMPORT FAIL: {exc}", file=sys.stderr)
    print("Hint: ensure `pip install -e .` has been run in the venv.", file=sys.stderr)
    raise SystemExit(2)


def banner(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def check(name: str, record: dict, expected_entropy=None) -> bool:
    entropy = record["expected_draw_entropy_nats"]
    spread = record["between_draw_spread_variance_probability_squared"]
    ok = bool(np.isfinite(record["mixture_mean"]).all() and np.isfinite(entropy).all() and np.isfinite(spread).all())
    if expected_entropy is not None:
        ok = ok and bool(np.allclose(entropy, expected_entropy, rtol=1e-15, atol=0.0))
    print(f"  mixture mean                 : {record['mixture_mean']}")
    print(f"  expected draw entropy (nats) : {entropy}")
    print(f"  between-draw spread (prob^2) : {spread}")
    print(f"  disagreement status          : {record['disagreement_status']}")
    print(f"  -> {name}: {'PASS' if ok else 'FAIL'}")
    return ok


all_ok = True

banner("Probe 1: all-zero draws (exact zero entropy expected; no clip floor)")
all_ok &= check("zeros", draw_decomposition(np.zeros((10, 3))), expected_entropy=0.0)

banner("Probe 2: all-one draws (exact zero entropy expected)")
all_ok &= check("ones", draw_decomposition(np.ones((10, 3))), expected_entropy=0.0)

banner("Probe 3: mixed boundary values (0.0, 1.0, 1e-15, 0.5, 1 - 1e-15)")
all_ok &= check("mixed", draw_decomposition(np.array([[0.0, 1.0, 1e-15, 0.5, 1.0 - 1e-15]] * 3)))

banner("Probe 4: interior p = 0.5 (log 2 expected)")
all_ok &= check("interior_0.5", draw_decomposition(np.full((20, 3), 0.5)), expected_entropy=np.log(2.0))

banner("Verdict")
if all_ok:
    print("A1 does not arise: the exact entropy is finite at both boundaries, with no clipping.")
    raise SystemExit(0)
print("A NON-FINITE OR INEXACT VALUE WAS OBSERVED. Investigate.")
raise SystemExit(1)
