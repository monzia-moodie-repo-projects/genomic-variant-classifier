# INCIDENT 2026-06-01 — mc_dropout NaN entropy

## Summary

`mc_dropout` uncertainty decomposition emitted NaN aleatoric/total entropy for fully-confident predictions, with "divide by zero in log" and "invalid value in multiply" runtime warnings.

## Root Cause

In `src/genomic_variant_classifier/models/mc_dropout.py`, `_decompose_uncertainty` clipped probabilities to `[eps, 1-eps]` with `eps=1e-8`. In float32, machine epsilon is ~1.19e-7, so `1 - 1e-8` rounds to exactly `1.0`. A confident prediction then yields `1 - p_clipped = 0`, and the binary entropy term `p*log(p) + (1-p)*log(1-p)` computes `0 * log(0) = 0 * -inf = NaN`.

## Fix

Replaced the eps-clip entropy with exact binary entropy using the mathematical convention `0*log(0) := 0`, implemented via `np.where` masking (no eps, no clip floor, dependency-free, warning-free). A fully-confident prediction now yields entropy exactly 0.

A first attempt (`eps=1e-6` + `log1p`) was superseded because it introduced a clip-floor entropy `H(1e-6) ≈ 1.48e-5` that tripped the pre-existing boundary test `tests/unit/test_mc_dropout_uncertainty.py::TestDecomposeUncertaintyBoundary` (asserts boundary aleatoric < 1e-5).

## Validation

Both entropy test files pass; confident-pass entropy is exactly 0; no runtime warnings. Full unit suite 651 passed / 1 skipped / 0 failed.

## Lessons

- Float32 clip bounds must exceed machine epsilon (~1.19e-7); 1e-8 is below it.
- Prefer the exact `0*log(0):=0` convention over eps-clipping for entropy.

## Status

FIXED. Applied to working tree; commit pending.
