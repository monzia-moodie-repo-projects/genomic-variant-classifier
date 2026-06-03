# INCIDENT 2026-06-03 — ntqr API incompatibility

**Status:** OPEN — ntqr_evaluator.py in permanent stub mode  
**Severity:** LOW — ntqr_evaluator.py gracefully stubs; no pipeline impact  
**Discovered:** 2026-06-03, SR #31 smoke test run  
**HEAD at discovery:** post-D.1/D.2 commit (553efac + D.1/D.2 changes)

---

## What happened

Running `docs/preflight/ntqr_sr31_check.ps1` revealed:

```
ImportError: cannot import name 'Evaluator2' from 'ntqr.r2'
```

ntqr 0.8 (the current PyPI version as of 2026-06-03) does **not** contain the
`Evaluator2` class that `ntqr_evaluator.py` was written against.

---

## ntqr 0.8 actual API

ntqr 0.8 is designed for evaluating **trios of classifiers** without labelled
test data, using the No-Test-Quorums-Required algebraic approach. It does NOT
support the single binary-classifier accuracy bounds use case we need.

Relevant classes in `ntqr.r2.evaluators` (ntqr 0.8):

| Class | Takes | Methods |
|-------|-------|---------|
| `ErrorIndependentEvaluation` | `TrioVoteCounts` | `classifier_a_label_accuracy`, `classifier_b_label_accuracy` |
| `MajorityVotingEvaluation` | `TrioVoteCounts` | `classifier_label_accuracy`, `prevalences` |
| `SupervisedEvaluation` | `TrioLabelVoteCounts` | `classifier_label_accuracy`, `prevalences` |
| `TrioVoteCounts` | `{tuple: int}` vote map | — |
| `TrioLabelVoteCounts` | label vote counts | — |

Our `ntqr_evaluator.py` was written for a hypothetical `Evaluator2(n_0, n_1)` API
with `classifier_accuracy_bounds(q_00, q_01, q_10, q_11)` method. This API does
not exist in any currently available ntqr version.

---

## Impact

`ntqr_evaluator.py` already handles this correctly:

```python
_NTQR_AVAILABLE = False
try:
    from ntqr.r2 import Evaluator2 as _Evaluator2
    _NTQR_AVAILABLE = True
except ImportError:
    logger.warning("ntqr not installed — NTQREvaluator will return stub bounds (None).")
```

When ntqr 0.8 is installed, the `ImportError` is caught and `_NTQR_AVAILABLE = False`.
All `NTQREvaluator.evaluate()` calls return `NTQRBounds` with all bounds `None`.
**No pipeline failure. No silent error. Stub mode is correct.**

---

## What NOT to do

- Do NOT add `ntqr` to `requirements.txt` (will install an incompatible version)
- Do NOT attempt to rewrite `ntqr_evaluator.py` for `TrioVoteCounts` — this API
  requires 3 classifiers, which is not our use case

---

## Resolution path

Before `ntqr` can be productionised:

1. Identify whether ntqr has a version with binary-classifier `Evaluator2` support,
   OR whether a different package implements the same NTQR algebraic bounds
   for the binary single-classifier case.

2. Alternatively, implement the NTQR binary accuracy bounds algebraically from the
   paper (Platzer & Schmidhuber 2023) without using the ntqr package at all.
   The algebra is:
   - Given: n_0 (benign), n_1 (pathogenic), q_00, q_01, q_10, q_11
   - Solve: for accuracy_0 ∈ [0,1] and accuracy_1 ∈ [0,1]
   - Constraint: q_00 + q_01 = n_0, q_10 + q_11 = n_1
   - NTQR bounds are determined by the solution set of the system

3. Flag for Phase 3 evaluation (see RUN_15_PLAN.md).

---

## Files affected

- `src/genomic_variant_classifier/evaluation/ntqr_evaluator.py` — in stub mode; no change needed
- `docs/preflight/ntqr_sr31_check.ps1` — updated to report INCOMPATIBLE_API (not FAIL)
- `requirements.txt` — ntqr NOT added
