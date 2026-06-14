# Run 17 Scope -- POINTER (canonical doc moved)

The canonical Run-17 scope & pre-flight gate is **docs/runs/RUN17_SCOPE.md**.

This file previously held an earlier plan that referenced `--kg-path` threaded into
`scripts/train.py`. That framing is **SUPERSEDED** and was wrong against the code:
the activation runs through **`scripts/run_phase2_eval.py`** with **`--kg`** (not `--kg-path`,
not `train.py`). See docs/runs/RUN17_SCOPE.md and its AUDIT ADDENDUM 2026-06-14.

Author: Monzia Moodie.
