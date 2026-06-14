# Session 2026-06-14 -- Run-17 launch-readiness audit

## Question
"I am ready to do Run 17. Is everything set to go?"

## Verdict
NOT a one-command go. The entrypoint (run_phase2_eval.py) and gating code are ready and the suite is
green (1194/7), but af_1kg_* activation depends on a 1000G per-superpop AF parquet that is not built
(registry.py:112: 1kgp/1000genomes dirs EMPTY -> kg_path silent-zero), and Gate F had no script.

## Done this session
- Verified the real flag surface of run_phase2_eval.py (all Run-17 flags present; earlier
  "missing flags" reading was a single-line grep artifact on multi-line add_argument calls).
- Confirmed Gate-A DECISION (n_pathogenic_in_gene scope) is CLOSED (2026-06-13).
- Built scripts/preflight_run17.py + 15 tests (Gate F): kg activate-XOR-defer, 81-col schema gate,
  hard-gate-scripts check, drift-proof command emit.
- Reconciled the two RUN17_SCOPE copies (stale roadmap copy -> pointer; canonical gets the addendum).

## Open decision (gates the launch command)
KG: (A) build the per-superpop 1000G AF parquet and run with --kg, or (B) --defer-kg and run Run 17
with gnn_score-only (af_1kg_* deferred to Run 18). User to decide.

## Next
Once decided: run preflight_run17.py (GO), then the all-models smoke (smoke_all_models.py) on the
box, then the gated launch -> monitor -> fetch -> destroy.
