"""Exact statistical inference for the scientific stage (owner rulings 2026-10-02 / 2026-10-03 / 2026-10-03b).

exact_confirmation -- the conservative gene-level conjunction test and exact Holm (CONFIRMATION, not ranking).
endpoints          -- known-positive recovery at k and the primary contrast; assay-yield bounds (EVALUATION).
analysis_contract  -- the draft and sealed pre-registration contract (a PLAN, not a result).
ranking            -- DANDELION with minimum-score gene aggregation, the top-k tie audit, universe and method
                      identities, and the numerical sensitivity audit (the primary extended RANKING method's adapter).
backend_trace      -- strict post-processing of the R actual-call trace of DANDELION's q-value backend
                      (scripts/dandelion/dandelion_backend_recorder.R): digests, completeness, admissibility.
exposure_outcomes  -- one status per planned exposure from the actual-call exposure recorder, the frozen PairPlan, coverage
                      and the coverage rule, the exploratory partial ranking, the release-policy table and its identity, and
                      the shared burden-input summary (rulings 2026-10-08g, 2026-10-09).
method_trace       -- the predetermined method fixtures and the layered adjustment-to-endpoint judge (rulings 2026-10-08e/f/g).
run_intent         -- the pre-execution run and evaluation intents: stage, contract, plan, inputs, environment and release-policy
                      identities, sealed before execution and admitted only against their persisted digests (ruling 2026-10-09).
evaluation_boundary-- the derived execution assessment, the release decision and the reference-evaluator boundary: reference
                      evidence is opened only when the admitted intents and the derived assessment permit it (ruling 2026-10-09).
Pure Python standard library; no logging configuration (library convention).
"""
from __future__ import annotations
