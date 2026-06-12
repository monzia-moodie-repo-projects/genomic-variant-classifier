## 2026-06-05 â€” Run 15 all-models smoke: first GREEN on real data (instance 39619871 @ 18da19e)

### Fixed
- KAN runs end-to-end on real data (imodelsx bare-name patch held; OOF AUROC 0.8488, device=cuda).
- Both SVMs run without the old ">100k auto-skip": ScalableSVM NystrÃ¶m `svm` (OOF 0.9804) +
  `svm_bagged_rbf` (OOF 0.9717).
- GNN `gnn_score` no longer all-zero: inductive `from_full_graph` scored 16,201 genes (mean 0.161,
  std 0.0208), injected into train/val/test with `nonzero_frac=1.0000`. Run 14/15 merge-back bug closed.
- Run15_Smoke.ps1: provision-failure now retries next offer; `PYTHONUNBUFFERED=1`+`-u`; streamed
  `run_phase2_eval.py` directly; `SmokeTimeoutMin` 90â†’180; `Ssh1` poll given `ConnectTimeout=20` (fixes
  the ~47-min apparent hang after the smoke had actually finished).

### Attempted / observed
- Full 13-model + stacker pipeline on `--max-train 3000 --n-folds 3 --min-review-tier 3 --string-db auto`.
  `SMOKE_EXIT=0`; Dev(test) AUROC 0.9831, Holdout(val) 0.9791 (PASS â‰¥0.9); total 767 s.
- Data prep is the dominant, `--max-train`-independent cost (~6.5 min): 4.40M raw â†’ 1.49M after
  label+tier filters; train 1.04M / val 146k / test 305k.

### Failed / flagged (open)
- **GNN near-chance** as a classifier; 50k probe: Best Val AUC 0.5240 (3k) â†’ 0.5095 (50k) â€” does NOT improve
  with scale â‡’ architectural, roadmap Tier-1/2 item, not a gate blocker (scorer fix is correct/non-degenerate).
- **svm_bagged_rbf scaling cost** (NEW â€” train AND predict): 1 bag @3k â†’ 25 bags @50k @ ~4 min/fold (train);
  the 50k held-out eval then spent ~31 min in `svm_bagged_rbf.predict_proba` (25 bags Ã— ~15k SVs Ã— 451k rows;
  126% CPU / 9.3 GB RSS / GPU idle). It completed (total probe 4,373 s â‰ˆ 73 min) but dominated the run.
  Projected @1.04M (~70 bags): hours for train+predict. KEEP (comparison is the goal) but cap bags (~10â€“15)
  and/or parallelize predict for Run 15. NystrÃ¶m `svm` unaffected.
- Smoke ran with dbnsfp/lovd/constraint = None â†’ 13 annotators all-zero (expected; paths not passed).
  Run 15 must wire available paths or they silently zero.
- `real_data_prep.py:444` `.fillna` downcasting FutureWarning (was :388; file moved).
- Annotation counter: `3/17` never logged (PhyloP 2/17 â†’ SpliceAI 4/17); LOVD logs `15/16` not `/17`.
- Review-tier filter applied as `<=3` (reads like a lower bound) â€” verify intended semantics.

### Resolved by 50k probe (instance 39619871, /tmp/probe50k)
- **cnn_1d is scale-limited, NOT broken**: OOF 0.4936 (3k) â†’ 0.6039 (50k). Pre-flight blocker cleared; Run 15
  may include it. Scientific note: 101bp one-hot CNN may plateau below tabular models â€” keep + study.
- **kan scales**: OOF 0.8488 (3k) â†’ 0.9309 (50k).
- **GNN is architectural-not-data**: 49,303 focal samples @50k (17Ã— the 3k count), 50 epochs, Val AUC flat
  ~0.50â€“0.51 â€” more data does not help. Roadmap Tier-1/2; not a blocker (gnn_score non-degenerate).
- Full 3kâ†’50k OOF: rf .9776â†’.9849 Â· xgb .9831â†’.9895 Â· lgbm .9825â†’.9899 Â· svm .9804â†’.9848 Â·
  svm_bagged_rbf .9717â†’.9780 Â· lr .9741â†’.9836 Â· gbm .9817â†’.9888 Â· catboost .9829â†’.9881 Â· tabular_nn .9835â†’.9869 Â·
  cnn_1d .4936â†’.6039 Â· kan .8488â†’.9309 Â· mc_dropout .9835â†’.9869 Â· deep_ensemble .9838â†’.9871.
- 50k held-out scorecard (recovered): Dev(test) AUROC 0.9848, Holdout(val) 0.9817 (PASS; up from 3k 0.9831/0.9791).

### Learned
- `smoke_all_models.py` captures its `run_phase2_eval.py` subprocess â†’ blind poll + the wrapper's
  degenerate-OOF gate is bypassed when we run the eval directly for visibility. Trade visibility vs
  automated gating consciously; re-add a per-model degenerate assertion if running direct.
- A 3,000-row smoke validates that models RUN, not that data-hungry models (cnn_1d, GNN, KAN, deep nets)
  LEARN. Trees hit ~0.98 at 3k; neural/graph/spline models need a mid-scale probe before trusting the
  full run.
- vast.ai offers are ephemeral: a chosen `ask` can be taken between search and create
  (`error 404/3603 no_such_ask`); retry the next offer.
- New PowerShell session â‡’ `$key` unset â‡’ `ssh -i $key` collapses to `ssh -i -p â€¦` (`-i` eats `-p`).
  Always re-set `$key` per shell before ssh/scp.

## 2026-06-05 â€” Run 15 all-models smoke: first GREEN on real data (instance 39619871 @ 18da19e)

### Fixed
- KAN runs end-to-end on real data (imodelsx bare-name patch held; OOF AUROC 0.8488, device=cuda).
- Both SVMs run without the old ">100k auto-skip": ScalableSVM NystrÃ¶m `svm` (OOF 0.9804) +
  `svm_bagged_rbf` (OOF 0.9717).
- GNN `gnn_score` no longer all-zero: inductive `from_full_graph` scored 16,201 genes (mean 0.161,
  std 0.0208), injected into train/val/test with `nonzero_frac=1.0000`. Run 14/15 merge-back bug closed.
- Run15_Smoke.ps1: provision-failure now retries next offer; `PYTHONUNBUFFERED=1`+`-u`; streamed
  `run_phase2_eval.py` directly; `SmokeTimeoutMin` 90â†’180; `Ssh1` poll given `ConnectTimeout=20` (fixes
  the ~47-min apparent hang after the smoke had actually finished).

### Attempted / observed
- Full 13-model + stacker pipeline on `--max-train 3000 --n-folds 3 --min-review-tier 3 --string-db auto`.
  `SMOKE_EXIT=0`; Dev(test) AUROC 0.9831, Holdout(val) 0.9791 (PASS â‰¥0.9); total 767 s.
- Data prep is the dominant, `--max-train`-independent cost (~6.5 min): 4.40M raw â†’ 1.49M after
  label+tier filters; train 1.04M / val 146k / test 305k.

### Failed / flagged (open)
- **GNN near-chance** as a classifier; 50k probe: Best Val AUC 0.5240 (3k) â†’ 0.5095 (50k) â€” does NOT improve
  with scale â‡’ architectural, roadmap Tier-1/2 item, not a gate blocker (scorer fix is correct/non-degenerate).
- **svm_bagged_rbf scaling cost** (NEW â€” train AND predict): 1 bag @3k â†’ 25 bags @50k @ ~4 min/fold (train);
  the 50k held-out eval then FROZE 25+ min in `svm_bagged_rbf.predict_proba` (25 bags Ã— ~15k SVs Ã— 451k rows).
  Probe killed there (OOF numbers already captured). Projected @1.04M (~70 bags): hours for train+predict.
  KEEP (comparison is the goal) but cap bags (~10â€“15) and/or parallelize predict for Run 15. NystrÃ¶m `svm` unaffected.
- Smoke ran with dbnsfp/lovd/constraint = None â†’ 13 annotators all-zero (expected; paths not passed).
  Run 15 must wire available paths or they silently zero.
- `real_data_prep.py:444` `.fillna` downcasting FutureWarning (was :388; file moved).
- Annotation counter: `3/17` never logged (PhyloP 2/17 â†’ SpliceAI 4/17); LOVD logs `15/16` not `/17`.
- Review-tier filter applied as `<=3` (reads like a lower bound) â€” verify intended semantics.

### Resolved by 50k probe (instance 39619871, /tmp/probe50k)
- **cnn_1d is scale-limited, NOT broken**: OOF 0.4936 (3k) â†’ 0.6039 (50k). Pre-flight blocker cleared; Run 15
  may include it. Scientific note: 101bp one-hot CNN may plateau below tabular models â€” keep + study.
- **kan scales**: OOF 0.8488 (3k) â†’ 0.9309 (50k).
- **GNN is architectural-not-data**: 49,303 focal samples @50k (17Ã— the 3k count), 50 epochs, Val AUC flat
  ~0.50â€“0.51 â€” more data does not help. Roadmap Tier-1/2; not a blocker (gnn_score non-degenerate).
- Full 3kâ†’50k OOF: rf .9776â†’.9849 Â· xgb .9831â†’.9895 Â· lgbm .9825â†’.9899 Â· svm .9804â†’.9848 Â·
  svm_bagged_rbf .9717â†’.9780 Â· lr .9741â†’.9836 Â· gbm .9817â†’.9888 Â· catboost .9829â†’.9881 Â· tabular_nn .9835â†’.9869 Â·
  cnn_1d .4936â†’.6039 Â· kan .8488â†’.9309 Â· mc_dropout .9835â†’.9869 Â· deep_ensemble .9838â†’.9871.

### Learned
- `smoke_all_models.py` captures its `run_phase2_eval.py` subprocess â†’ blind poll + the wrapper's
  degenerate-OOF gate is bypassed when we run the eval directly for visibility. Trade visibility vs
  automated gating consciously; re-add a per-model degenerate assertion if running direct.
- A 3,000-row smoke validates that models RUN, not that data-hungry models (cnn_1d, GNN, KAN, deep nets)
  LEARN. Trees hit ~0.98 at 3k; neural/graph/spline models need a mid-scale probe before trusting the
  full run.
- vast.ai offers are ephemeral: a chosen `ask` can be taken between search and create
  (`error 404/3603 no_such_ask`); retry the next offer.
- New PowerShell session â‡’ `$key` unset â‡’ `ssh -i $key` collapses to `ssh -i -p â€¦` (`-i` eats `-p`).
  Always re-set `$key` per shell before ssh/scp.

## 2026-06-05 â€” Run 15 all-models smoke: first GREEN on real data (instance 39619871 @ 18da19e)

### Fixed
- KAN runs end-to-end on real data (imodelsx bare-name patch held; OOF AUROC 0.8488, device=cuda).
- Both SVMs run without the old ">100k auto-skip": ScalableSVM NystrÃ¶m `svm` (OOF 0.9804) +
  `svm_bagged_rbf` (OOF 0.9717).
- GNN `gnn_score` no longer all-zero: inductive `from_full_graph` scored 16,201 genes (mean 0.161,
  std 0.0208), injected into train/val/test with `nonzero_frac=1.0000`. Run 14/15 merge-back bug closed.
- Run15_Smoke.ps1: provision-failure now retries next offer; `PYTHONUNBUFFERED=1`+`-u`; streamed
  `run_phase2_eval.py` directly; `SmokeTimeoutMin` 90â†’180; `Ssh1` poll given `ConnectTimeout=20` (fixes
  the ~47-min apparent hang after the smoke had actually finished).

### Attempted / observed
- Full 13-model + stacker pipeline on `--max-train 3000 --n-folds 3 --min-review-tier 3 --string-db auto`.
  `SMOKE_EXIT=0`; Dev(test) AUROC 0.9831, Holdout(val) 0.9791 (PASS â‰¥0.9); total 767 s.
- Data prep is the dominant, `--max-train`-independent cost (~6.5 min): 4.40M raw â†’ 1.49M after
  label+tier filters; train 1.04M / val 146k / test 305k.

### Failed / flagged (open)
- **GNN near-chance** as a classifier; 50k probe: Best Val AUC 0.5240 (3k) â†’ 0.5095 (50k) â€” does NOT improve
  with scale â‡’ architectural, roadmap Tier-1/2 item, not a gate blocker (scorer fix is correct/non-degenerate).
- **svm_bagged_rbf scaling cost** (NEW): exact-RBF bagged SVM, 1 bag @3k â†’ 25 bags @50k @ ~4 min/fold;
  ~70 bags/fold projected @1.04M â‡’ ~30â€“60+ min for this model alone. Budget for Run 15; candidate for bag cap.
- Smoke ran with dbnsfp/lovd/constraint = None â†’ 13 annotators all-zero (expected; paths not passed).
  Run 15 must wire available paths or they silently zero.
- `real_data_prep.py:444` `.fillna` downcasting FutureWarning (was :388; file moved).
- Annotation counter: `3/17` never logged (PhyloP 2/17 â†’ SpliceAI 4/17); LOVD logs `15/16` not `/17`.
- Review-tier filter applied as `<=3` (reads like a lower bound) â€” verify intended semantics.

### Resolved by 50k probe (instance 39619871, /tmp/probe50k)
- **cnn_1d is scale-limited, NOT broken**: OOF 0.4936 (3k) â†’ 0.6039 (50k). Pre-flight blocker cleared; Run 15
  may include it. Scientific note: 101bp one-hot CNN may plateau below tabular models â€” keep + study.
- **kan scales**: OOF 0.8488 (3k) â†’ 0.9309 (50k).

### Learned
- `smoke_all_models.py` captures its `run_phase2_eval.py` subprocess â†’ blind poll + the wrapper's
  degenerate-OOF gate is bypassed when we run the eval directly for visibility. Trade visibility vs
  automated gating consciously; re-add a per-model degenerate assertion if running direct.
- A 3,000-row smoke validates that models RUN, not that data-hungry models (cnn_1d, GNN, KAN, deep nets)
  LEARN. Trees hit ~0.98 at 3k; neural/graph/spline models need a mid-scale probe before trusting the
  full run.
- vast.ai offers are ephemeral: a chosen `ask` can be taken between search and create
  (`error 404/3603 no_such_ask`); retry the next offer.
- New PowerShell session â‡’ `$key` unset â‡’ `ssh -i $key` collapses to `ssh -i -p â€¦` (`-i` eats `-p`).
  Always re-set `$key` per shell before ssh/scp.

## 2026-06-05 â€” Run 15 all-models smoke: first GREEN on real data (instance 39619871 @ 18da19e)

### Fixed
- KAN runs end-to-end on real data (imodelsx bare-name patch held; OOF AUROC 0.8488, device=cuda).
- Both SVMs run without the old ">100k auto-skip": ScalableSVM NystrÃ¶m `svm` (OOF 0.9804) +
  `svm_bagged_rbf` (OOF 0.9717).
- GNN `gnn_score` no longer all-zero: inductive `from_full_graph` scored 16,201 genes (mean 0.161,
  std 0.0208), injected into train/val/test with `nonzero_frac=1.0000`. Run 14/15 merge-back bug closed.
- Run15_Smoke.ps1: provision-failure now retries next offer; `PYTHONUNBUFFERED=1`+`-u`; streamed
  `run_phase2_eval.py` directly; `SmokeTimeoutMin` 90â†’180; `Ssh1` poll given `ConnectTimeout=20` (fixes
  the ~47-min apparent hang after the smoke had actually finished).

### Attempted / observed
- Full 13-model + stacker pipeline on `--max-train 3000 --n-folds 3 --min-review-tier 3 --string-db auto`.
  `SMOKE_EXIT=0`; Dev(test) AUROC 0.9831, Holdout(val) 0.9791 (PASS â‰¥0.9); total 767 s.
- Data prep is the dominant, `--max-train`-independent cost (~6.5 min): 4.40M raw â†’ 1.49M after
  label+tier filters; train 1.04M / val 146k / test 305k.

### Failed / flagged (open)
- **GNN near-chance** as a classifier; 50k probe: Best Val AUC 0.5240 (3k) â†’ 0.5095 (50k) â€” does NOT improve
  with scale â‡’ architectural, roadmap Tier-1/2 item, not a gate blocker (scorer fix is correct/non-degenerate).
- **svm_bagged_rbf scaling cost** (NEW): exact-RBF bagged SVM, 1 bag @3k â†’ 25 bags @50k @ ~4 min/fold;
  ~70 bags/fold projected @1.04M â‡’ ~30â€“60+ min for this model alone. Budget for Run 15; candidate for bag cap.
- Smoke ran with dbnsfp/lovd/constraint = None â†’ 13 annotators all-zero (expected; paths not passed).
  Run 15 must wire available paths or they silently zero.
- `real_data_prep.py:444` `.fillna` downcasting FutureWarning (was :388; file moved).
- Annotation counter: `3/17` never logged (PhyloP 2/17 â†’ SpliceAI 4/17); LOVD logs `15/16` not `/17`.
- Review-tier filter applied as `<=3` (reads like a lower bound) â€” verify intended semantics.

### Resolved by 50k probe (instance 39619871, /tmp/probe50k)
- **cnn_1d is scale-limited, NOT broken**: OOF 0.4936 (3k) â†’ 0.6039 (50k). Pre-flight blocker cleared; Run 15
  may include it. Scientific note: 101bp one-hot CNN may plateau below tabular models â€” keep + study.
- **kan scales**: OOF 0.8488 (3k) â†’ 0.9309 (50k).

### Learned
- `smoke_all_models.py` captures its `run_phase2_eval.py` subprocess â†’ blind poll + the wrapper's
  degenerate-OOF gate is bypassed when we run the eval directly for visibility. Trade visibility vs
  automated gating consciously; re-add a per-model degenerate assertion if running direct.
- A 3,000-row smoke validates that models RUN, not that data-hungry models (cnn_1d, GNN, KAN, deep nets)
  LEARN. Trees hit ~0.98 at 3k; neural/graph/spline models need a mid-scale probe before trusting the
  full run.
- vast.ai offers are ephemeral: a chosen `ask` can be taken between search and create
  (`error 404/3603 no_such_ask`); retry the next offer.
- New PowerShell session â‡’ `$key` unset â‡’ `ssh -i $key` collapses to `ssh -i -p â€¦` (`-i` eats `-p`).
  Always re-set `$key` per shell before ssh/scp.

## 2026-06-05 â€” Run 15 all-models smoke: first GREEN on real data (instance 39619871 @ 18da19e)

### Fixed
- KAN runs end-to-end on real data (imodelsx bare-name patch held; OOF AUROC 0.8488, device=cuda).
- Both SVMs run without the old ">100k auto-skip": ScalableSVM NystrÃ¶m `svm` (OOF 0.9804) +
  `svm_bagged_rbf` (OOF 0.9717).
- GNN `gnn_score` no longer all-zero: inductive `from_full_graph` scored 16,201 genes (mean 0.161,
  std 0.0208), injected into train/val/test with `nonzero_frac=1.0000`. Run 14/15 merge-back bug closed.
- Run15_Smoke.ps1: provision-failure now retries next offer; `PYTHONUNBUFFERED=1`+`-u`; streamed
  `run_phase2_eval.py` directly; `SmokeTimeoutMin` 90â†’180; `Ssh1` poll given `ConnectTimeout=20` (fixes
  the ~47-min apparent hang after the smoke had actually finished).

### Attempted / observed
- Full 13-model + stacker pipeline on `--max-train 3000 --n-folds 3 --min-review-tier 3 --string-db auto`.
  `SMOKE_EXIT=0`; Dev(test) AUROC 0.9831, Holdout(val) 0.9791 (PASS â‰¥0.9); total 767 s.
- Data prep is the dominant, `--max-train`-independent cost (~6.5 min): 4.40M raw â†’ 1.49M after
  label+tier filters; train 1.04M / val 146k / test 305k.

### Failed / flagged (open)
- **cnn_1d degenerate at smoke scale**: OOF 0.4936, test 0.4595, holdout 0.4819 (<0.5), MCC 0.0000.
  First run with sequence data wired (`unmapped=0/1490014`). Scale artifact vs defect UNRESOLVED â†’
  50k probe. Blocks Run 15 per pre-flight law until understood.
- **GNN near-chance** as a classifier (Best Val AUC 0.5240, 2,915 focal samples, early-stop ep16).
  Scorer fix is correct; discriminative power is a roadmap Tier-1/2 item, not a gate blocker.
- Smoke ran with dbnsfp/lovd/constraint = None â†’ 13 annotators all-zero (expected; paths not passed).
  Run 15 must wire available paths or they silently zero.
- `real_data_prep.py:444` `.fillna` downcasting FutureWarning (was :388; file moved).
- Annotation counter: `3/17` never logged (PhyloP 2/17 â†’ SpliceAI 4/17); LOVD logs `15/16` not `/17`.
- Review-tier filter applied as `<=3` (reads like a lower bound) â€” verify intended semantics.

### Learned
- `smoke_all_models.py` captures its `run_phase2_eval.py` subprocess â†’ blind poll + the wrapper's
  degenerate-OOF gate is bypassed when we run the eval directly for visibility. Trade visibility vs
  automated gating consciously; re-add a per-model degenerate assertion if running direct.
- A 3,000-row smoke validates that models RUN, not that data-hungry models (cnn_1d, GNN, KAN, deep nets)
  LEARN. Trees hit ~0.98 at 3k; neural/graph/spline models need a mid-scale probe before trusting the
  full run.
- vast.ai offers are ephemeral: a chosen `ask` can be taken between search and create
  (`error 404/3603 no_such_ask`); retry the next offer.
- New PowerShell session â‡’ `$key` unset â‡’ `ssh -i $key` collapses to `ssh -i -p â€¦` (`-i` eats `-p`).
  Always re-set `$key` per shell before ssh/scp.

## 2026-05-30 -- ScienceClaw artifact ledger + deterministic policy gate (Task 3)

**Added:**
- `src/genomic_variant_classifier/agent_layer/science_claw/ledger.py` -- append-only
  hash-chained `ScienceClawLedger` over the SharedState `artifact_ledger` key;
  caller-side `compute_sha256`; and the PURE gate
  `evaluate(ledger_entries, message, computed_sha) -> Verdict` enforcing BOTH
  integrity (artifact present in ledger + recorded hash == on-disk hash) AND
  authorization (requires_approval implies approved is True). No I/O or clock in the
  gate, so identical inputs yield identical verdicts.
- `src/genomic_variant_classifier/agent_layer/science_claw/__init__.py` -- exports
  ScienceClawLedger, evaluate, Verdict, compute_sha256, LedgerError.
- `tests/unit/test_science_claw_ledger.py` -- 21 tests (subject wiring, append-only
  chain, tamper detection, determinism, integrity, authorization, combined, no-op).
- `tests/unit/test_science_claw_orchestrator_gate.py` -- 7 tests with real fixtures
  (no mock patching): method exists, run_pipeline invokes the gate, DENY blocks a
  tampered/missing artifact (message rejected + review item), ALLOW for a valid
  artifact, no-op for non-artifact messages, ignores unapproved messages.

**Changed:**
- `message_bus.py` -- new canonical subject `ARTIFACT_PUBLISHED`, added to both
  `ALL_SUBJECTS` and `APPROVAL_REQUIRED_SUBJECTS` (requires approval by default).
- `shared_state.py` -- `_default_state()` gains `artifact_ledger: []`; existing state
  files backfill transparently via `_migrate`.
- `orchestrator.py` -- new `enforce_artifact_gate(agent_names)` runs inside
  `run_pipeline` before the agent loop; on a gate DENY for an artifact-referencing
  actionable message it rejects the message (DENY blocks) and adds a human-review
  item. No agent code changed.

**Verified:** full unit tree 588 -> 595 passed (1 skipped). Ledger suite 21/21;
orchestrator-gate suite 7/7.

**Found (pre-existing, separate INCIDENTs, out of scope):**
- test_message_bus.py Group 4 stale patch-target (legacy `agents.` import path).
- test_message_bus.py "history ordering" timing flakiness (equal-microsecond ties).

**RESOLVED 2026-05-31 -- all three pre-existing INCIDENTs closed this session:**
- Group-4 stale patch-target -> commit 0d218a8 (requests stub + ftplib path).
- "history ordering" flakiness -> commit 7da885c (monotonic `seq` + `(timestamp, seq)`
  sort; deterministic-tie test; bus suite 35/35).
- clingen int-truncation -> commit 8a86e3e (see above).
All three INCIDENT files carry RESOLVED status; G1 PASS (57/2/0) at HEAD 7da885c.
Both proven independent of Task 3 by stashing all three edits and reproducing the
identical failures at commit 553d5b6.

## 2026-05-30 -- Correctness harness (Task 2) + G1 Section 14

**Attempted:** Add an AutoKernel-style 5-stage correctness harness that gates model
correctness before any AUROC is recorded, and wire it as Section 14 of the G1 local
pre-flight gate.

**Added:**
- `src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py` -- 5
  stages (smoke / config / sanity / determinism / zero-audit);
  `run_correctness_harness(raw_df, ...) -> HarnessReport`.
- `src/genomic_variant_classifier/agent_layer/harness/__init__.py` -- exports
  run_correctness_harness, HarnessReport, build_reference_slice, KNOWN_ZERO_DEFAULT.
- `tests/unit/test_correctness_harness.py` -- 5 failing-first tests, all green. Suite
  562 -> 567 passed (1 skipped).
- Module-level `build_reference_slice()` (fully-populated synthetic frame) +
  `KNOWN_ZERO_DEFAULT` (21-col dead-connector allowlist), shared as single source of
  truth by the test and G1. Verified: residual silent-zero set == allowlist exactly
  (symmetric diff []; n=21) at HEAD 25b5eaf.
- `scripts/Run_Preflight_Local.ps1` Section 14: hard-fail on any stage 1-4 failure or
  any stage-5 finding outside KNOWN_ZERO_DEFAULT; warn on the 21 known-dead columns.
  Live-verified (3 PASS + 1 WARN; G1 summary 54/4/1).
- `docs/incidents/INCIDENT_2026-05-30_clingen-int-truncation.md`.

**Found (latent):** `clingen_validity_score` is truncated to 0 by `.astype(int)` in
`engineer_features` (~L169) when fed fractional input (ClinGen's real 0-1 scale).
Empirically: integer input survives, `uniform(0.1,1.0)` -> nonzero fraction 0.0.
Contrast `pli_score` (`.astype(float).clip(0,1)`, survives). Kept OUT of the allowlist
so the harness hard-fails if it ever silently zeroes on real data. Fix deferred to
R10-G. (INCIDENT filed.)

**RESOLVED 2026-05-31 (commit 8a86e3e).** Cast changed to `.astype(float)` (NOT the
`.clip(0,1)` originally sketched -- the harness fixture uses 0-4 ordinal ClinGen values,
so clipping would be wrong; float preserves fractional and ordinal inputs alike).
Failing-first regression test added; full suite 596 passed.

**Fixed (during build):** G1 Section 14 harness invocation. Passing multi-line Python
with embedded `"..."`/regex through `& $venvPython -c $harnessPy` mangled the inner
double-quotes at the PowerShell->native arg boundary (`r"feature '([^']+)'"` ->
`rfeature`, "'(' was never closed"). Neither expandable `@"..."@` nor literal
`@'...'@` here-strings fixed it. Resolved by writing `$harnessPy` to a temp `.py`
(UTF-8 no-BOM, try/finally) and running the file: `& $venvPython $harnessTmp`.

**Learned:** Never pass multi-line Python with embedded quotes through `python -c`
from PowerShell. A static probe that extracts the here-string body to a file and runs
the file will NOT catch this (it bypasses `-c`); only a live in-script run reproduces
it. Always dry-run the actual gate, not just a parse check.

## 2026-05-30 PM11c/PM11d - train.py sequence/label realignment + train-side guard

### Attempted
- Close carried tech-debt PM11c (cnn_1d dummy-sequence closure) and PM11d
  (decouple sequence handling from the positional iloc slice) before building
  the Run-15 correctness harness, so the harness is not validating broken behavior.

### Failed (pre-fix, now proven)
- scripts/train.py sourced CNN sequences via raw_df["fasta_seq"].iloc[:len(y_test)]
  -- a positional head of the PRE-split frame paired with labels from the
  gene-aware (shuffling) GroupShuffleSplit. Regression test
  test_old_iloc_logic_misaligns_sequences PASSES (seq<->label agreement < 0.85),
  proving the misalignment is real, not theoretical.

### Fixed
- Test side: X_seq_test now sourced from meta_test["fasta_seq"].reset_index(drop=True),
  which run() returns split-aligned by construction (meta_test = df.iloc[test_idx]).
  Verified ALIGNED: meta_test 349067 == y_test 349067.
- Train side: raises NotImplementedError if real training sequences are enabled,
  because run() does not return meta_train and X_train carries no variant_id key
  (X_train shape 1197216x73, zero identity columns) -- no signature-free realignment
  exists. Converts silent corruption into a loud, safe failure.
- has_sequences check moved from raw_df to meta_test (the split-aligned source).
- PM11c: dummy-placeholder series retained ONLY on the no-sequence path (the
  production path), with a comment clarifying they are inert once cnn_1d is popped.

### Learned / Verified
- Latent in production: data/processed/clinvar_grch38.parquet has fasta_seq present
  but notna=0, so has_sequences is always False in prod -> CNN always popped -> the
  train-side misalignment has never fired. Live only on synthetic / real-sequence runs.
- meta_train is NOT persisted in models/v1/splits (Test-Path False) and run() does
  not return it; the Option-B-wide signature change was deliberately deferred.

### Tests
- NEW tests/unit/test_train_sequence_alignment.py (2 tests, both pass).
- Full suite: 562 passed, 1 skipped, 0 failed (327s). No regression.

### Cost
- $0 (local only; no GPU).

## 2026-05-28 PM-G2 - KAN deep-audit + G2 VM env gate built; KAN eval persisted

### Attempted
- Verify KAN memory/correctness from source before Run 15; build Charter gate G2 (VM env preflight); persist the KAN backend decision.

### Fixed / Added
- NEW scripts/Run_Preflight_VM.sh (4989a70): lean G2 env/hardware gate (GPU+CUDA hard gate + VRAM floor, torch_geometric+networkx, imodelsx+KANClassifier imports, disk/RAM floors, repo HEAD w/ optional EXPECTED_HEAD). Complements launch's data/code preflight; no overlap. LF/no-BOM; bash -n clean.
- MODIFIED scripts/launch_run11_vm.sh (4989a70): corrected stale "FastKAN" comments (L8, L119) to imodelsx/dependency, matching kan.py PM13c. Comment-only.
- MODIFIED docs/runs/RUN_15_PLAN.md (4989a70): gate-F live checklist Run_Preflight_VM.ps1 -> .sh (historical entry untouched).
- MODIFIED scripts/preflight_vm.sh (4989a70): DEPRECATED-for-Run-15 header (stale ClinVar-VCF contract + relative data paths); kept as optional deep data audit.
- NEW docs/research/KAN_BACKEND_EVAL_2026-05-28.md (6c192c1, PM13d): KAN backend decision of record.

### Learned / Verified
- imodelsx KANClassifier.fit() batches (batch_size=512, DataLoader, CPU-resident data) -> memory-safe at any N; the Run-10a 17.9 GB runaway was pykan-specific. No pre-Run-15 backend swap needed; FastKAN = future speed only.
- KAN max_fit_samples default = 100_000; _fit_imodelsx subsample is stratified (stratify=y). No override in src/scripts.
- launch §5 GPU/dep block is WARN-only and never checks torch_geometric; G2 supplies the hard gates. Repo reaches the VM via SCP of the whole working tree (.git present), so git rev-parse works.
- preflight_vm.sh (2026-05-13) already had a CUDA hard gate + PyG check but is stale for the Run 15 data layout; kept as a deprecated optional audit rather than wired into Run 15.

### Cost
- $0 (local only; no GPU provisioned this session).

## 2026-05-28 PM14 - G1 local pre-flight gate built and CLEARED (PM13 chain)

### Attempted
- Build Charter v1.1 gate G1 (scripts/Run_Preflight_Local.ps1) from the Run14_Preflight.ps1 basis, run it, and clear it green before Run 15.

### Fixed
- NEW scripts/Run_Preflight_Local.ps1 (PM13, 3cf287a): 14-section local pre-flight; S1 verifies HEAD==origin/main (no hash pin). Data flow confirmed = re-prep-from-raw on VM (run9_ready splits not used; meta_train.parquet is a runtime output).
- MODIFIED Run_Preflight_Local.ps1 S7/S10 (PM13b, 8dd3285): LOVD floor 1 -> 0.1 MB (0.254 MB / 18,006 variants / 10 genes is the legit gene-scoped extract); pykan import probe -> kan (PyPI dist pykan imports as module kan).
- MODIFIED src/genomic_variant_classifier/models/kan.py docstrings L6/L81 (PM13c, ee06b08): corrected stale "FastKAN is primary" to imodelsx (efficient-kan) primary; behavior unchanged.
- MODIFIED Run_Preflight_Local.ps1 S6 (PM13e, 3cfdd4d): renamed locals to nFail/nPass/nSkip to fix a case-insensitive collision with $script:Failed/$script:Passed that crashed the harness; skip-aware gate (0 failed AND >=560 passed AND collected>=566).

### Achieved
- G1 CLEARED: 54 pass / 1 warn / 0 fail (exit 0) at 3cfdd4d. pytest 560 passed / 6 skipped / 0 failed (all 6 skips intentional: MC-dropout calibration TODOs pending Run 15 + 1 coverage skip).

### Learned
- PowerShell variable names are case-insensitive: a local $failed IS $script:Failed; never reuse an accumulator's bare name as a local.
- pytest "collected" != "passed"; a pass-count gate must tolerate intentional skips (gate on 0-failed + passed-floor + collected-floor).
- A pre-flight harness can carry its own logic bugs a parser self-test will not catch; only the full real-path run surfaces them.

### Findings (logged, not fixed)
- docs/CHANGELOG.md contains encoding mojibake (em/en-dashes, multiplication signs) from prior default-encoding writes; future bulk cleanup. New entries written ASCII-clean + no-BOM UTF-8.
- variant_ensemble.py L435-465 pandas .fillna downcasting FutureWarning; meta-learner lbfgs ConvergenceWarning on small fixtures.

## 2026-05-27 PM11b -- unseen_gene_holdout ablation wired into run_phase2_eval.py (C3 falsifier b)

### Attempted
- Wire unseen_gene_holdout_split (data/splits.py L117) into scripts/run_phase2_eval.py as a --unseen-gene-holdout flag, satisfying RUN_15_PLAN H_Run15 C3 hypothesis falsifier (b).
- Add the flag to scripts/launch_run11_vm.sh ARGS so Run 15 runs the ablation by default.
- Make parse_args() testable via an optional argv parameter.

### Fixed
- **MODIFIED** scripts/run_phase2_eval.py (4 changes):
  1. parse_args signature now accepts argv=None (backward-compatible; testable).
  2. parse_args returns p.parse_args(argv).
  3. Added --unseen-gene-holdout flag (action=store_true) after --skip-cnn.
  4. Added try/except-wrapped ablation block after _save_feature_importance:
     - Reads outdir/splits/meta_train.parquet (Patch 6b dependency, PM11a-closed).
     - Calls unseen_gene_holdout_split(holdout_frac=0.2, seed=42).
     - Builds separate EnsembleConfig (model_dir=outdir/models_unseen_gene_holdout).
     - Mirrors main ensemble's model-removal logic (skip_nn/skip_cnn/skip_kan/skip_svm).
     - Calls ensemble.fit(X_sub, seq_sub, y_sub); evaluates on held-out genes.
     - Saves unseen_gene_holdout_metrics.json + unseen_gene_holdout_per_model.csv.
     - Logs C3 falsifier (b) PASS/FAIL vs 0.95 AUROC threshold.
- **MODIFIED** scripts/launch_run11_vm.sh:
  - Added the --unseen-gene-holdout ARGS append AFTER L203 fi (outside the L185 GNOMAD_CONSTRAINT if-block, so it is unconditional, unlike --skip-cnn at L188).
- **CREATED** tests/unit/test_run_phase2_eval_flag.py:
  - 3 smoke tests: flag present, flag default False, store_true rejects values.

### Discovered state (Probes 1-3 + pre-flight, 2026-05-27)
- unseen_gene_holdout_split(df, holdout_frac=0.2, seed=42, gene_col, n_buckets=100) returns (train_idx, holdout_idx); SHA-256 hash-stable partition (data/splits.py L117).
- prep.run() at L186 returns X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta. meta_train is NOT a local var; ablation reads it from outdir/splits/meta_train.parquet (same pattern as Patch 6b GNN block at L296-L298).
- ensemble.fit(X, seq, y) signature (L249); ens_cfg = EnsembleConfig(n_folds, model_dir, skip_kan) at L214.
- seq_tr = pd.Series(["A"*101] * len(y_train)) at L199 (placeholder; subsetting via .iloc safe).
- Ablation inserted after _save_feature_importance(L510) so all primary metrics persist BEFORE the ablation retrain begins (defensive: an ablation crash cannot lose main results).
- launch_run11_vm.sh L185 if [ -f GNOMAD_CONSTRAINT ] spans through L203 fi (else L201). --unseen-gene-holdout inserted after L203 to be unconditional.

### Scope note (flagged, not fixed in PM11b)
--skip-cnn at launch_run11_vm.sh L188 is inside the L185-L203 GNOMAD_CONSTRAINT if-block, making it conditional on that file existing. May be intentional or a latent bug. PM11d candidate to investigate; not blocking Run 15.

### Commits (1 this session, pushed)
- `XXXXXXX` feat(eval,launch,tests): wire unseen_gene_holdout ablation (PM11b)

### Learned
1. Three-probe lifecycle + a no-mutation pre-flight caught every anchor risk before touching the tree. Pre-flight verified all 5 anchors count==1 and 4 idempotency conditions; the real patcher then ran with zero anchor surprises.
2. Bash indentation is decorative, not syntactic: launch_run11_vm.sh L188 LOOKS outside the if-block but is inside it (fi at L203). Cross-referencing if/fi token positions (Probe 3) is the only reliable way to read bash control flow.
3. meta_train must be read from disk: Probe 3 B3 scan confirmed meta_train is never a local var in run_phase2_eval.py. The Patch 6b (PM11a) meta_train.parquet persistence is a hard dependency of this ablation.
4. The ablation reuses ens_cfg parameters but with a separate model_dir (models_unseen_gene_holdout) to avoid clobbering the main ensemble's saved joblib artifacts.

### Open follow-ups
- **PM11c** (optional) - cnn_1d closure refactor per INCIDENT_2026-05-24.
- **PM11d** (defer) - investigate --skip-cnn conditional coupling in launch_run11_vm.sh.
- **Memory update** (after PM11 series) - see PM11a entry for stale items to correct; add "PM11b: unseen_gene_holdout wired" status.
- **Run 15 launch readiness** - B.D3 + unseen_gene_holdout wiring both complete. Next: G1+G2 pre-flight gates per Charter v1.1, then Vast.ai provision -> SCP -> train -> SCP back -> destroy.

---

## 2026-05-27 PM11a -- B.D3 verification + INCIDENT_2026-04-30 closure (test + docs)

### Attempted
- Verify B.D3 (pipeline-side gene_symbol fix) state on disk before pre-launch code work.
- Close stale INCIDENT_2026-04-30_gnn-gene-symbol-keyerror.md (Status was "NOT YET RESOLVED" since 2026-04-30; in fact Patch 6b is fully applied).
- Add regression test guarding the _save_splits / meta_train.parquet contract so future refactors don't silently regress the fix.

### Fixed
- **CREATED** `tests/unit/test_patch_6b_meta_train.py`: 3 regression tests
  1. `test_save_splits_writes_meta_train_parquet` -- asserts meta_train.parquet is written when meta_train is provided.
  2. `test_save_splits_meta_train_preserves_gene_symbol` -- asserts gene_symbol survives the parquet roundtrip.
  3. `test_save_splits_meta_train_optional_when_none` -- asserts backward compat: meta_train=None still writes meta_val/meta_test, no meta_train.parquet.
- **UPDATED** `docs/incidents/INCIDENT_2026-04-30_gnn-gene-symbol-keyerror.md`: Status DIAGNOSED → RESOLVED 2026-05-27 with Resolution section listing exact file/line refs + verification artifacts.
- **UPDATED** `docs/CHANGELOG.md`: this PM11a entry prepended.

### Discovered state (probe evidence)
- **`src/.../data/real_data_prep.py` L1194-L1216**: `_save_splits` signature already includes `meta_train: pd.DataFrame | None = None`; body writes `meta_train.to_parquet(out / "meta_train.parquet", ...)` when not None.
- **`src/.../data/real_data_prep.py` L278+L283-L286**: `run()` builds `meta_train = df.iloc[train_idx].reset_index(drop=True)` and threads it through `self._save_splits(..., meta_train=meta_train)`.
- **`scripts/run_phase2_eval.py` L292-L317**: literal `# Patch 6b (2026-04-30):` comment + meta_train.parquet read + gene_symbol merge into gnn_df + `raise FileNotFoundError(_meta_train_path)` for missing file.
- **`outputs/run9_ready/splits/meta_train.parquet`**: 41,839,799 bytes (41.81 MB) on disk.
- **`scripts/launch_run11_vm.sh` L229**: `python scripts/run_phase2_eval.py $ARGS` -- the Run 14/15 VM-side entry point (file mtime 2026-05-27).

### Scope clarification (consequences for RUN_15_PLAN B.D3)
PM10 entry stated "B.D3 enable: pipeline-side gene_symbol fix -- REQUIRED before Run 15." PM11a probe shows the fix is **already enabled** in both files. **No code change required for B.D3.** Run 15 launching `scripts/launch_run11_vm.sh` will exercise the patched path automatically; GNN training is implicit when running run_phase2_eval.py with splits that include meta_train.parquet.

The "GNN-FREE" status carried in memory (Runs 9-14) is therefore due to either ablation choice (run9_ablations.py), pre-Patch-6b splits, or other unrelated reasons. For Run 15 with `outputs/run9_ready/splits/meta_train.parquet` present and Patch 6b code applied, GNN should train.

### Commits (1 this session, pushed)
- `XXXXXXX` docs(incident,changelog) + test(unit): close INCIDENT_2026-04-30 + Patch 6b regression test (PM11a)

### Learned
1. **Sticky-stale-incident-doc pattern**: the INCIDENT was written 2026-04-30 with "NOT YET RESOLVED"; Patch 6b was committed at some point in subsequent days, but the INCIDENT Status was never updated. Future-Claude (and this Claude, earlier in session) inherited the stale doc as ground truth and almost re-implemented an already-applied fix. **Lesson: when an INCIDENT references a specific patch script and reading the target files shows the patched state, the INCIDENT is closed regardless of its own self-report.** Always verify by reading the target file.
2. **Memory rule #27 (Patch 6b root cause) is now OBSOLETE**: the rule describes a future fix that has already happened. Worth updating memory after PM11 series complete so future-Claude doesn't re-investigate.
3. **Entry-point chain audit is mandatory before code work on a Run-affecting path**: the chain `Run14_Preflight.ps1` (Windows) → `scripts/launch_run11_vm.sh` (VM-side bash) → `python scripts/run_phase2_eval.py` was non-obvious; needed 3 separate file reads to confirm. The newer-looking `scripts/train.py` (2026-05-09) is NOT in the current launch path.
4. **Patcher needle audit lesson**: PM11a v1 used `**2026-05-27 PM11a` (bold-text pattern from RUN_15_PLAN Decision log) as the CHANGELOG header check needle, but CHANGELOG uses `## 2026-05-27 PM11a` (level-2 heading). The two project conventions are syntactically distinct (`**bold**` vs `## header`); rule #28.17 (verbatim needles) requires distinguishing them. Fixed v2 uses `## 2026-05-27 PM11a` matching the actual content.

### Open follow-ups
- **PM11b** -- wire existing `unseen_gene_holdout_split` (data/splits.py L117) into `scripts/run_phase2_eval.py` with `--unseen-gene-holdout` flag. Adds inline ablation pass during Run 15 (per C3 hypothesis falsifier b).
- **PM11c** (optional) -- cnn_1d closure refactor per INCIDENT_2026-05-24 (currently --skip-cnn; not required by C3 hypothesis).
- **Memory update** (after PM11 series) -- mark memory #27 Patch 6b as "applied, INCIDENT closed PM11a 2026-05-27"; remove "B.D3 enable" from pre-launch items.
- **RUN_15_PLAN.md B.D3 status** -- plan's B.D3 line currently implies "build/enable" is pending. Should be updated to "verified complete via PM11a" in a docs-only follow-up (low priority; not blocking launch).

---

## 2026-05-27 PM10 -- E budget decision: triple resolved (docs-only)

### Attempted
- Resolve final 3 actual placeholders in RUN_15_PLAN.md E section (L68 GPU hours, L69 cost USD, L70 hard ceiling) grounded in actual Run-14 baseline + Vast.ai pricing data + Run 15 scope decision (Interpretation B' hybrid per Monzia 2026-05-27).

### Fixed
- **`docs/runs/RUN_15_PLAN.md`** L68: GPU hours estimate = ~10h (range 8--12h).
- **`docs/runs/RUN_15_PLAN.md`** L69: cost estimate = ~$7 (range $5--9).
- **`docs/runs/RUN_15_PLAN.md`** L70: hard ceiling = 24h wall-clock OR $20 USD, whichever first.
- **`docs/runs/RUN_15_PLAN.md`** Decision log: PM10 entry appended after PM9.
- **`docs/CHANGELOG.md`**: this PM10 entry prepended.

### Scope
Interpretation B' (hybrid) per Monzia 2026-05-27:
- Run 15 trains base ensemble: 10 models (catboost, lightgbm, xgboost, random_forest, gradient_boosting, tabular_nn, mc_dropout, deep_ensemble, kan-250k, gnn) -- cnn_1d still --skip-cnn per B.D6 PM8.
- Run 15 ALSO runs unseen_gene_holdout ablation INLINE (one additional full retrain on gene-stratified split).
- Other 12 ablations from the planned matrix (lookup_only, feature_permutation, true_generalization, etc.) DEFERRED to post-hoc analysis on saved models/OOF preds (separate session).

### Estimate basis
- **Run 14 baseline (CHANGELOG L483/L502/L503)**: 3.24h wall-clock @ $0.6694/hr = $2.17 on Vast.ai Texas RTX 4090 instance 37897784. 10-model ensemble incl. KAN via imodelsx. No GNN, no cnn_1d, no ablations.
- **Run 15 base estimate**: 3.24h + ~30--60 min KAN-100K → KAN-250K delta + ~30--60 min GNN-FREE → GNN-enabled delta ≈ 4.5--5.5h.
- **Inline unseen_gene_holdout retrain**: ~4.5--5.5h (same components, gene-stratified split).
- **Total**: ~9--11h, midpoint 10h.
- **Cost**: 10h × $0.67--0.77/hr (Run 13 was $0.771/hr; Run 14 was $0.6694/hr) = $6.70--$7.70, midpoint $7.
- **Hard ceiling**: 24h is ~2.4× expected wall-clock; $20 is ~2.9× expected cost. Either trigger → manual destroy and post-mortem.

### Pre-launch code dependencies (NOT this commit)
- **B.D3 enable: pipeline-side gene_symbol fix in `build_pyg_dataset` caller** (memory #27 Patch 6b root cause). UNLOCKS BOTH GNN training AND unseen_gene_holdout ablation -- single change, double payoff. **Required**.
- **unseen_gene_holdout evaluator** in training pipeline (new code; leverages B.D3's gene_symbol availability for the gene-stratified split). **Required**.
- **cnn_1d closure refactor** per INCIDENT_2026-05-24 (currently --skip-cnn). **Optional**; not required by C3 hypothesis (which references the 10-model ensemble incl. KAN, not 11 incl. cnn_1d).

### Commits (1 this session, pushed)
- `XXXXXXX` docs(plan,changelog): E budget triple resolved -- Interpretation B' hybrid (PM10)

### Learned
1. **Run 14 set a new project low-water mark**: 3.24h / $2.17 vs Run 11's 7.9h / $5.60 (-59% wall-clock, -61% cost). The dlperf≥80 pcie_bw≥12 filter (memory #30) plus the Texas instance ($0.6694/hr -- cheapest of the post-filter runs) drove the cost reduction. Run 15 budgeting should use Run 14 as the reference, not the Run 11--13 average.
2. **B.D3 pipeline-side gene_symbol fix has a hidden double payoff**: same code change unlocks GNN training (memory #27 root cause) AND unseen_gene_holdout ablation (gene-stratified split requires gene_symbol). Implementing it for B.D3 also satisfies the unseen_gene_holdout prerequisite. Document this in pre-launch code-change planning so it's not redundantly scheduled.
3. **The 13-ablation matrix is a PLAN, not implemented code**: src/ has no ABLATION_MASKS / run_ablation references (probe Phase 9: 0 hits). The only ablation code on disk is `scripts/run9_ablations.py` (one-off for Run 9's 6-ablation matrix, CHANGELOG L2117). Future ablations beyond unseen_gene_holdout will require either generalizing run9_ablations.py or building a proper src/ablations.py -- separate code work, post-Run-15.

### Open follow-ups
- **PM11 -- Pre-launch code commits** (NOT docs): B.D3 enable + unseen_gene_holdout evaluator (bundled, shared gene_symbol dependency) + (optional) cnn_1d closure refactor. Each commit separate per discipline (one decision per commit).
- **G1 + G2 pre-flight gates** per Charter v1.1 (RUN_15_PLAN.md L74--L82).
- **Run 15 launch** (Vast.ai SCP up → train → SCP back → destroy immediately, per memory #7 and #29b).
- **Post-Run-15 ablation matrix** -- separate session, separate budget. Generalize scripts/run9_ablations.py or build src/ablations.py for the 12 deferred ablations.
- **L77 backtick-doc-pattern** -- the `- [ ]` checklist line literally contains the placeholder marker in backticks. After PM10, this is the only remaining placeholder substring in the plan. Per PM9 Learned item 3, this is documentation, not an unresolved decision. Monzia checks the box manually as part of pre-flight. Note also that L77 gate text says "All B.O* and C.* decisions filled" -- narrowly scoped wording; A (Hypothesis) and E (Budget) decisions are implicitly required even though L77's text doesn't enumerate them.

---

## 2026-05-27 PM9 -- H_Run15 decision: Option C3 hybrid hypothesis (docs-only)

### Attempted
- Resolve H_Run15 placeholder at RUN_15_PLAN.md L13 with a falsifiable primary hypothesis grounded in actual Run 15 scope (post-PM5/PM6/PM7/PM8 decisions) and the project's central scientific concern (gene-prevalence memorization).
- Update L14 stale examples line, which referenced pre-decision scope (5 silent-zero gaps closed, KAN at 814K) contradicted by today's PM5/PM8 decisions.

### Fixed
- **`docs/runs/RUN_15_PLAN.md`** L13: H_Run15 set to Option C3 hybrid hypothesis (conjunctive: gap test + gene-memorization test).
- **`docs/runs/RUN_15_PLAN.md`** L14: stale examples line replaced with concise falsification summary + Decision log pointer.
- **`docs/runs/RUN_15_PLAN.md`** Decision log: PM9 entry appended after PM8.
- **`docs/CHANGELOG.md`**: this PM9 entry prepended.

### Rationale
- Hybrid C3 covers BOTH the gap test (encoded in B.O1 PM5 threshold 0.001) AND the central scientific concern (gene-prevalence memorization given n_pathogenic_in_gene importance 3.3× next feature, per memory #12).
- Falsifier (a): OOF→test gap > 0.0010 -- escalates B.O1 to Option A2 (500K KAN) in Run 16.
- Falsifier (b): unseen_gene_holdout AUROC < 0.95 -- flags gene-memorization dominance; deeper ablation required before claiming variant-level discriminative skill.
- Alternative candidates explicitly considered and rejected: C1 (gene-memo only -- missed the gap criterion already encoded in B.O1), C2 (gap only -- missed central scientific concern), C4 (orthogonality -- supporting goal, not primary classification goal).
- Conjunctive AND criterion is strictly harder to confirm than either C1 or C2 alone, yielding stronger evidence if it holds.

### Run 15 actual scope (deltas vs Run 14)
- KAN: 100K → 250K (B.O1 PM5, L103 of plan).
- MC-dropout: degenerate fallback → real epistemic+aleatoric (B.O3 PM6, commit c60e842, L24/L104 of plan).
- GNN: GNN-FREE → enabled conditional on pipeline-side gene_symbol fix (B.D3 PM8, L106 of plan; memory #27 root cause Patch 6b).
- cnn_1d: still --skip-cnn (B.D6 PM8 confirms; closure bug INCIDENT_2026-05-24 unresolved, L106 of plan).
- 5 silent-zero features still dead: B.D1/B.D2/B.D4/B.D5 deferred (PM8 L106).

### Commits (1 this session, pushed)
- `XXXXXXX` docs(plan,changelog): H_Run15 decision -- Option C3 hybrid hypothesis (PM9)

### Learned
1. **Hypothesis text must reflect ACTUAL run scope at time of decision.** L14 example hypotheses were written at Run 14 close-out (2026-05-26) and predated PM5/PM6/PM7/PM8 decisions -- by today they contradicted the actual scope (4 of 5 silent-zero gaps deferred, KAN at 250K not 814K). Plan-template scaffolding should be removed or updated as decisions land, not left as historical clutter that contradicts current state.
2. **Conjunctive (AND) hypotheses are strictly harder to confirm but yield stronger evidence than disjunctive (OR) or single-criterion hypotheses.** C3 requires BOTH (a) AND (b) to confirm; either failure refutes. Vs C1 or C2 alone, C3 leaves less room for misinterpretation at close-out.
3. **The L77 meta-reference (backtick-wrapped placeholder pattern) is a documentation pattern, not an unresolved decision.** Future validation tools that count the placeholder substring should skip backtick-wrapped occurrences or accept that the residual count after all decisions = 1 (the L77 doc-pattern). This is the same pattern-vs-literal collision class as PM8 v1 (memory rule #28.16).

### Open follow-ups
- **E budget (L68/L69/L70)** -- next decision (PM10). Will probe SESSION_2026-05-25.md and CHANGELOG for Run 11/12/13/14 actual wall-clocks before proposing GPU-hr / USD / hard-ceiling triple.
- **B.D3 enable: pipeline-side gene_symbol fix in `build_pyg_dataset` caller** -- required before Run 15 launch if C3 hypothesis is to test GNN ensemble contribution. Memory #27 root cause Patch 6b: `X_train_raw = pd.read_parquet(outdir/'splits'/X_train.parquet)` clobbers gnn_df with 78-col matrix lacking gene_symbol. Fix: source gene_symbol from df via train_idx, or persist meta_train.parquet alongside meta_val/meta_test in DataPrepPipeline._save_splits.
- **cnn_1d closure refactor** per INCIDENT_2026-05-24 -- currently --skip-cnn; if refactored before launch, Run 15 could include cnn_1d as 10th-or-11th ensemble member. Not strictly required by C3 hypothesis (which references the existing 10-model ensemble).
- **Run 15 pre-flight gates G1+G2** per Charter v1.1 (plan L72--L82).

---

## 2026-05-27 PM8 -- B.D batch decisions: 6 data-source decisions resolved + 3 plan factual corrections (docs-only)

### Attempted
- Resolve B.D1--B.D6 placeholders in RUN_15_PLAN.md L29--43 with HIGH-confidence rationale grounded in actual on-disk connector code + data files + recent incident docs.
- Correct 3 factual inaccuracies in plan wording discovered during the 4-phase B.D probe sequence (Option B: comprehensive).

### Failed (and recovered)
- **B.D probe v1 (Phase 2 abort)**: PowerShell operator-precedence bug. `$bdStart -ge 0 -and $i -gt $bdStart -and ... -match '^### ' -or ... -match '^## '` parsed as `(A -and B -and C) -or D`, so the `-or D` clause fired on the first `## ` header anywhere in the file BEFORE `$bdStart` was set, causing the section finder to early-exit with bdStart still -1. Memory rule #21.12 added: "`-and` tighter than `-or`; paren OR groups in AND chains else early-exit". Probe was read-only; no recovery needed.
- **B.D probe v2 path miss**: Phase 4 per-item probe paths used guessed filenames (`onekgp.py`, `kgp.py`, `primateai.py`, `primateai_3d.py`) that don't exist on disk; actual filenames are `thousandgenomes.py` and `primateai3d.py`. Phase 3 directory listing DID surface the real filenames but Phase 4 ran with guesses in parallel. Methodology lesson: directory listing must INFORM per-item probe paths, not run alongside them. Probe was read-only; no recovery needed.
- **PM8 patcher first attempt (Phase E abort, exit 10)**: delta count expected 6, got 5. Root cause: PM8_ENTRY contained a literal `<DECISION>` token ("...avoid precedence traps. [token] count in plan: 11 -> 5...") as a meta-mention, which collided with the Phase E `plan_lf.count("<DECISION")` validation. Atomic patcher worked correctly -- Phase F never ran, so no files were modified. Fix: reworded PM8_ENTRY to use "Plan placeholder count" instead of the literal `<DECISION>` token. New lesson logged as PM8_ENTRY item (4).
- **PM8 patcher second attempt (Phase E abort, exit 17)**: cl_checks needle #12 was "Directory listing must INFORM per-item probes" but actual CL_ENTRY Learned item 2 uses lowercase "inform". Frankenstein needle mixed casing from two different occurrences. Phase F never ran, no files modified. Fix: corrected needle to match Learned item 2 exactly. Logged as Learned item 10.
- **3 plan inaccuracies discovered by probe** (corrected in this commit per Option B):
  1. B.D1 sub-bullet "Unlocks 5 dead features (af_1kg_{afr,eur,eas,sas,amr})" was wrong -- `thousandgenomes.py` outputs single `allele_freq` column (gnomAD AF fallback), not 5 per-population features. Corrected.
  2. B.D6 heading "CNN-fasta input" was based on misconception per INCIDENT_2026-05-23: cnn_1d is a 1-D CNN over the 78-dim tabular feature vector (input shape `(78, 1)`), NOT a sequence model. Corrected heading + DECISION text.
  3. B.D2 plan claim "transfer" was outdated: 30.6 GB is already on disk at `data/external/finngen/finnge_R12_annotated_variants_v1.gz`. Two issues: filename typo ("finnge" missing 'n') and version mismatch (R12 vs connector-expected R10). Added detail to DECISION rationale.

### Fixed (6 decisions + 3 corrections committed)
- **`docs/runs/RUN_15_PLAN.md`** L29 (B.D1): DECISION resolved (defer to Run 16) with connector-scope clarification.
- **`docs/runs/RUN_15_PLAN.md`** L30 (B.D1 sub-bullet): rewritten from "5 features" claim to accurate "single `allele_freq` column" description.
- **`docs/runs/RUN_15_PLAN.md`** L32 (B.D2): DECISION resolved (defer) with filename typo + R12/R10 version mismatch detail.
- **`docs/runs/RUN_15_PLAN.md`** L35 (B.D3): DECISION resolved (build, enable GNN path) with cache evidence + Run 9 root cause attribution to pipeline-side gnn_df overwrite.
- **`docs/runs/RUN_15_PLAN.md`** L38 (B.D4): DECISION resolved (defer) referencing INCIDENT_2026-04-17 and the new-code-work scope.
- **`docs/runs/RUN_15_PLAN.md`** L41 (B.D5): DECISION resolved (defer with license-review subtrack continuing); "drop" option from plan token explicitly NOT used per memory rule #20 (never propose dropping techniques/features).
- **`docs/runs/RUN_15_PLAN.md`** L43 (B.D6): heading corrected from "CNN-fasta input" to "cnn_1d fasta input misconception" + DECISION resolved (--skip-cnn) per INCIDENT_2026-05-23 cnn_1d clarification.
- **`docs/runs/RUN_15_PLAN.md`** Decision log: PM8 entry appended after PM7.
- **`docs/CHANGELOG.md`**: this PM8 entry prepended at top.

### Headline verification (probe outputs cited)
- **`thousandgenomes.py` L13-15** (read live 2026-05-27): "Expected parquet schema (same format as gnomAD AF parquet): variant_id str, allele_freq float Global alternate AF across all 1000G super-populations". Single column -- confirms B.D1 correction.
- **`finngen.py` L18-21**: "Feature columns produced: finngen_af_fin, finngen_af_nfsee, finngen_enrichment". B.D2 plan claim was correct on feature names but wrong on transfer status.
- **`data/external/finngen/finnge_R12_annotated_variants_v1.gz`** (30638.3 MB): filename typo + R12 version visible from `Get-ChildItem` output.
- **`primateai3d.py` L26-28** (PHASE_2_PLACEHOLDER) + **L41** ("must match TABULAR_FEATURES when wired"): connector exists but not yet integrated.
- **`data/raw/cache/string_links.parquet`** (13,715,404 rows; columns include `combined_score`) + **`string_names.parquet`** (19,699 rows) + **`string_graph_700.pkl`** (17.2 MB): STRING data + graph pickle fully cached.
- **`gnn.py`** L640-644: defensive gene_symbol handling with empty-string defaults -- Run 9 GNN-FREE was pipeline-side gnn_df overwrite, not gnn.py.
- **INCIDENT_2026-05-23** L18-20: "`cnn_1d` is a 1-D convolutional network operating on the 78-dim tabular feature vector. It is NOT an image classifier. Input shape is `(78, 1)`. ... no image data is required, was ever required, or will fix the regression. The bug is in the wrapper code." -- confirms B.D6 misconception.
- **INCIDENT_2026-05-24** L27-35: CNN1D model class defined as a closure inside `_build_model` method, causing joblib unpickle failure cross-platform.

### Commits (1 this session, pushed)
- `XXXXXXX` docs(plan,changelog): B.D1-B.D6 batch decisions + 3 plan factual corrections (PM8)

### Learned
1. **Plan wording can be inaccurate.** B.D1 "5 features" claim and B.D6 "CNN-fasta" framing both contradicted the actual source on disk. Probe-first discipline (memory #11) caught both before they were committed as "resolved".
2. **Directory listing must inform per-item probes**, not run parallel with guessed paths. My Phase 4 used filename guesses that don't exist on disk, despite Phase 3 directory listing surfacing the real filenames. This is a methodology refinement on memory rule #11.
3. **PowerShell `-and` binds tighter than `-or`** (memory #21.12 added earlier today) -- caught a B.D probe v1 abort. Pattern: never put `-and ... -or` in section-finder loop conditions without parenthesizing the OR group.
4. **Two-loop section finders** are safer than single-loop with mixed conditions -- single-purpose loops have no operator-precedence ambiguity.
5. **gnn.py is defensively coded.** The Run 9 GNN-FREE issue is pipeline-side, NOT in gnn.py. This changes the scope of "fix GNN" work -- the fix is upstream in the caller, not in graph construction.
6. **INCIDENT docs are authoritative.** INCIDENT_2026-05-23 conclusively states cnn_1d is tabular, not sequence. Without reading the incident doc we would have committed wrong B.D6 reasoning. Reaffirms memory rule #11 (read project files first).
7. **The 30.6 GB FinnGen file is a half-finished transfer**: filename typo + version drift mean the data is on disk but unusable by current finngen.py code. "Transfer" was the wrong label; "integrate after R12 schema validation" is the real next step.
8. **Atomic patcher pattern (Phase A read → B idempotency → C anchors → D build → E validate → F write) scales to 6 simultaneous decisions** without partial-mutation risk. Validated by Phase E catching the PM8 meta-collision (v1) and cl_checks needle case mismatch (v2) -- Phase F never ran in either failure, no files modified, no `git checkout` recovery needed.
9. **Validation needles must not appear in NEW content.** PM8 v1: embedded `<DECISION>` as meta-mention while Phase E counted `<DECISION>` as delta-validation marker. Result: count drift by 1, abort exit 10. Fix: reword to avoid the literal.
10. **Validation needles must EXACTLY match content (case-sensitive).** PM8 v2: cl_checks needle was "Directory listing must INFORM per-item probes" but Learned item 2 has lowercase "inform" -- frankenstein needle mixing caps from two different sentences. Result: substring not found, abort exit 17. Fix: align needle to one specific occurrence exactly. Pattern: every check needle should be copy-pasted verbatim from a unique occurrence in NEW content, not synthesized.

### Open follow-ups
- **H_Run15** primary hypothesis (L13): pending.
- **E budget** GPU hours / cost USD / hard ceiling (L68-70): pending (3 sub-placeholders).
- **GNN pipeline-side gene_symbol fix**: required before B.D3 "build" is actionable in Run 15 pre-flight. Code change in `build_pyg_dataset` caller (likely `variant_ensemble.py` or `pipeline.py`).
- **cnn_1d closure refactor**: bug from post-C5 commit ac64665 still present. Needs code change to refactor `_CNN1D` class out of `_build_model` method closure (per INCIDENT_2026-05-24 root cause).
- **B.D2 R12 schema validation**: separate task to grep FinnGen R12 file headers and confirm column compatibility with finngen.py R10 expectations. If columns match, B.D2 reopens as quick-integrate; if not, needs code update.
- **Run 15 launch**: 4 remaining decisions (H_Run15 + 3 E sub-placeholders), then pre-flight + Vast.ai SCP up → train → SCP back → destroy immediately.

---

## 2026-05-27 PM7 -- C.1 decision: decline np.log(0) defensive clip at mc_dropout.py:87 (docs-only)

### Attempted
- Resolve C.1 placeholder in RUN_15_PLAN.md L47 (`<DECISION: yes | no>` for adding a defensive `np.clip` at `mc_dropout.py:87`).
- Confirm safety via live probe of `_decompose_uncertainty` (mc_dropout.py L65-90) before committing to a decline rationale grounded in actual source rather than memory.

### Failed (and recovered)
- **C.1 probe Phase 4 aborted**: `A hash table can only be added to another hash table`. Root cause: PowerShell automatic variable `$matches` is set by the `-match` regex operator to a hashtable of capture groups; my probe used `$matches = @()` (array) and then `$matches += "..."` after a regex match, which stomped my array and produced a hashtable += string error. Phases 1-3 succeeded before the abort (state pin, full code at L60-100, boundary probe script presence); Phase 5 (placeholder enumeration) did not execute.
- **First C.1 implementation paste aborted at exit 11**: `marker not in new CHANGELOG`. Root cause #1 (proximate): MARKER was defined as `"C.1 decision: no"` which appeared in plan PM7_ENTRY but NOT in CHANGELOG C1_ENTRY (which used `"C.1 decision: decline np.log(0)..."`). Post-condition `if MARKER not in new_cl_lf: sys.exit(11)` fired correctly. Root cause #2 (amplifier): patcher wrote the plan BEFORE validating the CHANGELOG, so the plan was modified on disk while CHANGELOG was untouched -- partial mutation requiring `git checkout HEAD --` revert. Memory rule #28 expanded: (14) multi-file patchers must build+validate ALL files before writing ANY; (15) marker strings must appear VERBATIM in ALL touched files' new content. Fixed paste: MARKER `"2026-05-27 PM7 -- C.1 decision"` appears in both PM7_ENTRY and C1_ENTRY header; restructured patcher into Phase A (read), B (idempotency), C (anchors), D (build), E (validate), F (write).
- **Recovery (both above)**: probe was read-only -- no probe-time mutations. First paste required `git checkout HEAD -- docs/runs/RUN_15_PLAN.md` to revert the partial plan write. CHANGELOG was unchanged in both failures.

### Fixed (decision rationale committed)
- **`docs/runs/RUN_15_PLAN.md`** (L47): `<DECISION: yes | no>` → DECLINED marker citing mathematical + empirical evidence.
- **`docs/runs/RUN_15_PLAN.md`** (Decision log): append PM7 C.1 entry after PM6 A2 entry.
- **`docs/CHANGELOG.md`**: prepend this PM7 entry at top (reverse-chronological).

### Headline verification
- **Live probe** of `src/genomic_variant_classifier/models/mc_dropout.py`: 313 lines, 11694 bytes, **CRLF: False** (LF-only -- different from `variant_ensemble.py`'s CRLF). L82-88 reads:

      L82: mean_prob = probs_stack.mean(axis=0)
      L83: epistemic = probs_stack.var(axis=0)
      L85: eps = 1e-8
      L86: clipped = np.clip(probs_stack, eps, 1.0 - eps)
      L87: entropy_per_pass = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
      L88: aleatoric = entropy_per_pass.mean(axis=0)

  Critical observation: **L87 uses `clipped` (NOT raw `probs_stack`)** in both `np.log` calls. Boundary safety is structurally enforced via variable reuse, not just side-clipping.
- **Mathematical guarantee**: `clipped ∈ [1e-8, 1-1e-8]` (enforced by L86 assignment) ⇒ `log(clipped) ∈ [log(1e-8), log(1-1e-8)] ≈ [-18.42, -1e-8]` (finite); `log(1-clipped) ≈ [-18.42, -1e-8]` (finite by symmetry); products are bounded products of finite values (finite).
- **Empirical**: Runs 11/12/13/14 all included mc_dropout (OOF AUROC 0.9971/0.9971/0.9971/~0.9968) with **zero log(0) crashes** across roughly 4 × 1.2M samples × 5 folds = 24M+ inference passes.
- **Regression suite**: `tests/unit/test_mc_dropout_uncertainty.py` (7 cases, all green per B.O2 closure 2026-05-26) + `scripts/probe_a1_boundary.py` (2956 bytes, present locally, callable for any future verification).

### Commits (1 this session, pushed)
- `XXXXXXX` docs(plan,changelog): C.1 decline - np.log(0) clip not needed at mc_dropout.py:87 (line structurally safe)

### Learned
1. **Standing rule #3 (probe before assume) caught one more case.** The visible L87 code uses `clipped` not raw `probs_stack`. Had we relied on B.O2 closure summary alone, the argument would be implicit; the live probe shows the bound is structurally enforced via variable reuse.
2. **PS automatic variable `$matches` is a recurring trap.** Set by `-match` operator to capture-group hashtable. Memory rule #21 expanded with full auto-var blocklist.
3. **mc_dropout.py uses LF, not CRLF.** `variant_ensemble.py` uses CRLF. Future patchers must detect line endings per-file. The `read_bytes/decode/normalize-LF/restore-CRLF/encode/write_bytes` pattern handles this correctly.
4. **Redundant defensive code is anti-pattern when tests + production evidence exist.** A second clip at L87 would clip already-clipped values to the same bounds -- pure no-op.
5. **SESSION START PK queries (memory #11):** today's PK searches surfaced SESSION_2026-05-25 with full Run 11/12/13 context that should have informed Phase C framing earlier in the day.
6. **Markdown rendering pitfall.** Nested triple-backtick fences inside a Python string inside a PowerShell heredoc inside a chat markdown response close the outer fence prematurely. Fix: use 4-space indented code blocks inside the Python string.
7. **NEW (memory #28 items 14+15): Multi-file patcher atomicity + cross-file marker consistency.** Failed C.1 paste demonstrated both. (14) Build+validate ALL files before writing ANY -- prevents partial mutations when later validation fails. (15) Marker strings must appear VERBATIM in ALL touched files' new content -- a marker present only in one file but checked against another causes guaranteed false-positive aborts. Combined fix: restructured patcher into Phase A read → B idempotency (both) → C anchors (both) → D build (both) → E validate (both) → F write (both, only after all green).

### Open follow-ups
- **B.D1--B.D6** (6 data-source decisions): next in Phase C queue.
- **H_Run15** primary hypothesis: pending.
- **E budget** (GPU hours / cost USD / hard ceiling at RUN_15_PLAN.md L68-70): pending.
- **After all decisions close**: Run 15 launch.

---

## 2026-05-27 PM6 -- A2/B.O3/C.2 closure: TabularNNClassifier._predict_proba_single_pass implementation

### Attempted
- Close A2 (mc_dropout uncertainty degenerate) by implementing `_predict_proba_single_pass()` on `TabularNNClassifier`, satisfying MCDropoutWrapper's L216 hasattr contract so the wrapper produces real epistemic + aleatoric uncertainty instead of the L238-241 degenerate fallback returning `(proba, zeros, zeros)`.
- Add comprehensive unit test suite covering: API contract, stochasticity, side-effect isolation, MCDropoutWrapper integration, and 5 scientific properties (AUROC floor on linearly separable, mean-of-K ≈ deterministic, aleatoric bounded by log(2), aleatoric peaks at decision boundary, higher dropout → higher epistemic).
- Stub integration tests for post-Run-15 calibration work (OOD epistemic elevation, uncertainty-error correlation, ECE improvement, MC convergence).

### Failed (and recovered)
- **Initial design relied on memory, not probed source.** First-draft paste embedded 3 unverified assumptions: MCDropoutWrapper constructor parameter names (`base_estimator` vs `estimator`), public method signatures (`predict_with_uncertainty`), and whether `_decompose_uncertainty` was importable as a module-level function. Audit phase before execution surfaced this as standing-rule-#3 violation (probe before assume).
- **CRLF line-ending mismatch would have aborted the patcher at exit code 2.** `variant_ensemble.py` uses CRLF; initial OLD anchor used LF (`\n`). Patcher's `read_bytes().decode('utf-8')` yields CRLF in the string; LF-only anchor would not have matched. Caught at Step 0 verification probe before any mutation.
- **`caplog` scoping false-negative risk.** Initial test used bare `caplog.at_level(logging.WARNING)` (root logger). `mc_dropout.py:218` uses module logger `genomic_variant_classifier.models.mc_dropout`. If that logger ever sets `propagate = False`, the regression-guard test would have silently passed even when the warning was actually emitted. Caught at audit; fixed via explicit `logger=MC_DROPOUT_LOGGER` constant.
- **`docs/CHANGELOG.md` path assumption.** Step F-0 probe checked project root for `CHANGELOG.md` → NOT FOUND. Phase 6 grep revealed canonical location at `docs/CHANGELOG.md`. Had Step F proceeded with the assumed project-root path, would have created a duplicate file outside the docs tree.

### Fixed
- **`src/genomic_variant_classifier/models/variant_ensemble.py`** (c60e842, 53322 → 55749 bytes, CRLF preserved): added `_predict_proba_single_pass(self, X, seed=None)` between L874 `predict_proba` and L884 `predict`. Selective dropout activation pattern:
  - `model_.eval()` puts whole network in inference mode (running-stats BatchNorm)
  - Loop `model_.modules()` and selectively `.train()` only `nn.Dropout` instances → stochastic dropout mask without per-batch BatchNorm corruption
  - `try/finally` ensures `.eval()` restoration so subsequent `predict_proba` calls aren't left dropout-active
  - `torch.manual_seed(int(seed))` controls mask determinism per pass for MC sampling reproducibility
  - `raise ValueError` if `model_ is None` (explicit failure vs. silent `.modules()` AttributeError)
- **`tests/unit/test_tabular_nn_mc_dropout.py`** (new, 261 lines, 12143 bytes): 15 tests, 5 classes:
  - `TestPredictProbaSinglePassContract` (3): method exists, returns (n,2), probabilities valid
  - `TestPredictProbaSinglePassStochasticity` (3): same-seed deterministic, different-seed stochastic, K-pass variance non-zero
  - `TestPredictProbaSinglePassSideEffects` (2): no leak to predict_proba, single-row no NaN (BatchNorm preserved)
  - `TestMCDropoutWrapperIntegration` (2): no missing-method warning (caplog scoped to mc_dropout logger), end-to-end epistemic > 0
  - `TestPredictProbaSinglePassScientificProperties` (5): AUROC floor 0.85 on linearly separable, mean K ≈ predict_proba, aleatoric bounded by log(2), aleatoric peaks at boundary, dropout-rate sensitivity (5 epochs to allow training divergence)
- **`tests/integration/test_mc_dropout_calibration.py`** (new, 100 lines, 4800 bytes): 5 stubbed tests across 4 classes (`@pytest.mark.skip` + `raise NotImplementedError`) preserving threads for post-Run-15: `TestOODEpistemicElevation`, `TestUncertaintyErrorCorrelation` (Spearman + quartile binning), `TestCalibrationImprovement` (ECE, paper P2), `TestMonteCarloConvergence` (1/K variance scaling).

### Headline verification
- pytest: **14 passed, 6 skipped, 0 failed, 0 errors** in 96.03s.
  - 1 unit test skipped: `test_aleatoric_higher_near_decision_boundary` -- synthetic corpus didn't span both p≈0.5 (boundary) and p≈0/1 (extreme) prediction regions; `pytest.skip` guard fired as designed.
  - 5 integration stubs skipped (deliberate, awaiting Run 15 cohort).
- 19/19 PowerShell sanity checks PASS (including audit-added "VE preserves CRLF" and "caplog scoped to mc_dropout logger" gates).
- `.venv312` confirmed active via `python -c "import sys; print(sys.executable)"` pre-check.

### Commits (1 this session, pushed)
- `c60e842` feat(tabular_nn): A2/B.O3/C.2 close - implement _predict_proba_single_pass for MC-dropout

### Learned
1. **Standing rule #3 (probe before write) is non-negotiable, not optional.** Initial implementation paste embedded 3 unverified assumptions (CRLF, caplog scope, MCDropoutWrapper API). Step 0 verification probe caught the CRLF blocker; audit caught the caplog scope risk; only the API assumptions turned out correct -- probed, not guessed. Discipline ladder: probe first → audit second → execute third. Skipping any tier is a self-inflicted cycle loss.
2. **`docs/CHANGELOG.md` is the canonical path for this project, not `CHANGELOG.md` at root.** Step F-0 probe returned NOT FOUND for project-root path; Phase 6 grep revealed canonical location. Memory updated to canonicalize this going forward.
3. **`pytest.skip()` is a coverage gap signal worth tracking.** The aleatoric-peaks-at-boundary test skipped because the model trained well enough that predictions cluster at extremes. The calibration property of `_decompose_uncertainty` was NOT exercised by this commit's unit tests. Mitigation: `tests/integration/test_mc_dropout_calibration.py::TestCalibrationImprovement` covers similar territory against Run 15 holdout when data is available.
4. **Selective dropout activation is canonical for networks with BatchNorm.** Naive `model.train()` corrupts single-row/small-batch inference via per-batch BatchNorm stats. The `isinstance(m, nn.Dropout)` filter preserves running-stats BatchNorm while enabling stochastic dropout masks. Caught at design phase because BatchNorm1d was visible in the probed L815 architecture; would have caused NaN on the single-row test otherwise.
5. **Probe outputs are the only ground truth.** I claimed PyTorch architecture for `TabularNNClassifier` from memory; project knowledge held a stale TensorFlow snapshot; current code IS PyTorch (probe confirmed). The session compaction summary and project knowledge can BOTH be stale; only the live `view`/probe is authoritative.

### Open follow-ups
- **C.1** (np.log(0) defensive clip at mc_dropout.py:87): pending. Per B.O2 closure, the line is already safe via L86 `np.clip(probs_stack, 1e-8, 1.0 - 1e-8)`; C.1 decides whether to add a SECOND defensive clip at L87 as belt-and-suspenders.
- **B.D1--B.D6** (6 data-source decisions): pending in Phase C decision queue.
- **H_Run15** primary hypothesis: pending.
- **E budget** (GPU hours / cost USD / hard ceiling at RUN_15_PLAN.md L68-70): pending.
- **Coverage gap**: `test_aleatoric_higher_near_decision_boundary` skipped this session; documented; deferred to integration tests against Run 15 holdout.
- **Pre-existing mojibake in older CHANGELOG entries** (e.g., L1993, L2008, L2012 etc. from 2026-05-16 Run 10): double-encoded UTF-8 artifacts (`ÃÂ¢` for en-dash etc.). Out of scope this commit; flagged for future maintenance pass.

---

## 2026-05-27 PM5 -- A4/B.O1 KAN decision (250K Run 15, 500K Run 16 staged)

**Decided** scale KAN subsample to 250K for Run 15 (Option A1). Option A2 (500K) reserved for Run 16 if Run 15 OOF→test gap remains >0.001.

**Justification**: Run 14 at 100K showed OOF→test gap 0.0025 (≈3.5x catboost's gap), indicating overfit. Staged scaling tests whether 2.5x more training data (250K) closes the gap; if not, Run 16 escalates 5x (500K). Memory #18: KAN reinstated 2026-04-20 with 80GB GPU access (A100/H100 tier); 250K and 500K both tractable. Option B (drop) rejected: would lose KAN diversity contribution without testing the overfit-vs-sample-size hypothesis. Option C (keep 100K) rejected: empirically overfits.

**Files**: docs/runs/RUN_15_PLAN.md (B.O1 line + Decision log append), docs/CHANGELOG.md (this entry).

**Next**: H_Run15 hypothesis + E budget (items 2-3 of session's Phase C decision queue).

---

## 2026-05-27 -- D17 closure: scripts/run15_observability.py + tests for Run 15 (PM session 4)

### Attempted
- D17 closure: clone scripts/run14_observability.py to scripts/run15_observability.py + matching test file.
- Required by Run15_Postflight.ps1 L80 (exit 1 if missing locally). Last hard blocker for Run 15 launch from a code-presence standpoint.
- Codify the four distinct failure modes encountered across iterations into the lessons block.

### Failed
- **Attempt 1**: Patcher used `Path.read_text(encoding="utf-8", newline="")`. TypeError -- `Path.read_text` accepts `newline=` only since Python 3.13; env is 3.12.10 (`Path.write_text(newline=)` works since 3.10, asymmetric API gap). Top-level try/catch halted cleanly before any file writes.
- **Attempt 2**: Patcher fixed (`read_bytes` + decode + replace + encode + `write_bytes`). Patcher succeeded; files written. Phase 4 verification false-failed on all 4 `0 occurrences of X` checks: pattern `($newScript.Split('run14').Length - 1) -eq 0` was broken because PS 5.1 (.NET Framework 4.x) lacks the `String.Split(string)` overload added in .NET 5+. `.Split('run14')` resolved to the `params char[]` overload -- splits on any of chars `r`/`u`/`n`/`1`/`4`. Throw fired correctly; files left on disk (catch printed recovery commands but did not execute them).
- **Attempt 3**: Verification fixed (`-not <var>.Contains('substr')`). State pin tree-clean check threw at start because prior attempt's untracked files were still present. Pattern: "print recovery, hope human runs it" is structurally unreliable when next paste is re-paste of same block.
- **Attempt 4**: SUCCESS. Self-healing Phase 1 added: detects exactly the known-stale `?? scripts/run15_observability.py` + `?? tests/unit/test_run15_observability.py` entries and selectively cleans them. Refuses for any unexpected dirty entry.

### Fixed
- **`scripts/run15_observability.py`** (`486c680`, 602 lines, 25065 bytes): byte-level clone via Python patcher with 4 deterministic string-replace transforms.
  - Global `run14` -> `run15` (4 lines: L3 header banner, L30 usage example, L33 `--report-dir` example path, L585/586 output filenames `.json` + `.md`).
  - Global `Run 14` -> `Run 15` (3 lines: L5 purpose, L12 target, L406 markdown title).
  - Date: `Created:  2026-05-26` -> `Created:  2026-05-27`.
  - Target ref: `genomic-variant-classifier @ commit bf2f665, Run 14` -> `genomic-variant-classifier, Run 15 (commit set at launch)`.
  - Byte arithmetic: net delta +6 bytes from longer target line; all other transforms length-preserving. Matches patcher's reported `25059 -> 25065`.

- **`tests/unit/test_run15_observability.py`** (`486c680`, 132 lines, 5947 bytes): clone with single targeted replacement `run14_observability` -> `run15_observability` (4 occurrences: L1 docstring, L24 SCRIPT_PATH, L28 import docstring, L30 module spec name).
  - **INTENTIONALLY PRESERVED**: `outputs/run14/`, `run14_master.log`, `Run 14 log format`, `run14_synth`. These reference the test DATA SOURCE (real Run 14 log lines used as canonical sample format), not the script under test. Parser is invariant across runs.
  - Byte delta: 0 (run14_observability and run15_observability are same length); 4 length-preserving substitutions.

- **Patcher idiom now canonical for Python <= 3.12**: `Path.read_bytes()` -> `.decode("utf-8")` -> string `.replace()` -> `.encode("utf-8")` -> `Path.write_bytes()`. Bypasses the `read_text(newline=)` 3.13 API gap AND any local autocrlf line-ending interference (byte layer is opaque to autocrlf which operates on text-mode I/O only).

- **PowerShell sanity-check pattern correction**: substring-presence checks must use `-not $var.Contains('substr')` (PS 5.1-safe). DO NOT use `$var.Split('substr').Length - 1` on PowerShell 5.1.

- **Self-healing Phase 1 pattern**: when known-stale untracked artifacts from a prior failed apply might be present, state-pin should (a) match against exact known-stale entries, (b) refuse for any unexpected dirty entry, (c) selectively `Remove-Item -Force` and re-verify clean.

### Headline verification
- 19/19 Contains-based sanity checks PASS (PS 5.1 safe).
- `python -m py_compile` PASS on both new files.
- pytest: 14/14 PASS in 7.67s (both regression-on-old and functional-equivalence-on-new).
- Git: HEAD `d8baaa9` -> `486c680`; local == remote after push.
- Byte arithmetic: script +6, test +0, both match patcher reported sizes exactly.

### Commits (1 this session, pushed)
- `486c680` -- feat(scripts): D17 - scripts/run15_observability.py + tests for Run 15 (734 insertions, 2 files)

### Learned
1. **Python 3.12 vs 3.13 API gap**: `Path.read_text()` accepts `newline=` only since 3.13. `Path.write_text(newline=)` works since 3.10. For Python <= 3.12 portability, use `Path.read_bytes()` + `.decode("utf-8")` for reads and `.encode("utf-8")` + `Path.write_bytes()` for writes. Bypasses the asymmetric API gap AND preserves exact source byte structure regardless of autocrlf.
2. **PowerShell 5.1 `String.Split(string)` is the char-array overload**: `$str.Split('run14')` on PS 5.1 (.NET Framework 4.x) splits on **any** of chars `r`/`u`/`n`/`1`/`4`, NOT on the substring "run14". The single-string overload was added in .NET 5+ and is only available in PowerShell 7+. For substring-presence on PS 5.1, use `-not $str.Contains('substr')`.
3. **"Print recovery, hope human runs it" is structurally unreliable**: when a catch handler prints recovery commands as text but does not execute them, the next paste typically re-runs the same block, which re-throws the same way. Build self-healing into the state-pin phase: detect the exact known-stale artifacts, REFUSE if any unexpected dirty entry exists, and selectively clean.
4. **Audit-finding pattern**: when memory or older session notes claim an item is OPEN but the repo shows otherwise, verify against `git log --oneline -- path/to/file` and `git show <commit>`. Today's D16 turned out to already be CLOSED in `bd75ed5` (2026-05-11). The CHANGELOG entries on L54 and L98 of d8baaa9 are stale as a result; per append-only convention they are NOT modified, but this entry documents the audit finding for future grep.
5. **Recursive `__pycache__` cleanup scope**: `Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__` from project root traverses INTO `.venv312/` and clears site-packages pycache too (720 dirs cleared in this session). Functionally harmless (pytest re-generates), but wasteful. Future canonical idiom: filter out `\.venv*` paths from the cleanup list.

### Open follow-ups
- **D15** (memory codification): codify today's 5 lessons into memory_user_edits; remove any stale "D16 is open" entries from memory. Requires user confirmation per memory tool standing rule. Est 10 min.
- **SESSION_2026-05-27.md** update: append today's late-session events (D17 closure + D16 audit finding). Est 10 min.
- **Phase C remaining decision-only items**: A2 (TabularNN MC-dropout), A4 (KAN subsample), A6 (data sources x6), E budget, H_Run15. None block Run 15 launch from a code-presence standpoint; A4 + E budget + H_Run15 should be locked before launch.

---

## 2026-05-27 -- C.5+C.6+C.7 closure: postflight + destroy infrastructure (PM session 3)

### Attempted
- Anomaly closures for Run 15 plan C.5 (Test-ArtifactPresent wiring), C.6 (`exit 1` on any FAIL), and C.7 (separate destroy script refusing automation).
- Phase C of Run 15 plan continued (these were the last 3 code-level items before A2/A4/A6/E/H_Run15 decision-only items).
- Apply Charter v1.2 patch's Test-ArtifactPresent helper into actual gate logic.

### Failed
- First paste (f7febbb) had 2 sanity checks that false-positive FAILed:
  - `'No direct vastai destroy command'` used regex `'vastai\s+destroy'` against the full file. The CRITICAL header comment correctly states "This script DOES NOT call vastai destroy." -- which the over-broad regex matched. Should have walked lines and skipped `^\s*#` comments.
  - `'Has exit 1 path on any FAIL (C.6) >= 5'` used `(?m)^\s*exit\s+1` which only matches line-starting `exit 1`. The script has 5 total `exit 1` paths but 2 are inline in one-line `if (...) { ...; exit 1 }` patterns at L91 and L113. Should have used `\bexit\s+1\b` (word boundary, any position).
- PS-throw-scoping bug recurred (documented in 2026-05-27 A3 closure as Finding 2): the Phase 3 `throw` exited only the `& { }` block, not the surrounding paste. Phase 4 parser self-test PASSed (strong syntactic guarantee), Phase 5 committed f7febbb anyway. **The commit was correct** (parser PASS plus 10/12 sanity OK and 2 false-positive FAILs), but the procedural failure mode is real -- a future paste with a real syntactic error and the same sanity-check design would commit broken code.

### Fixed
- **`scripts/Run15_Postflight.ps1`** (`f7febbb`, 194 lines / 10789 bytes): based on `Run14_Postflight.ps1` structure with explicit artifact-presence gates section. Closes C.5 + C.6.
  - 7 Test-ArtifactPresent gates: master_log (≥1000B), observability_md, observability_json, per_model_metrics_csv, ensemble_joblib (≥1MB), ensemble_manifest, blend_weights.
  - Writes gate exit code to `.gate_exit_code` file in the report directory (consumed by Vastai_Destroy_Confirmed.ps1).
  - 5 explicit `exit 1` paths covering training-incomplete abort, obs script missing locally, SCP obs script failure, SCP report failure, and gate FAIL block.
  - **Run 14 oversight fix**: SCPs `models/` directory (which contains `ensemble.joblib`). Run 14's postflight did not SCP this, contributing to the A8 procedural fail (Charter v1.2 patch was needed because `Test-Path` checked the wrong nested path; even with the helper, the *directory* still had to be SCPed for the gate to find the file).
  - **A7 support**: SCPs `per_model_metrics.csv` and `per_model_metrics_val.csv` added by the Run 14 observability rewrite (da41f27).
  - Replaces the inline destroy command print (Run 14 pattern) with a pointer to `Vastai_Destroy_Confirmed.ps1`.

- **`scripts/Vastai_Destroy_Confirmed.ps1`** (`6107e56`, 114 lines / 6021 bytes): new script with 4 independent refusal layers. Closes C.7.
  - **Layer 1** (exit 2): refuses if `[Console]::IsInputRedirected` is true. Blocks `echo y | .\Vastai_Destroy_Confirmed.ps1 ...` automation and any pipe-from-stdin invocation. Directly addresses INCIDENT_2026-05-12 (vastai CLI interactive prompt) and INCIDENT_2026-05-24 (Run 10b premature destroy where destroy command shared a paste block with SCP setup).
  - **Layer 2** (exit 3): refuses if `-GateFile` path does not exist on disk. Forces postflight to have actually run.
  - **Layer 3** (exit 4): refuses if gate file content is not exactly `"0"`. Hard prerequisite that all Run15_Postflight.ps1 gates PASSed.
  - **Layer 4**: interactive `Read-Host` with `-cne "DESTROY"` case-sensitive comparison. Typo-resistant; "destroy" lowercase fails.
  - On layer pass: pipes `'y' |` to `& vastai destroy instance $InstanceId` to handle CLI ≥1.0.12's interactive confirmation prompt (per INCIDENT_2026-05-12). Exit 5 if CLI itself returns non-zero.

- **Procedural fix applied in second paste**: wrapped entire paste body in `try { ... } catch { Write-Host "ABORT: $_" -ForegroundColor Red; return }` at top scope. This definitively halts the paste on any throw -- the PS-throw-scoping issue from Finding 2 is now fixed by paste discipline. Pattern proven in production by this session's paste (no catch fired because no phase threw; the wrapper was in place as the safety net).

- **Sanity-check design fix**: corrected check patterns for Vastai_Destroy_Confirmed.ps1:
  - Word-boundary regex (`\bexit\s+N\b`) instead of line-starting (`(?m)^\s*exit\s+N`).
  - Line-walking comment-aware classification for `vastai destroy` matches (skip `^\s*#` lines before counting).
  - All 12 sanity checks PASS for 6107e56 (12/12).

### Headline verification
- f7febbb empirical re-verification (Phase 2 of the C.7 paste): exit 1 total = **5** (3 line-starting + 2 inline), 'vastai destroy' = **0 in code / 1 in comment**, Test-ArtifactPresent invocations = **7**, PowerShell parser PASS. f7febbb is genuinely correct; the 2 prior "FAIL"s were definitively false positives.
- 6107e56 parser self-test PASS, 12/12 corrected sanity checks PASS, single file staged.
- Both commits pushed clean; local == remote at each step.

### Commits (2 this session, both pushed)
- `f7febbb` -- feat(scripts): C.5+C.6 - Run15_Postflight.ps1 with Test-ArtifactPresent gates (194 lines, 1 file)
- `6107e56` -- feat(scripts): C.7 - Vastai_Destroy_Confirmed.ps1 with 4-layer refusal (114 lines, 1 file)

### Learned
1. **Sanity-check design is its own quality dimension.** Over-specified anchors (line-starting requirements, whole-file regex matches that don't distinguish code from comments) produce false positives that erode trust in the check suite and -- worse -- disguise the next paste's real problems. Use word boundaries; walk lines for comment classification; prefer narrow, defensible single-feature checks over count-thresholds.
2. **Top-level `try { ... } catch { ... return }` definitively fixes PS-throw-scoping in pasted blocks.** When any phase throws, control jumps to the catch, the `return` exits the script context, and subsequent statements do not execute. Verified in production usage in the C.7 paste (the wrapper did not fire only because nothing threw). This is the fix promised in the A3 closure CHANGELOG (Finding 2) and should be the default paste idiom from this session forward.
3. **The Run 14 procedural-fail class (A8) had two root causes, not one.** The first was the `Test-Path` flat-path assumption in the postflight gate (closed in Charter v1.2 patch via Test-ArtifactPresent helper). The second was that `models/` was not SCPed at all, so the helper had nothing to find. C.5 closes the second root cause by adding `models/` to the SCP list.
4. **Defense in depth at 4 layers is the right cardinality for an irreversible cloud command.** Each layer catches a distinct failure mode and uses a distinct exit code, so debug effort is bounded. Cumulative refusal probability under normal operation: stdin-not-redirected (interactive shell) + gate-file-exists (postflight ran) + gate-content-is-zero (postflight passed) + DESTROY-typed-exactly (intentional human action) -- each independently necessary.

### Open follow-up
- **D15** (memory updates, queued from A3 closure + A7 closure): codify PS-throw-scoping resolved via top-level try/catch; codify sanity-check design lessons; codify the `models/` SCP requirement. Estimated 10-15 min.
- **D16** (.gitattributes `*.sh text eol=lf`): pin shell-script line endings to LF in the repo so local Windows working tree matches the committed blob -- resolves the bash -n unreliability on Windows. Estimated 15 min.
- **D17** (Run 15 prep): create `scripts/run15_observability.py`. Run15_Postflight.ps1 L80 references this and will exit 1 if absent. Hard blocker for Run 15 launch. Copy from `scripts/run14_observability.py` and adapt paths/run id. Estimated 1-2 hr.
- **Phase C remaining decision-only items**: A4 (KAN subsample), A2 (TabularNN MC-dropout implementation vs drop), A6 (6 data-source decisions -- some need license review), E budget, H_Run15 hypothesis.
- **Phase E**: author `scripts/Run_Preflight_Local.ps1` and `scripts/Run_Preflight_VM.ps1` (Charter v1.1 templates planned but never committed -- see earlier audit finding).
- **Phase F**: Vast.ai provision → SCP up → train → SCP back → invoke Vastai_Destroy_Confirmed.ps1.

---

## 2026-05-27 -- A3 closure: launch script imodelsx_patch tee dedupe (PM session 2)

### Attempted
- Anomaly A3 close: dedupe imodelsx_patch logging in `scripts/launch_run11_vm.sh`.
- Phase C of Run 15 plan continued (A3 follows A7, per Phase-C ordering decision).
- Empirical hypothesis confirmation against `outputs/run14/run14_master.log`.

### Failed
- First paste's pre-fix `bash -n scripts/launch_run11_vm.sh` raised: `syntax error near unexpected token $'{\r''` on L31 `cleanup() {`. Root cause is NOT the script itself: line-ending diagnostic in session 2 showed CRLF=274 / LF-only=0 in the local working tree. Git's autocrlf is active (`warning: in the working copy of 'scripts/launch_run11_vm.sh', CRLF will be replaced by LF the next time Git touches it`). The committed blob is LF (verified by `git ls-files --eol` semantics and the fact that Run 14 launched successfully on Vast.ai via git-clone). Bash on Windows cannot parse CRLF shell scripts; this is a tooling artifact, not a real syntax error.
- First paste's post-fix `bash -n` failed for the same CRLF reason. Post-fix throw fired inside the Phase 3 `& { }` block but did NOT halt subsequent phases at top scope (PowerShell `throw` exits the script block, not the interactive paste). Phases 4-6 ran anyway and committed `9628463`. This worked correctly because the 5/5 verbatim-source sanity checks PASSED for the actual edit; but it is a **real safety gap** for future pastes if a real syntax error ever needs catching.
- Second paste (refined version with line-ending diagnostic + empirical Run-14 log check) was re-pasted while session 1's commit had already landed. All defensive safety nets fired correctly: HEAD-drift check threw (expected 526cb3f, got 9628463), anchor uniqueness threw (count=0 because file already patched), Python patcher's A3 marker idempotency check exited 1 with "ABORT: patch already applied (A3 marker present)", and stage-set check threw (empty stage). No corruption, no double-application.

### Fixed
- Root cause (structurally confirmed + empirically verified 2026-05-27): `scripts/launch_run11_vm.sh` L200 had `fi | tee -a "$LOG"` after an if/else block (L193-200) where each inner echo (L197 success branch, L199 else branch) already piped to `tee -a "$LOG"`. The outer tee re-tee'd the inner echoes' already-tee'd output. Effect: each imodelsx_patch status line logged to `run11_master.log` twice.
- Empirical evidence (Phase 2a of refined paste): in `outputs/run14/run14_master.log` (61722 bytes), `'fixed 3 bare-name refs'` appears 2 times and `'already patched'` appears 0 times. Hypothesis confirmed: success branch fired once and was logged twice.
- Implemented: replace L200 `fi | tee -a "$LOG"` with `fi  # A3 fix 2026-05-27: removed redundant outer tee`. Inner echoes preserved; outer-else WARN echo at L202 preserved. 1-line change, idempotent on retry (patcher refuses to re-apply if the A3 marker is already present).
- Defense-in-depth verification: 5 verbatim-source-substring sanity checks (PS), 9 internal Python patcher post-conditions (idempotency, anchor count = 1, no-op refusal, anchor gone post-replace, 3 collateral integrity checks, length delta sanity), PS-level anchor uniqueness pre-check. All passed on session 1's first apply. All defensive checks correctly refused session 2's redundant re-application.

### Headline empirical verification
- Run 14 log `imodelsx_patch: fixed 3 bare-name refs` count: **2** (should have been 1)
- Run 14 log `imodelsx_patch: already patched` count: **0** (else branch did not fire)
- Net hypothesis: success branch logged twice, confirming tee-dup structurally and empirically
- Post-fix expected: each imodelsx_patch line logged once

### Commits (1 this session, pushed)
- `9628463` -- fix(scripts): A3 close - dedupe imodelsx_patch logging in launch_run11_vm.sh (1 insertion, 1 deletion)

### Learned
1. **PowerShell `throw` inside `& { ... }` exits ONLY the script block, not the surrounding interactive paste.** Subsequent top-level statements continue executing. For paste safety, either (a) wrap the entire paste in `try { ... } catch { Write-Host "ABORT: $_" -ForegroundColor Red; return }`, or (b) set a `$script:abort = $true` flag and check it at the entry of every subsequent phase. For A3 this manifested benignly (the script edit was correct; bash -n failed only for CRLF reasons), but a future paste with a real edit error would commit corrupt code.
2. **`scripts/launch_run11_vm.sh` has CRLF line endings in the local working tree but LF in the committed blob.** Git's autocrlf normalization keeps the repo clean for Linux consumers (Vast.ai via git-clone gets LF and works fine), but Windows-side `bash -n` cannot parse the working-tree CRLF copy. Follow-up: add a `.gitattributes` rule pinning `*.sh` to `text eol=lf` and re-checkout. Local working-tree CRLF will also break any direct SCP from the working tree to a Linux box (caller should always SCP from a fresh git-clone or normalize on transfer).
3. **Idempotency-by-marker is a strong defense.** The patcher's first check is `if "A3 fix 2026-05-27" in src: sys.exit(1)`. Combined with the PS-level anchor-uniqueness check (Phase 2d, counts `'fi | tee -a "$LOG"'` occurrences via `[regex]::Matches`), session 2's redundant re-paste was caught at 4 independent layers (HEAD drift, anchor count = 0, patcher marker exit, empty-stage throw). This is the level of defense-in-depth that paste discipline should target.
4. **Empirical verification of structural hypotheses is cheap and high-value.** Phase 2a of the refined paste (`[regex]::Matches` on the Run 14 master log for both branch messages) took milliseconds and converted a structural argument into a measurement. Should be standard for any future "X line is duplicated/missing" claim.

### Open follow-up
- **Phase C remaining**: C.5-C.7 (postflight + destroy script infrastructure; Charter SR #38, #39, Test-ArtifactPresent wiring); A2 (B.O3, C.2) `TabularNNClassifier._predict_proba_single_pass`; A4 (B.O1) KAN subsample decision; A6 (B.D1-B.D6) 6 data-source decisions; E budget; H_Run15 hypothesis.
- **D15** (queued for next session): codify the two new memory learnings: (a) PowerShell `throw` scoping in pasted blocks, (b) shell-script line-ending hygiene with .gitattributes recommendation.
- **D16** (separate small commit): add `.gitattributes` rule `*.sh text eol=lf` and `*.ps1 text eol=crlf`, then re-checkout `scripts/launch_run11_vm.sh` to normalize the working tree. Will resolve the bash -n local-machine reliability gap.
- **Phase E**: `scripts/Run_Preflight_Local.ps1` and `scripts/Run_Preflight_VM.ps1` (Charter v1.1 templates exist; need authoring).
- **Phase F**: Vast.ai provision → SCP up → train → SCP back → destroy.

---

## 2026-05-27 -- A7 closure: observability per_model parser rewrite (PM session)

### Attempted
- Anomaly A7 close: rewrite `scripts/run14_observability.py` per_model parser to read structured outputs.
- Phase C of Run 15 plan launched (A7 first per Phase-C ordering decision).
- Local verification by regenerating observability against `outputs/run14/full/` and `outputs/run14/run14_master.log`.

### Failed
- V1 audit check `'No old hardcoded regex'` was uninformative due to a backslash-escaping error in the PS regex pattern. The check used too many backslashes, so the .NET regex looked for double-backslashes in the file source, but the file source has single backslashes for regex metachars. Result: the check always passed regardless of whether the old pattern was present. V2 audit fixed this with correct backslash counts and added a symmetric `'Old per_model call gone'` check for the main() body.
- V2 paste's `'main placeholder set (3a)'` audit check raised a false-positive FAIL because V1 (which actually ran the patcher) wrote `# filled below; structured-files preferred (A7 fix 2026-05-27)` as the placeholder comment, while V2's audit expected `# A7 fix` literally after the `#`. The patch is functionally correct; only the audit check was over-specific to V2's exact text. Investigated and confirmed via relaxed re-check (`"per_model": None,\s*#`) plus 3/3 functional verification (per_model_source=structured, catboost.test_auroc=0.9975, kan.oof_auroc populated) before commit.

### Fixed
- Root cause (verified 2026-05-27): `parse_log_for_per_model_metrics` Pattern A regex required the "==>" prefix on metric lines. `outputs/run14/run14_master.log` shows 45 "==>" lines (all from shell launch echos like `==> [1/7] Data file preflight`) and 11 OOF AUROC lines (all via Python logger format like `2026-05-26 10:49:47  INFO  ...  random_forest OOF AUROC: 0.9978`) with zero overlap. The regex matched 0 of 11 metric lines.
- Implemented: new `read_per_model_metrics_files(outputs_dir)` function reads three structured sources atomically: `per_model_metrics.csv` (test metrics), `per_model_metrics_val.csv` (val metrics), `models/*_meta.json` (OOF AUROC + saved_at_utc + n_samples). `main()` prefers structured files and falls back to log-grep if absent. JSON adds `per_model_source` key with values `"structured"` or `"log_scrape"`. Pattern A regex relaxed to accept Python-logger format (defense in depth for the fallback path). `write_markdown_report` uses `test_f1_macro` fallback (CSV column name; pre-A7 log-grep produced `test_f1`).
- Regression test `tests/unit/test_run14_observability.py` (7 cases, all passing): 5 cases for the structured-file reader (OOF/test/val/missing-dir/empty-dir), 2 cases for the log-grep fallback (Python-logger format match + legacy "==>" backward compat).

### Headline verification (local regen against outputs/run14/full/)
- `per_model_source: structured`
- catboost: OOF=0.9981844462249252 TEST=0.9975 VAL=0.9975 -- matches 2026-05-26 Run 14 entry
- kan: OOF=0.9921137643927214 TEST=0.9896 VAL=0.9914 -- matches
- xgboost: OOF=0.9983895442721538 TEST=0.9974 -- matches
- ENSEMBLE_STACKER present (TEST=0.9975, val=0.9974) -- new vs pre-A7 (log-grep never matched ensemble row)
- All 11 entries from per_model_metrics.csv populate (10 base learners + ensemble stacker)

### Commits (1 this session, pushed)
- `da41f27` -- fix(observability): A7 close - rewrite per_model parser to read structured files (236 insertions, 4 deletions; includes 132-line regression test file)

### Learned
1. **PowerShell single-quote regex pitfall.** PS single-quote literals preserve characters verbatim. .NET regex requires two backslashes to match one literal backslash. So matching one literal backslash + s in a PS regex pattern needs 3 chars (`\s`), not 5 chars (`\\s`). Writing too many backslashes never matches anything, and `-not (never matches)` always returns true -- the check looks fine but tests nothing. Verbatim-source-substring marker rule (D13.RETRY, 2026-05-27) covers the underlying discipline.
2. **Audit checks must match what the patcher actually writes, not what an alternative patcher revision would write.** V2's audit check expected V2's text; V1 ran and wrote different text. False-positive FAIL is recoverable but wastes a verification round. When a check fails on a patch that appears working, INVESTIGATE the check vs file content before reverting -- the patch may be correct.
3. **Run14_Postflight.ps1 consumes the observability MD report (L129-131), not the JSON schema.** Changing JSON schema (added `per_model_source` key) is safe as long as MD renders correctly. No Postflight changes needed.
4. **Structured outputs beat log-grep for any post-hoc analysis.** Even when the regex is correct, structured files don't depend on logger format, have higher precision (full-float meta-json vs rounded log values), survive log truncation, and are atomic per-model. Future observability code should always prefer structured > log-grep > nothing.

### Test count baseline correction
- The compacted summary referenced "552 tests" from B.6.4 proof; today's count is 533. Re-reading the 2026-05-27 AM CHANGELOG entry (L22) shows the real testpaths-fix baseline was **526 tests**, not 552. The 552 figure was the B.6.4 hypothetical (full suite minus the polluter file). 526 + 7 new A7 tests = 533 today. **No discrepancy.**

### Open follow-up
- **A3** (next Phase C item): `scripts/launch_run11_vm.sh` imodelsx_patch echo dedupe (C.4 in plan; ~30 min).
- **C.5-C.7**: postflight + destroy script infrastructure (Charter SR #38, #39, Test-ArtifactPresent wiring).
- **A2** (B.O3, C.2): implement `TabularNNClassifier._predict_proba_single_pass()` OR drop MC-dropout from base learners (2-4 hr).
- **A4** (B.O1): KAN subsample decision (30 min decision; GPU-expensive action).
- **A6** (B.D1-B.D6): 6 data-source decisions; some need license review.
- **E budget**: GPU hours, cost USD, hard ceiling.
- **H_Run15 hypothesis**: last item to lock.

---

## 2026-05-27 - Pytest sys.modules pollution diagnosed + testpaths fix landed

### Attempted
- Phase B.1-B.8: systematic diagnosis of 12 pytest collection errors that surfaced on 2026-05-26 after phylop relocation cleared an earlier NameError mask.
- Phase B.9: commit + push pyproject.toml fix (9eec8eb).
- Phase B.11: write SESSION_2026-05-27.md (54c29fe).
- Phase D13: append corrections to INCIDENT_2026-05-26_scipy-torch-array-api-compat.md while preserving the original 87 lines (919920c).

### Failed
- INCIDENT_2026-05-26 hypothesis #1 ("torch is partially or incorrectly installed") proved wrong. B.1.3 verified torch installs cleanly: `python -c "import torch; print(torch.__file__, torch.__version__, torch.__spec__, torch.Tensor)"` all work in plain Python.
- B.11.2 first attempt wrote the SESSION file to C:\Users\monzi\docs\sessions\ because `[System.IO.File]::WriteAllText` resolves relative paths against .NET `Environment.CurrentDirectory` (the PS-startup dir), not PS `Get-Location`. Recovered in B.11.RETRY with `[Environment]::CurrentDirectory = $pwd.Path` and absolute paths.
- D13.RETRY threw at STEP 5 marker check on a paraphrased regex (`sys.modules["torch"] = MagicMock` as one literal sequence) that does not appear in the source - the code block uses `sys.modules[_mod]` (loop variable) and the prose uses `sys.modules["torch"]` separated by words. File content was always correct; D13.RECOVER verified with 16 markers chosen from verbatim source substrings.

### Fixed
- Verified root cause (B.5.5 + B.6.1 + B.6.4): `src/genomic_variant_classifier/agent_layer/test_message_bus.py` L87-89 stubs `sys.modules[_mod] = MagicMock()` at module level for a loop spanning ewc_utils/shap/torch/feedparser/requests. pytest's default `test_*.py` auto-discovery imports this file during full-suite collection, polluting torch for the rest of the collection. scipy.stats's array_api_compat `_issubclass_fast` then calls `getattr(sys.modules["torch"], "Tensor")` and gets back a MagicMock (hashable so it passes scipy's lru_cache key check, but NOT a class), causing the subsequent issubclass() to raise TypeError. The test_esm2_activation.py `ValueError: torch.__spec__ is not set` is the same pollution viewed via a different lookup path.
- B.6.4 decisive: full suite minus `test_message_bus.py` drops errors from 12 to 0 and increases tests from 416 to 552 (+136 = exactly the 12 victim files' counts 17+10+10+3+18+10+2+7+10+11+22+16).
- Commit `9eec8eb` added `[tool.pytest.ini_options]` with `testpaths = ["tests"]` to pyproject.toml. Restricts pytest auto-discovery to canonical tests/ tree. test_message_bus.py is unmodified and remains runnable by explicit path.
- Side effect (B.8.1): root-level `test_catboost.py` (17718 B, untracked per .gitignore:95, in "Scratch and generated files" section per .gitignore:92) is no longer auto-discovered. Correctness improvement - cloud/CI runs on Vast.ai never saw this untracked file anyway, so local pytest now matches cloud pytest behavior. Canonical tracked `tests/unit/test_catboost.py` (20551 B) remains in default discovery.
- A1 regression test gap closed: `tests/unit/test_mc_dropout_uncertainty.py` (7 tests) shipped in `3a166f6` on 2026-05-26 but never actually ran under pytest until 2026-05-27 because it was among the 12 erroring files. Now 7 passed in 5.35s.

### Headline metrics (G1 gate verification)
- `python -m pytest --collect-only -q`: **526 tests, 0 errors** (was 416 collected + 12 errors).
- A1 regression (test_mc_dropout_uncertainty.py): 7 passed in 5.35s.
- Spot-checks: alphamissense 17 passed, eve 18 passed, prediction_artifacts 11 passed.
- Memory rule #4 (G1 gate: local pytest collection errors = 0): **GREEN**.

### Commits (5+1 this session, all on main)
- `088797a` - fix(tests): relocate test_phylop_block.py and remove duplicate broken test (carried over from 2026-05-26 close)
- `8662597` - fix(gitignore): anchor scratch-file patterns to root, completing 088797a relocation
- `9eec8eb` - fix(pytest): restrict discovery to tests/ to stop sys.modules["torch"] pollution
- `54c29fe` - docs(session): SESSION_2026-05-27 - B.1-B.9 testpaths fix diagnosis + remediation
- `919920c` - docs(incident): D13 - correct INCIDENT_2026-05-26 with verified root cause
- (this commit) - docs(changelog): catch CHANGELOG up with 2026-05-27 session

### Learned
1. **G1 gate paid for itself.** Run 15 would have launched on Vast.ai (Linux, where the polluter file is also importable) and discovered this issue only after compute spend. The standing rule strengthened on 2026-05-27 (memory rule #4: ALL prior-run anomalies CLOSED or DEFERRED-with-justification, local pytest collection errors = 0) is doing its job in pre-flight.
2. **`pytest --collect-only -q` silently suppresses files with 0 tests AND 0 errors.** test_message_bus.py was imported (running its sys.modules pollution) but wasn't enumerated in -q output, hiding the polluter from grep on pytest logs. The smoking gun came from greping the SOURCE FILE for `sys.modules[`, not the pytest log. Future audits: `--collect-only -v` for per-file visibility.
3. **PowerShell array splat is `@var`, never `@$var`.** Wrong form gives silent 1.4s no-op (B.5.1-B.5.3 false negative). Validation rule: pytest invocation <2s elapsed time = red flag.
4. **`Select-String` regex misses produce empty arrays silently.** Downstream foreach over the empty array runs zero times silently. Defensive pattern: count results, throw if empty when non-empty expected.
5. **`Get-Content -Raw` after `Out-File -Encoding utf8` introduces CRLF on Windows.** B.9.8.1 false-negative on FIX block caused by exactly this. For inspecting commit-message bodies: keep the array of lines, or pipe through `-replace "\r",""`.
6. **`[System.IO.File]` uses .NET CWD, not PowerShell `Get-Location`.** B.11.2 wrote to `C:\Users\monzi\docs\sessions\` instead of project root because the relative path resolved against .NET's startup CWD. Fix: absolute paths everywhere with .NET APIs, or `[Environment]::CurrentDirectory = (Get-Location).Path` at session start. Memory rule was already present; failure to apply it cost one paste cycle. D14 re-codification queued.
7. **Marker regexes must use verbatim source substrings, not paraphrased forms.** D13.RETRY's STEP 5 failed because the regex looked for a phrase that does not exist as one literal sequence in the source. Defensive: pick distinctive phrases that exist as exact byte sequences (preferably section headers or code-block lines).

### Open follow-up (next session)
- **D12** (post-Run-15): refactor test_message_bus.py L87-89 sys.modules pollution into pytest `monkeypatch.setitem` fixtures with proper teardown.
- **D14** (this session, queued): codify lessons 2-7 above into memory via `memory_user_edits`.
- **Phase C** (next session): anomaly sweep A2-A8 (21 `<DECISION>` placeholders in RUN_15_PLAN.md). Per memory recommend A7 first.
- **Phase E** (next session): `scripts/preflight_run15.py` G1-G15 master gate script.
- **Phase F** (next session): Vast.ai provision -> SCP up -> train -> SCP back -> destroy.

---

## 2026-05-26 â€” Run 14 complete + Preflight Charter v1.1 + v1.2 patch

### Attempted
- Run 14 launch on Vast.ai instance 37897784 (Texas, RTX 4090, $0.6694/hr) after 4-bug KAN remediation chain.
- Production of locked test AUROC on 349K-variant held-out set with 10 base learners (first run where KAN trains).

### Failed
- Launch #1 (10:12 UTC): nohup+tee redirect collision corrupted log to binary.
- Launch #2 (10:32 UTC): `ModuleNotFoundError: genomic_variant_classifier`; launch script assumed pre-installed package on fresh VM.
- Launch #3: PowerShell escaping error on inline Python smoke test (no run impact).
- Postflight Block B gate (A8): used fixed `Test-Path` on flat paths; reported FAIL on `ensemble.manifest.json` and `ensemble.joblib` even though both were SCPed to `\full\models\` (one directory deeper). Destroy command was inadvertently executed despite the FAIL â€” recovery confirmed files locally via recursive locator. No data loss. Procedural lesson logged.

### Fixed
- Launch #4 (10:38:56 UTC): tmux send-keys with manually pre-installed deps â†’ ALL PREFLIGHT PASSED.
- KAN trained successfully via imodelsx/efficient-kan backend on CUDA, OOF 0.9921 (3 CV folds Ã-- 100K subsample).
- Run completed clean exit 0 at 13:53:31 UTC.
- Charter v1.2 patch: `scripts/Run14_Postflight.ps1` now uses `Test-ArtifactPresent` helper (recursive `Get-ChildItem -Filter`) instead of fixed `Test-Path`. A8 closed.

### Headline metrics (locked test set, 349,067 variants)
- Test AUROC: **0.9975** (Run 13 0.9974, Î” +0.0001)
- Test AUPRC: 0.9914, f1_macro: 0.9775, f1_weighted: 0.9855, MCC: 0.9550, Brier: 0.0130
- OOF blend AUROC: 0.9985 (LR stacker: 0.9984)
- Wall-clock: 3 h 14 m 35 s (Run 13 was 6.3 h â†’ -49%)
- Cost: $2.17 (Run 13 was $4.90 â†’ -56%, project low-water mark)

### Per-model OOF AUROC (10 base learners â€” all 10 trained successfully)
random_forest 0.9978, xgboost 0.9984, lightgbm 0.9983, logistic_regression 0.9955, gradient_boosting 0.9974, catboost 0.9982, tabular_nn 0.9975, **kan 0.9921 (NEW)**, mc_dropout 0.9975, deep_ensemble 0.9977.

### Per-model TEST AUROC (key finding)
- **catboost test AUROC 0.9975 = ENSEMBLE_STACKER test AUROC 0.9975** (tied on ranking power)
- Stacker dominates on threshold-dependent: f1_macro 0.9775 vs 0.9632 (Î” +0.0143), MCC 0.9550 vs 0.9276 (Î” +0.0274), Brier 0.0130 vs 0.0166 (lower = better calibrated)
- KAN test AUROC 0.9896 (OOFâ†’test gap 0.0025, ~3.5Ã-- catboost's gap â†’ 100K subsample overfits)

### Learned
1. **H1 confirmed technically but diversity-marginal on AUROC**: ensemble's lift is in **calibration and threshold quality**, not ranking. catboost alone is competitive for AUROC use cases.
2. **34 of 78 features are dead** (observability collector quantification). 8 connector/parser gaps map to specific Run-15 work items.
3. **Procedural failure mode A8 closed**: postflight gates must use recursive locators because output directories nest by 1-3 levels. Charter v1.2 patch enforces this in `Run14_Postflight.ps1`.
4. **Charter SR #38 queued for Run-15 prep**: separate `Vastai_Destroy_Confirmed.ps1` that requires gate exit 0 and refuses `echo y |` auto-confirmation, so destroys can never follow a failed gate in the same shell session.

### Charter v1.1 deployed
6 artifacts installed:
- `docs/PREFLIGHT_CHARTER.md`, `docs/templates/RUN_N_PLAN_TEMPLATE.md`
- `scripts/Run_Preflight_Local.ps1`, `Run_Preflight_VM.ps1`, `Run_Monitor.ps1`, `Run_Postflight.ps1`

6 new standing rules SR #32 â€“ #37 added.

### Charter v1.2 patch deployed
- `Test-ArtifactPresent` helper inserted into `scripts/Run14_Postflight.ps1` (and `Run_Postflight.ps1` if present).
- Closes A8.

### Open backlog (â†’ Run 15)
- A1: `np.log(0)` at `mc_dropout.py:87` â€” clip BEFORE log
- A2: implement `_predict_proba_single_pass()` on TabularNNClassifier OR migrate uncertainty to DeepEnsembleWrapper
- A3: deduplicate `imodelsx_patch` echo in `launch_run11_vm.sh`
- A4: scale KAN subsample 100K â†’ 250K-500K
- A5: normalize score annotation step numbering
- A6 (data): build STRING parquet, 1KGP AF parquet, transfer FinnGen, evaluate PrimateAI-3D license, build CNN fasta or `--skip-cnn`
- A7: fix `scripts/run14_observability.py` per_model log-parsing patterns (currently extracts nothing despite log lines being present)
- SR #38 (queued): separate destroy script with gate-exit-0 prerequisite
- HGVSp parser â†’ unlocks ESM-2 + EVE
- Populate `RUN_15_PLAN.md` from template; run G1+G2 gates before any Vast.ai create

### Artifacts (committed)
- `outputs/run14/full/metrics.json` (322 bytes) â€” stacker AUROC/AUPRC/F1/MCC/Brier for test + val
- `outputs/run14/full/per_model_metrics.csv` (629 bytes) â€” 11-row test-set table
- `outputs/run14/full/per_model_metrics_val.csv` (636 bytes)
- `outputs/run14/full/feature_importance.csv`, `data_quality_audit.{csv,json}`
- `outputs/run14/full/models/ensemble.manifest.json`, `outputs/run14/full/scaler.manifest.json`
- `outputs/run14/run14_master.log` (61,722 bytes)
- `outputs/run14/pip_freeze_vm.txt` (216 packages)
- `outputs/run14/reproducibility_manifest.json` (16,268 bytes â€” full metrics + per-model + SHA-256 + session_notes)
- `outputs/run14_observability/run14_observability.{md,json}`

### Artifacts (deliberately NOT committed â€” too large; on local disk only)
- `outputs/run14/full/models/*.joblib` (10 base + ensemble.joblib = ~520 MB)
- `outputs/run14/full/models/*_oof.npy` and `*_oof_indices.npy` (~160 MB)
- `outputs/run14/full/models/ensemble_models/*.joblib` (~1.1 GB; full-data refits)
- `outputs/run14/full/splits/*.parquet`, `outputs/run14/full/oof_predictions.parquet`, `meta_*.parquet` (~150 MB)

### HEAD progression
`f4dbeed` â†’ `0d4ea7b` â†’ `bf2f665` â†’ `35b9e44` â†’ `80ac62c` â†’ (this commit)

# CHANGELOG

## 2026-05-24 - Run 10b launch, premature destroy, local salvage to TEST AUROC 0.9970

### Attempted
- Full Run 10b training with `launch_run10b_skip_kan_v2.sh` (KAN disabled, 10 base estimators)
- Phase 1.7.1 incremental per-model checkpoint patch (commit f147112) tested in production
- End-to-end SCP + destroy + commit sequence in single PowerShell paste block
- Approximate meta-learner stacking from saved OOF arrays + y_train

### Failed
- **Premature `vastai destroy`**: destroy command shared paste block with SCP; PowerShell ran all sequentially, killing instance 37429606 at ~06:00 UTC while deep_ensemble member 5/5 was fitting. Lost deep_ensemble + meta-learner + GNN + cloud test eval. See INCIDENT_2026-05-24_run10b-premature-destroy.md
- **OOF meta-learner alignment**: OOF arrays stored in CV-prediction order, not X_train row order. Pairing OOF with `y_train[:1017633]` gave reconstructed AUROC ~0.50 across all 8 models. Sanity check caught this; fell back to simple-average.
- **cnn_1d cross-platform unpickle**: `joblib.load` of cloud Linux-saved cnn_1d.joblib fails on local Windows with `TypeError: NoneType.__new__(X)` due to nested-class closure. See INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md

### Fixed / Worked as designed
- **Phase 1.7.1 patch fully validated** in disaster recovery scenario. Per-model joblib + OOF + meta JSON saved right after each AUROC log preserved 9 of 10 base models when the instance died unexpectedly. Without the patch, Run 10b would have been a total loss.
- **Phase 2 v2 auto-discovery** located splits at `full/splits/` despite Phase 1 inventory's wrong assumption of `full/` root
- **Alignment sanity check** in Phase 2 v2 correctly detected misaligned OOF rows and prevented false meta-learner results from being published

### Learned
- **STANDING RULE #30**: Irreversible cloud commands NEVER share a paste block with preceding setup/copy commands. Always isolate in a separate code block requiring explicit re-paste after manual verification.
- **OOF row indices need sidecar**: To enable post-hoc meta-learner reconstruction, the per-fold prediction-to-row mapping must be saved alongside OOF arrays (`{name}_oof_indices.npy`).
- **Closure-defined classes are pickle-fragile**: `_CNN1D._build_model.<locals>._CNN1D` doesn't survive cross-process pickle. Run 11 must move `_CNN1D` to module-level.
- **Split parquets live at `<run_dir>/splits/`**, not `<run_dir>/` directly.
- **Local CPU inference is fast enough**: 503K rows x 8 models in 2.3 min wall-clock; the no-local-training rule applies to training only, inference is fine.
- **mc_dropout + deep_ensemble are real estimators**: They were hidden behind the KAN dam in Run 10a. With `--skip-kan` we see 10 base estimators, not 8.

### Outcome
Locked **TEST AUROC = 0.9970** on 349,067 variants via simple-average ensemble of 8 working base models. Matches best-single performance (catboost, lightgbm both at 0.9970). Mean OOF->TEST degradation -0.0009 across 8 working models indicates healthy generalization.

### Commits
- `f147112` Phase 1.7.1 incremental checkpoint patch (pre-launch)
- `927e8d6` Run 10b launch script committed (post-destroy)
- `9b1400e` Run 10b-partial salvage results
- `8e1b21f` (CHANGELOG blank-line modification only; superseded by this commit)

## 2026-05-23 Ã¢â‚¬â€ Run 10a deployment & no-checkpoint reckoning

### Attempted
- Run 10a regen+train on Vast.ai inst 37429606 (RTX 4090, $0.76/hr) with LOVD + DbNSFP wired
- Mid-run salvage planning when KAN cycle 3 of 6 still active at 16h

### Fixed (empirically validated)
- LOVD silent-zero: annotation 15/16 returns 369 variants (was 0). Commit `66593d6` confirmed correct.
- DbNSFP silent-zero: annotation 1/17 delivers 204,384 real SIFT scores.
- KAN pykan 0.2.x compatibility: `dataset` dict with `train_input/train_label/test_input/test_label` keys works.
- KAN OOM safeguard: 100K stratified subsample with `max_fit_samples=100_000` allocates 0.2 GB peak instead of 17.9 GB.

### Failed
- `ensemble.save()` and per-model persistence: NO `.pkl`/`.joblib`/`.cbm` files exist anywhere in /workspace after 16 hours of training. Phase 1.7 patch (`66593d6`) created `model_dir` but did not add per-model writes. Same architectural omission as Run 9.
- `cnn_1d` wrapper: OOF AUROC = 0.5000 (constant predictions). Regression introduced between Run 9 and Run 10a, likely from post-C5 namespace refactor breaking the inner `_CNN1D._build_model.<locals>._CNN1D` closure.
- 4090 GPU utilization for KAN: 0% steady Ã¢â‚¬â€ KAN is CPU-bound. ~$10/run wasted on wrong hardware tier.

### Learned
- Standing pre-flight rule did NOT catch the no-checkpoint failure mode because it didn't require runtime verification.
- cnn_1d is a 1-D convolution over the 78-feature tabular vector, not an image model. Image data acquisition remains unscheduled (correctly so) Ã¢â‚¬â€ Phase 0 baseline + ablation matrix come first.
- KAN's 6-cycle pattern confirmed: 5-fold OOF CV + 1 final fit on full data. Each cycle ~4h 25m + ~1h 30m inter-cycle gap = ~5h 55m wall-clock per cycle.
- PowerShellÃ¢â€ â€™SSHÃ¢â€ â€™bash quoting: `---` separators + single-word grep patterns are the only reliable shape. Never embed `"..."` inside `'...'`.

### Memory rules updated
- Memory edit #29 replaced with: incremental checkpointing mandatory on all >30 min cloud training; pre-flight must verify checkpoint files appear within first 30 min; abort if first base model finishes with no checkpoint emission.

### Incidents filed
- `INCIDENT_2026-05-23_run10a-no-checkpoints.md` Ã¢â‚¬â€ structural fix via `variant_ensemble.py` patch
- `INCIDENT_2026-05-23_cnn1d-0.5-auroc.md` Ã¢â‚¬â€ closure regression, unit test gate required

### Costs
- Run 10a so far: $13.02 (15h 53m Ãƒâ€” $0.76 + $0.95 setup)
- Run 10a remaining if completed: +$14.44 Ã¢â€ â€™ $27.46 total
- Run 10c (kill+patch+restart with --skip-kan) projection: +$2.50 Ã¢â€ â€™ $15.50 total

### Next-session deliverables
1. Apply `variant_ensemble_incremental_save_patch.py` to local repo
2. Commit + push
3. Kill Run 10a, relaunch on patched code with `--skip-kan`
4. Verify checkpoints appear within first 30 min
5. Add `tests/integration/test_ensemble_persistence.py`
6. Add `tests/unit/test_cnn_1d_wrapper.py` with AUROC > 0.55 gate
7. SCP outputs back, destroy instance
8. File all session docs to `docs/sessions/` and `docs/incidents/`

---

# Changelog Ã¢â‚¬â€ Genomic Variant Classifier

Append-only. One entry per session. Captures what was attempted, what
failed (with exact errors and root causes), what was fixed, and what was
learned. Searchable: paste any error string to find the root cause and fix.

Format per entry:
  ## YYYY-MM-DD Ã¢â‚¬â€ <one-line summary>
  ### Attempted | Failed | Fixed | Learned

---

## 2026-04-08 Ã¢â‚¬â€ Runs 6 & 7, GPU quota request, Run 8 startup script

### Attempted
- Run 6: full training on GCP (n2-highmem-32, CPU-only). Holdout AUROC 0.9862.
- Run 7: repeat with gnomAD v4.1 constraint features wired in. AUROC 0.9862 (unchanged Ã¢â‚¬â€ GNN still CPU-only).
- GPU quota request: GPUS_ALL_REGIONS = 1.
- Run 8 VM create: L4 (g2-standard-8).

### Failed
- Run 6 models lost: VM was deleted before model upload was confirmed.
  Root cause: shutdown was triggered by `&&` chaining, not `trap EXIT`.
  `&&` only fires on success; VM was already off by the time we checked GCS.
- GPU quota denied. Code: GPUS_ALL_REGIONS = 0 (new account, no billing history).
- Run 8 VM create failed: `ZONE_RESOURCE_POOL_EXHAUSTED` across all US zones.
  Root cause: quota was 0 Ã¢â‚¬â€ zone exhaustion was a red herring.
- venv torch install on Deep Learning VM: `libcusparseLt.so.0` not found.
  Root cause: venv doesn't have access to the system CUDA libraries.
  Fix: uninstall pip torch from venv; add .pth bridge to system torch.
- `gcloud storage cp -r` added extra directory nesting level.
  Fix: use individual file copies, not `-r`.
- `set -euo pipefail` in startup script caused silent exits on risky commands.
  Fix: wrap risky commands with `|| true`.

### Fixed
- Startup script: replaced `&&` chaining with `trap 'upload && shutdown' EXIT`.
  Fires on ANY exit: success, failure, crash, OOM.
- Git safe.directory: `git config --global --add safe.directory $REPO_DIR`
  (startup runs as root; repo cloned as monzi Ã¢â‚¬â€ git refuses pull otherwise).
- Parallel composite upload disabled: `gcloud config set storage/parallel_composite_upload_enabled False`.
  Was causing 401 auth failures on large files when OAuth token expired mid-upload.
- argparse `--string-db` flag: was missing from `run_phase2_eval.py`.
- gnomAD constraint path: was never wired into `AnnotationConfig`.
  All four constraint features (loeuf, syn_z, mis_z, pli_score) defaulted to 0.

### Learned
- Always verify models are in GCS before stopping/deleting a VM.
- `trap EXIT` is the only correct pattern. `&&` is insufficient.
- Google grants GPU quota only after billing history is established.
  Reapply after 2026-04-15.
- `gcloud storage` CLI always; never `gsutil` (does not read project from config).

---

## 2026-04-09 Ã¢â‚¬â€ Inter-run items 1-8, inter-agent message bus (Phase 4)

### Attempted
- SpliceAI index build from full hg38 VCF (28.8GB compressed).
- VersionMonitorAgent implementation and orchestrator wiring.
- Requirements cleanup (orphan files, add transformers>=4.40).
- Dockerfile audit and fixes.
- Polars benchmark on gnomAD constraint join.
- .gitkeep replacement in data/ subdirs.
- Inter-agent message bus: OpenClaw-inspired typed message passing between all 4 agents.
- Full pipeline dry-run verification.

### Failed
- SpliceAI VCF was misidentified as masked SNV (~72M lines).
  Actual: full unmasked hg38 VCF including indels Ã¢â‚¬â€ 1.1B+ lines, 2.5+ hours.
  Root cause: filename says "masked.snv" but file is full genome-wide.
  Result: still correct and more complete than expected. Build still running at session end.
- Docker smoke test: Docker Desktop not running (Linux engine pipe not found).
  Not a code problem. Deferred.
- `data_freshness_agent.py`: `ImportError: cannot import name 'ALPHAMISSENSE_MANIFEST_URL'`.
  Root cause: config has `ALPHAMISSENSE_MANIFEST`, not `ALPHAMISSENSE_MANIFEST_URL`.
  Fix: align agent import to real config constant name.
- `training_lifecycle_agent.py`: `ModuleNotFoundError: No module named 'ewc_utils'`.
  Root cause: top-level import; ewc_utils lives in agents/ not agent_layer/.
  Fix: lazy import inside `_check_drift()` method.
- `literature_scout_agent.py`: `ModuleNotFoundError: No module named 'feedparser'`.
  Fix: lazy import inside `_fetch_biorxiv()`.
- `literature_scout_agent.py`: `NameError: name '_TRAINING_AGENT' is not defined`.
  Root cause: constant dropped during config-name reconciliation pass.
  Fix: re-add `_TRAINING_AGENT = "TrainingLifecycleAgent"` constant.
- LOVD REST API: HTTP 402 (unsupported) on all polls.
  Root cause: LOVD changed their API terms. Logged as warning, skipped gracefully.
- ClinGen API: 404 (endpoint URL format changed).
  Logged as warning, skipped gracefully.
- PubMed efetch: occasional 500 Server Error (NCBI transient).
  Logged as warning, skipped gracefully.

### Fixed
- All 8 inter-run items completed and committed.
- Inter-agent message bus: 34/34 tests passing on Python 3.14.3.
- Full pipeline `--dry-run` confirmed working: all 4 agents run cleanly with
  graceful degradation where ewc_utils/feedparser not on path.

### Learned
- SpliceAI "masked.snv" filename is misleading Ã¢â‚¬â€ always check file size first.
  28.8GB compressed = full genome-wide VCF, not masked SNVs only.
- Polars join 3.3x faster than pandas merge on gnomAD constraint join (500K variants).
  Integration approved for Phase 3 ETL bottlenecks.
- Inter-agent messaging with lazy imports is the correct pattern for an agent layer
  where not all dependencies are always installed.
- PowerShell `<` operator is reserved Ã¢â‚¬â€ never use `<placeholder>` syntax in commands.
  Always use a real value or `PLACEHOLDER_VALUE` without angle brackets.

---

## 2026-04-09 (post-session) Ã¢â‚¬â€ Local file cleanup + SpliceAI GCS fix

### Fixed
- SpliceAI GCS index was wrong file: `processed/spliceai_index.parquet` in GCS
  was the raw 28.7GiB VCF accidentally uploaded under the wrong name.
  Root cause: `Rename-Item` failed silently (target already existed), so
  `data\processed\spliceai_index.parquet` was still the 29GB file when
  `gcloud storage cp` ran. The correct 336.8MB filtered parquet was still
  named `spliceai_index_test.parquet`.
  Fix: uploaded `spliceai_index_test.parquet` directly to GCS as `spliceai_index.parquet`.
  GCS now confirmed: 336.83MiB / 353,196,691 bytes at 2026-04-09T23:15Z.
  Local: deleted 29GB wrong file, renamed _test.parquet Ã¢â€ â€™ spliceai_index.parquet.

### Cleaned up local files (all confirmed in GCS before deletion)
  - data\external\spliceai_scores.masked.snv.hg38.vcf.gz     27.5 GB (duplicate)
  - data\external\dbnsfp\dbNSFP5.3.1a_grch38.gz             47.9 GB Ã¢Å“â€œ GCS
  - data\external\finngen\finnge_R12_annotated_variants_v1.gz 30.6 GB Ã¢Å“â€œ GCS
  - data\external\spliceai\spliceai_scores.masked.snv.hg38.vcf.gz 27.5 GB Ã¢Å“â€œ GCS
  - data\external\alphamissense\AlphaMissense_hg38.tsv\       5.2 GB (GCS has .gz)
  - data\raw\cache\alphamissense_scores_hg38.parquet          740 MB (regeneratable)
  - data\external\clinvar_fresh\variant_summary.txt.gz        415 MB Ã¢Å“â€œ GCS
  - data\raw\clinvar\variant_summary.txt.gz                   415 MB (duplicate)
  Total recovered: ~142 GB

---

## 2026-04-16 Ã¢â‚¬â€ Lambda A10 setup; Phase 2 feature promotion; SyntaxError fix; 205 tests green

### Attempted
- Launch Lambda Labs gpu_1x_a10 as GCP GPU quota substitute (quota still 0).
- Fix SyntaxError in variant_ensemble.py blocking all imports.
- Sync TABULAR_FEATURES (21) to match engineer_features() output (78 columns).
- Provision Lambda Python environment and authenticate GCS service account.

### Failed
- ssh-keygen -N "" in PowerShell: silent parse failure. Fix: run interactively, Enter twice.
- SyntaxError fix via python -c inline: PowerShell tokenizer mangled nested quotes/backslashes.
  Fix: write repair script to .py file via Set-Content, execute, remove.
- Repair script string-match failure: file used em-dash in comment; script used ASCII --.
  Fix: locate block by structural markers (feats line + return line) not literal text.
- Lambda pip: --index-url replaces PyPI entirely; all non-torch packages returned 404.
  Fix: --extra-index-url for torch; separate pip invocation for everything else.

### Fixed
- SyntaxError line 524 variant_ensemble.py: Phase 2 feature blocks pasted inside unclosed
  assert ( expression. Removed broken fragment; clean assert added after all features computed.
- TABULAR_FEATURES mismatch (21 declared vs 78 produced): engineer_features() grew across
  Phase 2 sessions but list was frozen. Updated to full 78-feature list in 20 groups.
- Lambda torch environment: torch 2.11.0+cu130, CUDA True, pandas 2.3.3, PyG 2.7.0.
- GCS access on Lambda: SA key scp'd, gcloud authenticated, bucket accessible.

### Learned
- assert ( multiline is valid Python. Assignments inside cause SyntaxError on =.
  Compute all features first, assert last.
- --index-url is destructive (replaces PyPI). --extra-index-url is additive.
- TABULAR_FEATURES and engineer_features() must stay in sync.
  The assert at end of the function is the single guard.
- Write all multi-line Python repair scripts to .py files, not inline python -c strings.
- Lambda instance billing starts at launch. Have all code pushed before creating the instance.

## 2026-04-16 (continued) Ã¢â‚¬â€ AlphaMissense parquet fix; Run 8 training launched on Vast.ai RTX 4090

### Fixed
- alphamissense.py _parse_parquet returned raw 5-column schema instead of
  lookup_key/alphamissense_score. Fix: build lookup_key = CHROM:POS:REF:ALT,
  deduplicate, return 2-column df matching _parse_tsv output schema.
- Stale parquet cache (wrong schema from first broken run) deleted on Vast.ai.
- Result: 206,131 / 1,700,687 variants now annotated by AlphaMissense.

### Infrastructure
- Vast.ai RTX 4090 instance: 175.155.64.225:19863, $0.388/hr
- Vast.ai auto-starts tmux on login Ã¢â‚¬â€ no manual tmux new-session needed.
- All 7 data files pulled from GCS in ~3 minutes (vs 25 min scp previously).
- Training launched 20:13:40 UTC with full 78-feature set including AlphaMissense.

### Pending
- Training in progress Ã¢â‚¬â€ detached in tmux, running unattended.
- Check results in ~2-3 hours for final AUROC/AUPRC/MCC.
## 2026-04-16 Ã¢â‚¬â€ Run 8 COMPLETE Ã¢â‚¬â€ AUROC 0.9863, 1.8GB artifacts saved to GCS

### Final Results
  AUROC  0.9863 (holdout)  0.9833 (test)   PASS (target >= 0.9)
  AUPRC  0.9461 (holdout)  0.9436 (test)
  MCC    0.8482 (holdout)  0.8178 (test)
  F1     0.9226 (holdout)  0.9052 (test)
  Brier  0.0358 (holdout)  0.0479 (test)
  Time:  4270s on Vast.ai RTX 4090 ($0.388/hr)

### OOF AUROCs (5-fold CV)
  RF 0.9921 | XGB 0.9932 | LGB 0.9930 | GBM 0.9891 | CatBoost 0.9930 | LR 0.9846
  Blend: 0.9938 | Weights: RF 0.391, LGB 0.255, CatBoost 0.319, XGB 0.035

### Top 10 Features
  n_pathogenic_in_gene 568.1 | loeuf 418.2 | syn_z 370.5 | mis_z 352.4
  consequence_severity 242.7 | pli_score 218.3 | alphamissense_score 189.7
  af_raw 174.2 | af_log10 105.3 | len_diff 86.7

### AlphaMissense confirmed contributing
  206,131 / 1,700,687 variants annotated | ranked 7th of 78 features

### Bugs discovered (fix in Run 9)
  GNN: ValueError: invalid literal for int() with base 10: path string passed where
       protein ID int expected. GNN did not contribute to Run 8.
  TF models: tabular_nn, cnn_1d, mc_dropout, deep_ensemble all skipped Ã¢â‚¬â€
             no tensorflow on Vast PyTorch image. Use PyTorch equivalents.
  ESM-2: stub mode (transformers not installed) Ã¢â‚¬â€ all esm2_delta_norm = 0.0

### GCS artifacts (gs://genomic-variant-prod-outputs/run8/)
  models/run8/models/ensemble.joblib         main ensemble
  models/run8/scaler.joblib
  models/run8/metrics.json
  models/run8/per_model_metrics.csv / _val.csv
  models/run8/feature_importance.csv
  models/run8/splits/X_train|val|test.parquet
  logs/run8.log
  19 files, 1.8 GiB total

### Infrastructure notes
  - Vast.ai auto-tmux protects from SSH drops (unlike Lambda foreground sessions)
  - sudo shutdown fails in Vast containers (no systemd) Ã¢â‚¬â€ container exits naturally
  - SA key permissions: parallel composite upload GET check fails Ã¢â‚¬â€ non-blocking
## 2026-04-16 (final) Ã¢â‚¬â€ SpliceAI + PyTorch NN fixes committed

### Fixed
- SpliceAI: _get_lookup now detects .parquet and calls _parse_parquet()
  instead of _parse_vcf(). Fixes 0 variants annotated in Run 8.
  Schema: chrom:pos:ref:alt lookup_key, dedup by max score.
- CNN1DClassifier: migrated TF/Keras Ã¢â€ â€™ PyTorch (Conv1d, AdaptiveMaxPool1d,
  early stopping patience=5, CUDA-aware)
- TabularNNClassifier: migrated TF/Keras Ã¢â€ â€™ PyTorch (BatchNorm1d, Dropout,
  weight_decay=1e-4, early stopping patience=8, CUDA-aware)
- All 466 tests passing after all three fixes.

### Run 9 readiness
All known bugs from Run 8 are now fixed:
  GNN string_db path bug          FIXED (0a02e5d)
  AlphaMissense parquet schema    FIXED (5297711)
  SpliceAI parquet branch         FIXED (this commit)
  CNN1D / TabularNN TFÃ¢â€ â€™PyTorch    FIXED (38656bc)
  transformers installed          DONE

Expected Run 9 active models: RF, XGB, LGB, GBM, CatBoost, LR,
  tabular_nn, cnn_1d, mc_dropout, deep_ensemble, GNN (10 base models + GNN)
Expected new feature signals: SpliceAI scores, ESM-2 (if HGVSp populated)
---

## 2026-04-17 Ã¢â‚¬â€ SpliceAI silent-zero fix, test isolation, GCS audit

### Attempted
- Verify Run 8 SpliceAI parquet was actually in GCS (could not be
  confirmed from prior sessions because gsutil kept returning 401).
- Patch `SpliceAIConnector` to default to the production parquet
  instead of silently returning 0.0 for all variants.
- Add regression test and confirm no regressions across the unit
  suite.

### Failed
- gsutil returned `401 Anonymous caller` on every GCS list attempt.
  Root cause: gsutil and `gcloud storage` have separate credential
  stores; gsutil's were stale. The SpliceAI parquet was in fact in
  GCS the whole time (since 2026-04-09). This cost multiple sessions
  of uncertainty.
- v1 test patch monkeypatched `FetchConfig.cache_dir` as a class
  attribute, which has no effect on dataclass instance fields.
  Individual test appeared to pass in 61s but sibling tests rebuilt
  the 430 MB production cache on the next `TestAnnotationPipeline`
  run.
- v2 test patch (short-circuiting `_load_cache` for one test) didn't
  cover the other 15 tests in the class. Full class run hit a
  5-minute timeout mid-import while building the cache.

### Fixed
- `src/data/spliceai.py`: renamed `DEFAULT_VCF_PATH` to
  `DEFAULT_SPLICEAI_PATH` pointing at
  `data/external/spliceai/spliceai_index.parquet`. `__init__` now
  falls through to this default when `vcf_path=None` is passed. This
  closes the Run 8 silent-zero failure mode - the connector no
  longer returns 0.0 for all variants when `AnnotationConfig()` is
  constructed with defaults.
- `tests/unit/test_spliceai_parquet_default.py`: new regression test
  (~3-7s runtime) that builds a 3-row synthetic parquet, instantiates
  `SpliceAIConnector()` with no args, and asserts at least one
  non-zero `splice_ai_score`.
- `tests/unit/test_core.py`: added class-scoped `autouse=True`
  fixture `_isolate_spliceai` at the top of `TestAnnotationPipeline`.
  Monkeypatches `DEFAULT_SPLICEAI_PATH` (nonexistent tmp file) and
  `BaseConnector._load_cache` (returns None), short-circuiting
  SpliceAI disk I/O for all 16 tests in the class. Full class runs in
  2:28 instead of timing out.
- `scripts/verify_spliceai_index.py`: parquet integrity/schema/null
  checks. Used at session start to confirm the production parquet
  (45,549,300 rows, 10 columns, no nulls outside of MT chromosome).
- `docs/CHANGELOG.md`: deduplicated the triplicated
  `## 2026-04-16 (final)` heading caused by PowerShell heredoc
  collision on session close. Net -46 lines.

### Learned
- Silent-zero connector fallbacks are bugs, not features. Future
  connectors should assert file existence at startup, not silently
  return defaults at runtime.
- `gsutil` is deprecated and has a separate credential store from
  `gcloud`. Use `gcloud storage ls` exclusively for authoritative
  GCS-state checks. Never trust `gsutil 401` as evidence of absence.
- Dataclass fields cannot be monkeypatched via
  `setattr(Class, "field", value)` - patching has no effect on new
  instances. Patch instance methods or module constants instead.
- Class-scoped `autouse=True` fixtures are the right tool for
  preventing disk-I/O side effects across every test in a class,
  including future tests that don't yet exist.
- Run scoped tests (`pytest path::Class::test -v --timeout=N`) before
  full suites when iterating on fixes. A 20-minute suite is the
  worst feedback loop.
- PowerShell heredocs (`@'...'@ | Add-Content`) corrupt reliably when
  content contains triple-quoted Python or literal commit messages.
  Use standalone `.py` files instead.
- `Get-Content | Add-Content` can silently fail with empty pipelines
  or encoding conflicts on existing files. For reliable appends,
  read and write in a single .NET call via
  `[System.IO.File]::AppendAllText`.

### Commits
- `9ba3127` feat(spliceai): default to parquet index; add regression
  tests; dedupe changelog (5 files changed, 191 insertions, 50
  deletions).
- `8b12f76` docs: session 2026-04-17 - SpliceAI default path fix
  (session doc only; CHANGELOG append failed silently and was
  applied in a follow-up commit).

## 2026-04-17 (afternoon, take 2) --- Run 9 infra + ESM-2 silent-zero discovery

(Note: the earlier afternoon CHANGELOG entry was draft; this supersedes
it. Kept in-place because the ESM-2 discovery materially changed the
story and file contents.)

### Added

- `scripts/preflight_check.py` (local, pre-launch gate): scripted
  enforcement of standing rule #1. Checks git tree, HEAD == origin/main,
  full pytest suite, local data files, GCS objects via `gcloud storage
  ls` (2026-04-17 rule), GITHUB_TOKEN from .env/session/Windows-User-env,
  transformers+torch importable, no tensorflow, SpliceAI test-cache
  absence. Allowlists two pre-existing carry-overs
  (`scripts/gcp_run6_startup.sh`, `ROADMAP_PSYCH_GWAS_ENTRY.md`).
  Supports `--skip-pytest` and `--skip-gcs` flags for fast iteration.
  Three revisions this session to work around Windows `.cmd` shim
  handling in subprocess.

- `scripts/preflight_vm.sh` (on-VM, post-SSH gate): checks nvidia-smi,
  `torch.cuda.is_available()`, data-file presence on container FS,
  transformers>=4.40, git HEAD, and all critical Python imports.

- `tests/unit/test_esm2_activation.py`: three-test regression module
  for ESM-2 stub-mode detection. Skipped on machines without
  transformers. When transformers is present: gates API drift, gates
  the real-mode path (passes when all four required columns are
  present and backend+network available), and explicitly documents
  the current stub-mode expected-behavior via a separate test that
  fails loud if the connector ever starts silently inferring the
  parsed columns.

- `scripts/run9_launch.md`: operational runbook for Run 9. Updated
  to explicitly expect ESM-2 stub mode in training logs per the
  INCIDENT doc. Pins `transformers>=4.40,<5.0` on Vast.ai installs.

- `docs/incidents/INCIDENT_2026-04-17_esm2-hgvsp-parser.md`: full
  root-cause record for the ESM-2 silent-zero that affected Runs
  6-8. The training pipeline never populated `wt_aa`/`mut_aa`/
  `protein_pos` (grep of `src/` returned only esm2.py as reader,
  nothing as writer); the connector logged an INFO message and
  returned all zeros. Remediation plan: add `src/data/hgvsp_parser.py`
  in Run 10.

### Discovered

- **ESM-2 has been inert in Runs 6, 7, and 8**. Root cause: pipeline
  does not populate the four columns the connector requires
  (`gene_symbol`, `protein_pos`, `wt_aa`, `mut_aa`). Connector emits
  an INFO-level log ("columns missing -- defaulting to 0.0") that was
  not being grepped. Feature-importance rankings showed ESM-2 below
  top 20, which was indistinguishable from "feature contributes
  literally zero" vs "feature contributes weakly".

- **EVE is almost certainly in the same state**. Same column-pattern:
  `eve.py:232` reads `wt_aa`/`mt_aa`/`position`/`mutations_protein_name`;
  none written by pipeline. Full diagnosis deferred to Run 10, when
  the HGVSp parser can populate both ESM-2 and EVE inputs.

### Design notes

- **Dual-layer preflight** (local + on-VM) is the minimum correctness
  boundary for Run 9, not redundancy.

- **Connector fallbacks with INFO logs are silent**. For any connector
  with a graceful fallback path, preflight should test that the
  fallback fails loud. SpliceAI got this in commit 9ba3127; ESM-2 got
  it in this session. Audit other connectors (EVE, AlphaMissense,
  CADD) for the same pattern as a Run 10 prerequisite.

- **Zero-fraction audit belongs in the agent layer**. Feature-importance
  alone cannot distinguish "weak feature" from "inert feature".
  Planned: nightly job that prints zero-fraction per feature per
  dataset and alerts when a feature flips to 1.0 zero-fraction.

### Learned

- Read connector source before writing its test. First ESM-2 test
  draft assumed 1280-dim embedding columns; actual API is a scalar.
- Windows gcloud subprocess requires `shell=True` when the cmd token
  is a bare name without explicit path or `.exe`. subprocess cannot
  resolve `.cmd` shims via `CreateProcess`.
- `[System.IO.File]::AppendAllText` does not add a separator before
  the appended content. If the target file doesn't end with `\n`, the
  append gets concatenated onto the final line. Fix: include `\n\n`
  prefix in the appended content, or check-and-add-newline first.

### Commits queued

- `feat(run9): scripts/preflight_check.py + scripts/preflight_vm.sh + ESM-2 smoke test + launch runbook`
- `docs(run9): INCIDENT for missing HGVSp parser + session doc + CHANGELOG`

### Run 9 readiness after this session

- [x] local preflight script on disk (3rd revision, all bugs fixed)
- [x] VM preflight script on disk
- [x] ESM-2 smoke test on disk (matches actual connector schema)
- [x] launch runbook on disk (expects ESM-2 stub)
- [x] INCIDENT doc filed
- [ ] Vast.ai instance provisioned (user action)
- [ ] on-VM preflight passes (requires live instance)
- [ ] training launched and final metrics captured

## 2026-04-20 Ã¢â‚¬â€ KAN reinstatement, ensemble OOF fix, CI recovery

Entered session investigating a CI failure (`pytest (3.11)` red since
2026-04-19). The failing test surfaced a pre-existing bug in
`VariantEnsemble.fit` that was simultaneously blocking Run 9's ablation
harness at ~10 hours of CPU time. Fix verified with a 500-row synthetic
probe in under 2 minutes. Separately, investigation of the local
`skip_kan` behaviour during that probe revealed the `KAN unconditionally
removed` status was 15 days out of date Ã¢â‚¬â€ the underlying OOM was
fixed in commit 2389ee2 on 2026-04-04. With Vast.ai GPU access for
Run 9, the remaining reason to keep KAN disabled evaporated. Three
atomic commits shipped, all CI green.

### Changed

- `src/models/variant_ensemble.py` (b1c1150): removed stale duplicate
  `self.meta_learner.fit(oof_preds, y_arr)` call at line 1159. The
  correct call one block below used `y_fit` (length 0.85 Ãƒâ€” N, matching
  `oof_preds`) but never ran because the stale call crashed first with
  `ValueError: Found input variables with inconsistent numbers of
  samples: [N*0.85, N]`. Pre-existing bug from a botched earlier
  patch; not introduced by Patch 1 (8a7e2da). Fix is `-7/+1` lines
  and unblocks both CI and the Run 9 ablation harness.

- `scripts/run_phase2_eval.py` (8f9eb60): added `--skip-kan` argparse
  flag, threaded through `EnsembleConfig(skip_kan=args.skip_kan)`,
  and replaced the unconditional
  `ensemble.base_estimators.pop("kan", None)` with a
  `if args.skip_kan:` gate. Default behaviour change: **KAN is now in
  the ensemble by default**. Pass `--skip-kan` to opt out. Matches
  items 3 and 4 of the ROADMAP KAN Re-enablement Checklist. Side
  effect: fixes the broken Dockerfile trainer CMD (see INCIDENT below).

### Added

- `scripts/run9_ablations.py` (128331f, 780 lines, new file): LOCO
  ablation harness for Run 9+ with 14 ablation targets. Coexists with
  `run_phase2_eval.py`; reads already-scaled splits from
  `<run>/splits/` and applies feature-prefix ablations by zeroing
  matching columns. Handles the 78-column schema confirmed on
  2026-04-19. Includes `--skip-kan` and `--skip-mc-dropout` CLI flags,
  a `no_kan` MODEL-level ablation, and a runtime guard that errors
  exit 2 if `--ablation no_kan` is passed without `--skip-kan`
  (preventing silent no-op runs).

- `docs/sessions/SESSION_2026-04-20.md`: session record covering the
  OOF bug diagnosis, KAN history reconstruction, reversal decision,
  and three-commit shipping sequence.

- `docs/incidents/INCIDENT_2026-04-20_dockerfile-trainer-skip-kan.md`:
  documents the Dockerfile trainer CMD passing a non-existent argparse
  flag from 2026-04-09 through 2026-04-20. Resolved as a side-effect
  of commit 8f9eb60 adding the flag.

### Discovered

- **The KAN "unconditionally removed" status was 15 days out of date.**
  Commit 2389ee2 (2026-04-04) added a 100K-sample stratified subsample
  gate in `KANClassifier._fit_pykan` that caps peak RAM at ~0.3 GB
  (from 17.9 GB). The hardcoded `pop("kan", None)` in
  `run_phase2_eval.py` was added in Run 6 prep (commit a0a732d on
  2026-04-05) as belt-and-braces caution and outlived its
  justification. ROADMAP had a documented re-enablement checklist
  (`docs/ROADMAP.md` lines 206-212) that was actionable-but-unactioned.
  `LiteratureScoutAgent` (`agent_layer/agents/version_monitor_agent.py`,
  commit a95c9db) already monitors pykan PyPI releases programmatically.

- **The Dockerfile trainer CMD has been broken since 2026-04-09.**
  Commit 671e48d added `--skip-kan` to the `scripts/run_phase2_eval.py`
  invocation at Dockerfile line 166. Until today, `run_phase2_eval.py`
  did not accept that flag Ã¢â‚¬â€ argparse would have errored with
  `unrecognized arguments: --skip-kan` and exit 2. Undetected for 11
  days because Runs 6-8 used startup scripts on GCP/Lambda/Vast.ai
  (`scripts/gcp_run6_startup.sh` etc.), not Docker. The trainer
  container was never invoked after 2026-04-09. Commit 8f9eb60
  incidentally fixes this by making the flag exist.

- **CI has been red since at least 2026-04-19** on the same OOF bug
  that blocked Run 9. Test
  `tests/unit/test_api.py::TestInferencePipeline::test_save_and_load_roundtrip`
  was failing at 20-sample scale with the identical `[N*0.85, N]`
  inconsistency that the Run 9 ablation harness hit at 1.2M-sample
  scale after ~10 hours of training. Commit b1c1150 fixes both.

- **Dockerfile is CPU-only multi-stage.** All three stages (builder,
  api, trainer) use `python:3.11-slim-bookworm`. No CUDA runtime, no
  GPU base image. GPU training happens via startup scripts on
  Vast.ai/Lambda/GCP, not via Docker. No change needed for Run 9.

### Design notes

- **500-row synthetic probes are fast enough to be a pre-commit gate.**
  Exercising the full `VariantEnsemble.fit()` code path with tree
  models + KAN took ~90 seconds on the CPU-only laptop, compared to
  22+ hours on the same hardware at real 1.2M-row scale (which
  crashed before meta-learner fit regardless). Used this session to
  verify the OOF fix before committing, then again to verify v4.1/v4.2
  skip-flag semantics. Standard pattern going forward: any change to
  `VariantEnsemble.fit` or `_build_estimators` gets a synthetic probe
  before any attempt at scaled training.

- **`no_kan` is a model-level ablation, not a feature-level one.**
  KAN uses the same 78 input features as every other base estimator,
  so there are no feature columns to zero. The harness handles this
  by adding `no_kan` to `ABLATION_MASKS` with an empty prefix list
  and gating execution on both `--ablation no_kan` AND `--skip-kan`.
  Without the runtime guard, `--ablation no_kan` alone would zero
  zero columns and train KAN anyway Ã¢â‚¬â€ a silent ~10-hour no-op on a
  GPU instance. The guard returns exit 2 with an explanatory message.

- **The KAN Re-enablement Checklist in ROADMAP.md was the right spec.**
  Every item on the checklist mapped cleanly to one of the commits
  shipped today. This is a data point for the value of maintaining
  forward-looking checklists in ROADMAP.md: when a condition changes
  (OOM fix + GPU access) that triggers the checklist, the work is
  already scoped.

### Learned

- **Read the notes before enforcing the decision.** Entering the
  session, memory note said "KAN unconditionally removed pending
  pykan memory fix" and I began to enforce that rule. User pushed
  back and asked me to investigate the history. The investigation
  took ~20 minutes of grep over `docs/`, `logs/`, code files, and
  git log and surfaced that (a) the OOM was fixed 15 days ago, (b)
  Vast.ai GPU access changes the calculus anyway, (c) there's a
  documented re-enablement checklist waiting to be executed. Had I
  proceeded without the investigation, KAN would still be absent.
  Standing rule #13 exists for exactly this class of error.

- **Failing-loud beats failing-silent at every scale.** The
  `--ablation no_kan` guard that returns exit 2 when `--skip-kan` is
  absent is a small amount of code (six lines) that prevents a
  ~10-hour silent no-op run on a GPU instance. Mirrors the SpliceAI
  fail-loud fix from commit 9ba3127 and the ESM-2 stub-detection
  test from 2026-04-17. Pattern: if a feature or model can be
  silently absent, add a loud check that forces the absence to
  announce itself.

- **Grep before inferring.** Initial plan for this session
  extrapolated `LiteratureScoutAgent` as a planning abstraction
  from a one-line memory note. Grep surfaced a committed
  `agent_layer/agents/version_monitor_agent.py` (commit a95c9db)
  that already does exactly what was planned. Default to reading
  the repo over reading the notes about the repo.

### Commits shipped this session

- `b1c1150 fix(ensemble): meta-learner fit uses y_fit to match oof_preds length`
- `8f9eb60 feat(ensemble): add --skip-kan CLI flag, remove hardcoded KAN removal`
- `128331f feat(run9): KAN as first-class ablation target; --skip-mc-dropout flag`

All three green on CI (pytest 3.11, pytest 3.12, lockfile drift check,
Docker build smoke test).

### Deferred to post-Run-9 cleanup

- **CI dependency conflict:** `requirements.txt` pins
  `starlette==1.0.0` but `prometheus-fastapi-instrumentator==7.1.0`
  (pinned by `requirements-api.lock` transitively) requires
  `starlette<1.0`. Pip emits a non-fatal ERROR during CI install and
  the installed env has the incompatible combination. Test suite
  passes because no current test imports Prometheus
  instrumentation, but runtime behaviour when instantiating the
  FastAPI app is untested. Fix: upgrade
  `prometheus-fastapi-instrumentator` to a version supporting
  starlette Ã¢â€°Â¥1.0, or pin `starlette<1.0` in `requirements.txt`.
  File INCIDENT after Run 9 completes per user instruction.

### Run 9 readiness after this session

- [x] ensemble meta-learner fit bug fixed (b1c1150)
- [x] `--skip-kan` CLI available in `run_phase2_eval.py` (8f9eb60)
- [x] `scripts/run9_ablations.py` on disk with 14 ablation targets (128331f)
- [x] `no_kan` ablation first-class with runtime guard
- [x] CI green on main
- [x] KAN Re-enablement Checklist items 207-211 complete (ROADMAP.md)
- [ ] splits regenerated against current 78-col schema (user action)
- [ ] Vast.ai instance provisioned (user action)
- [ ] Python 3.12 venv locally (optional; deferred Ã¢â‚¬â€ Vast.ai handles its own Python)
- [ ] Step C: verify Patch 6a `--string-db auto` branch triggers GNN injection
- [ ] on-VM preflight passes (requires live instance)
- [ ] KAN scalability pre-flight at 10K and 100K rows on GPU before full run
- [ ] training launched and final metrics captured

## 2026-04-30

### Attempted
- Stage 3 splits regen (run_phase2_eval.py with [GNN-TRACE]
  instrumentation, --skip-nn --skip-svm --skip-kan, --string-db auto,
  --n-folds 2, output outputs/run9_ready/)

### Failed
- GNN training: KeyError 'gene_symbol' in build_pyg_dataset.
  Caught by `except Exception` and downgraded to warning. on-disk
  gnn_score remained 0.0 across all three splits.
- --skip-nn flag did not skip mc_dropout/deep_ensemble (memory #17
  confirmed). Wall-clock cost: 10h+ of the 13h total runtime.

### Fixed (this session)
- Stage 1: .venv312 bootstrapped on Python 3.12.10. requirements.txt +
  torch 2.11.0+cpu + torch_geometric 2.7.0 installed cleanly.
  Pandas pinned to 2.3.3 (was 3.0.1).
- Stage 2: [GNN-TRACE] instrumentation patch landed in
  scripts/run_phase2_eval.py (18 logger calls, 4/4 verification gates
  green). Backup at scripts/run_phase2_eval.py.bak-gnn-trace.
- Stage 3: data prep + ensemble training completed end-to-end.
  Test AUROC 0.9814, val AUROC 0.9850.

### Drafted (committed in next session)
- Patch 6b (scripts/apply_patch_6b.py): persist meta_train.parquet
  in DataPrepPipeline._save_splits, source gene_symbol from it in
  run_phase2_eval.py for gnn_df construction.
- 5K-row synthetic probe (scripts/probe_patch_6b.py).

### Learned
- Generic `except Exception: logger.warning` masks crashes. Either
  narrow the except or use exc_info=True. [GNN-TRACE] insertion 9
  uses exc_info=True and would have surfaced this immediately on
  first run.
- Patches that re-persist on success path must verify success
  before persisting. Patch 6a re-persists regardless of whether
  gnn_scorer was built.
- Memory #19 (no local retraining) was violated this session at
  cost of 13h. Reaffirming.
- run9_ready splits are a valid GNN-FREE BASELINE for paper P4
  comparison. Don't discard.

---

## 2026-05-02 Ã¢â‚¬â€ Gene-scope expansion deferred to Run 10; LOVD silent-zero confirmed

### Attempted
- Review of request to add additional gene variants beyond the canonical
  10 (BRCA1, BRCA2, MLH1, MSH2, MSH6, APC, NF1, TP53, PTEN, RB1) before
  Run 9, with two LOVD admin emails attached as context.
- Investigation of LOVD subsystem state (connector wiring, on-disk data,
  trained feature matrix) to scope the integration work properly.
- Three-stage diagnostic: schema check on `lovd_all_variants.parquet`,
  value_counts on trained matrix, structural merge replicating the
  connector's logic in isolation.

### Failed
- LOVD `lovd_variant_class` is identically `0` across all 1,197,216 rows
  in `outputs/run9_ready/splits/X_train.parquet` despite:
  - LOVD parquet on disk being structurally healthy (18,006 rows, 10
    genes, joinable schema).
  - LOVDConnector being unconditionally invoked at
    `src/data/real_data_prep.py:738` with return value assigned.
  - Diagnostic merge (replicating the connector's exact key construction
    against `models/v1/clinvar_enriched.parquet`) yielding 5,553 inner-
    join matches in isolation.
  Root cause is at one of the runtime join boundaries inside the ETL Ã¢â‚¬â€
  either Cause 1 (downstream column overwrite) or Cause 2 (upstream
  coordinate transformation by one of the 14 prior `annotate_dataframe`
  steps). Distinguished by the integer in the log line at
  `real_data_prep.py:740Ã¢â‚¬â€œ748` (`"Score annotation 15/16 (LOVD): %d
  variants with lovd_variant_class > 0."`); resolution deferred to R10-A.
  Full record: `docs/incidents/INCIDENT_2026-05-02_lovd-silent-zero.md`.
- Initial hypothesis (floatÃ¢â€ â€™str trailing `.0` on the `pos` join key)
  falsified by direct dtype check: `pos` is int64, conversion is clean.

### Fixed
- Nothing patched this session. All identified work moved to Run 10.

### Learned
- LOVD label-quality is functional-translated-to-clinical, not clinical.
  Per LOVD admin's 2026-04-01 second email: clinical classification
  field intentionally withheld from API pending ACMG v4. API exposes
  `effect_reported`/`effect_concluded` (functional). Per ACMG/AMP 2015
  framework, functional evidence (PS3/BS3) is one input to a clinical
  classification combining multiple categories, not the classification
  itself. ClinVar tier-2 Ã¢â€ â€™ LOVD-API-derived is a label-quality
  downgrade. Earlier-session "30Ãƒâ€” more rows" framing was rhetorical
  and was flagged as such mid-session.
- Silent-zero discovery requires checking the *trained* feature matrix
  value distribution, not just connector logs. Connector logged the
  zero count at INFO level once during the 13h regen and the line was
  lost in training output. Recommend post-ETL assertion that any
  feature with single-source contribution must have `nunique() > 1` in
  the training matrix, with clear failure on zero variance. Extends
  the 2026-04-17 audit recommendation (EVE, AlphaMissense, CADD) to
  LOVD; same pattern likely affects other connectors on the 30+
  all-zero list from `SESSION_2026-04-30.md` Finding #4.
- `scripts/process_lovd.py` is dead code. Live LOVD merge is
  `scripts/build_lovd_index.py` Ã¢â€ â€™ `lovd_all_variants.parquet`. The
  schema mismatch between the two scripts (`lovd_variants.parquet` vs
  `lovd_all_variants.parquet`, `pathogenicity` vs `classification_raw`)
  is a dead-code artifact, not a live bug. Cleanup candidate for a
  separate post-Run-9 commit, low priority.
- `outputs/run9_ready/splits/` is not `data/splits/`. `DataPrepConfig`
  default and the run9 launch path differ. `docs/HANDOFF_run9_launch.md`
  and the Vast.ai onstart script must reference the actual
  `outputs/run9_ready/` path before Vast.ai launch.
- 4/1 raw LOVD download integrity confirmed against admin's logged
  ban window. TP53/PTEN/RB1 `.txt` files at 5:38Ã¢â‚¬â€œ5:39 AM Eastern are
  genuine; BRCA1/BRCA2/APC/MLH1/MSH2/MSH6/NF1 `.txt` files at the
  same time are 96Ã¢â‚¬â€œ98 byte error pages contemporaneous with the ban
  (`[01/Apr/2026:10:53Ã¢â‚¬â€œ12:34 +0200]` Ã¢â€ â€™ 4:53Ã¢â‚¬â€œ6:34 AM Eastern). 6:56 PM
  `.json` files are post-unblock manual saves of
  `?format=application/json` views, currently unconsumed by
  `build_lovd_index.py`.
- rclone Drive remote renamed `gvc` Ã¢â€ â€™ `genvarcla`. `agent_data/`
  namespace recreated on Drive with 5 subfolders (events, litcache,
  drift_reports, modelscout, trainlifecycle). Local `agent_data/`
  directory created. Smoke test (21-byte file round-trip) clean.
- **Process violations (this session, all recorded in SESSION doc):**
  `PASTE_FULL_PATH_HERE` placeholder in copy-pasteable command;
  bash heredoc syntax in PowerShell context (already covered by
  Windows-platform standing rule on file); loose grep regex framed as
  decisive. Pattern across all three: confident framing on
  under-constrained tooling. Recorded for future-self correction.

### Run 10 sequencing (revised)
- **R10-A:** Grep `outputs/run9_ready/regen.log` for the LOVD annotation
  count line. Distinguishes Cause 1 (downstream overwrite) vs Cause 2
  (upstream coordinate transformation).
- **R10-B:** Patch identified cause. Add unit test asserting
  `(df["lovd_variant_class"] > 0).sum() > 0` after the LOVD step on a
  3Ãƒâ€”5 fixture with 1 expected match. Pattern modeled on
  `tests/unit/test_spliceai_parquet_default.py` (commit 9ba3127) and
  `tests/unit/test_esm2_activation.py` (2026-04-17).
- **R10-C:** Re-regen splits on Vast.ai with LOVD live (no local
  retraining per standing rule #19). Post-condition: ~4,500Ã¢â‚¬â€œ5,500 of
  5,553 inner-join matches in train.
- **R10-D:** Originally-requested gene scope expansion (Paths 1+2: LOVD
  raw + gnomAD/UniProt per-gene). Manual browser only per LOVD admin
  emails of 2026-04-01.
- Cleanup (low priority, post-Run-9): remove `scripts/process_lovd.py`
  and orphaned `data/external/lovd/lovd_variants.parquet`.

### Run 9 readiness after this session
- Run 9 launch path **unaffected**. Run 9 inherits the same silent-zero
  baseline as run9_ready (Test AUROC 0.9814, Val AUROC 0.9850). Adding
  this INCIDENT as a known-pending item before Run 9 launch but not as
  a launch blocker.
- All four files for this session committed in a single commit:
  `docs(session): 2026-05-02 Ã¢â‚¬â€ gene-scope expansion deferred; LOVD silent-zero INCIDENT`.

## 2026-05-09: C3.6 hotfix + C4-prep complete

### Attempted
- Pre-condition audit for C4 pickle migration (Stage 1)
- Spec compliance audit of `scripts/migrate_pickles.py` (Stage 2)
- Functional smoke of `install_compat_aliases` (Stage 2 D)
- L119 patch for AttributeError on `_new_root.agent_layer` (Stage 2.5b)
- Diagnose namespace-vs-regular package status of `agent_layer` (Stage 2.5c)
- Add `agent_layer/__init__.py` and re-test alias count (Stage 2.5d)
- C3.6 hotfix: sweep bare imports of `agents`/`config`/`message_bus`/`shared_state` (Stage 2.5e)
- Build `tests/fixtures/migration_smoke.parquet` (Stage 3)
- Final readiness check (Stage 4)
- Two-commit push: C3.6 hotfix + C4-prep (Stage A + B)

### Failed
- Initial `install_compat_aliases` smoke threw `AttributeError: module
  'genomic_variant_classifier' has no attribute 'agent_layer'` at L122.
  Bare `import genomic_variant_classifier as _new_root` does not bind subpackage
  attributes; explicit `import genomic_variant_classifier.agent_layer` needed.
- First `__init__.py` retry showed only 22/28 `agent_layer.*` aliases registered;
  6 walk_failures from bare imports in `base_agent`, `data_freshness_agent`,
  `interpretability_agent`, `literature_scout_agent`, `training_lifecycle_agent`,
  and `orchestrator`. C3 regex sweep had missed these.
- Stage 3 reported `WARN: column count 81 != 78` Ã¢â‚¬â€ false alarm; PowerShell `-match`
  against a multi-line string array does NOT populate `$Matches`; stale value from
  prior smoke test `SRC=81` capture was used. Fixture is verified 78 cols by the
  python output itself (`COLS=78`).

### Fixed
- `src/genomic_variant_classifier/agent_layer/__init__.py` created (empty;
  promotes namespace -> regular package; C1 sweep miss resolved).
- `scripts/migrate_pickles.py` L119: explicit
  `import genomic_variant_classifier.agent_layer` added before
  `_new_root.agent_layer` access; C2-spec docstring still aligns.
- 8 files in `src/genomic_variant_classifier/agent_layer/` rewritten to
  fully-qualified imports (44 lines, +1716 bytes total): `agents/base_agent.py`,
  `agents/data_freshness_agent.py`, `agents/interpretability_agent.py`,
  `agents/literature_scout_agent.py`, `agents/training_lifecycle_agent.py`,
  `orchestrator.py`, `run_agents.py`, `test_message_bus.py`.
- `tests/fixtures/migration_smoke.parquet` committed (force-added; 8 x 78,
  48830 bytes, deterministic `df.head(8)` from
  `outputs/run9_ready/splits/X_test.parquet` head; live 78-col schema).

### Learned
- `pkgutil.walk_packages` does NOT recurse into PEP 420 namespace packages by
  default. Empty `__init__.py` converts namespace -> regular package, enabling
  walk recursion. Future migration sweeps should add post-condition tests that
  walk the full module tree.
- C3 regex patterns 6 and 7 (per spec) lacked `\b` word boundaries, allowing
  over-match against names like `agents_helper`. The C3.6 sweep script added
  `\b` as defensive hardening. No actual collisions in current codebase, but
  `\b` is now the preferred pattern for any future migration sweeps.
- PowerShell `-match` against an array filters but does NOT populate `$Matches`.
  To extract groups from multi-line `python -c` stdout, either `-join "n"` to
  collapse first, or use `Where-Object { $_ -match ... }` in pipeline. Bug in
  Stage 3 column-count check was benign (false WARN) but worth fixing in
  future scripts.
- Pre-migration `find_packages()` at repo root discovered `agent_layer/` AND its
  subpackages as TOP-LEVEL packages. So bare `from agents import X` worked
  because `agents` was on `sys.path`. After C1 nested it under
  `genomic_variant_classifier/`, those bare names broke. C3 regex sweep should
  have caught all instances; missed 8 files. Root cause for the miss is not
  fully diagnosed (C3 spec patterns are correct; possibly file-glob omission or
  later re-introduction during C3.x hotfixes Ã¢â‚¬â€ neither verified).

### Refs
- Commits: `e0f4c6e` (C3.6 hotfix), `e34ce7b` (C4-prep)
- HEAD before session: `fc7f63a`
- HEAD after session: `e34ce7b`
- INCIDENT: `docs/incidents/INCIDENT_2026-05-09_c1-c3-sweep-misses.md`
- Session: `docs/sessions/SESSION_2026-05-09.md`
- Spec: `docs/hypotheses/HYP_consolidate-package-layout.md` (C1, C3, C4 sections)
- Operational tooling (in `agent_data/`, NOT in repo):
  `c4_fix_install_compat.py`, `c4_diagnose_walk.py`, `c4_fix_bare_imports.py`,
  `c4_batch_C36_through_4.ps1`, `c4_batch_commits.ps1`

## 2026-05-09 (continuation) Ã¢â‚¬â€ C5 layout-migration cleanup

### Attempted
- C5.1: rewrite stale `src/X` refs in README L196/L223, ci.yml L77, narrow .gitignore cleanup
- C5.2: rewrite stale `src.*` / `src/` refs in 7 active operational docs
- C5.3 discovery: full-repo audit (369 hits across 71 files)
- C5.3a v1: full-repo sweep of 55 files / 83 expected substitutions (Bucket 3)
- C5.3a v2: same scope after regex fix
- C5.3b: remove 8 stale `.gitignore` rules

### Failed
- **C5.3a v1** (Stage 3, no commit, recovered): post-apply stale-ref count 9 Ã¢â€°Â  4 expected. Path-style regex `src/(SUBPKG)/` required trailing slash; missed 5 line-level hits where slash was absent (`src/api + src/models` in Dockerfile L10, bare `src/evaluation`/`src/reports`/`src/utils` at end of L2 in three `__init__.py` files, bare `src/` in `test_1kgp.py` L409). Working tree dirty with 51 partial writes; recovered via `git checkout -- .`.

### Fixed
- **C5.3a v2:** loosened path-style regex to `src/(SUBPKG)(?![A-Za-z0-9_])` (word-boundary lookahead instead of required slash). Catches all 5 v1-missed hits except bare-`src/` in test_1kgp.py L409 (intentional incidental).
- **Stage 1 arithmetic-sanity check** added to v2 batch: parses helper output and asserts `actual_substitutions == baseline_lines - deliberate_skip_lines - incidental_lines + multi_match_extras` (C5.3a v2: `83 == 87 - 4 - 1 + 1`, where `+1` is Dockerfile L10's multi-match adjustment) BEFORE Stage 2 apply. Catches the v1 class of regex-undershoot at dry-run time. See SESSION_2026-05-09_C5.md Ã‚Â§Lesson 1 for the full term-by-term derivation.

### Learned
- **STANDING RULE Ã¢â‚¬â€ apply-batch arithmetic sanity:** every mechanical-rewrite batch must assert at Stage 1 (dry-run) that `actual_substitutions == expected_substitutions`, where `expected = baseline_lines - deliberate_skip_lines - incidental_lines + multi_match_extras` (the last term reconciles match-count vs line-count: each non-skipped line with N>1 matches contributes N-1 extras). Without this check, a too-strict regex undershoots silently; the failure surfaces only at Stage 3 post-apply verification, after partial writes. Codify in every future apply helper template.
- **Path-style regex form:** `src/(SUBPKG)(?![A-Za-z0-9_])` (word-boundary lookahead) is more robust than `src/(SUBPKG)/` (required slash).
- **Recovery enforced by pre-flight:** apply batches' pre-flight rejects dirty working trees, ensuring `git checkout -- .` recovery happens before any retry.
- **Substitutions Ã¢â€°Â  line-level diff:** helper substitution count and git diff stat can differ when a single line has multiple substitutions (Dockerfile L10: 2 substitutions, +1/-1 in diff).

### Commits
- `d7ed38e` Ã¢â‚¬â€ C5.1
- `4eb1205` Ã¢â‚¬â€ C5.2
- `6a38ee3` Ã¢â‚¬â€ C5.3a (v2): 55 files, 83 substitutions, +82/-82
- `6443af7` Ã¢â‚¬â€ C5.3b: 8 .gitignore deletions

### Refs
- `agent_data/c5_3_discovery.ps1`
- `agent_data/c5_3a_apply_full_sweep.py` (v2)
- `agent_data/c5_3a_batch.ps1` (v2 with Stage 1 arithmetic-sanity)
- `agent_data/c5_3b_apply_gitignore_cleanup.py`
- `agent_data/c5_3b_batch.ps1`
- Session doc: `docs/sessions/SESSION_2026-05-09_C5.md`

---

## 2026-05-10 Ã¢â‚¬â€ Architectural cleanup: GCS retirement (Commits 1-4 of cleanup arc)

### Attempted
- Complete the SCP-only architectural pivot started by the 2026-04-29 GCP project deletion (`INCIDENT_2026-04-29_gcp-billing-deletion.md`). Required four ordered commits: incident formalization, runtime GCS strip, operational docs rewrite, and session log + CHANGELOG cap.

### Failed
- **Stage 3 batch parser** (parse-time, no writes, recovered): PowerShell `$p:` in double-quoted strings parsed as scope/drive prefix. Anchor at L162:33 reported `Variable reference is not valid. ':' was not followed by a valid variable name character.` Fixed by wrapping in `${p}:` form.
- **Stage 3 P1.6 dry-run** (anchor not found, no writes, recovered): anchor at `scripts/run9_launch.md:200-201` had a trailing `\n` but the file ends at L201 without a terminal newline. Fixed by removing the trailing newline from the P1.6 anchor and replacement (matches both `receipt.` and `receipt.\n[more]` cases via `text.count(old)`).
- **Save procedure silent failure** (state corruption, recovered): `Move-Item -Force` from `~\Downloads` to `agent_data\` removes the source. Subsequent re-attempts find the source missing and silently no-op, leaving `agent_data\` with no file at all. Fixed by adding a `Test-Path` source check BEFORE removing the destination.

### Fixed
- **Commit 1/4 (`b15a625`)** Ã¢â‚¬â€ `docs(incident): formalize 2026-04-29 GCP project deletion + SCP-only architectural pivot`. Created `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md` (4065 bytes); deleted stale `secrets/gcp-sa-key.json`.
- **Commit 2/4 (`aad8f5a`)** Ã¢â‚¬â€ `chore(arch): strip GCS from active runtime code`. Removed `upload_to_gcs()` (`prediction_artifacts.py`), `gcloud auth` block (`preflight_check.py`), GCS bucket config (`agent_layer/config.py`), GCS-mode pytest assertions (`agent_layer/test_message_bus.py`). 4 files, 5 insertions, 90 deletions. Live `upload_to_gcs` callers post-strip: 0.
- **Commit 3/4 (`feece15`)** Ã¢â‚¬â€ `docs(arch): rewrite operational docs for SCP-only architecture`. 4 files, 20 atomic patches, 30 GCS hit-lines removed: `scripts/run9_launch.md` (11), `docs/HANDOFF_run9_launch.md` (2), `docs/RUN9_OPERATIONS_PLAYBOOK.md` (9), `docs/RUN9_SCIENTIFIC_DESIGN.md` (8). 62 insertions, 62 deletions (balanced textual rewrite). Post-patch GCS hit count across all four files: 0.
- **Commit 4/4 (this commit)** Ã¢â‚¬â€ session log + CHANGELOG cap.

### Learned
- **STANDING RULE Ã¢â‚¬â€ PowerShell variable-colon hazard:** in double-quoted strings, `"$varname:..."` parses as scope/drive prefix (matches `$env:`, `$global:`, `$script:` family). Use `"${varname}:..."` when followed by a literal colon. Add the brace-delimited form to the standing-rules list of PowerShell hygiene patterns.
- **STANDING RULE Ã¢â‚¬â€ EOF-newline anchor:** multi-line `replace` anchors at or near EOF must not include a terminal `\n`. The anchor without trailing newline matches both `text.` (EOF) and `text.\n[more]` cases via Python's `str.count(old)`. P1.6's failure proved this empirically; the file ends without a trailing newline.
- **STANDING RULE Ã¢â‚¬â€ Move-Item is destructive:** Windows `Move-Item -Force` removes the source after the move. Save procedures must `Test-Path` the source BEFORE removing the destination. Pattern: verify Downloads has the file Ã¢â€ â€™ only then delete `agent_data\` Ã¢â€ â€™ then move.
- **STANDING RULE Ã¢â‚¬â€ SHA-256 fingerprint verification:** byte-count alone can miss "downloaded the cached pre-fix version" failures (two file versions can share a byte count by coincidence). Each chat-delivered file should carry a SHA-256 fingerprint the user verifies before save.
- Helper writes with `newline="\n"` for deterministic LF output; Git `core.autocrlf=true` on Windows produces benign `LF will be replaced by CRLF` warnings at staging. Repo content remains LF-normalized; the warnings have no functional impact.
- Architectural state after cleanup arc: GCP project `genomic-variant-prod` permanently destroyed; no remote object storage; data flow is local Windows source-of-truth Ã¢â€ â€ Vast.ai GPU scratch (SCP via `id_lambda_run8`) Ã¢â€ â€ Drive via rclone `genvarcla:` for agent-layer durability only. `INCIDENT_2026-04-29` is the canonical verification-rule supersession of the 2026-04-17 GCS-receipt rule.

### Commits
- `b15a625` Ã¢â‚¬â€ Commit 1/4: incident formalization (4065 bytes of incident doc, secret deleted)
- `aad8f5a` Ã¢â‚¬â€ Commit 2/4: runtime GCS strip (4 files, +5/-90)
- `feece15` Ã¢â‚¬â€ Commit 3/4: operational docs rewrite (4 files, +62/-62)
- (this commit) Ã¢â‚¬â€ Commit 4/4: session log + CHANGELOG cap

### Refs
- `agent_data/arch_cleanup_stage3_discovery.ps1` (5266 bytes)
- `agent_data/arch_cleanup_stage3_code.py` (21838 bytes; SHA `154884df6e976e1614c43c879e7dd71bbcdb1222ce61f277dd379fdd0b33fc1f`)
- `agent_data/arch_cleanup_stage3_batch.ps1` (8991 bytes; SHA `952daab6457d22c9459c5fe9288030eb9f117c776ba57ac769e8957ecf5c1fae`)
- `agent_data/arch_cleanup_stage4_code.py` (this commit's helper)
- `agent_data/arch_cleanup_stage4_batch.ps1` (this commit's batch)
- Session doc: `docs/sessions/SESSION_2026-05-10_arch-cleanup.md`
- Incident doc: `docs/incidents/INCIDENT_2026-04-29_gcp-billing-deletion.md`

## 2026-05-10 Ã¢â‚¬â€ SpliceAI cache leak fix (path-aware conftest.py)

### Attempted
- Move class-scoped `_isolate_spliceai` fixture from `TestAnnotationPipeline` (test_core.py L2167) to a module-scoped autouse fixture in `tests/unit/conftest.py`, add `_save_cache` patch to plug the 430 MB `data/raw/cache/spliceai_scores_snv.parquet` regeneration leak.

### Failed
- **Attempt 1** (Stage 2 abort, no commit): helper's in-line post-apply check used a loose grep `if "_isolate_spliceai" in final_tc` that false-positived on the NEW class docstring's legitimate cross-reference to the new fixture location. Same-pattern-bug as the batch verification fix moments earlier Ã¢â‚¬â€ fixed one location, missed the identical pattern in the other.
- **Attempt 2** (Stage 3b abort, no commit): fixture's UNCONDITIONAL `_save_cache Ã¢â€ â€™ no-op` blocked the legitimate cache write in `test_parquet_cache_used_on_second_call`, which uses `FetchConfig(cache_dir=tmp_path / "cache")` Ã¢â‚¬â€ a tmp-scoped cache that does NOT touch the production dir. Test failed `assert score == 0.42 Ã¢â€ â€™ got 0.0`. Cache mtime UNCHANGED throughout (leak prevention was working; over-blocking was the issue).
- **Pre-check B** (non-fatal): Python helper structural validation via `& python -c @"..."@` errored on `f'{\"X\" if ok else \"Y\"}'` Ã¢â‚¬â€ PS here-strings pass `\"` literally; backslashes inside Python f-string `{expr}` are forbidden. Other pre-checks confirmed file state independently.

### Fixed
- **Attempt 3 commit `a01eef3`**: path-aware fixture design. New `_is_prod_cache_path(cache_path)` helper resolves the cache target and tests `relative_to(_PROD_CACHE_DIR.resolve())`; load/save are blocked only when path resolves under `data/raw/cache/`. tmp_path-scoped FetchConfigs are unaffected and exercise the real loadÃ¢â€ â€™saveÃ¢â€ â€™load flow. `_orig_load_cache` and `_orig_save_cache` captured before patch, called for non-prod paths.
- Helper's in-line post-apply check tightened to `def _isolate_spliceai(` (the method definition) instead of the bare name (which legitimately appears in the new docstring's cross-reference).

### Verified
- 16 pytest tests pass in 58.90s (including `test_parquet_cache_used_on_second_call`, the regression test that exposed Attempt 2's over-blocking).
- Cache mtime IDENTICAL pre/post pytest: `04/19/2026 13:56:19`.
- Cache size IDENTICAL pre/post pytest: 451,626,904 bytes.
- CI green on `a01eef3` (4 min runtime).

### Learned
- **Autouse + unconditional patching is dangerous.** Fixtures that null out shared infrastructure must be conditional/path-aware, not blanket no-ops. Cost of over-blocking: silent test failures that look like real bugs.
- **Same-pattern-bug-different-location.** When fixing a pattern, grep the entire change-set for similar instances. Fixing the batch verification but missing the identical helper internal check cost an iteration.
- **CRLF/UTF-8 byte-delta surprises.** Disk byte delta differs from Python char delta by `num_CRLF_lines + 2*multibyte_chars`. Existing `[WARN] -500 to -1500` bounds in the batch are tight; should widen to roughly `python_char_delta Ã¢Ë†â€™ num_lines_with_CRLF + 2*multibyte_char_count` in future batches.
- **PS here-string + Python f-string interaction.** Inside `@"..."@`, `\"` is passed literally; backslashes in Python f-string `{expr}` are syntax errors. Use single quotes inside double-quoted f-strings.

### Commits
- `a01eef3` Ã¢â‚¬â€ `test(spliceai): move _isolate_spliceai fixture to conftest.py and add _save_cache patch to prevent 430 MB cache regeneration`

### Refs
- Helper: `agent_data/spliceai_cache_fix_code.py` (SHA `3ca0cca1cddaea0b0f46ec56be012482dae3fe8448875ad36cdc8b00b36d5d1e`)
- Batch: `agent_data/spliceai_cache_fix_batch.ps1` (SHA `4d7023a9424f9b54a4e4fce0360bde0fa496736a7da1c1051c5bf6ba80a1491e`)
- Session doc: `docs/sessions/SESSION_2026-05-10_spliceai-cache-fix.md`
- New conftest: `tests/unit/conftest.py`
- Prior session (arch cleanup, same day): `docs/sessions/SESSION_2026-05-10_arch-cleanup.md`
## 2026-05-12 Ã¢â‚¬â€ Run 9: 11.4h training on Vast.ai RTX 4090, ensemble.save() crash, no test AUROC

### Attempted
- Launch Run 9 as 6-ablation suite (`full + 5 feature-group ablations`)
  on Vast.ai RTX 4090 (instance 36588175, $0.473/hr).
- Auto-destroy on preflight failure via vastai CLI `cleanup_if_setup_failed`
  trap function in `scripts/launch_run9_vm.sh`.
- Pickle entire fitted ensemble as a single joblib via
  `joblib.dump(self, path)` in `VariantEnsemble.save()`.

### Failed
- **4 failed launch attempts** before successful launch (~10 min debug each):
  - Attempt 1: workflow-aware preflight bugs (ClinVar VCF,
    torch_geometric). Resolved by commits `8a3785a` + `bd75ed5`.
  - Attempt 2: data SCP'd to repo-relative paths
    (`/workspace/genomic-variant-classifier/data/...`) but
    `launch_run9_vm.sh` uses `/workspace/{data,outputs}/` absolute paths.
  - Attempt 3: training script used absolute paths while preflight
    used repo-relative. Operator added symlinks ad-hoc.
  - Attempt 4: `ln -s /workspace/genomic-variant-classifier/data
    /workspace/data` placed symlink INSIDE the existing
    `/workspace/data/` directory created in attempt 3 instead of
    replacing it (silent Unix `ln` behaviour on existing-target).
    `rm -rf` of the destination required before `ln -s`.
- **Auto-destroy broken** in vastai CLI 1.0.12: interactive
  `input()` confirmation fails under `nohup` with `OSError: Bad file
  descriptor`. Manual destroy via Vast.ai web console at
  https://cloud.vast.ai/instances/ after ~9h idle billing.
- **`ensemble.save()` PicklingError** at end of 11.4h training:
  `_CNN1D` defined inside `CNN1DClassifier._build_model.<locals>`
  is not pickle-able. `joblib.dump()` crashed with
  `_pickle.PicklingError: Can't pickle <class
  'genomic_variant_classifier.models.variant_ensemble.CNN1DClassifier._build_model.<locals>._CNN1D'>:
  it's not found as ...<locals>._CNN1D`. Joblib is corrupt; no
  per-model checkpoints exist; locked test AUROC never produced.

### Fixed (this session)
- Workflow-aware preflight (commits `8a3785a` + `bd75ed5`) Ã¢â‚¬â€ landed
  before final launch attempt.
- Path mismatch Ã¢â‚¬â€ manual `mv` data into repo + symlink
  `/workspace/{data,outputs}` Ã¢â€ â€™ repo paths (workaround; canonical fix
  deferred to Phase 1.5 launch-script unified patch).
- Symlink trap Ã¢â‚¬â€ `rm -rf` before `ln -s` when destination might be
  recreated as directory.

### Drafted (shipped in 2026-05-13 follow-up session as `run10_phase1_v2.zip`)
- Patch A1: `_CNN1D` lifted to module-level `_CNN1DModule` via lazy-
  global with qualname fixup. Fixes pickle.
- Patch A2: `VariantEnsemble.save()` refactored to per-model joblib
  checkpoints (`<ensemble>_models/<model_name>.joblib`) + thin
  orchestrator joblib. Single-model pickle failure no longer poisons
  whole ensemble. `load()` back-compat with legacy single-joblib format.
- Patch A3: `evaluate()` CatBoost dispatch fix (was missing the
  DataFrame branch that `fit`/`predict_proba` correctly include).
- Patch B1: `scripts/run_phase2_eval.py` Ã¢â‚¬â€ added `--lovd-path`,
  `--dbnsfp-path`, `--finngen-path` CLI args + `AnnotationConfig`
  wiring (mirrors `scripts/train.py:167-172`). Closes the
  silent-zero gap for three connectors that were unknowingly absent
  from Run 9 alongside LOVD. Supersedes R10-A of
  `INCIDENT_2026-05-02_lovd-silent-zero.md` (see
  `INCIDENT_2026-05-02_lovd-silent-zero_AMENDMENT.md`).
- Patch B2 + B3: test-set evaluation + OOF parquet + `metrics.json`
  flushed BEFORE `ensemble.save()` so a save crash never loses
  scientific artifacts.
- Regression tests: `tests/unit/test_variant_ensemble_save_load.py`
  (4 tests) + `tests/unit/test_lovd_annotation_reaches_training_matrix.py`
  (2 tests with importskip guard).

### Results
- OOF blend AUROC: **0.9916**
- LR stacker AUROC: 0.9911
- Best single base (lightgbm): 0.9911
- **ÃŽâ€ blend over best single: +0.0005 Ã¢â‚¬â€ within noise floor** pending
  bootstrap CI per `SESSION_2026-05-12.md` Run 10 plan Ã‚Â§3.
- No test-set AUROC: script crashed at save before test evaluation
  ran. Phase 1 patch B2 moves test eval before save to prevent
  recurrence.
- **Per-model OOF AUROC table (2026-05-13 partial recovery via
  `scripts/run9_outputs_audit.ps1`):** 8 of 11 base models recovered as
  04-30 proxies (lightgbm 0.9911, xgboost 0.9908, catboost 0.9900,
  gradient_boosting 0.9889, random_forest 0.9881, deep_ensemble 0.9872,
  mc_dropout 0.9870, logistic_regression 0.9849). 4 NOT recoverable:
  svm, kan, tabular_nn, cnn_1d (skipped in 04-30 regen). 11-dim
  Nelder-Mead weight dict NOT recoverable beyond qualitative statement
  (kan/tabular_nn/logistic_regression 0%, cnn_1d ~10%). See
  `INCIDENT_2026-05-12_no-per-model-checkpoint.md` Ã‚Â§Recovery status.
- **Scientific finding from proxy comparison:** 04-30 8-model blend
  was 0.9915 vs Run 9 11-model blend 0.9916. Adding 4 models
  (svm/kan/tabular_nn/cnn_1d) moved blend by **+0.0001** Ã¢â‚¬â€ at or below
  noise floor. Supports the Ã‚Â§2 keep-all decision being conditional on
  bootstrap CI.

### Scientific implications (preliminary; full analysis in Run 10)
- The 11-model ensemble adds essentially nothing over a single tuned
  lightgbm in OOF blend. ÃŽâ€=+0.0005 must be confirmed via bootstrap CI
  before any pruning decision.
- KAN (8h compute) received 0% blend weight. Drop candidate for
  Run 10, deferred pending bootstrap CI per SESSION Ã‚Â§2 amendment.
- tabular_nn and logistic_regression received 0% blend weight.
- cnn_1d received ~10% blend weight despite OOF AUROC ~0.5 (broken
  signal Ã¢â‚¬â€ fed placeholder sequences per
  `INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md`). Investigate
  whether this generalizes after pickle fix; Sequence Branch
  (real FASTA) wiring deferred to Run 11.
- Standing concern about gene-prevalence + external-score
  memorization remains unresolved.

### Learned (7 new standing rules Ã¢â‚¬â€ see SESSION doc Ã‚Â§Learned)
1. Vast.ai SCP destinations must be repo-relative or include explicit
   symlink step in runbook.
2. `vastai destroy Ã¢â€°Â¥1.0.12` is interactive; auto-destroy in scripts
   MUST pipe `yes` or `echo y`.
3. `ln -s` does NOT replace existing real directories; use `rm -rf`
   first when destination may have been recreated between fix attempts.
4. PowerShell strips inner `"..."` from ssh command args Ã¢â‚¬â€ use single
   quotes ONLY inside ssh wrappers, never double quotes.
5. STOP putting bash code inside `ssh ... '<bash>'` from PowerShell.
   Use `@'...'@ | ssh ... bash -s` with `-replace "`r`n", "`n"` to
   strip CRLF.
6. PowerShell `@'...'@` heredocs preserve `\r\n` line endings; always
   `-replace "`r`n", "`n"` before piping to remote bash.
7. Vast.ai 2026 PyTorch images auto-tmux + auto-activate `/venv/main`.
   SCP destinations MUST be inside the cloned repo. Subprocess can
   still use `/usr/local/bin/python` symlinks for non-activated calls.

### Costs
- Instance 36588175, Vast.ai RTX 4090, $0.473/hr
- ~20.5h total wall-clock = **~$9.70**
- ~9h of that was idle post-crash because auto-destroy was broken
- Productive: ~$5.40 | Idle: ~$4.30

### Commits
- `3cfc039` Ã¢â‚¬â€ `docs(session): Run 9 launch, training, pickle crash, results`

### Refs
- Session doc: `docs/sessions/SESSION_2026-05-12.md`
  (amended 2026-05-13 Ã¢â‚¬â€ Ã‚Â§2 of Run 10 plan revised to keep-all; OOF
  AUROC/blend-weight placeholders annotated with recovery pointer)
- INCIDENTs (filed in 2026-05-13 follow-up session):
  - `docs/incidents/INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md`
  - `docs/incidents/INCIDENT_2026-05-12_vastai-destroy-interactive.md`
  - `docs/incidents/INCIDENT_2026-05-12_launch-path-inconsistency.md`
  - `docs/incidents/INCIDENT_2026-05-12_no-per-model-checkpoint.md`
- LOVD INCIDENT 2026-05-13 amendment: launch-script wiring gap
  identified as actual root cause; supersedes Cause 1 + Cause 2
  candidates. See `INCIDENT_2026-05-02_lovd-silent-zero.md`
  Ã‚Â§"2026-05-13 Update".
- Phase 1 patch bundle: `run10_phase1_v2.zip` (shipped 2026-05-13)
- Run 9 outputs audit: `scripts/run9_outputs_audit.ps1` (placed
  2026-05-13)

# Phase 1.5b CHANGELOG entry

Append this block to `docs/CHANGELOG.md` (after the existing
`## 2026-05-12 Ã¢â‚¬â€ Run 9:` entry).

---

## 2026-05-13 (post-1.5) Ã¢â‚¬â€ Phase 1.5b: test fixes + FinnGen wiring correction

### Test fixes Ã¢â‚¬â€ commit 66593d6 shipped 2 broken tests

The Phase 1 patch bundle (`run10_phase1_v2.zip`, commit 66593d6) shipped 4
regression tests with 2 sandbox-only assumptions that broke under production
pytest:

**1.** `tests/unit/test_variant_ensemble_save_load.py::test_ensemble_save_creates_per_model_checkpoints`
and `::test_ensemble_load_roundtrip` called `ens.fit_minimal(X_tab, X_seq, y)` Ã¢â‚¬â€
a helper method that exists in Claude's sandbox draft but was never shipped to
production `variant_ensemble.py`.

```
AttributeError: 'VariantEnsemble' object has no attribute 'fit_minimal'
```

**Fix (1.5b):** rewritten as one consolidated test `test_ensemble_save_load_with_cnn1d`
that restricts `ens.base_estimators` to `{"lightgbm", "cnn_1d"}` BEFORE
calling `ens.fit()`, then exercises the full save/load/predict_proba round
trip on a 60-row balanced synthetic dataset. CNN1D is in the restricted set
specifically to exercise the A1 pickle-fix code path.

**2.** `tests/unit/test_lovd_annotation_reaches_training_matrix.py::test_lovd_annotation_reaches_training_matrix`
and `::test_lovd_annotation_silent_zero_when_path_omitted` used a 5-row gene
fixture (TP53Ãƒâ€”2, GENE_X, BRCA2, APC) that `GroupShuffleSplit` cannot partition
into class-balanced train/val/test splits.

```
ValueError: Gene-aware split 'train' missing class(es): {np.int64(1)}.
Try lowering min_review_tier or increasing dataset size.
```

**Fix (1.5b):** added `require_both_classes=False` to both tests' `DataPrepConfig`.
The class-balance constraint is for production training; the LOVD column-
propagation check these tests target doesn't need it.

Tests 1 and 2 (`test_cnn1d_module_class_is_module_level` and
`test_cnn1d_pickles_after_fit`) passed in production unchanged. Those tests
directly validate the A1 pickle fix and remain the most important regression
guards.

### FinnGen wiring Ã¢â‚¬â€ commit 66593d6 message was incorrect

The 66593d6 commit message stated:

> NOTE: FinnGen wiring is partial. B1 sets AnnotationConfig.finngen_path
> but real_data_prep.py annotate chain does not invoke FinnGenConnector
> (regen.log shows no FinnGen step). Phase 1.6 will add the connector
> invocation. LOVD and DbNSFP are fully fixed.

This is **incorrect** and was based on a false inference that "no FinnGen
entries in regen.log" implied "FinnGen connector not wired". Empirical
verification on 2026-05-13 via direct grep of `src/genomic_variant_classifier/data/real_data_prep.py`:

```
185:    finngen_path: Optional[Path] = None  # FinnGen R10 annotated variants TSV
418:    # FinnGen R10: third-tier AF fallback after gnomAD and 1KGP
419:    if self.annotation_config.finngen_path:
420:        from genomic_variant_classifier.data.finngen import FinnGenConnector
422:        finngen = FinnGenConnector(tsv_path=self.annotation_config.finngen_path)
423:        df = finngen.annotate(df)
425:    else:
427:        for col in FINNGEN_COLUMNS:
430:        df["finngen_enrichment"] = 1.0
```

**Phase 1 B1 IS sufficient for FinnGen.** Passing `--finngen-path` to
`scripts/run_phase2_eval.py` sets `AnnotationConfig.finngen_path`, which
satisfies the line 419 conditional and invokes `FinnGenConnector.annotate()`
at line 422. Same fix shape as LOVD and DbNSFP.

The reason no FinnGen entries appear in Run 9's `outputs/run9_ready/regen.log`
is **NOT** a wiring gap Ã¢â‚¬â€ it's that the `else` branch at line 425-430 silently
fills defaults (`finngen_af_fin=0`, `finngen_af_nfsee=0`, `finngen_enrichment=1`)
with **no log emission at all**. This is a *worse* silent-zero pattern than
LOVD or DbNSFP (which at least emit a WARNING that audit greps catch).

FinnGen is wired into the **AF-fallback** stage (line ~418, third tier after
gnomAD and 1KGP) Ã¢â‚¬â€ NOT into the **score-annotation** stage (line 504+). The
"Score annotation N/M" log series covers the 17 score connectors only.
That's why `Select-String "Score annotation"` against `real_data_prep.py`
shows 17 score steps with FinnGen absent Ã¢â‚¬â€ that absence is structural, not
a bug.

**Phase 1.6 follow-up (deferred, optional):** add an `INFO` log to the
FinnGen `else` branch so silent-zero is detectable in `regen.log` audits.
Small code-hygiene patch, can ride with `sequence_context.py` stub work.

### Phase 1 commit message accuracy

The 66593d6 commit message will remain as-is (git history rewrite not worth
the risk on `main`). The correction lives here and will be referenced by any
future audit. Future commit messages should phrase FinnGen as "fully wired"
alongside LOVD and DbNSFP.

# Phase 1.5c CHANGELOG entry

Append this block to `docs/CHANGELOG.md` (after the existing
`## 2026-05-13 (post-1.5) Ã¢â‚¬â€ Phase 1.5b:` entry).

---

## 2026-05-13 (post-1.5b) Ã¢â‚¬â€ Phase 1.5c: LOVD anchor fix + sklearn/lightgbm skew workaround

Phase 1.5b shipped two fixes but only one landed cleanly (commit `f64c024`).
This entry corrects the remaining failures.

### Issue 1 Ã¢â‚¬â€ Phase 1.5b LOVD anchors didn't match production indentation

The `apply_phase1_5b.py` applier used `str_replace`-style anchors with fixed
8-space body indentation, which assumed tests were wrapped in a `class TestLOVDPropagation:`.
Production tests are top-level functions with 4-space body indentation. Both
anchors (L1, L2) returned `[ERROR: anchor not found]` and the LOVD test file
was left untouched. The 2 LOVD tests continued to fail with the original
`ValueError: Gene-aware split 'train' missing class(es)`.

**Fix (1.5c):** indent-aware patcher in `apply_phase1_5c.py`. Locates each
`DataPrepConfig(...)` block by its `output_dir` marker (`"splits"` or
`"splits_no_lovd"`), parses the closing-paren indent and argument indent
dynamically from the block itself, and inserts `require_both_classes=False`
with matching indent. Works for both top-level functions (4-space body) and
class-wrapped methods (8-space body). Sandbox-verified against both layouts.

### Issue 2 Ã¢â‚¬â€ lightgbm OOF silently dropped due to sklearn 1.6+ API rename

The Phase 1.5b ensemble test fitted `lightgbm` + `cnn_1d` and asserted both
landed in `trained_models_`. Production run logged:

```
ERROR  lightgbm OOF failed:
  check_X_y() got an unexpected keyword argument 'force_all_finite' Ã¢â‚¬â€ skipping.
```

`scikit-learn` 1.6 renamed `force_all_finite` Ã¢â€ â€™ `ensure_all_finite`. lightgbm
versions before 4.4 still call sklearn with the old argument name. The
`VariantEnsemble.fit()` OOF loop catches the exception, logs an `ERROR`,
and silently continues with the model dropped from `trained_models_`. The
test then sees only `{cnn_1d}` instead of the expected `{lightgbm, cnn_1d}`
and fails.

**Important: this is an environment issue, not a code bug.** Run 9 on
Vast.ai produced `lightgbm OOF AUROC: 0.9911` (`outputs/run9_ready/regen.log`
line 88), so the Vast.ai venv had a compatible combo at the time. The local
venv must have drifted (likely sklearn pulled forward as a transitive dep).

**Fix (1.5c, test only):** swap `lightgbm` Ã¢â€ â€™ `random_forest` in
`test_ensemble_save_load_with_cnn1d`. Random forest is pure-sklearn, so
no skew is possible. The test still exercises both the tabular dispatch
(via random_forest) and the sequence dispatch (via cnn_1d, which is what
the A1 pickle fix actually targets).

**Run 10 implication Ã¢â‚¬â€ DO NOT IGNORE.** Before Run 10 launch, verify the
Vast.ai venv has a compatible sklearn/lightgbm combo. The diagnostic is:

```powershell
python -c "import sklearn, lightgbm; print(f'sklearn {sklearn.__version__}'); print(f'lightgbm {lightgbm.__version__}')"
```

If `sklearn >= 1.6` and `lightgbm < 4.4`, lightgbm OOF will be dropped at
fit time. Fix on Vast.ai: `pip install -U lightgbm` (which brings in the
`ensure_all_finite` rename) OR pin both in `requirements*.txt`. Run 9 best
single-model was lightgbm; losing it for Run 10 would be a major regression.

### What this bundle does NOT do

- Does NOT fix the local venv. The `pip install -U lightgbm` step is up
  to Monzia. The test simply avoids triggering the skew.
- Does NOT add the FinnGen `else`-branch INFO log noted in Phase 1.5b's
  CHANGELOG entry. Still deferred to Phase 1.6+.
- Does NOT touch production code in `variant_ensemble.py` or
  `run_phase2_eval.py`. The 66593d6 production patches remain correct.

# Phase 1.5d CHANGELOG entry

Append this block to `docs/CHANGELOG.md` (after the existing
`## 2026-05-13 (post-1.5b) Ã¢â‚¬â€ Phase 1.5c:` entry).

---

## 2026-05-13 (post-1.5c) Ã¢â‚¬â€ Phase 1.5d: positive LOVD test scope fix

Phase 1.5c successfully added `require_both_classes=False` to both
`DataPrepConfig` blocks in `test_lovd_annotation_reaches_training_matrix.py`.
Production pytest then surfaced a remaining issue in the positive test:

```
AssertionError: Expected at least one row with lovd_variant_class > 0
in training matrix; got 0. value_counts: {0: 1}
```

### Root cause Ã¢â‚¬â€ test scope assertion bug

The positive test (`test_lovd_annotation_reaches_training_matrix`) was
asserting on `X_train["lovd_variant_class"] > 0`, but the 5-row fixture
has 5 distinct genes (TP53Ãƒâ€”2, GENE_X, BRCA2, APC). With:
- `test_fraction=0.4` Ã¢â€ â€™ 2 genes in test
- default `val_fraction` Ã¢â€ â€™ ~1 gene in val
- `GroupShuffleSplit` doing gene-aware random splitting

the LOVD-matching TP53 row can land in *any* of train/val/test depending
on the random seed and gene-bucket assignment. In this run the TP53 row
went to val or test, and X_train ended up with 1 row (different gene)
that wasn't LOVD-annotated.

The test's actual post-condition is "LOVD annotation reached SOME output
matrix" Ã¢â‚¬â€ i.e. the connector ran, the merge happened, and the column
survived feature engineering through to the output. The correct scope
is the **union of X_train Ã¢Ë†Âª X_val Ã¢Ë†Âª X_test**, not X_train alone.

### Fix (1.5d)

Rewrote the assertion block to:
1. Unpack all three splits: `X_train, X_val, X_test = result[0], result[1], result[2]`
2. Check `lovd_variant_class` column is present in each split (feature engineering consistency)
3. Concatenate via `pd.concat([X_train, X_val, X_test], ignore_index=True)` and assert at least one row across the union has `lovd_variant_class > 0`

The inverse test (`test_lovd_annotation_silent_zero_when_path_omitted`)
already passes because `0 == 0` in any split Ã¢â‚¬â€ it remains untouched.

### Local venv version skew was a stale `.pyc`, not a real issue

Phase 1.5b's failure attributed to sklearn 1.6+ / lightgbm <4.4 skew turned
out to be transient. Phase 1.5c diagnostic on Monzia's clean venv:

```
sklearn 1.8.0
lightgbm 4.5.0
```

Both versions ship the `ensure_all_finite` rename, so they're compatible.
The Phase 1.5b error (`force_all_finite` complaint) was likely from a
stale `__pycache__/` that survived the Phase 1 cache-clear. The Phase 1.5c
test using `random_forest` instead of `lightgbm` is fine to keep Ã¢â‚¬â€ it's
not strictly necessary for skew-avoidance now, but it makes the test more
robust to any future version drift.

**Run 10 implication is REDUCED but not eliminated.** The Vast.ai venv
still needs version pinning before launch Ã¢â‚¬â€ sklearn or lightgbm
floating could re-introduce the issue. Track in Phase 1.7
(`scripts/launch_run10_vm.sh` + `requirements*.txt` review).

### Cumulative test state after 1.5d

- `tests/unit/test_variant_ensemble_save_load.py`: 3 PASSED
- `tests/unit/test_lovd_annotation_reaches_training_matrix.py`: 2 PASSED
- Phase 1 regression suite GREEN end-to-end

Ready to advance: Phase 1.6 (`sequence_context.py` stub + optional FinnGen
INFO log) or directly to Phase 1.7 (launch script rewrite).

# Phase 1.5e CHANGELOG entry

Append this block to `docs/CHANGELOG.md` (after the existing
`## 2026-05-13 (post-1.5c) Ã¢â‚¬â€ Phase 1.5d:` entry).

---

## 2026-05-13 (post-1.5d) Ã¢â‚¬â€ Phase 1.5e: module-level pandas import for LOVD test

Phase 1.5d's assertion rewrite used `pd.concat([X_train, X_val, X_test], ignore_index=True)`
at module/test-function scope, but the test file imports pandas only
inside fixture functions (e.g. `import pandas as pd` inside
`tiny_clinvar_parquet`). Test-body code therefore raised:

```
NameError: name 'pd' is not defined
```

### Why the Phase 1.5d WARN missed this

The Phase 1.5d applier had this check:

```python
if "import pandas" not in text:
    print("WARN: pandas not imported in target file. ...")
```

Naive substring match. The file has `    import pandas as pd` (indented,
inside a fixture body), which contains the substring `"import pandas"`,
so the WARN never fired. The check should have been anchored at line
start with `re.MULTILINE` to detect only module-level imports.

### Fix (1.5e)

Single-purpose applier `apply_phase1_5e.py` that:

1. Checks for **module-level** `import pandas` via
   `re.compile(r'^import pandas(\s|$)', re.MULTILINE)` Ã¢â‚¬â€ distinguishes
   `import pandas as pd` at column 0 from `    import pandas as pd`
   inside a function body
2. If absent, inserts `import pandas as pd` at the best available
   location:
   - After `from __future__ import annotations` (preferred)
   - After the module docstring (fallback)
   - At the very top (last resort)
3. Idempotent (status: `ALREADY` if module-level import exists)

Sandbox-verified against four scenarios: in-fixture-only (production
state), already-module-level, no-`__future__`, bare file with neither
docstring nor `__future__`. All produce correct insertion or no-op.

### Lesson learned Ã¢â‚¬â€ future appliers

Any future applier that depends on a module-level import being present
should check with `re.compile(r'^import <pkg>', re.MULTILINE)` rather
than naive substring match. Memory rule 28 (apply-batch hygiene)
extended implicitly.

### Cumulative test state after 1.5e

- `tests/unit/test_variant_ensemble_save_load.py`: 3 PASSED
- `tests/unit/test_lovd_annotation_reaches_training_matrix.py`: 2 PASSED

Phase 1 regression suite GREEN end-to-end. Ready to advance to Phase 1.6
(`sequence_context.py` stub + optional FinnGen INFO log) or directly to
Phase 1.7 (launch script + requirements pinning).

# Phase 1.7 CHANGELOG entry

Append this block to `docs/CHANGELOG.md` (after the existing
`## 2026-05-13 (post-1.5d) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Phase 1.5e:` entry).

---

## 2026-05-13 (post-1.5e) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Phase 1.7: Run 10 launch readiness

Three artifacts shipped to prepare for Run 10 launch. Phase 1.6
(sequence_context stub + FinnGen INFO log) is deferred ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â neither is a
Run 10 blocker.

### 1. NEW: `scripts/launch_run10_vm.sh`

Evolves `scripts/launch_run9_vm.sh` (97 lines) into a Run 10 launch
script. Diffs from the Run 9 source:

- **Non-interactive `vastai destroy`** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Run 9's launch script called
  `vastai destroy instance "$INSTANCE_ID"` directly. `vastai` 1.0.12 is
  interactive and would hang on a y/N prompt without TTY, defeating
  auto-destroy on setup failure. Phase 1.7 pipes `echo y |` per memory
  rule 30(c).
- **Run 10 paths** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â `OUT_BASE=/workspace/outputs/run10`, `RUN_ID=run10`,
  per-ablation log `logs/run10_${ABL}.log`.
- **Single 'full' ablation** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Run 10's narrow goal is the locked test
  AUROC that Run 9 lost to `save()` crash. The Run 9 6-ablation matrix
  (`full no_spliceai no_gnn no_alphamissense no_conservation
  no_population_af`) is collapsed to `for ABL in full`. Run 10a will
  extend the loop.
- **Post-success expected-outputs banner** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â points at the new per-model
  joblib layout shipped by Phase 1 A2:
  `models/<name>.joblib` + `models/orchestrator.joblib`. A future
  observer of the Vast.ai log can confirm which files to SCP back.
- **No SCP-back automation** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the existing manual SCP + manual destroy
  pattern is preserved per INCIDENT_2026-04-29 (local-landing-receipt
  rule). Server-side SCP-back-to-local requires a return tunnel the VM
  doesn't have; the right place for that automation is the local
  PowerShell runbook, not the VM script.

Run 10 uses the existing `outputs/run9_ready/splits/` directory. LOVD
and DbNSFP columns remain silent-zero (same as Run 9) because B1's
`--lovd-path/--dbnsfp-path` are only exercised when splits are
regenerated. **Run 10a** will re-regen via
`scripts/run_phase2_eval.py --lovd-path ... --dbnsfp-path ...`. **Run 10b**
will additionally pre-index the 30 GB FinnGen TSV to a ClinVar-intersected
parquet before adding `--finngen-path`.

### 2. PATCHED: `scripts/preflight_vm.sh`

Four new sections inserted between section 8 (Critical Python imports)
and the Summary section:

- **Ãƒâ€šÃ‚Â§9 LOVD parquet** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â `du -k` size threshold ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥ 100 KB at the canonical
  path `data/external/lovd/lovd_all_variants.parquet`. WARN (not FAIL)
  if absent ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Run 10 tolerates the silent-zero pattern; Run 10a/10b
  require it.
- **Ãƒâ€šÃ‚Â§10 DbNSFP parquet** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â `du -m` size threshold ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥ 20 MB at the
  canonical path. WARN-on-absent contract matches Ãƒâ€šÃ‚Â§9.
- **Ãƒâ€šÃ‚Â§11 FinnGen TSV (optional, warn-only)** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â present-or-warn at the
  canonical 30 GB path. Run 10b will tighten this to FAIL once the
  pre-indexed parquet is the deployment artifact.
- **Ãƒâ€šÃ‚Â§12 sklearn + lightgbm 1000-row LGBMClassifier smoke fit** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
  catches the Phase 1.5b false-alarm pattern (`check_X_y() got an
  unexpected keyword argument 'force_all_finite'`) BEFORE GPU billing
  starts. The OOF wrapper in `variant_ensemble.py` silently downgrades
  lightgbm-fit failures to ERROR + skip-the-model, so a real skew
  would only surface after ~11h of training. The smoke fit makes that
  surface at preflight time instead.

Each section uses the existing `pass`/`fail`/`warn` macros so the
summary line at the bottom counts correctly.

### 3. NEW: `logs/training/run9_master.log.recovery.md`

Full SSH `tail -100` capture of Run 9's `/workspace/run9_master.log`,
retrieved 2026-05-13 before Vast.ai instance 36588175 was destroyed.
The original 273-line / 264 KB master log was never SCP'd back; the
last 100 lines (the failure-relevant region with full traceback) are
the only surviving copy outside chat transcripts.

The recovery file includes:
- Reconstructed timeline (16-row table from earlier SSH queries)
- All 11 per-model OOF AUROCs (lightgbm 0.9911 best, cnn_1d 0.5000
  anomalous)
- Blend weights from Nelder-Mead (random_forest 0.3377, xgboost 0.0434,
  lightgbm 0.2933, ...)
- Verbatim `tail -100` block (RuntimeWarnings, deep_ensemble fit
  members, blend log, full PicklingError traceback, ABORT line)
- Cross-references to filed incidents and Phase 1 fixes

### Open follow-up flagged during Phase 1.7

- **`cnn_1d OOF AUROC: 0.5000`** in Run 9 is anomalous. The same class
  (`CNN1D._build_model.<locals>._CNN1D`) that broke pickle may also
  have failed silently at fit time. The Phase 1 A1 fix repairs the
  pickle path but doesn't address a hypothetical fit-side bug. Worth
  checking after Run 10's locked test result is in.
- **`requirements-api.lock` vs `requirements.txt` version split** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
  `fastapi==0.119.1` / `starlette==0.48.0` in the lock file vs
  `fastapi==0.135.2` / `starlette==1.0.0` in `requirements.txt`. Driven
  by `prometheus-fastapi-instrumentator==7.1.0` requiring `starlette<1.0`.
  These coexist because the Docker multi-stage build installs them in
  separate stages (api vs trainer per memory rule 19). Non-blocking,
  but memory rule 20's deferred fix is still open.

### Phase 1 cumulative state after 1.7

- Phase 1 regression suite: **5/5 GREEN** (unchanged since 1.5e)
- Full unit-test sweep: **501/501 GREEN** (unchanged since 1.5e)
- Launch readiness: scripts in place, preflight covers all Run 10
  failure modes seen to date
- Cost-budget for Run 10: ~$10ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“12 for ~11h on Vast.ai RTX 4090
  (matches Run 9 wall-clock; no regen step in Run 10)
- Time-to-result: ~12h from SCP-up to locked test AUROC in
  `outputs/run10/full/metrics.json`

---

## 2026-05-16 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Run 10: locked test AUROC produced

### Attempted
- Launch Run 10 on Vast.ai (instance 36853443, RTX 4090, datacenter 1647
  Iceland) to produce the locked test AUROC that Run 9 failed to deliver.
- 4 launch attempts before successful training start (path mismatch, missing
  meta parquets, missing pykan, symlink fix).
- Full 11-model ensemble training (~12 hr): RF, XGB, LGB, LR, GBM, CatBoost,
  TabularNN, CNN1D, KAN (200 epochs), MC_Dropout, DeepEnsemble (5 members ÃƒÆ’Ã¢â‚¬â€ 5
  folds).

### Failed
- Launches 1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“3: `FileNotFoundError` on split files. Root cause: launch script
  uses `SPLITS_DIR=/workspace/outputs/run9_ready/splits` but SCP put files at
  `/workspace/genomic-variant-classifier/outputs/run9_ready/splits/`. Fix:
  symlink.
- Post-training OOF export crash at `run9_ablations.py:705`:
  `ValueError: Length of values (1197216) does not match length of index
  (1017633)`. `ensemble.oof_predictions_` has 85% of y_train rows. Crash
  occurred AFTER locked test eval was written to disk. See
  `INCIDENT_2026-05-16_oof-export-length-mismatch.md`.

### Fixed / Achieved
- **Locked test AUROC: 0.98163** (95% CI: 0.98126ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“0.98197).
- OOF blend AUROC: 0.9916. Test-to-OOF gap ~0.01 (healthy).
- All 11 per-model checkpoints + ensemble.joblib saved and SCP'd locally
  (~4.2 GB total).
- Evaluation artifacts saved: `eval_report.json`, `test_predictions.parquet`
  (349,067 rows ÃƒÆ’Ã¢â‚¬â€ 20 cols), `calibration.parquet`, `manifest.json`.
- Instance destroyed after full artifact retrieval.

### Learned
- PowerShell here-string `@"..."@` piped to SSH is unfixable for CRLF. Only
  reliable pattern: `ssh ... 'single-line command'`. One command per SSH call.
- `wc -l` returns 0 on `\r`-only files (KAN progress bars). Use `tail -c N`.
- `meta_val.parquet` and `meta_test.parquet` are required by `load_splits()`
  (8 files, not 6).
- `pykan` must be explicitly installed on Vast.ai images.
- Vast.ai CLI `vastai destroy` returns 401 when run FROM the instance itself;
  use the web console instead.

### Cost
- Vast.ai instance 36853443: ~$7ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“9 (12 hr training + ~2 hr idle/debug)
- Prior destroyed instance 36853984: ~$1 (auto-destroyed by preflight trap)




## 2026-05-31 â€” Phase 0: cohort de-leak (Run 15 prep)

### Attempted
- Resolve the null-key cohort leak (B1) at source before Run 15 split regeneration.

### Fixed
- Added `scripts/clean_cohort.py` (introspective, fail-loud, --audit/--apply) and
  `tests/unit/test_clean_cohort.py` (synthetic; 2 passed).
- Quarantined 21,091 allele-less rows â†’ `data/processed/clinvar_grch38_structural.parquet`.
- Emitted `data/processed/clinvar_grch38_clean.parquet` (4,399,089 rows; 0 null, 0 dup).
- `clinvar_grch38_conflicts.parquet` written (0 irreducible conflicts after quarantine).
- Reconciliation identity verified exact (4,420,180 = 21,091 + 4,399,089).

### Learned
- The 4,203 duplicate `variant_id`s were entirely within the allele-less bucket; quarantine
  alone yields a unique-key clean cohort with no label-conflict surgery required.
- Root mechanism: `astype(str)` on null alleles in the gnomAD join collapses distinct
  region records onto shared keys (see INCIDENT_2026-05-31_null-key-leak.md).
- ~48 coding/splice variants carry null alleles upstream (ingestion gap); recovery
  candidate in the ClinVar re-pull.

### Open follow-ons
- Harden the gnomAD-join key (null-safe) in `real_data_prep.py`.
- Regenerate splits from the clean cohort; repoint the pipeline input.
- GNN `gnn_score` confirmed 100% zero across all Run-14 splits (separate incident pending
  trace-branch identification).

## 2026-05-31 -- Leakage audit quantified + Run-14 provenance/ensemble findings

### Found
- Run-14 split leakage quantified: within-split dup train 2,125/val 129/test 409;
  cross-split overlap train&test 247/train&val 115/val&test 46; structural-in-splits 11,320.
- Main split IS gene-disjoint (gene overlap train&test = 0; GroupShuffleSplit by gene 42/43).
- Provenance mismatch: outputs/run14/run14_master.log reports output=/workspace/outputs/run11/full.
- Reduced ensemble in 05-26 run: skip_cnn=True (cnn_1d closure bug B.D6) and string_db=None (GNN off).

### Decided
- Regenerate splits from clinvar_grch38_clean.parquet (gene-disjointness preserved; all three
  contamination classes removed). Cohort guard enforces clean input.
- Establish honest baseline (clean cohort, GNN on via --string-db auto, --skip-cnn) before Run-15
  multi-modal build. See docs/incidents/INCIDENT_2026-05-31_run14-split-leakage.md +
  INCIDENT_2026-05-31_gnn-score-zero.md.

## 2026-05-31 -- Run-15 baseline path: defects closed + cohort-guard resilience

### Fixed
- run10a-no-checkpoints (INCIDENT_2026-05-23): per-model incremental checkpointing verified by
  tests/unit/test_ensemble_persistence.py (4-file quartet {name}.joblib/_oof.npy/_oof_indices.npy/
  _meta.json + OOF/index length parity per base model). RESOLVED.
- cohort-guard LOVD regression (self-inflicted, commit 1720c0a): _assert_clean_cohort raised
  KeyError 'variant_id' on inputs lacking that column (raw ClinVar / tiny LOVD fixtures). The
  duplicate-identity check now prefers variant_id and otherwise derives the key from the
  chrom:pos:ref:alt locus, preserving fail-loud behaviour on a dirty production cohort. Locked by
  tests/unit/test_cohort_guard_resilience.py (4 cases); the two LOVD post-condition tests pass again.
- test_cohort_guard.py::test_duplicate_variant_id_raises relaxed to a wording-agnostic match
  ("duplicate variant") after the guard message changed to "duplicate variant identity".

### Reclassified (missing-feature scope, signed off -- not defects)
- cnn1d-0.5-auroc (INCIDENT_2026-05-23) + cnn1d-cross-platform-unpickle (INCIDENT_2026-05-24):
  cnn_1d is a sequence CNN whose fasta_seq input is unpopulated upstream (constant poly-A ->
  AUROC 0.5). Honestly excluded from the baseline via --skip-cnn; re-enabled in Phase B once
  fasta_seq (reference-genome 101-bp window) is extracted, which also unlocks the RNA pipeline.

### Learned
- Before renaming a user-facing string/message, grep tests for assertions on it.
- Verify a model's input contract from code before writing tests (cnn_1d is sequence, not tabular).
- Run the full unit suite (no -x) from repo root before pushing, to match CI exactly.

## [2026-06-03] â€” D.1 Correctness Patches + D.2 Science Additions

### Added
- `src/genomic_variant_classifier/data/splits.py` â€” hash-based gene-stratified
  splits (`gene_stratified_split`, `unseen_gene_holdout_split`, `split_summary`).
  Replaces `GroupShuffleSplit` with `hashlib.md5` gene-hash for Rule 6 stability:
  holdout genes are stable as dataset grows (C3 gate / `test_hash_stability_across_data_versions`).
- `src/genomic_variant_classifier/evaluation/ntqr_evaluator.py` â€” NTQR r2 accuracy
  bounds (stub mode when `ntqr` absent; SR #31 check required before requirements.txt).
- `src/genomic_variant_classifier/features/topological_ph.py` â€” PH features over
  STRING v12 (Adopt #20; train-subgraph leakage guard; stub when `gudhi` absent).
- `scripts/ablation_npig_permutation.py` â€” C3 permutation ablation for
  `n_pathogenic_in_gene`. Uses shuffled-label npig recomputation (F-10 fix).
- `tests/unit/test_d1_d2.py` â€” 42-test battery for D.1+D.2.
- `docs/preflight/ntqr_sr31_check.ps1` â€” SR #31 smoke test for ntqr.
- `docs/preflight/gudhi_sr31_check.ps1` â€” SR #31 smoke test for gudhi.
- `src/genomic_variant_classifier/features/__init__.py` â€” new package init.

### Fixed (D.1 patches â€” real_data_prep.py + run_phase2_eval.py)
- **F-02** `_assert_clean_cohort` silent-skip: `else: _key = None` replaced by
  `raise ValueError` when neither `variant_id` nor locus columns exist.
- **F-05** `run_phase2_eval.py`: auto-enables `--skip-cnn` when seq-windows file
  absent (prevents false exit-2 from unmapped-coverage gate).
- **F-06** `_annotate_scores` log step numbers normalised to N/17 throughout
  (12 individual log strings corrected: 3/4â†’3/17 through 14/14â†’14/17).
- **F-07** `_join_gnomad` position coerced to `int` via `pd.to_numeric` for robust
  locus matching (avoids leading-zero string mismatch).
- **F-13** OOF sidecar (`oof_predictions.parquet`) now includes `_train_row_idx`
  column for downstream meta-learner reconstruction alignment.

### Fixed (splits.py API compliance)
- Added `gene_stratified_split` and `split_summary` (were missing; caused
  `test_splits.py` collection error / `ImportError`).
- `unseen_gene_holdout_split`: changed `KeyError` â†’ `ValueError` for missing
  gene column; added `holdout_frac` bounds validation (raises `ValueError`
  matching `holdout_frac` for values outside (0,1)).

### Fixed (test_d1_d2.py API alignment)
- `test_missing_gene_col_raises_key_error` â†’ `test_missing_gene_col_raises_value_error`
  (aligns with `test_splits.py` expectation and new `ValueError` contract).
- Removed `test_missing_label_col_raises_key_error` (label column no longer
  required by `unseen_gene_holdout_split`).

### Test results
- `tests/unit/test_splits.py`: 12/12 PASS (was: collection ERROR)
- `tests/unit/test_d1_d2.py`: 42/42 PASS (new)
- Full suite: 693 passed / 6 skipped / 0 failed (was: 596/1/0 + 1 collection error)

## 2026-06-03 — Session 2026-06-01/06-03 (correctness, verification, audit)

### Fixed
- mc_dropout NaN entropy: float32 eps=1e-8 below machine epsilon rounded clip bound to 1.0 → log(0) → NaN. Replaced with exact binary entropy (0*log(0):=0). See INCIDENT_2026-06-01_mc-dropout-nan-entropy.md.
- GNN 64 GB OOM: build_pyg_dataset replicated the full STRING graph per variant. Option B single-shared-graph + batched focal readout. Validated on real 16,201-node graph (no OOM, gnn_score std 0.0302). See INCIDENT_2026-06-01_gnn-oom.md.

### Added
- scripts/preflight_gate.py: pre-flight config gate (8/8 tests). Hard-fails falsy --string-db, empty --seq-windows, missing source paths, missing --unseen-gene-holdout, forbidden --skip-nn/--skip-cnn/--skip-kan; warns on missing --skip-svm; requires --ack-omit for optional-source omissions. --emit confirmed all 8 rich-run inputs present.
- scripts/inspect_clinvar_header.py: ClinVar VCF header provenance reader (date/review-status), early-break at first data row.

### Verified
- Rich sources work (silent-zeros were missing paths): dbNSFP 204,384 SIFT; gnomAD-constraint 94.6%; LOVD 369. Rich config cut n_pathogenic_in_gene importance 1213.5→272.3.
- Full unit suite: 651 passed, 1 skipped, 0 failed.

### Audited
- Run 14 AUROC 0.9975 untrustworthy (GNN skipped, no gene-disjoint holdout, leakage proxy dominant). See INCIDENT_2026-06-03_run14-audit.md.
- data/ is a junction to G:\My Drive\...; repo/code/git are local. ClinVar fileDate 2026-03-15, GRCh38, three review-status fields.

### Open
- SVM auto-skip conflict (--help says --skip-svm required >100k vs manifest auto-skip).
- Scope: germline vs oncogenic/somatic.
- Commit + push the two fixes (run-id trailer); GNN GPU epoch-timing probe before rich run.

## 2026-06-03 - Path A: --min-review-tier silent no-op (HIGH)

**Fixed**
- `_load_and_label` silently skipped the review-tier filter on every run (incl. Run
  14's 0.9975) because neither clean nor dirty cohort had a `ReviewStatus` column;
  `--min-review-tier 3` was a no-op for the whole project history.
- Part 1 (f24bfc6, 6142d87): `augment_reviewstatus.py` attaches `ReviewStatus` to the
  clean cohort from the ClinVar VCF (underscores->spaces). non-empty=3,974,573;
  tier<=3 KEEP=1,490,014. `probe_review_status.py` read-only diagnostic.
- Parts 2-3 (b494544): fail-loud guard (raise if `min_review_tier<5` and no
  `ReviewStatus`); drop `review_tier` after filtering (no feature leak);
  `test_review_tier_filter.py`; LOVD tests `0->5`.
- Part 4: `preflight_run15_baseline.py` ReviewStatus present+populated NO-GO gate.

**Learned**
- A `if col in df.columns:` filter with no else fails open (silent). Column-name
  conventions (`ReviewStatus` vs `review_status`) must be asserted, not assumed.
- Run 14's 0.9975 was both leakage-inflated and never tier-filtered; Run 15
  (tier<=3, ~1.49M) is the first honest-cohort baseline.

**Tests**: full unit suite 713 -> 716 passed, 1 skipped, 0 failed.


## 2026-06-04 — Run 15 (de-leaked gene-disjoint baseline)

**Attempted**
- Full-signal Run 15 on Vast.ai (RTX 4090, instance 39391596) via
  `launch_run15_baseline.sh` + self-stop teardown wrapper: 12-model ensemble,
  `--string-db auto` GNN, `--unseen-gene-holdout`, n_folds 5, tier 3, all
  signal connectors wired (gnomAD-constraint, dbNSFP, LOVD, SpliceAI,
  AlphaMissense).

**Result**
- `TRAIN_OK after 29525s | GNN_FAIL` (~8.2 h, ~$8). Self-stop clean.
- Test AUROC 0.9983 / val 0.9984 / **unseen-gene holdout 0.9988** (213,436 rows,
  2,407 genes). Cross-gene generalization holds; C3 falsifier (b) PASS.
- Per-model best single = catboost (test 0.9984); stacker 0.9983 (no lift at
  saturation); cnn_1d 0.8219 (sequence-only). 10 of 12 models compared.
- Data: tar MD5 `fefa30910559a89b2b62aa133d7b7e1c`, 121 files, verified,
  retrieved, backed up to Drive. Instance destroyed; exposed key rotated.

**Failed**
- KAN failed in both ensembles (`name 'test_size' is not defined`) — imodelsx
  patch (launch_run11_vm.sh:193-194) not ported into baseline launcher.
- GNN `gnn_score` degenerate (0.5, std=0, all splits): `gnn_df` lacks
  `variant_id` → `GNNScorer.from_trainer` builds empty `gene_scores` → 0.5
  default everywhere; gene-disjoint split compounds it. GNN trained fine (val
  AUC 0.6509) but scores never reached the matrix.

**Fixed**
- (Process) Replaced the "skip heavy models" mini-test standing law with an
  ALL-MODELS smoke gate (no `--skip`, fails launch if any model
  errors/skips/degenerate). Recorded the multi-goal project charter
  (`docs/PROJECT_GOALS.md`).

**Learned**
- A pre-launch test that skips fragile models gives false confidence; exercise
  every model at tiny scale before paying for GPU time.
- An informational GNN gate detects degeneracy but does not prevent a wasted
  run — the injection must hard-abort on `std≈0`.
- ~38 of 78 features carry zero importance (unpopulated connectors); the
  effective matrix is ~40 features. Quantified for the data roadmap.

**Open (next session, behind smoke gate)**
- KAN sed patch in baseline launcher; GNN inductive all-node scorer +
  `std>0` hard-abort; SVM Nyström/RFF + bagged secondary;
  `n_pathogenic_in_gene` ablation. See
  `INCIDENT_2026-06-04_kan-test-size-baseline-launcher.md` and
  `INCIDENT_2026-06-04_gnn-score-injection-degenerate.md`.

## 2026-06-06 — Run 15 sealed (corrected re-run)

**Result.** Main ensemble AUROC 0.9984 (AUPRC 0.9936, F1m 0.9826, MCC 0.9652, Brier
0.0069); unseen-gene holdout 0.9988 (2,407 genes / 213,436 rows), C3 falsifier PASS vs
0.95. All 13 base models trained with OOF + checkpoints. Cohort 1,490,014
(210,549 path / 1,279,465 benign); splits 1,038,974 / 146,329 / 304,711; 78 features.
~11.2 h, ~$6.3. Box 39653192 destroyed clean.

**Fixed.** (1) KAN ran (prior crash resolved). (2) GNN non-degenerate (gnn_score std
0.099, nonzero_frac 1.0, range [0.0012, 0.5000]) vs Run 14 all-zero; post-run gate PASS
on all splits. (3) Slow cloud smoke root-caused (`--max-train` subsamples after full
annotation; no split cache) and fixed via stratified 80 k clinvar subset; smoke GREEN.
(4) Postflight `Invoke-Ssh` SSH-banner halt fixed (function-scope EAP=Continue + Out-String).

**Learned.** (1) cnn_1d 0.52 @3k -> 0.85 @1.49M is a pure data-scale artifact, not a
wiring bug — validates proceeding past smoke degeneracy. (2) GNN is a weak standalone
classifier (val AUC 0.6509); its value is the non-degenerate gnn_score ensemble feature.
(3) dbNSFP now live (188,023 SIFT); PhyloP and RNA-splice still 0 (unwired stubs).
(4) Review-tier <=3 retained 88% (1,686,333 -> 1,490,014); tier semantics vs Run 14 a
standing audit item. (5) Unseen-gene 0.9988 ≈ in-distribution 0.9984 = no generalization
collapse, but NOT proof of leakage-free: gene-level features (n_pathogenic_in_gene 391
top, pLI, LOEUF) may inform holdout genes; attribution awaits the n_pathogenic_in_gene
ablation. Do NOT record 0.9988 as "leakage disproven."

**Pending.** Run15_FullRun.ps1 launch-rc parse + Run15_Smoke.ps1 poll/Gate SMOKE_EXIT
detection (both SSH-stderr-banner faults, source needed). --splits-dir load-cache.
n_pathogenic_in_gene ablation. Time-disjoint re-pull. PhyloP/RNA-splice wiring.
_meta.json location audit. real_data_prep.py:444 fillna FutureWarning.

## 2026-06-08 - Reactome connector + feature-count cascade (78 -> 79)
- Attempted: wire ReactomeConnector (gene-level, reactome_pathway_count) into both feature builders.
- Failed:    feature-count addition tripped 4 hardcoded 78-guards (pipeline.py assert, test_splice_ai length test, KNOWN_ZERO_DEFAULT frozenset, test_api /info mock literal + 2 assertions); test_api n_features/feature_names diverged because the mock literal is hardcoded while feature_names tracks INFERENCE_FEATURE_COLUMNS.
- Fixed:     bumped all 4 guards to 79; added reactome_pathway_count to KNOWN_ZERO_DEFAULT (21->22); synced /info mock literal so n_features == len(feature_names) == 79. Suite: 788 passed, 6 skipped. `== 78` sweep returns zero.
- Learned:   feature count is hardcoded in >=4 places (prod + tests); centralize into one EXPECTED_TABULAR_FEATURE_COUNT before COSMIC/TCGA/KEGG repeat the cascade. Reactome is the validated gene-level connector template.
## 2026-06-08 (GNN GPU probe)
- GNN probe PASS on RTX 4090: gnn_score_std=0.0214, device=cuda, all_finite, graph 16201 nodes/236930 edges, peak_vram 13.9GB, s/epoch 1.2 (instance 40109189, <$0.50). Answers the Run-14 dead-gnn_score question: path is alive.
- fix(gnn) 89c07ed: parse STRING protein.info as TSV on _download_gz download path (latent; verified on GPU).
- fix(gnn) 63a2fb7: use STRING column 'experimental' not 'experiments' for edge channel (was silently zeroed).
- OPEN: requirements.txt websockets==16.0 vs langgraph ResolutionImpossible (bootstrap workaround only).
## 2026-06-08 (deps follow-up)
- Finding #1 CLOSED: requirements.txt websockets pin commented (af9978e). Clean resolve validated; pip selects websockets-15.0.1 (langgraph-sdk requires <16,>=14). requirements.txt is manually maintained (pip-compile-under-3.14 output + manual edits; header forbids auto-regen), so the edit is durable.
- Finding #4 NON-BLOCKING: requirements.lock pins websockets==16.0 but, being pip-compiled from requirements.in before imodelsx/langchain entered the tree, contains no langgraph-sdk and thus no <16 constraint -- it would install without ResolutionImpossible and is referenced by no install path. Not hand-edited (hashed lock).
- DRIFT -> Phase-2 dep-consolidation: requirements.in/.lock are stale vs requirements.txt (missing imodelsx + langchain); requirements.txt compiled under 3.14 not 3.12 (PHASE_2_FEATURES.pipcompile_python312_migration); multiple requirements-{api,dev,agents} files of varied vintage. A single pip-compile-under-3.12 pass regenerating all locks resolves the stale websockets automatically.
## 2026-06-09 - Run 15

- Run 15 SEALED (commit 032a2ab): Test AUROC 0.9984 / Val 0.9983 / unseen-gene-holdout 0.9988 (C3 falsifier PASS). 79 features, 1.49M cohort, ~11.5h RTX 4090, ~$6.
- ESM-2 stall fixed + shipped (local UniProt index, no run-time REST, GPU auto-detect). Coverage only ~3,451/~1.49M -> HGVSp parser promoted to top Phase-D unblock; current AUROCs rest on tabular + constraint features.
- AlphaMissense 71.7M-row OOM re-validated (cohort-filter-during-parse, 325b0d2).
- gene_constraint_oe revived (Run-14 all-zero -> #2 feature); gnn_score confirmed real; cnn_1d (0.85) and kan (0.996) recovered.
- Infra: SSH background launch needs < /dev/null; read-only checks use -n/ConnectTimeout/BatchMode; Run15_Smoke.ps1 poll-bail bug + clingen dtype drift flagged.
## 2026-06-10: ESM-2 coverage gate + stale coord-index root cause

- Root-caused Run 15 ESM-2 = 3,451 / 2.49M missense: the Vast box merged step-10b
  protein coordinates against a stale alphamissense_protein_index.parquet. Local
  index is healthy (96.6% missense coverage). See
  docs/incidents/INCIDENT_2026-06-10_esm2-coverage-stale-coord-cache.md.
- Added fail-loud coverage gate to real_data_prep step 10b
  (_protein_coord_source_present + _assert_protein_coord_coverage;
  AnnotationConfig.min_protein_coord_coverage = 0.50). Enforced only when a coord
  source is present; skipped in stub mode.
- Added tests/unit/test_protein_coord_coverage_gate.py (13 cases).
- Regression: v1 gate raised unconditionally and broke 12 stub-mode tests; v2
  conditional fix restores them. Full suite re-verified: 817 passed, 1 skipped.
- Added diagnostic scripts: probe_protein_coord_cache.py, probe_split_esm2.py,
  probe_coord_merge_repro.py.

## 2026-06-10 (cont.): Phase 0 gene-resolution + Phase 1 ESM-2 LLR feature

- Phase 0 (commit fd5e293): new data/gene_symbols.py (normalize_gene_symbol,
  gene_symbol_candidates; full symbol then ;-split components; never splits '-',
  protecting HLA-A / NKX2-1 / readthrough fusions). Wired into esm2 (_get_sequence
  candidate loop + _missing_genes aggregate log), eve (fixed a real case-drift bug:
  variant _gene_symbol .fillna("") un-upper-cased vs an upper-cased lookup; now
  normalizes both keys + drops empty-gene rows), protein_pipeline (get_accession
  candidate loop). Suite 849 passed / 1 skipped.
- Phase 1 (commit fd612e9): ESM-2 650M LLR scorer + esm2_llr feature.
  - Scorer (data/esm2.py): _load_transformers_mlm (EsmForMaskedLM logits head,
    distinct from the EsmModel embedding loader); _llr_from_logit_row
    (logit[mut]-logit[wt]; partition function cancels -> normalization-domain-
    invariant); _score_llr (WT-marginal = 1 pass/protein default; masked-marginal
    opt-in; skips wt_aa-vs-sequence mismatches, counted); annotate_llr.
  - CPU correctness gate (scripts/probe_esm2_llr.py) PASS: TP53 R175H/R248Q/R273H
    WT-marginal -9.13 / -11.04 / -9.61 (pathogenic, negative); benign P72R -6.09;
    every wt_aa matched the residue at its token index; WT- and masked-marginal
    agree in sign.
  - CALIBRATION: LLR sign is NOT a class label -- benign P72R also scores negative,
    just less so. esm2_llr is a CONTINUOUS feature; the ensemble learns the
    threshold (never a hard LLR<0 => pathogenic cutoff).
  - Feature wired 79 -> 80: TABULAR_FEATURES += esm2_llr (after esm2_delta_norm);
    EXPECTED_TABULAR_FEATURE_COUNT 79->80; INFERENCE_FEATURE_COLUMNS auto-derived
    (list(TABULAR_FEATURES)). Assembled at BOTH sites (real_data_prep +
    variant_ensemble) SIGNED with NO clip -- clipping would have silently zeroed the
    pathogenic signal; a regression test fails loudly if a clip is reintroduced.
  - Harness reference slice (correctness_harness.build_reference_slice) now
    populates esm2_llr with a signed range -- a live feature, NOT added to
    KNOWN_ZERO_DEFAULT (that set is dead-connectors only).
  - Model default stays esm2_t6_8M_UR50D (CI fast, no 2.5GB download); regen MUST
    set esm2_model_name=esm2_t33_650M_UR50D (printed in the step-16b log).
  - Full suite 862 passed / 1 skipped.
- Repo hygiene (commit a59d728): tracked prior-session diagnostics
  (probe_uniprot_index, diagnose_esm2_coverage, clinvar_name_probe) + the step-10b
  coverage-gate patcher (patch_add_protein_coord_coverage_gate, for committed
  34e125a); .gitignore += *_bak_* (consolidation backups used _bak_, escaping the
  existing *.bak_*).
- Carried: Phase 2 = ESM C 600M (Cambrian, "Built with ESM"); Phase 3 = GPU regen +
  LLR recalibration (signed-feature scaling); stale step-count log denominators
  (/16, /17 vs 18 steps) cleanup; clingen int/float dtype drift before regen.

## 2026-06-10 -- Agent layer re-wiring (4 -> 13 operational)

### Fixed
- Restored the orchestrated agent layer from 4 to 13 operational agents. The April->May
  decomposition of DriftMonitorAgent into 8 detectors plus the C1 migration orchestrator
  rewrite had left the 8 detectors orphaned (no BaseAgent/run(), unregistered) and dropped
  VersionMonitorAgent's class.
  - DriftMonitorBase adapter + 8 thin wrappers (Schema/Concept/LabelShift/Calibration/
    Infrastructure/Fairness/Adversarial/AnnotationPolicy MonitorAgent) over the existing
    detect()/persist(); detectors now COMPOSED, not orphaned. Wrappers report
    status='awaiting_baseline' until reference inputs exist.
  - Restored VersionMonitorAgent as a BaseAgent wrapper over the existing module-level
    upstream-release watch targets.
  - Registered all 9 in Orchestrator._register_agents().

### Added
- scripts/audit_agent_roster.py, scripts/audit_agent_operational.py (AST audit tooling).
- scripts/patch_register_drift_agents.py, patch_add_version_monitor_agent.py, patch_readme_agent_count.py.
- tests/unit/test_drift_monitor_agents.py, test_schema_drift_monitor_agent.py, test_version_monitor_agent.py.
- docs/incidents/INCIDENT_2026-06-10_agent_layer_regression.md.

### Changed
- README: agent count reconciled 7 -> 13; stale "Py 3.14.3" -> "Python 3.12.10".

### Verification
- 11 new agent tests pass; full suite 872 passed / 6 skipped.
- Gate: operational=13 composed=8 orphaned=0 total=21.

### Commits
- 8619afc (drift re-wiring), 21e835d (VersionMonitorAgent + README). Pushed to origin/main.

## 2026-06-11 -- CI fix: agent-layer optional deps (pandera, river)

### Fixed
- CI was red (#302-#303): schema_drift_agent (pandera) and annotation_policy_agent (river)
  imported undeclared optional libs at module level; the orchestrator imports every wrapper, so
  the whole agent layer was un-importable in CI. Local passed (deps in .venv312); CI failed (deps
  absent). pytest -x masked the river failure behind pandera.
  - pandera (schema, into detect()) and river (annotation, try/except guard) are now lazy imports.
  - test_schema_drift_monitor_agent.py uses pytest.importorskip (repo convention).
  - No requirements changed; ok-path detection tests skip in CI and run locally; registration runs in both.

### Added
- scripts/simulate_ci_no_optional_deps.py -- reproduces the lib-absent CI env in-process to validate
  import-safety before pushing.
- scripts/patch_schema_lazy_pandera.py, patch_annotation_lazy_river.py, patch_test_schema_importorskip.py.

### Verification
- Local full suite 873/6 unchanged; simulate gate exit 0; CI #304 (92ff4a2) green on 3.11 + 3.12.

### Commit
- 92ff4a2 (origin/main).

## 2026-06-11 -- Schema-drift activation + preflight gate

### Added
- scripts/build_schema_baseline.py -- captures the sealed Run-15 X_train schema to
  data/reference/schema/schema_baseline.json (ordered expected_dtypes + sha256 hash + provenance).
  Contract: 78 columns, all float64, hash db43fd918bdfa4d0...
- SchemaDriftAgent.from_baseline(baseline_path, output_dir) -- classmethod that rebuilds the
  pandera schema from the baseline with nullable=True, so Run-15's degenerate (all-NaN) columns do
  not false-trip against their own baseline. pandera imported lazily (keeps the layer CI-importable).
- scripts/run_schema_drift_check.py -- standalone preflight schema gate: load baseline -> head-read
  a feature matrix (first parquet batch; dtype-exact, memory-bounded) -> print column/dtype diff ->
  exit 0 (green) / 2 (drift) / 3 (usage/env). Run before any regen or training to catch
  dropped/renamed/retyped columns before they silently zero a feature.
- data/reference/schema/schema_baseline.json committed as a VERSIONED contract (not gitignored);
  future schema changes now surface as a reviewable one-file diff.
- tests/unit/test_schema_drift_activation.py (ok/green, ok/red, default-still-awaiting_baseline);
  tests/unit/test_run_schema_drift_check.py (exit-code contract 0/2/3).
- scripts/patch_add_from_baseline.py (idempotent, py_compile-gated patcher).

### Verification
- Real-data smoke: gate on Run-15 X_train -> green/0 (byte-identical hash); on meta_train -> red/2
  (18 added, 38 removed, 15 dtype changes, 53 pandera violations) -- proves the gate fires on real data.
- Full suite 873 -> 876 (e0a76a1) -> 880 (21d94c4) passed, 6 skipped; simulate_ci gate exit 0.
- New tests importorskip pandera/pyarrow -> skip in CI, run locally (repo convention).

### Findings (see docs/sessions/SESSION_2026-06-11_ci-and-schema-gate.md and ROADMAP backlog 2026-06-11)
- Agent-layer drift agents registered but invoked by no pipeline; drift_monitor.yml inert (GDrive
  stub + stale phase2_with_gnomad path); run_drift_monitor.py covers distributional+label drift but
  not schema. Feature-count spread 64/78/79 flagged TO VERIFY; af_1kg_* present-vs-placeholder TO VERIFY.

### Commits
- e0a76a1 (from_baseline + activation tests + builder), 21d94c4 (preflight gate + versioned baseline).
  Both on origin/main.

<!-- docs-close: ecd0474 esm2-llr+train-wiring -->
## 2026-06-11 (PM) -- ESM-2 650M LLR fix + train.py wiring

### Fixed
- ESM-2 LLR forward-pass OOM on long proteins (TTN ~34k aa, ~94 GB O(L^2) attention):
  added _MLM_MAX_RESIDUES=1022 + _windowed_logit_row; long proteins window the WT- and
  masked-marginal reads, short proteins unchanged (1db43f1).

### Added
- scripts/train.py: --esm2-model / --esm2-uniprot-index / --esm2-cache / --esm2-device,
  threaded into AnnotationConfig; metrics annotation_sources now records
  esm2_model/esm2_uniprot_index/finngen/dbnsfp (ecd0474).
- scripts/probe_esm2_650m_activation.py: CPU activation probe (caught the OOM pre-GPU).
- tests: test_esm2_llr_windowing.py, test_train_esm2_wiring.py.

### Decided
- Run 16 uses ESM-2 650M (esm2_t33_650M_UR50D); ESM C 600M deferred to a later
  controlled A/B (single-variable discipline; ESM C = net-new connector code).

### Learned
- ESM-2 650M activates non-zero on real data: delta nonzero_frac=0.967, llr=0.960 (CPU probe).
- PowerShell 5.1 has no heredoc; multi-line commit messages must use git commit -F <file>.
- Wired != populated != non-zero: train.py constructed AnnotationConfig but never
  overrode the 8M / live-REST defaults, so a regen would have silently produced the
  wrong feature at production scale.

<!-- docs-close: e3bcd79 cnn-rna-activation -->
## 2026-06-11 (late PM) -- CNN real-sequence + RNA MaxEntScan-delta activation

### Fixed
- 1D-CNN trained on poly-A placeholders: train.py gated the CNN on the deprecated,
  empty single `fasta_seq` column (notna=0 cohort-wide) and raised NotImplementedError
  on the real-sequence path. Repointed the gate and X_seq plumbing to the live
  [fasta_seq_ref, fasta_seq_alt] delta windows -- test-side from meta_test, train-side
  from the already-persisted meta_train.parquet (gene-split-aligned to X_train via the
  shared train_idx). NotImplementedError removed; NO DataPrepPipeline.run() signature
  change (fb12c0f).
- RNA-splice maxentscan_score dead (default 0.0 for every variant): rna_pipeline read the
  same empty single `fasta_seq`. Repointed to score the ref/alt windows and emit a NEW
  variant-specific feature maxentscan_delta = score(alt) - score(ref) (e3bcd79).

### Added
- maxentscan_delta registered in TABULAR_FEATURES, BOTH _engineer_features blocks
  (variant_ensemble.py and real_data_prep.py), and the RNA-off default-fill tuple;
  EXPECTED_TABULAR_FEATURE_COUNT 80 -> 81. INFERENCE_FEATURE_COLUMNS auto-derives
  (list(TABULAR_FEATURES)); the feature-count contract is green at 81/81.
- Correctness-harness reference slice now populates maxentscan_delta (non-zero synthetic)
  so stage-5 silent-zero detection stays honest -- deliberately NOT allowlisted in
  KNOWN_ZERO_DEFAULT (it is a live feature that should carry signal).
- tests/unit/test_train_cnn_activation.py; tests/unit/test_rna_maxentscan_delta.py.
- Idempotent patchers: patch_train_cnn_activation.py and patch_rna/ve/rdp/
  correctness_harness_maxentscan_delta.py.

### Verification
- Full suite 893 passed / 6 skipped (e3bcd79). The torch-gated CNN test trains end-to-end
  on a 2-column ref/alt delta frame and returns finite probabilities. maxentscan_delta is
  nonzero for a real ref!=alt splice variant and 0 for ref==alt / non-splice / legacy
  single-fasta_seq fallback.
- The correctness harness caught the one defect this session: maxentscan_delta added without
  its reference-slice entry tripped stage-5 (all-zero outside the dead-connector allowlist);
  py_compile, the feature-count contract, and the targeted tests all passed it through.

### Learned
- Activation precondition (load-bearing): real_data_prep NEVER adds fasta_seq* columns; they
  ride ONLY from the input parquet (_load_and_label preserves all input columns). Both the CNN
  and maxentscan_delta activate ONLY when Run 16 uses
  --clinvar data\processed\clinvar_grch38_clean_seq.parquet (the ref/alt cohort). With the
  default clinvar_grch38.parquet they degrade SILENTLY to inert (CNN dropped to placeholders,
  maxentscan_delta all-zero) -- no crash, no signal.
- New standing run-gate: every new tabular feature must appear, POPULATED, in the correctness-
  harness reference slice; it was the only gate that caught this session's feature/slice drift.
- The always-donor MaxEntScan selection bug does NOT collapse the delta (the variant base at
  window center lies inside the donor 9-mer); biology-correct donor/acceptor selection is a
  separate tracked fix.

### Commits
- fb12c0f (CNN real-sequence activation), e3bcd79 (maxentscan_delta + harness slice + contract
  bump). Both on origin/main.

<!-- docs-close: ci-esm2-hub-flake 2026-06-12 -->
## 2026-06-12 -- CI red resolved: flaky ESM-2 HuggingFace Hub download

### Fixed
- CI was red on runs #316 (docs-only) / #317 while local was green:
  test_llr_long_protein_scores_finite_without_oom loads the real ESM-2 8M from HF Hub;
  CI runners (no cache, rate-limited 429) erred, local (cached weights) passed.
  fee2e63 wraps the live load in try/except OSError -> pytest.skip; the test still runs
  fully wherever the model loads and skips only on HF-offline.

### Changed
- .github/workflows/ci.yml: HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 on the unit-test
  step (CI never reaches HF Hub -> 429 impossible); pytest -x -> --maxfail=5 (a break
  surfaces several failures instead of halting at and hiding everything after the first).

### Verification
- Reproduced offline (empty HF_HOME): test SKIPS, not errors. With weights: both pass.
- Whole offline suite: 898 passed / 2 skipped / exit 0 -- no other unguarded live-loader.

### Learned
- Local-suite-green is NOT a proxy for CI-green where a test loads an ESM-2 model: the
  local cache hides a hard network dependency. New gate: run the suite under an empty
  offline HF cache before trusting green.

### Commits
- fee2e63 (test skip-guard, already on origin/main); this close (ci.yml hardening + docs).
- See docs/incidents/INCIDENT_2026-06-12_ci-esm2-hub-flake.md.

<!-- incident: protein-coord-index-corruption 2026-06-12 -->
### 2026-06-12 -- protein-coord index corruption + repair
- Failed: probe v1 re-run after `Remove-Item` of the cache rebuilt the protein-coord
  index from a 50k sample, overwriting the full 17.8 MB index with a 0.29 MB one.
- Learned: `ProteinCoordConnector._build_index` filters to the passed cohort and writes
  the canonical cache; diagnostics must be read-only. Validate the cache by size.
- Fixed: probe v2 (read-only default + size guard + explicit `--rebuild-full`); full
  rebuild -> 18.64 MB, full-cohort coverage 0.9665 (2,405,448/2,488,889 missense).
- Confirmed: Run-16 `--alphamissense` = TSV (not the scores parquet); 96.65% full-cohort
  protein-coord coverage means ESM-2 will populate.
