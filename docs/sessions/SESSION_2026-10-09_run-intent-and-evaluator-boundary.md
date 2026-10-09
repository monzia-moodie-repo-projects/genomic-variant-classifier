# Session record, 2026-10-09: the run intent, the reference-evaluator boundary, derived completeness and shared burden inputs

Author: Monzia Moodie. Design authority: owner ruling 2026-10-09 (790 lines, SHA-256
`a2c2e9d62483103745e06dd3a8cad9a1065f967d335a0441ccf9715a6340b250`, read in full). Base: GitHub main `702ebfe`, tree
`93b2f13881ca4ad92bbd77b318029e7499ce35f0`.

## 1. What the ruling asked for

The ruling's bounded next implementation has seven steps; this change delivers the first four:

1. Bind the stage and the release-policy identity in a pre-execution intent.
2. Derive completeness from admitted exposure and pair evidence.
3. Enforce reference withholding before evaluation is invoked.
4. Add the shared burden-input summary from the existing actual-call trace.

Steps 5 to 7 (environment admission and isolated qualification, the frozen feasibility run, a later evaluation operation) remain.

## 2. Measurements taken before any design decision

| Question | Measurement | Consequence |
| --- | --- | --- |
| Does an operation-intent mechanism exist? | No. `maintenance_channel.py` claimed "`operation_intent` already provides the root" and "owned by OperationRecord"; no RuntimePaths property and no class of either name exists in the tree, and `git log -S` finds the strings only in `50aea17`, the commit that wrote the sentence. | The ruling's alternative applies: a typed, versioned intent record. The false docstring is corrected beside the claim. |
| Can `admission.RunRecord` hold the intent? | No. It is a mutable execution-status record (`started` to a terminal state); its `stage` field names the active execution step. | A separate immutable record; RunRecord is unchanged. |
| Does the exposure recorder capture the effective burden input? | No. Version 1 records `n_trans`, `n_valid`, `pi0a`, `pi0b` and the mixture weights only. The installed `run_dandelion_for_exposure` body (printed by position from DANDELION 0.1.0) holds the post-clamp, gene-named burden vector `p_b` at position 18, the pi0 guard. | Recorder version 2 records `names(p_b)` and the exact values there. |
| What exactly is `clamp_p`? | Printed from the installed namespace: `p <= 0` becomes `.Machine$double.xmin` (`0x1p-1022`), `p >= 1` becomes `1 - 1e-15` (`0x1.ffffffffffff7p-1`, measured in R and Python). | The judge checks every recorded input against `clamp_p` of the planned burden p-values, exactly. |
| Does the estimator depend on order? | `nonnullPropEst` averages cosines over the vector; floating summation is order-dependent. | The numerical identity is the ORDERED vector; grouping never sorts. |

## 3. Design, component by component

**Run and evaluation intents** (`inference/run_intent.py`). A RunIntent binds the run identifier, the declared stage, the sealed
contract digest, the frozen pair-plan digest, the exact input identities, the environment identity, the implementation tree, the
release-policy identity, a declared prior-knowledge statement and the earlier intents that informed it. An EvaluationIntent names
one admitted computation, its exact score bytes and, for a confirmatory evaluation only, the reference digest. Both accept only
their canonical rendering, are sealed exclusively before execution and are admitted only against the digest recorded at sealing
and the release policy of the implementation now running.

**Release policy identity** (`inference/exposure_outcomes.py`). The decision table `endpoint_release` reads is one immutable
structure whose canonical rendering has the identity `b56f33dcec5f68f55633487cf917be31bda0a677bbddf0f07cdc2c43e42f3abe`. A
changed table or rule changes the identity: that is an amendment, and every intent bound to the earlier identity is refused.

**The frozen plan** (`PairPlan`). Planned exposures, the pair grid, exclusions, usable-pair counts and ordered valid genes, with
invariants that tie them together. The fixture planner (`method_trace.plan_for`) and an independently hand-built plan of fixture
T2 produce the same digest (`3cf569ff...3f11`).

**Derived completeness** (`score_coverage`, `assess_execution`). The coverage rule refuses an empty eligible set, an unclassified
or infrastructure outcome, a grid mismatch, a score for an excluded pair and scores that disagree with the outcomes; it withholds on
a recorded mixture failure. It reports every applicable reason. Four derived fields stay separate: execution integrity, method
completion, evaluation permission and scientific interpretation. One refinement of the ruling is recorded: method completion is
`not_determined` when execution integrity is refused, because completion cannot be known then.

**The evaluator boundary** (`inference/evaluation_boundary.py`). `release_decision` has no reference parameter. `evaluate`
re-derives the assessment itself (it accepts none), refuses changed score bytes before any reference access, requires a
confirmatory evaluation of a feasibility computation to disclose it in `informed_by`, and opens the reference only when the
release table permits a reference recovery for the admitted evaluation stage.

**Recorder version 2 and the burden summary.** One file per distinct effective burden input; the reader checks numbering, orphans,
duplicates, gene counts and exact values. The summary groups exposures by their complete identity (biological support and
numerical input), never by a tolerance, and reports agreement across identical inputs, failure counts and the genes that lose
coverage.

## 4. Findings, defects and corrections recorded during the session

- **A silent last-value-wins defect in the judge (corrected).** The judge built the score matrix line by line in a dictionary, so
  a duplicated cell kept the last of two conflicting values; a malformed line raised a bare ValueError rather than a refusal. The
  score artifact, `mat_sig.tsv` and `nominations.tsv` are now read strictly.
- **A false documentation claim (corrected beside it).** See section 2.
- **A dated record that is not the change's date (correction note added).** The roadmap entry "Environment qualification 5" says
  2026-10-10, and so does its branch name; GitHub shows commit `6c3fdb6` and merge `a27ec6c` on 2026-10-06.
- **Errors of mine, caught by tests before delivery:** (1) a hand-typed hexadecimal literal for `1 - 1e-15` (`...fffbp-1`) that
  R and Python both contradict (`...fff7p-1`); the test now cites the measured value. (2) A test case for a missing stage that
  replaced a substring the canonical rendering does not contain (the last key has no trailing comma), so it tested the intact
  intent; a guard test now requires every parse case to change the bytes. (3) A runtime cross-check in `assess_execution` that no
  input can reach; the seeded-defect run showed it survived removal, so it was removed and the property test that proves the
  agreement it guarded is cited instead. (4) The R library path: after the container restarted, the R-gated tests first ran with a
  library path that omitted the site library holding qvalue's dependencies, so qvalue was "absent" and four route predictions
  failed; the correct path (`/usr/lib/R/site-library`) was measured, not assumed.
- **A pre-existing skip set, unchanged.** Eight cases in `test_inference_backend_trace.py` skip because the pinned DANDELION is not
  in R's default library in this environment; they are outside this change.
- **A load-sensitive performance test (pre-existing, recorded, not changed here).** `test_alphafold::test_rsa_performance_beats_naive`
  compares two WALL-CLOCK timings and requires fast < 0.80 x naive. It failed once, only while two full suites shared the sandbox's
  two processors (4.85 s against 0.8 x 5.36 s), and passed 6 of 6 times alone. Unloaded, the ratio measured 0.65 to 0.70 in both wall
  and processor time: a thin margin. A robust check would time processor use and compare growth across two input sizes; that is a
  separate change with its own evidence.

## 5. Verification

- Targeted inference tests: all pass (both R-gated method-fixture cases executed with the real DANDELION and qvalue).
- Node identities: +174 / -1 (the -1 is a renumbered parametrize identifier); suite 8,163 -> 8,336.
- Full suite by node identity, development sandbox (its 60 failures and 30 errors come from packages absent there and are identical
  before and after): 0 outcome changes on 8,047 shared tests; 174 added, all passing. Under the Windows-newline simulation (injection
  proven: the 14 known POSIX-only failures appear in both runs): the same, except the load-sensitive test of section 4.
- Seeded defects: 53 of 54 Python defects detected (the survivor led to the removal described above); on the final code 53 of 53;
  and 7 of 7 R recorder defects detected.
- The six forbidden-call cases of the ruling run with a reference loader that raises if it is called. A callback test verifies
  control flow; it does not prove filesystem isolation of the worker process.

## 6. What remains, in order

1. Commit the first artifact-inventory record (waiting for the evidence bundle `0d14514d...8df6` to be uploaded).
2. Admit the lockfile transition and finish environment admission and the isolated qualification (ruling step 5).
3. Write the real-data feasibility runner and seal its run intent BEFORE the estimator runs; run the frozen feasibility analysis
   (ruling step 6). Its report carries the assessment and the burden-input summary; it opens no reference.
4. Admit a later evaluation operation only under the declared policy and the documented prior-knowledge history (ruling step 7).
