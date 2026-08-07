# SESSION 2026-08-06 — OP-1 step 4: the two operating-point selectors

**Base: `HEAD = origin/main = 0a3041d`. Result: `3ddf617`, pushed.**

**Ratchet 4300 → 4353 (+53). Armed full suite 4347 passed, 6 skipped, 0 failed
in 16m08s; 4353 collected. Skip surface unchanged at 6.**

D12 closes. The twelve-defect register opened 2026-08-01 is now fully closed.
Neither legacy selector is touched, `evaluator.py` is untouched, and nothing
imports the new selectors yet.

This document also records the falsification of **GUARD-1**, which the roadmap
had carried across six deltas, and seven defects of the author's — every one
caught before it reached the tree, and one of them caught by a crash rather
than by a check.

---

## 1. What changed

`thresholds.py` grows from 850 to 1,524 lines. Eleven anchored edits and one
appended section:

| id | change |
|---|---|
| E1 | `from .population import EvaluationPopulation, PopulationComparison` |
| E2 | `__all__` rebuilt, 11 → 17 entries |
| E3 | `ExactThresholdSweep.__init__` calls the confusion-state invariant |
| E4 | `SELECTION_EVALUATION_INDEPENDENCE_NOT_ESTABLISHED` blocker member |
| E5 | its `_BLOCKER_PROSE` entry |
| E6 | `OperatingPointOutcome.selection`, defaulted, quoted forward reference |
| E7 | `__post_init__` refuses an outcome and a selection record that disagree |
| E8 | `refused()` accepts and forwards `selection` |
| E9 | `to_dict()` emits `"selection"` last |
| E10 | `_validate_unique_confusion_states`, above `ExactThresholdSweep` |
| E11 | the certification-blocker completeness gate in the outcome tests |
| E12 | the step-4 banner section: three vocabularies, `OperatingPointSelection`,
five helpers, two public selectors |

`tests/unit/test_operating_point_selection.py` is created: 27 functions, four
parametrised at ×3, ×20, ×3 and ×4, for 53 cases. All 53 come from that file;
`test_operating_point_outcome.py` gained a dictionary entry and no test.

## 2. D12: what it was, and what closes it

`if diff < best_diff` is strict, so the FIRST candidate reaching the minimum
wins and the answer depends on TRAVERSAL DIRECTION. The two legacy selectors
traverse opposite ways: `_find_operating_point` walks `np.linspace(0, 1, 1000)`
ascending, permissive to conservative (`evaluator.py:1515`);
`_find_high_ppv_point` walks `np.sort(np.unique(p))[::-1]`, conservative to
permissive (`evaluator.py:1603`). `sweep_thresholds` runs conservative to
permissive, so a selector written naturally against it would have INVERTED the
sensitivity-target tie policy while appearing to preserve it.

What closes D12 is not the rule but the ARTIFACT THAT STATES THE RULE. A
threshold chosen by a declared order and reported without it is, to a reader,
indistinguishable from one chosen by traversal accident. So a frozen
`OperatingPointSelection` travels with every outcome, naming the objective, the
tie-break, the target, the status, the candidate count, the feasible count and
the selected index — and it serialises.

## 3. Every criterion in the persisted policy can decide something

A four-stage draft ended each key with the canonical threshold declaration.
Sabotage could not detect its removal. Measured over 400 random cohorts with
duplicate scores forced: on a canonical sweep `n_flagged` is STRICTLY
increasing, so "fewer flagged" and "most conservative threshold" induce THE SAME
ORDER, and the suffix was unreachable. A persisted policy code naming a stage
that can never run would overstate what chose the candidate.

The canonical order is real and is tested — as a property of the SWEEP, not as
a selector criterion. That distinction is the whole content of the finding.

## 4. The uniqueness invariant moves to the type

The shortened keys are total exactly when each candidate holds a distinct
`(true positive, false positive)` pair. Without that, every key ties,
`np.lexsort` is stable, and the winner falls back to ARRAY ORDER — D12
reopening through the back door in the commit that closes it.

So `ExactThresholdSweep.__init__` refuses such a sweep. A check repeated in
every selector is one a future consumer forgets, and the policy would then rest
on an unstated producer convention instead of a contract. It rejects nothing the
suite already builds: 300 random cohorts through the live `sweep_thresholds`,
zero violations, and no test constructed the type directly before this commit.

## 5. Two required changes the handoff's change list did not contain

Both found by reading the live module rather than the replica the payloads were
drafted against. Both are CALL-TIME failures, invisible to `--collect-only`.

* `thresholds.py` imported only `EvaluationPopulation` from `population.py`,
  while `_selection_certification_blockers` references `PopulationComparison` in
  a runtime expression. Unfixed, the first successful selection raises
  `NameError`.
* `OperatingPointOutcome.refused()` took no `selection` argument, and all four
  refusal paths pass one. Unfixed, every refusal raises `TypeError`.

The targeted run of 124 tests is where these would have surfaced — after the
installer had already written the file.

## 6. The verification was redone against the real class

The payloads were verified against a replica, and a replica agreeing with its
author is the defect that produced the step-3c `NameError`. So the live
`OperatingPointOutcome` was reconstructed from `inspect.getsource`, the four
edits applied to it, and the whole battery re-run against the PATCHED
PRODUCTION CLASS: **53 of 53 cases pass, 14 of 14 mutations detected**, plus 21
permutation orderings with zero divergent.

The replica could not have exposed either finding in section 5 — it defines
`PopulationComparison` and accepts `selection` in the same file. That is what
Part 10.4 of the handoff meant by unverified, discharged by measurement.

## 7. GUARD-1 is falsified, and in both directions

GUARD-1 has read, across six roadmap deltas: *step 6's cutover must scope this
guard to the legacy path or extend it deliberately.* Measured 2026-08-06:

* Both legacy selectors compute `preds = (p >= t).astype(int)` inline
  (`evaluator.py:1516` and `1607`). Neither calls `apply_decision_threshold`.
* A repository-wide search finds `apply_decision_threshold` at `metrics.py`
  711, 754, 778, 1524 and `registry.py` 927, 932, 1022, 1024. **No line in
  `evaluator.py`.**
* `thresholds.py` does not bind the name at all, and neither
  `sweep_thresholds`, nor either new selector, nor `metrics_from_counts` calls
  it. The exact sweep derives counts by sorting and cumulative sums.
* A live `evaluate` call: the selectors ran **3 times** and the report printed
  three distinct operating-point thresholds — 0.592, 0.544, 0.329 — while
  `apply_decision_threshold` was invoked 18 times at exactly **one** distinct
  threshold, `(0.5, ">=")`.

`test_report_construction_performs_no_threshold_comparison` patches
`metrics_module.apply_decision_threshold` and asserts one distinct
`(0.5, ">=")`. It is green, and it has never observed an operating point. The
predicted step-6 collision cannot occur, because the replacement is invisible to
that guard for the same reason the legacy path is.

**GUARD-1 is DISCHARGED by falsification.** The real gap it was pointing at —
the operating-point path has no threshold-provenance guard of any kind — is
absorbed into **OPCOV-1**, whose statement is restated accordingly. Discharging
without moving the gap would have closed the item and lost the concern.

## 8. Seven defects of the author's, all caught before the tree

The pattern matters more than the count, and it is one pattern: **a conclusion
that agreed with its author for the wrong reason.**

1. An abstract-syntax-tree probe matching only `ast.Name` calls, blind to
   `T.ExactThresholdSweep(...)`. Its empty result was consistent with the claim
   and did not establish it; the stronger instrument had already run.
2. A recommendation to add `from typing import Optional`, reasoned from the
   import block without reading the class that uses it — which already carries
   three such annotations and works under PEP 563. Withdrawn; the exposure is
   F821-1's, not step 4's.
3. A ratchet-shape hypothesis inferred from a byte-count coincidence: 368,334
   bytes is within one percent of 4,300 node identifiers at 85 characters.
   Falsified by counting bare-integer lines — exactly one.
4. An unquoted forward reference in the new `selection` field, correct only
   because line 27 carries `from __future__ import annotations`. Caught by
   EXECUTING the patched class, not by reading it. Now quoted.
5. A warning that the installer's backups would reach the commit. `.gitignore`
   line 262 has matched `*.bak_*` since 2026-07-11. Retracted on measurement.
6. A GUARD-1 probe that reported the legacy selectors NOT FOUND, because it
   used `getattr(module, name)` and they are METHODS on `ClinicalEvaluator`.
   The functions were exactly where the roadmap always said.
7. That probe's successor crashed with `UnicodeEncodeError` on `\u2192`,
   printing repository source to a code-page-1252 console. The call-graph trace
   never ran. **Caught by a crash, not by a check** — which is luck, not
   method, and is recorded as such.

Standing lessons added: an interactive PowerShell `else` must share a line with
the preceding closing brace; and any probe printing repository source must call
`sys.stdout.reconfigure(encoding="utf-8")` first, because this repository
legitimately contains `→`, `—` and `≥`.

## 9. Follow-ups: one discharged, five raised, two restated

Twenty-nine carried in, ENUMERATED from the step-3c delta rather than
estimated. One discharged, five raised: **thirty-three open.**

* **PERSIST-1** — the selection record serialises
  (`OperatingPointOutcome.to_dict()` emits `"selection"` last) but no artifact
  carries one, because the selectors are unwired by design. Until step 6, D12's
  closure is demonstrated by tests rather than by a written artifact.
* **BACKUP-1** — roughly 115 stale `.bak` files across the working tree from
  guarded-patcher runs in June and July 2026, all git-ignored. No commit risk;
  a reader can still open the wrong file.
* **DOCLOC-1** — session documents split across `docs/` and `docs/sessions/`
  by date, with the changelog citing both paths.
* **DOCX-1** — `docs/ROADMAP.docx` dated 2026-06-16 against `ROADMAP.md` at
  2026-08-06.
* **CONSOLE-1** — the report's `≥` renders as mojibake in a code-page-1252
  console; a `Tee-Object` capture would carry it into a log file, which is the
  round trip that put 301 mojibake lines in the changelog.

**Restated, not discharged:**

* **OPCOV-1** now also carries GUARD-1's real gap: no threshold-provenance
  guard observes the operating-point path, legacy or new.
* **CHANGELOG-2**'s literal claim is stale — the newest entry is now
  2026-08-05, not 2026-07-31 — but the gap it named is not closed: the file
  jumps from 2026-08-05 to 2026-07-31 while `docs/` holds session documents for
  thirteen milestones in between. A reader checking only the literal claim would
  discharge it wrongly.

## 10. Acceptance

| | |
|---|---|
| base | `0a3041d` |
| result | `3ddf617` |
| `thresholds.py` | `5cc26b455ac279a3…` → `2b457d6ff2240816…`, 850 → 1,524 lines |
| ratchet | 4300 → 4353 (+53), measured by the installer |
| collected | 4353, measured; predicted 4353 by two independent counts |
| targeted tests | 124 passed in 12.29s |
| coupled tests | 100 passed in 31.18s |
| full suite, unarmed | 4347 passed, 6 skipped, 0 failed, 33 warnings, 18m20s |
| full suite, ARMED | 4347 passed, 6 skipped, 0 failed, 33 warnings, 16m08s |
| skip surface | unchanged at 6 |
| new tests | 27 functions, 53 cases |
| sabotage | 14 mutations, 14 detected, 0 undetected |
| permutation | 21 orderings, 0 divergent |
| production files touched | one |

The 33 warnings are pre-existing scikit-learn degenerate-cohort warnings from
absence and population tests; these selectors import no scikit-learn. The armed
run was 2m12s FASTER than the unarmed one on the same tree, which settles the
duration question by measurement rather than supposition.

## 11. Next

**Step 5 — the shadow comparison.** Both implementations run over the same
cohorts and every difference is attributed. OPCOV-1 is the constraint to read
first: the legacy selectors are thinly covered, and a shadow comparison rests
on that coverage.

**Step 6 — the cutover.** It does NOT have to reckon with GUARD-1; that was
measured today and the item is discharged. It does have to reckon with the
absence GUARD-1 was standing in for, which is now OPCOV-1's.

Thirty-three follow-ups are open. The five raised here are recorded and none is
touched.
