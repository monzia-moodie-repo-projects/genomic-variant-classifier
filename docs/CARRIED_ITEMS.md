# Carried items — the register

**Established 2026-07-28 at commit `0cc663d`, on completion of Tier 1 item 6.**

## Why this file exists

Carried items were declared inside per-commit roadmap deltas, and their status
changes were recorded in *later* deltas. An item's current state could therefore
only be reconstructed by grepping the whole roadmap and reading the deltas in
chronological order.

That is two sources of truth for one fact, with no mechanism to detect
divergence — the same defect the metric stack spent fourteen commits removing
from the evaluation path. It had already gone wrong: **item (l) was discharged in
commit 2a-1 and still read as open eleven commits later.**

**This file is the single source of truth for carried-item status.** Every row
carries a VERIFICATION PREDICATE — a statement that can be checked by running
code — and `tests/unit/test_carried_item_register.py` runs them. An item claiming
to be OPEN whose predicate reports it closed is a test failure, not a stale
document nobody re-reads.

## A namespace collision, resolved

`docs/ROADMAP.md` uses single letters for **two different things**:

    root patterns (a)-(d)     recurring failure shapes, lines 1163-1194
                              e.g. "(d) A green result from a mutated
                              environment is evidence about the environment,
                              not about the code"
    carried items (a)-(s)     deferred work

`carried item (a)` and `root pattern (a)` are unrelated and sit 500 lines apart.
Carried items are referred to in this register by the prefix **CI-** and are
never written as a bare letter again.

---

## Open

| id | statement | raised | verification predicate |
|---|---|---|---|
| **CI-s** | Deferred imports in `registry.py` are a load-bearing contract, not a style. The module must import without scikit-learn, so any module-scope construction must bind kernels by name and resolve at call time. | 3b-2 | *inverted* — the predicate asserts the contract HOLDS; this row records the rule, and a violation is a failure |

## Discharged

| id | statement | raised | discharged | verification predicate |
|---|---|---|---|---|
| **CI-k** | Interior-edge agreement coverage between the two calibration implementations. | 2a-1 | **2b-1** | the binning suite exercises every interior edge |
| **CI-l** | The transitional label mask `select_finite_reference_labels`. | 2a | **2a-1** | the symbol exists nowhere in `src/` as code |
| **CI-m** | `metrics.evaluate` filters non-finite rows through `clean_arrays` rather than refusing. The item claimed a caller **cannot tell how many observations a number describes**. **MEASURED 2026-07-29: it can** — `n_input`, `n_dropped` and `n_pos` are all returned, and `CleanArrays` carries three separate drop counts. The counts were added by the fail-closed work of 2026-07-20 and the item was never updated. Its predicate tested a PROXY — whether `clean_arrays` is called — which is true by design and would have held the item open forever. | 2a | **this commit** | `evaluate` reports the attempted and surviving row counts |
| **CI-n** | `cohort_version` was called a weak provenance identity: most call sites pass a generic `"v2"`, so two cohorts could share it. **MEASURED 2026-07-29: they cannot.** `_derive_population_source_id` incorporates the ORDERED variant-identifier sequence — which is exactly what the item said was missing — and its own docstring says so. Frames differing in variants, or in variant ORDER alone, yield distinct identities. Its predicate tested whether the function ACCEPTS a `cohort_version` parameter, which is permanently true. | 2a | **this commit** | two frames sharing a version and partition but differing in variants yield distinct identities |
| **CI-o** | The evaluator abstract-syntax-tree guard, deferred until the evaluator was actually retired rather than merely observed. | 2b-2 | **3b-2** | `test_computation_path_guards.py` exists and inspects the report path |
| **CI-p** | `MetricResult.to_dict` emitted raw `NaN` while `from_dict` documented reading `null` back, and `dump_strict_json` refuses `NaN` by design — so every refused result was unpersistable through `to_dict` alone. **The reader was always right; only the writer disagreed.** Its claimed Family B blast radius was disproved by measurement: no Family B type is persistence-reachable. | 3a | **this commit** | `to_dict()` on a refused result returns `value: None` and round-trips |
| **CI-q** | `compare_models` scored several models against one shared cohort and could not prove it: each model built its own population, so the comparison's entire premise was true in fact and unrecorded. **Its predicate was itself defective** — a text scan matching four docstrings and an unrelated function, which would have reported the item open forever. | 3b-0 | **this commit** | `compare_models` passes `population=` to every evaluation, parsed from its source |
| **CI-u** | The FLAT report surface could not represent absence. On a single-class cohort `auroc` and the receiver-operating-characteristic curve were `NaN` and strict JSON refused them, so a scientifically valid evaluation over a degenerate cohort produced **no artifact at all** — three of five measured cohorts. Staged u-1 unify the writers, u-2 the absence vocabulary, u-3 wire it with schema 4. | CI-p investigation | **this commit** | a single-class report serialises, and every absent scalar is declared |
| **CI-r** | The frozen report oracle is blind to interval certification — both captured fields are `False` in all ten cohorts because the capture used `n_bootstrap=0`. **That premise is accurate; its implication was not.** MEASURED 2026-07-29: forcing the success-path assignment to `False` FAILS `test_evaluator_produces_a_certified_interval_when_genes_are_present`. The property is covered directly and in both directions by `test_bootstrap_reconciliation`. Its predicate described a FROZEN FIXTURE nobody intends to change rather than a coverage gap. | 3b-1a | **this commit** | the bootstrap suite asserts interval certification positively |
| **CI-t** | The report path called scikit-learn functions on raw `(y, p)` with no validation. Their behaviour disagreed across three defect classes, and the operating-point sweep shipped a wrong decision threshold because `NaN >= t` is `False`. **Discharged prematurely on a HAND COUNT of ten call sites; a parsed enumeration found twelve.** The subgroup breakdown was reachable only with `meta` supplied, which no corrupt-model test did. | CI-q investigation | **2026-07-28, corrected same day** | a parsed enumeration finds no metric call whose enclosing function lacks a validator governing a branch |

## Unverifiable from the repository alone

These are recorded so they are not lost, but no predicate can decide them here.
They require the continuous-integration environment or a training cohort.

| id | statement | raised | why no predicate |
|---|---|---|---|
| **CI-i** | Five Monte Carlo Dropout tests are unconditionally skipped pending a Run 15 cohort. | pre-2a | needs the cohort; a skip count alone cannot distinguish these from other skips |
| **CI-j** | Four tests have no continuous-integration coverage because they are Windows-only. | pre-2a | needs the continuous-integration matrix, not the working tree |
| **CI-a** | Certification of the canonical seam remains open. | pre-2a | scope predates this register; see `docs/ROADMAP.md` line 1705 |

---

## Rules

**A status change is made HERE first.** A roadmap delta may describe the change;
this register decides it.

**Every open item carries a predicate that can fail.** An item with no runnable
check is in the unverifiable table, where its inability to be checked is explicit
rather than implied.

**Discharge is proved, not asserted.** An item moves to Discharged only when its
predicate reports the condition gone, and the predicate stays in the suite
afterwards so a regression re-opens it as a failure.

---

*Established 2026-07-28. Amend by editing this file and its test together.*
