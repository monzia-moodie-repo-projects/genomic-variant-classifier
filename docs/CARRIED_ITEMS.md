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
| **CI-m** | `metrics.evaluate` is a survivor-filtering path: it drops non-finite rows through `_clean` rather than refusing, so a caller cannot tell how many observations a number describes. | 2a | `metrics.evaluate` exists **and** `_clean` is called from it |
| **CI-n** | `cohort_version` is a weak provenance identity. Most call sites pass a generic `"v2"`, so two different cohorts can share it; the ordered variant-identifier sequence is what actually distinguishes them. | 2a-1 | `_derive_population_source_id` still accepts `cohort_version` as a free string |
| **CI-p** | `MetricResult.to_dict` emits raw `NaN` while `from_dict` documents reading `null` back as `NaN`, and `dump_strict_json` refuses `NaN` by design. Every refused result is unpersistable through `to_dict` alone. Normalised at the report layer in 3a; five Family B call sites remain. | 3a | `to_dict()` on an UNDEFINED result returns a non-finite `value` |
| **CI-r** | The frozen report oracle is blind to interval certification. Both captured certification fields are `False` in all ten cohorts because the capture used `n_bootstrap=0`, so a defect forcing them `False` is invisible to it. | 3b-1a | the oracle's `auroc_ci_certification_eligible` has exactly one distinct value |
| **CI-s** | Deferred imports in `registry.py` are a load-bearing contract, not a style. The module must import without scikit-learn, so any module-scope construction must bind kernels by name and resolve at call time. | 3b-2 | *inverted* — the predicate asserts the contract HOLDS; this row records the rule, and a violation is a failure |

## Discharged

| id | statement | raised | discharged | verification predicate |
|---|---|---|---|---|
| **CI-k** | Interior-edge agreement coverage between the two calibration implementations. | 2a-1 | **2b-1** | the binning suite exercises every interior edge |
| **CI-l** | The transitional label mask `select_finite_reference_labels`. | 2a | **2a-1** | the symbol exists nowhere in `src/` as code |
| **CI-o** | The evaluator abstract-syntax-tree guard, deferred until the evaluator was actually retired rather than merely observed. | 2b-2 | **3b-2** | `test_computation_path_guards.py` exists and inspects the report path |
| **CI-q** | `compare_models` scored several models against one shared cohort and could not prove it: each model built its own population, so the comparison's entire premise was true in fact and unrecorded. **Its predicate was itself defective** — a text scan matching four docstrings and an unrelated function, which would have reported the item open forever. | 3b-0 | **this commit** | `compare_models` passes `population=` to every evaluation, parsed from its source |
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
