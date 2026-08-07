# OP-1 BUILD SPECIFICATION — CANONICAL, reconciled 2026-08-07

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Reconciled 2026-08-07, against `HEAD = origin/main = d208240`, tree clean,
ratchet 4353.**
**Supersedes nothing wholesale. Reconciles two source specifications, both
committed alongside this file.**

---

## 0. Authority

Two build specifications existed, neither acknowledging the other. Measured
2026-08-07: the 2026-08-04 document contains no occurrence of `supersede`,
`replaces`, `2026-07-31` or `OP-2`, and the 2026-07-31 document does not
anticipate a successor.

| concern | authority |
|---|---|
| architecture and implementation layout | `OP1_BUILD_SPEC_2026-08-04.md`, plus steps 1–4 as built |
| step-5 comparison semantics | `OP1_BUILD_SPEC_2026-07-31.md`, **STEP K**, retained normatively and reproduced in section 3 |
| sequence and closure | this document, section 4 |

**The 2026-07-31 module layout is superseded**, and that is a measurement, not
a preference. None of the following exists in `src/` as of 2026-08-07:
`operating_point.py`, `_finalize_metric_result`, `CountMetricSpecification`,
`confusion_counts_at_threshold`, `FIXED_GRID`, `select_operating_point`. Five
of five symbols absent, one of two modules absent. Only `legacy_projection.py`
survives, from a different lineage.

**STEP K is not superseded.** It is the only specification of the shadow
comparison anywhere in the project, and the 2026-08-04 document names step 5
without defining it.

---

## 1. What was built, steps 1 through 4

All four are complete and pushed. The twelve-defect register opened 2026-08-01
is fully closed.

| step | commit | what |
|---|---|---|
| 1 | `0030544` | `ExactThresholdSweep` and `sweep_thresholds` — every achievable operating point as owned immutable arrays, O(n log n). Closes D1, D9, D10, D11. |
| 2 | — | `OperatingPointOutcome` and `metrics_from_counts` — typed refusals, no fabricated `0.0`, nothing rounded at storage. Closes D2–D5 and D6. |
| 3a | — | Oracle C1: the count path reproduces the registry's status, reason and value on six shared estimands. |
| 3b | — | Oracle C2's difference, measured rather than assumed. |
| 3c | `58929e9`, `1be72e4` | `_registry_metadata_prefix` — one support authority. |
| 4 | `3ddf617` | `select_nearest_sensitivity_target` and `select_max_sensitivity_at_ppv_floor`. Closes D12. |

Everything lives in `src/genomic_variant_classifier/evaluation/thresholds.py`,
organised as step-numbered banner sections. Nothing is wired: `evaluator.py`
imports neither selector, by design.

---

## 2. The two objectives, named

```
Objective A:  max sensitivity(t)  subject to  PPV(t) >= floor
Objective B:  the most permissive t before the first floor violation
```

OP-0 (`d4b4259`, 2026-08-04) froze B and named it. OP-1 implements A for
`at_high_ppv`. Measured on `y = [1, 0, 1]`, `p = [0.9, 0.8, 0.7]`, floor 0.60,
they differ by DOUBLE the sensitivity: B selects `t=0.90` at sensitivity 0.5,
while `t=0.70` satisfies the same floor at sensitivity 1.0 and is unreachable
because the break has already fired.

---

## 3. STEP K — the shadow comparison, retained normatively

Reproduced from `OP1_BUILD_SPEC_2026-07-31.md` lines 486–520. This is the
authority for OP-1 step 5.

Classify every field comparison:

```
EQUAL
EXPECTED_EXACT_SWEEP_IMPROVEMENT      the exact sweep found a superior feasible point
EXPECTED_UNDEFINEDNESS_CORRECTION     a legacy 0.0 is a canonical UNDEFINED
UNEXPECTED_MOVEMENT                   must be EMPTY
```

**Do not require thresholds to match when decision sets match:**

```python
def same_decision_partition(first, second) -> bool:
    return first.counts == second.counts
```

Two numerically different thresholds can be decision-equivalent.

**Selection dominance:** for each legacy policy, the new candidate satisfies the
same constraint, the new objective is not worse, and any difference is
attributable to exact enumeration, explicit tolerance, burden minimisation or
undefinedness semantics.

**Create a FROZEN MOVEMENT SET from actual measurements before OP-2.**

**Evaluator shadow guard:** the legacy selector remains authoritative; the new
selector runs only in shadow; no report field is assigned from the new bundle.
**Invert in OP-2.**

**Operating-point source guard:** reject direct division involving TP/FP/FN/TN,
per-metric calls to `apply_decision_threshold`, scikit-learn confusion metrics,
and inline denominator guards. **Allow** cumulative count construction and the
vectorised candidate table.

### 3.1 Three refinements adopted 2026-08-07

**Assert by identity, never by count.** The precedent is the calibration shadow
comparison of 2026-07-28: *"before 3b-1a 6 mismatches; after 3b-1a exactly 2,
and exactly the right two… Asserted by IDENTITY, not count — the wrong four
could have vanished."*

**The movement set is a committed artifact**, not a test constant, keyed by
fixture and report field, carrying both commits. A changed movement set
requires deliberate review; it is never regenerated to make a test pass.

**The guard is on the source path, not on `apply_decision_threshold`.** GUARD-1
was discharged by falsification on 2026-08-06: both legacy selectors compute
`preds = (p >= t)` inline, `apply_decision_threshold` appears nowhere in
`evaluator.py`, and `thresholds.py` does not bind it. A guard patching that
function observes neither implementation. The real gap is OPCOV-1's.

---

## 4. Sequence, and where OP-1 ends

Ruling of 2026-08-07, resolving the disagreement between the two sources.

```
OP-1 step 5   the shadow comparison
              MEASUREMENT ONLY. No production authority changes.
              Produces the frozen movement set.
              OP-1 CLOSES HERE.

OP-2          authority inversion
              the three report fields cut over; schema bump
              consumes step 5's movement set as a PRECONDITION
```

The 2026-08-04 document called the cutover "step 6"; the 2026-07-31 document
said "invert in OP-2". **The separation is adopted**, because step 5 produces
the evidence that authority inversion consumes, and combining measurement with
cutover under one sequence would make the result feel preordained.

---

## 5. Naming — NAMING-1

"Step 5" has three referents in this project:

```
PHASE step 5        metric-stack wiring (DECISION_2026-07-25)
OP-1 step 5         the shadow comparison
hierarchy step 5    a tie-break stage (2026-07-31 spec, line 395)
```

**Always write the qualifier.** `OP-1 step 5`, `PHASE step 5`, never a bare
"step 5". This is the remedy `CARRIED_ITEMS.md` already applies to the
collision between root patterns (a)–(d) and carried items (a)–(s), resolved
there by mandating the `CI-` prefix.

---

## 6. OPCOV-1, the standing caution

From `OP1_BUILD_SPEC_2026-08-04.md`: *"a shadow comparison that agrees proves
agreement on the cases exercised — which are few."*

Measured 2026-08-04 and re-measured 2026-08-07, unchanged:
`_find_high_ppv_point` was exercised by nothing before OP-0; the operating
points are asserted on **seven lines across four files**
(`test_bootstrap_reconciliation.py`, `test_core.py`, `test_population_wiring.py`,
`test_report_input_gates.py`). `test_operating_point_semantics.py` adds four
tests, all semantic — none characterises returned values numerically.

OPCOV-1 now carries three claims under one identifier: the original count, an
"almost no coverage" paraphrase, and GUARD-1's absorbed threshold-provenance
gap. **It should be split before step 5 leans on it.**

---

*Reconciles `OP1_BUILD_SPEC_2026-07-31.md` (sha256 e44bd79bf31cbd8d…) and
`OP1_BUILD_SPEC_2026-08-04.md` (sha256 c0fcc58bfc38686f…), both committed
alongside this file, unchanged.*
