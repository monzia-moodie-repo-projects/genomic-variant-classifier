# SESSION 2026-08-04 — THR-1a: the threshold vocabulary moves to a bottom layer

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = afa7a90`. Full suite 4165 passed, 6 skipped, 0 failed; 4171 collected.**
**Ratchet 4165 → 4171 (+6). ZERO MOVEMENT: no behaviour, no value, no pre-existing test outcome changed.**

Companion documents:
`SESSION_2026-08-04_op0-legacy-selector-semantics.md`,
`SESSION_2026-08-04_reg2-denominator-status.md`.

---

## 1. Why a bottom layer, and why not the two obvious homes

OP-1's exact threshold sweep must describe each swept candidate with a
`ThresholdParameters`. That single requirement rules out both obvious places.

**A sweep in a module that imports `registry.py` reverses the layering.** Future
registry count-applicability code could then not import the sweep without a
cycle.

**`metrics.py` is worse.** It imports scikit-learn **at module level**, and
`evaluation/__init__.py` forbids importing it for exactly that reason — commit
`015ff94` restored the file but added `from ... import metrics`, pulling
scikit-learn eagerly and breaking the Phase-5 contract, *"trading one silent
failure for another."* Two tests lock it. A reusable algorithm placed there could
never be depended on from the package root.

So the vocabulary moves **down**:

```
capabilities.py / population.py
             |
        thresholds.py        ThresholdOperator, ThresholdSource,
             |               ThresholdParameters
    +--------+--------+
registry.py       metrics.py
```

`thresholds.py` imports `numpy`, `dataclasses`, `enum` and `logging` — **neither
registry, nor metrics, nor scikit-learn** — and a **structural** test asserts
that over the source. A runtime check would pass merely because scikit-learn
happens to be installed.

---

## 2. Re-exported by identity, not by equality

`ThresholdParameters`'s own docstring states the constraint this move could not
break:

> One instance is shared by a descriptor, its kernel adapter and its
> applicability predicate, and that sharing is asserted **BY IDENTITY** at import
> time. Three copies of a threshold that merely happen to be equal today is how a
> threshold comes to differ tomorrow.

A re-export producing a **distinct** class object would leave every `isinstance`
check comparing against a different type. `registry.py` asserts identity at
import, so the import itself would have failed rather than the suite — which is
the good failure mode, but only because someone had written that assertion.

The classes moved **verbatim**. The post-check compared
`before.replace(MOVED, "")` against `after.replace(REPLACEMENT, "")` to prove
everything outside the block was byte-identical — a property that does not depend
on how `difflib` groups hunks.

---

## 3. A sequencing failure of mine, recorded without softening

A zero-movement extraction needs an inventory **before** the move: every import
of the classes, every attribute lookup through `registry`, any package-level
export, any `__all__`, any use of `__module__`/pickle/qualified names, any
monkeypatch, any source-inspection test.

**Seven checks. I ran none before the move and all of them afterwards, when
prompted.**

Nothing broke, and the suite proves it. But *"nothing broke"* is the **outcome**,
not the discipline. An inventory exists so that a move is **known** safe rather
than **found** safe, and running it afterwards converts a check into a
**postmortem**.

### 3.1 Three of the seven were clean by arrangement, not by design

**`registry.__all__`** holds 12 entries and **none of the three** is among them —
measured from the live module rather than read from a source window. Unchanged by
the move, because they were never declared there, and all three remain reachable
as attributes, which is what every existing import uses.

**Every pickle and joblib reference** across `src`, `tests` and `scripts` is a
**model** artifact — ensembles, scalers, pipelines. None names a threshold type,
so no historical artifact refers to a class whose module moved.

**Three tests inspect `registry` source** — `test_metric_registry.py:437` and
`:907`, `test_prediction_input_contract.py:294`. All passed: none expected the
threshold definitions to be there.

Each of those could have been otherwise. They were fine because of how the code
happened to be arranged.

### 3.2 The fourth had genuinely changed, and was unrecorded

`ThresholdParameters.__module__` moved from
`genomic_variant_classifier.evaluation.registry` to
`...evaluation.thresholds`. Intended — it is the point of the extraction — and
written down nowhere.

It affects `repr`, generated documentation, pickled output bytes and type-name
provenance. **An unrecorded change to `__module__` is an unrecorded change to
what an artifact says about itself.**

### 3.3 And this project already had the precedent

`tests/unit/test_metric_result_relocation.py:60-63`:

```python
def test_capabilities_owns_metric_result():
    assert MetricResult.__module__ == (
        "genomic_variant_classifier.evaluation.capabilities"
    ), "MetricResult must be DEFINED in the vocabulary layer, not re-exported into it"
```

`MetricResult` was relocated into `capabilities.py`, re-exported from
`clustering_metrics`, and the tests pin **both** halves — where it is defined,
and that consumers resolve the same object.

**THR-1a wrote the second half and omitted the first.** The completing test uses
the precedent's wording, because it is the same claim about a different layer,
and it was falsified against `...registry` and `...capabilities` before shipping.

The precedent sat in the repository the whole time, unconsulted until prompted.

---

## 4. Six tests

| test | what it pins |
|---|---|
| identity of all three through the re-export | `registry.X is thresholds.X` |
| the shared-instance invariant | `isinstance` through either name |
| serialisation unchanged | `to_mapping()` byte-identical |
| every refusal moved with the class | four validation paths |
| `thresholds.py` imports nothing forbidden | **structural**, over source |
| **`__module__` ownership** | the gap §3.2 found |

---

## 5. No sabotage line

Nothing behavioural changed, so nothing behavioural can be mutated. The identity
and ownership assertions are the mutation-equivalent, and **both were
exercised** — identity by construction, and `__module__` by confirming the
assertion fails against `...registry` and `...capabilities`.

---

## 6. Acceptance

| item | value |
|---|---|
| full suite | 4165 passed, 6 skipped, 0 failed (18m06s, and 15m37s under the armed gate) |
| collected | 4171 |
| ratchet | 4165 → 4171 (+6) |
| `test_readme_claims` | 10 passed |
| pre-existing tests | **every one passed unchanged** — the whole acceptance criterion |

The old ratchet value and today's pass count are both 4,165. Different
quantities; the coincidence is arithmetic accident and the entry says so.

---

## 7. Not included, deliberately

**THR-1b** — adding `ThresholdSource.EVALUATION_SWEEP`. An additive
persisted-vocabulary change has a different failure mode from an extraction, and
combining them would give a regression two possible origins.

**Step 1** — the array-backed sweep. Its algorithm is already verified: zero
mismatches against the brute-force definition across ten cohorts, all eleven
refusal paths raising, and a 1000-point grid shown to reach only **2 of 5**
achievable operating points on scores spaced at 0.0001. What changes is where it
lives, what it returns, and that it carries the `EvaluationPopulation` from the
first type.

---

## 8. Follow-ups — eighteen

| id | item |
|---|---|
| **EXTRACT-1** | *new.* A zero-movement extraction has a pre-move inventory; skipping it converts a check into a postmortem |
| SWEEP-1 | two equivalent tie-aware sweeps in `metrics.py` (322, 1781), agreeing across eleven cohorts, with nothing asserting they agree |
| REG-2-b | `_requires_interior_specificity` returns `INSUFFICIENT_SUPPORT` with reason `specificity_undefined` |
| ICI-1 | `integrated_calibration_index` declared applicable, then returns non-finite |
| F1-1 | `f1` returns `ok` with 0.0 from an undefined positive predictive value |
| OPCOV-1 | the operating-point selectors have almost no coverage |
| GITIGNORE-1 | `*.bak_*` appears three times in `.gitignore` |
| STRUCT-1 | structural guards now used four times, on four defect classes |
| POP-1b-M03 | no test distinguishes the source distance from the parent distance |
| POP-1b-M07 | nothing asserts on `print_report` output |
| ZERO-1 | 24 dead-connector defaults still zero |
| INF-1 | an infinite reference label is pooled with NaN as *withheld* |
| ABS-1 | the ranking channel's refusal reported as `undefined_on_cohort` |
| DEAD-1 | ~40 lines of dead absence computation in `evaluate` |
| DEAD-3 | `_assert_absence_biconditional` computes `observed_curves` twice |
| PRE-2 | section 5's PASS line swallows the KAN banner |
| LINT-1 | no lint gate anywhere |
| F821-1 | 18 undefined names; 9 need assessment |

`CMP-1` is carried in the roadmap and omitted here only to keep this table to the
items this session touched or created.

---

## 9. Next

**THR-1b**, then **step 1**: `ConfusionCounts`, the exact sweep, and the lazy
candidate view — owned immutable arrays, no object-per-candidate storage, linear
byte scaling asserted, and selection working on arrays rather than iterating
Python objects.

Seven register defects remain open: D1, D2-D5, D6, D9, D10, D11, D12. One sort
with cumulative sums closes D1, D9, D10 and D11 together.
