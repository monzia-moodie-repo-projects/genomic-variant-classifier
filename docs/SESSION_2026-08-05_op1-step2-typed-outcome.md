# SESSION 2026-08-05 — OP-1 step 2: the typed operating-point outcome

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = 0030544`. Full suite 4219 passed, 6 skipped, 0 failed; 4225 collected.**
**Ratchet 4203 → 4225 (+22), measured by the installer. STILL NO SELECTOR — nothing imports the outcome.**

Companion documents:
`SESSION_2026-08-04_op1-step1-exact-sweep.md`,
`SESSION_2026-08-04_thr1b-evaluation-sweep-source.md`.

---

## 1. The register that opened with twelve defects now has one

| defect | closed by |
|---|---|
| D7, D8 | OP-0 — `d4b4259` |
| D1, D9, D10, D11 | step 1 — `0030544` |
| **D2-D5, D6** | **this step** |
| **D12** | **open** — closes in step 4 |

D12 is the undeclared tie-break: `if diff < best_diff` is strict, so with an
ascending grid ties go to the lower, more liberal threshold. Defensible,
undocumented, and invisible in the result. It closes when the selector declares
and persists its rule.

---

## 2. D2-D5 and D6 close by construction, not by convention

`MetricResult.__post_init__` enforces — and its docstring states — that

```
a non-OK status REQUIRES a nonempty reason;
a non-OK status carries value NaN;
an OK status carries a finite value and no reason.
```

So there is **no way to store a fabricated `0.0` where a refusal belongs.** The
constructor refuses it, **however the code is later edited.** That is a
structural guarantee rather than a careful branch someone could simplify away.

### 2.1 D5 is broken structurally

The legacy form computed F1 from a positive predictive value that might itself be
a fabricated `0.0` — **a second fabrication derived from the first**, with no
record that either occurred.

`2TP/(2TP+FP+FN)` is the same quantity computed from **counts** and cannot
inherit anything. Measured on a cohort with nothing flagged: the positive
predictive value **refuses** while F1 is `ok` at 0.0, and a test pins exactly
that pairing.

### 2.2 D6 closes because nothing is rounded at storage

The legacy selectors stored `round(value, 4)` and computed F1 from **unrounded**
inputs, so a stored F1 could not be recomputed from the stored values it was
supposedly derived from.

Measured here:

```
stored     0.7551020408163265
recomputed 0.7551020408163266
```

Agreeing to 1e-12, differing only in the last bit by floating-point
associativity. `round_for_display` returns a **new mapping** and leaves the
record alone — rounding is a presentation concern and stays one.

---

## 3. The vocabulary is the registry's, deliberately

Step 3's Oracle C1 exists to prove this count path reproduces the registry's
status semantics, and it can only do that if the two speak **one** vocabulary.

Every reason string here is one the registry already emits:

```
empty_predicted_positive_set        positive predictive value, TP+FP = 0
empty_predicted_negative_set        negative predictive value, TN+FN = 0
positive_class_support_required     sensitivity, TP+FN = 0
negative_class_support_required     specificity, TN+FP = 0
zero_f1_denominator                 F1, 2TP+FP+FN = 0
zero_confusion_margin               Matthews, any vanishing margin
```

And the statuses follow REG-2 (`afa7a90`), which measured the registry's own
convention across 24 descriptors and 6 cohort shapes: a vanishing **denominator**
is `UNDEFINED`; an absent reference **class** is `INSUFFICIENT_SUPPORT`.

A private vocabulary would make the oracle compare two dialects and report every
difference as a defect.

---

## 4. Prevalence is absent, and a test asserts its absence

Decision 3 (2026-08-04): prevalence is a **population** statistic, not a
threshold-derived one. It does not depend on the threshold, on predicted-positive
membership, on the policy, or on the sweep.

Its canonical value comes from `registry.compute(by_name("prevalence"), ctx)`.
Computing a second one here would invent two prevalences that agree until a
population bug makes them diverge.

---

## 5. Four reads before a line was written, and each changed the design

**The layering.** `capabilities.py` and `population.py` import only stdlib and
NumPy — both pure leaves. So importing `MetricResult` **cannot close a cycle**,
and the typed outcome may sit beside the sweep. Had `capabilities.py` imported
`thresholds.py`, the outcome would have needed plain fields instead, and that is
a different design.

**`MetricResult`'s enforced invariants.** D2-D5 close **by construction** rather
than by branches I would otherwise have written carefully — and carefully-written
branches are what a later edit removes.

**`MetricResult`'s generic-by-decision comment.** Its own source records that 35
of its 53 construction sites are embedding-space probes for which population
scope has no epidemiological meaning, and that forcing the field there *"would
make the contract ceremonial exactly where it cannot be checked."* So population
identity is carried **once, on the outcome**, in the `MetricMetadataKey`
vocabulary — rather than re-opening a decision from the wrong layer.

**`_is_finite` existing.** Imported rather than reimplemented. Importing an
underscore-prefixed name across modules is a smell; a **second implementation of
the predicate whose agreement governs serialisation** is the SWEEP-1 shape
documented one commit earlier. The duplication is the worse of the two, and the
choice is recorded rather than left to look careless.

**That is EXTRACT-1 applied for a second consecutive step** — and this time it
*changed* a design decision rather than confirming safety.

---

## 6. A tenth prose-matching post-check, recorded as recurrence

Three expectations in the installer were wrong:

| token | expected | measured | why |
|---|---|---|---|
| `MetricResult.not_ok` | 2 | 3 | a docstring names it |
| `round(` | 1 | 2 | a comment describes the legacy defect |
| `prevalence` | 2 | 1 | it appears in one docstring, not two |

**Every expectation was right about the code and wrong about the text.** The
`code_only` tokenising helper existed in earlier installers and this one did not
carry it — that is the recurrence, not the arithmetic. It now does, and the
expectations are stated against code counts.

---

## 7. No sabotage line

Nothing existing changed, and the guarantees are **constructor-enforced**: a
mutation that removed a careful branch would still be refused by
`MetricResult.__post_init__`. That is a stronger statement than any mutation
score, because it holds for mutations nobody thought to write.

---

## 8. Acceptance

| item | value |
|---|---|
| full suite | 4219 passed, 6 skipped, 0 failed (17m07s, and 16m19s under the armed gate) |
| collected | 4225, measured by the installer |
| ratchet | 4203 → 4225 (+22) |
| `test_readme_claims` | 10 passed |
| tests | 22, none parametrised — functions and cases coincide |
| nothing else moved | nothing imports the outcome |

The structural import guarantee survived a **second** relative import:
`test_thresholds_imports_nothing_it_must_not` still passes, because
`capabilities.py` is neither registry, nor metrics, nor scikit-learn — and is
itself a pure leaf.

---

## 9. Next

**Step 3 — the two oracles.** C1 compares this count path against
`registry.compute` on identical cohorts: status, reason, value with exact NaN
semantics. C2 compares full `MetricResult` identity at the canonical threshold,
including metadata, support counts, population scope and fingerprint,
certification eligibility and the serialised form.

The vocabulary decision made deliberately in §3 gets **tested** there rather than
asserted here — which is the point of having made it deliberately.

Then step 4 (the selector, Objective A, closing D12), step 5 (the shadow
comparison), and step 6 (the cutover, which must reckon with **GUARD-1**).

Nineteen follow-ups carried, unchanged by this step.
