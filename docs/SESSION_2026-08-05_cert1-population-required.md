# SESSION 2026-08-05 — CERT-1: an OK outcome requires a population

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = 58bf4e1`. Full suite 4272 passed, 6 skipped, 0 failed; 4278 collected.**
**Ratchet 4271 → 4278 (+7). A CORRECTION TO SHIPPED CODE.**

Companion documents:
`SESSION_2026-08-05_op1-step3b-c2-measurement.md`,
`SESSION_2026-08-05_op1-step2-typed-outcome.md`.

---

## 1. The defect, and how it was found

`OperatingPointOutcome` landed in `f0db01f` two commits earlier with an
eligibility rule that **contradicted the registry's**.

```python
# registry, _certification_eligibility
if not ctx.population.is_attributed:
    return False, "unattributed_population"

# thresholds.py, OperatingPointOutcome
return self.is_ok and not self.certification_blockers
```

The outcome **never consulted the population**, so an OK outcome with
`population=None` and no blockers reported **eligible** — and a shipped test
positively asserted that.

The registry's rule is older and better argued: *a certified claim asserts
something about a NAMED set of rows*, and an unattributed population's
fingerprint is absent, so comparison with any other returns UNKNOWN rather than
SAME or DIFFERENT.

It also contradicted **step 2's own rationale**. Population identity was placed
on the outcome because *"n=980 beside n=980 says nothing about WHICH 980."* An
OK, certifiable outcome with no population at all is that argument abandoned.

**Found by reading `_certification_eligibility` while inventorying for step 3c.
Not by a failing test — the suite asserted the defect.**

---

## 2. Three states, now separated

| population | numerically valid | certifiable | behaviour |
|---|---|---|---|
| absent | no | no | **the constructor refuses** |
| unattributed | yes | no | a typed blocker is emitted |
| attributed | yes | depends on blockers | eligibility follows |

**The middle row is why this is not simply a stronger boolean.** Adding `and
self.population.is_attributed` would return the right answer and leave the
serialised artifact unable to explain why — a reader would see
`certification_eligible: false` beside an **empty blocker list**.

### 2.1 Declared and derived blockers stay separate

`certification_blockers` is what the caller declared;
`effective_certification_blockers` adds what the outcome's state implies.

Inserting the derived one via `object.__setattr__` would make constructor
arguments differ from the stored object — a caller passing `()` and reading back
a one-element tuple.

### 2.2 One vocabulary

The blocker code is `unattributed_population`, matching the registry's
machine-readable reason **character for character**.

---

## 3. The gate that should have existed

CERT-1 added a member to `OperatingPointCertificationBlocker` and found **no
completeness gate** guarding it — THR-1b built exactly that for
`ThresholdSource` one day earlier.

Three gates added: the member set asserted exactly, prose agreement in both
directions, and **the registry's own source read** for the reason string.

### 3.1 Falsified, with the prediction wrong in the informative direction

Renaming the value to `no_population_identity` gave **4 failed, 25 passed** —
two more than I predicted.

The extras are correct: CERT-1's own earlier tests already assert the string,
from the serialised artifact and directly. **Four independent statements of one
fact, each from a different angle**, and a rename should break all four.

**The prose gate correctly did not fire** — the member still exists and still has
prose, so its subject is untouched by a value change. That is the T3 property: a
gate failing alongside everything else measures the change rather than the
property.

---

## 4. Two defects in my own installers, found by review rather than by a gate

**Dead code.** The CERT-1 installer defined `VOCAB_OLD`, `VOCAB_NEW` and
`VOCAB_TESTS`, checked the file existed, and **never read, patched, wrote or
manifested it.** Its docstring claimed *"two edits to the test files"*; one test
file was edited. The discussion of updating a vocabulary gate described work the
installer never did — and the gate it should have built is §3's.

**A check weaker than its claim.** The fixture installer computed an abstract
syntax tree dump for every assertion and then compared only their **count**. I
described it as establishing that *"the repair cannot quietly alter what any test
claims."* **It did not**: every assertion could be rewritten and the count would
hold. It also missed `pytest.raises` context managers entirely.

The claim was verified afterwards by `git diff` — **zero removed assertion
lines**, every hit an addition from the new tests. So it was **true by luck
rather than by check**, and the distinction belongs in the record.

---

## 5. Six refusals today, all mine, all caught before writing

**Five were miscounts of my own code.** One was a search for text the source
never contains contiguously.

| | |
|---|---|
| `UNATTRIBUTED_POPULATION` | expected 4, measured 3 — the binding and its guard are one occurrence |
| `EvaluationPopulation` | expected `== 1`, measured 6 — a **presence** question asked with an **exactness** test |
| `population=None` | expected 3, measured 6 — I assumed six existed beforehand; nine did |
| `SHARED_ESTIMANDS` etc. | three counts, all under |
| `must carry an EvaluationPopulation` | **absent from the source** — split across adjacent literals, joined only at runtime |

**The common root: I write a check by describing what I believe is there, rather
than by measuring what is there.**

The remedies differ — tokenise for prose, **derive** for counts, assert structure
rather than strings for composed text — but the habit is single. Every gate
caught it; the gates should not have had to.

### 5.1 The remedy that generalises

The last post-check derives its expectation from the measured before-count —
`before - 3` — rather than hardcoding a number, and was verified across nine
cases.

**That is the same principle the finaliser design adopted**: derive from what is
there at the time, do not enumerate what you believe is there. `support()`
taught it, and it applies to post-checks as readily as to key sets.

---

## 6. Acceptance

| item | value |
|---|---|
| full suite | 4272 passed, 6 skipped, 0 failed (17m20s, and 18m15s under the armed gate) |
| collected | 4278, measured by the installer |
| ratchet | 4271 → 4278 (+7) — five tests replacing one, plus three gates |
| `test_readme_claims` | 10 passed |
| assertions through the fixture repair | **53 / 53 unchanged** |
| the oracle | 46 passed — CERT-1 leaves the sweep untouched |

`sweep_thresholds` still accepts `population=None`; only
`OperatingPointOutcome` requires one on an OK status. That distinction is
correct: a **sweep** enumerates candidates and may have no declared population; an
**outcome** claims performance about specific rows and may not.

---

## 7. Next

**Step 3c — the finaliser extraction**, specified against step 3b's two committed
reports and now against a corrected outcome type.

Its design constraint, established while inventorying: **the finaliser must
derive its key set from `support()` at call time, never enumerate it.**
`support()` returns four keys plus `N_CLUSTERS` **conditionally**, so a fixed
list of seven would be wrong for clustered contexts and no unclustered test would
catch it.

Then step 4 (the selector, Objective A, closing **D12** — the last of the
twelve), step 5 (the shadow comparison), and step 6 (the cutover, which must
reckon with **GUARD-1**).

Twenty follow-ups carried; C2-1 remains the newest.
