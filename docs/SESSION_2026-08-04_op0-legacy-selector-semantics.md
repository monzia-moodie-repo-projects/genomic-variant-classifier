# SESSION 2026-08-04 — OP-0: characterize and clarify the legacy operating-point selector

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = 90ed465`. Full suite 4155 passed, 6 skipped, 0 failed; 4161 collected.**
**Ratchet 4157 → 4161 (+4). NO SABOTAGE LINE, deliberately — see §6.**

Companion documents:
`SESSION_2026-08-01_pop1a-label-eligible-population.md`,
`SESSION_2026-08-02_pre1-preflight-contract-gate.md`,
`SESSION_2026-08-03_pop1b-population-surface.md`,
`SESSION_2026-08-03_reg1-metadata-ownership.md`.

---

## 1. What OP-0 is, and why it is not the rewrite

OP-1 will replace the operating-point subsystem with a typed, population-bearing
one. **OP-0 changes no arithmetic and no schema.** It makes the legacy contract
truthful and frozen, so that OP-1's shadow comparison is scientifically
interpretable.

Without it, an OP-1 difference could be read five ways: a sweep defect, a
tie-breaking defect, a threshold-order defect, a regression in positive
predictive value, or the intended policy change. With it, a difference is a
**declared policy change** and nothing else.

A rename-only commit would have been weaker. It would have removed the naming
defect and left the more consequential ambiguity — a docstring claiming a policy
the code does not implement — untouched.

---

## 2. The docstring claimed a policy the code does not implement

It read *"Highest-sensitivity threshold where PPV ≥ min_ppv."* The code walks
candidates from conservative to permissive and **breaks at the first violation**,
returning the last preceding candidate. Those differ whenever positive predictive
value is non-monotone in the threshold — which it is.

**Measured 2026-08-04**, `y = [1, 0, 1]`, `p = [0.9, 0.8, 0.7]`, `min_ppv = 0.60`:

```
t=0.90   ppv=1.0000   sensitivity=0.5000   FEASIBLE        <- selected
t=0.80   ppv=0.5000   sensitivity=0.5000   violates floor  <- the break fires
t=0.70   ppv=0.6667   sensitivity=1.0000   FEASIBLE, UNREACHABLE
```

The selected point has **half the sensitivity** of a candidate satisfying the
same floor. The legacy rule is legitimate — the most permissive threshold that
has never dropped below the floor — but it is not what the docstring said.

| | |
|---|---|
| legacy `_find_high_ppv_point` | **Objective B**, conservative prefix |
| OP-1's policy for `at_high_ppv` | **Objective A**, pointwise floor, maximise sensitivity over all candidates |

### 2.1 Neither name existed anywhere in the repository

Measured by the OP-1 preflight: `Objective A` and `Objective B` returned **zero**
occurrences across source and documentation. The decision adopting Objective A
was taken on **2026-08-01** and had never been written where the work happens.

That is a new member of a familiar family. Not a stored number that rotted, but
an **adopted decision recorded only in the conversation that took it**. The
roadmap's own standing rule is that a claim which cannot be followed is an
assertion wearing evidence's clothes; a decision that cannot be found is the same
thing one level up.

### 2.2 The break is deliberately not removed

Removing it would switch from B toward A **before** the typed subsystem, the
exact sweep, population identity, refusal semantics and the shadow comparison
exist to receive it. The resulting difference would be uninterpretable — which is
precisely what OP-0 exists to prevent.

---

## 3. A naming defect, repaired

```python
_find_high_ppv_point:   n_neg = tp + fp   # n_flagged
_find_operating_point:  n_neg = fp + tn   # genuinely the negatives
```

The same identifier held **two different quantities in two adjacent functions**.
The arithmetic was correct — every downstream use read it as the flagged count,
and the trailing comment admitted as much — and the name said the opposite. That
is one edit away from becoming numerical.

Renamed to `n_flagged`, which is already the public vocabulary: it is a field on
`OperatingPoint`. **The sibling is untouched**, and a test asserts it stays that
way so the rule cannot be satisfied by over-renaming.

---

## 4. The subsystem about to be rewritten has almost no coverage

Measured across the whole test tree on 2026-08-04:

- **`_find_high_ppv_point` was exercised by nothing.** The only match for either
  finder was a *comment* in `test_computation_path_guards.py:26` mentioning
  `_find_operating_point` in prose.
- The operating points themselves were asserted on **seven lines across four
  files**.

A shadow comparison against an unexercised selector proves very little. OP-1
should be read with that in mind, and it is recorded as **OPCOV-1**.

---

## 5. Four tests, in a new module

`tests/unit/test_operating_point_semantics.py`. A new module because both
candidate homes are about something else — `test_evaluator_meta.py` locks the
`meta=` breakdown contract and Run 16's silent-empty trap; `test_evaluator_phase5.py`
is Phase 5 work. Filing a selector characterization into either would be filing
**by accident rather than by subject**, and this project has already paid for one
misfiled entry.

| test | kind |
|---|---|
| the legacy contract, frozen on eleven measured values | behavioural |
| the skipped candidate is feasible **and** better, and is not returned | behavioural |
| `tp + fp` is never stored as `n_neg` | **structural** |
| the sibling's `n_neg = fp + tn` stays | **structural** |

The second exists so that *"Objective B is worse here"* is a check rather than a
claim in a docstring.

Conventions were taken from the suite, not from a probe:
`ClinicalEvaluator(n_bootstrap=0, random_state=42)`, the form used at twenty
sites. The OP-0 probe had used `random_state=0` — immaterial here, but adopting
the suite's form is the discipline that caught `_two_class_context()` not
existing in this repository on 2026-08-03.

---

## 6. No sabotage line, and the absence is explained

Every recent ratchet entry carries one. This one does not.

**OP-0 changes no arithmetic, so there is nothing behavioural to mutate.** The
installer proved that structurally: the token stream of `_find_high_ppv_point`
was compared before and after with the renamed identifier normalised, and came to
**418 tokens on both sides, identical**. A text diff cannot show that, because
the text also gains comments and a rewritten docstring.

### 6.1 Both structural guards were falsified individually

Reverting the rename in place:

```
AssertionError: line(s) [76] of _find_high_ppv_point store tp + fp under the
name n_neg ...
1 failed, 3 passed
```

**The other three passed unchanged**, which is the part that matters: the
structural guard is not entangled with behaviour it should not be watching.
Restoration was verified by digest, and the module returned to 4 passed.

That is stronger evidence for a commit of this kind than an aggregate mutation
count, and it is the fifth repair this week proven by making it fail first.

---

## 7. A tool defect found and fixed in passing

The `code_only` helper used by these installers' post-checks filtered
`tokenize.STRING`. **On Python 3.12, f-strings do not tokenise as `STRING`** —
they produce `FSTRING_START`, `FSTRING_MIDDLE` and `FSTRING_END`.

So f-string prose passed straight through a filter whose entire purpose was
excluding prose. **Every earlier use in this sequence had the same blind spot**;
the counts that passed did so because no f-string happened to contain the token,
not because the filter worked.

Found by measuring where the occurrences actually were rather than reasoning
about the discrepancy a third time. Fixed with `hasattr` guards so it stays
correct on older interpreters.

---

## 8. Acceptance

| item | value |
|---|---|
| full suite | 4155 passed, 6 skipped, 0 failed (18m21s, and 15m11s under the armed gate) |
| collected | 4161 |
| ratchet | 4157 → 4161 (+4) |
| `test_readme_claims` | 10 passed |
| arithmetic | **unchanged** — 418 tokens before, 418 after |

---

## 9. Follow-ups — fourteen

| id | item |
|---|---|
| **OPCOV-1** | *new.* `_find_high_ppv_point` exercised by nothing before OP-0; operating points asserted on seven lines across four files. OP-1's shadow comparison rests on thin ground. |
| **GITIGNORE-1** | *new, cosmetic.* `*.bak_*` appears three times in `.gitignore` (lines 155, 158, 262). Harmless duplication in a file whose other entries are single. |
| STRUCT-1 | structural guards have now been used twice — REG-1's derivation check and OP-0's naming check. Other invariants stated only in comments could be gated the same way. |
| POP-1b-M03 | no test distinguishes the source distance from the parent distance |
| POP-1b-M07 | nothing asserts on `print_report` output |
| ZERO-1 | 24 dead-connector defaults still zero — is the allowlist itself stale? |
| INF-1 | an infinite reference label is pooled with NaN as *withheld* |
| ABS-1 | the ranking channel's refusal reported as `undefined_on_cohort` |
| DEAD-1 | ~40 lines of dead absence computation in `evaluate` |
| DEAD-3 | `_assert_absence_biconditional` computes `observed_curves` twice |
| PRE-2 | section 5's PASS line swallows the KAN banner |
| LINT-1 | no lint gate anywhere |
| F821-1 | 18 undefined names; 9 need assessment |
| CMP-1 | `ModelComparison` carries a fingerprint with no scope beside it |

---

## 10. Next

**OP-1** — the typed, population-bearing operating-point subsystem. The legacy
contract is now frozen and named, so a shadow difference is a declared policy
change. Its own defect register from 2026-08-01 must still be re-read against the
current code, since it predates POP-1a, POP-1b and REG-1.

Then the drift monitor, whose red is roadmap 6.20's fix working.
