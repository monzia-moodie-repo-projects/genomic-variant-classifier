# SESSION 2026-08-03 — REG-1: metadata ownership differs between the refusal path and the success path

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = 41285af`. Full suite 4151 passed, 6 skipped, 0 failed; 4157 collected.**
**Ratchet 4152 → 4157 (+5). Sabotage: 6 mutations, 6 detected, 0 undetected, 0 anchor misses.**

Companion documents:
`SESSION_2026-08-01_pop1a-label-eligible-population.md`,
`SESSION_2026-08-02_pre1-preflight-contract-gate.md`,
`SESSION_2026-08-03_pop1b-population-surface.md`.

Committed evidence:
`docs/measurements/REG1_MUTATION_BASELINE_2026-08-03.txt` (preserved uncorrected),
`docs/measurements/REG1_MUTATION_R2_2026-08-03.txt`.

---

## 1. The defect

`registry.compute` merged descriptor-supplied metadata **two different ways**.

**The OK branch** (registry.py:1617-1628) derived a protected set, raised
`RegistryInvariantError` on overlap, and merged the verdict **first** so it lost
anyway.

**The refusal branch** (1558-1562) merged `verdict.metadata` **last**, so it
**won**, with **no collision check at all**.

So the same descriptor metadata that raised on the applicable path was silently
accepted on the refusal path.

### 1.1 The refusal branch was the worse one to leave open

`MetricContext.support` (registry.py:406-432) supplies `POPULATION_SCOPE`,
`POPULATION_FINGERPRINT`, `N_OBSERVATIONS`, `N_CLASSES_OBSERVED`, and
`N_CLUSTERS` when clusters are present. Its docstring says why they ride on
refusals too:

> Attached to EVERY result, refusals included… The cohort size behind a REFUSAL
> is equally informative: an `INSUFFICIENT_SUPPORT` on 3 rows and one on 300,000
> point at different problems.

The branch a descriptor could hijack was therefore **the one whose whole purpose
is stating what evidence base the refusal describes.** A refusal could claim a
membership fingerprint it never examined — and `capabilities.py:218-220` names
why that is the sharp case: *"n=980 beside n=980 says nothing about WHICH 980."*

### 1.2 Latent on 2026-08-01, live after POP-1b

Recorded on 2026-08-01 as latent: the population keys were thinly populated and
an overwrite had little to destroy. POP-1b (`00e180c`, 2026-08-02) made them
real, populated and load-bearing — schema version five exists so a reader can
tell a smaller cohort from a narrowed one.

**The defect did not change. Its consequences did.**

---

## 2. Version one was wrong, and the suite said so in twelve minutes

v1 derived **one** protected set and applied it to **both** branches. Applied
2026-08-03, it turned **29 tests red**:

```
verdict = Applicability(applicable=False, status=UNDEFINED,
                        reason='binary_class_support_required',
                        metadata={N_CLASSES_OBSERVED: 1, 'classes_observed': [1.0]})

RegistryInvariantError: auroc: applicability metadata attempted to set
registry-owned key(s) ['MetricMetadataKey.N_CLASSES_OBSERVED']
```

`auroc` is a **registered** descriptor. It refuses a single-class cohort and
reports `N_CLASSES_OBSERVED` as **the evidence for its refusal**. The guard
called that a violation.

**A guard that treats two paths as symmetric when they are not is a guard that
rejects correct behaviour.** v1's reasoning was right about the **derivation**
(one copy, never two) and wrong about the **ownership**.

---

## 3. The ownership line was measured, not argued

Probed 2026-08-03 across four cohort shapes — single-class positive, single-class
negative, two-class, degenerate probabilities — over every registered descriptor.
**27 refusals observed.**

| key | metrics claiming it on a refusal |
|---|---|
| `N_CLASSES_OBSERVED` | **7** — collides with `support()` |
| `REFERENCE_CLASS_SUPPORT` | 2 |
| `classes_observed` | 7 |
| `threshold` | 4 |
| `n_predicted_positive`, `n_reference_positive`, `specificity` | 2 each |

**Exactly one `support()` key** is claimed by descriptors on refusals, and seven
metrics claim it — a settled convention, not one descriptor's accident.
`N_OBSERVATIONS`, `POPULATION_FINGERPRINT` and `POPULATION_SCOPE` were claimed by
**none**.

I had listed `N_OBSERVATIONS` as registry-owned **by argument** before the probe.
The probe **confirmed** it — which is not the same as having been right, and had
any descriptor set it, v2 would have failed exactly as v1 did.

### 3.1 The contract, enforced

| key | refusal path | success path |
|---|---|---|
| `METRIC_NAME` | registry-owned | registry-owned |
| `POPULATION_SCOPE` | registry-owned | registry-owned |
| `POPULATION_FINGERPRINT` | registry-owned | registry-owned |
| `N_OBSERVATIONS` | registry-owned | registry-owned |
| `N_CLASSES_OBSERVED` | **descriptor-owned evidence** | **registry-owned** |
| `CERTIFICATION_*` | not descriptor-supplied | registry-owned |

---

## 4. Two constraints the code itself imposed, and both ruled out the easy fix

**Do not reorder the merge.** registry.py:1605-1613, verbatim:

> **PROTECTED KEYS ARE REJECTED, NOT SHADOWED.** Merge ORDER would also prevent a
> descriptor from overwriting registry-owned keys, but **silently**: the
> descriptor's value would simply vanish, and a descriptor author who believed
> they were setting the population scope would get **no signal at all**.

Neither branch's merge order was changed.

**Do not copy the derivation.** Lines 1614-1616 say the set is derived *"so a
future key added to `support()` is protected the moment it exists rather than the
moment somebody remembers to add it here."*

One helper, called from both branches, with the protected set as a **parameter**.
Only the exception is named, and only the refusal path subtracts it. The OK
path's behaviour is byte-identical.

---

## 5. Two mutation rounds, and the first is preserved uncorrected

### 5.1 The baseline: 6 mutations, 4 detected, 2 undetected

`docs/measurements/REG1_MUTATION_BASELINE_2026-08-03.txt` is committed **as
written**, including two wrong rationales that are themselves the finding.

**M05 was predicted detectable** because *"the pre-existing hijack test sets
`N_CLASSES_OBSERVED`."* **It does not** — it sets `POPULATION_SCOPE`,
`N_OBSERVATIONS` and `CERTIFICATION_ELIGIBLE`. I had that list from the REG-1
preflight and asserted the opposite from it. **M05 was a real missing test, not
an equivalent mutant.**

**M06 was called *"the one thing a test cannot catch"***, on the reasoning that
no test can catch a number that is still correct. **That reasoning was wrong.** A
*behavioural* test cannot; a *structural* one can, because the derivation is a
property of the **source**.

> Behavioural tests prove outputs and refusals.
> Structural tests prove ownership, derivation and authority paths.

### 5.2 Both gates falsified individually before round two

```
M05 alone → test_an_applicable_verdict_may_not_forge_n_classes_observed
            FAILED: DID NOT RAISE RegistryInvariantError

M06 alone → test_refusal_protected_keys_are_derived_from_ctx_support
            FAILED: "the refusal protected set was HAND-LISTED instead of
            derived from ctx.support()" — assert 0 == 1, where 0 = len([])
```

Both restorations hash-verified; the file returned to 54 passed. **Inferring
which test caught which mutation from an aggregate is weaker than showing it.**

### 5.3 Round two: 6 detected, every one by the named gate

R2 adds **oracle classification** and, more importantly, verifies that the
**named** detector fired — `DETECTED_BY_EXPECTED` distinguished from
`DETECTED_BY_OTHER`. The baseline could not tell the intended gate firing from an
unrelated test failing for an unrelated reason.

**All six were caught by the test written to catch them.**

M04 proves the exception is necessary. M05 proves it must not leak into the
success path. M06 proves the set must be derived rather than enumerated. Three
mutations, three halves of one model.

---

## 6. Five defects of mine in this commit alone, none caught by review

**A raise-site check that counted the class definition.** `RegistryInvariantError (`
also matches `class RegistryInvariantError(RuntimeError)` at registry.py:176. The
check would have refused a correct file. Now prefixed with `raise`.

**A dead helper.** `_refusal_hijack`, written then bypassed — dead code in the
commit whose siblings are DEAD-1, DEAD-2 and DEAD-3.

**A tautology.** `assert result.metadata["population_scope"] == ctx_scope_of(result)`,
where the helper returned that same value. **A comparison of a thing with
itself** — the identical defect I repaired at `test_bootstrap_reconciliation.py:696`
that morning, reproduced hours later.

**A count written against code I meant to write.** `object.__setattr__` expected
+2; the dead helper made it +4.

**A fifth post-check matching its own prose.** `_DESCRIPTOR_OWNED_ON_REFUSAL`
counted 3 — one in code, two in docstrings. Fixed as a **class** by tokenising,
not as an instance by tuning.

### 6.1 And a sixth, of a different kind

The ratchet installer's post-check refused a **correct** entry because
`Sabotage: 6 mutations, 6 detected, 0 undetected, 0 anchor misses.` **already
appears in an earlier entry** — a previous commit that also ran six mutations and
detected all six. The check counted over the whole file when it should have
counted within the appended entry.

That is distinct from the others: the expectation was right and the **scope** was
wrong. My fixture could not have caught it, so the fixture was rebuilt to
**contain a duplicate**, reproducing the exact hazard, and the corrected check
verified against it.

### 6.2 A fourth stale constant of the same shape

I predicted the ratchet would move to 4,155. Collection said **4,157**: REG-1
added **five** test functions, three in the first battery and two more to close
M05 and M06, and my figure was carried forward from before the closure tests
existed. Collection is the only authority.

---

## 7. Acceptance

| item | value |
|---|---|
| full suite | 4151 passed, 6 skipped, 0 failed (17m49s, and 18m15s under the armed gate) |
| collected | 4157 |
| ratchet | 4152 → 4157 (+5) |
| `test_readme_claims` | 10 passed; ratchet and badge agree |
| sabotage | 6 mutations, 6 detected, 0 undetected, 0 anchor misses |

---

## 8. Follow-ups — twelve, none touched

| id | item |
|---|---|
| POP-1b-M03 | no test distinguishes the source distance from the parent distance |
| POP-1b-M07 | nothing asserts on `print_report` output |
| ZERO-1 | 24 dead-connector defaults still zero — is the allowlist stale? |
| INF-1 | an infinite reference label is pooled with NaN as *withheld* |
| ABS-1 | the ranking channel's refusal reported as `undefined_on_cohort` |
| DEAD-1 | ~40 lines of dead absence computation in `evaluate` |
| DEAD-3 | `_assert_absence_biconditional` computes `observed_curves` twice |
| PRE-2 | section 5's PASS line swallows the KAN banner |
| LINT-1 | no lint gate anywhere |
| F821-1 | 18 undefined names; 9 need assessment |
| CMP-1 | `ModelComparison` carries a fingerprint with no scope beside it |
| **STRUCT-1** | *new.* The structural-test technique proved its worth here. Other invariants stated only in comments — the merge orders, the "rejected not shadowed" rule — could be gated the same way. |

---

## 9. Next

**OP-1** — the operating-point subsystem this whole sequence set out to build. It
now rests on a population that says what it describes (POP-1a, POP-1b) and a
registry whose metadata ownership is enforced on every path (REG-1).

Then the drift monitor, whose red is roadmap 6.20's fix working — exactly as
preflight 13c's red was PRE-1a working, and as REG-1 v1's 29 red tests were the
suite working.
