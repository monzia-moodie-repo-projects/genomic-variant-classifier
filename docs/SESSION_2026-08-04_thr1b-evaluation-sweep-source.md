# SESSION 2026-08-04 — THR-1b: `EVALUATION_SWEEP`, and the vocabulary gains a gate

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = b4cbdc4`. Full suite 4169 passed, 6 skipped, 0 failed; 4175 collected.**
**Ratchet 4171 → 4175 (+4), measured by the installer rather than written into it.**

Companion documents:
`SESSION_2026-08-04_thr1a-threshold-vocabulary.md`,
`SESSION_2026-08-04_reg2-denominator-status.md`.

---

## 1. The addition is one line; the gate is the substance

`ThresholdSource` gains `EVALUATION_SWEEP = "evaluation_sweep"`.

Measured in the installer's own preconditions, **before** the change, across the
whole test suite:

```
'for member in'         0 occurrence(s)
'ThresholdSource}'      0 occurrence(s)
'_EXPECTED_THRESHOLD'   0 occurrence(s)
```

**Five tests referenced `ThresholdSource` and not one enumerated its members.**
They used it as a *value* — asserting `fixed_default` on three descriptors,
exercising type validation, and using `CALIBRATED` as a differing value in a
fingerprint variant.

So a member could have appeared, disappeared, or quietly changed its serialised
string, and **nothing in 4,171 tests would have objected.**

That is the REG-2 shape one layer down: there, a semantic correction to two
metrics changed no test outcome anywhere, and the repair was the assertions that
would notice it next time.

---

## 2. Why the rename case is the one that matters

`tests/unit/test_registry_vocabulary_completion.py:832`:

```python
assert mcc_print["threshold"] == (0.5, ">=", "fixed_default")
```

**The descriptor fingerprint embeds the serialised source string** in a tuple
compared by equality. That fingerprint is the immutability audit — so these
values are genuinely load-bearing, and a renamed value would silently orphan
every historical record carrying the old one.

Nothing gated that until now.

---

## 3. The gate runs in both directions

On the pattern of `test_conformal_package_exports.py`, which asserts that every
module reachable is declared **and** every declared name exists.

The member set is asserted **exactly**, so three distinct failures are caught:

| change | caught |
|---|---|
| a member added | yes — the set differs |
| a member removed | yes — the set differs |
| a **value renamed** | yes — the mapping differs |

All three were exercised against a constructed vocabulary before shipping.

### 3.1 And falsified against the live module afterwards

A throwaway `SNEAKY = "sneaky"` added to the enum:

```
Left contains 1 more item: {'SNEAKY': 'sneaky'}
AssertionError: the ThresholdSource vocabulary changed.
1 failed, 9 passed
```

**The other nine passed unchanged** — the completeness gate is not entangled with
identity, serialisation, ownership or validation. A gate that fails alongside
everything else is measuring the change rather than the property. That is the
same distinction T3 established for REG-2.

Restoration verified by digest; 10 passed after.

### 3.2 Compatibility is asserted separately from the set

The set test fails for **any** change. A second test names the three members that
predate THR-1b, so a failure says immediately whether an **existing** value moved
— which reinterprets artifacts — or a **new** one was added, which does not.

All three kept their strings, verified by post-check on the patched source.

---

## 4. Why `EVALUATION_SWEEP` and not `EVALUATION_SELECTED`

**A candidate exists before selection.** Every point an exact sweep enumerates
carries this source, and at most one is ever chosen. "Selected" would make the
vocabulary temporally false for all but one candidate.

And it keeps three facts apart, which is why this is a type rather than a comment:

```
SOURCE                  where the candidate came from        <- this enum
POLICY                  why it was chosen among candidates   <- OP-1's policy
CERTIFICATION BLOCKERS  why its performance is not
                        independently validated
```

A field conflating them would leave an artifact unable to distinguish *swept*
from *chosen* from *unvalidated*.

---

## 5. The ratchet measured its own target

Every previous ratchet in this sequence carried a hand-written value, and that
constant was **wrong three times today**:

| | predicted | measured |
|---|---|---|
| OP-0 | 4,155 | **4,157** |
| REG-2 | 48 test functions | **49** |
| THR-1a post-check | 2 | **3** |

Each was caught by a gate — the machinery working. But a target that is an
arithmetic guess is a **stale constant waiting to happen**, which is the shape
the roadmap's section 7 names.

So this installer ran `pytest --collect-only`, read the count, and wrote **that**.
The prediction of 4,175 was kept only as a **cross-check**, and if the two had
disagreed the installer would have **refused** and reported both — because a
surprise in the count means something landed that the commit did not intend.

They agreed. But the refusal path is the reason to build it this way.

---

## 6. No sabotage line

The addition changes no existing value and no behaviour, so there is nothing
behavioural to mutate. The falsification in §3.1 **is** the mutation, with its
outcome predicted before it was run.

---

## 7. Acceptance

| item | value |
|---|---|
| full suite | 4169 passed, 6 skipped, 0 failed (16m02s, under the armed gate) |
| collected | 4175, measured by the installer |
| ratchet | 4171 → 4175 (+4) |
| `test_readme_claims` | 10 passed |
| pre-existing serialised values | **all three unchanged** |

This was also the **first full execution** since the member landed — collection
proves the tests exist, not that they pass.

---

## 8. Next

**Step 1** — `ConfusionCounts`, the exact sweep, and the lazy candidate view.
Everything it needs now exists: `thresholds.py` as the bottom layer (THR-1a),
`EVALUATION_SWEEP` to label each candidate (THR-1b), the status vocabulary
correct and gated (REG-2), and an algorithm already verified against the
brute-force definition across ten cohorts.

**Its inventory comes first this time** — EXTRACT-1 applied rather than merely
recorded.

Seven register defects remain open: D1, D2-D5, D6, D9, D10, D11, D12. One sort
with cumulative sums closes D1, D9, D10 and D11 together.
