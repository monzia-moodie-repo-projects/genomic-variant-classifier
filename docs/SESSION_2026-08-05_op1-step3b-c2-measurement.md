# SESSION 2026-08-05 — OP-1 step 3b: the Oracle C2 measurement

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = 8a591b3`. Suite unchanged at 4271 collected.**
**A MEASUREMENT, NOT A FIX — no code, no tests, and the ratchet does not move.**

Committed evidence:
`docs/measurements/OP1_C2_DIFFERENCE_2026-08-05.txt` — **R1, preserved with its incomplete prediction intact**
`docs/measurements/OP1_C2_DIFFERENCE_R2_2026-08-05.txt` — R2, corrected and complete

Companion document: `SESSION_2026-08-05_op1-step3a-oracle-c1.md`.

---

## 1. The ratchet does not move, and that is the point

Every commit in this sequence so far has added tests. This one adds none.

Step 3b **measures** the difference between a plain `MetricResult` and a
registry-finalised one, so that the finaliser extraction can be specified against
evidence rather than against my reading. The deliverable is two reports.

The adopted ruling was explicit: **making C2 pass by copying registry metadata
into the count path would recreate a second implementation of the finalisation
contract.** So the measurement comes first and any fix follows from it.

---

## 2. A hypothesis falsified before the measurement ran

REG-1 established that `registry.compute` had **two** metadata merge branches
that looked symmetric and were not. Counting return statements found **five**
`MetricResult` construction sites, and REG-1's protected-set work touched two.

So I hypothesised that some path might construct a refusal **before**
`ctx.support()` attaches — leaving artifacts with refusals that carry no
population identity, which is exactly what POP-1b exists to prevent.

**That hypothesis is false.** All five carry `**ctx.support()`:

```
line  12   NOT_APPLICABLE  required_inputs_missing
line  41   refusal         verdict.status / verdict.reason
line  49   FAILED          metric_computation_failed
line  63   FAILED          applicable_metric_returned_non_finite
line 115   OK              the computed value
```

**A negative result worth recording.** The concern was legitimate given five
sites and two audited, and the code is already correct. Knowing that by reading
all five is different from assuming it from the two REG-1 happened to touch.

---

## 3. R1 disagreed, and the disagreement produced a better design

The measurement stated its prediction **in advance** and failed if the
measurement differed. It exited **1**, with one reported disagreement — and the
two halves of what it exposed are not the same kind of thing.

### 3.1 A defect of mine

`PREDICTED_OK_ONLY` held one key. **This file's own docstring table named two:**

```
CERTIFICATION_BLOCKED_BY   the OK path, when not eligible
```

Prose and code disagreed, and the code is what ran. That is the same defect class
as every prose-versus-code count in this sequence, **inverted**: previously prose
leaked *into* a check; here the check omitted what the prose stated.

### 3.2 A discovery

Four keys appeared on refusal paths that **no prediction covered**:

```
n_predicted_positive   n_reference_positive
reference_class_support   threshold
```

They come from `verdict.metadata` — the **descriptor's** statement of why it
refused, varying per metric and cohort. I named that category in prose and never
turned it into a prediction, so the measurement had nothing to compare against.

---

## 4. The ownership boundary, which is the finding

```
the finaliser owns    identity, support, certification      7 keys
the descriptor owns   verdict.metadata                      4 observed
```

**Had R1 agreed with its prediction, the extraction would have been specified
with a finaliser that swallowed verdict metadata too — and been wrong about who
owns it.**

`reference_class_support` is the clearest case: it is calibration's diagnostic
about a single-class cohort, recorded by the descriptor. A finaliser claiming to
own it would be claiming authorship of someone else's evidence.

That boundary came from the **disagreement**, not from agreement. It is the
strongest argument in this session for stating predictions in advance rather than
reporting what a measurement happens to find.

---

## 5. The measured difference set, complete

| surface | keys | count |
|---|---|---|
| every path | `metric_name`, `population_scope`, `population_fingerprint`, `n_observations`, `n_classes_observed` | 5 |
| OK path only | `certification_eligible`, `certification_blocked_by` | 2 |
| refusal paths only | `n_predicted_positive`, `n_reference_positive`, `reference_class_support`, `threshold` | 4 |

**24 comparisons** across four cohorts and six metrics. R2 exited **0** with the
three surfaces confirmed and verified disjoint.

**The count path supplies none of them**, and that is structural rather than an
omission: `metrics_from_counts(counts, parameters)` takes no context and no
population, so it *cannot* supply population keys — it has never been given a
population.

---

## 6. Why R1 is preserved rather than corrected away

The instinct after exit 1 is to fix the constant and re-run until it exits 0.
**That would destroy the finding.**

R1 records a prediction incomplete in two ways — one a defect, one a discovery —
and a repository holding only the corrected version has lost the evidence that
the disagreement produced a better design than agreement would have.

Same discipline as REG-1's baseline mutation report, committed with its wrong
rationales intact because the failed prediction *was* the finding.

R2 writes to a distinct filename, so both survive.

---

## 7. What this forbids, and what it specifies

**C2 cannot pass by enriching the count path.** Copying registry metadata into
`metrics_from_counts` would require handing it a context — at which point it is
doing the registry's job with a second implementation of the finalisation
contract. That is the SWEEP-1 shape at the level of a **contract** rather than an
algorithm.

The architecture the measurement points to is **one finaliser both paths call**:
it takes a bare result plus a context and attaches identity, support and
certification — and **does not touch verdict metadata**, because that is the
descriptor's.

---

## 8. Next

**Step 3c — the finaliser extraction**, specified against these two reports.
Then C2 can require complete identity with no ignored-key list, which is the
condition the ruling set.

Then step 4 (the selector, Objective A, closing **D12** — the last of the
twelve), step 5 (the shadow comparison), and step 6 (the cutover, which must
reckon with **GUARD-1**).

Nineteen follow-ups carried, unchanged by this measurement.
