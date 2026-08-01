# SESSION 2026-07-31 / 2026-08-01 — OP-1 preflight, source collection and defect register

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Repository state throughout: `HEAD = origin/main = 960f807`, working tree clean, suite-size ratchet 4121.**

---

## 1. What this session was for, and what it actually produced

The session opened to build **OP-1**, the typed operating-point subsystem: one
commit, non-authoritative, replacing the four defects recorded in
`ClinicalEvaluator._find_operating_point`. The handoff and the build
specification were complete; the starting prompt named exactly one outstanding
read.

No OP-1 code was written. The session was spent establishing, by measurement,
that the specification rested on several facts that were wrong, and that the
subsystem's surrounding code carries defects the specification does not mention.
Every claim below was produced by running something against the repository at
`960f807`, and every instrument was itself tested before its output was trusted.

The reading phase is now **complete**. Steps A through D can be written from
measured facts alone.

---

## 2. Repository state, verified

Measured 2026-07-31 04:49 and re-measured 2026-08-01 10:35. Identical
throughout; no file changed during the session.

| file | lines | bytes | SHA-256 (first 16) |
|---|---:|---:|---|
| `capabilities.py` | 568 | 27,936 | `02567d107d2e86fc` |
| `registry.py` | 1,675 | 80,796 | `048515a388304cfe` |
| `metrics.py` | 2,011 | 94,058 | `909c9cacb0188df4` |
| `population.py` | 512 | 25,765 | `4a6efa5c7b16baf8` |
| `evaluator.py` | 1,882 | 88,469 | `51ad26c7d8501401` |
| `legacy_projection.py` | 409 | 18,366 | `274c26e15eb3cc30` |
| `catalogue.py` | 285 | 13,920 | `a64b2cfe4b82f9c6` |
| `input_validation.py` | 171 | — | — |
| `operating_point.py` | ABSENT | — | OP-1 creates it |

`registry.py` matches the handoff exactly, so its `REPO`-labelled line numbers
are valid. `metrics.py` at 94,058 bytes against the earlier sandbox copy's
81,746 confirms the `SANDBOX` numbers in handoff section 3.12 are unusable; every
read in this session anchored on a name, never a number.

**`tests/EXPECTED_SUITE_SIZE` is not a bare integer file.** It carries a 4,305-line
comment header and the value `4121` on the final line. The handoff and the
starting prompt both describe it as a single token. It is parsed correctly by
`tests/conftest.py:58` (`_read_expected_suite_size`), and there are 28 references
to it across the repository. Any new gate must match that convention rather than
invent a second one.

---

## 3. Instrument defects — three of mine, all caught before their output was believed

Recorded in full because each would have put a false statement into a durable
document, and because the pattern matters more than the individual faults.

### 3.1 The double-wrapped array (2026-07-31, collector version one)

`Read-SourceLines` ended `return , $raw` while every call site wrapped the result
in `@( )`. The comma operator stops pipeline unrolling; `@( )` re-collects. Together
they double-wrapped, and every consumer declaring `[string[]]$Lines` then
**silently coerced** the nested array into one joined string.

Consequences: nine self-test assertions failed; the transcript reported
`registry.py is 1 lines` while the fingerprint table one line above — which never
used the helper — correctly reported 1,675; and it reported
`class Applicability absent` from two files. **That last claim was false and said
nothing about where the class lives.** Diagnosed from the assertion-11 failure
message, which showed the entire fixture rendered as one space-joined string;
the spacing reconstructed the blank lines exactly, which is what made the
diagnosis certain rather than plausible.

Fix: return the array plain; add an independent cross-check inside the reader
(the file is read a second time as one raw string and the expected line count
derived from its line-feed count); add `Assert-LineArray`, deliberately untyped,
because a `[string[]]` parameter would perform the very coercion being tested
for. Eleven regression assertions.

### 3.2 Implicit mandatory validation on a string array (2026-07-31, version two)

`[Parameter(Mandatory = $true)]` applies an implicit `ValidateNotNullOrEmpty`,
and on a `[string[]]` parameter that attribute validates **every element**. The
fixture's blank lines are empty strings, so the first navigation call was
rejected with `Cannot bind argument to parameter 'Lines' because it is an empty
string`. `[AllowEmptyCollection()]` permits an empty *array*, not an empty
*element*. Since `registry.py` and `metrics.py` are full of blank lines, the real
collection would have died identically.

Fix: `Mandatory` removed from every parameter, replaced by explicit body guards,
rather than relying on an attribute-interaction rule that cannot be verified
without executing PowerShell. Assertions changed to take **script blocks**, so an
exception becomes an attributed failure instead of killing the run with a raw
binding error naming no assertion. Nine new regression assertions passing arrays
containing empty strings through every function.

### 3.3 The indentation heuristic (2026-08-01, final reads)

`enclosing_def` located a function by walking backwards tracking minimum
indentation, with no awareness of string literals. This codebase writes
flush-left section headers inside docstrings — `WHY THIS MODULE EXISTS`,
`ADDRESSING MODEL`, `THREE CHANNELS, NOT ONE GATE`. Such a line reports
indentation zero; thereafter no `def` can satisfy `indent < 0` and the search
runs off the top of the file. Reproduced on a ten-line fixture.

Fix: **not a smarter heuristic.** Python parses Python — replaced wholesale with
the `ast` module, whose nodes carry exact `lineno` and `end_lineno` including
decorators. Verified against fixtures carrying the exact flush-left headers that
defeated the heuristic.

### 3.4 A fourth, minor, disclosed here for completeness

The `restrict` call-site audit classified a line as a comment only if it began
with `#`. The occurrence at `population.py:347` is inside a **docstring**, so it
was reported as a `CALL`. The correct count of non-comment `restrict` call sites
under `src/` is **zero**, not one. This does not change any conclusion below —
it strengthens it — but the count as printed is wrong.

**The pattern.** Every one of these was caught by a self-test or a required-read
failure, never by a transcript that read plausibly. Defect 3.1 would have entered
the record as "the `Applicability` class does not exist in your repository."

---

## 4. Facts established for OP-1 steps A through D

### 4.1 The `Applicability` contract (`registry.py:447-474`)

Handoff section 3.8 was right; the 2026-07-30 installer draft was wrong.

```python
if self.status is None or self.status is MetricStatus.OK:
    raise ValueError("an inapplicable decision requires a non-OK status; ...")
```

A refusal without a status raises. `APPLICABLE = Applicability(applicable=True)`
at line 474, one declaration, one file.

### 4.2 Acceptance criterion 19, measured

No prior record captured the **status** column. Oracle C compares it as strictly
as it compares the reason.

| registered metric | status | reason | verdict metadata |
|---|---|---|---|
| `sensitivity` | `INSUFFICIENT_SUPPORT` | `positive_class_support_required` | `{REFERENCE_CLASS_SUPPORT: "single_class"}` |
| `specificity` | `INSUFFICIENT_SUPPORT` | `negative_class_support_required` | `{REFERENCE_CLASS_SUPPORT: "single_class"}` |
| `positive_predictive_value` | `INSUFFICIENT_SUPPORT` | `empty_predicted_positive_set` | `{"threshold": 0.5}` |
| `negative_predictive_value` | `INSUFFICIENT_SUPPORT` | `empty_predicted_negative_set` | `{"threshold": 0.5}` |
| `f1` | `UNDEFINED` | `zero_f1_denominator` | `{"n_predicted_positive": …, "n_reference_positive": …}` |
| `matthews_correlation_coefficient` | `UNDEFINED` | `zero_confusion_margin` | same two keys |
| `prevalence` | `INSUFFICIENT_DATA` / `NOT_APPLICABLE` / `FAILED` | `empty_cohort` / `reference_labels_required` / `nonfinite_reference_labels` | `{"n": 0}` / — / `{"n_nonfinite_labels": …}` |

**Criterion 19 as written — "reason strings identical" — is too narrow.** It must
become identity of the `(status, reason, metadata)` triple.

### 4.3 Nine corrections the build specification needs

1. Every factory takes `ThresholdParameters` **positionally** plus a `metric=`
   keyword. Section C.3 writes `_requires_class_support(positive=True)`; the
   landed form is `_requires_class_support(tp, *, positive, metric)`.
2. `_requires_class_support` tests `wanted not in ctx.classes_observed`, not a
   count. `classes_observed` is the sorted unique of the **finite** label values.
3. Metadata is wrong in both the specification and the handoff:
   `_requires_nondegenerate_confusion` attaches **two** keys, and
   `_requires_class_support` attaches an **enum-keyed** entry.
4. Every confusion predicate calls `_requires_probabilities(ctx)` first, which
   can refuse four ways. **Three cannot be expressed from four integers**
   (`probabilities_required`, `nonfinite_predicted_probabilities`,
   `values_are_not_probabilities`). The `*_from_counts` functions are **tails**,
   not equivalents.
5. Defect D6 has a **seventh** inline-counting site, not six:
   `_requires_flagged_margin` at `registry.py:958-962` as well as
   `_requires_nondegenerate_confusion` at 863-880. Confirmed by cross-module
   search: exactly four references to `apply_decision_threshold` outside
   `metrics.py`, all in `registry.py`, and **zero** references to
   `_margins_at_threshold` or `_confusion_counts` outside `metrics.py` across 178
   files — so section C.2's removal permission is confirmed by measurement.
6. The identity validator is the hazard flagged earlier, and the code says so.
   `_validate_registry:1461` asserts `getattr(d.applicability,
   "_threshold_parameters", None) is not tp`. Live audit: **9 metrics share one
   object by identity, 15 have none, 0 mismatches.** Any rewritten predicate must
   carry `predicate._threshold_parameters = tp` forward; an equal-but-distinct
   instance passes field comparison and fails this.
7. `compute()` merges verdict metadata in **opposite orders** in two branches.
   The inapplicable branch builds `{METRIC_NAME, **ctx.support(),
   **verdict.metadata}` — verdict last, wins, **with no collision check**. The OK
   branch runs the protected-key check then `{**verdict.metadata, **meta}` —
   verdict first, loses. Step B preserves only the second. A finalizer unifying
   them would change the first, and Oracle A would miss it.
8. That asymmetry is a latent defect: an inapplicable verdict can today overwrite
   `POPULATION_SCOPE` or `POPULATION_FINGERPRINT` in a refusal record silently.
   The comment at `registry.py:659-662` shows the author avoiding exactly that
   collision by hand — the discipline is maintained by care, not enforced.
9. Three different sevens. `COUNT_METRIC_SPECIFICATIONS` has eight entries; seven
   have registered descriptors (the eight minus `flagged_fraction`, absent from
   the 24 registered names); `OperatingPointMetrics` carries a **different**
   seven. The specification says "the seven metrics" as though it named one set.

### 4.4 Two conflicts that block step C as specified

**The predictive-value metadata.** Both verdicts carry `{"threshold": 0.5}`. A
function taking four integers cannot produce that key, and Oracle C requires the
complete metadata mapping with no excepted field. Either
`*_applicability_from_counts` takes the `ThresholdParameters` alongside the
counts, or the oracle excepts a field it explicitly forbids excepting.

**Prevalence is a different estimand.** The registered kernel is
`np.mean(y_arr == 1)` over the array it was handed, and its docstring refuses to
filter. Counts arrive through `_margins_at_threshold` → `_clean` →
`clean_arrays`, whose mask **drops rows**. `(tp + fn) / n` is a proportion over
the cleaned population; `prevalence(y)` is one over the full population.

---

## 5. The population divergence — demonstrated, then confirmed reachable

### 5.1 The mechanism

`clean_arrays:251-254` builds `keep = isfinite(y) & isfinite(score) &
isfinite(probability)` and drops rows. `_requires_probabilities` guarantees the
probabilities are finite before dispatch, so `fs` and `fp` cannot drop anything
on the registry path. **Nothing in the applicability layer checks the reference
labels.** Meanwhile two predicates count over the **raw** context arrays.

### 5.2 Reproductions, against the installed package

| cohort | predicate | kernel |
|---|---|---|
| `y=[0, nan]`, `p=[0.1, 0.9]` | flagged count 1 → APPLICABLE | margins `(0,0,0,1)` → NaN |
| `y=[1, nan]`, `p=[0.9, 0.1]` | margins `(1,1,1,1)` → APPLICABLE | counts `(1,0,0,0)` → NaN |
| `y=[1,1,0,nan]`, `p=[0.9,0.1,0.2,0.8]` | counts `(1,1,1,1)` → ppv 0.5000 | counts `(1,0,1,1)` → ppv 1.0000 |

Two controls held: class support does **not** diverge, because
`ctx.classes_observed` already filters non-finite labels; and on all-finite
labels the two paths are bit-identical.

### 5.3 End to end through `registry.compute()`

The first two return `status failed`, reason
`applicable_metric_returned_non_finite` — whose own comment at `registry.py:1579`
calls that state an implementation defect, not a property of the cohort.

**The third returns something worse:**

```
value   1.0
status  ok
reason  None
CERTIFICATION_ELIGIBLE   True
N_OBSERVATIONS           4
N_CLASSES_OBSERVED       2
POPULATION_FINGERPRINT   sha256:9ff577fc17ef440d857457c266308e3aa4c467177139cdc6e36871e2560bf87f
```

A value computed over **three** rows, carrying the **four**-row population's size
and fingerprint, status OK, certified eligible, no reason, no diagnostic. The
fingerprint attests to a membership set the value was not computed over.

Compare the incident at `registry.py:530-535`, which motivated the 2026-07-27
fail-closed ruling: *twenty non-finite probabilities in a thousand rows produced
a Brier score over 980 rows reported as `n_observations = 1000`, status ok,
certification_eligible True.* **This is that defect on the label axis, live.**

### 5.4 Reachability — confirmed through the supported path

`ClinicalEvaluator.evaluate` is the only `MetricContext` construction in the
entire source tree (`evaluator.py:957`). `label_check` is assigned at 846 and read
at 901-902, feeding `ranking_usable` and `probability_usable`, which gate the
**curves**. The construction passes `y_true=y.astype(float)` and
`y_prob=p.astype(float)` **unconditionally**; only `y_score` is gated, and on the
ranking check.

**The registry receives raw labels whatever the label verdict says.**

`population.py:13-14` records that **withheld labels are first-class in this
project and are carried as NaN by `CanonicalVariantTable`**. These are not
pathological cohorts. They are the ordinary representation of data this pipeline
ingests.

### 5.5 One input, three verdicts, inside a single call

Within one `ClinicalEvaluator.evaluate()` on a cohort with withheld labels:

- the three **operating points** refuse and return `None` with a logged warning
  (gated at 1279-1293 and 1351-1362 — this gating is correct and was added
  2026-07-28 under CI-t);
- the **curves** are withheld through `_absence_maps`;
- the **registry metrics** are computed and reported OK, certified eligible.

That is precisely the incoherence the 2026-07-28 commit 3b-2 says it exists to
remove: *"One input, two layers, opposite verdicts"* (`evaluator.py:871-878`,
944-953). It survives, one layer further down, on a different axis.

**One inference, flagged as such and not yet confirmed:** `_absence_maps` marks a
field absent only when `absence_for_value(...)` returns non-`None`, which the
surrounding idiom suggests happens only for missing values. A field carrying a
**finite** number computed over a narrowed population would therefore pass
through unannotated even when `label_check.ok` is false. I have not read
`absence_for_value`; this needs confirming before it is asserted.

### 5.6 The replacement mechanism exists, is tested, and is not wired in

`population.py:11-20` records that the 2026-07-27 ruling was enforced in halves;
the label half was deferred **deliberately**, parked behind a named transitional
selector `metrics.select_finite_reference_labels`, and *"This module is that
target's replacement."* The selector is correctly gone — one mention under
`src/`, in that docstring, no definition anywhere.

Call-site audit of `EvaluationPopulation.restrict`:

- **`src/` (179 files): zero real call sites.** The single occurrence is the
  docstring at `population.py:347`.
- **`tests/` (289 files): 22 call sites**, and they encode the exact intended
  production idiom, vocabulary included:
  - `test_evaluation_population.py:423` — `eligible = attempted.restrict(np.isfinite(y), scope="label_eligible", …)`
  - `test_prediction_input_contract.py:359` — the same line
  - `test_evaluation_population.py:380, 400` and `test_registry_vocabulary_completion.py:535` — `population = attempted if mask.all() else attempted.restrict(…)`, the guard `restrict` requires because it refuses a no-op narrowing
  - scope `"label_eligible"`, reason `"reference_label_withheld"`
- `scripts/` (443 files): zero.

**The mechanism is fully built, fully tested, its vocabulary is settled, and
production code never calls it.** `clean_arrays` still narrows silently inside
the kernel — the exact act the ruling forbids, and precisely the defect shape
`population.py:32-34` says an `EvaluationPopulation` cannot lose.

---

## 6. Defect register for the OP-1 target functions

Both read in full at `960f807`: `_find_operating_point` at `evaluator.py:1254-1326`,
`_find_high_ppv_point` at `1328-1400`, `class OperatingPoint` at `218-232`.

### D1 — the thousand-point linear sweep (`1297`)

`for t in np.linspace(0, 1, 1000)`. The achievable operating points are
determined by the **unique probability values**, not by a uniform grid, so the
grid can miss the exact threshold achieving the target sensitivity entirely.

### D2 — specificity fabricated as `0.0` (`1311`, and `1384` in the sibling)

`tn / n_neg if n_neg > 0 else 0.0`. With no negatives, `0.0` asserts that every
negative was misclassified. The value is undefined, not zero.

### D3 — positive predictive value fabricated as `0.0` (`1312`)

Same shape, when nothing is flagged.

### D4 — negative predictive value fabricated as `0.0` (`1313`, `1385`)

Same shape, when nothing is cleared.

### D5 — F1 inherits the fabrications (`1314-1315`, `1386-1387`) — NEW

`f1 = 2·ppv·sensitivity/(ppv+sensitivity) if (ppv+sensitivity) > 0 else 0.0`,
computed from a `ppv` that may itself be a fabricated `0.0`. A second
fabrication, derived from the first, with no record that either occurred.

### D6 — rounding at construction destroys the computed value (`1317-1322`, `1390-1395`) — NEW

Every float is stored as `round(value, 4)`. `f1` is computed from **unrounded**
`ppv` and `sensitivity` and then rounded, so recomputing `2ps/(p+s)` from the
**stored** `ppv` and `sensitivity` will not generally reproduce the **stored**
`f1`. The record is internally inconsistent, and a clinical decision threshold is
quantized with no statement that it was.

### D7 — `n_neg` names two different quantities in one class (`1304` vs `1373`) — NEW

```
1304    n_neg = fp + tn          # the NEGATIVE count
1373    n_neg = tp + fp          # n_flagged  -- the FLAGGED count
```

Both feed division. The arithmetic at 1378 is correct; at 1396 the misnamed
variable is passed as `n_flagged=int(n_neg)`, which is right and reads wrong.
Sixty-nine lines apart in the same class.

### D8 — the high-positive-predictive-value function states two objectives — NEW

Its docstring says both:

- line 1335 — *"Highest-sensitivity threshold where PPV ≥ min_ppv"* — **objective A**
- lines 1338-1339 — *"the most permissive threshold that never drops below min_ppv"* — **objective B**

The `break` at 1379-1381 and the unconditional overwrite of `best` implement **B**.
Measured on `y=[1,0,1]`, `p=[0.9,0.8,0.7]`, `min_ppv=0.6`, where the positive
predictive value runs 1.0000 → 0.5000 → 0.6667 (**not monotone in the threshold**):

- objective A gives `t=0.7`, sensitivity **1.0000**, ppv 0.6667
- objective B gives `t=0.9`, sensitivity **0.5000**, ppv 1.0000
- the landed code returns `t=0.9`

**B is a legitimate clinical objective** — it guarantees that at this threshold or
anything more conservative, precision holds. The defect is that the summary line,
the function name and the parameter name all say A. *Correcting my own earlier
framing: I described this as an unsound break. It is sound for B; the docstring
is contradictory and the naming points at A.* Which objective is intended is a
scientific decision, not a code fix.

### D9 — quadratic cost at production scale (`1366`) — NEW

`_find_high_ppv_point` loops over `np.unique(p)` and recomputes `(p >= t)` plus
four boolean reductions inside, so it is **O(k·n)** with `k = |unique(p)|`. For a
cohort of 1.5 million variants with distinct probabilities that is
**2.25 × 10¹²** element operations.

| n | `linspace` sweep, O(1000·n) | unique sweep, O(k·n), k=n | sort + cumulative sum, O(n log n) |
|---:|---:|---:|---:|
| 1,000 | 1,000,000 | 1,000,000 | 9,965 |
| 100,000 | 100,000,000 | 10,000,000,000 | 1,660,964 |
| 1,500,000 | 1,500,000,000 | **2,250,000,000,000** | **30,774,796** |

A single sort with cumulative sums yields **exact** counts at every achievable
threshold in O(n log n) — simultaneously fixing D1's inexactness and D9's cost.

### D10 — loop-invariant work inside the sweep (`1303-1306`) — NEW

`n_pos = tp + fn` is the total count of actual positives and `n_neg = fp + tn`
the total count of actual negatives; both are **invariant across thresholds**. The
guard `if n_pos == 0: continue` is therefore evaluated 1,000 times to reach the
same answer, and on a cohort with no positives the function performs the entire
sweep to return `None`.

### D11 — two sweep strategies in one class (`1297` vs `1363`) — NEW

`_find_operating_point` sweeps `np.linspace(0, 1, 1000)`; `_find_high_ppv_point`
sweeps `np.sort(np.unique(p))[::-1]`. Two operating-point functions, one class,
two incompatible notions of which thresholds exist.

### D12 — an undeclared tie-break (`1309`) — NEW

`if diff < best_diff` is strict, so with `linspace` ascending the **first**
threshold achieving the minimum difference wins, and ties go to the **lower**,
more liberal threshold. Defensible, undocumented, and invisible to a reader of
the result.

---

## 7. Decisions required before step C is written

None of these is mine to make.

1. **Do the count-level applicability functions take `ThresholdParameters`?**
   Without it the predictive-value metadata `{"threshold": 0.5}` cannot be
   reproduced and Oracle C cannot hold as written.
2. **Does Oracle C compare the `(status, reason, metadata)` triple?** The measured
   four-`INSUFFICIENT_SUPPORT` / two-`UNDEFINED` split forces this.
3. **Is count-level prevalence a different estimand**, or excluded from Oracle C?
4. **Which objective does `_find_high_ppv_point` intend, A or B?**
5. **Is the label-finiteness wiring in scope for OP-1, or its own commit?**
6. **Are the two composite certification blockers given explanatory prose?**

---

## 8. Recommended shape, offered as a recommendation

**OP-1 lands as specified, plus D5 through D12**, because those defects are inside
the very functions the commit replaces; fixing them is the job, not scope creep.
The sort-plus-cumulative-sum sweep resolves D1, D9, D10 and D11 together and
makes the result exact rather than gridded.

**Step D's rerouting is included and declared a behaviour change**, not described
as bit-identical, with its own test — it closes the definedness half of the
population divergence by construction.

**The label-finiteness wiring is its own commit and requires authorisation.** It
completes a scientific ruling, and its shape is already designed, tested and
named by the test suite:

```python
mask = np.isfinite(y)
population = attempted if mask.all() else attempted.restrict(
    mask, scope="label_eligible", reason="reference_label_withheld")
```

placed where `evaluate` builds the population, so the narrowing becomes explicit,
reasoned, and carried in the lineage exactly as `population.py` intends — instead
of happening silently inside `clean_arrays` where nothing downstream can see it.

**The `compute()` branch-2 protected-key gap is recorded and left for its own
commit**, so OP-1's "zero movement" claim stays literally true for everything it
does not declare.

---

## 9. Artifacts produced

| instrument | outcome |
|---|---|
| `op1_stepAD_collector_2026-07-31.ps1` | failed; defect 3.1 |
| `op1_stepAD_collector_v2_2026-07-31.ps1` | failed; defect 3.2 |
| `op1_stepAD_collector_v3_2026-07-31.ps1` | 58/58 self-test, exit 0, 6,510-line transcript |
| `op1_population_divergence_probe_2026-07-31.py` | exit 1, five findings, zero failures |
| `op1_reachability_probe_2026-08-01.py` | exit 1, three findings, zero failures |
| `op1_label_validation_reader_2026-08-01.py` | exit 0; under-read one section (3.4 note) |
| `op1_final_reads_2026-08-01.py` | failed; defect 3.3 |
| `op1_ast_reads_2026-08-01.py` | exit 0, zero divergences — read complete |

Transcripts in `C:\Users\monzi\Downloads\`.
