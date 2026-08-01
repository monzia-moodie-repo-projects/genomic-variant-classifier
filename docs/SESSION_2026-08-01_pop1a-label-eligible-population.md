# SESSION 2026-08-01 — POP-1a: explicit label-eligible evaluation populations

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = 960f807`, working tree clean at session start, ratchet 4121.**
**Outcome: POP-1a applied, corrected, pinned by 19 tests, sabotage-tested, ratchet at 4140, armed gate green.**

Companion document: `SESSION_2026-08-01_op1-preflight-and-defect-register.md`,
which covers the OP-1 preflight that preceded this work and the twelve-defect
register for the operating-point subsystem. This document begins where that one
ends.

---

## 1. What POP-1a is, and why it was necessary before OP-1

Ruled 2026-07-27:

> No numerical kernel may select, filter, normalise or redefine its evaluation
> population. Population construction is an explicit upstream operation, and
> every result must describe exactly that population.

Commit 2a enforced that for predicted scores and probabilities and
**deliberately left the label half standing**. `population.py:11-20` records why:
withheld labels are first-class in this project — carried as `NaN` by
`CanonicalVariantTable` — so selecting on them is a *population* decision, not a
kernel one. The behaviour was parked behind a named transitional selector,
`metrics.select_finite_reference_labels`, and `population.py` was written as that
selector's replacement.

The selector was retired. **The replacement was never wired in.** Measured
2026-08-01: `EvaluationPopulation.restrict` had twenty-two call sites in
`tests/` and **zero in `src/`** — the single occurrence under `src/` was the
docstring at `population.py:347`.

### 1.1 The defect, measured before any change

Against the installed package, `registry.compute` on `y = [1, 1, 0, nan]`,
`p = [0.9, 0.1, 0.2, 0.8]` returned for `positive_predictive_value`:

```
value 1.0    status ok    reason None
CERTIFICATION_ELIGIBLE True
N_OBSERVATIONS 4
POPULATION_FINGERPRINT sha256:9ff577fc...
```

A value computed over **three** rows, carrying the **four**-row population's size
and fingerprint, certified eligible, with no reason and no diagnostic. The
narrowing happened inside `metrics.clean_arrays`, where nothing downstream could
see it.

`registry.py:530-535` records the identical shape on the probability axis — a
Brier score over 980 rows reported as `n_observations = 1000`, status ok,
certification_eligible True — which is the incident the 2026-07-27 ruling was
written to eliminate. The fingerprint is the sharpest part: it attested to a
membership set the value was not computed over.

### 1.2 Reachability, established rather than assumed

`ClinicalEvaluator.evaluate` is the **only** `MetricContext` construction in the
source tree. `label_check` was assigned at line 846 and read twice, at 901-902,
feeding `ranking_usable` and `probability_usable` — which gate the **curves**.
The context construction passed `y_true` and `y_prob` **unconditionally**.

So the registry received raw labels whatever the label verdict said, and the
divergence was reachable through the supported path on ordinary input.

### 1.3 One input, three verdicts

Within a single `evaluate` call on a withheld-label cohort:

| subsystem | verdict |
|---|---|
| the three operating points | refuse, return `None`, log a warning |
| the curves | withheld through `_absence_maps` |
| the registry metrics | **computed, status OK, certification eligible** |

That is the incoherence commit 3b-2 says it exists to remove — *"One input, two
layers, opposite verdicts"* — surviving one layer down, on a different axis.

---

## 2. The wiring

### 2.1 Fourteen consumers, enumerated before a line was written

`evaluate` spans `evaluator.py:790-1132`. Every consumer of the evaluation
arrays was listed from the source first, because **a projection reaching
thirteen of fourteen produces a fingerprint that is actively wrong rather than
merely over-broad**, and the one missed is the one a spot check will not visit.

The population is now constructed **before** the input gates:

```python
attempted   = EvaluationPopulation.full(n_source, scope="attempted_cohort",
                                        source_id=source_id)
label_mask  = np.isfinite(np.asarray(attempted.take(y_attempted), dtype=float))
population  = attempted if bool(label_mask.all()) else attempted.restrict(
                  label_mask, scope="label_eligible",
                  reason="reference_label_withheld")
```

`scope="label_eligible"` and `reason="reference_label_withheld"` are **not new
vocabulary**. They are what the suite already asserted at twenty-two call sites;
POP-1a adopts them rather than coining a parallel set.

The `mask.all()` guard is not a workaround for the strict-narrowing rule.
`restrict` refuses a mask that removes *nothing*, because
`label_eligible(n=4)` beneath `attempted_cohort(n=4)` would assert a restriction
that never happened.

### 2.2 Three decisions made from measurement, not preference

**Gene clusters resolve on the projected frame.** `_resolved` refuses
all-or-nothing on any missing gene label, and `partitions_equivalent` compares
whole columns. Resolving on the full frame would withhold a certified interval
because a row *excluded from the evaluation* lacked a gene symbol — the 3b-1a
over-restriction, repeated. Resolving on `meta_eval` also makes `cluster.values`
already `population.n` long, so the bootstrap cannot misalign.

**`compare_models` restricts once and hands down one object.** Nothing in
`model_comparison.py` asserts object identity at runtime, but
`evaluator.py:1870-1875` records the shared population's fingerprint and count
into the comparison artifact while hard-coding
`comparison_is_like_for_like=True`. Restricting per call would make the artifact
carry the *attempted* fingerprint while every report it summarised described the
narrower set — this defect, one layer up.

**The metadata frame is projected with `meta.iloc[population.indices]`**, correct
because those indices are absolute positions into the original source frame per
`population.py:49-53`.

### 2.3 The registry path cannot be half-wired; the other thirteen consumers can

`MetricContext.__post_init__` enforces three checks: the population type,
`y_true` against `population.n`, and every other array against `y_true`. Under
POP-1a that becomes a guarantee — an unprojected array raises immediately.

**Only the registry path is defended.** `_interval_fields`, the three operating
points, both breakdowns and the three curve calls take bare arrays with no length
check. That asymmetry is why the acceptance battery names each consumer
individually rather than sampling.

### 2.4 The empty cohort: three failures, guarded once

An all-withheld cohort projects to `n = 0`, and `restrict` permits it — it
refuses a mask removing *nothing*, not one removing *everything*. Three failures
lie behind that state:

| line | failure |
|---|---|
| 822 | `n_pos / n * 100` — `ZeroDivisionError` |
| 962 | `project_legacy_fields` — `LegacyProjectionError` |
| 1531 | `r.prevalence * 100` — **never reached, therefore uncharacterised** |

The second is correct behaviour: the single-class area-under-the-precision-recall
-curve rule substitutes prevalence, and on an empty cohort prevalence is itself
refused. `legacy_projection` is right to refuse, and refuses by raising.

Guarding them one at a time would be patchwork, and the third was never even
characterised. **The cohort is refused once**, with its lineage in the message.
That is *not* the component-level refusal the input gates perform: those withhold
one quantity from a cohort that exists. Here there is no cohort.

### 2.5 A defect repaired in passing, declared rather than folded in

`n_pos = int(y.sum())` sat at line 817, **above every input gate**, and raised
`ValueError: cannot convert float NaN to integer` on a withheld label — measured
on six of nine probed dtypes, including a plain float array and a pandas
`Series`, which is exactly what the signature advertises. `evaluate` died there
rather than reaching the gates built for that input. Moving the line below the
projection repairs it by construction.

---

## 3. Four defects of mine, and what caught each

Recorded in full because the pattern matters more than the instances: **not one
was caught by my own reading of my own work.**

### 3.1 The scores regression — caught by the full suite

The first POP-1a projected `scores` *before* `validate_ranking_scores`, so a
mis-sized array raised from `population.take` instead of being refused. The
comment six lines below that gate, dated 2026-07-28, says exactly this:

> REFUSED MEANS NOT FORWARDED. A first version validated the array and passed it
> to the registry anyway, so a mis-sized `scores` raised a ValueError from the
> context's own length check — turning a refusal this gate exists to make
> graceful back into an exception three layers down.

POP-1a reintroduced it one layer earlier. Caught by the full suite: **1 failed,
4130 passed**. Only *one* of that test's two parametrisations failed — a
correctly sized array passes `take` and is refused gracefully — so a spot check
would have missed it.

The asymmetry I had missed: `y` and `p` are validated with `n_expected=n` because
they are already projected; `scores` arrives **source-aligned**, so it must be
checked against `n_source` and projected only once it validates.

### 3.2 A vacuous test — caught by review of the fixture arithmetic

The companion score test asserted only that the curves were non-empty. On the
label-eligible cohort `[1, 1, 0]`:

```
probability channel  [0.9,  0.1,  0.2 ]   AUROC = 0.5
the score fixture    [0.95, 0.05, 0.15]   AUROC = 0.5      IDENTICAL
```

It would have passed while the evaluator validated the array, projected it, and
then **ignored it**. It asserted that something happened, not that the right
thing did. Replaced with `[0.95, 0.85, 0.15]`, which yields 1.0 and separates the
channels; the smoke run confirms the baseline, the same cohort with no scores
reporting 0.5.

### 3.3 Two blank-line defects — caught by reading the applied diff

An appender idiom, `BLOCK.lstrip("\n")`, stripped each block's own leading
newlines, landing two appends hard against the preceding function. The file
parsed and nineteen tests passed.

**There is no lint gate in this repository** — no `ruff`, `flake8`, `pycodestyle`
or pre-commit configuration, and none of the five workflows runs one. `E302`
covers exactly this. Formatting discipline rests entirely on review.

### 3.4 Three post-checks that matched their own prose — caught by fixtures

Three separate installer post-checks counted a token and found their own
explanatory comments: `int(y.sum())`, `REFUSED MEANS NOT FORWARDED`, and the
score fixtures. Each refused a correct patch.

Tuning the expected counts would have fixed three instances and left the class.
The post-checks now **tokenise the source, discard every comment and string
literal, and count per enclosing function**. A count over source that discusses
itself is structurally unreliable.

---

## 4. Acceptance

`tests/unit/test_population_wiring.py`, 19 tests, all green. Every expectation is
a value the applied code produced on 2026-08-01; none is predicted. Fingerprints
are never hard-coded — the tests assert the *relationship*, because a literal
digest pins the fixture rather than the property.

Falsification was performed rather than assumed. The scores regression test was
run against the pre-fix evaluator and **failed with `PopulationError`**, then
passed against the fixed one. A regression test never seen to fail is a test
trusted on faith.

`test_existing_registry_results_do_not_move` **moved nothing** — its
`label_restricted` fixture already applied this exact idiom before POP-1a
existed, so the registry's behaviour under POP-1a was pinned by a passing test
before the commit was written.

Full suite: **4134 passed, 6 skipped, 0 failed**, 4140 collected, skip set
unchanged at 6. The armed gate — `--assert-suite-size`, which no run had used all
day — reported `suite-size ratchet OK (collected 4140 == EXPECTED_SUITE_SIZE
4140)`.

---

## 5. Sabotage

Fourteen mutations in a disposable `git worktree`. **Two defences ran first**,
because POP-1a was uncommitted and an editable install would have resolved
imports to the real `src` rather than the worktree — making every mutation land
on a file nothing imports and reporting *0 detected / 14 undetected*, the exact
inverse of the truth, into a permanent log.

`PYTHONPATH` was pinned to the worktree and `evaluator.__file__` confirmed inside
it. Then a **canary** break was required to turn the suite red, with the run
aborting as `HARNESS_ERROR` otherwise. The canary produced 37 failures.
Restoration was verified against the pristine digest after every one of fifteen
runs.

**Result: 14 mutations, 13 detected, 1 undetected, 0 anchor misses.**

### 5.1 M10, undetected and recorded

Counting positives from the attempted labels with `np.nansum` is **equivalent on
every fixture cohort**: `nansum` ignores `NaN`, and the projection removes exactly
the `NaN` rows.

| cohort | projected sum | `nansum` attempted |
|---|---|---|
| `[1,1,0,nan]` | 2 | 2 |
| `[1,1,0,0]` | 2 | 2 |
| `[nan,nan,nan]` | 0 | 0 |
| `[1,1,0,inf]` | 2 | **OverflowError** |

They diverge only on an **infinite** label, which `np.isfinite` excludes and
`np.nansum` does not. A real but narrow gap — and it raises a question deferred
deliberately: an infinity is not a *missing* label, it is a *corrupt* one, and
treating it as withheld pools it with genuine `NaN`.

### 5.2 A concentration worth knowing about

M04, M05 and M13 were each detected by **exactly one** test,
`test_the_metadata_frame_is_projected_with_the_arrays`. Deleting that single test
would make three mutations invisible.

---

## 6. PRE-1 — a dead gate on the paid-launch path

Discovered because a preflight crash was investigated rather than dismissed.

### 6.1 The crash

`Run_Preflight_Local.ps1:321` — section 13c, *"RUN_17_PLAN.md feature contract
matches the CODE"* — died with *"does not contain a method named 'Trim'"*. The
summary reported **53 passed, 1 warned, 1 failed**, and 13c contributed to none
of the three. **A gate that reports nothing when it breaks is worse than no gate,
because it looks like coverage.**

Measured cause: the import chain writes the KAN repair banner to standard error.
`2>&1` merges it as an `ErrorRecord` **object**, so the result is a two-element
array and `.Trim()` member-enumerates and dies. Exit code was 0 — the interpreter
succeeded; only the PowerShell handling failed.

**The guard for exactly this exists one line too late.** Line 322 reads
`if ($codeCount -notmatch '^\d+$') { Fail "13c: could not read..." }`. It was
written to catch a non-numeric result. Line 321 throws before it can run.

Confirmed pre-existing and not POP-1a's: the crash reproduces identically with
`evaluator.py` stashed to its pre-POP-1a state.

### 6.2 What it was hiding

| source | value |
|---|---|
| `docs/runs/RUN_17_PLAN.md:11` | `<!-- FEATURE_CONTRACT: 97 -->` |
| package | `EXPECTED_TABULAR_FEATURE_COUNT = 95` |
| `len(TABULAR_FEATURES)` | 95 |

The code is internally consistent and its contract tests pass 7 of 7. The plan's
marker is stale.

Provenance, anchored on `^[+-]EXPECTED_TABULAR_FEATURE_COUNT` rather than on
commit dates:

```
Jun 11  80 -> 81      Jun 26  87 -> 88      Jul  6  91 -> 97
Jun 13  81 -> 82      Jun 27  88 -> 91      Jul 14  97 -> 95   <- 03:16:59
Jun 17  82 -> 87
```

`RUN_17_PLAN.md` was last committed **2026-07-13 00:25**. The count moved to 95 on
**2026-07-14 03:16** — twenty-seven hours later. **The marker was correct when
written.** The plan's `91` at lines 118 and 125 is accurate history: it held from
27 June to 6 July.

And the three commits that built this defence sit fourteen minutes apart on
12 July:

```
23:26  6f4904b  plan said 91 features; the contract is 97. G1 now DERIVES it.
23:36  01afe93  make the feature-count claim machine-checkable, and emphasis-proof
23:40  721a23e  assert the feature contract, do not scrape it from prose
```

**The gate was built to catch this exact drift, and twenty-seven hours later the
drift occurred and the gate could not fire.** Its own comment reads *"Neither can
silently drift from the code again."* It has been unable to fire for eighteen
days.

### 6.3 Two commits, in this order

**PRE-1a** — filter the merged stream by type so line 322's existing guard can
run. **PRE-1b** — correct the marker `97 → 95` with a dated note.

**PRE-1a strictly first**, so the gate is *seen to fail* on the stale marker
before the marker is corrected. A repair never observed to work is not a verified
repair.

### 6.4 Two misreadings of mine, recorded

While establishing this provenance I stated the timeline wrongly **twice**, both
times by inferring chronology from a listing of commits that touched the file
rather than anchoring on commits that changed the value. First I said the code
moved on 20 and 21 July; it did not. Then I said the marker was "stale on the day
it was written, never correct"; it was correct, and went stale a day later.

Both were corrected by anchored reads. The lesson is the standing one: order must
be measured, not assumed.

---

## 7. Follow-ups — recorded, none touched

| id | item |
|---|---|
| **PRE-1a** | section 13c cannot run; the `.Trim()` on a merged `ErrorRecord` stream |
| **PRE-1b** | `RUN_17_PLAN.md` marker `97 → 95`, with dated provenance |
| **ABS-1** | the ranking channel's refusal reported as `undefined_on_cohort`; `_absence_maps` takes `ranking_check` and never reads it |
| **DEAD-1** | ~40 lines of dead absence computation in `evaluate` (1181-1220), discarded at 1240, already carrying a reason string that disagrees with the live one |
| **LINT-1** | no lint gate; `ruff` reports 603 `I001`, 500 `UP045`, 409 `BLE001`, 267 `F401` |
| **F821-1** | 18 undefined names; 7 are the deliberate `_ensure_sklearn` global injection, 9 need assessment, `metrics.py:1486` first |
| **CMP-1** | `ModelComparison` carries a population fingerprint with no population scope beside it |
| **INF-1** | an infinite reference label is pooled with `NaN` as *withheld*; it is corrupt, not missing |

`ruff` was installed during investigation and has been uninstalled; `.venv312`
matches `requirements.txt`.

---

## 8. Final state

| item | value |
|---|---|
| `evaluator.py` | `0efdcdc293b6583b95a38479f4bbce4b2b22b01facb92562e8d8efd2c4e616dd` |
| `tests/unit/test_population_wiring.py` | `e84818f105bd41b4294bdc52790757508aa2c8b2672539ab9de050f026720cfc` |
| `tests/EXPECTED_SUITE_SIZE` | `6f1c7458e81dfab02c17066758c4c2b21019d3ca0a2fc73c7c58daa495ab6eaa` |
| `README.md` | `f44a9656cd552f5995bd209a9f7d53edd0f0ccf971e92e35942e1d6a22c94938` |
| suite | 4134 passed, 6 skipped, 4140 collected |
| armed ratchet | `collected 4140 == EXPECTED_SUITE_SIZE 4140` |
| preflight | 53 passed, 1 warn (1000 Genomes absent, deferred B.D1), 1 fail (uncommitted tree, expected) |

Next: POP-1b (the report surface — `n_source`, `n_label_eligible`,
`n_reference_label_withheld`, scope and parent fingerprint, with a schema version
bump), then REG-1, then OP-1. PRE-1a and PRE-1b are promoted ahead of the
follow-up list.
