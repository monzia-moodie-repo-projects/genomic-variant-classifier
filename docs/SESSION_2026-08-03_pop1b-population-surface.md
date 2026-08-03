# SESSION 2026-08-02/03 — POP-1b: the report names its evaluation population

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Commit: `00e180c`. Pushed. 8 files, 543 insertions, 18 deletions.**
**Outcome: schema version five; full suite 4146 passed, 6 skipped, 0 failed; ratchet 4140 → 4152, armed gate verified by demonstrating it aborts.**

Companion documents:
`SESSION_2026-08-01_op1-preflight-and-defect-register.md`,
`SESSION_2026-08-01_pop1a-label-eligible-population.md`,
`SESSION_2026-08-02_pre1-preflight-contract-gate.md`.

---

## 1. What POP-1b adds, and why POP-1a withheld it

POP-1a (`1577f0b`) made `n_samples` the **label-eligible** count and deliberately
added no fields, because schema surface belongs in its own commit with its own
version bump. Landing a schema change inside a wiring change is patchwork.

The consequence was a real gap: a reader of a version-4 artifact could not tell a
smaller cohort from a narrowed one, and the artifact carried a population
**fingerprint** with nothing beside it saying what the population **was**.

Five fields, schema version five:

| field | source |
|---|---|
| `n_source` | `population.n_source` |
| `n_label_eligible` | `population.n` |
| `n_reference_label_withheld` | `population.n_excluded_from_source` |
| `population_scope` | `population.scope` |
| `population_parent_fingerprint` | `population.parent.membership_fingerprint`, or `None` |

Measured 2026-08-02 before any change: `n_label_eligible`,
`n_reference_label_withheld` and `population_parent_fingerprint` appeared
**nowhere** in the repository. `population_scope` had 31 occurrences and
`population_fingerprint` 20 — but those are `MetricMetadataKey` members on
individual results, not report-level fields. The names are consistent by design
and must not be conflated.

---

## 2. The finding that shaped the design, and it is invisible in the dataclass

`from_serialized` (`evaluator.py:648`) and `from_serialized_v2` (`738`) both do:

```python
known = {f.name for f in dataclasses.fields(cls)}
accepted = {k: v for k, v in payload.items() if k in known and k != "metric_results"}
return cls(**accepted)
```

A payload written before POP-1b contains none of the five names. **Declared
without defaults, they would raise `TypeError: missing 5 required positional
arguments` on every version-1 through version-4 artifact ever written.**

Nothing in the dataclass shows this. It appears only when the **deserialisers**
are read, which the preflight did and a reading of `EvaluationReport` alone would
not have.

### 2.1 Why the sentinel is −1 and not 0

Zero is a legitimate measurement for a cohort that was attempted and yielded
nothing. A historical artifact must not be readable as though it recorded a
measurement it never took, and a negative sentinel cannot be mistaken for one.

`test_the_sentinel_is_negative_and_not_zero` pins this.

### 2.2 `n_excluded_from_source`, not a second copy

The build specification recommended deriving `n_reference_label_withheld` as
`n_source - n`. Reading `population.py` showed that derivation **already exists**
as a named property at lines 397-400, with the docstring *"Rows absent relative
to the ORIGINAL frame, however many narrowings"* — precisely the reasoning given.

Recomputing it inline would have created a **second copy of one quantity**:
shape (a) inside the commit whose specification quotes shape (a). POP-1b reads
the property.

`n_excluded_from_parent` is the wrong distance when `compare_models` supplies an
already-restricted population — see §6.2, where a mutation proved this is
untested.

### 2.3 The three counts are enforced

`_validate_population_fields` asserts
`n_source == n_label_eligible + n_reference_label_withheld`, and refuses a
**partial recording** where two are measurements and one is a sentinel.

A field trio that must sum is exactly the shape that rots silently: any one can
drift and nothing notices. The battery **constructs violations**, because an
invariant nothing can violate in a test is an invariant nothing checks.

---

## 3. Four failures found only by the full suite

POP-1b's own battery passed. `test_bootstrap_reconciliation.py` passed at 63.
`test_explicit_absence.py` passed at 27. **Four tests failed in two files nothing
in the commit had opened.**

The preflight had measured **347 test lines** matching
`EvaluationReport|n_samples|schema`. That number is why a schema change is not
declared done on the strength of its own tests.

| test | cause | repair |
|---|---|---|
| `test_exactly_one_report_field_was_added` | a cumulative growth guard | append five names; rename |
| `test_the_typed_schema_version_is_declared_and_readable` | `SUPPORTED == {1,2,3,4}` | extend the set |
| `test_evaluate_now_emits_the_typed_schema_version` | bare `== 4` | threshold |
| `test_the_report_emits_the_typed_surface` | `== _ABSENCE` | threshold |

### 3.1 The 2b-3 snapshot fixture was NOT regenerated

The growth guard reads `tests/fixtures/report_snapshot_2b3.json`, frozen at
commit 2b-3, and lists every field added since. **Rebasing it onto today would
have made it permanently blind to everything added before now — green while
guarding nothing, which is worse than red.**

The five names were **appended**, in the order `dataclasses.fields` returns them,
measured from the live class rather than assumed. The installer verifies the
fixture's digest after writing and restores every file if it moved; that abort
path was exercised on a fixture before delivery.

### 3.2 A test name that had already gone stale

`test_exactly_one_report_field_was_added` asserted **three**. It had been
extended once and the name was left behind — and that stale name cost a full turn
of this session, reading as a historical claim about one commit when it is a
running record. Renamed `test_no_report_field_appeared_unannounced`.

---

## 4. Five true assertions protected from change

POP-1b adds version five. It does not renumber anything. These are **true and
were deliberately left alone**, each guarded by a post-check requiring it to
survive byte-for-byte:

```
_TYPED == 3          the typed capability version
_VERSION == 2        the base version
removed == []        the load-bearing half of the growth guard
_ABSENCE == 4        the absence maps did arrive at four
4/4                  the contract-test count in RUN_17_PLAN.md
```

**This session nearly retargeted three of them.** Each time the actual text
corrected the plan: `4/4` was accurate and I had claimed seven from a different
file; `_ABSENCE == 4` I twice proposed rewriting before reading its function; and
the `removed == []` line I briefly flagged as possibly-dead on a window that had
cut off one line short.

Retargeting a true statement is how a correct document becomes a false one.

---

## 5. Six defects of the author's, and not one was caught by review

Recorded in full because the pattern matters more than the instances.

**An orphaned comment.** Edit 1 anchored on `SUPPORTED_REPORT_SCHEMA_VERSIONS`
without knowing a comment sat above it, so `# Versions this codebase can READ.`
would have captioned the new constant while the line it describes was pushed
sixteen lines down. Caught by reading the rendered diff.

**A docstring in a form this file uses nowhere.** The house style is `# N <prose>`
**above** each version constant; I wrote a triple-quoted string **below** it.
Caught by reading twelve lines of the real file.

**Two names the target file does not import.** R1 and R2 used
`EVALUATION_REPORT_SCHEMA_VERSION_POPULATION` and
`SUPPORTED_REPORT_SCHEMA_VERSIONS`; `test_bootstrap_reconciliation.py` imports
neither. Both repairs would have raised `NameError`. Caught by rendering the
repairs in situ.

**A rename that would have destroyed the guarantee it was meant to preserve.**
I planned to rename `test_the_schema_version_advances_to_four` to
`..._is_the_highest_supported`. Reading its body showed the docstring claims
something else entirely — *"version 4 is what tells a reader the absence maps
exist"* — so the rename would have replaced a real guarantee with one already
covered elsewhere. Caught by refusing to patch a body I had not read.

**A fixture supplying six of thirty-seven required fields.**
`EvaluationReport` has 56 fields, 37 required. Six tests died at construction
before reaching their assertions. **The repair was not to list the other
thirty-one** — that fixture would go stale the moment a field is added, the same
failure rebuilt larger — but to derive from real reports via
`dataclasses.replace()` and payload deletion.

**A line-count constant carried between two installers that count differently.**
One counts `len(source.split("\n"))`, including the empty string after the
trailing newline; the other subtracts it. I read 297 from the first and typed it
into the second, and it **refused a byte-identical file** whose SHA-256 matched
exactly. Shape (a) at its smallest scale.

### 5.1 And a post-check that matched its own prose, four times

On `int(y.sum())`, on `REFUSED MEANS NOT FORWARDED`, on the score fixtures, and
on `dataclasses.replace(` — each time a check counted a token and found the
installer's own explanatory comment, refusing a correct patch.

Tuning the expected number fixes the instance and leaves the class. The checks
now **tokenise the source and count over code with every comment and string
literal removed**. A count over source that discusses itself is structurally
unreliable.

---

## 6. Sabotage

Eleven mutations in a disposable `git worktree`, behind the same two defences as
POP-1a: `PYTHONPATH` pinned with `evaluator.__file__` confirmed inside the
worktree, and a canary required to turn the suite red or the run aborts.

**Result: 11 mutations, 9 detected, 2 undetected, 0 anchor misses.**

### 6.1 M01 was the mutation the battery existed for

M01 removes a default from `n_source`. **DETECTED**, breaking four tests.

The distinction matters: `test_a_pre_population_artifact_still_deserialises` —
the test that proves the defaults work — **failed on 2026-08-02** for an
unrelated reason, was repaired, and only then passed. A test that has passed once
is not the same as a test that can fail. M01 shows it can.

### 6.2 Two undetected, and they are REAL GAPS

POP-1a's single undetected mutation was an **equivalent mutant**, indistinguishable
on any cohort whose non-finite labels are NaN. These are not.

**M03** substitutes `n_excluded_from_parent` for `n_excluded_from_source`. Every
fixture population is **one level deep**, where the two coincide. They diverge at
two levels — precisely the `compare_models` case cited in §2.2 as the reason for
the choice. The design decision is correct and entirely unprotected.

**M07** makes the `print_report` narrowing guard always true, so a fully labelled
cohort gains a population line it should not have. Undetected because **nothing
asserts on `print_report` output at all.** The guard exists to keep unrestricted
output byte-identical and nothing checks that claim.

Both are recorded, neither quietly repaired. Two tests would close them — a
two-level population, and a `capsys` assertion — and they belong in their own
commit with their own sabotage run.

---

## 7. The armed gate, verified by making it abort

The full suite under `--assert-suite-size` printed **nothing** about the ratchet.
The 2026-08-01 confirmation line, `suite-size ratchet OK (collected 4140 ==
EXPECTED_SUITE_SIZE 4140)`, comes from `Run_Preflight_Local.ps1` section 6, which
formats that message itself. Bare pytest is silent on success.

**Silence cannot be verified by observing silence.** So the gate was run against a
twelve-test subset:

```
actually collected:  12
4140 FEWER test(s) than expected.
*** TESTS HAVE VANISHED. ***
no tests ran in 1.05s
```

It aborts, and its message names the failure mode it exists to catch — including
the `importorskip` collapse that let the graph-neural-network branch go untested
for 508 continuous-integration runs (roadmap 6.17).

So the silence on the full run is a **genuine pass**. That is the third repair
this session proven by making it fail first, after the scores regression test and
preflight section 13c.

---

## 8. Acceptance

| item | value |
|---|---|
| full suite | 4146 passed, 6 skipped, 0 failed — twice (17m31s, 15m45s) |
| collected | 4152 |
| ratchet | 4140 → 4152 (+12), armed gate verified |
| `test_readme_claims` | 10 passed; ratchet and badge agree |
| `HEAD` = `origin/main` | `00e180c` |

Twelve, not eleven: `test_a_pre_population_artifact_still_deserialises` is
parametrised over schema versions 2 and 4, which pytest counts as two. **Both
pass**, so `from_serialized` and `from_serialized_v2` — different code paths — do
not diverge.

One standing warning: `UndefinedMetricWarning: Only one class is present in
y_true` from `test_the_metadata_frame_is_projected_with_the_arrays`, POP-1a's
test, where the clustered bootstrap resamples a three-row cohort into
single-class replicates. Inherent, informative, flagged when written.

---

## 9. Follow-ups — twelve, none touched

| id | item |
|---|---|
| **POP-1b-M03** | *new.* No test distinguishes `n_excluded_from_source` from `n_excluded_from_parent`; needs a two-level population |
| **POP-1b-M07** | *new.* Nothing asserts on `print_report` output; needs a captured-output test |
| ZERO-1 | 24 dead-connector defaults still zero — is the allowlist itself stale? |
| INF-1 | an infinite reference label is pooled with NaN as *withheld*; it is corrupt, not missing |
| ABS-1 | the ranking channel's refusal reported as `undefined_on_cohort`; `_absence_maps` takes `ranking_check` and never reads it |
| DEAD-1 | ~40 lines of dead absence computation in `evaluate`, discarded at 1240 |
| DEAD-3 | `_assert_absence_biconditional` computes `observed_curves` twice, discarding the first |
| PRE-2 | section 5's PASS line swallows the KAN banner and a progress bar |
| LINT-1 | no lint gate anywhere |
| F821-1 | 18 undefined names; 9 need assessment |
| CMP-1 | `ModelComparison` carries a fingerprint with no scope beside it |
| TEST-1 | *closed by this commit* — the tautology at `test_bootstrap_reconciliation.py:696` |

**DEAD-2 closed** by this commit. **DEAD-4 withdrawn**: `removed` is used at line
460; my reading window had cut off one line short, and I flagged it as suspected
rather than asserting it.

---

## 10. Next

**REG-1** — protected metadata ownership on every result path. The last
prerequisite before OP-1.

**OP-1** — the operating-point subsystem this sequence set out to build. It now
sits on a population that says what it describes, which is what makes its
strongest oracle mean anything.

Then the drift monitor, whose red is roadmap 6.20's fix working — *"THE SCHEDULED
DRIFT MONITOR HAD NEVER CHECKED ANYTHING"* — exactly as preflight 13c's red was
PRE-1a working.
