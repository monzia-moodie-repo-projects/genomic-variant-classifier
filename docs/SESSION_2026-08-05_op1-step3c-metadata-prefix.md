# SESSION 2026-08-05 — OP-1 step 3c: the registry metadata prefix

**Base: `HEAD = origin/main = c33c208`. Result: `58929e9`, plus `1be72e4` repairing a
red badge that `58929e9` introduced.**

**Ratchet 4278 → 4300 (+22). Full suite 4294 passed, 6 skipped, 0 failed; 4300
collected. Skip surface unchanged at 6.**

`MetricContext.support()` now has exactly one caller in `registry.py`, and
`compute` has none. Eleven of the twelve-defect register of 2026-08-01 are
closed; D12, the undeclared tie-break, remains and closes in step 4.

This document records both commits, because the second exists only because of a
sequencing defect in the first and separating them would make the record read
better than the work was.

---

## 1. What changed

`registry.compute` built its metadata inline at five `MetricResult` construction
sites. What all five genuinely shared was exactly two things —
`{METRIC_NAME: d.name}` and `**ctx.support()` — and everything else differed by
branch. That is why this is not "one finaliser both paths call", a phrase used
in earlier sessions which does not survive reading the sites.

`_registry_metadata_prefix(descriptor, ctx, pre_support=None)` returns

```
METRIC_NAME
the caller's branch-specific fields, in the order given
the COMPLETE runtime support snapshot
```

and is now the module's only caller of `support()`.

## 2. The decision: five options costed, four rejected

**A, narrow.** The helper replaces the five metadata literals; the guards keep
calling `support()` separately. Leaves the seam the adopted ruling names, and
buys nothing the others do not. Dominated.

**B, full hoist.** `support = ctx.support()` once near the top of `compute`. It
preserves insertion order perfectly, but the helper's signature becomes
`(descriptor, support)` — wrong for the step-5 caller, which holds a descriptor
and a context and would then have to call `support()` itself, reinstating the
second authority the extraction exists to remove.

B's apparent efficiency advantage does not exist. The seven `support()`
locations are **five mutually exclusive branches**; per invocation the refusal
path made two calls, the OK path two, the other three one each. The worst case
was two, and every single-snapshot option reduces it to one.

**C, helper returns both.** `(metadata, support_keys)`. The second element is
derivable from the first — the protected sets are the returned mapping's keys,
minus an exception or plus two certification keys — so it is a redundant return
discarded at three of five sites, and it puts the assembler on the wrong side of
the assembly/enforcement boundary.

**D, guards derive from the returned mapping.** Correct, and the cleanest
diff — but it moves branch fields *across* the support expansion at four of five
sites, because those sites place their extras between the metric name and the
expansion.

**F-strict, adopted.** `pre_support` carries the branch fields so they stay where
they are, and the helper refuses a caller-supplied key colliding with the metric
name or the live snapshot.

An extras-accepting helper using `**kwargs` **cannot exist**:
`MetricMetadataKey.CERTIFICATION_ELIGIBLE` is an enum member and cannot be a
keyword-argument name, so the OK site is unreachable through that shape.
`pre_support` is a Mapping for that reason, not by preference.

## 3. The seam was latent, not live — and the distinction is recorded

`compute` called `support()` twice on each verdict-bearing path: once for the
guard's protected set, once for the attached metadata.

Measured 2026-08-05: the twenty-seven lines between the OK-path snapshot at 1647
and its guard at 1675 are **entirely comment**, and `MetricContext` is a frozen
dataclass. The two snapshots cannot diverge today.

They could the moment `support()` gains a caller-visible dependency. On the
refusal path `verdict.metadata` merges **last**, so a key the first snapshot
failed to protect would overwrite a registry-owned key attached from the second
— the forgery REG-1 exists to prevent, reachable through a hole REG-1 did not
close.

The ruling is therefore prophylactic. Saying so is the point: a record that
described a latent hazard as a live one would misprice every future decision
that cites it.

## 4. Insertion order is not cosmetic, which is why D was not enough

`MetricMetadataKey` is a `(str, Enum)` mixin, so `hash(member) ==
hash(member.value)` and a plain string and its enum member are **the same
dictionary key**.

Confirmed three independent ways: the enum's own docstring, dated 2026-07-27; a
Python 3.12 replica built from the measured method resolution order; and the
live class. A first hypothesis — that `Enum.__hash__` hashes the member *name*,
making the two distinct keys — was **falsified** by the replica before it reached
any code.

So a plain-string branch field can collide with an enum support key, and
insertion order decides which value survives. It is a precedence question
wearing a formatting question's clothes.

## 5. Preserved byte for byte, proven by execution before the installer existed

A replica of all five metadata literals and both protected-set expressions was
run on a clustered **and** an unclustered context, asserting:

* `list(old) == list(new)` — key **order**, not merely equality — at every site
* both protected sets set-identical to the expressions they replace
* `support()` evaluated exactly once
* the verdict still winning `N_CLASSES_OBSERVED` on the refusal path
* the certification blocker still last on the OK path

`_reject_registry_owned_keys` is untouched. Only its fourth argument expression
changes, at two call sites.

## 6. A pre-existing structural gate detected the change, and was rewritten

`test_refusal_protected_keys_are_derived_from_ctx_support`, added 2026-08-03
after REG-1 mutation M06 went undetected, asserted that a literal
`ctx.support()` call appears **inside** the refusal guard's protected-set
expression.

Its protected property survived step 3c intact; its **instrument** recognised
only one spelling of it. It is not a false positive, not harmless staleness and
not mere brittleness, and none of those labels is used here, because each erases
the distinction between a valid property and an obsolete instrument.

### 6.1 The replacement proves the derivation

* the guard's protected set **reads a local**
* that local's **nearest preceding** assignment calls the prefix
* the prefix takes exactly **one** snapshot
* `compute` takes **none**

**Nearest** is load-bearing. The OK path rebinds `meta = {**dict(verdict.metadata),
**meta}` after its guard, so a check asking merely whether `meta` is *ever*
assigned from the prefix would stay green if the guard moved below that rebinding
and began protecting a mapping that already carried descriptor keys.

### 6.2 It now covers both guarded branches

Before step 3c the refusal and OK paths derived independently and only the
refusal path was pinned here — a gap nobody had noticed. After step 3c they share
one authority, so a gate covering one would certify half the architecture.

The pre-step-3c direct form is **deliberately not accepted**: permitting
`ctx.support()` inside a guard expression would allow a regression to two
support authorities while leaving the test green.

### 6.3 Sabotage

Eight mutations, eight detected, zero undetected, with the correct shape accepted
first. The two that matter are the two a name-recognising gate would be fooled
by: `base` assigned from an ordinary dict literal, and the OK guard moved below
the `meta` rebinding.

### 6.4 Two support-counting helpers are not duplication

`test_registry_metadata_prefix.py` keeps the general, nesting-aware form, because
it walks every function in the module and must not attribute a nested body's
calls to its parent. The gate in `test_metric_registry.py` counts calls in
exactly two named functions and **first asserts neither contains a nested
definition**, which makes a plain walk provably sound. The scopes differ, so the
instruments differ, and the simpler one carries the precondition that licenses
it.

## 7. C2-1 observed, DIAG-1 raised

Addendum A section 9 recorded as permitted-but-never-measured whether any metric
is applicable on a single-class cohort while carrying `reference_class_support`.
Enumerating the live catalogue rather than naming five metrics found
**`integrated_calibration_index`** doing exactly that.

**C2-1 is not discharged here.** Its erratum frames it more broadly — a heading
claiming a structural split that is really cohort-dependent — and discharging a
register item on a reading of its scope is the drift the register exists to
prevent. The observation is recorded; the item stays open pending a ruling.

That descriptor is applicable, its kernel returns `nan`, and `compute` takes the
non-finite branch — which does not merge `verdict.metadata` at all. Neither does
the exception branch. So a `FAILED` result reports what the kernel returned but
not the cohort fact that explains it. Pre-existing, unchanged by step 3c, and
outside its scope: **DIAG-1**. The absence is *asserted* in the test, so resolving
DIAG-1 must update it deliberately rather than let it pass by accident.

## 8. Zero new skips

A first version of the diagnostics test named five metrics and skipped the two
that attach nothing on the cohort. Naming carriers is guessing, and a skip is not
coverage: it reports success while asserting nothing.

Replaced by one loop over all twenty-four registered descriptors with its
non-vacuity assertion folded inside it, and `_missing_inputs` asserted empty for
every descriptor so a future one requiring a new input **fails by name** instead
of dropping silently out of coverage. A second latent skip, on a deterministic
certification condition, became an assertion.

Coverage went from five named metrics to twenty-four; new skips went from two to
zero; collected cases fell by five, which is why the ratchet reads +22 and not
+27.

## 9. Fifteen defects of the author's, and the pattern matters more than the count

**Four were caught by gates that refused a write.** A repository-wide
`*_manifest.json` sweep that rejected three committed project artifacts. A
guard-body statement count asserted at 2 where the body has 3, refusing a patch
that changed nothing in the function it guarded. A fatal `\U` escape in a non-raw
docstring containing a Windows path. A stale-name check written as a raw
substring search, which fired on the replacement's own docstring quoting the name
it replaced.

**The tenth is the one worth the space.** A stand-in file written to verify the
test patch imported `ast` at module level; the real `test_metric_registry.py`
does not, and imports it locally inside the bodies that need it. The fixture was
chosen to agree with its author, the patch was reported verified, and it failed
with `NameError` on the first real run. Not a wrong belief encoded in a check — a
check that only looked where the author had already looked.

**Then a citation, and a search that agreed with the author for the wrong
reason.** A draft cited a prior invalid-escape defect at "line 1966" of the
ratchet, from memory. The search written to verify it used needles including
`download_finngen`, which occurs at line 594 inside an enumeration of script
names — so the check reported success on a line that records nothing. Narrowed to
diagnostic phrases, the third attempt found the genuine record at line 602. The
recollection was right about the defect and wrong about the location; only the
third attempt measured it.

**And the fifteenth reached the remote.** See section 10.

The remaining four: a phantom revert record (section 10.1); a diagnostics test
asserting a contract that never existed; a ratchet-entry format derived from the
maximum line width of the preceding entry, inheriting a single 92-character
outlier; and a placeholder duration inside a runnable command block, pasted
verbatim — the third time this session a wrong value inside a pasteable block was
pasted, with the instruction to substitute it printed only in prose beneath. That
class is closed by measurement: the ratchet installer parses the suite output
instead of accepting numbers by argument.

## 10. The one that reached `origin/main`

`58929e9` moved the ratchet to 4300 and left the README badge at 4278.
`test_readme_test_count_equals_the_suite_size_ratchet_exactly` asserts equality
with no tolerance, and it failed on the pushed commit. `1be72e4` repairs it.

**Sequencing, not coverage.** The full suite ran green *before* the ratchet bump.
Afterwards only `--collect-only --assert-suite-size` ran, which exercises
collection and executes nothing. So the one test coupling the README to the
ratchet never ran against the committed tree.

The evidence was already in hand and had been quoted in the same session: the
ratchet's reconstructed history names commit `50bb9fa` as "derive the README
badge", and the CERT-1 delta records its commit carrying "the ratchet **and the
badge**". The ratchet and the badge move together in this repository, and the
step-3c ratchet installer did not touch the badge.

**The remedy is a rule, not a resolution.** A ratchet bump changes the tree after
the last executing run, so the sequence must be: bump, then *execute* — at
minimum the README and ratchet tests, which take thirty seconds — and only then
commit. A collection-only check cannot substitute, because the coupling it must
catch lives in an assertion.

An armed full suite was run afterwards against `1be72e4`: 4294 passed, 6 skipped,
0 failed in 17m10s, with `--assert-suite-size` enforced by execution.

### 10.1 A phantom revert record

The step-3c code installer backed up `tests/EXPECTED_SUITE_SIZE` and then
deliberately never wrote it, so its manifest listed a file it had not touched.
Running `--revert` would have restored the ratchet to 4278.

This is not inference. The ratchet installer's manifest, written three and a half
hours later, recorded the **identical** `sha256_before` —
`b87848659c85a5b7…` — for the same file, proving both backups captured
byte-identical content and the first installer never wrote it. Both manifests
were read before deletion, which is why the evidence exists at all.

## 11. Acceptance

| | |
|---|---|
| base | `c33c208` |
| result | `58929e9`, then `1be72e4` |
| `registry.py` | `833816545b2a342f…` → `b3fe3d11416fd9f2…` |
| ratchet | 4278 → 4300 (+22), measured by the installer |
| collected | 4300, measured on the staged tree |
| full suite, armed | 4294 passed, 6 skipped, 0 failed, 33 warnings in 17m10s |
| skip surface | unchanged at 6 |
| new tests | 18 functions, 22 cases in `test_registry_metadata_prefix.py` |
| `test_metric_registry.py` | 53 functions, 58 cases — unchanged; one gate replaced by one gate |
| sabotage | 8 mutations, 8 detected, 0 undetected |
| production files touched | one |

The 33 warnings are the pre-existing scikit-learn degenerate-cohort warnings and
are identical in count to the pre-bump run. A duration of 14m36s appears in the
ratchet entry; it comes from the pre-bump run, whose counts the armed run
corroborates exactly. No test asserts a duration.

## 12. Next

**Step 4 — the selector, Objective A**, which closes **D12**, the last of the
twelve. **Step 5 — the shadow comparison.** **Step 6 — the cutover**, which must
reckon with **GUARD-1**: `test_computation_path_guards.py` asserts every applied
threshold is `(0.5, ">=")`, and the exact sweep applies every unique score.

Twenty-nine follow-ups are open. The six raised here — SUPPORT-1, MERGE-1,
JSONKEY-1, RENDER-1, TYPING-1, DIAG-1 — and the two raised while surveying the
documentation — CHANGELOG-1, CHANGELOG-2 — are all recorded and none is touched.
