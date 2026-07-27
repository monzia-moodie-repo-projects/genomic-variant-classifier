# SESSION 2026-07-27 -- the metric registry

**Branch point:** `origin/main` at `d3851a3`, ratchet 3185, tree clean.
Roadmap Tier 1 item 6, commit 2 of 3.

---

## 1. THE PROBLEM THE REGISTRY SOLVES

The project runs TWO conventions for a metric's answer. The panels return
`MetricResult` -- a value that always knows whether it is a value. The kernel in
`metrics.py` returns BARE FLOATS, and `evaluate()` returns a plain `dict`.

A bare float cannot say "I am not a value." What that costs is recorded in
`evaluate()`'s own docstring, on `y = [1,1,1,1]`, `p = [.9,.8,.85,.95]`:

```
auroc NaN   auprc NaN            <- correct, ranking is undefined
brier 0.01875   ece 0.125        <- NUMBERS
calibration_valid True           <- asserting those numbers are sound
```

That 0.125 is merely `1 - 0.875`: the gap between the mean prediction and the
only label present, from a reliability diagram with one occupied row.

**THAT SPECIFIC CASE WAS FIXED ON 2026-07-21**, inside `evaluate()`, by widening
the calibration gate beyond `is_probability(p)`. Verified against the current
implementation on 2026-07-27: the same input now returns `brier NaN`, `ece NaN`,
`calibration_valid False`. My first draft of the registry docstring presented the
defect as LIVE. That was wrong and is corrected -- it is quoted as the worked
example of why a bare float is the wrong return type, not as a live bug.

The general problem the fix did not solve: every new metric must remember to
return NaN and every new caller must remember to check. The registry makes the
status structural.

---

## 2. APPLICABILITY IS DECIDED BEFORE THE KERNEL IS CALLED

A post-hoc `status = UNDEFINED if isnan(value) else OK` could not have caught the
case above, because 0.125 is not NaN. So the order is load-bearing:

```
validate the context ONCE
  -> evaluate the descriptor's applicability predicate
      -> if inapplicable, return a typed MetricResult WITHOUT invoking the kernel
      -> otherwise invoke, then validate what came back
```

An inapplicable metric is never computed. A test proves the ORDER rather than the
outcome, using a kernel that raises `AssertionError` if it is ever called.

An applicable metric that returns a non-finite value has FAILED -- an
implementation defect, not a property of the cohort. Calling it UNDEFINED would
blame the data and send the fix to the wrong place.

---

## 3. THREE AXES, KEPT SEPARATE

```
status                   does a value exist at all?
scientific interpretability   is the estimand meaningful for this cohort?
certification_eligible   may a certified claim rest on it?
```

On a single-class cohort:

```
auroc / auprc / auprc_gain   UNDEFINED             the estimand does not exist
expected_calibration_error   INSUFFICIENT_SUPPORT  computable, uninterpretable
brier_score / log_loss       OK, cert=False        correct proper score, and
                                                   inadmissible for a claim
```

**A DEFECT IN MY FIRST DRAFT.** `certification_eligible` was hard-coded `True`
for every OK result, collapsing the third axis into the first -- the exact
collapse the design ruling told me to avoid. Now derived by
`_certification_eligibility`, with `certification_blocked_by` recording why. This
mirrors the bootstrap separation, where `status` asked "was an interval
produced?" and `certification_eligible` asked "is it admissible?".

---

## 4. A TYPE THAT DOES NOT EXIST

The design called for `supported_capabilities: frozenset[EvaluationCapability]`.
There is no `EvaluationCapability` in this project. `CapabilityState` exists and
measures a different axis -- how far a capability has PROGRESSED, from
NOT_IMPLEMENTED to VALIDATED -- and a metric does not "support" NOT_IMPLEMENTED.

Rather than invent a type to match a name, the static filter is `required_inputs`
(concrete and checkable) and the real gate is the applicability predicate, which
sees the actual context. Recorded so a later reader knows the omission was
deliberate.

The status vocabulary also proved richer than the design assumed: NINE values,
not four. `NOT_APPLICABLE`, `INSUFFICIENT_DATA` and `NOT_IMPLEMENTED` already
existed and are used rather than approximated.

---

## 5. THE DECLARATION IS IMMUTABLE

`_METRICS` is a frozen tuple, mirroring `monitoring/registry.py`, not a mutable
`dict[str, Callable]` anything can write to at import. `_validate_registry` runs
at import over that fixed declaration: unique lower-case names, non-empty
required inputs, LABELS always required, `requires_clusters` implying the
CLUSTERS input, a mandatory applicability policy, a callable function. A
malformed declaration fails the IMPORT, not the run.

`MetricContext` validates alignment ONCE. Descriptors never reinterpret array
lengths -- the defect `CleanArrays` was built to remove, where independent masks
produced two arrays of the same length describing DIFFERENT ROWS and every
calibration metric silently paired a probability with the wrong label.

---

## 6. WHAT IS DELIBERATELY UNTOUCHED

`metrics.evaluate()` is unchanged and remains the legacy untyped compatibility
interface. It is NOT registered as one composite descriptor: its five metrics
have five different applicability rules, and one capability decision cannot
honestly govern all of them. A test asserts the registry never CALLS it.

---

## 7. THREE DEFECTS OF MY OWN, ALL CAUGHT

1. **The docstring described a fixed defect as live.** Caught by running the
   legacy path and comparing, rather than trusting its docstring.
2. **`certification_eligible` hard-coded True.** Caught by printing the metadata
   for a single-class cohort instead of only the status.
3. **A guard that grepped source text.** `test_the_registry_does_not_call_metrics_evaluate`
   searched raw source and failed on the module docstring, which discusses
   `evaluate()` at length. A textual guard cannot distinguish a reference from a
   call. Rewritten to parse the ABSTRACT SYNTAX TREE, and proven to still catch a
   real wrap: sabotaging `_auroc` to call `evaluate()` fails it correctly.

And one repeat: the `StrEnum` floor guard fired on a docstring saying "not
StrEnum" -- the identical trip from the 2026-07-26 bootstrap session. The guard
permits the BACKTICKED spelling; the docstring now uses it and says why.

---

## 8. VERIFIED

```
27 registry tests + 16 relocation tests
385 passed across twelve affected files
3.10 floor: the registry RUNS, not merely imports
no-scikit-learn contract: green
sabotage: inapplicable metric never invoked; missing inputs never invoked;
          NaN from an applicable metric is FAILED not UNDEFINED; a kernel
          exception preserves metric identity and a STABLE reason; a finite
          value cannot confer certification eligibility
hygiene: LF only, no byte-order mark, pure ASCII
```

Ratchet 3185 -> 3212 (+27).

Next: commit 3, orchestration adoption -- the evaluator and report path consume
registered results, with the compatibility boundary explicitly maintained.

---

---

## 9. THE 2026-07-27 SCOPE DOCUMENTS: WHAT WAS ADOPTED, WHAT WAS REJECTED

Two documents arrived after this module was drafted:
`REGRESSION_CQR_PROJECT_SCOPE_v1.md` and a review of it,
`revised_metric_and_analytic_scope`.

The review's own assessment of the first document is `Direct implementation
readiness: No`, with roughly 25 per cent to be removed from the immediate
specification. It is a superseded design document, not a build order. Its
architectural principles, however, read as a codification of what this session
already established -- including "claim golden reproduction without loading a
golden artifact", which is verbatim the defect fixed in the R2 reconciliation
hours earlier.

The review's central ruling is that regression and conformal quantile regression
MUST NOT create a third evaluation convention, and it names the target
architecture as capabilities -> registry -> bootstrap -> versioned artifact.
That is what commits 1 and 2 build.

### 9.1 ADOPTED: support attachment

The review lists six orchestrator responsibilities. Five were already
implemented. The sixth -- "attaching sample and cluster support" -- was not.

Every result now carries `n_observations` and `n_classes_observed`, plus
`n_clusters` when clusters are supplied, and it is attached to REFUSALS and
FAILURES as well as to values. An INSUFFICIENT_SUPPORT on 3 rows and one on
300,000 point at different problems, and nothing previously recorded which had
occurred.

`n_clusters` counts DISTINCT clusters. It is deliberately not an effective sample
size: that is a property of a resampling design and already lives in
BootstrapResult beside the design effect and replicate accounting. A second,
weaker answer to a question already answered properly is how two numbers come to
disagree. A test asserts the support dictionary holds exactly three keys and
nothing named "effective" or "design_effect".

NO THRESHOLD IS APPLIED. Whether a minimum observation or cluster count should
block certification is a scientific policy decision; inventing one silently is
the class of guess this project removes. A two-row cohort reports OK and records
that it had two rows.

### 9.2 REJECTED, because each would degrade a correction already made

  * `MetricResult(value=None, status=FAILED, ...)` -- section 7 of the scope
    document. The project's invariant requires NaN for a non-OK result. Tested:
    `TypeError: ufunc 'isfinite' not supported for the input types`. The
    specification's example does not construct against the real class.
  * `metric_ok(value, certification_eligible=True)` unconditionally. That is the
    defect caught in this module's own first draft: on a single-class cohort it
    would certify a Brier score. Eligibility is DERIVED. The reviewing document
    independently requires "recording certification eligibility", so the two
    documents disagree and the reviewer is right.
  * `ApplicabilityRule -> MetricResult | None`, with None meaning applicable.
    Such a rule may return ANY result, including an OK one carrying a value, so
    "ruled inapplicable" and "computed" become indistinguishable at the type
    level. `Applicability` refuses that structurally: an applicable verdict
    carries no status or reason; an inapplicable one requires both.

All three rejections are recorded in the module docstring, in place, so the
decisions are durable rather than re-litigated.

### 9.3 A defect worth carrying forward

The review found, in the conformal quantile regression prototype:

```python
raw_lower = np.minimum(lower, upper)     # crossing destroyed here
raw_upper = np.maximum(lower, upper)
...
np.mean(np.asarray(raw_lower) > np.asarray(raw_upper))   # necessarily 0.0
```

The crossing rate is measured AFTER the crossing has been erased, so it is
structurally zero and reports perfect health. That is the same disease as
`n01 + n11 == 203` and the tautological estimate pin: a check that cannot fail.
It belongs in the regression workstream, recorded here so it is not rediscovered.

### 9.4 A process failure of my own

While folding in support attachment, `registry.py` reached a state I could not
account for: it contained a `support()` implementation that was not in the
hash-verified payload and that my edit script had aborted before writing. Rather
than ship a file whose provenance I could not explain, I reset to the packaged
payload -- verified by hash as `ff1cfd3b93deb048` -- and rebuilt the change
deterministically, confirming the clean base with precise checks first. The
unexplained version was preserved for comparison rather than discarded.

Ratchet 3185 -> 3220 (+35).

---

---

## 10. SECOND REVIEW: population_scope, and a principle

### 10.1 One recommendation was already implemented

The review recommended replacing `ApplicabilityRule -> MetricResult | None` with a
dedicated `ApplicabilityDecision` type. That is exactly what this registry
already does, under the name `Applicability` -- same four fields, plus a
`__post_init__` the recommendation does not have: an applicable decision carries
no status or reason, an inapplicable one REQUIRES a non-OK status and a nonempty
reason. The rejection recorded in section 9.2 was of the SPECIFICATION's
`MetricResult | None`, not of the concept. We agree.

### 10.2 ADOPTED: population_scope, required

`MetricContext` will not construct without naming its population, and the name
travels into every result.

Support counts alone do not identify the DENOMINATOR. This session produced two
cases where correct numbers described different populations and the difference
was invisible:

  * 53 and 63 were both correct, over universes that differed by ten variants,
    and the word "canonical" was applied to both;
  * 85 was printed beside 107 as a breakdown of 107, where 85 + 107 = 192.

A number without its population is not evidence. Two results with identical
support counts and different denominators are now distinguishable in the
artifact, asserted by a test that constructs exactly that pair.

The support dictionary is now exactly four keys -- `population_scope`,
`n_observations`, `n_classes_observed`, `n_clusters` -- and a test still refuses
anything named `effective` or `design_effect`, because effective sample size
belongs to the resampling design in BootstrapResult.

### 10.3 ADOPTED as a standing principle

    Preserve raw state until diagnostics complete. Canonicalisation occurs only
    after diagnostic measurements have been computed.

Three defects this session shared one shape -- destroy the distinction, measure
the destroyed distinction, declare success:

```
n01 + n11 == 203     held only after applicability had been erased
85 and 107           printed as a partition after the overlap was forgotten
np.minimum/maximum   sorted the quantile bounds BEFORE the crossing rate was
                     measured, so that rate is structurally zero
```

It applies to quantile crossings, overlapping populations, duplicate mappings,
cluster identities, calibration exclusions and bootstrap degeneracy. Recorded in
the registry module docstring so it is read by whoever extends it next.

### 10.4 An install accident, and the guard that caught it

The metric registry was installed twice. The FIRST run used an installer still in
Downloads from before the support-attachment repackage, so it installed the
PRE-support registry -- `ff1cfd3b93deb048`, 27 tests -- and moved the ratchet to
3212. The SECOND run used the current installer and correctly ABORTED with
"working tree is not clean", refusing to install over the first run's uncommitted
state.

Nothing was committed and nothing was lost. The lesson is that a stale installer
in Downloads is indistinguishable by name from a current one, so the installer
now prints the payload hash it expects BEFORE any file is touched, and the
recovery path resets to the last commit rather than layering.

Ratchet 3185 -> 3227 (+42).

---

*Written 2026-07-27.*
