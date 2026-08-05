# HANDOFF ADDENDUM A — verbatim source for OP-1 step 3c

**Author: Monzia Moodie**
**Written 2026-08-05, after `docs/HANDOFF_2026-08-05_op1-step3c.md` was found
deficient in use.**

**This document is SELF-SUFFICIENT. It requires no repository access, no command
execution, and no project-knowledge lookup. Everything an installer needs to
anchor against is quoted verbatim below.**

---

## 0. THE DECISION, FIRST

The parent handoff buries this in its Part 5. It is the only open question and it
belongs at the top.

`registry.compute` calls `ctx.support()` **twice on each verdict-bearing path** —
once to build the guard's protected set, once to build the metadata. The adopted
ruling requires a **single snapshot**, because two calls create a
time-of-check/time-of-use seam.

Satisfying that widens step 3c from **five assembly sites** to **seven
locations**, and the two extra locations are REG-1's guard calls.

**Option A — narrow.** The helper replaces only the five metadata literals; the
guards keep calling `ctx.support()` separately. Leaves the seam the ruling names.

**Option B — full hoist.** `support = ctx.support()` once near the top of
`compute`, threaded into the helper and both protected-set derivations. Satisfies
the ruling; changes the guard call sites; the helper then takes `support` as a
parameter rather than calling `ctx.support()` itself.

**Option C — helper returns both.** `_identity_and_support_metadata` returns
`(metadata, support_keys)` so the caller derives its protected set from the same
snapshot. One call, no hoist, but a tuple return.

**Recommend one with reasoning and wait for a ruling before writing code.**

---

## 1. THE PARENT HANDOFF'S CONTRADICTION, RESOLVED

Its **Part 10** says the handoff must still be copied into the repository and
committed. Its **starting prompt (Part 8)** says the handoff is in the
repository.

**The handoff IS committed.** Part 10 was written before the commit and the
prompt after; they were never reconciled. Part 10's instruction is spent — do not
re-run it.

That is the same defect REC-1 §1 recorded in the CERT-1 delta: a document
disagreeing with its own body. Recorded here rather than by editing the parent,
on the convention that a correction sits beside a record.

---

## 2. `registry.compute` — THE FIVE SITES, VERBATIM

Read from `inspect.getsource(R.compute)` on 2026-08-05. The function begins at
`registry.py:1570`. **Line numbers below are RELATIVE to the function**; the
absolute line is `1569 + relative`.

Indentation is exact. These are the anchors.

### Site 1 — relative 10-16, absolute 1579-1585

```python
    missing = _missing_inputs(d, ctx)
    if missing:
        return MetricResult(
            value=nan, status=MetricStatus.NOT_APPLICABLE,
            reason="required_inputs_missing",
            metadata={MetricMetadataKey.METRIC_NAME: d.name, "missing_inputs": list(missing),
                      **ctx.support()})
```

### Site 2 — relative 37-44, absolute 1606-1613 (the refusal branch)

```python
        _reject_registry_owned_keys(
            d, ctx, verdict,
            frozenset({MetricMetadataKey.METRIC_NAME} | set(ctx.support()))
            - _DESCRIPTOR_OWNED_ON_REFUSAL)
        return MetricResult(
            value=nan, status=verdict.status, reason=verdict.reason,
            metadata={MetricMetadataKey.METRIC_NAME: d.name, **ctx.support(),
                      **dict(verdict.metadata)})
```

**Note: `ctx.support()` appears TWICE here** — relative 39 and relative 43. This
is one of the two seams.

### Site 3 — relative 46-57, absolute 1615-1626

```python
    try:
        raw = d.function(ctx)
    except Exception as exc:                                   # noqa: BLE001
        return MetricResult(
            value=nan, status=MetricStatus.FAILED,
            reason="metric_computation_failed",
            metadata={MetricMetadataKey.METRIC_NAME: d.name,
                      "exception_type": type(exc).__name__,
                      # the message is recorded for a human, but the machine-
                      # readable reason above is stable; exception text is not.
                      "exception_message": str(exc)[:200],
                      **ctx.support()})
```

### Site 4 — relative 59-67, absolute 1628-1636

```python
    value = float(raw)
    if not np.isfinite(value):
        # APPLICABLE and non-finite is an implementation defect, not a property
        # of the cohort. Calling it UNDEFINED would blame the data.
        return MetricResult(
            value=nan, status=MetricStatus.FAILED,
            reason="applicable_metric_returned_non_finite",
            metadata={MetricMetadataKey.METRIC_NAME: d.name, "returned": repr(raw),
                      **ctx.support()})
```

### Site 5 — relative 75-78, absolute 1644-1647 (the OK path, base)

```python
    eligible, why = _certification_eligibility(d, ctx)
    meta = {MetricMetadataKey.METRIC_NAME: d.name,
            MetricMetadataKey.CERTIFICATION_ELIGIBLE: eligible,
            **ctx.support()}
```

### Site 5 continued — relative 106-115, absolute 1675-1684

```python
    _reject_registry_owned_keys(
        d, ctx, verdict,
        frozenset({MetricMetadataKey.METRIC_NAME,
                   MetricMetadataKey.CERTIFICATION_ELIGIBLE,
                   MetricMetadataKey.CERTIFICATION_BLOCKED_BY}
                  | set(ctx.support())))
    meta = {**dict(verdict.metadata), **meta}
    if not eligible:
        meta[MetricMetadataKey.CERTIFICATION_BLOCKED_BY] = why
    return MetricResult(value=value, status=MetricStatus.OK, metadata=meta)
```

**Note: `ctx.support()` appears TWICE on this path too** — relative 78 and
relative 111. This is the second seam.

**Note the OPPOSITE MERGE PRECEDENCE.** Site 2 puts `verdict.metadata` LAST;
site 5 puts it FIRST (`{**dict(verdict.metadata), **meta}`). **Whether that is
intentional is NOT ESTABLISHED.** Preserve both byte-for-byte.

---

## 3. `_reject_registry_owned_keys` — VERBATIM, WHOLE

```python
def _reject_registry_owned_keys(d: "MetricDescriptor", ctx: "MetricContext",
                                verdict: "Applicability",
                                protected: frozenset) -> None:
    """Refuse descriptor metadata that would overwrite a registry-owned key.

    EXTRACTED 2026-08-03 (REG-1) FROM THE OK BRANCH, WHERE IT WAS THE ONLY PLACE
    IT RAN. The refusal branch merged `verdict.metadata` LAST with no check, so
    the same metadata that raised on the applicable path was silently accepted on
    the refusal path -- the branch whose whole purpose is saying what evidence
    base the refusal describes. A refusal could claim a membership fingerprint it
    never examined, and "n=980 beside n=980 says nothing about WHICH 980".

    THE PROTECTED SET IS A PARAMETER because the two paths DO NOT OWN THE SAME
    KEYS. A first version of this change derived one set for both and turned 29
    tests red: `auroc` refusing a single-class cohort reports N_CLASSES_OBSERVED
    as the GROUND of its refusal, and the guard called that a violation. The
    derivation is still single-sourced; only the ownership differs, because the
    paths genuinely differ.

    REJECTED, NOT SHADOWED. Merge order would also prevent the overwrite, but
    silently: the descriptor's value would vanish and its author would get no
    signal. That reasoning is recorded in `compute` and is why neither branch's
    merge order was changed.
    """
    overlap = protected & set(verdict.metadata)
    if overlap:
        raise RegistryInvariantError(
            f"{d.name}: applicability metadata attempted to set registry-owned "
            f"key(s) {sorted(str(k) for k in overlap)}. A descriptor states what "
            "is true of the COHORT; the registry states what is true of the "
            "RESULT, and the two must not be able to disagree.")
```

**Four lines of body. Returns `None`. Assembles nothing.** The boundary between
assembly and enforcement is exactly where the extraction wants to cut.

```python
_DESCRIPTOR_OWNED_ON_REFUSAL = frozenset({MetricMetadataKey.N_CLASSES_OBSERVED})
```

---

## 4. `MetricContext.support()` — VERBATIM, WHOLE

```python
    def support(self) -> dict:
        """The evidence base a metric was computed over.

        Attached to EVERY result, refusals included. A metric computed on twelve
        rows and one on four hundred thousand are not equally trustworthy, and
        without this an artifact cannot say which it holds. The cohort size
        behind a REFUSAL is equally informative: an INSUFFICIENT_SUPPORT on 3
        rows and one on 300,000 point at different problems.

        `n_clusters` counts DISTINCT clusters, not an effective sample size.
        Effective sample size under clustering is a property of a resampling
        design and already lives in BootstrapResult beside the design effect and
        replicate accounting. Duplicating an approximation here would create a
        second, weaker answer to a question already answered properly.

        NO THRESHOLD IS APPLIED. Whether a minimum observation or cluster count
        should block certification is a scientific policy decision; inventing one
        silently is the class of guess this project removes.
        """
        out = {MetricMetadataKey.POPULATION_SCOPE: self.population_scope,
               MetricMetadataKey.POPULATION_FINGERPRINT:
                   self.population.membership_fingerprint,
               MetricMetadataKey.N_OBSERVATIONS: self.n,
               MetricMetadataKey.N_CLASSES_OBSERVED: self.n_classes_observed}
        if self.clusters is not None:
            out[MetricMetadataKey.N_CLUSTERS] = self.n_clusters
        return out
```

**Four keys unconditionally, `N_CLUSTERS` conditionally.** This is why the helper
must expand the mapping wholesale and enumerate nothing.

---

## 5. `_certification_eligibility` — VERBATIM, WHOLE

Included because step 3c must not disturb it, and because REG-REASON-1 concerns
its return type.

```python
def _certification_eligibility(d: MetricDescriptor, ctx: MetricContext) -> tuple:
    """May a computed value support a certified claim?

    Separate from `status`, which answers whether a value exists at all. The
    bootstrap work established the same separation for intervals: `status` asks
    "was an interval produced?", `certification_eligible` asks "is it admissible
    for certified claims?" -- independent axes.

    Returns (eligible, reason_if_not).
    """
    if not ctx.has_both_classes:
        return False, "single_class_cohort"
    if ctx.n == 0:
        return False, "empty_cohort"
    # AN UNATTRIBUTED POPULATION CANNOT SUPPORT A CERTIFIED CLAIM (2026-07-28).
    #
    # A certified claim asserts something about a NAMED set of rows. An
    # unattributed population has no source identity, so its membership
    # fingerprint is absent and comparison against any other population returns
    # UNKNOWN rather than SAME or DIFFERENT. A claim that cannot be tied to an
    # identifiable cohort is not certifiable, however sound its arithmetic.
    #
    # This does NOT make the value wrong or the evaluation useless: unattributed
    # evaluation is a legitimate exploratory mode, and every metric still reports
    # its status, value and support. It makes the ADMISSIBILITY explicit rather
    # than leaving a reader to infer it from an absent fingerprint.
    if not ctx.population.is_attributed:
        return False, "unattributed_population"
    return True, None
```

---

## 6. `compute`'s SIGNATURE AND OPENING

```python
def compute(d: MetricDescriptor, ctx: MetricContext) -> MetricResult:
    """Compute ONE registered metric, or explain why it was not computed.

    The order is load-bearing. Inputs, then applicability, then -- only then --
    the kernel. A metric ruled inapplicable is NEVER INVOKED, so a finite but
    scientifically unsupported number cannot be produced and then explained away.
    """
    nan = float("nan")
```

**The descriptor is bound as `d`, not `descriptor`.** The ruling's helper
signature names its parameter `descriptor`; call sites pass `d, ctx`.

**If Option B is chosen, `support = ctx.support()` goes immediately after
`nan = float("nan")`** — before `_missing_inputs`, so all five sites and both
guards can use it.

---

## 7. THE ONE PRODUCTION CALLER — VERBATIM

`registry.py:1728`, inside `evaluate_registered`:

```python
    return {d.name: compute(d, ctx) for d in chosen}
```

Everything else that calls `compute` is a test: 18 sites in
`tests/unit/test_metric_registry.py`, 1 in
`tests/unit/test_oracle_c1_count_path.py`.

---

## 8. WHAT THE PARENT HANDOFF GOT WRONG, SO IT IS NOT TRUSTED BLINDLY

* **It paraphrased the five sites instead of quoting them**, then instructed the
  reader not to re-derive Part 4. An installer needs byte-exact anchors; a
  paraphrase cannot produce one.
* **Part 10 contradicts Part 8** on whether the handoff is committed. It is. See
  §1.
* **It assumed live-repository access.** Project knowledge is a July-era snapshot
  with no `evaluation/` package, so every "re-measuring command" routes through
  the human. This addendum removes that dependency for step 3c.
* **It was 29,009 bytes in one file**, with the specification in the middle.
  Length was the failure mode.

---

## 9. WHAT REMAINS UNVERIFIABLE WITHOUT THE LIVE REPOSITORY

State these plainly rather than assuming:

* **whether `registry.py` has changed since 2026-08-05.** `HEAD` was `05d4261`
  plus the handoff commit. Confirm with `git log --oneline -3` before anchoring.
* **whether the opposite merge precedence is intentional.** Not established;
  preserved byte-for-byte for that reason.
* **whether any metric is applicable on a single-class cohort while carrying
  `reference_class_support`.** Permitted by `registry.py:662-668`, never observed
  in measurement. This is **C2-1**.

---

## 10. THE SIX TESTS STEP 3c OWES

1. a direct contract test for the helper — returns exactly
   `{METRIC_NAME} | set(ctx.support())`
2. a conditional-key test proving `N_CLUSTERS` propagates **without the test
   enumerating it** — compare against `ctx.support()` at runtime
3. every construction path carries the base mapping — drive all five branches
4. descriptor diagnostics preserved — a refusal carrying
   `reference_class_support` still carries it
5. **refusal asymmetry preserved** — `auroc` on a single-class cohort does not
   raise `RegistryInvariantError`. This is the case that turned 29 tests red in
   REG-1's first attempt
6. a structural one-authority gate on the **abstract syntax tree**, not raw text
   counts
