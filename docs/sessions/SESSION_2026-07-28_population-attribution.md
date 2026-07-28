# SESSION 2026-07-28 — population attribution (commit 3b-0)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `132bcc2`, ratchet 3445
**Roadmap position:** Tier 1 item 6, commit 3b-0 — prerequisite to the legacy projection
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. The blocker this commit removes

Commit 3b makes `evaluate()` build a `MetricContext`, which requires an
`EvaluationPopulation`, which required a non-empty `source_id`. But `evaluate()`
receives arrays, not a `CanonicalVariantTable`. It has no source identity to give.

Three ways of inventing one were measured, and all three fail:

| strategy | outcome |
|---|---|
| one fixed sentinel string | two DIFFERENT equal-length cohorts share a fingerprint — the fingerprint certifies an equivalence nobody established, which is the exact defect it exists to prevent |
| derive from the labels | ruled out 2026-07-27: a label-policy change must be visible through `cohort_version`, not embedded in an opaque row digest |
| per-call unique identifier | safe, but NON-DETERMINISTIC: the artifact changes every run, breaking reproducibility and every byte-identity oracle in this project |

I initially proposed the sentinel. That was wrong, and the ruling of 2026-07-28
explains precisely why: combined with the normal fingerprint algorithm, a
sentinel produces a value that **looks cryptographically authoritative while
identifying only `sentinel + n_source + positions`**. A reader might notice the
sentinel; every generic comparison of `population_fingerprint` would not.

## 2. Absence represented as absence

`source_id` is now `Optional[str]`. `None` means the caller could not identify
the frame — a real and common state, now representable without invention.

An unattributed population has **no fingerprint at all**:

    attributed    membership_fingerprint = "sha256:..."
    unattributed  membership_fingerprint = None

That is not a fingerprint of nothing; it is the absence of one. A digest there
would let `a.fingerprint == b.fingerprint` answer `True` for two populations
whose equivalence is unknown.

A blank string is still refused. `None` states "unattributed"; a blank string
states nothing at all, and admitting it would give two spellings of absence, one
of them accidental.

## 3. Comparison is three-valued, because two are not enough

    SAME       proven to describe the same rows
    DIFFERENT  proven to describe different rows
    UNKNOWN    not knowable from the provenance available

`compare_membership` is the authoritative comparison. A boolean cannot express
"not knowable", and collapsing it into `False` would read as "different rows",
which is itself a claim.

**The trap this closes is exact, and is pinned by a test.** `None == None` is
`True` in Python, so a caller comparing two absent fingerprints directly
concludes sameness. The test asserts that the naive comparison returns `True`
*and* that the authoritative comparator returns `UNKNOWN`, so the divergence
between the obvious answer and the correct one is recorded rather than assumed.

Measured:

    unattributed vs unattributed  -> unknown
    attributed, same frame        -> same
    attributed, different frames  -> different
    attributed vs unattributed    -> unknown

## 4. What was deliberately NOT tested

An earlier plan of mine was a test asserting that two different equal-sized
cohorts collide under a sentinel. The ruling rejected it, correctly: such a test
documents the defect but also **institutionalises it as intended behaviour**.

The measured fact remains true — equal-size unattributed calls cannot be
distinguished — but it is an epistemic limit, not a behaviour to enshrine. The
tests now assert that the system **refuses to claim comparability**, which is the
property that matters.

## 5. Attribution is inherited, never invented

A restriction of an unattributed population is unattributed. A child cannot
acquire an identity its parent lacked, nor discard one its parent had — the
existing `source_id` inheritance check already covers `None` on both sides, so no
new guard was needed and none was added.

## 6. Verification

### 6.1 Nothing moved

All **41 pre-existing population tests pass unchanged**. The frozen report oracle
— 10 cohorts, 48 fields, **480 values** — shows **zero movements**. The 38
modules touching the evaluation stack produce a BYTE-IDENTICAL `FAILED` list of
40, all sandbox dependency gaps. No test was lost; ten were added.

### 6.2 Sabotage matrix

Eight breaks, **eight detected, zero undetected**.

| break | detected |
|---|---|
| B1 an unattributed population gets a sentinel fingerprint | yes |
| B2 comparison returns SAME for two unattributed populations | yes |
| B3 comparison collapses UNKNOWN into DIFFERENT | yes |
| B4 same_membership_as claims True for unattributed pairs | yes |
| B5 a blank source_id is accepted as attribution | yes |
| B6 is_attributed reports True regardless | yes |
| B7 the child may invent an identity its parent lacked | yes |
| B8 comparison accepts a non-population | yes |

## 7. What this commit does NOT do

`evaluate()` is untouched. It does not yet accept `source_id`, does not build a
population, and does not populate `metric_results`. That is commit 3b, whose
acceptance criterion is the four declared field-cohort movements.

The certification consequence — `certification_eligible = False` when the
population is unattributed — also belongs to 3b. The scope condition in the
ruling is satisfied: the frozen oracle captures the 48 flat report fields and NOT
`metric_results`, so a certification change on typed results cannot violate it.
That was verified, not assumed.

## 8. Files

    src/genomic_variant_classifier/evaluation/population.py   optional attribution
    tests/unit/test_evaluation_population.py                  41 -> 51 tests

Ratchet 3445 -> 3455 (+10), measured by `pytest --collect-only`.

## 9. Delivery-convention findings from the previous commit

Commit 3a's first installer failed to parse: an apostrophe inside a
single-quoted PowerShell literal terminates the string. It failed safe, but the
generator self-check had never looked for unbalanced quote literals. Worse, the
first repair silently did nothing — `read_text` normalises line endings, so a
`split` on carriage-return-newline returned the whole file as one element and the
pattern never matched, and writing back with an empty newline argument then
stripped every carriage return. One root cause, two symptoms, and the same broken
file would have shipped a second time had the new check not just been added.

Three conventions adopted:

- installers are checked for **unbalanced single-quoted literals**;
- prose destined for a PowerShell literal is **escaped at generation**;
- every generated file is **re-read from disk and re-checked after any repair**,
  because an edit that silently matched nothing looks exactly like one that
  succeeded.

---

*Written 2026-07-28.*
