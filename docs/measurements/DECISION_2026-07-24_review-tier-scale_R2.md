# Decision 2026-07-24 (second) -- the two open review-tier questions, in full

**Status:** complete context. Both questions are answerable today and neither is deferred
here. Recommendations are stated with their evidence and their counter-arguments.

**Repository state:** `main` at `e3a4795ad0e1806c41ae01eecf0af3bbdaec562d`,
2026-07-24T05:22:26-04:00.

**Supersedes the deferral in** `docs/measurements/DECISION_2026-07-24_review-tier-scale.md`
section 9, which listed these as open. That document is not modified.

**Acronyms on first use.** ClinVar = the National Center for Biotechnology Information's
Clinical Variation archive. AUPRC = area under the precision-recall curve.

---

## 1. Facts common to both decisions

**`review_tier` is a filter axis and nothing else.** It is created at
`real_data_prep.py:526`, used once at line 536 for `df[df["review_tier"] <=
min_review_tier]`, and **dropped at line 544**. It never reaches a model, never becomes a
feature, and is never persisted. The numeric value carries no meaning beyond its position
relative to the threshold.

**The whole-cohort tier distribution**, measured 2026-07-24 and committed at
`docs/measurements/REVIEWSTATUS_GAPS_2026-07-24.txt`:

| Tier | Production filter (substring) | `clean_cohort` (exact) | Nested field (substring) |
| ---: | ---: | ---: | ---: |
| 1 | 17,220 | 17,220 | 21,384 |
| 2 | 621,707 | 621,707 | 648,425 |
| 3 | 3,077,747 | 3,077,747 | 3,209,058 |
| **4** | **94,285** | **157,229** | **160,997** |
| 5 | 588,130 | 94,285 | 352,421 |
| 6 | 0 | 430,901 | 6,804 |

**Correction, 2026-07-24.** The "Production filter (substring)" column above was
previously filled with the `clean_cohort` exact-map values for tiers 4-6
(showing 157,229 / 519,434 / 5,752). That was wrong: the production map at
`real_data_prep.py:132` is a substring map, and `criteria provided, conflicting
classifications` -- 157,229 rows -- matches no production key, so it falls to the
tier-5 default, not tier 4. Re-derived by simulating the real production map on
the committed cohort value counts, the production column is tier 4 = 94,285
(`no assertion criteria provided`, its only member), tier 5 = 588,130, and no
tier 6 (the production map has no key mapping to 6; unmatched defaults to 5). The
`clean_cohort` and nested columns were correct and are unchanged. Logged as D14.

**Retention at each threshold**, same source:

| `min_review_tier` | Keep, current column | Keep, nested field | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 17,220 | 21,384 | +4,164 |
| 2 | 638,927 | 669,809 | +30,882 |
| **3** | **3,716,674** | **3,878,867** | **+162,193** |
| 4 | 3,873,903 | 4,039,864 | +165,961 |
| 5 | 4,393,337 | 4,392,285 | **-1,052** |

**Thresholds in use:** 3 (`real_data_prep.py:235` default, `launch_run15_baseline.sh:135`,
`launch_run17_r12only.sh:278`), 2 (`continual_trainer.py:78`), and 5 as the sentinel
meaning filtering is off. **4 is used nowhere in the repository.**

---

## 2. Decision one -- `no assertion criteria provided`: tier 4 or tier 5?

### 2.1 What it means

Verified against ClinVar documentation and a live record on 2026-07-24: **zero stars**. A
live ClinVar page reads *"Review status: (0/4) 0 stars out of maximum of 4 stars -- no
assertion criteria provided."* A submitter made a classification and supplied no assertion
criteria to support it.

It is distinct from `no classification provided`, where **no classification exists at
all**. That distinction is the crux.

### 2.2 What the three maps say

| Map | Tier | Lookup |
| --- | ---: | --- |
| `real_data_prep.py:132-140` -- the production filter | **4** | substring, unmatched to 5 |
| `augment_reviewstatus.py:21-27` | **4** | substring, unmatched to 5 |
| `clean_cohort.py:126-136` | **5** | exact after normalisation, unmatched to 6 |

Two to one for 4, but the dissenter is the most complete and most recently corrected map.

### 2.3 How many rows, and where the choice bites

**94,285 rows carry tier 4 under the production map**, of which
`no assertion criteria provided` is the only member -- the map assigns nothing else to 4.
That is **2.143 percent of the 4,399,089-row cohort**. (An earlier version of this
subsection gave 157,229 rows and 3.57 percent; that count is the whole-cohort total
for `criteria provided, conflicting classifications`, which the production substring
map places in tier 5, not tier 4. Corrected 2026-07-24, logged as D14.)

The choice changes the outcome **only at `min_review_tier = 4`**, which no script, launcher
or configuration uses. At 3 they are dropped under both maps; at 5 they are kept under
both.

**At threshold 4, the difference is the 94,285 tier-4 rows** -- not small if that
threshold is ever used, exactly zero today, since no configuration sets it.

### 2.4 The argument for tier 4

A classification without criteria is still a classification. Someone examined the variant
and reached a conclusion; they did not document the framework. The information content is
low but non-zero, and it is categorically different from a record where nobody classified
anything.

Tier 4 preserves that ordering. It says: better than nothing, worse than criteria-backed.
And `real_data_prep.py:235` carries the intent in its own comment --
`# exclude tier 4-5 (no criteria)` -- which groups 4 and 5 as "no criteria" while keeping
them ordered.

### 2.5 The argument for tier 5

`clean_cohort` runs at cohort-construction time and its job is conservatism. A zero-star
record is not evidence a classifier should learn from; grouping it with the other
zero-information records is honest about what it is worth. The finer gradation, if wanted,
belongs in the training filter rather than the cleaner.

There is also a consistency argument: `clean_cohort` reserves 6 for "no classification"
and 5 for "no criteria", giving a three-level bottom (4 conflicting, 5 no criteria, 6 no
classification) where the production map has a two-level bottom. `clean_cohort`'s scale is
internally more expressive.

### 2.6 Recommendation: **tier 4**

Three reasons, in order of weight.

**It is star-faithful.** ClinVar's own precedence ordering for aggregating submissions is
practice guideline, expert panel, criteria provided single submitter, **no assertion
criteria provided**. ClinVar ranks it last among things that are still assertions, and
does not rank it against "no classification" at all -- because the latter is not an
assertion. A scale mirroring ClinVar's should put it at the bottom of the assertion range,
which is 4, not merged into the non-assertion range.

**`no classification provided` is removed earlier anyway.** A row with no classification
has no `clinical_sig` in `PATHOGENIC_TERMS` or `BENIGN_TERMS`, so it is dropped at the
label filter, `real_data_prep.py:516`, **twenty lines before the tier filter runs**. Tiers
5 and 6 are therefore largely theatre for the training path: their occupants are already
gone. Tier 4 is the last tier whose occupants actually reach the filter, which makes it
the meaningful bottom of the scale.

**It changes nothing today and is cheaper to adopt.** Two of three maps already say 4;
choosing 4 changes one map, choosing 5 changes two.

**What would change my recommendation:** evidence that variants with no assertion criteria
are systematically mislabelled -- that their `clinical_sig` disagrees with better-evidenced
records for the same variant at a materially higher rate. That is measurable and has not
been measured. If it were true, they would be actively harmful rather than merely weak,
and merging them into the discard range would be right.

---

## 3. Decision two -- should an unmatched review status raise?

### 3.1 What happens today

`real_data_prep.py:530-531`:

    .map(lambda s: next((v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5))

An unrecognised string silently becomes tier 5. `clean_cohort` silently becomes tier 6.

**Two values in the cohort are currently unmatched by the production map:**

| Value | Rows in the blank-status set | Assigned tier |
| --- | ---: | ---: |
| `criteria provided, conflicting classifications` | 3,768 | 5 |
| `no classification for the single variant` | 115 | 5 |

The first is a **one-star ClinVar status with submitted criteria**. It receives the same
tier as a record with no classification at all, for no reason other than absence from a
dictionary.

### 3.2 What raising would cost

**A crash, at data preparation, before training.** `_load_and_label` and the tier filter
run early -- minutes into a job, not hours. The cost of a raise is a failed job and a
one-line map addition, not a lost eleven-hour run.

**A future ClinVar vocabulary change would break every run until the map is updated.**
That is the genuine cost, and it is real: ClinVar renamed `conflicting interpretations` to
`conflicting classifications` and `no classification for the individual variant` to
`...single variant` within the last two years. Both renames are already in this cohort.
**A raise would have caught both on the day they appeared instead of silently demoting
3,883 rows.**

### 3.3 The precedent already exists in the same function

`real_data_prep.py:545-553` already raises rather than degrading silently:

> `min_review_tier=... requested but the cohort has no 'ReviewStatus' column, so the
> review-tier filter cannot be applied (it would silently keep all review levels).
> Re-build the cohort with ReviewStatus ... or set min_review_tier=5 to disable tier
> filtering explicitly.`

**That is exactly the argument, applied one level up.** The function already refuses to
filter when the *column* is missing, on the grounds that silence would misrepresent the
result. A missing *value* misrepresents the result in precisely the same way, for the rows
it touches.

### 3.4 Recommendation: **yes, raise -- with three conditions**

**Condition one: the error must name the unknown values and their counts.** Not "unmatched
review status" but "3,768 rows carry `criteria provided, conflicting classifications`,
which is absent from REVIEW_STATUS_TIER; add it or set the escape hatch". The fix must be
obvious from the message.

**Condition two: an explicit escape hatch, matching the existing precedent.** Line 551
already offers `set min_review_tier=5 to disable tier filtering explicitly`. The same shape
applies: an `allow_unmatched_review_status` flag, defaulting **False**, that when set
restores the tier-5 default and logs a WARNING naming every unmatched value. Explicit
opt-in to the old behaviour, never silent.

**Condition three: it lands with the map reconciliation, not before.** Raising against
three divergent maps would produce three different failures. One map, one default, one
raise.

**Why this is the change I would keep if only one were possible:** every defect in this
chain has the same shape -- the blank `ReviewStatus` becoming tier 5, the dead
`individual variant` key, the two unmapped statuses, and the 98.834 percent deletion join
failure that started it. In each, **a lookup miss silently became a quality judgement.**
The tier decisions above adjust where a boundary sits; this one removes the mechanism that
lets boundaries be crossed by accident.

**What would change my recommendation:** evidence that ClinVar's vocabulary changes often
enough that fail-closed would block routine work. Two renames in two years does not meet
that bar, and both were silent losses that a raise would have caught.

---

## 4. Summary of what is being asked

| # | Question | Recommendation | Rows affected today | Blocking |
| --- | --- | --- | ---: | --- |
| 1 | `no assertion criteria provided` tier | **4** | **0** (threshold 4 unused) | no |
| 2 | unmatched status raises | **yes**, with three conditions | **0** (behaviour change, not a reclassification) | no |

Neither changes the current cohort. Both should land with the map reconciliation in Phase
1, and both should be recorded in the commit that carries them with the reasoning above,
so the next reader inherits the decision rather than the ambiguity.
