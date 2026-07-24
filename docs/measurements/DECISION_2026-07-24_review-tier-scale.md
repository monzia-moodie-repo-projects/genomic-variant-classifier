# Decision memo 2026-07-24 -- what tier should `no assertion criteria provided` receive?

**Status:** analysis and recommendation. **The decision is Monzia's.** Nothing is
implemented and no code is changed by this document.

**Repository state:** `main` at `c968976a3cc25deb9e6c32f85b61d9b907024958`.

**Predecessors:** `docs/measurements/MEASUREMENT_2026-07-24_nested-reviewstatus-validation.md`
and `docs/measurements/MEASUREMENT_2026-07-24_review-tier-map-divergence.md`. Neither is
modified.

**Acronyms on first use.** ClinVar = the National Center for Biotechnology Information's
Clinical Variation archive. ACMG = American College of Medical Genetics and Genomics.
SCV = a single submitted record in ClinVar. VCV = ClinVar's aggregated variant record.

---

## 1. The first thing to say: this divergence has no current exposure

Before any argument about which tier is correct, the size of the question.

`no assertion criteria provided` is tier **4** in `real_data_prep.py` and tier **5** in
`clean_cohort.py`. The filter is `review_tier <= min_review_tier`. So the two maps
disagree about these rows **only at `min_review_tier = 4`**.

Verified across the entire repository on 2026-07-24: **`min_review_tier = 4` is used
nowhere.** No script, no shell launcher, no configuration, no test. The values actually in
use are:

| Value | Where |
| --- | --- |
| 3 | `real_data_prep.py:235` default; `launch_run15_baseline.sh:135`; `launch_run17_r12only.sh:278` |
| 2 | `continual_trainer.py:78` |
| 5 | sentinel meaning "filtering disabled", per the guard at `real_data_prep.py:545` |

At 2 and at 3, `no assertion criteria provided` is dropped under both maps. At 5 it is
kept under both. **Today the divergence changes nothing.** It is latent, not active, and
I should have established that before describing it as something 12,228 rows "hang on".
They hang on it only conditionally.

That does not make it safe to leave. It makes it a defect to fix deliberately rather than
urgently.

---

## 2. What ClinVar actually means by these terms

Verified 2026-07-24 against ClinVar documentation and live records rather than asserted
from memory.

| Review status | ClinVar stars |
| --- | ---: |
| practice guideline | 4 |
| reviewed by expert panel | 3 |
| criteria provided, multiple submitters, no conflicts | 2 |
| criteria provided, single submitter | **1** |
| criteria provided, conflicting classifications | **1** |
| no assertion criteria provided | **0** |
| no classification provided | not on the star scale -- no interpretation exists |

A live ClinVar record shows the zero explicitly: *"Review status: (0/4) 0 stars out of
maximum of 4 stars -- no assertion criteria provided."* ClinVar's own precedence order for
aggregating submitted records is: practice guideline, expert panel, criteria provided
single submitter, no assertion criteria provided.

**Two consequences for this project's maps.**

**`criteria provided, conflicting classifications` is a ONE-star status**, the same star
count as `criteria provided, single submitter`, which both maps place at tier 3.
`clean_cohort` places conflicting at tier 4 -- a deliberate demotion below its star
rating. That may well be right, but it is a judgement and it is not recorded as one.

**`no assertion criteria provided` is zero stars**, and so is `no classification
provided` in the sense that neither carries any assessed evidence. ClinVar does not rank
them against each other.

---

## 3. The real problem: one number is carrying two different meanings

The project's tier scale is trying to express two independent things at once.

**Meaning A -- how strong is the evidence behind the classification?** That is exactly
what ClinVar's star rating measures, and it maps cleanly onto tiers 1 through 4.

**Meaning B -- does a usable training label exist at all?** That is a different question
with a different answer, and it is already handled elsewhere.
`real_data_prep._load_and_label` at lines 511-517 assigns a binary label only for terms in
`PATHOGENIC_TERMS` or `BENIGN_TERMS` and then drops every row where the label is null.
A record with **no classification provided** has nothing to classify, so it is removed by
the label filter **regardless of its tier**.

**That is why the 4-versus-5 question feels unanswerable: it is the wrong question.**
Ordering `no assertion criteria provided` against `no classification provided` asks which
of two zero-star statuses is worse, when one of them is not a strength-of-evidence
statement at all -- it is an absence-of-data statement, and the label filter already
removes it.

`real_data_prep.py:235` carries a comment that says this without quite noticing:

    min_review_tier: int = 3  # exclude tier 4-5 (no criteria)

**Tiers 4 and 5 are described as one bucket, "no criteria".** The distinction between
them was never intended to carry weight. The scale grew a level it does not use.

---

## 4. What I recommend, and what it costs

**A star-faithful scale, with absence handled separately.**

| Review status | Stars | Proposed tier |
| --- | ---: | ---: |
| practice guideline | 4 | 1 |
| reviewed by expert panel | 3 | 1 |
| criteria provided, multiple submitters, no conflicts | 2 | 2 |
| criteria provided, single submitter | 1 | 3 |
| criteria provided, conflicting classifications | 1 | **3** |
| criteria provided, conflicting interpretations | 1 | **3** |
| no assertion criteria provided | 0 | **4** |
| no classification provided | -- | **5** |
| no classification for the single variant | -- | 5 |
| no classification for the individual variant | -- | 5 |
| **anything unmatched** | -- | **raise, do not default** |

Four changes from the status quo, each with its consequence stated.

**Change 1 -- `no assertion criteria provided` stays at 4.** This adopts
`real_data_prep`'s value over `clean_cohort`'s 5, on the grounds that it is the last tier
where a human made an assessment. It is zero-star evidence, but it is evidence. Cost:
`clean_cohort` must change. Rows affected at present thresholds: **none**.

**Change 2 -- conflicting classifications move from 4 to 3.** This is star-faithful:
ClinVar rates conflicting at one star, the same as a single submitter. Conflict is a
*label* problem, and `real_data_prep` already has a separate `exclude_conflicting`
mechanism for it. Encoding conflict a second time in the tier double-counts it. Cost:
3,768 rows in the blank-status set would become tier 3 and therefore **retained at the
default threshold**, where they are currently dropped. **This is the only change in this
memo that alters today's cohort, and it needs a stated before-and-after count before it
lands.**

**Change 3 -- both spellings of the no-classification key, at tier 5.** `clean_cohort`
already carries both; the other two carry one. Cost: none, 115 rows, currently unmatched.

**Change 4 -- an unmatched status RAISES.** This is the change I would argue hardest for.
Every defect in this whole chain -- the blank `ReviewStatus`, the two unmapped statuses,
the dead `individual variant` key -- has the same shape: **a lookup miss silently becoming
a quality judgement.** A default of 5 or 6 is what converts "I do not recognise this
string" into "this variant is poorly reviewed". Those are not the same statement and the
code should not be allowed to conflate them.

The pattern already exists in this repository: `real_data_prep.py:545-553` raises rather
than filtering silently when `ReviewStatus` is absent, and `probe_reviewstatus_gaps`
closes by demanding the same discipline for `clean_cohort`'s schema. Extending it to the
value lookup is consistent, not novel.

---

## 5. What the alternative argument is, because it is not weak

**The case for `clean_cohort`'s 5 rather than 4.**

`clean_cohort` runs at cohort-construction time and its job is conservatism: quarantine
anything doubtful, let downstream decide. Under that reading, a zero-star record is
doubtful and belongs with the other doubtful records, and the finer gradation belongs in
the training filter rather than in the cleaner. Two maps with two purposes is then a
feature, not a defect.

**The reason I do not adopt it:** two maps with two purposes is exactly what produced the
present situation, in which the cohort you get depends on which module answered the
question and nothing makes that visible. If the two really do need different behaviour,
that should be one map with two documented *policies* selected explicitly, not two maps
that happen to differ and whose difference nobody stated until today.

---

## 6. What I am not recommending, and why

**I am not recommending promoting `clean_cohort`'s map wholesale**, which is what my
previous document implied was the natural move. It is the most complete map -- the only
one with the conflicting keys, both no-classification spellings, and a named default --
but adopting it entire would demote `no assertion criteria provided` from 4 to 5 for no
stated reason, and would keep conflicting at 4 against ClinVar's one-star rating. Complete
is not the same as correct.

---

## 7. A separate question this analysis surfaced, unresolved

`real_data_prep._load_and_label` labels from **`clinical_sig`** (line 510), using
`PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}`
matched with `.isin(...)` -- **exact and case-sensitive**, with spaces and initial
capitals.

`scripts/probe_tier_filter_impact.py` warns in its own assumptions block: *"The label
column consumed downstream is `pathogenicity` (what clean_cohort auto-detects). If
real_data_prep uses `clinical_sig`, re-run with --label-col."* It also records that the
data uses **underscores** (`likely_pathogenic`).

**The probe told me to verify which column production uses. I passed
`--label-col pathogenicity` without verifying, and production uses `clinical_sig`.**
`clean_cohort.py:110` lists `pathogenicity`, `clinical_significance`, `clinical_sig` and
`clnsig` as label-column candidates, so the cohort may carry more than one, possibly with
different capitalisation conventions.

Two outcomes are possible and I cannot distinguish them from the repository:

- `clinical_sig` carries ClinVar's original spacing and capitalisation, in which case
  `.isin` matches correctly and the probe's label figures describe a different but
  equivalent column.
- `clinical_sig` carries the underscore form, in which case
  **`.isin({"Likely pathogenic", ...})` matches nothing** and every likely-pathogenic
  variant is dropped as unlabelled.

The second would be a defect larger than anything recorded so far. **It is a five-second
check against the cohort and it must be run before anything else in this memo is acted
on.** The command is in section 8.

---

## 8. The check that must precede any decision

```powershell
$Repo = "C:\Projects\genomic-variant-classifier"
$Dl   = "C:\Users\monzi\Downloads"
Set-Location $Repo
python "$Dl\probe_label_column_terms_2026-07-24.py" --repo "C:\Projects\genomic-variant-classifier"
"EXIT CODE = $LASTEXITCODE"
```

A bash here-document was written here first. **PowerShell 5.1 has no here-document
syntax**, so `python - <<'PY'` would have failed on Monzia's machine, and the project's
own convention forbids `python -c` with multi-line code containing embedded quotes. The
check is therefore delivered as a script.

Read the repr strings. If `clinical_sig` shows `'Likely pathogenic'` the exact match is
safe. If it shows `'likely_pathogenic'` it is not, and that becomes the highest-priority
item in the project.

---

## 9. The decision requested

1. **`no assertion criteria provided`: tier 4 or tier 5?** My recommendation is 4, on
   star-faithfulness, with zero current exposure either way.
2. **`criteria provided, conflicting classifications`: tier 3 or tier 4?** My
   recommendation is 3, with conflict handled by `exclude_conflicting`. This one **does**
   change today's cohort -- 3,768 rows -- and should not land without a before-and-after
   count.
3. **Should an unmatched status raise instead of defaulting?** My recommendation is yes,
   and it is the change I would keep if only one were possible.

None of these is implemented. Section 8 runs first regardless of how they are answered.
