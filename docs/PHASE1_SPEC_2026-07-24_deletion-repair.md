# Phase 1 specification 2026-07-24 -- the deletion repair and the map reconciliation

**Status:** specification. **No code has been changed.** This document states exactly what
would change, where, what it would cost in rows, what must be measured before and after,
what tests must accompany it, and what is deliberately out of scope. It exists so that the
repair can be approved or rejected on its detail rather than on its summary.

**Repository state:** `main` at `e3a4795ad0e1806c41ae01eecf0af3bbdaec562d`,
2026-07-24T05:22:26-04:00.

**Authorised by:** both Phase 1 entry criteria are met -- criterion 4 at
2026-07-24T07:32:59Z, criterion 3 at 08:05:21Z -- and both open decisions have
recommendations with evidence in
`docs/measurements/DECISION_2026-07-24_review-tier-scale_R2.md`.

**Acronyms on first use.** VCF = variant call format. ClinVar = the National Center for
Biotechnology Information's Clinical Variation archive. AUROC = area under the receiver
operating characteristic curve. AUPRC = area under the precision-recall curve.

---

## 1. What is being repaired, and what is not

**In scope, and only this:**

1. The review-status source: `scripts/augment_reviewstatus.py:64` stops deriving
   `ReviewStatus` from a VCF join that fails on 98.834 percent of deletions, and takes it
   from `metadata.review_status`, which is already in the source parquet.
2. The three divergent tier maps become one.
3. An unmatched review status raises instead of silently becoming tier 5.
4. The cohort is regenerated, with a stated before-and-after row count.

**Explicitly out of scope, and why:**

- **The AlphaFold structural resolver.** It has the wider blast radius -- 225 artifacts
  against 79, 98 days against 45 -- but it needs the UniProt index and the AlphaFold
  service, not the cohort, so it can proceed in parallel as a separate change. Bundling
  two repairs into one commit makes a bisect meaningless.
- **Retraining and revalidation.** That is Phase 2. This specification produces a repaired
  cohort; it does not evaluate what the repair does to any metric.
- **Provenance manifests, the source registry, typed evidence states.** Phase 3.
- **Regenerating the four affected smoke cohorts.** Required before the all-models smoke
  law can gate a post-repair launch, but it depends on this repair completing first, so it
  is the immediate follow-on rather than part of this change.

---

## 2. Change 1 -- the review-status source

**File:** `scripts/augment_reviewstatus.py`

**Current, at line 64:**

    key = (df["chrom"].map(_norm_chrom) + ":" + df["pos"].astype("int64").astype(str)
           + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str))
    df["ReviewStatus"] = key.map(vmap).fillna("")  # unmatched -> "" -> tier 5

**The defect:** the key mis-normalises indels, so the VCF lookup misses 98.834 percent of
deletions, and `.fillna("")` converts every miss into the worst tier. A join failure is
indistinguishable from a quality judgement.

**Replacement behaviour:**

- Read `metadata.review_status` from the source parquet. It is already present, agrees
  with the VCF-derived column on **3,974,573 of 3,974,573** rows where both are populated
  with **zero disagreements**, and is a strict superset -- `rows metadata-missing but
  ReviewStatus present: 0`.
- Normalise `-` and the other tokens in `MISSING_TOKENS` to the empty string, so "missing"
  has one representation.
- **Retain the VCF join as a cross-check, not as the source.** Where both are populated,
  disagreement must be zero; any non-zero count aborts the script. That converts today's
  measured agreement into a standing assertion instead of a one-off observation.
- Keep the three existing post-conditions at lines 68-71 -- row count, null count and
  duplicate count unchanged -- and add a fourth: the count of populated `ReviewStatus`
  must **increase**, and by how much must be printed.

**Predicted effect, from `docs/measurements/REVIEWSTATUS_GAPS_2026-07-24.txt`:**

| Quantity | Before | After |
| --- | ---: | ---: |
| deletions with a populated review status | 2,210 | 180,773 |
| deletions retained at `min_review_tier <= 3` | 1,938 | 163,391 |
| deletion share of the surviving cohort | 0.0521 % | 4.2123 % |
| binary trainable rows | 1,490,324 | 1,620,592 |
| positive rate | 14.145 % | 18.546 % |

**The positive-rate shift is the number that matters.** 14.145 to 18.546 percent means
**no AUPRC computed before this repair is comparable with one computed after.** That must
be stated in `docs/METRICS.md` beside every affected row, not left for a reader to infer.

---

## 3. Change 2 -- one tier map

**Files:** `src/genomic_variant_classifier/data/real_data_prep.py:132-140`,
`scripts/clean_cohort.py:126-136`, `scripts/augment_reviewstatus.py:21-27`.

Three maps become one, defined once and imported. The single map, with both decisions from
`DECISION_2026-07-24_review-tier-scale_R2.md` applied:

| Review status | ClinVar stars | Tier | Source of the value |
| --- | ---: | ---: | --- |
| practice guideline | 4 | 1 | all three agree |
| reviewed by expert panel | 3 | 1 | all three agree |
| criteria provided, multiple submitters, no conflicts | 2 | 2 | all three agree |
| criteria provided, single submitter | 1 | 3 | all three agree |
| criteria provided, conflicting classifications | 1 | **3** | star-faithful; `clean_cohort` says 4 |
| criteria provided, conflicting interpretations | 1 | **3** | the pre-2024 spelling, same status |
| no assertion criteria provided | 0 | **4** | decision one; `clean_cohort` says 5 |
| no classification provided | -- | 5 | `clean_cohort` says 6 |
| no classification for the single variant | -- | 5 | the spelling ClinVar uses |
| no classification for the individual variant | -- | 5 | retained for older releases |
| **unmatched** | -- | **raise** | decision two |

**Lookup semantics: exact, after normalisation.** `clean_cohort`'s `_norm_term` --
lowercase, underscores to spaces, collapse whitespace -- becomes the single normaliser.
The substring lookup is retired: it returns the first key in dictionary insertion order
that appears in the string, which makes the result depend on dictionary ordering. Inert
today, fragile by construction, and unnecessary once the map is complete.

**Resolution-level effect of the map reconciliation, holding the legacy
`ReviewStatus` source constant.** The unified map changes either the numeric tier or the
resolution provenance of three observed statuses. `no assertion criteria provided` is
absent from these tables because it is an explicit tier-4 status under both the production
map and the unified map -- neither its tier nor its resolution path changes. (An earlier
version of this section listed it here at 157,229 rows, which is both the wrong status --
that is `criteria provided, conflicting classifications`' total -- and the wrong basis.
Corrected 2026-07-24 per Option C; logged as D14.)

**Table 3A -- whole-cohort resolution changes, legacy source held constant.** Counts are
whole-cohort totals from the legacy `ReviewStatus` column of `clinvar_grch38_clean.parquet`.

| Status | Rows (whole cohort) | Old resolution | Old tier | New resolution | New tier | Numeric change |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `criteria provided, conflicting classifications` | 157,229 | unmatched fallback | 5 | explicit key | 3 | -2 |
| `no classification for the single variant` | 512 | unmatched fallback | 5 | explicit key | 5 | 0 |
| `no classifications from unflagged records` | 121 | unmatched fallback | 5 | explicit key | 5 | 0 |

Only `criteria provided, conflicting classifications` changes numeric tier. The other two
remain tier 5 but cease to be unmatched vocabulary -- the fallback-to-explicit provenance
change the `TierResolutionPath` enum in `review_status.py` records.

**Table 3B -- source-repair interaction among legacy-gap rows.** These counts are the
subset whose legacy `ReviewStatus` was blank and whose `metadata.review_status` supplies a
previously unmatched recognised status. Each equals the metadata total minus the overlap
for its status (the transition table's new-only column,
`docs/measurements/COHORT_DELTA_FORENSICS_2026-07-24.txt`); the unflagged-records count of
12 was independently re-measured on the cohort on 2026-07-24 and matches.

| Nested status among legacy-gap rows | Rows in gap subset | Legacy outcome | Repaired + old map | Repaired + unified map |
| --- | ---: | --- | --- | --- |
| `criteria provided, conflicting classifications` | 3,768 | missing-tier path | unmatched fallback tier 5 | explicit tier 3 |
| `no classification for the single variant` | 115 | missing-tier path | unmatched fallback tier 5 | explicit tier 5 |
| `no classifications from unflagged records` | 12 | missing-tier path | unmatched fallback tier 5 | explicit tier 5 |

**The conflicting rows are not actually retained.** A variant whose review status is
`criteria provided, conflicting classifications` carries a `clinical_sig` of *"Conflicting
classifications of pathogenicity"*, which is in neither `PATHOGENIC_TERMS` nor
`BENIGN_TERMS`, so it is dropped at the **label filter, `real_data_prep.py:516`, twenty
lines before the tier filter runs**. `exclude_conflicting` defaults to `True` and fires at
line 557 as a second, redundant guard on a population already gone.

**So the map change moves zero rows into training.** That must be verified, not assumed --
see section 6, measurement 3.

---

## 4. Change 3 -- unmatched raises

**File:** `real_data_prep.py:526-534`, and the same at the other two call sites once the
map is shared.

**Current:**

    .map(lambda s: next((v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5))

**Replacement behaviour**, with the three conditions from decision two:

- Exact lookup after normalisation. On a miss, collect the unrecognised value.
- After the pass, if any unrecognised values exist, **raise**, naming each value and its
  row count, so the fix is a one-line map addition and is obvious from the message.
- A `DataPrepConfig.allow_unmatched_review_status` flag, defaulting **`False`**, restores
  the old tier-5 default and logs a `WARNING` naming every unmatched value and count.
  Explicit opt-in, never silent.

**This matches an existing precedent in the same function.** `real_data_prep.py:545-553`
already raises when the `ReviewStatus` *column* is absent, on the grounds that filtering
silently *"would silently keep all review levels"*. A missing *value* misrepresents the
result identically, for the rows it touches, and line 551 already offers an explicit escape
hatch in exactly this shape.

**On the first run after this change, the raise will fire** on the three unmatched values
in Table 3A -- `criteria provided, conflicting classifications`, `no classification for the
single variant`, and `no classifications from unflagged records` -- unless they are added to
the map in the same commit. They are, so it will not. That is the point of landing the map
and the raise together.

---

## 5. Tests that must accompany the change

None of these exist today.

1. **Every value in the cohort maps.** Parametrised over the distinct review-status values
   the cohort actually contains, asserting each yields a tier and none raises. This is the
   test that would have caught the dead `individual variant` key in May.
2. **An unrecognised value raises**, and the message names the value.
3. **`allow_unmatched_review_status=True` does not raise**, returns tier 5, and logs a
   warning naming the value.
4. **The map is the only map.** As built, a CONTENT-based detector
   (`tests/unit/_review_map_detector.py`) that flags any dictionary literal whose keys are
   ClinVar review-status vocabulary and whose values are integers, under any binding name.
   This is stronger than the name-scoped guard first envisaged: it found a fourth tier map,
   `clinvar_tracker.py:160::REVIEW_TIER`, that a search for the name `REVIEW_STATUS_TIER`
   could not see. The inventory is frozen on `path::name` and fails in both directions, so
   it shrinks 8 -> 4 -> 1 as consumers are rewired. This pins the *class* of defect.
5. **Normalisation is applied before lookup** -- `Criteria_Provided,__Single_Submitter`
   resolves to tier 3.
6. **The deletion retention floor.** An assertion that after augmentation, deletions with
   a populated review status exceed 150,000, so a regression to the join-based source
   turns the suite red rather than quietly re-censoring the cohort.

These tests landed on 2026-07-24 in commit `45525fb`, and the suite-size ratchet moved
2893 to **2950** -- not the 2899 forecast here, because the rebuild also added a
content-based tier-map detector (`tests/unit/_review_map_detector.py`) with a nine-case
sabotage battery and direct tests for `resolve`, `tier_of` and `tiers_for`. The ratchet
and README badge moved together, since `tests/unit/test_readme_claims.py:221` enforces
equality with no tolerance.

---

## 6. Measurements that must bracket the change

Each is a number recorded before and after, in the commit message.

1. **Cohort reconciliation.** Row count, null-allele count and duplicate-identifier count
   before and after augmentation. The existing assertions at
   `augment_reviewstatus.py:68-71` already check the first three; the printed values go in
   the record.
2. **Retention by variant class at thresholds 1 through 5**, from
   `scripts/probe_tier_filter_impact.py`. The before figures are already committed at
   `docs/measurements/TIER_FILTER_IMPACT_2026-07-24.txt`.
3. **The map-change-only effect.** Run the tier filter with the new map against the *old*
   review-status source and confirm the surviving trainable row IDENTITIES are unchanged --
   `old_survivor_ids.equals(new_survivor_ids)`, not merely `old_count == new_count`, because
   a count can hold while different rows enter and leave. This isolates the map change from
   the source change. If the identity sets differ, section 3's claim that the conflicting
   rows never reach the filter is wrong and the repair stops.
4. **Cross-source disagreement.** The count of rows where the VCF-derived and nested
   values are both populated and differ. Expected zero; any other value aborts.
5. **The label-column agreement question**, still open: whether `clinical_sig` and
   `pathogenicity` agree in content, not merely in format.
   `probe_label_column_terms_2026-07-24.py` answers it. Not blocking, but the repaired
   cohort should not be published without it, because every figure in section 2 was
   measured on `pathogenicity` while production labels from `clinical_sig`.

---

## 7. Order of operations, and the rollback

1. Land the shared map, the raise, and the six tests. **No cohort is regenerated.** The
   suite must be green and the ratchet bumped. Measurement 3 runs here.
2. Land the `augment_reviewstatus.py` source change. Still no regeneration. Measurement 4
   runs here.
3. Regenerate the cohort into a **new file**, not over the existing one. Measurements 1
   and 2 run here.
4. Certify the new cohort against the clean-cohort guard and the sweep before anything
   consumes it.
5. Only then switch the launch scripts.

**Rollback:** steps 1 and 2 are ordinary commits and revert cleanly. Step 3 writes a new
file and touches nothing existing, so rollback is deleting the new file. **The existing
cohort is never overwritten**, which is also why step 3 must not reuse the current
filename.

**The current cohorts remain AFFECTED and are not deleted.** They are the provenance
record for Runs 15 and 16 and for every artifact the lineage sweep classified. Per the
governing principle, they are superseded, not replaced.

---

## 8. What this specification does not settle

- Whether the repaired cohort changes AUROC, AUPRC, calibration or gene ranking
  materially. Unknown, and unknowable until Phase 2.
- Whether `clinical_sig` and `pathogenicity` agree in content.
- The disposition of the 251 lineage-less model artifacts.
- The AlphaFold structural repair, which proceeds separately and in parallel.

---

## 9. What approving this means

That the repair may be built as specified: three code changes, six tests, four
measurements, five ordered steps, one new cohort file, nothing overwritten.

**It does not authorise regeneration.** Step 3 is a separate go, taken after steps 1 and 2
are green and measurement 3 has confirmed that the map change alone moves zero rows.
