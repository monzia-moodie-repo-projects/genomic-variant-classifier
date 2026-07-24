# Measurement 2026-07-24 (second) -- the three review-tier maps, and a correction to the first record

**Predecessor:** `docs/measurements/MEASUREMENT_2026-07-24_nested-reviewstatus-validation.md`,
committed at `c968976`, 2026-07-24T03:57:50-04:00. That record is **not modified**. Per the
governing principle adopted in `docs/CONTAINMENT_2026-07-24.md`, no scientific artifact is
silently replaced; this document supersedes one claim in it and adds new measurement.

**Repository state:** `main` at `c968976a3cc25deb9e6c32f85b61d9b907024958`.

**Acronyms on first use.** ClinVar = the National Center for Biotechnology Information's
Clinical Variation archive.

---

## 1. The correction

The predecessor states, of the two unmapped review statuses:

> New finding: two review-status values are unmapped and silently dropped. Neither the
> incident nor either probe flags this.

**The second sentence is wrong.** `scripts/clean_cohort.py` flags one of them explicitly,
in its own module header, item 5:

> **5. DEAD TIER KEY REMOVED.** `"no classification for the individual variant"` never
> matched; ClinVar says `"...for the single variant"`. Both spellings are now present.

And its `REVIEW_STATUS_TIER` maps both statuses the predecessor called unmapped:

    "criteria provided, conflicting classifications": 4,
    "criteria provided, conflicting interpretations": 4,
    "no classification for the single variant": 6,      # the spelling ClinVar uses
    "no classification for the individual variant": 6,  # retained: older releases

Dated by commit: the conflicting-classifications key landed **2026-05-31** in `764d38d`;
the single-variant spelling landed **2026-07-08** in `2ffcb4c`, the same commit that
made `clean_cohort` nested-aware in direct response to the incident.

**I did not check all three implementations before writing "neither flags this."** I
checked the incident and the two probes and stopped. The correct statement is narrower and
sharper, and it is in section 3.

---

## 2. What the finding actually is

The fix exists in this repository and has not reached the module that applies the filter.

`src/genomic_variant_classifier/data/real_data_prep.py:529-532` is where
`min_review_tier` is enforced, and it uses **its own** `REVIEW_STATUS_TIER`, not
`clean_cohort`'s:

    .map(lambda s: next((v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5))

`real_data_prep.py` was last touched **2026-07-11** (`aa99ac6`), three days after
`clean_cohort` was corrected, and did not receive the corrected keys.

**This is the same shape as the AlphaFold defect recorded in
`docs/INCIDENT_2026-07-23_protein_pipeline_alphafold_fetch.md`:** a correct implementation
in one module, an uncorrected copy in the production path, and a remediation that stopped
at the module boundary instead of ending with a repository-wide search for the pattern.
Twenty-one days there. Sixteen days here.

---

## 3. The three maps, tabulated

| Review status | `clean_cohort` | `real_data_prep` | `augment_reviewstatus` | Agree |
| --- | ---: | ---: | ---: | --- |
| practice guideline | 1 | 1 | 1 | yes |
| reviewed by expert panel | 1 | 1 | 1 | yes |
| criteria provided, multiple submitters, no conflicts | 2 | 2 | 2 | yes |
| criteria provided, single submitter | 3 | 3 | 3 | yes |
| criteria provided, conflicting classifications | **4** | absent | absent | **NO** |
| criteria provided, conflicting interpretations | **4** | absent | absent | **NO** |
| no assertion criteria provided | **5** | **4** | **4** | **NO** |
| no classification provided | **6** | **5** | **5** | **NO** |
| no classification for the individual variant | **6** | **5** | **5** | **NO** |
| no classification for the single variant | **6** | absent | absent | **NO** |

**Six of ten keys disagree.** The divergence is in the VALUES, not merely in the default,
which is what the predecessor implied by describing it as a difference of semantics.

Three further differences:

| | `clean_cohort` | `real_data_prep` | `augment_reviewstatus` |
| --- | --- | --- | --- |
| unmatched default | **6** (`TIER_UNMATCHED`, documented) | **5** | **5** |
| lookup | **exact**, after underscore normalisation | **substring** | **substring**, on a lowercased raw string |
| underscore handling | normalises before matching | none | none |

The substring lookup is `next((v for k, v in MAP.items() if k in s), default)`, which
returns the **first key in insertion order** that appears in the string. That makes the
result depend on dictionary ordering. No current status matches two keys, so it is inert
today; it is fragile by construction.

---

## 4. What the divergence costs, in rows

Using the counts measured 2026-07-24 from
`docs/measurements/REVIEWSTATUS_GAPS_2026-07-24.txt`:

| Status | Rows (blank-ReviewStatus set) | `clean_cohort` | `real_data_prep` | Divergent |
| --- | ---: | ---: | ---: | --- |
| `criteria provided, conflicting classifications` | 3,768 | 4 | 5 (unmatched) | **yes** |
| `no classification for the single variant` | 115 | 6 | 5 (unmatched) | **yes** |
| `no assertion criteria provided` | 12,228 | 5 | **4** | **yes** |

**At `min_review_tier <= 3`, which the launch scripts use, all three are dropped under
both maps**, so today's cohort is unaffected by the divergence. The exposure is
conditional and appears the moment anything runs at a different threshold:

- **At `min_review_tier <= 4`**, `real_data_prep` **keeps** all 12,228
  `no assertion criteria provided` rows and `clean_cohort` **drops** them, and
  `real_data_prep` drops the 3,768 conflicting rows while `clean_cohort` keeps them.
- `scripts/run_phase2_eval.py:1197` prints, as operator guidance,
  *"Try --min-review-tier 2 for expert-reviewed labels only"*, so non-default thresholds
  are an anticipated operation, not a hypothetical.

**The cohort you get depends on which module answered the question**, and nothing in the
code makes that choice visible.

---

## 5. What is required before regeneration

The predecessor listed two prerequisites. Both stand; one is now specified precisely.

1. **Reconcile the three maps into one function with one documented default.** This is
   `probe_reviewstatus_gaps`'s own unconditional recommendation. `clean_cohort`'s map is
   the most complete -- it is the only one carrying the conflicting-classification keys
   and both spellings of the no-classification key -- and its `TIER_UNMATCHED = 6` is the
   only default that is named and documented. It is the natural single source, but
   promoting it changes `no assertion criteria provided` from 4 to 5 in the production
   filter, which is a scope decision and not a refactor.
2. **Decide the tier for conflicting classifications deliberately.** `clean_cohort`
   assigns 4. ClinVar treats `criteria provided, conflicting classifications` as a
   one-star status, the same star count as `criteria provided, single submitter`, which
   both maps put at 3. Whether conflict warrants a demotion to 4, or should be handled by
   `real_data_prep`'s separate `exclude_conflicting` path rather than by tier at all, is a
   scientific judgement. It should be made and recorded, not inherited.

**Neither is a refactor.** Both change which rows enter training at some thresholds, so
both belong in Phase 1 with a stated before-and-after row count, not in a cleanup pass.

---

## 6. What still stands from the predecessor

Everything else. The eighteen-figure reproduction, the corrected reference-class analysis
showing deletions resemble insertions at 2.14 percentage points against 4.60 for
single-nucleotide variants, the independent 178,563 cross-check, the missing-token and
tier-semantics findings, and the process correction recording the withdrawn tool.

Phase 1 entry criterion 4 remains closed. Criterion 3, the artifact lineage sweep, remains
open.
