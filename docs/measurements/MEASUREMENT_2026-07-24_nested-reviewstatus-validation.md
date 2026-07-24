# Measurement 2026-07-24 -- Phase 1 entry criterion 4: is the nested review-status remedy valid on deletions?

**Result: yes, and on stronger evidence than the incident or my own design proposed.**
Two new defects were found along the way, neither previously recorded.

**Repository state:** `main` at `7ac4fb5a0e82afc2f961694613828c72ef9ec6b9`,
2026-07-24T03:14:11-04:00.

**Sources.** `scripts/probe_tier_filter_impact.py` and `scripts/probe_reviewstatus_gaps.py`
run 2026-07-24, both named in the incident's own header as its reproduction commands.
`probe_nested_tier_distribution_2026-07-24.py` run at 2026-07-24T07:32:59Z for the one
question the existing probes do not answer.

**Acronyms on first use.** AUPRC = area under the precision-recall curve. SNV = single-
nucleotide variant. MNV = multi-nucleotide variant. VCF = variant call format.

---

## 1. The incident reproduces exactly. All eighteen figures.

Before asking whether the remedy is valid, the prior question is whether the incident's
measurements still hold sixteen days later. Every figure it cites was re-derived today.

| Quantity | Incident, 2026-07-08 | Probe, 2026-07-24 |
| --- | --- | --- |
| deletion blank rate | 98.834 % | 98.834 % |
| SNV blank rate | 5.771 % | 5.771 % |
| insertion blank rate | 0.483 % | 0.483 % |
| MNV/other blank rate | 0.519 % | 0.519 % |
| agreement where both populated | 3,974,573 / 3,974,573 | 3,974,573 / 3,974,573 |
| deletion validation coverage | 1.166 % | 1.166 % |
| deletions kept, current, tier <= 3 | 1,938 | 1,938 |
| deletions kept, metadata, tier <= 3 | 163,391 | 163,391 |
| deletion share, current | 0.0521 % | 0.0521 % |
| deletion share, metadata | 4.2123 % | 4.2123 % |
| pathogenic kept, current | 34.556 % | 34.556 % |
| likely_benign kept, current | 95.236 % | 95.236 % |
| binary rows, current | 1,490,324 | 1,490,324 |
| positive rate, current | 14.145 % | 14.1450 % |
| binary rows, metadata | 1,620,592 | 1,620,592 |
| positive rate, metadata | 18.546 % | 18.5462 % |
| unfiltered binary rows | 1,848,225 | 1,848,225 |
| unfiltered positive rate | 26.725 % | 26.7252 % |

**Eighteen of eighteen.** The incident is current, not stale, and its numbers may be
cited as measured today.

---

## 2. The validation question, and why agreement cannot answer it

The remedy replaces the top-level `ReviewStatus` column with the nested
`metadata.review_status` field. Its headline support is 3,974,573 of 3,974,573 rows
agreeing with zero disagreements. But validation coverage **by class** is:

    SNV        3,865,093 of 4,101,824 = 94.229%
    insertion     90,778 of    91,219 = 99.517%
    MNV/other     16,492 of    16,578 = 99.481%
    deletion       2,210 of   189,468 =  1.166%   <-- the class the remedy exists to rescue

The join that fills the top-level column fails on 98.834 percent of deletions, so for
98.834 percent of them there is nothing to agree with. The perfect agreement is carried
by the classes that did not need rescuing.

---

## 3. The distributional answer, and a correction to my own design

`probe_nested_tier_distribution_2026-07-24.py` measured the review-tier profile the
nested field assigns each class, over populated rows only, under the substring semantics
the production filter uses.

| class | n populated | tier 1 | tier 2 | tier 3 | tier 4 | tier 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SNV | 3,865,871 | 0.39 % | 15.72 % | 77.48 % | 2.26 % | 4.15 % |
| **deletion** | **180,773** | **2.34 %** | **14.83 %** | **73.21 %** | **6.86 %** | **2.75 %** |
| insertion | 90,804 | 2.19 % | 13.27 % | 75.35 % | 6.54 % | 2.64 % |
| MNV/other | 16,493 | 1.16 % | 11.62 % | 79.03 % | 4.69 % | 3.49 % |

**My script compared deletions against single-nucleotide variants. That was the wrong
reference class**, and the right one is in the same table.

| Comparison | Largest tier-share difference | At tier |
| --- | ---: | --- |
| deletion vs SNV | 4.60 pp | 4 |
| **deletion vs insertion** | **2.14 pp** | **3** |
| insertion vs SNV | 4.28 pp | 4 |
| MNV/other vs SNV | 4.10 pp | 2 |

**Insertions are the validated class most similar in kind**: 99.517 percent validation
coverage, 90,778 of 90,778 agreeing, zero disagreements. Deletions resemble insertions
(2.14 percentage points) far more closely than they resemble single-nucleotide variants
(4.60). Both indel classes carry roughly 2.2 to 2.3 percent tier 1 against 0.39 percent
for SNVs, and roughly 6.5 to 6.9 percent tier 4 against 2.26 percent.

**The indel review profile is a real, consistent shape, and the validated half of it
agrees with the unvalidated half.** That is a materially stronger argument for the remedy
than comparison against single-nucleotide variants produces, and stronger than the
incident itself offers.

**It is still not proof.** No threshold was declared in advance and none is applied here.
The claim is: the nested field treats deletions the way it treats the indel class it is
validated on, so adopting it for deletions is consistent with observed behaviour rather
than an unexamined leap.

### An independent cross-check of the rescue count

My add-on measured 180,773 deletions with the nested field populated. The probe measures
2,210 deletions with both fields populated. 180,773 minus 2,210 is **178,563** -- exactly
the incident's stated rescue count, derived from two independent tools.

---

## 4. New finding: two review-status values are unmapped and silently dropped

`REVIEW_STATUS_TIER` in `real_data_prep.py:132-140` has seven keys. The cohort contains
values that match none of them, and neither the incident nor either probe flags this.

| Value present in the data | Rows | Substring tier | Exact tier | Mapped |
| --- | ---: | ---: | ---: | --- |
| `criteria provided, conflicting classifications` | **3,768** | 5 | 6 | **NO** |
| `no classification for the single variant` | **115** | 5 | 6 | **NO** |

**3,883 rows across two values**, all dropped at `min_review_tier <= 3`.

`criteria provided, conflicting classifications` is a legitimate ClinVar review status
**with submitted criteria**. It receives the worst tier and is dropped, which is the same
outcome as a blank for an entirely different and unrelated reason. Whether it should be
tier 2, 3 or excluded on other grounds is a scientific judgement; being dropped by
falling off the end of a lookup table is not a judgement at all.

`no classification for the single variant` differs from the mapped
`no classification for the individual variant` by **one word**. That is a near-miss in a
substring table, and the substring semantics do not save it because neither string
contains the other.

These counts are small beside 161,453 wrongly excluded deletions, but they are the same
defect class: a lookup miss silently becoming a quality judgement.

---

## 5. Findings restated from the probes that are not in the incident

**A third missing-value token.** `metadata.review_status` uses `-` as its missing marker,
present on 245,148 rows, all of which also have a blank top-level column. Broken down:
SNV 235,953, deletion 8,695, insertion 415, MNV/other 85. The metadata field is a strict
superset of the top-level column -- `rows metadata-missing but ReviewStatus present: 0` --
so no information is lost by switching, only gained.

**Tier semantics disagree on 425,149 rows.** Substring puts 519,434 rows at tier 5 and
5,752 at tier 6; exact-map puts 94,285 at tier 5 and 430,901 at tier 6. The difference is
425,149 in both directions. Three implementations of review-tier mapping exist --
`clean_cohort`, `augment_reviewstatus`, `real_data_prep` -- and `probe_reviewstatus_gaps`
recommends reconciling them into one function with one documented default. That has not
been done.

**An inversion at tier 5.** At `min_review_tier <= 5`, the metadata source keeps 1,052
FEWER rows than the current column, the only tier where the delta is negative. This
follows from metadata carrying 6,804 rows at tier 6 against 5,752, and is a consequence
of the unmapped values in section 4.

**`probe_reviewstatus_gaps` states an unconditional requirement that is not done.** Its
closing verdict: *"FIX (c) IS REQUIRED IMMEDIATELY AND UNCONDITIONALLY: clean_cohort.py
needs a hard PRE-condition (no resolvable review column => raise, never silent
all-tier-5) and a hard POST-condition (ReviewStatus present in the written schema). It
guards rows but not its own schema."* No such guard exists.

**The latent underscore bug is confirmed live.** `clean_cohort.py` matches only `benign`
and `pathogenic`; `likely_benign` and `likely_pathogenic` are silently mapped to -1.
Currently inert because the source has zero duplicate `variant_id`, so the conflict
machinery never runs -- but it will mis-detect conflicts the moment a duplicate appears.

**A minor imprecision in the probe's own verdict.** It states *"Runs 1-17 used the
VCF-derived column"*. Run 17 has not executed; only its smoke tests have. Recorded in
`INCIDENT_2026-07-08_R2.md` section 5.

---

## 6. What this licenses, and what it does not

**Phase 1 entry criterion 4 is CLOSED.** The remedy is validated on deletions to the
extent the data permits: not by agreement, which is impossible at 1.166 percent overlap,
but by demonstrated consistency with the validated indel class. The finding is recorded
with its limits rather than as a clean pass.

**It does not license regenerating the cohort yet.** Two things must accompany the
regeneration and neither is done:

1. The two unmapped review statuses in section 4 must be resolved deliberately -- mapped
   to a tier on stated reasoning, or excluded on stated reasoning. They must not remain
   unmapped through a repair whose entire subject is values silently receiving the worst
   tier.
2. The three tier-mapping implementations must be reconciled, per
   `probe_reviewstatus_gaps`'s own recommendation, so that the repaired cohort is built
   by one function with one documented default rather than by whichever of three
   implementations a given entry point happens to call.

**And the wording for anything built on the repaired cohort is fixed by this
measurement:** deletion review statuses in the repaired cohort are adopted on
distributional consistency with insertions, not on verified agreement, because verified
agreement covers 1.166 percent of them. That sentence belongs wherever the repaired
cohort is cited.

**Phase 1 entry criterion 3, the lineage sweep, remains open.**
