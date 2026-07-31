# SESSION 2026-07-31 — the priority was backwards

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**On top of:** `30d184a`
**Ratchet:** 4121, unchanged — this session writes documents only
**Nothing was built.** This session read, and found that what had been written
needed correcting.

---

## 1. What was asserted, and what is true

Five roadmap deltas written on 2026-07-30 assert that `INCIDENT_2026-07-08`
*"remains OPEN"*, is a *"Tier 0 VALIDITY failure"*, and *"outranks all of it"*,
citing `project_metrics.txt` as the authority.

**The project's own adopted decision says the opposite.**
`docs/measurements/DECISION_2026-07-25_cohort-v2-authorization-and-phase-split.md`,
status **ADOPTED**, section 9:

    5. Proceed with cohort-agnostic metric-stack wiring
    6. Begin dedicated cohort-v2 construction in its own focused session
    7. Complete and certify v2
    8. Backfill, calibrate and certify production metrics on v2
    9. Authorize downstream cutover

**Metric-stack wiring is step 5. Cohort v2 is step 6.** And the decision rejects
the alternative explicitly — *"continue directly into v2 construction, holding
all other work"* — on the stated ground that it *"needlessly delays
metric-stack engineering that has no cohort dependency."*

---

## 2. Three defects in what was written

**One: the ordering is inverted.** Recorded above.

**Two: the cited authority is not in the repository.** A recursive search for
`project_metrics.txt` across the whole tree returns nothing. It exists only as a
file uploaded in an earlier session. Five deltas cite, as authority, a document a
reader cannot open.

That is the same defect `INCIDENT_2026-07-08_R2` section 3 recorded against
revision 1 — two commit hashes that fail `git cat-file -e` — and the rule
adopted in response reads: *"Every commit hash cited as evidence must be verified
to resolve in the repository the document lives in, at the time the document is
written, and the verification command must be recorded beside it."* **The rule
was written for hashes and applies to filenames identically.**

**Three: the characterisation is six days stale.** The deltas describe the
incident as though its subject were the variant-call-format join. It is no longer
that.

---

## 3. What the incident actually became

Investigating the `clean_cohort.py` strict-resolver rewire surfaced a defect
deeper than the join: **the duplicate-group representative selection is
input-order dependent** — a stable sort followed by positional `.iloc[0]` —
so **the physical Parquet row order participates in adjudication.**

    legacy order-sensitive representative selections   1,610
    under deterministic policies P2 through P6             0
    P6 order-invariance                verified across reverse and three
                                       within-group permutations, with
                                       representative set, labels, states and
                                       quarantine all identical

    P6 labels withheld (Rule 4)          14
    P6 labels recovered (Rule 5)        189   previously lost to file order
    net trainable-row delta            +175   reconciles exactly: -14 + 189
    additional quarantines               22

    clinical_sig distinct values        102
    recognised by the parser        102/102
    unconsumed compound tokens            0   fail-closed gate SATISFIED

So the plan is a **certified cohort v2** built by a group-level evidence
adjudicator over a lossless multi-axis parse of `clinical_sig`, with the binary
label from a separate versioned target policy. Not a join repair.

**Phase 1b-E — evidence and design — COMPLETE.**
**Phase 1b-C — v2 construction — AUTHORIZED_NOT_IMPLEMENTED**, checklist
C1 through C10, and the decision states it belongs *"in its own focused session."*

---

## 4. What today's work was, measured against the decision

Section 5 of the decision lists what may proceed:

> metric interface definitions; typed metric result objects; capability and
> validation states; binary, calibration, conformal, selective-prediction,
> subgroup, attribution and out-of-distribution metric implementations;
> cross-fitting contracts; bootstrap infrastructure; gene-cluster resampling;
> synthetic sabotage tests; serialization and reporting schemas; deterministic
> aggregation; metric provenance; failure and insufficient-support contracts.

That is a precise description of registry commit 2, the three absent metrics, and
`risk_control`. And section 5's blocked list — production expected values,
empirical baseline tables, area-under-curve comparisons, calibration baselines,
conformal quantiles, gene-ranking comparisons, release claims — was not touched
by any of it.

**The project has been executing step 5 continuously for six days:**

    2026-07-26    2 session documents
    2026-07-27    8
    2026-07-28   10
    2026-07-29    8
    2026-07-30    5
                 33 total

The metric registry, the evaluation population contract, the authority switch,
the carried-item register, the typed report surface, absence vocabulary, registry
commit 1 — then today's five. **It was never blocked on the incident.**

---

## 5. Two further findings

**The guard the specification asked for was never built.**
`docs/PHASE1_SPEC_2026-07-24_deletion-repair.md` section 5, test 6, requires *"an
assertion that after augmentation, deletions with a populated review status exceed
150,000, so a regression to the join-based source turns the suite red rather than
quietly re-censoring the cohort."*

A search for `150000`, `150_000`, `retention floor` and `deletion.*populated`
across every test file returns **nothing**. Five of the six specified tests
landed at `45525fb`; the sixth — the only one that measures the actual defect
— did not. **The suite is green because the detector is absent, not because the
repair happened.**

**And the defect line survives, renumbered.** The incident cites
`augment_reviewstatus.py:64`. The file is now 155 lines and the line is at 118:

    df["ReviewStatus"] = key.map(vmap).fillna("")

Only the trailing comment changed, from `tier 5` to `TIER_MISSING`. **Step 1b did
land** — the local tier map and the substring resolver were removed, and
unknown vocabulary now raises with a complete inventory via
`_build_strict_tier_lookup`. That is the silent-demotion fix. The join-source fix
is a different change and has not landed.

---

## 6. What is corrected, and what is not

**Corrected:** the roadmap gains a dated correction section recording the adopted
phase model, the execution order, the unresolvable citation, the order-dependence
defect, and the missing guard.

**Not corrected:** the five deltas of 2026-07-30 are **not edited**. They are
dated records of what was written that day. `ROADMAP.md:1355` was amended rather
than rewritten on 2026-07-30 for the same reason, and
`INCIDENT_2026-07-08_R2` line 10 states the principle: *"no scientific artifact
is ever silently replaced. Every correction creates a new version linked to its
predecessor, with an explicit account of what changed, why, and which downstream
artifacts are affected."*

**Nothing in the incident is repaired here.** Repairing the join would repair a
symptom of a defect the project has already characterised more deeply, under a
plan the 2026-07-25 decision superseded.

---

*Written 2026-07-31. The carried-item register decides status; `tests/EXPECTED_SUITE_SIZE`
decides the count.*
