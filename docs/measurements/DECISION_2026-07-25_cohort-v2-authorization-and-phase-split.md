# Decision record: clean-cohort v2 authorization and Phase 1b split (Option C)

- **Date:** 2026-07-25
- **Status:** ADOPTED
- **Scope:** clean-cohort adjudication, clinical-significance ontology, downstream metric baselines
- **Supersedes framing of:** the open "clean_cohort strict-resolver rewire" as a single Step 1b consumer task
- **Does NOT supersede:** the shipped `review_status.py` resolver, or the two completed Step 1b rewires (augment_reviewstatus ce5731a, real_data_prep 78631a5)

---

## 1. Context

The Step 1b work to move review-status consumers onto the canonical strict resolver
reached `clean_cohort.py`, the certified cohort builder. Investigating that rewire
surfaced a defect deeper than a map divergence: the builder's duplicate-group
representative selection is **input-order dependent** (a stable sort followed by
positional `.iloc[0]`), so the physical Parquet row order participates in adjudication.
Pursuing this to ground truth produced a sequence of read-only measurement probes and,
through four Monzia decision documents (path D, P5+P6, multi-axis ontology, and this
Option C), a fully specified redesign: a group-level evidence adjudicator (P6) operating
over a lossless multi-axis parse of `clinical_sig`, with the binary training label
derived by a separate, versioned target policy.

The evidence phase is now complete. This record fixes the phase boundary, authorizes the
construction phase without starting it, and records the dependency that governs when
production metric work may be certified.

## 2. Decision

Adopt **Option C**: formally close the evidence/design phase, preserve its artifacts
immutably, and permit cohort-independent metric-stack infrastructure to proceed under a
hard dependency gate. Precisely:

> The clean-cohort **investigation and design** phase (Phase 1b-E) is **COMPLETE**. The
> clean-cohort **v2 implementation** phase (Phase 1b-C) is **AUTHORIZED but OPEN**.
> Metric-stack architecture may proceed, but **no production metric baseline, scientific
> comparison, or clinical conclusion may be certified against the superseded v1 cohort or
> before v2 cutover.**

Rejected alternatives:

- **(a) Continue directly into v2 construction, holding all other work.** Rejected: it
  converts a validated measurement track into an open-ended implementation epic
  (ontology, typed evidence contracts, target-policy API, group adjudication,
  deterministic selection, evidence sidecar, certification states, manifests, output
  hashing, launch refusal, cohort migration) and needlessly delays metric-stack
  engineering that has no cohort dependency.
- **(b) Declare the clean-cohort track done and proceed to the metric stack.** Rejected as
  stated: it risks implying v1 metric results are current scientific truth. v1 and v2 are
  not interchangeable -- v2 is expected to change representative rows, binary labels,
  quarantine membership, trainable count, positive count, and source semantics.

Option C is the disciplined form of (b): it preserves the evidence, respects the original
metric-stack sequencing, avoids rushed foundational construction, and prevents obsolete v1
semantics from contaminating the new evaluation framework.

## 3. Status assertions

```text
Clinical-significance ontology v1.0 : MEASUREMENT-VALIDATED (specification version)
P6 group adjudication               : DESIGN-VALIDATED
Certified cohort v2                 : AUTHORIZED, NOT IMPLEMENTED
Current v1 cohort                   : retained for lineage
                                      NOT reproducibly derived under a deterministic
                                        adjudication policy
                                      NOT to be overwritten
Production metric rebaselining      : BLOCKED_BY_COHORT_V2
```

`CLINICAL_SIGNIFICANCE_ONTOLOGY_VERSION = 1.0` denotes a **validated specification**, not
a shipped source-module version. The census probe hash and the locked 102-value
vocabulary inventory are the frozen reference against which the production implementation
will be checked.

## 4. Measured basis (all read-only, 2026-07-25, raw cohort clinvar_grch38.parquet, 4,420,180 rows)

Order dependence and P6 adjudication (from the corrected P6 audit):

- Legacy order-sensitive representative selections: **1,610** (P0); naive-unified: 1,612 (P1).
- Deterministic policies P2-P6 order-sensitive selections: **0**.
- P6 order-invariance: verified True across reverse and three within-group permutations
  (representative set, labels, states, quarantine all identical).
- Binary-vs-explicit-conflict labels withheld by P6 (Rule 4): **14**.
- Binary-vs-uncertain labels recovered by P6 (Rule 5, previously lost to file order): **189**.
- Net P6 trainable-row delta: **+175** (reconciles exactly: -14 + 189 = +175).
- Additional irreducible-conflict quarantines under P6: **22**.
- P6 representative-row changes vs legacy: 232; like-for-like kept-row label changes: 63;
  group-adjudicated label changes (stricter, separate basis): 203 (= 14 + 189).

Clinical-significance ontology census (from the ontology census probe):

- Distinct normalized `clinical_sig` values: **102**.
- Values recognized by the multi-axis parser: **102/102**.
- Unconsumed compound tokens: **0**. Rows with unconsumed tokens: **0**. (Fail-closed gate SATISFIED.)
- Ontology-only positive expansion (inclusive - production target policy): **+208** positives;
  negatives identical (1,359,716) across production/strict/inclusive views.
- Evidence-axis row coverage: VUS 2,299,678; B/LB 1,359,816; P/LP 341,401; absent 253,278;
  explicit-conflict 161,555; other-non-binary 2,730; pharmacogenomic 1,984; risk-allele 306;
  low-penetrance 209; VUS-subtype 30.

These deltas are the design basis; the production implementation must reproduce this
validated behavior before v2 certification.

## 5. What may proceed now; what is blocked

**May proceed (cohort-agnostic or synthetic-validated):** metric interface definitions;
typed metric result objects; capability and validation states; binary, calibration,
conformal, selective-prediction, subgroup, attribution, and out-of-distribution metric
implementations; cross-fitting contracts; bootstrap infrastructure; gene-cluster
resampling; synthetic sabotage tests; serialization and reporting schemas; deterministic
aggregation; metric provenance; failure and insufficient-support contracts.

**Blocked until v2 certification (`BLOCKED_BY_COHORT_V2`):** production expected metric
values; class-balance-dependent thresholds; empirical baseline tables; AUROC/AUPRC
comparisons; calibration baselines; conformal quantiles; decision-curve baselines;
subgroup prevalence; gene-ranking comparisons; release claims; clinical utility findings.

```text
Metric-stack implementation        : may proceed
Production metric backfill/certify  : BLOCKED_BY_COHORT_V2
```

**Architectural safeguard.** The metric stack must not import probe code or any temporary
P6 implementation. Metric code consumes a generic evaluation-table contract, defined so
infrastructure binds to a cohort *contract* rather than to legacy or v2 builder internals:

```python
class CanonicalVariantRecord(TypedDict):
    variant_id: str
    canonical_binary_label: int | None
    canonical_evidence_state: str
    adjudication_reason: str
    cohort_version: str
```

## 6. Dependency rule (governs order; the session label does not)

The metric-stack *implementation* may proceed in any order relative to v2 construction.
Production metric *backfill and certification* are blocked until v2 is certified.

**Exception:** if the next metric-stack task requires real production labels,
cohort-specific expected values, or the new evidence-summary fields, then v2 construction
moves ahead of that task. The dependency -- not the session label -- determines the order.

## 7. Phase model

**Phase 1b-E -- Evidence and design closure -- COMPLETE.** Outputs: adjudication audit;
order-dependence measurement; P0-P6 comparison; full `clinical_sig` vocabulary census;
validated multi-axis parser prototype; sensitivity views; reconciled label and quarantine
deltas; v2 design decision; hashes and execution provenance. Answers: *what is wrong, what
semantics should replace it, and is the design sufficiently specified to implement?* -- 
answered.

**Phase 1b-C -- Certified cohort v2 construction -- AUTHORIZED_NOT_IMPLEMENTED.** Outputs
will be: production ontology module; production target-policy module; production P6
adjudicator; deterministic metadata selector; evidence-summary artifact; v2 Parquet;
certification manifest; reproducibility and sabotage suite; downstream cutover
authorization. Answers: *has the validated design been correctly implemented and
certified?* -- not yet begun.

**Step 1b is NOT complete.** Record that 1b-E is complete and 1b-C is open; do not mark
Step 1b done until 1b-C completes.

## 8. Construction checklist (Phase 1b-C -- each gate independently testable and committable)

```text
C1  production ontology module (parser + typed evidence object; fail-closed invariant wired)
C2  target-policy module (versioned derive_binary_label; production/strict/inclusive views)
C3  P6 adjudication module (evidence-vector pathogenicity channel; Rules 2/3/4/5/6)
C4  deterministic representative selector (tier, semantic_rank, canonical_key; never file order)
C5  evidence-summary contract (sidecar; AdjudicationReason codes)
C6  order-invariance sabotage suite (within-group permutation; injected-defect catch)
C7  full-cohort dry run (4.4M rows; complexity + correctness gates)
C8  v2 artifact and manifest (clinvar_grch38_clean_adjudication-v2_<hash>.parquet, NOT overwriting v1)
C9  certification (ontology/target-policy/group-adjudication versions in manifest; gates 0/0/true)
C10 downstream cutover authorization (separate, explicit)
```

Manifest fields to record at C9:

```json
{
  "clinical_significance_ontology_version": "1.0",
  "binary_target_policy_version": "2.0",
  "group_adjudication_policy_version": "P6.1",
  "unrecognized_clinical_significance_values": 0,
  "unconsumed_compound_tokens": 0,
  "order_invariance_verified": true
}
```

## 9. Execution order adopted

```text
1. Commit all clean-cohort and ontology measurement artifacts (evidence-only)   <- this commit
2. Commit this v2 decision/authorization record                                 <- this commit
3. Mark Phase 1b-E complete                                                      <- ROADMAP update
4. Mark Phase 1b-C authorized and open                                          <- ROADMAP update
5. Proceed with cohort-agnostic metric-stack wiring
6. Begin dedicated cohort-v2 construction in its own focused session (C1-C10)
7. Complete and certify v2
8. Backfill, calibrate, and certify production metrics on v2
9. Authorize downstream cutover
```

The evidence is committed and reviewed as a coherent package before any production code is
written, so the exact design basis cannot be lost under subsequent edits.
