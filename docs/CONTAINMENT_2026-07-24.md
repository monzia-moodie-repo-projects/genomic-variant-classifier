# CONTAINMENT 2026-07-24 -- Phase 0

**Purpose:** stop further contamination. No design changes, no repairs, no regeneration.
Phase 0 is a governance act that costs hours and is fully reversible.

**Repository state:** `main` at `d2d02df0bc477e64e4172f97b254c11093d7209f`,
2026-07-24T02:26:14-04:00.

**Governing principle, adopted 2026-07-24:** no scientific artifact is ever silently
replaced. Every correction creates a new version linked to its predecessor, with an
explicit account of what changed, why, and which downstream artifacts are affected. This
applies to incidents, datasets, calibration models, conformal calibrators, feature
matrices, trained models and evaluation reports alike.

**Acronyms on first use.** AUROC = area under the receiver operating characteristic
curve. AUPRC = area under the precision-recall curve. LOVD = Leiden Open Variation
Database. VUS = variant of uncertain significance.

---

## 1. The four phases, and why Phase 0 is separable

| Phase | Content | Scale |
| --- | --- | --- |
| **0 -- Contain** | stop contamination; no design changes | hours |
| 1 -- Repair | the deletion join and the AlphaFold resolver. Nothing else. | days |
| 2 -- Revalidation | statistical only; no architectural work | days |
| 3 -- Architecture | source registry, typed evidence, provenance manifests, certification, evidence reliability layer | weeks |

The sequencing matters. An earlier draft of this work bundled containment with
architecture, which would have delayed containment behind a design programme. The
immediate hazard is that **220,590 cohort variants currently carry structural feature
values that are wrong rather than absent**, and that a training run launched today would
inherit a label-correlated deletion censoring. Both are stopped by Phase 0 alone.

---

## 2. Artifact lineage states

Adopted 2026-07-24. The primary object of containment is the **artifact**, not the run.
A run is affected if and only if one or more of its upstream artifacts is affected, and
every downstream derivative inherits the state.

| State | Meaning |
| --- | --- |
| `VALID` | produced by a path with no known defect, and its inputs are `VALID` |
| `PROVISIONAL` | produced by a path under active investigation; may become `VALID` or `AFFECTED` |
| `AFFECTED` | descends from a known-defective input or process |
| `SUPERSEDED` | replaced by a newer artifact; retained for provenance, not for use |
| `REGENERATED` | rebuilt after repair; carries a pointer to the artifact it replaces |

The state travels downstream automatically. Feature matrices, parquet exports,
out-of-fold predictions, probability calibrators, conformal calibrators, SHAP values,
permutation analyses, gene rankings, benchmark tables, figures and manuscripts all
outlive the runs that made them, and all inherit.

---

## 3. Containment item 1 -- the deletion defect

Basis: `docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md` and its
revision 2, `docs/incidents/INCIDENT_2026-07-08_R2.md`.

| Artifact | State | Basis |
| --- | --- | --- |
| `data/processed/clinvar_grch38_clean.parquet` after `augment_reviewstatus.py` | **AFFECTED** | the join fails on 98.834 percent of deletions |
| `data/processed/clinvar_grch38_clean_seq.parquet` after the same | **AFFECTED** | same |
| every training split derived from either | **AFFECTED** | inherited |
| Run 15 artifacts, including its two METRICS.md rows | **AFFECTED** | descends from the above |
| Run 16 artifacts | **AFFECTED** | descends from the above; no METRICS.md row exists |
| Runs 9 through 14 | not affected **by this defect** | the review-tier filter never executed; separately not comparable to regime v1 |
| Run 17 | **BLOCKED**, not affected | has not executed |

### Published metrics are archived, not invalidated

This is a deliberate distinction and the wording is load-bearing.

**Known defective** and **known materially changed** are different claims. The cohort is
known to be biased. Whether AUROC, AUPRC, calibration or gene ranking change materially
**is not yet known** and cannot be until the repaired cohort is regenerated and the
comparison measured. Declaring the metrics invalid today would assert something
unevidenced.

The wording applied to Run 15's METRICS.md rows is:

> These metrics were generated from a cohort subsequently found to contain a
> label-correlated deletion-selection defect (INCIDENT_2026-07-08). They remain archived
> for provenance but must not be compared directly with metrics from a repaired cohort.

### Run 17 launch blocker

    Run 17
      Status: BLOCKED
      Release criteria, in order:
        1. the deletion incident is closed
        2. the cohort is regenerated from the nested review-status field
        3. the regenerated cohort passes validation, including deletion-specific checks
        4. only then is training permitted

Rationale: if Run 17 trains from an affected cohort **after** the defect was identified,
the project loses the ability to characterise the defect as historical. That is a
governance loss, not merely a data one.

---

## 4. Containment item 2 -- the AlphaFold structural features

Basis: `docs/INCIDENT_2026-07-23_protein_pipeline_alphafold_fetch.md` and
`docs/audits/AUDIT_2026-07-24_alphafold_structural_coverage.md`.

`src/genomic_variant_classifier/pipelines/protein_pipeline.py:171` takes `data[0]`
unconditionally. Above the 2,699-to-2,700 residue boundary observed in the evaluated
AlphaFold release, no canonical model exists, so `data[0]` is an arbitrary isoform.
Measured: 559,786 of 4,399,089 cohort variants affected, **220,590 of them receiving
values that are wrong rather than absent**.

**All four Phase D structural features produced through this resolver are quarantined**:
`alphafold_plddt`, `solvent_accessibility`, `secondary_structure_context`,
`dist_to_active_site`.

### Feature provenance states

Adopted 2026-07-24. A single numeric column with a sentinel hides distinctions that
matter. Every structural feature should carry one of:

| State | Meaning |
| --- | --- |
| `VALID` | canonical model, sequence-identity confirmed, coordinates compatible |
| `MISSING` | no model available; genuinely absent, not wrong |
| `INVALID_SEQUENCE_MATCH` | a model was returned but its sequence does not match the canonical index |
| `INVALID_ISOFORM` | an isoform model was substituted for a canonical protein |
| `STALE_REFERENCE` | the upstream sequence version predates the local index |
| `UNSUPPORTED` | the source does not model this class of entity |
| `NOT_COMPUTED` | the pipeline did not attempt it |

Downstream code can then ignore invalid values, impute missing ones, model the
missingness explicitly, or analyse the failure -- instead of treating every case as a
sentinel. The value of this outlives the incident.

**Not implemented in Phase 0.** The states are declared here so that Phase 1's repair has
a target. Phase 0 quarantines; Phase 1 implements.

---

## 5. Containment item 3 -- uncertified cohort files

Basis: the cohort inventory of 2026-07-24T04:00:09Z.

**Nine of thirteen cohort parquet files in `data/processed/` are measurably NOT CLEAN**,
including two whose names assert otherwise:

| File | Rows | Null or empty alleles | Duplicate identifiers |
| --- | ---: | ---: | ---: |
| `clinvar_grch38_clean_v2_verified.parquet` | 4,420,180 | **21,091** | 4,217 |
| `clinvar_grch38_clean_v3_verified.parquet` | 4,400,192 | **1,103** | 0 |
| `clinvar_grch38_noalleles.parquet` | 4,420,180 | **4,420,177** | 513,428 |

`real_data_prep.py:476` raises on any of them. The four `_structural` files are correctly
not clean; they are the quarantine.

Measurably CLEAN: `clinvar_grch38_clean.parquet`, `clinvar_grch38_clean_seq.parquet`,
`cohort_fresh.parquet`, `cohort_stale.parquet` -- **all four of which are nevertheless
`AFFECTED` under section 3**, because clean and unaffected are different properties.

**Containment:** no training run may be launched against a cohort file that has not been
certified by measurement. Certification manifests are Phase 3; Phase 0 requires only that
the cohort inventory be run and its output read before any launch.

---

## 6. Containment item 4 -- LOVD expansion frozen

`docs/LOVD_ACQUISITION_PLAN_rev2_2026-07-24.md` is marked **PROVISIONAL**.

The plan's central ranking is deletion-weighted, and the deletion scarcity it responds to
-- 0.0521 percent of the training cohort -- is **substantially produced by the defect in
section 3**, not by the underlying data. Repairing the cohort raises deletion retention
from 1.023 percent to 86.237 percent, which will change the ranking counts the plan is
built on.

**No gene is acquired until the cohort is repaired and the opportunity ranking is
recomputed.** The eleven-gene Batch 1 is withdrawn as a first operational unit. When
expansion resumes it should begin with a small instrumented pilot spanning different
operational profiles, not a large batch, and the pilot's purpose is to measure the
acquisition funnel -- acquired rows, parseable, normalizable, cohort-joining,
independent, informative -- rather than to add volume.

---

## 7. What Phase 0 does not do

It does not repair anything. It does not regenerate any cohort. It does not retrain. It
does not delete or overwrite any artifact. It does not implement lineage or provenance
states in code -- they are declared here as the target for Phase 1 and Phase 3.

It also does not resolve the open questions in `INCIDENT_2026-07-08_R2.md` section 7,
including whether the nested-field remedy holds on deletions specifically, where
validation coverage is 1.166 percent.

---

## 8. Phase 1 entry criteria

Phase 1 may begin when all of the following hold:

1. This containment record is committed and pushed.
2. `INCIDENT_2026-07-08_R2.md` is committed, and revision 1 carries its pointer.
3. The affected-artifact inventory in section 3 has been checked against the actual
   contents of `data/processed/`, `outputs/` and `models/` -- section 3 lists what is
   known, and a lineage sweep has not been performed.
4. The nested-field remedy has been validated on deletions specifically, not only on the
   3,974,573 rows where both fields are populated.

Criteria 3 and 4 are measurements, not decisions, and neither has been made.
