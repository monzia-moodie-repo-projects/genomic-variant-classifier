# CONTAINMENT 2026-07-24 -- REVISION 2, dated 2026-07-24

**This document does not replace revision 1.** `docs/CONTAINMENT_2026-07-24.md` remains
immutable except for a pointer added to its header. Per the governing principle it itself
adopted, no scientific artifact is silently replaced; every correction creates a new
version linked to its predecessor.

**Why revision 2 exists.** Revision 1 was written before the artifact lineage sweep. It
said so plainly -- section 3 listed the artifacts *known* to be affected and section 8
made the sweep a Phase 1 entry criterion precisely because that inventory was incomplete
by construction. The sweep ran at 2026-07-24T08:05:21Z across 2,595 artifacts and 13.55
gibibytes. Three of revision 1's statements now need amending, and one of its assumptions
needs reversing.

**Revision 1 SHA-256 at time of writing:**
`3C78F53E2F72B899165170698F4E96EA139CBCC15634869DE7ABBEB84CB893B2`, 10,286 bytes,
214 lines.

**Repository state:** `main` at `c968976a3cc25deb9e6c32f85b61d9b907024958`.

**Evidence:** `docs/measurements/MEASUREMENT_2026-07-24_artifact-lineage-sweep.md` and
`docs/audits/evidence/2026-07-24/ARTIFACT_LINEAGE_SWEEP_2026-07-24.json`.

**Acronyms on first use.** AUROC = area under the receiver operating characteristic curve.
AUPRC = area under the precision-recall curve. LOVD = Leiden Open Variation Database.

---

## 1. Amendment 1 -- the AlphaFold quarantine scope, named

Revision 1 section 4 quarantined the four Phase D structural features without establishing
which runs produced them. The sweep establishes it.

**225 artifacts carry all four quarantined features**, with no partial cases. The date
range is **2026-03-30T02:47:07** (`outputs/phase4_retrain/splits/X_train.parquet`) to
**2026-07-06T00:05:49** (`outputs/smoke_cnn_tier1/meta_val.parquet`) -- a span of
**98 days**.

**Runs carrying structural features but NOT a top-level `ReviewStatus`** -- that is,
regime-v0 runs the deletion incident correctly exempts, which are nonetheless affected by
the structural defect:

    run9_ready      run9_fresh      run10b_final    run11
    run12           run13           run14           phase4_retrain
    verify_sources  d1_d2_minitest  probe_patch6b   experiments

`docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md` states that Runs 9
through 14 are unaffected. **That is true of the deletion defect and false of the
project's artifacts overall.** Both statements must travel together wherever either is
cited.

**Lineage state for all 225: AFFECTED.**

---

## 2. Amendment 2 -- `VALID` does not mean fit for use

The sweep classifies `clinvar_grch38_clean_v2_verified.parquet` and
`clinvar_grch38_noalleles.parquet` as **VALID**, on the basis that they carry neither a
top-level `ReviewStatus` nor a quarantined structural feature. That classification is
correct.

It is also not a clearance. The cohort inventory of 2026-07-24T04:00:09Z measured
**21,091** and **4,420,177** null-or-empty allele rows in those two files. Both would
raise at `real_data_prep.py:476`.

**`VALID` means "descends from neither known defect". It does not mean "fit for use".**

Revision 1 section 5 already says clean and unaffected are different properties. The
sweep's own output does not restate it, and a reader of the sweep alone could take
`VALID` as an endorsement. **The state definition in revision 1 section 2 is amended to
read:**

> `VALID` -- produced by a path with no known defect, and its inputs are `VALID`.
> **This is a statement about LINEAGE only. It carries no claim about content, schema,
> cleanliness or fitness for any purpose. An artifact may be lineage-`VALID` and still
> fail `_assert_clean_cohort`.**

---

## 3. Amendment 3 -- the smoke cohorts are affected, and they gate launches

Four smoke-test cohorts carry a top-level `ReviewStatus` and are therefore **AFFECTED**:

| File | Rows | Modified |
| --- | ---: | --- |
| `data/processed/clinvar_smoke.parquet` | 80,000 | 2026-06-05 |
| `data/processed/clinvar_smoke_seq.parquet` | 5,000 | 2026-06-12 |
| `data/processed/clinvar_smoke3000.parquet` | 3,000 | 2026-07-05 |
| `data/processed/clinvar_smoke3000_seq.parquet` | 3,000 | 2026-07-05 |

Smoke results are not scientific claims, so this is not a metrics problem. **It is a gate
problem.** The all-models smoke law requires every model to fit on a tiny smoke cohort
before any full or cloud run, and any model that errors, skips or produces a degenerate
out-of-fold result blocks launch. **That gate has been evaluating against defective inputs
since 2026-06-05.**

A smoke pass on an affected cohort does not establish that the same models will fit on a
repaired one -- the repaired cohort has a different class balance, a different deletion
share, and 178,563 rescued deletions. **The smoke cohorts must be regenerated with the
repaired cohort before the smoke law can gate a post-repair launch.**

---

## 4. The assumption that needs reversing -- phase ordering

Revision 1 section 1 sequences Phase 1 as "the deletion join and the AlphaFold resolver".
Written in that order, and the whole session up to the sweep treated the deletion incident
as the lead item on severity.

The artifact evidence says the reverse:

| | Structural defect | Deletion defect |
| --- | ---: | ---: |
| artifacts carrying the direct marker | **225** | 79 |
| carrying only this marker | **161** | 15 |
| earliest artifact | **2026-03-30** | 2026-06-03 |
| span | **98 days** | 45 days |
| runs reached that the other does not | **12** | 0 |

**This document does not reorder Phase 1.** Which defect to repair first is a scientific
judgement about whether breadth or label-correlation dominates, and it belongs to Monzia.
What revision 2 records is that the ordering in revision 1 was written without this
evidence and should not be treated as settled by default.

Two considerations, stated so the judgement can be made on them rather than on momentum:

**For the deletion defect first.** Its bias is *label-correlated* -- 34.556 percent of
pathogenic variants survive against 95.236 percent of likely-benign -- which is the kind
of defect that distorts a classifier's learned decision boundary rather than merely
degrading a feature. It also gates the LOVD work, whose deletion-weighted ranking rests on
a scarcity this defect largely manufactures.

**For the structural defect first.** It is broader (225 against 79), older (98 days
against 45), and it reaches twelve runs the deletion defect does not. It also produces
values that are *wrong rather than absent* for 220,590 variants, which is the worse of the
two failure modes: a sentinel can be modelled, a plausible wrong number cannot.

---

## 5. New containment item -- artifacts with no determinable lineage

The sweep found **985 artifacts, 38.0 percent of everything inspected, whose lineage
cannot be established at all.**

| Count | Bytes | Group |
| ---: | ---: | --- |
| **251** | 807.8 MiB | serialised models with no lineage recorded in the file |
| 734 | 698.2 MiB | non-model artifacts modified after 2026-06-03 carrying no marker |

**Nothing in a serialised estimator records which cohort trained it.** Not joblib, not the
CatBoost binary, not the PyTorch checkpoint. This is not a gap in the sweep; it is a gap
in the artifacts, and it is unresolvable after the fact by any amount of inspection.

**Containment:** no model artifact in `outputs/` or `models/` whose lineage is
UNDETERMINED may be cited in any comparison, promoted to a baseline, or used to seed a
warm start, until either a provenance manifest is reconstructed for it or it is
regenerated. That covers 251 files and 807.8 mebibytes.

This is the concrete argument for the provenance manifests revision 1 deferred to Phase 3.
**The cost of not having them is now measured: 38 percent of the artifact estate is
unclassifiable.**

---

## 6. New containment item -- an existing quarantine convention, undocumented

`data/processed/_invalidated_2026-07-09/clinvar_grch38_clean_v3_verified.parquet`, 134.8
mebibytes, 4,400,192 rows, sits beside a file of the same name and identical size and row
count at the top level of `data/processed`, six hours newer.

Someone has quarantined a cohort before, by directory convention, on 2026-07-09. There is
no manifest, no README in that directory, and no mention of it in any document read during
this session.

**Either adopt the convention formally** -- a documented `_invalidated_<date>/` directory
carrying a note stating what was invalidated, why, and what replaced it -- **or retire
it**, moving the file somewhere its status is explicit. An undocumented quarantine
directory is a trap for exactly the reader this containment exists to protect.

---

## 7. What is unchanged from revision 1

Every containment item in revision 1 sections 3 through 6 stands: the deletion defect
scope, the AlphaFold feature quarantine, the uncertified-cohort rule, and the LOVD freeze.
The five lineage states and seven feature-provenance states stand, with the `VALID`
definition amended per section 2 above. The four-phase sequencing stands, with the
ordering *within* Phase 1 reopened per section 4.

Phase 1 entry criteria 3 and 4 are both now **met**. Criterion 4 was closed at
2026-07-24T07:32:59Z; criterion 3 by the sweep at 08:05:21Z.

---

## 8. What still blocks Phase 1, and it is not an entry criterion

`probe_label_column_terms_2026-07-24.py` has not been run.

`real_data_prep._load_and_label` labels from `clinical_sig` using
`.isin({"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"})` -- exact and
case-sensitive. `scripts/probe_tier_filter_impact.py` records that the data uses
underscores. If `clinical_sig` carries the underscore form, that match returns nothing and
**every likely-pathogenic variant is dropped as unlabelled** -- a defect larger than either
of the two being contained here.

It takes seconds and it outranks everything in this document. Until it has run, no repair
should begin, because a repair built on a cohort whose labelling is itself broken would
have to be redone.
