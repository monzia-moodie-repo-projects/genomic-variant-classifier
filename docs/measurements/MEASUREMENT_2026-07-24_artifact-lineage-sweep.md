# Measurement 2026-07-24 (third) -- artifact lineage sweep: Phase 1 entry criterion 3

**Result: criterion 3 is CLOSED, and it inverts the priority I have been working under.**
The AlphaFold structural defect touches nearly three times as many artifacts as the
deletion defect and reaches back three times further in time. 985 artifacts -- 38 percent
of everything inspected -- have no determinable lineage at all.

**Repository state:** `main` at `c968976a3cc25deb9e6c32f85b61d9b907024958`.

**Source:** `sweep_artifact_lineage_2026-07-24.py`, SHA-256
`55E0175739EB6527635AE71BECA16F3DD30FC392B2387B0639A8301A02D14853`, run
2026-07-24T08:05:21Z across `data/processed`, `data/interim`, `outputs`, `models` and
`experiments`. All five roots present; zero scan problems reported.

**Scale:** **2,595 artifacts, 13.55 gibibytes.**

---

## 1. The headline

| State | Files | Share of files | Gibibytes | Share of bytes |
| --- | ---: | ---: | ---: | ---: |
| VALID | 675 | 26.0 % | 5.75 | 42.5 % |
| **AFFECTED** | **240** | **9.2 %** | **3.41** | **25.2 %** |
| PROVISIONAL | 695 | 26.8 % | 2.91 | 21.5 % |
| **UNDETERMINED** | **985** | **38.0 %** | 1.47 | 10.9 % |

The header count field in the report agrees with an independent recount of the record
array. Nothing was inferred from a summary.

---

## 2. The priority inversion

I have been treating `INCIDENT_2026-07-08`, the deletion defect, as the lead item on
severity, with the AlphaFold defect second. **The artifact evidence says the reverse.**

| | Structural defect | Deletion defect |
| --- | ---: | ---: |
| artifacts carrying the direct marker | **225** | 79 |
| artifacts carrying only this marker | **161** | 15 |
| carrying both | 64 | 64 |
| earliest artifact | **2026-03-30** | 2026-06-03 |
| latest artifact | 2026-07-06 | 2026-07-18 |
| span | **98 days** | 45 days |

All 225 structural artifacts carry **all four** quarantined features --
`alphafold_plddt`, `solvent_accessibility`, `secondary_structure_context` and
`dist_to_active_site` -- with no partial cases.

**Runs touched by the structural defect but NOT by the deletion defect**, that is,
regime-v0 runs the deletion incident correctly exempts:

    run9_ready, run9_fresh, run10b_final, run11, run12, run13, run14,
    phase4_retrain, verify_sources, d1_d2_minitest, probe_patch6b, experiments

`INCIDENT_2026-07-08` states that Runs 9 through 14 are unaffected. **That is true of the
deletion defect and false of the project's artifacts overall.** Those runs carry
structural features produced through the unsafe resolver. `docs/CONTAINMENT_2026-07-24.md`
quarantined the features without establishing which runs; this sweep establishes them.

The earliest structural artifact is `outputs/phase4_retrain/splits/X_train.parquet`,
2026-03-30T02:47:07 -- **four months before the defect was found.**

---

## 3. What is UNDETERMINED, and why it cannot be resolved by reading

985 artifacts, 1.47 gibibytes, in two groups:

| Count | Bytes | Group |
| ---: | ---: | --- |
| **251** | 807.8 MiB | serialised models with no lineage recorded in the file |
| 734 | 698.2 MiB | non-model artifacts modified after 2026-06-03 carrying no marker |
| 0 | -- | other |

By extension: **249 `.joblib`**, 239 `.txt`, 222 `.npy`, 169 `.json`, 35 `.tsv`, 30 `.csv`,
25 `.log`, plus one `.pt`, one `.cbm`, one `.sqlite` (121 MiB) and one `.bak_2026-07-18`
(523.6 MiB).

**Nothing in a serialised estimator records which cohort trained it.** Not the joblib, not
the CatBoost binary, not the PyTorch checkpoint. The largest are 54.34 MiB
(`ablation_probe/timing/models/ensemble_models/random_forest.joblib`, 2026-06-06) and a
run of seven 26.08 MiB support-vector machines from the 2026-06-19 ribonucleic-acid
sequencing ablations.

**This is the finding, not a gap in the sweep.** A provenance manifest written beside each
artifact at production time would resolve both groups. Neither is resolvable afterwards by
inspecting the artifact, which is precisely why the containment record lists provenance
manifests as Phase 3 work.

---

## 4. A limitation of my own sweep that its output does not state

`clinvar_grch38_clean_v2_verified.parquet` and `clinvar_grch38_noalleles.parquet` are both
classified **VALID**, with the basis *"no top-level ReviewStatus; nested review_status
present, so it predates or bypasses the defective augmentation."*

That classification is correct **and it is not an endorsement.** The cohort inventory of
2026-07-24T04:00:09Z measured **21,091** and **4,420,177** null-or-empty allele rows in
those two files respectively. Both would raise at `real_data_prep.py:476`.

**`VALID` means "descends from neither known defect". It does not mean "fit for use".**
`docs/CONTAINMENT_2026-07-24.md` section 5 says clean and unaffected are different
properties. The sweep's own output does not restate it, and a reader of the sweep alone
could take `VALID` as a clearance. The state vocabulary needs a note to that effect, or a
separate fitness axis.

---

## 5. Four findings the sweep produced that were not being looked for

**The smoke cohorts are affected.** Four smoke-test cohorts carry a top-level
`ReviewStatus`: `clinvar_smoke.parquet` (80,000 rows, 2026-06-05),
`clinvar_smoke_seq.parquet` (5,000, 2026-06-12), `clinvar_smoke3000.parquet` and
`clinvar_smoke3000_seq.parquet` (3,000 each, 2026-07-05). **Every smoke test since
2026-06-05 has run against a defective cohort.** Smoke results are not scientific claims,
so this is not a metrics problem -- but the all-models smoke law is a launch gate, and the
gate has been passing against affected inputs.

**`data/processed` holds 29 parquet files, not the 13 previously inventoried.** The
earlier inventory filtered to names beginning `clinvar` or `cohort` and therefore omitted
the annotation indices: `alphamissense_index.parquet` (71,697,556 rows),
`spliceai_index.parquet` (45,549,300), `dbsnp_index.parquet` (2,867,527),
`gnomad_v4_exomes.parquet` (2,951,148), `gtex_v11_gene_expression.parquet`,
`seq_windows.parquet` (4,420,180) and others. All are VALID by lineage. **A filter chosen
for one question silently bounded the answer to another.**

**A previously invalidated cohort is still on disk.**
`data/processed/_invalidated_2026-07-09/clinvar_grch38_clean_v3_verified.parquet`, 134.8
MiB, sits beside an identical-size, identical-row-count file of the same name at the top
level, six hours newer. Someone has done a manual quarantine before, by directory
convention, with no manifest and no record in the documents I have read. That convention
should either be adopted formally or retired.

**`clinvar_enriched.parquet` exists in four places.** One of them,
`outputs/run16/clinvar_enriched.parquet` at 213.0 MiB and 1,686,333 rows, is AFFECTED. The
row count matches the pre-tier-filter figure the incident records for regime v1.

---

## 6. Where PROVISIONAL sits

695 artifacts, 2.91 gibibytes, dated 2026-06-04 to 2026-07-10, concentrated in:

    outputs/run15_baseline       135 files   830.8 MiB
    outputs/run15_rerun_report   133 files   826.8 MiB
    outputs/ablation_run15       123 files   674.8 MiB
    models/                      144 files    59.2 MiB
    outputs/run17_smoke_stage1    79 files    70.4 MiB
    outputs/run16                 74 files   521.5 MiB

These are consistent with descent from a defective cohort by path and date and carry no
marker proving it. **Modification time is not lineage**, and the sweep says so in every
basis string rather than promoting them to AFFECTED.

---

## 7. What this changes

**Criterion 3 is closed.** Both declared Phase 1 entry criteria are now met: criterion 4
by the measurement of 2026-07-24T07:32:59Z, criterion 3 by this sweep.

**The containment record needs three amendments**, none of which I have made:

1. The AlphaFold quarantine scope should name the runs, which now range from
   `phase4_retrain` (2026-03-30) through `smoke_cnn_tier1` (2026-07-06), and should state
   that it reaches regime-v0 runs the deletion incident exempts.
2. The `VALID` state needs the note in section 4 -- lineage-valid is not fitness-valid.
3. The smoke cohorts should be listed explicitly, because the all-models smoke law is a
   launch gate and it has been gating on affected inputs.

**And the phase ordering deserves re-examination.** Phase 1 as written repairs the
deletion join and the AlphaFold resolver in that order. By artifact count, date range and
the fact that it reaches runs the deletion defect does not, **the structural resolver has
the larger blast radius.** Which to repair first is a judgement about whether breadth or
label-correlation dominates, and it is not mine to make.

---

## 8. What remains open

| # | Item |
| --- | --- |
| 1 | The label-column and term-set check, `probe_label_column_terms_2026-07-24.py`. Still unrun, and it outranks everything here: if `clinical_sig` carries underscore terms, `.isin` matches nothing and every likely-pathogenic variant is dropped as unlabelled. |
| 2 | The three review-tier decisions in `docs/measurements/DECISION_2026-07-24_review-tier-scale.md`. |
| 3 | The three containment amendments in section 7. |
| 4 | 251 serialised models with no recoverable lineage. Resolvable only by a manifest written at production time. |
| 5 | The `_invalidated_2026-07-09/` convention: adopt formally or retire. |
