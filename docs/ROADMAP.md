# GenAssoc / genomic-variant-classifier — Living Roadmap

**Version:** 2026-06-08 (v1, re-baselined)
**Owner:** Monzia Moodie
**Repo:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Status of this document:** authoritative source of truth; supersedes phase
language scattered across prior session docs. To be updated at the END of every
session (see Standing Disciplines). Drive copy pushed via `rclone genvarcla:`.

> **Maintenance note (honest):** this roadmap had drifted out of date — session
> work (ESM-2 activation, the split-health audit, the data-source expansion) was
> not being folded back in. This re-baseline fixes that and the document is now
> the gating artifact, not an afterthought.

---

## 1. Project identity & goals

A production-grade, multi-modal genomic disease-association program. Current
core: an ACMG/AMP-style variant pathogenicity classifier over ~1.7M ClinVar
variants (2,488,903 missense in the working cohort), 78 engineered features,
an 8-model ensemble + stacking meta-learner + STRING-DB GNN + KAN.

**Dual goal (both first-class):**
1. Classify variants and draw statistical inferences to disease categories/phenotypes.
2. Empirically measure/compare/validate how different ML algorithms behave on
   large, complex, real-world data — including newer architectures (KAN, GNN, GraphGPS) —
   even when performance differences are small.

**Default stance:** implement / keep / study. Never drop models or features on
marginal-AUROC grounds. Data is central; maximize information.

---

## 2. Phase model (PROPOSED canonical — pending Monzia's sign-off)

Prior docs use three conflicting numbering schemes (PHASE_1_ASSESSMENT's
"Phase 1 bugfix -> Phase 2 VEP/SpliceAI/AM"; a "Phase 0 foundation" framing; and
a "Phase 2 = 7 databases" framing). They cannot all be right. Proposed single
scheme, mapping the legacy labels onto it:

| Phase | Name | Scope | Legacy label it absorbs | State |
|-------|------|-------|-------------------------|-------|
| F | Foundation | Bug-fixed core pipeline, 8-model ensemble + GNN + KAN, real ClinVar, run discipline | "Phase 1" (bugfix), "Phase 0" | LARGELY DONE (Run 14/15) |
| D | Data expansion | Wire annotation sources; activate dead feature columns; add new sources | "Phase 2 (VEP/SpliceAI/AM)" partly done; "Phase 2 = 7 databases" | IN PROGRESS (this is where we are) |
| P | Performance/infra | Polars data layer, Rust inference microservice | roadmap "Phase 3/5" | NOT STARTED |
| A | Advanced modeling | Julia custom training, VAE, Bayesian UQ, GraphGPS/GNN opts | roadmap "Phase 4" | NOT STARTED |
| X | Productionization | REST API, Docker deploy, clinical eval/report hardening | "Phase 2 (API/Docker)" framing | NOT STARTED |

> So the honest answer to "are we still in Phase 1?": the foundation is done; we
> are in **Phase D (data expansion)**. The label confusion is exactly why this
> table exists. Adjust the names/letters if you prefer; then this is canonical.

---

## 3. Current state snapshot (2026-06-08)

- **Run 14 sealed** (Test AUROC 0.9975, OOF blend 0.9985, KAN OOF 0.9921, $2.17).
- **Run 15 splits exist** at `outputs/run15_rerun_report/full/splits/` (9 parquets:
  X/meta/y x train/val/test; train 1,038,974 / val 146,329 / test 304,711; X=78 cols).
- **ESM-2 activation in flight:** batched + device path shipped (`5a4d103`),
  equivalence-gated; CPU confirmed compute-bound (38.9 ms/variant at 4 physical
  cores; ~31 h full-cohort CPU) -> GPU regen is the chosen path. 18,621/19,383
  cohort genes (96.1%) prefetched into `esm2_cache.sqlite`.
- **Split-health audit (`scripts/audit_split_feature_health.py`):** of 96 columns,
  **54 healthy / 42 degenerate.** Splits predate the HGVSp/coords step (`8696ede`)
  and the ESM-2 device path, so they are stale. `gnn_score` is HEALTHY here
  (the Run-14 merge-back zero is not present). Full categorization in
  `docs/SPLIT_HEALTH_2026-06-08.md`.
- **Known code anomaly:** `gene_constraint_oe` / `gene_is_constrained` ALL_ZERO
  while `loeuf`/`pli_score` (same source) are healthy — suspected superseded
  schema columns. Tracer: `scripts/diagnose_constraint_columns.py`.
- **Suite:** 775 passed / 6 skipped (torch_geometric Windows DLL importorskip).

---

## 4. Data-source registry (THE answer to "can X be wired?")

Honest triage. "Level" = where the signal attaches. "Fit" reflects relevance to
*this* variant-pathogenicity feature model, not the resource's general value.

### 4A. Wired & healthy (producing signal today)
ClinVar (labels + cohort), gnomAD v4 (LOEUF, pLI, AF), AlphaMissense, SpliceAI
(`splice_ai_score`, `is_splice`), dbNSFP (SIFT, CADD, REVEL, GERP, PolyPhen),
STRING-DB v12 (GNN graph -> `gnn_score`), ClinVar-derived `n_pathogenic_in_gene`.

### 4B. Scaffolded but DEAD — column exists, source not wired (Phase D targets)
| Source | Dead column(s) | Access | Note |
|--------|----------------|--------|------|
| ESM-2 | `esm2_delta_norm` | local model + cache | UNBLOCKING NOW (GPU regen) |
| (schema) | `gene_constraint_oe`, `gene_is_constrained` | n/a | vestige; fix via tracer |
| PhyloP | `phylop_score` | free bigWig | conservation |
| GTEx | `gtex_*` (6) | free | eQTL/expression |
| 1000 Genomes | `af_1kg_*` (5) | free VCF | population AF |
| dbSNP / RefSNP | `dbsnp_af` | free | rsID + common-variant + AF |
| OMIM | `omim_*` (2) | license (free academic w/ registration) | disease/inheritance |
| ClinGen | `clingen_validity_score` | free API | gene-disease validity |
| FinnGen | `finngen_*` (3) | free (summary) | population enrichment |
| MaxEntScan | `maxentscan_score` | free tool | splice strength |
| VEP | `codon_position`, `exon_number` | free tool | coding context |
| Protein structure | `alphafold_plddt`, `solvent_accessibility`, `secondary_structure_context`, `dist_to_active_site`, `has_uniprot_annotation` | free (AlphaFold DB) | see 4C |
| EVE | `eve_score` | free score files | needs the score files |
| HGMD | `hgmd_*` (2) | **PAID, procurement-blocked** | label-leakage rules apply |
| LOVD | `lovd_variant_class` | free | tiny coverage |

### 4C. New candidates assessed (incl. your latest list) — honest verdicts
| Source | Fit | Level | Access / license (verified) | Effort | Verdict |
|--------|-----|-------|------------------------------|--------|---------|
| **AlphaFold DB** | HIGH | residue/protein | free; per-UniProt model **with pLDDT**; bulk download | Moderate | DO — revives the entire dead structure block (DSSP -> SS + accessibility; UniProt features -> active site) |
| **RefSNP / dbSNP** | HIGH | variant | free | Low | DO — directly fills `dbsnp_af` + adds common-variant flag |
| **COSMIC** | HIGH | variant/gene | free academic w/ registration; **commercial licensed** | Moderate | DO (academic) — somatic recurrence/hotspots; treat as feature NOT label (leakage) |
| **TCGA** | MED-HIGH | gene/variant | open tier free via GDC; controlled tier via dbGaP | Moderate | OPTIONAL — somatic recurrence + expression context; controlled parts blocked |
| **Reactome** | MED | gene/pathway | free, open | Low-Med | OPTIONAL — pathway-membership features + GNN edge augmentation |
| **KEGG** | MED | gene/pathway | API free academic; **bulk FTP licensed** | Low-Med | OPTIONAL — overlaps Reactome; mind bulk license |
| **BioGRID** | MED | gene/interaction | free | Low-Med | OPTIONAL — interaction edges; OVERLAPS STRING (marginal) |
| **DepMap** | MED | gene | free CSV (quarterly, ~1,320+ lines, Chronos/CERES) | Low | OPTIONAL — gene essentiality; ablate vs LOEUF/pLI for redundancy |
| **CPTAC** | MED | gene/protein | processed tables open via `cptac` pip pkg (**CC-BY-NC-ND**); raw via dbGaP | High | LATER — cancer proteomics; non-commercial license; heavy MS matrices |
| **dbGaP** | n/a | (gateway) | **controlled — needs eRA Commons + DAR + institutional Signing Official; BLOCKED w/o R1 faculty sponsor** | n/a | NOT A SOURCE — it is the access gate for TOPMed/controlled-TCGA/CPTAC-protected |
| **ProteomeXchange** | LOW | dataset index | free | High | SKIP for now — federation of MS repos (PRIDE/PeptideAtlas/MassIVE/jPOST); dataset-discovery, not a clean feature; CPTAC covers the useful slice |
| **SRA / ENA / DDBJ DRA** | OUT | raw reads | free | Very high | SKIP — INSDC raw-sequencing archives; not annotation features unless the project pivots to reprocessing raw reads |
| **SILVA** | OUT | rRNA taxonomy | free | n/a | SKIP — ribosomal-RNA / microbial taxonomy DB; not relevant to human variant pathogenicity |

**Reading of your list:** strong direct fits are **AlphaFold DB, RefSNP/dbSNP,
COSMIC, TCGA, Reactome/KEGG**; **BioGRID** is graph augmentation that overlaps
STRING; **dbGaP** is an access prerequisite (and blocked), not a connector;
**ProteomeXchange/SRA/ENA/DDBJ-DRA/SILVA** do not produce variant-pathogenicity
features for this model and are out of scope absent a pipeline pivot. Including a
source for the "study many features" goal is fine — but raw-read archives and
rRNA taxonomy do not yield variant features without a different pipeline.

---

## 5. Immediate plan

1. **Fix the constraint vestige** — run `diagnose_constraint_columns.py`; drop or
   alias `gene_constraint_oe`/`gene_is_constrained` so the regen doesn't reproduce
   a zero column.
2. **Regen strategy decision (open):**
   - **(A) ESM-2-only regen now** — fastest to a Run-15 retrain with ESM-2 signal;
     a second regen later for new sources.
   - **(B) Wire a Phase-D batch first, then one comprehensive regen** — e.g.
     AlphaFold-structure + dbSNP + PhyloP + GTEx + 1KGP (+ optionally DepMap/COSMIC),
     all banked in one GPU pass. More upfront connector work; avoids paying for the
     regen twice; serves the maximize-information goal.
3. **GPU regen (measure-first):** prep coords locally (cheap CPU lookups; sequences
   pre-cached, no UniProt on the box), ship variant frame + `esm2_cache.sqlite` to a
   cheap 4090, run a short `--device cuda` throughput probe and QUOTE the measured
   number before committing, forward-only regen with incremental checkpointing,
   rebuild splits, `echo y | vastai destroy` (separate re-paste block).
4. **AFTER check:** re-run `audit_split_feature_health.py`; ESM-2 + the 3 ALL_NULL
   prerequisites must flip healthy; the 54 healthy must stay healthy.
5. **ALL-MODELS smoke** (tiny `--max-train`, no `--skip`, every model incl GNN/KAN)
   before the Run-15 retrain.

---

## 6. Modeling & infra roadmap

- **Ensemble:** RF, XGBoost, LightGBM, SVM, LR, GBM, 1D-CNN, TabularNN + stacking
  meta-learner; CatBoost; MC-Dropout; Deep Ensemble. Per-model algorithm comparison
  documented every run.
- **GNN (Phase D/A):** fix-already-confirmed `gnn_score` healthy in Run-15; opts =
  bf16 AMP (NOT fp16 on Ada/cc8.9), PyG SparseTensor/CSR, GraphGPS hybrid
  (`GPSConv` wrapping `GATConv` + Performer), Laplacian PE/RWSE, 3-channel STRING
  weights. GPU-only; 2-epoch probe before any full run.
- **KAN:** reinstated (imodelsx, instance-attribute fix). Study vs MLP/trees.
- **Performance (Phase P):** Polars data layer; Rust inference microservice.
- **Advanced (Phase A):** Julia training, VAE, Bayesian UQ.
- **Productionization (Phase X):** REST API, Docker, clinical eval/report.

---

## 7. Standing disciplines (condensed; full text in session docs)
- Pre-flight gate; local mini-test before cloud; goal realignment each run.
- Measure-first (no time estimates without a probe); ALL-MODELS smoke before training.
- Incremental checkpointing on long runs; irreversible/cloud cmds in separate re-paste.
- Count-guarded, backup-first, idempotent, sandbox-validated patchers; byte-IO on Windows.
- Document every run (algorithm comparison + living metrics glossary); keep THIS roadmap current.
- Never drop models/features; scope ambiguity -> STOP + ask with options + pros/cons.

---

## 8. Blockers
- **HGMD Professional** — procurement (QIAGEN trial; institutional seat; PI sponsorship).
  REVEL/VEST4/FATHMM/MutPred2 must NOT be labels if HGMD is a label source.
- **dbGaP / TOPMed / controlled-TCGA / CPTAC-protected** — need institutional Signing
  Official; blocked without R1 faculty sponsor.
- **EVE** — needs score files before `eve_score` is real.

---

## 9. Changelog
- **2026-06-08 (v1 re-baseline):** roadmap reconstructed; phase model proposed;
  data-source registry added (incl. AlphaFold/dbSNP/COSMIC/TCGA/Reactome/KEGG/BioGRID/
  DepMap/CPTAC + out-of-scope SRA/ENA/DDBJ/SILVA/ProteomeXchange + dbGaP clarification);
  split-health audit + constraint-vestige finding recorded; ESM-2 GPU-regen plan set.
