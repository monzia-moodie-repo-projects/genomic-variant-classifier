# GenAssoc / genomic-variant-classifier - Living Roadmap

**Version: 2026-06-09 (v2)  |  Owner: Monzia Moodie**

Repo: github.com/monzia-moodie-repo-projects/genomic-variant-classifier

*Authoritative source of truth; supersedes phase language in prior session docs. Updated at the end of every session; Drive copy via rclone genvarcla:.*

*v2 note: Run 15 completed (full cohort, ESM-2 mechanically active, constraint columns revived, unseen-gene-holdout ablation). Findings folded into the snapshot, data-source registry, and immediate plan below.*

# 1. Project identity & goals

Production-grade multi-modal genomic disease-association program. Core: an ACMG/AMP-style variant pathogenicity classifier over ~1.49M cohort rows (from ~2.49M ClinVar missense), 80 features, 13-model ensemble + stacking meta-learner + STRING-DB GNN + KAN.

**Dual goal (both first-class):**

- Classify variants and infer disease categories/phenotypes.

- Empirically measure/compare/validate ML algorithms on large complex data, incl. KAN/GNN/GraphGPS, even at small performance differences.

**Default stance: implement / keep / study. Never drop models/features on marginal-AUROC grounds.**

# 2. Phase model

| **Phase** | **Name** | **Scope** | **Legacy label** | **State** |
| --- | --- | --- | --- | --- |
| F | Foundation | Bug-fixed core, ensemble+GNN+KAN, real ClinVar | Phase 1 / Phase 0 | DONE |
| D | Data expansion | Wire sources; activate dead columns; add new sources | Phase 2 (partly) | IN PROGRESS |
| P | Performance/infra | Polars layer; Rust inference service | roadmap 3/5 | NOT STARTED |
| A | Advanced modeling | Julia training, VAE, Bayesian UQ, GraphGPS | roadmap 4 | NOT STARTED |
| X | Productionization | REST API, Docker, clinical eval/report | Phase 2 (API/Docker) | NOT STARTED |

*We are in Phase D (data expansion). Foundation is done.*

# 3. Current state snapshot (2026-06-10)

- **Run 15 SEALED (commit 032a2ab):** Test AUROC 0.9984, Val 0.9983, AUPRC 0.9935/0.9919, MCC 0.9655/0.9614, Brier 0.0069/0.0071. Cohort Train 1,038,974 / Val 146,329 / Test 304,711; 79 features. ~11.5 h on RTX 4090, ~$6.

- **Unseen-gene-holdout ablation: ENSEMBLE_STACKER AUROC 0.9988** on 213,436 rows / 2,407 gene-disjoint genes (C3 falsifier (b) PASS vs 0.95). Strong generalization to unseen genes; the corpus-scope leakage question on n_pathogenic_in_gene remains open (see §5).

- **ESM-2 mechanically active** (local UniProt index, no run-time REST, GPU auto-detect; commits 7b267ea/032a2ab) BUT coverage is only ~3,451/~1.49M (HGVSp-parser gap). esm2_delta_norm is ~99.7% zero in the full run; current AUROCs rest on tabular + constraint features.

- **gene_constraint_oe REVIVED** (Run-14 all-zero -> Run-15 #2 feature) via loeuf->oe derivation patch. **gnn_score REAL** (non-degeneracy gate PASS). cnn_1d (0.85) and kan (0.996) recovered from smoke underfit.

- **AlphaMissense OOM fixed** (325b0d2, re-validated): 71.7M rows parsed in ~2 min via cohort-filter-during-parse.

- Suite: 862 passed / 1 skipped (2026-06-10, post Phase-0/1 gene-resolution + ESM-2 LLR wiring; features 79 -> 80).

# 4. Data-source registry

## 4A. Wired & healthy

ClinVar (labels+cohort), gnomAD v4 (LOEUF, pLI, AF, **mis_z, syn_z, gene_constraint_oe**), AlphaMissense, SpliceAI, dbNSFP (SIFT/CADD/REVEL/GERP/PolyPhen), STRING-DB v12 (GNN -> gnn_score, confirmed real), n_pathogenic_in_gene, Reactome (reactome_pathway_count, wired 2026-06-08).

## 4B. Scaffolded but DEAD / partial (Phase D targets)

| **Source** | **Dead/partial column(s)** | **Access** | **Note** |
| --- | --- | --- | --- |
| ESM-2 | esm2_delta_norm (secondary), **esm2_llr** (primary, NEW) | local model+index | Phase 1 DONE 2026-06-10: esm2_llr LLR scorer (EsmForMaskedLM logits head; WT-marginal default, masked opt-in) + feature wired (79->80 lockstep; SIGNED, NOT clipped). CPU sign/index gate PASS; sign != class (continuous). Realizes after Run 16 coord-sync with esm2_model_name=esm2_t33_650M_UR50D. ESM C 600M = Phase 2 |
| PhyloP | phylop_score | free bigWig | conservation |
| GTEx | gtex_* (6) | free | eQTL/expression |
| 1000 Genomes | af_1kg_* (5) | free VCF | population AF |
| dbSNP/RefSNP | dbsnp_af | free | stub-mode step; activation = data + config |
| AlphaFold structure | alphafold_plddt, solvent_accessibility, secondary_structure_context, dist_to_active_site, has_uniprot_annotation | free (AlphaFold DB) | stub-mode step; activation = data + config |
| OMIM | omim_* (2) | free academic w/ reg. | disease/inheritance |
| ClinGen | clingen_validity_score | free API | **dtype drift: int vs float across prep/inference - fix before regen** |
| FinnGen | finngen_* (3) | free summary | population enrichment |
| MaxEntScan | maxentscan_score | free tool | splice strength |
| VEP | codon_position, exon_number | free tool | coding context |
| EVE | eve_score | free score files | needs score files + HGVSp coords |
| HGMD | hgmd_* (2) | PAID, blocked | label-leakage rules |
| LOVD | lovd_variant_class | free | tiny coverage |

*Note: gene_constraint_oe / gene_is_constrained moved OUT of 4B (constraint vestige resolved; oe now healthy and #2 feature).*

## 4C. New candidates - verdicts (unchanged from v1)

Strong fits: AlphaFold DB (DO), RefSNP/dbSNP (DO), COSMIC (DO, academic; feature NOT label), TCGA (OPTIONAL), Reactome (DONE), KEGG (OPTIONAL, overlaps Reactome). BioGRID overlaps STRING. dbGaP is an access prerequisite (blocked), not a connector. ProteomeXchange / SRA / ENA / DDBJ DRA / SILVA out of scope.

# 5. Immediate plan

- **ESM-2 coverage (RESOLVED 2026-06-10):** the ~3,451 cap was a stale AlphaMissense protein-coord index on the training box, not an HGVSp-parser gap (protein_pos/wt_aa/mut_aa are populated by step 10b; hgvsp_parser.py / protein_coords.py already exist). Coverage gate shipped (34e125a; local ceiling 96.6%); Run 16 prereq is an operational coord-index sync. Method/model migration to LLR + ESM-2 650M -> ESM C 600M now in progress (Phase 1).

- **n_pathogenic_in_gene computation-scope audit:** confirm train-only-per-fold vs corpus-wide; recompute train-only if corpus-wide, to close the leakage question the UGH 0.9988 result left open.

- **Fix clingen_validity_score dtype drift** (int in real_data_prep vs float in variant_ensemble) before the next regen.

- Remaining Phase-D connectors: activate dbSNP + AlphaFold-structure stub steps (data + config), then build COSMIC / TCGA / KEGG.

- One comprehensive GPU regen after the accessible public connectors are wired; measure-first probe; ALL-MODELS smoke before any billable retrain.

# 6. Modeling & infra roadmap

- Ensemble: RF, XGBoost, LightGBM, SVM (nystrom + bagged_rbf), LR, GBM, 1D-CNN, TabularNN + meta-learner; CatBoost; MC-Dropout; Deep Ensemble; KAN. Per-model comparison every run.

- GNN (Phase D/A): bf16 AMP, PyG SparseTensor/CSR, GraphGPS, Laplacian PE/RWSE, 3-channel STRING weights. GPU-only; 2-epoch probe first.

- CatBoost GPU-memory hardening (empty_cache between families / expandable_segments / order before torch models).

- Performance (P): Polars; Rust inference service. Advanced (A): Julia, VAE, Bayesian UQ. Productionization (X): REST API, Docker, clinical eval/report.

# 7. Standing disciplines

- Pre-flight gate; local mini-test before cloud; goal realignment each run.

- Measure-first (no estimates without a probe); ALL-MODELS smoke before training.

- Incremental checkpointing; irreversible/cloud cmds in separate re-paste blocks.

- Count-guarded, backup-first, idempotent, sandbox-validated patchers; byte-IO on Windows.

- **Background launch over SSH uses `< /dev/null`; read-only SSH checks use `-n -o ConnectTimeout=20 -o BatchMode=yes`; single-quoted SSH bodies; single-word grep patterns.**

- Document every run (algorithm comparison + metrics glossary); keep this roadmap current.

- Never drop models/features; scope ambiguity -> STOP + ask with options + pros/cons.

# 8. Blockers

- HGMD Professional - procurement; REVEL/VEST4/FATHMM/MutPred2 not labels if HGMD is a label source.

- dbGaP / TOPMed / controlled-TCGA / CPTAC-protected - need institutional Signing Official; blocked w/o R1 faculty sponsor.

- EVE - needs score files + HGVSp coords before eve_score is real.

# 9. Changelog

- 2026-06-10: ESM-2 coverage root-caused + gated; UniProt gap measured; protein-LM upgrade decided; roadmap consolidated.
  * ESM-2 ~3,451-of-2.49M coverage cap was a STALE AlphaMissense protein-coord index on the training box -- NOT the HGVSp parser. Local coord index covers 96.6% of missense. Step-10b fail-loud coverage gate shipped (commit 34e125a). Run 16 prereq is an operational coord-index sync, not parser work. SUPERSEDES the Section 4B "gated on HGVSp parser" line and the Section 5 "HGVSp parser (highest leverage)" item: protein_pos/wt_aa/mut_aa are already populated by step 10b from AlphaMissense, and hgvsp_parser.py / protein_coords.py already exist.
  * UniProt gene-symbol gap measured at 0.27% (6,742/2.49M missense; index healthy at 20,190 genes; MYH11/NDE1 PRESENT). "MYH11;NDE1" was a semicolon-joined multi-gene symbol, not a missing gene. Not a blocker; observability patch (aggregate unmatched-gene logging + safe ;-split) planned (Phase 0).
  * Protein-LM upgrade decision (research-backed): switch scoring METHOD embedding-delta -> log-likelihood-ratio (LLR, WT-marginal); switch MODEL esm2_t6_8M -> ESM-2 650M baseline (config-only via ESM2_MODEL_NAME; facebook/{name} mapping confirmed in _load_transformers_model) -> ESM C 600M (Cambrian Non-Commercial License; "Built with ESM" attribution). esm2_delta_norm demoted to SECONDARY; new esm2_llr primary feature (feature count +1, lockstep). ESM3-open / ESM C 6B reserved as future escalation. Cloud: RunPod added alongside Vast.ai (provider-agnostic layer; pin one provider during validation).
  * Roadmap consolidation: the pre-rebaseline repo-root ROADMAP.md archived verbatim into Appendix A and removed from repo root; *.bak_* gitignored; README live-link disambiguated. Single ground-truth living roadmap.
  * Phase 0 (commit fd5e293): shared gene_symbols.py resolution helper wired into esm2/eve/protein_pipeline; aggregate missing-gene logging; fixed a real eve case-drift bug; safe ;-join recovery. Suite 849 passed.
  * Phase 1: ESM-2 650M LLR scorer (annotate_llr; EsmForMaskedLM logits head; WT-marginal default, masked opt-in) + esm2_llr feature wired (TABULAR_FEATURES 79->80, both assembly sites, SIGNED/NOT clipped; INFERENCE_FEATURE_COLUMNS auto-derived). CPU sign/index gate PASS (TP53 hotspots negative; benign P72R less negative). CALIBRATION: LLR sign != class -> continuous feature (no hard cutoff). Harness reference slice populates esm2_llr (live, NOT allowlisted). Suite 862 passed / 1 skipped. Model default stays 8M; regen sets esm2_model_name=esm2_t33_650M_UR50D (visible in step-16b log).
- 2026-06-09 (v2): Run 15 sealed (Test 0.9984 / Val 0.9983 / UGH 0.9988). ESM-2 stall fixed + shipped (local index + GPU), but coverage ~3,451 -> HGVSp parser promoted to top Phase-D item. AlphaMissense OOM re-validated. gene_constraint_oe revived (#2 feature); gnn_score confirmed real; cnn_1d/kan recovered. Infra lessons (SSH stdin-detach, fast-fail flags, poll-bail bug) recorded. clingen dtype drift flagged.

- 2026-06-08 (v1 re-baseline): roadmap reconstructed; phase model proposed; data-source registry added (incl. out-of-scope determinations + dbGaP clarification); split-health audit + constraint vestige recorded; ESM-2 GPU-regen plan set.


---

## Appendix A -- Archived pre-rebaseline roadmap (Mar-May 2026, SUPERSEDED)

> ARCHIVED HISTORICAL SNAPSHOT. Retained verbatim for the project's running
> historical record. This is the repo-root ROADMAP.md that predates the
> 2026-06-08 v2 re-baseline. Its metrics (64 features, AUROC 0.9847), feature
> counts, repo slug, and phase labels reflect the Mar-May 2026 state and are
> NOT current -- the live roadmap is Sections 1-9 above. Frozen; do not edit.
# Genomic Variant Classifier -- Project Roadmap

**Author:** Monzia Moodie
**Repository:** `monzia-moodie/genomic-variant-classifier`
**Last updated:** March 2026 -- Phase 7/8 complete, Phase 4 in progress

---

## Vision

A production-grade, multi-modal genomic variant pathogenicity classifier that:

1. Achieves clinically actionable AUROC >= 0.90 on held-out ClinVar data
2. Provides calibrated uncertainty estimates for Variants of Uncertain Significance (VUS)
3. Integrates population-level WGS controls alongside disease cohort data
4. Serves predictions via a REST API with Docker deployment
5. Benchmarks multiple ML algorithm families to rigorously compare their
   effectiveness on large-scale genomic data

---

## Current State (March 2026)

| Item                                                                   | Status                    |
| ---------------------------------------------------------------------- | ------------------------- |
| 64-feature tabular ensemble (LightGBM, XGBoost, RF, GBM, LR)          | **Done**                  |
| Holdout AUROC (gene-stratified, 154K variants)                         | **0.9847**                |
| RNA splice pipeline (MaxEntScan; 4 features)                           | **Done**                  |
| Protein structure pipeline (AlphaFold/UniProt; 4 features)             | **Done**                  |
| FastAPI REST service (/predict, /batch, /health, /gene, /rsid, /info)  | **Done**                  |
| X-API-Key auth + slowapi rate limiting                                 | **Done**                  |
| Structured JSON logging + Prometheus /metrics                          | **Done**                  |
| Multi-stage Dockerfile + docker-compose (api / trainer / monitoring)   | **Done**                  |
| GitHub Actions CI (lockfile check, pytest, docker build)               | **Done**                  |
| Docker image pushed to GHCR                                            | **Done -- v2.0.0**        |
| Conformal prediction intervals (scripts/conformal_prediction.py)       | **Done**                  |
| External validation script (scripts/validate_external.py)             | **Done**                  |
| Calibration analysis (scripts/calibration_analysis.py)                | **Done**                  |
| METHODS.md (publication-ready methods section)                         | **Done**                  |
| dbSNP index parquet (2.87M ClinVar-matched rs-IDs)                     | **Done**                  |
| ESM-2 connector (src/genomic_variant_classifier/data/esm2.py)                                     | **Done -- ready for retrain** |
| MC Dropout / Deep Ensemble uncertainty (src/genomic_variant_classifier/models/mc_dropout.py)      | **Done**                  |
| KAN classifier (src/genomic_variant_classifier/models/kan.py)                                     | **Done**                  |
| Algorithm benchmark framework (src/genomic_variant_classifier/evaluation/benchmark.py)            | **Done**                  |
| Model retrain incorporating Phase 4 features                           | Pending data + compute    |

---

## Phase 4 -- Algorithm Expansion and Benchmarking

**Goal:** Rigorous comparison of ML families; add ESM-2 and uncertainty features.

### 4A -- ESM-2 Sequence Embeddings

- [x] `src/genomic_variant_classifier/data/esm2.py` connector -- HuggingFace transformers backend, SQLite cache
- [x] `esm2_delta_norm` added to `PHASE_4_FEATURES` (ready for next retrain)
- [ ] Install `transformers torch` in training environment and run annotation
- [ ] Retrain ensemble with 65-feature set; measure AUROC lift (+0.03-0.06 expected)

Expected AUROC lift: +0.03-0.06 on missense variants.
Install: `pip install transformers torch`

### 4B -- KAN (Kolmogorov-Arnold Network)

- [x] `src/genomic_variant_classifier/models/kan.py` -- pykan / efficient-kan backends; MLP fallback
- [x] sklearn-compatible interface; `plot_edge_functions()` for interpretability
- [ ] Run in benchmark framework; compare OOF AUROC against MLP

Install: `pip install pykan`

### 4C -- Bayesian Uncertainty Quantification

- [x] `src/genomic_variant_classifier/models/mc_dropout.py` -- MCDropoutWrapper + DeepEnsembleWrapper
- [x] Uncertainty decomposition: epistemic (variance) + aleatoric (entropy)
- [x] `annotate_uncertainty()` helper for DataFrame annotation
- [ ] Run DeepEnsembleWrapper(LightGBM, n_members=5) on holdout; measure ECE improvement
- [ ] Annotate VUS subset with uncertainty flags; export for clinical review

### 4D -- GNN over Protein-Protein Interaction Network

- [x] `src/genomic_variant_classifier/models/gnn.py` -- GAT convolutions over STRING DB graph
- [ ] Wire STRING DB edge weights into GNN training (currently uses uniform weights)
- [ ] Late fusion: concat GNN gene embedding with TABULAR_FEATURES before stacking

### 4E -- Algorithm Comparison Framework

- [x] `src/genomic_variant_classifier/evaluation/benchmark.py` -- cross-validated benchmark across all families
- [x] Metrics: AUROC, AUPRC, Brier, ECE, train time, inference latency, memory
- [ ] Run full benchmark on ClinVar holdout
- [ ] Produce comparison table for METHODS.md / manuscript

Run:

```bash
python -m genomic_variant_classifier.evaluation.benchmark \
    --parquet data/processed/clinvar_grch38.parquet \
    --output  outputs/benchmark \
    --n-folds 5
```

---

## Phase 3 -- Data Expansion

### 3A -- Population Controls

| Source                      | Data           | Status                                          |
| --------------------------- | -------------- | ----------------------------------------------- |
| 1000 Genomes Project (IGSR) | 2,504 WGS      | `data/external/1000genomes/` empty -- pending   |
| gnomAD v4.1 exomes          | ~730M variants | Done (filtered parquet)                         |

`population_1kg_af` added to `PHASE_4_FEATURES`; pending data download.

### 3B -- Disease Cohorts (Controlled Access)

Apply for these in parallel -- each takes 2-8 weeks for approval.

| Source                          | Data                                    | Application                      | Priority    |
| ------------------------------- | --------------------------------------- | -------------------------------- | ----------- |
| dbGaP / NCBI                    | TOPMed (300K WGS), CMG rare disease     | eRA Commons + institutional      | High        |
| EGA                             | European cancer + rare disease WGS      | Data Access Agreement            | High        |
| CMG (Centers for Mendelian Genomics) | High-quality rare disease trios    | Via dbGaP / AnVIL                | High        |
| UK Biobank                      | 470K WES + 200K WGS                     | Formal application               | Medium-high |
| All of Us                       | 250K+ WGS diverse ancestry              | Researcher Workbench (free)      | Medium      |

### 3C -- Pending Downloads

| File                    | Size     | Source                              | Status  |
| ----------------------- | -------- | ----------------------------------- | ------- |
| dbNSFP4.7a.zip          | ~30 GB   | Google Drive (registration required) | Pending |
| 1000G VCF chr*.vcf.gz   | ~100 GB  | IGSR portal (free)                  | Pending |
| SpliceAI scored VCF     | 27 GB    | On Drive                            | Done    |

---

## Phase 5 -- Clinical Validation and Deployment

- [ ] Prospective validation on gene panels (BRCA1/2, TP53, PTEN, ATM)
- [ ] Comparison against ClinVar star-rating on expert-reviewed variants
- [ ] Model card: training data, known limitations, ancestry coverage
- [ ] Manuscript draft: multi-modal genomic variant classifier with algorithm benchmarking

---

## Feature Roadmap

**Live (64 features -- current model):**
see `TABULAR_FEATURES` in `src/genomic_variant_classifier/models/variant_ensemble.py`

**PHASE_4_FEATURES (pending retrain):**

```text
esm2_delta_norm       -- ESM-2 embedding L2 distance (wt vs. mut); ~+0.03-0.06 AUROC (SECONDARY)
esm2_llr              -- ESM-2 650M log-likelihood-ratio (logit[mut]-logit[wt]); SIGNED, negative=damaging; CONTINUOUS (sign != class; benign TP53 P72R also negative ~-6.09); WT-marginal default / masked opt-in (PRIMARY)
population_1kg_af     -- 1000 Genomes allele frequency
uncertainty_epistemic -- Deep Ensemble epistemic uncertainty (inference-time)
uncertainty_aleatoric -- Deep Ensemble aleatoric uncertainty (inference-time)
```

---

## Complexity Reference

| Algorithm                | Training           | Genomic scale                  |
| ------------------------ | ------------------ | ------------------------------ |
| GBTs (XGBoost/LightGBM)  | O(I\*n\*d\*log n)  | Excellent                      |
| MLP / KAN                | O(n\*L\*H^2[\*G])  | Good                           |
| ESM-2 (frozen inference) | N/A                | Excellent                      |
| GNN (sparse PPI)         | O(G\*(V+E)\*d)     | Excellent -- independent of n  |
| Deep Ensemble (M members)| O(M \* base)       | Good                           |
| SVM (RBF)                | O(n^2)             | INFEASIBLE above ~100K samples |

SVM is excluded from all production runs (n > 100K).

---

*This roadmap is a living document. Update after each phase gate.*


## Backlog additions -- 2026-06-10 (agent layer)

- **Populate drift-agent reference baselines** (moves the 8 drift agents from
  awaiting_baseline -> active detection). Per agent, supply its inputs:
  - SchemaDriftMonitorAgent: expected schema baseline (expected_dtypes + hash) + current matrix.
  - ConceptDriftMonitorAgent: NannyML CBPE estimated AUROC + BBSE p-value.
  - LabelShiftMonitorAgent: reference confusion matrix + p_train + a prediction-log window.
  - CalibrationDriftMonitorAgent: labeled predictions with per-class posteriors + baseline ECE.
  - InfrastructureDriftMonitorAgent: pinned packages + expected DAG hash + golden set + replay.
  - FairnessSubgroupMonitorAgent: per-stratum p_train + predictions + axis columns.
  - AdversarialSubmissionMonitorAgent: weekly ClinVar submission feeds + baselines.
  - AnnotationPolicyMonitorAgent: SVI publication feed + ClinVar review-status deltas + submitter history.
  Note: data/reference/ recorded absent on 2026-05-07 -- create it first.
- **README registry-table precision** (optional): rename the 8 detector rows to the registered
  *MonitorAgent wrapper class names.
- **alibi-detect**: install only if a future detector imports it (none do today).
- Ref: docs/incidents/INCIDENT_2026-06-10_agent_layer_regression.md.
