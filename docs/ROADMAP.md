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

- **Unseen-gene-holdout ablation: ENSEMBLE_STACKER AUROC 0.9988** on 213,436 rows / 2,407 gene-disjoint genes (C3 falsifier (b) PASS vs 0.95). Strong generalization to unseen genes; the corpus-scope leakage question on n_pathogenic_in_gene RESOLVED 2026-06-13 (train-only at L1 689787f + L2 6b38985; lone-feature probe 0.7181 -> ~0.50) (see §5).

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
| 1000 Genomes | af_1kg_* (5) | free VCF | population AF -- WIRED 2026-06-13 (fill_population_af + build_1kg_parquet.py, a0ce407); activate via --kg <per-superpop AF parquet> |
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

- **n_pathogenic_in_gene computation-scope audit (RESOLVED 2026-06-13):** was corpus-wide; recomputed train-only at Level 1 (data prep, 689787f) and Level 2 (stacking OOF, 6b38985). Lone-feature probe 0.7181 -> ~0.50; leaky vs leak-free inner-OOF 0.7755 vs 0.6633. Run-17 Gate-A leakage decision CLOSED.

- **clingen_validity_score dtype drift (RESOLVED 2026-06-13):** both builders cast to float (variant_ensemble.engineer_features and real_data_prep._engineer_features, the latter with a "match inference builder" comment); verified aligned -- no regen blocker remains.

- Remaining Phase-D connectors: activate dbSNP + AlphaFold-structure stub steps (data + config), then build COSMIC / TCGA / KEGG.

- **Heterogeneous-KG modeling track (2026-06-13):** hetero-GNN ENGINE done (models/hetero_gnn.py, HeteroConv multi-relation gene graph, 54158f7) + KG edge-connectors done (data/kg_edges.py co-membership primitive + Reactome/KEGG/GO/OMIM/ClinGen adapters, 8c19f9b). NEXT: live hetero_gnn_score wiring -- a HeteroGNNScorer mirroring GNNScorer, plus the schema decision (hetero_gnn_score as a guarded 82nd feature, both builders in lockstep + EXPECTED_TABULAR_FEATURE_COUNT 81->82, vs enriching the gnn_score graph in place).
- **af_1kg_* WIRED (2026-06-13, a0ce407):** fill_population_af + build_1kg_parquet.py; activate at Run 17 via --kg <1000G per-superpopulation AF parquet>.

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

- 2026-06-13 (session 2, pandas-3.0 future-proofing & agent hardening): four commits; full suite 924 -> 956 passed / 6 skipped, zero new warnings.
  * Null-safe, pandas-version-stable variant/merge key construction (commit 36348fd): module-level _variant_key() maps missing alleles to a "" sentinel BEFORE astype(str), and the gnomAD merge keys now match the SpliceAI fillna("") pattern. Byte-identical keys validated under pandas 2.3.3 AND 3.0.2. DEFENSE-IN-DEPTH only: the clean-cohort guard (real_data_prep.py L378-390, INCIDENT_2026-05-31) already raises on null ref/alt, so the 3.0 key-collapse was already mitigated in production; this removes reliance on that guard. +4 tests.
  * VersionMonitorAgent extended to watch Python + ALL installed packages + the PyG companion ABI (commit 8ff2310): _check_python (running vs latest patch/series/EOL via endoflife.date), _check_dependencies (pip list --outdated; _is_major_bump flags major jumps e.g. pandas 2->3), _check_pyg_abi (import-tests torch_scatter/torch_sparse, catching the OSError / 0xc0000139 ABI break it previously missed). 5 -> 8 watch targets; all new state under literature_scout.*; no roster/coordinator change. +13 tests. First live run: pandas 2.3.3 -> 3.0.3 flagged [deps:major] (migration now self-monitoring); torch 2.11 -> 2.12 correctly NOT major; pyg_abi_alert empty (companions absent post-fix); 99 outdated / 19 major bumps.
  * dtype-family-aware schema gate (commit 37d60be): _dtype_family() collapses the string family (object/string/str/pyarrow variants) to one token and is IDENTITY for every numeric dtype; wired into hash_schema + detect + the pandera schema. object<->pandas-3.0 string no longer reads as drift, while float64<->int64 is still RED. The committed baseline (81x float64) hashes IDENTICALLY -> NO rebuild. Validated end-to-end under real pandas 2.3.3 + pandera 0.31.1. +5 tests. data/agent_state.json gitignored (runtime state).
  * FINDING -- neither named "pandas-3.0 blocker" is actually live: (a) the variant-key collapse is clean-cohort-guard-mitigated; (b) the schema baseline is 81/81 float64, so the object->string flip cannot occur on the current matrix. Both fixes are defense-in-depth, NOT live-bug fixes; this corrects the earlier "unmitigated blocker" framing. pandas stays PINNED at 2.3.3; the migration is future scope, now agent-monitored.
  * RESOLVES the two "Carried" items from the 2026-06-13 (variance mask) entry below: (a) test_ablate_gnn now PASSES locally (3 passed) after uninstalling the mismatched torch_scatter 2.1.2+pt25cu124 / torch_sparse wheels -> PyG falls back to native scatter (ENV-ONLY, no commit; the Vast.ai GPU box keeps its CUDA companions); now guarded by the VersionMonitorAgent pyg_abi watch so it cannot silently recur. (b) the .fillna downcasting FutureWarning was fixed in commit 4d56423 (@_suppress_fillna_downcast on both feature builders; suite warnings 220 -> 41).
  * Run 17 pre-flight gate authored (commit 94bf6ae; docs/runs/RUN17_SCOPE.md): activates gnn_score (--string-db auto) + af_1kg_* (--kg <1000G Phase-3 AF parquet>) via scripts/run_phase2_eval.py (NOT train.py). Corrected flag discrepancies: the real flag is --kg (a parquet), not --kg-path (a VCF). VALUE activation only -> 81-col schema + hash unchanged; no baseline rebuild.
  * FINDING -- schema baseline provenance + DEFAULT_MATRIX footgun: the committed schema_baseline.json was captured from models/smoke_run16b/splits/X_train.parquet (run16b-smoke, 81 cols) and is GREEN against that matrix; it is RED against outputs/run15_rerun_report/full/splits/X_train.parquet, which is STALE (78 cols; predates esm2_llr / maxentscan_delta / reactome_pathway_count). build_schema_baseline.py DEFAULT_MATRIX still points at the stale run15 path, so a no-arg rebuild would regress 81 -> 78. Fixed in a follow-up (--matrix required + column-count regression guard).
  * RESOLVED <DECISION> (2026-06-13): n_pathogenic_in_gene is now train-only at Level 1 (689787f) and Level 2 (6b38985); the Run-17 Gate-A leakage decision is CLOSED. Lone-feature probe 0.7181 -> ~0.50; leaky vs leak-free inner-OOF 0.7755 vs 0.6633.
- 2026-06-13: neural variance mask landed (commit 5de7806); 81->51 schema trim attempted then REVERTED.
  * Run 16 census: 37/81 matrix columns constant on the gene-disjoint test split (29 unpopulated/stub/blocked, codon_position == protein_pos, plus deferred gnn_score / af_1kg_* / sparse-real is_mitochondrial / lovd_variant_class).
  * Attempted to relocate 30 columns TABULAR_FEATURES -> PHASE_2_FEATURES (81->51), unifying both feature builders on a fail-loud select. Unit-green (contract + test_api at 51/80) but the full suite raised 40 failures across 10 files. Those are the deliberate Phase-4 contract (fully-promoted schema + connector->matrix wiring + safe defaults; the *_in_tabular_features / *_flows_into_feature_matrix / phase_2_is_empty tests are silent-failure guards). REVERTED; schema stays 81-col.
  * Learned: a constant column is a data-availability state, not dead code. The real risk (constant neural inputs) is handled in the MODEL layer: TabularNNClassifier learns a fit-time variance mask (var>0) applied at predict, inherited by mc_dropout/deep_ensemble; cnn_1d/trees/LR/CatBoost/KAN untouched. No schema/contract/inference/schema-baseline change; backward-compatible with pre-mask pickles; no OOF leakage (per-fold recompute). Suite 924 passed / 6 skipped, zero new warnings. See docs/design/neural_variance_mask.md and docs/sessions/SESSION_2026-06-13_feature-trim-and-variance-mask.md.
  * Carried: test_ablate_gnn skips locally on torch_scatter/torch_sparse 0xc0000139 (GNN coverage absent on the Windows box) -- confirm runnable before Run 17 activates gnn_score; pandas .fillna downcasting FutureWarning in variant_ensemble.py wants an explicit cast.
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


## Backlog additions -- 2026-06-11 (drift wiring + schema gate)

**Delivered this session (first delivery against the 2026-06-10 "Populate drift-agent reference
baselines" item):** SchemaDriftMonitorAgent now has a versioned baseline
(`data/reference/schema/schema_baseline.json`, 78 cols / all float64), a `from_baseline` loader,
and a standalone preflight gate (`scripts/run_schema_drift_check.py`, exit 0/2/3). The schema agent
is the worked example; the other seven drift agents still await their reference inputs.

### Drift-wiring findings (recorded 2026-06-11)
- The eight agent-layer drift MonitorAgents are registered in `Orchestrator._register_agents` but
  invoked by nothing (absent from `PIPELINE_DEFINITIONS`; `run_agents.py --pipeline full` runs only
  the four framework agents).
- `.github/workflows/drift_monitor.yml` is effectively inert: GDrive download is a stub, so the job
  skips via "No reference splits available"; it also points at the stale
  `outputs/phase2_with_gnomad/splits/` path (pre-Run-15).
- `scripts/run_drift_monitor.py` covers distributional (PSI/KS/MMD) + label drift but NOT
  schema/column/dtype drift. The new gate is additive.

### Proposed action items (NOT done -- deliberate design decisions for a future session)
- [ ] **Pipeline-wire the drift agents.** Add a `drift` key to `PIPELINE_DEFINITIONS` listing the
  eight MonitorAgents, reachable via `run_agents.py --pipeline drift`. They are registered but
  currently unreachable from any CLI path.
- [ ] **Fix `drift_monitor.yml`.** Repoint the stale `outputs/phase2_with_gnomad/splits/` ->
  `outputs/run15_rerun_report/full/splits/`, and replace the GDrive-download stub (a no-op that
  makes the monthly job skip) with a real fetch or an honest hard-skip that is logged, not silent.
- [ ] **Add the schema gate as a `drift_monitor.yml` step.** Run `scripts/run_schema_drift_check.py`
  (exit-2 gates the job) so schema drift -- which `run_drift_monitor.py` does not cover -- is checked
  monthly alongside PSI/label drift.
- [ ] **Reconcile the two parallel drift systems.** The agent-layer drift agents and the
  `src/monitoring/` + `run_drift_monitor.py` system overlap conceptually; consolidate into one
  documented entrypoint so "drift monitoring" has a single source of truth.

### Feature-count reconciliation (TO VERIFY -- not asserted)
- [ ] Reconcile the **64 / 78 / 79** spread: notes say 79; on-disk `X_train` is 78 (verified green
  by the gate); this ROADMAP still says "Live (64 features)". Identifier/label/target columns live in
  `meta_*`, separate from the 78 `X_*` features. Settle the canonical count.
- [ ] Verify whether `af_1kg_afr/amr/eas/eur/sas` (present in the 78-col `X_train` per the gate diff)
  are populated or placeholder-zero, and reconcile against `population_1kg_af` being listed under
  PHASE_4_FEATURES (pending).

- Ref: docs/sessions/SESSION_2026-06-11_ci-and-schema-gate.md;
  docs/incidents/INCIDENT_2026-06-11_ci-optional-deps.md.

<!-- docs-close: ecd0474 esm2-llr+train-wiring -->
## ROADMAP delta -- 2026-06-11 (PM)

### Done
- [x] ESM-2 LLR long-protein OOM fixed via windowing (1db43f1)
- [x] ESM-2 650M activation validated on real data (CPU probe GREEN)
- [x] DECISION resolved: ESM-2 650M for Run 16 (ESM C 600M = later A/B)
- [x] train.py ESM-2 wiring: model + offline index + cache + device flags (ecd0474)
- [x] metrics annotation_sources provenance extended (esm2/finngen/dbnsfp)

### Run 16 launch contract (preflight MUST Test-Path each)
- --esm2-model esm2_t33_650M_UR50D
- --esm2-uniprot-index data\external\uniprot\uniprot_human_reviewed.parquet
- --alphamissense <AlphaMissense scores path>

### Open -- blocking Run 16
- [ ] CNN train-sequence NotImplementedError: verify cohort fasta_seq density; plumb
      meta_train (Option-B-wide) or keep CNN on placeholder seqs (INCIDENT_2026-05-30).
- [ ] ONE preflight script (3 mandatory paths + instance/SSH/key vars, validated).
- [ ] Standing run gates: all-models smoke, full suite green, zero known bugs.

### Open -- post-regen / parallel
- [ ] Schema baseline refresh 78 -> 79 (esm2_llr newly live).
- [ ] EVE activation (EVE_scores_ASM acquisition) -- separate data track.
- [ ] Doc drift: AnnotationConfig docstring 17 vs code 18 steps (Reactome already runs);
      reconcile per-step log labels (15/16, 16/17, 17/17, 18/18).
- [ ] Hygiene: non-ASCII em-dash in real_data_prep.py esm2_delta_norm comment.

## ROADMAP delta -- 2026-06-11 (late PM, CNN + RNA activation)

### Done
- [x] 1D-CNN activated on real [fasta_seq_ref, fasta_seq_alt] delta windows; train-side via the
  persisted meta_train.parquet (gene-split-aligned to X_train); NotImplementedError removed
  (fb12c0f). SUPERSEDES the "Open -- blocking Run 16" item "CNN train-sequence
  NotImplementedError ... (INCIDENT_2026-05-30)".
- [x] RNA MaxEntScan activated: maxentscan_delta = score(alt) - score(ref), a NEW
  variant-specific splice-disruption feature. The MaxEntScan source moves from Section 4B
  (Scaffolded but DEAD/partial) to LIVE. maxentscan_score keeps its meaning (ref-window score).
  EXPECTED_TABULAR_FEATURE_COUNT 80 -> 81 (e3bcd79).

### Run 16 launch contract -- ADDITION (preflight MUST Test-Path)
- --clinvar data\processed\clinvar_grch38_clean_seq.parquet  (the ref/alt cohort). Without it
  BOTH the CNN and maxentscan_delta degrade to inert: _load_and_label preserves input columns,
  but real_data_prep never adds fasta_seq* itself, so the ref/alt windows exist on the frame
  ONLY if the input cohort carries them. Joins the existing --esm2-model esm2_t33_650M_UR50D,
  --esm2-uniprot-index, and --alphamissense requirements.

### Open -- post-regen / parallel (updated)
- [ ] Schema baseline refresh: regenerate data/reference/schema/schema_baseline.json from the
  post-Run-16 X_train. Target = EXPECTED_TABULAR_FEATURE_COUNT (now 81: +esm2_llr +maxentscan_delta
  vs the sealed-78 baseline). SUPERSEDES the earlier "Schema baseline refresh 78 -> 79" line. The
  pre-existing 78/79/80 spread (Feature-count reconciliation, TO VERIFY) reconciles AT this regen
  by diffing actual X_train columns against TABULAR_FEATURES -- not asserted here.
- [ ] Always-donor MaxEntScan selection bug: donor/acceptor choice is bounds-based (always donor
  for a 101bp window), so maxentscan_delta measures a donor perturbation even for acceptor-region
  variants. Biology-correct selection (drive from dist_to_donor/dist_to_acceptor) is the next RNA
  item. Does NOT block Run 16.

### Standing discipline -- ADDITION
- Every new tabular feature must appear, POPULATED (non-zero / non-degenerate), in the
  correctness-harness reference slice (build_reference_slice). This session the harness stage-5
  silent-zero tripwire was the ONLY gate that caught a feature added without its slice entry.

## ROADMAP delta -- 2026-06-12 (CI ESM-2 Hub flake resolved)

### Done
- [x] CI restored to green. test_llr_long_protein_scores_finite_without_oom was loading
  the real ESM-2 8M from HF Hub; CI (no cache, 429) flaked red while local (cached) passed.
  fee2e63 skip-guards the live load; ci.yml now forces HF offline and uses --maxfail=5.
  See docs/incidents/INCIDENT_2026-06-12_ci-esm2-hub-flake.md.

### Standing disciplines -- ADDITIONS
- Offline-suite gate: before trusting "suite green", run tests/unit under an empty offline
  HF cache (HF_HOME=<empty>, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1). Anything that ERRORS
  (vs passes/skips) is a network-coupled test to guard. Local-green != CI-green for any test
  that loads an ESM-2 model (the local cache hides the dependency).
- CI surfaces failures broadly: pytest -x replaced by --maxfail=5 so a break is not reported
  as a single isolated failure with the rest of the suite hidden.

### Note (unchanged, benign)
- 0xc0000139 torch_scatter/torch_sparse dumps during collection are the known-benign
  importorskip path (missing PyG C-extensions on the CPU box); suite exit stays 0.

<!-- roadmap-delta: protein-coord-rebuild 2026-06-12 -->
## ROADMAP delta -- 2026-06-12 (protein-coord index repair + Run-16 input contract)
- Protein-coord index rebuilt full-cohort: 18.64 MB, 0.9665 coverage (ESM-2 ready).
- Run-16 launch contract (mandatory flags): `--clinvar` clean_seq cohort (ref/alt);
  `--esm2-model esm2_t33_650M_UR50D`; `--esm2-uniprot-index uniprot_human_reviewed.parquet`;
  `--alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz` (TSV, NOT the
  scores parquet); `--gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv`.
- Data staging: ship the 18.64 MB protein-coord index to Vast.ai co-located with the
  `--alphamissense` dir so the regen loads it (no 613 MB TSV re-scan).
- Open: decouple the protein-coord source from `--alphamissense` (elegant fix removing
  the scores-parquet-vs-TSV trap); fold a coverage/size check into the preflight.

<!-- roadmap-delta: run16-smoke-gate 2026-06-12 -->
## ROADMAP delta -- 2026-06-12 (Run-16 smoke gate cleared)
- Run-16 all-models smoke: COMPLETE end-to-end (724s); ENSEMBLE_STACKER test AUROC 0.9934.
  The `--fast` smoke is the authoritative gate; the input preflight (now 6 checks) is a fast
  necessary pre-screen only.
- Run-16 launch contract (mandatory data-feeding flags): `--clinvar` clinvar_grch38_clean_seq
  (ref/alt + ReviewStatus); `--esm2-model esm2_t33_650M_UR50D`; `--esm2-uniprot-index
  uniprot_human_reviewed.parquet`; `--alphamissense AlphaMissense_hg38.tsv.gz` (TSV, NOT the
  scores parquet -- the parquet cache OOMs at 16 GiB); `--gnomad-constraint
  gnomad.v4.1.constraint_metrics.tsv`.
- Data staging: ship the TSV (not the 740 MB scores cache) + the 18.64 MB protein-coord index
  co-located under data/external/alphamissense/.
- Open decisions: production optional flags (--gnomad allele-freq, --uniprot, --dbnsfp-path,
  --lovd-path, --finngen-path) + paths; optional AlphaMissense cohort-filtered cache read;
  optional decouple of protein-coord source from --alphamissense.
- Watch (full regen): cnn_1d (~0.5) and kan (~0.74) at scale; gene-disjoint splits + cross-fold
  count leakage on gene-level features; AlphaFold-structure stub (pLDDT=50.0 default).

## 2026-06-12 -- Run 16 launch-ready; Run 17 scope committed

Run 16 (tabular): VALIDATED + launch-ready. Flag set frozen
(docs/launch/LAUNCH_CONTRACT_run16.md). Schema sealed at 81 (run16b-smoke). gnn_score,
af_1kg_*, and uniprot features dormant-by-design (sealed, will activate later).

Run 17 (COMMITTED, not deferred -- docs/roadmap/RUN17_SCOPE.md):
- Track A: 1000 Genomes AF -- build_1kg_parquet.py + --kg <per-superpop AF parquet> + validate AF-fill (fill_population_af, a0ce407);
  af_1kg_* per-population columns WIRED 2026-06-13 (a0ce407); activation = per-superpopulation AF parquet via --kg.
- Track B: STRING-DB GNN -- gnn_score live + LEAKAGE-FREE via gene-disjoint cross-fitting,
  held-out-gene no-leak check, WITH/WITHOUT ablation.
Both gated by full-scale feature-population audit + schema drift-check + gene-disjoint
integrity verification before Run 17 trains.
