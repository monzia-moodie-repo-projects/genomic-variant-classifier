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

- **ESM-2 mechanically active** (local UniProt index, no run-time REST, GPU auto-detect; commits 7b267ea/032a2ab). **The ~3,451 coverage cap was RESOLVED 2026-06-10 and this line previously misattributed it to an "HGVSp-parser gap" -- see 4B/section-5 note and the 2026-06-10 entry below, which SUPERSEDE that framing.** Root cause was a stale AlphaMissense protein-coord index on the training box; `hgvsp_parser.py` and `protein_coords.py` both exist, are wired (`real_data_prep.py:995`, `protein_coords.py:35`) and are tested. Local coord index covers 96.6% of missense; coverage gate shipped (34e125a). The `esm2_delta_norm ~99.7% zero` figure is a **RUN-15 measurement** and is retained as history; it predates the coord-index sync and is NOT a current statement. Re-measuring esm2 coverage on the 4,399,089-row cohort is an OPEN item.

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
| 1000 Genomes | af_1kg_* (5) | free VCF | population AF -- ACTIVE 2026-06-15: kg_grch38_af.parquet built (chr1-22 + X, 437,668 variants = ~9.9% cohort; 5 super-pops non-zero); activate via --kg. chrY/MT structurally absent from the 1000G high-coverage panel (404-confirmed) -> 3,191 Y + 3,124 MT cohort variants get af_1kg=0; gnomAD Y/MT allele_freq RESOLVED 2026-06-16 (PAR X->Y fix): Y 1047/3155, MT 2731/3124 |
| dbSNP/RefSNP | dbsnp_af | free | DONE+VERIFIED 2026-06-26 (build_dbsnp_parquet.py; dbsnp157_cohort.parquet 3.75M rows, 46% AF>0). End-to-end audit 2026-07-01: 37.45% cohort coverage, dbsnp_af>0 confirmed through DbSNPConnector. Wired: --dbsnp-path -> AnnotationConfig -> real_data_prep step 10. |
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

- Remaining Phase-D connectors: AlphaFold-structure stub steps (data + config), then build COSMIC / TCGA / KEGG. (dbSNP DONE+VERIFIED 2026-06-26; see the dbSNP row above and the 2026-07-01 end-to-end coverage audit.)

- **Heterogeneous-KG modeling track (2026-06-13):** hetero-GNN ENGINE done (models/hetero_gnn.py, HeteroConv multi-relation gene graph, 54158f7) + KG edge-connectors done (data/kg_edges.py co-membership primitive + Reactome/KEGG/GO/OMIM/ClinGen adapters, 8c19f9b). SCORER + SCHEMA DONE 2026-06-13: HeteroGNNScorer (547e2dc, mirrors GNNScorer; torch-free assembly + PyG train/score) and hetero_gnn_score landed as the 82nd feature (Option A -- SEPARATE from gnn_score to preserve the homogeneous-vs-heterogeneous comparison; EXPECTED_TABULAR_FEATURE_COUNT 81->82, both builders lockstep, reactome stays last; contract green, suite 1000). EVAL-OVERWRITE DONE 2026-06-14 (a54ef38): run_phase2_eval --hetero-gnn + --kg-edges source:path builds HeteroGNNScorer from STRING interacts_with + KG relations and overwrites hetero_gnn_score per split (opt-in; until run with the flag it stays the 0.5 default, mirroring gnn_score). ONLY REMAINING (Run-17): schema_baseline regen 81->82 from the real matrix.
- **af_1kg_* ACTIVE (2026-06-15):** kg_grch38_af.parquet built chr1-22 + X (437,668 variants, 6.7 MB) via build_1kg_parquet.py per-chr shards + merge; activate via --kg. chrY/MT not in the 1000G high-coverage panel (404-confirmed -> af_1kg=0 for 3,191 Y + 3,124 MT cohort variants; gnomAD Y/MT allele_freq RESOLVED 2026-06-16). WIRED 2026-06-13 (a0ce407).
- **gnomAD Y/MT allele_freq ACTIVE (2026-06-16):** PAR canonicalisation fix -- gnomAD reports PAR variants on X, ClinVar on Y; `y_key()` remaps PAR1 X->Y identical + PAR2 X-98,813,480 + MSY pass-through. `build_gnomad_ymt_af.py` merges Y/MT frequencies into `gnomad_v4_exomes.parquet`: Y 1047/3155 (33% = honest gnomAD ceiling; remainder = gnomAD-uncalled Y positions + 264 na:na structural), MT 2731/3124 (87%). Commit 112967d.
- **LiteratureScout broadened (2026-06-13, a42e723 + a9c0326):** provenance (authors/publication_date/journal across PubMed/bioRxiv/ClinGen/Zenodo) + new Zenodo source (_fetch_zenodo) + PubMed queries 11->19 / keywords 32->46 into architecture/methodology gaps + journal allow-list relevance boost. +8 tests.

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

**Delivered 2026-06-14 (6a05481) -- schema agent ACTIVATED + generic enabler:**
`SchemaDriftMonitorAgent.from_default_baseline` loads the `SchemaDriftAgent` detector from the canonical
baseline (`data/reference/schema/schema_baseline.json`; same path as `run_schema_drift_check.py` +
`build_schema_baseline.py`), and `Orchestrator` now prefers `from_default_baseline(state)` when the class
defines it (else `cls(state)`). The schema agent runs active detection once a matrix is supplied
(arg -> `GVC_SCHEMA_CURRENT_MATRIX` env); it is no longer awaiting its *baseline*, only its run-time
*matrix*. The orchestrator hook is the single generic enabler the other seven reuse -- each just defines
`from_default_baseline`. **Buildability split for the seven:** BUILDABLE NOW (no trained model) =
LabelShift (reference label distribution from cohort labels), Infrastructure (pinned packages + DAG hash),
likely AnnotationPolicy + AdversarialSubmission (config/heuristic references); RUN-17-DEPENDENT (need
predictions) = Concept (NannyML CBPE AUROC + BBSE), Calibration (per-class posteriors + ECE),
FairnessSubgroup (per-subgroup predictions). Verified end-to-end: `test_schema_drift_baseline.py`
(5 tests -- green/red/awaiting/env/bare), suite 1009.

**Delivered 2026-06-14 (continued) -- drift-baseline campaign + FeatureCoverageSentinel (c0dec47, a7abca0,
469a7b6, e4f96df, e61a2dc, 8206673, 376aa2e):** the buildability split above is resolved. ACTIVE NOW
(model-free / config-threshold): Infrastructure (a7abca0), AnnotationPolicy + AdversarialSubmission
(469a7b6), plus Schema. MACHINERY-READY (real baseline at Run-17): LabelShift (c0dec47). A NINTH drift
agent, FeatureCoverageSentinel, was built/tested/wired/activated against the split-health audit reference
(54/42/96): feature_health shared module (e4f96df), detector (e61a2dc), builder + monitor (8206673),
orchestrator wiring + the `drift` pipeline (376aa2e). REMAINING (RUN-17-DEPENDENT, need model predictions):
Concept (NannyML CBPE + BBSE), Calibration (per-class posteriors + ECE), FairnessSubgroup (per-subgroup
predictions) -- machinery next. Suite 1009 -> 1063.

**Delivered 2026-06-14 (trio) -- Concept + Calibration + FairnessSubgroup activated (19fb2a0):** the three
RUN-17-DEPENDENT detectors gained detector from_baseline + monitor from_default_baseline (same pattern as
Schema/LabelShift/FeatureCoverage; orchestrator hook 6a05481 routes them). Builders: build_concept_baseline.py
(thin writer for the two CBPE scalars), build_calibration_baseline.py (computes baseline_ece via the detector's
OWN detect() -> single code path), build_fairness_baseline.py (p_train_per_stratum from reference predictions +
axes). 20 tests; suite 1063 -> 1083. DRIFT SET NOW 8 of 8 WIRED + FeatureCoverageSentinel (9th): 5 ACTIVE,
4 machinery-complete (LabelShift + the trio) awaiting Run-17 model artifacts. FairnessSubgroup PHASE_2 stubs
unchanged + documented (per-stratum AUROC proxy; max_dpd_change=0.0 pinned by test_dpd_stub_is_zero).

- **Test-gate finding (2026-06-14): flaky pre-existing warning inflates the count.** Two back-to-back
  `pytest -q` runs gave 41 then 141 warnings. The +100 is a benign, FLAKY sklearn UserWarning
  (`sklearn.utils.parallel.delayed` ... parallel.py:144) from `test_correctness_harness.py`:
  `run_correctness_harness` builds `EnsembleConfig(skip_svm=...)` with the default `n_jobs=-1`, so the Stage-1
  smoke fits the tiny slice via loky Parallel; loky emits the warning per worker dispatch (0..~100). NOT a
  regression; no pass/fail impact. The DETERMINISTIC baseline stays 41. **Proposed fix:** pass `n_jobs=1` into
  the harness smoke's `EnsembleConfig` (tiny-slice parallelism is pointless) -> deterministic + faster + no
  loky warning. Low-risk, single-line. **FIXED 2026-06-14 (fe2289d)** -- both back-to-back `pytest -q` runs now stable at 1083 passed / 6 skipped / 41 warnings (the +100 parallel.delayed block gone); the 5 test_correctness_harness stages still pass.

### Drift-wiring findings (recorded 2026-06-11)
- The eight agent-layer drift MonitorAgents are registered in `Orchestrator._register_agents` but
  invoked by nothing (absent from `PIPELINE_DEFINITIONS`; `run_agents.py --pipeline full` runs only
  the four framework agents). **[RESOLVED 2026-06-14 (376aa2e):
  `PIPELINE_DEFINITIONS["drift"]` now lists all 9 drift agents; reachable via --pipeline drift.]**
- `.github/workflows/drift_monitor.yml` is effectively inert: GDrive download is a stub, so the job
  skips via "No reference splits available"; it also points at the stale
  `outputs/phase2_with_gnomad/splits/` path (pre-Run-15).
- `scripts/run_drift_monitor.py` covers distributional (PSI/KS/MMD) + label drift but NOT
  schema/column/dtype drift. The new gate is additive.

### Proposed action items (NOT done -- deliberate design decisions for a future session)
- [x] **Pipeline-wire the drift agents -- DONE 2026-06-14 (376aa2e).** `PIPELINE_DEFINITIONS["drift"]`
  now lists all NINE drift MonitorAgents (the eight + FeatureCoverageSentinelMonitorAgent), reachable via
  `run_agents.py --pipeline drift` (run_agents builds --pipeline choices from PIPELINE_DEFINITIONS.keys()).
  Verified live: the dry-run drift pipeline runs all 9 agents.
- [x] **Fix `drift_monitor.yml` -- DONE 2026-06-14 (partial; path + honest GDrive skip).** Stale `outputs/phase2_with_gnomad/splits/` -> `outputs/run15_rerun_report/full/splits/` (6 occurrences); GDrive stub no longer fabricates a 'credentials loaded' message (the skip is logged). REMAINING: a real rclone/gdown fetch (still a placeholder). Original note: Repoint the stale `outputs/phase2_with_gnomad/splits/` ->
  `outputs/run15_rerun_report/full/splits/`, and replace the GDrive-download stub (a no-op that
  makes the monthly job skip) with a real fetch or an honest hard-skip that is logged, not silent.
- [x] **Add the schema gate as a `drift_monitor.yml` step -- DONE 2026-06-14.** A GUARDED step runs `scripts/run_schema_drift_check.py --matrix .../X_train.parquet` (exit 0/2/3), skipping honestly when baseline/matrix are absent. REMAINING: tighten to gate-the-job on exit-2 (currently continue-on-error, matching the job's notify-not-fail design). Original note: Run `scripts/run_schema_drift_check.py`
  (exit-2 gates the job) so schema drift -- which `run_drift_monitor.py` does not cover -- is checked
  monthly alongside PSI/label drift.
- [ ] **Reconcile the two parallel drift systems.** The agent-layer drift agents and the
  `src/monitoring/` + `run_drift_monitor.py` system overlap conceptually; consolidate into one
  documented entrypoint so "drift monitoring" has a single source of truth.

- **Delivered 2026-06-14 -- ReclassificationSentinel (10th drift agent).** clinvar_tracker-backed label-drift
  sentinel (b6e5958 detector / 9662569 monitor + reference builder / 0c6c049 wiring); 17 tests. DRIFT SET NOW
  10 of 10 WIRED. Run-17-gated: the (variant_id, split) reference (build_reclassification_reference.py against
  the real splits) + the OLD/NEW ClinVar release parquets.
- **Reconcile finding (a) -- pandera effectively required.** The agent-layer drift pipeline cannot construct
  SchemaDriftMonitorAgent.from_default_baseline without pandera (from_baseline imports it), despite the
  "optional dep" docstring. No functional gap (the monthly job uses run_drift_monitor.py), but graceful
  from_baseline/detect degradation is a decision for this reconcile work. CI tests guard it with
  importorskip("pandera") (5a6b0d0); see INCIDENT_2026-06-11 + its 2026-06-14 recurrence.
- **Reconcile finding (b) -- legacy meta_TEST mislabel.** run_drift_monitor.run_label_drift reads
  meta_TEST.parquet and assigns those ids to training_variant_ids, so its flip_rate_training is the TEST-set
  rate. The new ReclassificationSentinel does per-split extraction correctly; consolidate the two.
- **Infra note (2026-06-14).** The repo's data/ was a Windows Junction -> G:\My Drive\...\data (Google Drive
  for Desktop) and dangled when G: was unmounted, failing 20 tests via the fail-loud guard. Restored via
  git checkout -- data/ -> data/ is now a PLAIN LOCAL dir (see INCIDENT_2026-06-14_data-junction-dangling).
  Large untracked assets remain on G: -- re-hydrate before any real run. Recommend local data//outputs/ +
  rclone genvarcla: for durability, NOT a live G: junction. Check outputs/ for the same dangling condition.

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
  post-Run-16 X_train. Target = EXPECTED_TABULAR_FEATURE_COUNT (now 82: +esm2_llr +maxentscan_delta +hetero_gnn_score
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

## 2026-06-14 (continued) -- data-source registry, freshness monitor, dead-agent audit, proposed-agent roadmap

### Delivered (wired + healthy)
- monitoring/registry.py: single source of truth, 24 sources (11 ACTIVE, 7 probeable). FinnGen R12->R14 (embargo).
- DatabaseFreshnessMonitorAgent: registry-driven HITL freshness over ALL DBs; 'database_monitor' pipeline +
  'full' + weekly workflow + documented FRESHNESS report. Supersedes DataFreshnessAgent's 4-source polling.
- Dead-agent audit closed: both gcloud-dataproc dead paths neutralized; InterpretabilityAgent tested; dead dep
  dropped. 'all' pipeline + cadence semantics documented. No dead agents.

### Proposed agents (Phase D/E -- assessed 2026-06-14, NOT yet built; awaiting go-ahead) -- priority order:
1. ModelInsightsAgent (HIGH value, LOW risk, read-only): post-run, ingest each base model's OOF + metrics
   (AUROC/AUPRC/MCC/Brier/calibration) + SHAP + the per-model algorithm comparison; produce a DOCUMENTED per-model
   comparison report, alert significant findings, FLAG leakage/over-fit (e.g. suspiciously high AUROC ->
   n_pathogenic_in_gene-style memorization). GUARDRAIL: diagnostics + integrity flags only; does NOT auto-tune
   toward higher AUROC (scientific-integrity-over-metrics). Reuses evaluator.py + shap_utils + metrics glossary.
2. DataReadinessAgent (HIGH value, MED risk): pre-run, orchestrate the existing audits (feature-health, smoke
   population, schema-drift, FeatureCoverageSentinel) + freshness monitor + critical_assets() into ONE documented
   pre-run readiness gate; HITL-block a run if data is stale/degenerate/silent-stubbed. INVOKES + VERIFIES +
   DOCUMENTS real_data_prep.py (no silent data mutation).
3. GpuOrchestratorAgent / FinOps (HIGH value, HIGH risk): cross-platform (Vast.ai + RunPod) preflight + optimal
   instance selection ($/hr x reliability x dlperf x mem) + run init + auto-terminate + billing report. RunPod is
   NET-NEW (zero code today). COST-SAFETY: HITL-approve before provisioning (real money); budget caps;
   confirm-on-terminate; never auto-spend. Build LAST + most carefully.
4. AgentOpsMonitor (meta): ONE FLAT heartbeat/last-run/error-rate monitor over agent_state.json (alerts on
   staleness/error/conflict/perf-drift); monitors itself too. NOT recursive (no agent-of-agent tower).


## Proposed-agent roadmap -- update 2026-06-14
- [DONE] ModelInsightsAgent (was: proposed #1, HIGH value / LOW risk). Shipped read-only per-model comparison +
  integrity monitor over run OOF. Next proposed agents remain: DataReadinessAgent (#2, pre-run gate orchestrating
  existing audits), GpuOrchestratorAgent/FinOps (#3, cross-platform Vast.ai+RunPod, cost-safety HITL, build LAST),
  AgentOpsMonitor (#4, flat heartbeat/error-rate meta-monitor over agent_state.json).


## Proposed-agent roadmap -- update 2026-06-14
- [DONE] DataReadinessAgent (was: proposed #2). Shipped verify-only pre-run readiness gate (assets + feature
  health -> GO/NO_GO, HITL override). Remaining: AgentOpsMonitor (#4, flat heartbeat/error-rate meta-monitor over
  agent_state.json -- LOWER risk, recommended next) and GpuOrchestratorAgent/FinOps (#3, cross-platform
  Vast.ai+RunPod, cost-safety HITL -- HIGHEST risk, build LAST). Optional follow-up for DataReadinessAgent:
  active-invocation mode (shell out to smoke_all_models / preflight_gate with HITL) if desired.


## Proposed-agent roadmap -- update 2026-06-14
- [DONE] AgentOpsMonitorAgent (was: proposed #4). Shipped flat heartbeat/backlog/flags meta-monitor. Only
  GpuOrchestratorAgent/FinOps (#3) remains: cross-platform Vast.ai+RunPod preflight + optimal instance selection
  + auto-terminate + billing -- HIGHEST risk (provisions paid infra), build LAST and only behind cost-safety
  guardrails (HITL-approve before spend, budget caps, confirm-on-terminate, never auto-spend). Recommend a design
  review before building it.
- Optional follow-up (enables AgentOpsMonitor error-rate/perf-drift): add orchestrator run-telemetry to a new
  'agent_runs' state section (per-run status/duration/error), then extend the ops detector.


## Proposed-agent roadmap -- update 2026-06-14
- [DONE] agent_runs telemetry follow-up: AgentOpsMonitor error-rate + perf-drift now backed by real
  orchestrator-recorded telemetry (was the documented gap). Only GpuOrchestratorAgent/FinOps (#3) remains --
  highest risk (provisions paid infra); recommend a design review before building (ground the existing Vast.ai
  workflow + RunPod gap + cost-safety guardrails: HITL-approve before spend, budget caps, confirm-on-terminate).


## Proposed-agent roadmap -- update 2026-06-14
- GpuOrchestratorAgent/FinOps (#3): DESIGN REVIEW landed (docs/design/GPU_FINOPS_DESIGN.md). Decision pending --
  recommend-only/emit-only advisor (zero spend, reuses launch_run16.pick_offer) recommended as the first slice;
  autonomous provisioning gated behind a separate sign-off. No money-adjacent code until confirmed.


---

<!-- roadmap-delta: 2026-06-14-to-2026-07-12 catch-up -->
# ROADMAP delta -- 2026-07-12 (FOUR-WEEK CATCH-UP: the roadmap had gone stale)

**This delta exists because the roadmap broke its own rule.** The header says *"Updated at the
end of every session"*. Its last content entry was **2026-06-14**; the file was last touched
**2026-07-01**. In the intervening four weeks the project ran **~130 commits**, changed the
feature contract twice, found and fixed a cohort coordinate corruption, rebuilt the evaluation
metric stack, and discovered that the test suite had been red for days. None of it was here.

A roadmap that is not current is not a roadmap -- it is a stale snapshot that invites exactly
the drift this project spends its time hunting. Recorded plainly so the failure is visible.

## 0. CORRECTIONS to stale headline facts above

| where | said | actual (2026-07-12) |
|---|---|---|
| §1 Project identity | "80 features" | **97** (`EXPECTED_TABULAR_FEATURE_COUNT = 97`, variant_ensemble.py:164) |
| §3 Snapshot | "Current state snapshot (2026-06-10)"; Run 15, 79 features | superseded -- see §5 below |
| `docs/runs/RUN_17_PLAN.md` §H_Run17 | "the expanded 91-feature contract (88 + 3 FinnGen R13 columns)" | **97** -- KEGG + COSMIC + Nucleotide Transformer landed 2026-07-06 (`80eb9c8`). The *runbook* was corrected (`61c2b04`); the *plan's hypothesis text* was not. **Open drift -- fix before Run 17 launch.** |

**Feature-contract history (each step is an audit, not a bump):**
79 → 80 (esm2_llr) → 81 → 82 (hetero_gnn_score) → 87 (rnaseq_*) → 88 (omim_n_diseases_molecular)
→ 91 (FinnGen R13) → **97** (KEGG ×2, COSMIC ×2, Nucleotide Transformer ×2, 2026-07-06).

## 1. Phase D -- data expansion: the connector wave (2026-06-15 → 2026-07-06)

| source | date | commit | note |
|---|---|---|---|
| 1000 Genomes `af_1kg_*` | 06-15 | `bd5eecf` | chr1-22+X, 437,668 variants |
| gnomAD chrY / chrMT allele frequency | 06-16 | `832f023` | root cause was PAR variants mapping X→Y |
| GTEx bulk median transcripts-per-million | 06-17 | `00a58f1` | |
| Reactome pathway count | 06-17 | `66337fc` | |
| RNA-seq family (`rnaseq_*`) | 06-17→19 | `de19458` | recount3/GTEx differential expression; leakage-validated |
| OMIM genemap2 + molecular feature #88 | 06-26 | `499ccc6` | fixed the "88 bug" |
| EVE (real CSV schema) | 06-21 | `369cc61` | |
| ClinGen wiring | 06-21 | `b79364d` | |
| dbSNP | 07-01 | `d5791c1` | DONE + verified |
| FinnGen R12 + R13 dual-release | 06-27→28 | `2884e9b`, `5344ddb` | contract 88 → 91 |
| AlphaFold Phase-D | 07-02→03 | `eba5c40`, `7d49b54` | v6/API URL resolution, O(n) Relative Solvent Accessibility, canonical isoform selection, deterministic row order |
| **KEGG + COSMIC + Nucleotide Transformer** | **07-06** | **`80eb9c8`** | **contract 91 → 97** |

## 2. Cohort integrity -- the coordinate incident (2026-07-08 → 07-11). THE most consequential work of the period.

- **`9df1221` (07-08) INCIDENT (critical):** deletion `ReviewStatus` loss -- the VCF join missed
  **98.8% of deletions**.
- **`cd9edfb` (07-08) ROOT CAUSE:** cohort `pos` is variant_summary `Start`, **not** `PositionVCF`
  -- an off-by-one on every padded deletion.
- **`193afa0` (07-08) → cohort-v2:** `pos -= 1` where `alt` is a padded prefix. **187,245 rows corrected.**
- **`3ff1d13` / `6f9ebe7` (07-09) GENOME-VERIFIED against GRCh38:** 187,235 / 187,245 padded
  deletions match at the corrected position (**99.9947%**); SNV control **2,000/2,000** (SNVs are
  never shifted, so they are a data-derived control that a slice error could not pass). The 10
  residual mismatches are genuine ClinVar-vs-GRCh38 disagreements → disposition **FLAG, not correct**.
- **`ef5e909` (07-09):** guard tolerates ≤0.1% genuine disagreement, writes every mismatch to a
  TSV, and adds an SNV control that **hard-fails on any slice/build error regardless of tolerance**
  -- a coordinate bug cannot be tolerated away.
- **`8626955` (07-09):** sequence windows rekeyed to cohort-v2 (pure key remap, content unchanged);
  distinguishes COVERAGE_GAP (tolerated) from KEY_MISMATCH (hard fail).
- **`da18481` (07-11):** ClinVar *"Conflicting classifications of pathogenicity"* → **uncertain**,
  at the connector level.
- **`322c23a` / `e3e422e` (07-11):** allele classification + alleleless-variant recovery + provenance.
- **`e2e7f1a` (07-11):** **cohort-v3** -- fresh ClinVar ingestion, rebuild, and diff tooling.
- **`07cb781` (07-11):** gene-disjoint 4-way split re-baseline + gene-disjoint external calibration.

## 3. Evaluation stack rebuild (2026-07-08 → 07-11)

- `959244f`, `4d7b3a5`, `e387927` -- metric core: AUROC (average-rank tie handling), AUPRC
  (tied scores collapse correctly), Expected Calibration Error / Maximum Calibration Error.
- `85fd6b0` -- overflow-safe sigmoid in Iteratively Reweighted Least Squares; **refuses to report**
  Brier / Expected Calibration Error / calibration slope on non-converged fits rather than printing
  a number nobody can trust.
- `baddd72` -- `evaluation/__init__.py` must not eagerly import scikit-learn (a lazy-import contract).
- `5190b50` -- **leakage question RESOLVED**: no univariate leak; top standalone feature is
  `is_loss_of_function`.
- `811a11b` (07-11) -- **Expected Calibration Error computation corrected**.

## 4. Test + repository integrity (2026-07-08 → 07-12) -- see `docs/status/REMEDIATION_2026-07-11_test-suite-red.md` for the full write-up

- **`ac47972` (07-08) TRIAGE: the suite was RED, 24 of 1,616, and had been for days.** Four clusters.
  The standing brief said *"596 tests pass"*; it had been wrong for long enough that nobody knew when
  it stopped being true.
- **`a4fd129` (07-11): 24 → 0.** All four clusters plus one regression and one cluster the triage
  never saw (a `sys.path` leak that published a counterfeit `genomic_variant_classifier` package).
- **`aa99ac6` (07-11): SINGLE SOURCE OF TRUTH.** There were **two** hand-synced implementations of
  the feature matrix. The five-stage correctness harness validated `variant_ensemble.engineer_features`;
  the training pipeline ran `DataPrepPipeline._engineer_features`. **The gate audited a code path the
  pipeline never executed** -- a silent zero there was structurally invisible. Proved equivalent over
  **117 adversarial comparisons** (exact on column set, ORDER, dtype and values; forcing every
  `df.get` default and every integral-input truncation path), then collapsed. **−376 lines.**
- **`343cc66` (07-11):** four TRACKED tests imported UNTRACKED scripts -- **a clean clone had a red
  suite**. Fixed.
- **`88af150` (07-12): the suite is now HERMETIC and IDEMPOTENT.** It had been writing 620 KB into its
  own repository on every cold run -- including a **live network download** from
  `https://alphafold.ebi.ac.uk` -- and two runs of identical code disagreed (1805/17 vs 1812/10).
  Invisible on a developer machine (warm caches write nothing) and invisible to `git status`
  (`data/raw/` is gitignored). One defect, five instances: **a library hard-coding a
  working-directory-relative writable path.**

**Current test state (2026-07-12):** local **1815 passed / 0 failed**; cold clone **1615 passed /
0 failed**; two consecutive runs identical; `Test-Path data/raw/cache` after a full cold run → **False**.

## 5. CURRENT STATE SNAPSHOT (2026-07-12) -- supersedes §3 above

- **Feature contract: 97.** Single implementation (`variant_ensemble.engineer_features`); the
  correctness harness and the training pipeline now run **the same code**.
- **Cohort: v3** (fresh ClinVar). v2 is genome-verified; the padded-deletion coordinate bug is closed.
- **Run 15 remains the last SEALED run** (commit `032a2ab`; 79 features, Test AUROC 0.9984).
  **Run 17 is planned and gated, NOT launched.**
- **Test suite: green, hermetic, idempotent, and green on a clean clone** -- none of which was true
  on 2026-07-08.
- **Continuous Integration: exists and gates the container path** (`.github/workflows/ci.yml`,
  Python 3.11 + 3.12, 508 runs). It was **RED on `e3e422e`** and merged past.

## 6. OPEN -- carried forward, dated, not lost

| # | item | status |
|---|---|---|
| 6.1 | Continuous Integration `--maxfail=5` -- reported **5** failures when the truth was **24**. | **CLOSED 2026-07-12** (`0849da3`). Removed, and `-rs` added so every skip states its reason. |
| 6.2 | ~~**Continuous Integration runs `tests/unit/` only** -- `tests/conformal/` (7 files), `tests/integration/` (1), and **22 root-level `tests/test_*.py`** = **30 test files never run in Continuous Integration.**~~ Widened to `pytest tests/` on branch `ci/widen-test-scope`, pull request **#2**, merged **2026-07-13** (`0996dec`). The 30 dark files include the entire cohort-construction and data-provenance layer -- `build_cohort_*`, `clean_cohort`, `dedup_collapse`, `ingest_clinvar_snapshot`, `recover_alleleless_*`, `seq_window_manifest`, and **`split_protocol_v2`, the gene-disjoint split on which every leakage claim in this project rests**. In 508 Continuous Integration runs, none had ever executed. **PREDICTION WAS WRONG:** widening was expected to go RED; it went **GREEN with zero new failures** (1587 -> 1785 passed, +198). The only two failures were the pre-existing Kolmogorov-Arnold Network defect (6.16), which the widening did not cause and which was already failing in the narrow gate. | **CLOSED 2026-07-13** (`9f701ec`, merged `0996dec`). Continuous Integration is green on Python 3.11 + 3.12 running the FULL suite. |
| 6.3 | ~~"The rented-GPU path bypasses Continuous Integration entirely."~~ **CORRECTED 2026-07-12.** `Run_Preflight_Local.ps1` (G1) **does** run `pytest tests/` -- the FULL tree, more than Continuous Integration does -- and hard-fails on any failure. The gate was never missing; it had **ROTTED**: its floors were `1485/1496` against a suite of **1,823 collected / 1,815 passed**, so ~330 tests could have vanished and it would still have said PASS. Floors refreshed to `1805/1815` (`0b93d30`). **Still open:** `Run_Preflight_VM.sh` / `vm_bootstrap_run.sh` run no pytest, and G1 has a `-SkipPytest` escape hatch. | PARTLY CLOSED |
| 6.4 | `RUN_17_PLAN.md` hypothesis said **91** features; actual **97**. | **CLOSED 2026-07-12** (`721a23e`). Corrected, and G1 §13c now **DERIVES** `EXPECTED_TABULAR_FEATURE_COUNT` from the package and hard-fails on disagreement. Negative-tested three ways: correct marker PASSES, stale marker FAILS, absent marker FAILS. |
| 6.5 | Correctness-harness sanity model **does not converge** (lbfgs, max_iter 1000 *and* 200; reproduced on Python 3.11 and 3.12 in Continuous Integration). Stage 3 is weakened as evidence while its own reference model is unconverged. **The most scientifically substantive item left.** | OPEN |
| 6.6 | ~~LightGBM **feature-name mismatch** -- fitted on a named DataFrame, predicted on a bare ndarray; column order trusted implicitly. Silently wrong if it ever drifts.~~ **THIS ENTRY WAS WRONG IN EVERY CLAUSE.** LightGBM is fitted on an ndarray and predicted on an ndarray, consistently, in `fit`, `_leakfree_oof`, `predict_proba` and `evaluate`. There was no mismatch and no order hazard. Instrumenting the warning (2026-07-13) rather than reading its text found **three real things** -- 6.6a/b/c below. | **CLOSED 2026-07-13** (`7d42409`, `f49d8c0`) -- and **corrected**. Full write-up: `docs/status/REMEDIATION_2026-07-13_warnings-and-silent-model-drop.md` |
| 6.6a | **Silent base-model erasure.** `VariantEnsemble.fit` wrapped each base model's out-of-fold step in a bare `except Exception`: on any failure it logged one line, set the out-of-fold column to 0.5, and `continue`d -- which also skipped `model.fit()`. The model was never fitted, never checkpointed, and absent from `trained_models_`, `oof_model_names_`, the blend, and every comparison artifact. **A 13-model ensemble became a 12-model ensemble, the survivors reported normal metrics, and the run looked healthy.** This directly corrupts the project's stated goal of comparing every algorithm: a dropped model does not look like a failure, it looks like a model that was never a candidate. (The constant-0.5 column was *not* fed to the meta-learner -- `valid_cols` drops it. That was mis-stated twice during triage and is corrected in the code comments.) Discovered because a **spurious** warning, escalated under `-W error::UserWarning`, was sufficient to delete LightGBM from a run. **Noise could delete a model.** | **CLOSED 2026-07-13** (`7d42409`). Now **raises** by default (`EnsembleConfig.allow_base_model_dropout = False`); opt-in dropout logs at ERROR with a traceback and records name + cause in `VariantEnsemble.dropped_models_`. 6 tests. |
| 6.6b | **LightGBM does not enforce its own feature names.** Measured 2026-07-13: fit on a DataFrame, then predict with the same data in a *different column order* -> **silently wrong predictions, max delta 0.855 in probability, no error, no warning, even under `-W error`.** It maps columns POSITIONALLY. scikit-learn and XGBoost raise `ValueError`; CatBoost reorders by name. **LightGBM is the sole outlier in the roster.** The `X_tab.values` dispatch is therefore **load-bearing**, not stylistic: "cleaning it up" to pass DataFrames would arm a silent-corruption bug in the pathogenicity classifier. This was very nearly done. | **CLOSED 2026-07-13** (`f49d8c0`). `tests/unit/test_feature_name_contract.py` (7 tests) records the measured behaviour of all four libraries, pins the dispatch, and acts as a **library-upgrade tripwire**: if LightGBM ever fixes positional mapping, the test fails on purpose and reports that the constraint can be lifted. |
| 6.6c | The 11 LightGBM warnings themselves: **spurious.** LightGBM 4.6.0 populates `feature_names_in_` with synthetic names (`Column_0`, ...) even when fitted on a bare ndarray; scikit-learn leaves it unset. That asymmetry -- nothing else -- produced the warning. | **CLOSED 2026-07-13** (`f49d8c0`). Suppressed in `pyproject.toml` **pinned to that exact message** (never a blanket `ignore::UserWarning`), with the premise gated by 6.6b's tests. Verified afterwards that the 18 `n_components` warnings remained visible -- a filter that silenced both would have recreated the very condition that let these defects survive. |
| 6.13 | **`n_components > n_samples` (18 warnings/run) -- NOT "test-scale noise".** A latent correctness bug: `ScalableSVM._build_headline` clamped the Nystrom/RFF map dimension against `n_samples`, but `calibrate=True` (the DEFAULT) hands the pipeline to `CalibratedClassifierCV`, which **refits Nystroem on each cross-validation training fold** -- strictly smaller than `n_samples`. The clamp was off by the cross-validation factor and scikit-learn silently truncated the map. Measured: cv=3 -> 3 warnings, cv=5 -> 5 warnings, `calibrate=False` -> 0. One per fold, exactly. **Invisible at production scale** (n≈1.7e6, D=1024: the clamp never binds), which is why it was waved past for weeks. | **CLOSED 2026-07-13** (`f49d8c0`). New `_rows_the_map_is_fitted_on()` = `n - ceil(n/k)`; `_map_dim()` is now the single source of truth (the `fit` log line held a **second hand-kept copy** of the formula and would have begun reporting the old D while the model trained with the new one). 20 tests; the fold size is derived **empirically from `StratifiedKFold`**, not asserted against itself. Suite runtime 486s -> 430s as a side effect. |
| 6.14 | **The G1 pre-flight pytest floor rotted TWICE IN TWO DAYS.** Refreshed 1485 -> 1805/1815 on 2026-07-12 beneath an emphatic *"RAISE THIS WHENEVER YOU ADD TESTS"* comment. By 2026-07-13, 33 tests had been added and the floor still read 1805 against a suite passing **1,852** -- **47 tests of dead slack, created in the same session that wrote the roadmap entry naming this exact failure pattern.** A **third**, stale copy (`test floor 1485/1480`) also sat in the script's header, contradicting both live values. **This is a DESIGN defect, not a discipline defect: the comment does not enforce itself, and no amount of emphasis will make it.** The fix is the one this project already uses successfully for features -- a single committed constant behind a fail-loud guard, exactly as `EXPECTED_TABULAR_FEATURE_COUNT` guards `TABULAR_FEATURES`. **Proposal:** one committed suite-size constant; a `conftest` collection hook enforcing it **only** under an explicit `--assert-suite-size` flag (so running a subset locally is unaffected); G1 and Continuous Integration both pass that flag and both read the same constant. Adding a test then turns the suite red until the constant is bumped -- the ratchet cannot be forgotten, because forgetting it fails loudly. | **CLOSED 2026-07-13.** **THE RATCHET IS BUILT.** `tests/EXPECTED_SUITE_SIZE` holds ONE number (**1882**); `tests/conftest.py` aborts the run under `--assert-suite-size` if the **collected** count disagrees **in either direction** (fewer = tests VANISHED; more = ratchet not bumped). G1 and Continuous Integration **both** pass the flag and read the **same file** — the two hard-coded floors (`$MinPytest`, `$minPass`) are **DELETED**, so the number no longer exists in four places disagreeing with itself in three. **It asserts `collected`, not `passed`,** because the passed/skipped split is environment-dependent (Windows 1875p/7s vs Linux 1856p/13s/1xf) while the collected count is not — and it only became environment-independent today, since closing 6.17 stopped a module-level `importorskip` on a missing `torch_geometric` from collapsing eleven tests into one skip entry. **Negative-tested in BOTH directions (exit code 4 each time), and its first act was to catch its own author:** adding its 12 self-tests without bumping the number turned the suite RED immediately. 12 self-tests (roadmap 6.9: *a guard with no self-test is a rumour*), including every way of corrupting the ratchet file — a malformed ratchet is the one failure that would leave the guard silently inert while still carrying a reassuring name. |
| 6.15 | **A stale sandbox mount silently reverted `docs/ROADMAP.md`, and the corruption was committed and pushed (`f49d8c0`).** A three-line string substitution was performed on the tracked file via a `python` read-modify-write through the Linux sandbox's mount of the repository. That mount served a **stale cached copy** predating `f377659`; the write put it back. Result: the entire four-week catch-up delta *and* §6 (the open register) were deleted -- **158 deletions, 0 insertions** -- and the roadmap edits made moments earlier with the Windows-side file tools were discarded. The commit output said `158 deletions(-)` and it was read past. The mount had **already** been recorded as unsafe (it produced a phantom `SyntaxError`, a truncated `real_data_prep.py`, and a fabricated content-loss diff on 2026-07-12); the recorded rule was *"git must run on Windows, never in the sandbox"*, and it was then violated with an operation strictly more dangerous than the reads that prompted it. Restored from `e1ef05b`; verified 636 lines, §6 present, delta present. | **CLOSED 2026-07-13** (restore commit). **STANDING RULE: tracked files are edited ONLY with the Windows-side file tools. The sandbox shell is for running code, never for writing into the repository.** |
| 6.16 | **THE KOLMOGOROV-ARNOLD NETWORK HAD BEEN SILENTLY ABSENT FROM EVERY CONTINUOUS INTEGRATION RUN SINCE MAY.** Found by 6.6a's fail-loud handler on its FIRST clean run -- one day before Run 17. **Root cause is in `__init__`, not `fit`:** `imodelsx` 1.0.13 (the latest release) accepts `test_size` / `random_state` / `shuffle` and **throws them away** (verified: `hasattr(m, "test_size")` is False after construction); `fit()` then reads them as **bare names** and raises `NameError`. `KANClassifier.fit()` cannot run at all on an unmodified install. Since 2026-05 the launch scripts `sed`-ed the **installed `site-packages` file** to redirect the lookup onto `self`, and `kan.py` supplied `self.<name>` -- **two halves of one mechanism; neither works alone.** The `sed` ran on the developer's laptop and the Run 11 / Run 16 instances. It was **NEVER in Continuous Integration, NEVER in Docker, and NOT in `scripts/vm_bootstrap_run.sh` -- the Run 17 path.** So KAN raised `NameError` in every Continuous Integration run, the old bare `except Exception` ate it, and a **TWELVE-model ensemble trained and reported normal metrics**. **Run 17 would have provisioned a fresh box, PASSED EVERY PRE-FLIGHT** (section E checked that `KANClassifier` *imports* -- it imports fine; the bug is in `fit()`), **trained for eleven hours, and published a twelve-model algorithm comparison with a headline model silently missing.** Second defect: the `sed` left the developer's `.venv312` holding a **mutated `site-packages`**, so local tests exercised a code path **no clean machine had** -- *"it passes on my machine"* was structurally load-bearing for two months. | **CLOSED 2026-07-13** (`98ebd6b`). In-process repair (`kan.py::_repair_imodelsx_kan_bare_names()`, BOTH bindings, every environment); `sed` deleted from all three launch scripts + `RUN16_RUNBOOK.md`; `patch_runbook_kan_and_offer.py` refuses to run; `.venv312` restored to **pristine** (stack held: pandas 2.3.3, transformers 4.46.3); `imodelsx==1.0.13` pinned; **pre-flight now FITS every base model, not imports**; `VariantEnsemble.ensemble_completeness_` written to run artifacts. 6 tests incl. an upstream tripwire and a divergence detector. Full write-up: `docs/status/REMEDIATION_2026-07-13_warnings-and-silent-model-drop.md` section 9. |
| 6.17 | **NEW -- THE GRAPH-NEURAL-NETWORK BRANCH IS ENTIRELY UNTESTED IN CONTINUOUS INTEGRATION.** Visible only because `-rs` was added on 2026-07-12. `torch_geometric` is not installed on the runner, so **eleven test entries skip**: `test_ablate_gnn`, `test_gnn_experimental_channel`, `test_gnn_gps`, `test_gnn_optim`, `test_gnn_shared_graph`, `test_gnn_tier2_denoise`, `test_hetero_gnn` (x2), `test_hetero_gnn_scorer` (x2), `test_hetero_inductive_pyg` (x2) -- the STRING-DB graph neural network, the hetero-graph neural network, GraphGPS, the ablation, and the inductive path. **`gnn_score` is a REAL, non-degenerate ensemble feature** (roadmap section 3: non-degeneracy gate PASS). Also skipped for missing dependencies: `pandera` (5 schema-drift tests), `pyspark` (3 in `test_core`), `river` (1), `mapie` (1 conformal cross-check). **This is the SAME DEFECT SHAPE as 6.16: a headline component whose failure the gate cannot see.** Widening 6.2 brought the FILES in; it did nothing about the DEPENDENCIES that skip them out again. **ROOT CAUSE (6.18):** `requirements.txt` -- the ONLY file anything installs -- was missing `torch`, `torch-geometric`, `networkx`, `numba`, `pandera`, `pyspark` and `river` **entirely**, because it is a `pip-compile` artifact of `requirements.in` **compiled under Python 3.14**, where `torch` has no wheels. `requirements.lock` (Python 3.12) has all of them, and **nothing installs it**. `torch` reached every environment only as a transitive dependency of `imodelsx` -- undeclared and unpinned, in a deep-learning project. `torch_geometric` reached the rented instance only via an ad-hoc unpinned `pip install` at `vm_bootstrap_run.sh:147`. The Docker **training** image (`Dockerfile:159`) installs the same file, so it cannot run the graph neural network either. **AND `mapie` was declared NOWHERE:** `tests/conformal/test_mapie_crosscheck.py` -- the ONLY independent check of this project's from-scratch conformal prediction against a mature reference -- **had never executed on any machine, ever.** Declared and run 2026-07-13: **all 3 PASS.** Our LAC (Least Ambiguous set-valued Classifier) sets match MAPIE element-wise EXACTLY; our APS (Adaptive Prediction Sets) is confirmed a valid, *tighter* variant rather than a bug. A correct test that never runs is not a test. | **CLOSED 2026-07-13** (`a77c4a2`). `requirements.txt` completed with `torch==2.11.0`, `torch-geometric==2.7.0`, `networkx==3.6.1`, `numba==0.65.1`, `pandera==0.24.0`, `pyspark==3.5.8`, `river==0.23.0`, `mapie==1.4.1` -- every version **measured** from the green environment and matching `requirements.lock` exactly. New Continuous Integration step **FAILS THE BUILD** if any coverage-critical dependency is absent, instead of letting its tests skip while the suite reports green. Suite 1860 -> **1863 passed, 7 skipped, 0 warnings**. |
| **CORRECTION** | Commit `a77c4a2`'s message asserts *"Neither nannyml nor evidently is imported anywhere in src/."* **That is FALSE.** It came from a malformed PowerShell glob that silently matched nothing -- a negative asserted from a search that had not been validated. `evidently` is imported in **4** files, `nannyml` in **1**, `river` in **1**. The true state is worse than the false claim, and is recorded as **6.19**. Logged here because the commit message cannot be rewritten and the register must not inherit the error. | **CORRECTED 2026-07-13** |
| 6.18 | **`requirements.in` / `requirements.lock` vs `requirements.txt`: a DUAL SOURCE OF TRUTH that has drifted, and the lock is a TRAP.** `requirements.txt` is a `pip-compile` artifact of `requirements.in` -- and its own header records that it was compiled **UNDER PYTHON 3.14**, where `torch` has no wheels. `pip-compile` therefore resolved the graph WITHOUT the torch stack and emitted a file with `torch`, `torch-geometric`, `networkx`, `numba`, `pandera`, `pyspark` and `river` **entirely absent**. `requirements.lock` (compiled under Python 3.12) has all of them. **Only `requirements.txt` -- the sole file that Continuous Integration, the Dockerfile and every launch script actually install -- was missing them** (see 6.17). The header even names the fix: *"re-run pip-compile under Python 3.12 once .venv312 is bootstrapped"*. It IS bootstrapped. **But regenerating today would be a disaster:** `requirements.in` has NO UPPER BOUND on `transformers` (`>=4.40` → resolves **5.8.0**, the family that **killed a Run 17 smoke test** -- the Nucleotide Transformer imports a symbol `transformers` 5.x removed) or on `pandas` (`>=2.0` → the 3.0.4 Windows cp312 `date_range` segfault). Both ceilings were bought with real incidents and exist **only** as exact pins in `requirements.txt`. **`requirements.lock` is therefore installed by nothing, and installing it would re-introduce a break the project already paid to escape.** Also: the `lockfile-check` Continuous Integration job verifies `requirements-api.lock` against `requirements-api.txt` -- it **never** checks `requirements.lock` against `requirements.in`. A drift gate guarding the lock nobody uses, blind to the one that matters (**root pattern (c)**). | **OPEN.** Stage 1 done (`a77c4a2`: `requirements.txt` completed by hand with the exact, measured, known-good versions). Stage 2 = add the missing ceilings to `requirements.in`, recompile **under Python 3.12**, verify the new lock reproduces the known-good stack EXACTLY, extend `lockfile-check` to the main pair, then make ONE hash-pinned file the single install target. **Deliberate and separately verified -- never a side effect of another change.** |
| 6.19 | **THE DRIFT-MONITORING SUBSYSTEM IS DECLARED, WIRED, AND SILENTLY DEAD.** Found 2026-07-13 while closing 6.17. `pip check` reports **six broken requirements** that nothing in this project has ever run -- not Continuous Integration, not G1, not the bootstrap. Measured: (a) **`nannyml` 0.13.0 -- the Confidence-Based Performance Estimation engine -- CANNOT IMPORT AT ALL.** It imports `pyspark.pandas`, which uses `np.NaN`, **removed in NumPy 2.0**; we pin `numpy==2.4.4`. (b) **`evidently` 0.7.6 is called through the 0.4.x API** (`from evidently.report import Report`, `from evidently.metric_preset import ...`) -- **both modules no longer exist**. `scripts/run_drift_monitor.py:192` wraps the import in `try/except ImportError` and calls `logger.warning`, so **the Evidently drift report is never generated and the only trace is a log line nobody reads**. (c) `scripts/verify_drift_libs.py` -- the script whose entire job is to VERIFY the drift libraries -- imports `alibi_detect` at module level, **a library `requirements.in` explicitly records as DROPPED**, so the script cannot run. (d) The scheduled `drift_monitor.yml` workflow installs only `requirements-api.txt` + `scipy`, which contains **neither `evidently` nor `nannyml`** -- so it has been running with its drift libraries absent. (e) `river` (online drift) is the ONLY member of the stack that actually works. **The `<4` ceiling on `pyspark` in `requirements.in` is what keeps `nannyml` dead: PySpark 4.x is the version that supports NumPy 2.** `requirements.in` credits this stack with replacing `alibi-detect` for drift monitoring. **That capability does not exist.** | **CLOSED 2026-07-13.** **THE CAPABILITY IS NOW REAL, AND IT NEVER EXISTED BEFORE.** **CORRECTION TO THIS ENTRY:** it originally said `nannyml` was *"declared, wired, and silently dead."* **It was NOT wired.** It was imported by exactly ONE file -- `scripts/verify_drift_libs.py`, which printed its version number. **There was no Confidence-Based Performance Estimation implementation anywhere in this codebase.** `requirements.in` claimed a capability that had never been built. **AND: `nannyml` 0.13.0 AND 0.13.1 both require `lightgbm>=3.3,<4.6`** (verified by reading both wheels' METADATA) **while the ensemble TRAINS on `lightgbm` 4.6.0 -- a BASE MODEL.** Reviving it in-place would mean downgrading a model of the classifier to satisfy a monitoring library: changing the science to suit the instrument. **Refused.** **THE FIX:** an **isolated drift environment** (`requirements-drift.txt`: `nannyml` 0.13.1, `evidently` 0.7.6, `lightgbm` 4.5.0, `plotly` 5.24.1, **NO `pyspark`**, and the shared-data contract `numpy`/`pandas`/`scikit-learn` **pinned to MATCH training** so the monitor reads the same parquet files the pipeline writes). `nannyml` breaks *because* `pyspark` is importable -- it optionally imports it, and `pyspark` 3.5.x calls `np.NaN`, **removed in NumPy 2.0**; it is not a `nannyml` bug and no version fixes it. **Built:** `monitoring/performance_estimator.py` (fails LOUD, never a `logger.warning`), wired into `run_drift_monitor.py --estimate-performance`, **12 tests in `tests_drift/`** -- a separate top-level directory outside `testpaths`, because a module-level `importorskip` would collapse N tests into one skip entry and **silently break the suite-size ratchet (6.14)**. **Verified by execution:** CBPE estimates ROC AUC 0.802 on data with the labels removed (realized 0.821), and collapses toward 0.5 when fed a model emitting noise. **Evidently PORTED** to the 0.7 API (`evidently.report` / `metric_preset` / `pipeline.column_mapping` are all **DELETED**; `DataQualityPreset` is now `DataSummaryPreset`; **`Report.run()` takes `current_data` FIRST** -- positional args would have silently produced a BACKWARDS drift report). The old code caught the `ImportError` and logged *"Evidently AI not installed. Run: pip install evidently"* -- **Evidently WAS installed; the API had been deleted.** The code misdiagnosed its own failure and the report was silently never produced for months. **`verify_drift_libs.py` rewritten** -- it imported `alibi_detect` (explicitly DROPPED) at module level and therefore **could not run at all**; the script whose job was to verify the drift libraries was itself broken by a library the project had removed. **`pip check` is now a HARD GATE** in Continuous Integration (occurrences before today: **zero**), and `lockfile-check` now asserts the lock **MATCHES** `requirements-api.txt` rather than merely that it *installs* -- **root pattern (c), fourth instance**. **`prometheus-fastapi-instrumentator` raised past `<8.0`**: 8.0.2 requires `starlette>=1.0.0,<2.0.0`, exactly what we run, resolving the one real conflict pip had been reporting in an ERROR line on **every one of 508 runs** while exiting 0. **AND A CORRECTION I OWE:** the "six broken requirements" I attributed to the project were **LOCAL POLLUTION** -- stray `nannyml`/`evidently`/`fairlearn`/`great_expectations` in `.venv312` that `requirements.txt` never declared. Removing them took local `pip check` from **6 conflicts to 0**. I measured a mutated environment and reported the result as a property of the project: **root pattern (d), third instance, mine.** |
| 6.20 | **NEW -- THE SCHEDULED DRIFT MONITOR HAS NEVER RUN, AND REPORTED "NO DRIFT" EVERY MONTH ANYWAY.** Found 2026-07-13 while closing 6.19, and it is **the worst instance of this project's central failure pattern**, because unlike every other one, **it does not fail -- it SUCCEEDS.** `.github/workflows/drift_monitor.yml` fires on the first of every month (`cron: "0 6 1 * *"`). Its "Download reference splits from Google Drive" step is a **PLACEHOLDER**: it prints *"real download NOT wired yet"* and does nothing but `mkdir -p` an **empty directory**. The next step then checks whether that directory is empty -- **it always is** -- and on finding it empty does: `echo "No reference splits available -- skipping drift check (avoids false alert)"; echo "exit_code=0"; echo "drift_level=none"; exit 0`. **So every month it created an empty folder, noticed the folder was empty, and reported `drift_level=none` WITH A GREEN TICK, having never looked at a single row of data.** Both steps also carried `continue-on-error: true`, so a genuine failure would have been swallowed too. And the catch-all `*)` branch mapped **every unrecognised exit code to "none"** -- a crash, a segfault or an out-of-memory kill would have been reported as a clean bill of health. **"No drift detected" and "drift was not checked" are not the same statement. Reporting the second as the first is how a monitoring system lies.** A clinical variant classifier has had a monthly drift monitor reporting "no drift" since the day it was written, having never once checked. **The workflow also installed `requirements-api.txt` + `scipy`** -- the INFERENCE stack, containing no `evidently`, no `nannyml`, no `pandera`, no `river` -- **and never installed the project package at all**, so `python scripts/run_drift_monitor.py` could not even have imported `monitoring.drift_detector`. | **CLOSED 2026-07-13** (the lie), **PARTLY OPEN** (the capability). The **lie is fixed**: absent data reports `drift_level=UNKNOWN` and exits non-zero; unrecognised exit codes report `UNKNOWN`, never `none`; `continue-on-error` is gone from the file entirely; the workflow installs the **isolated drift environment** and the project package, and runs `pip check` + `verify_drift_libs.py`. **The Google Drive download was not wired -- it was DELETED.** Wiring it was the wrong fix: it needs a credential, moves a 23.8 MB cohort matrix into a hosted runner on a PUBLIC repository every month, and from Run 17 that matrix carries real OMIM-derived values (OMIM is `tier: controlled`; `data_manifest.yaml:347` requires `sync:false` for a built artifact derived from a controlled cohort). Replaced by an **aggregate-only reference profile** (`monitoring/drift_reference_profile.py`, `scripts/build_drift_reference_profile.py`): per-feature histogram counts + quantile grids, **1,450 KB, committed to git**, no variant rows, no identifiers, no per-variant annotation values. **The Population Stability Index computed from it is BIT-IDENTICAL to the raw matrix** -- measured on the real 1,038,974-variant Run-15 matrix, **worst delta across all 78 features = 0.000e+00**, asserted with `==` and not `pytest.approx` (`tests/unit/test_drift_reference_profile.py`, 21 tests). What the profile CANNOT do -- the multivariate Maximum Mean Discrepancy and Székely-Rizzo energy tests, which need real reference samples -- it reports as **NOT COMPUTED** (`joint_tests_run=False`), never as passing: a silently-substituted `mmd_pvalue = 1.0` would have permanently disarmed the urgent-retrain escalation while looking perfectly healthy. **A SECOND LIE WAS FOUND ONE LAYER DOWN AND FIXED:** `run_drift_monitor.py::run_feature_drift` **returned 0 -- "no drift" -- when given no new data**, and the workflow invoked it with `--features-only` and NO `--new-data`, so it took that branch every month. Fixing the workflow alone would have MOVED the lie, not killed it. It now exits `EXIT_NOT_CHECKED = 4`, a code of its own -- deliberately **not** 3, because 3 means `urgent_retrain` and firing a SEVERE DRIFT alarm for a check that never ran is the same lie in the opposite direction. Also killed: `reindex(columns=..., fill_value=0.0)`, which **invented absent features as columns of zeros and drift-checked them as though measured**. **STILL OPEN:** the monitor has a reference but no NEW-release feature matrix on a hosted runner, so the monthly run reports **UNKNOWN (exit 4)** -- honest, red, and loud. |
| 6.21 | **NEW -- RUN 15 TRAINED, EVALUATED, AND PUBLISHED AN AUROC OF 0.998 WITH 36 OF ITS 78 FEATURES CONSTANT ZERO.** Found 2026-07-13, by accident, while building the drift reference profile (6.20) -- because the profile forced someone to look at the *values* in the matrix rather than at the file list, the logs, or the launcher. **46% of the feature space did not exist**, across 1,038,974 variants. Whole sources were silently stubbed to 0.0: **GTEx (6 features), 1000 Genomes (5), FinnGen (3), AlphaFold/protein structure (4), splice/MaxEntScan (4), UniProt (2), OMIM (2), HGMD (2), ESM-2, EVE, dbSNP, PhyloP, ClinGen, codon_position, gene constraint.** The reported 0.998 was produced by the **38 features that were real**. Nothing in the pipeline said a word. **ROOT CAUSE:** connectors do not raise when their source file is absent -- they return zeros. `omim.py:105` is the canonical shape: `if gene_table.empty: result[...] = DEFAULT_N_DISEASES; return result` -- no log, no warning, no raise. And `variant_ensemble.engineer_features` does the same one layer up: `df.get("omim_n_diseases", pd.Series([0] * len(df)))`. **WHY THE EXISTING GATES DID NOT CATCH IT** -- and this is the whole lesson: `scripts/launch_run17_*.sh` hard-aborts (exit 8) if the OMIM `genemap2`, PhyloP, dbSNP, AlphaFold, ClinGen, EVE, UniProt or FinnGen **FILE** is missing. Right instinct, **wrong layer**. A file-existence check is a **PROXY**. A present-but-empty file, a schema change, a renamed column, or a failed gene-symbol join all sail straight through it and still deliver a column of zeros. That is section 7, root pattern (c), and it is the most expensive one this project has. The correctness harness's `_stage5_zero_audit` DOES check for all-zero columns -- but it runs on `build_reference_slice()`, a **synthetic fixture where every feature is populated by construction**, so it validated a code path, not the data. **A FILE THAT EXISTS IS NOT A FEATURE THAT VARIES.** | **CLOSED 2026-07-13.** Two gates that assert **the feature itself**, against the real engineered matrix, sharing ONE definition of "dead" so they cannot drift apart. (a) `variant_ensemble.feature_census()` + a **pre-training census in `run_phase2_eval.py`**, placed on the FULL matrix *before* the `--max-train` subsample (so a sparse feature is never mistaken for a dead one by looking at a slice). It prints every dead feature **with the data source and the CLI flag responsible for each** (`FEATURE_SOURCE`), then **exits 2**. `--allow-dead-features` exists, defaults OFF, and records the casualties in the run artifacts rather than hiding them. (b) `EnsembleConfig.allow_zero_variance_features = False` + `VariantEnsemble._assert_no_dead_features()`, armed at the top of `fit()` as a backstop. Row threshold 10,000 so small synthetic fixtures WARN rather than fail -- otherwise the guard would redden every test in the repo and be switched off within a day, which is how guards die. 12 tests (`tests/unit/test_zero_variance_guard.py`), including the exact Run-15 casualty list as a regression. **A source that fails to populate now costs SECONDS in the smoke run, not eleven hours of paid compute followed by a twelve-model algorithm comparison built on a feature space that was half imaginary.** |
| 6.21a | **HGMD REMOVED from the feature contract** (97 -> 95), 2026-07-13. Two independent reasons, either sufficient. **(1) No access:** HGMD Professional is a paid QIAGEN licence that is not held (6.x, "PAID, blocked"). The connector `data/hgmd.py` is fully implemented and tested -- but was **never wired** (no `--hgmd-path` flag reaches the pipeline, no data file exists), so `hgmd_is_disease_mutation` and `hgmd_n_reports` were **constant zero for the entire life of the project**, contributing nothing while occupying two slots in the roster. **(2) Label leakage -- and this reason SURVIVES the licence arriving:** HGMD "DM" means *disease-causing mutation*; the training label is ClinVar Pathogenic (`real_data_prep.py:512`). They are the same quantity under two vendors' names, and HGMD-DM overlaps ClinVar-Pathogenic heavily. As a **variant-level** feature it is an answer key, and the gene-aware split cannot help because the leak lives *inside every fold*. The deployment failure is the damning part: a novel **variant of uncertain significance** -- precisely what this classifier exists to score -- has no HGMD entry, so the flag reads 0, and the model concludes "not a disease mutation" and leans benign. **It would post a superb AUROC on a test set of catalogued variants and systematically under-call the variants that matter.** The project already draws this line elsewhere: `real_data_prep.py:1169` stubs COSMIC at 0.0 and names the reason `feature-not-label`. **The old tests asserted `hgmd_* == 0` when the column was absent -- the defect written down AS A REQUIREMENT, passing every run while certifying that a zero-information feature was behaving correctly.** | **CLOSED 2026-07-13.** Removed from `TABULAR_FEATURES`; `EXPECTED_TABULAR_FEATURE_COUNT` 97 -> 95; `INFERENCE_FEATURE_COLUMNS` auto-follows. The **connector is KEPT and its 7 tests kept passing** -- dormant, not deleted. Absence is now PINNED by tests in `test_hgmd.py`, `test_splice_ai_promotion.py` and `test_schema_baseline_matches_contract.py`, because a two-line deletion is exactly what a well-meaning merge restores. **When the licence arrives, do NOT restore the two lines:** wire it **gene-level and leave-one-out** (`n_hgmd_dm_in_gene`, counting HGMD-DM variants in the gene while EXCLUDING the variant being scored), mirroring the existing `n_pathogenic_in_gene`. Same biological signal, no answer key. Note also `data_manifest.yaml:286`: REVEL/VEST4/FATHMM/MutPred2 are **trained on HGMD-DM**, and `revel_score` is already a feature -- so even a gene-level HGMD introduces circularity with predictors already in the roster, and deserves a held-out ablation rather than an assumption. |
| 6.22 | **NEW -- THE SCHEMA GATE, THE G1 PREFLIGHT, AND THE METHODS DOCUMENT WERE ALL DESCRIBING A MODEL THAT DOES NOT EXIST.** Found 2026-07-13 while regenerating the schema baseline after 6.21a. Three separate hand-kept copies of the feature count, all stale, none of which any test re-derived. **(a) `data/reference/schema/schema_baseline.json` was TEN COLUMNS behind `TABULAR_FEATURES`** (`cosmic_recurrence`, `cosmic_sig_tier`, 3x `finngen_r13_*`, 2x `genomiclm_*`, 2x `kegg_*`, `omim_n_diseases_molecular`). Its own `captured_from` field confessed why: *"derived: run16b-smoke baseline + hetero_gnn_score + 5 rnaseq_* **surgically added**"* -- it was never captured, it was **hand-maintained**. So the gate whose entire job is to catch a silently-changed column set **WAS ITSELF a silently-changed column set**, and would have fired on Run 17 for its own staleness. **A gate that cries wolf about itself gets switched off, and then the next REAL schema change goes through unseen.** **(b) `scripts/preflight_run17.py` hard-coded `EXPECTED_SCHEMA_COLS = 87`** -- in the G1 gate that guards **eleven hours of paid compute**. It had **already been hand-patched once** (`scripts/patch_preflight_schema_cols_87.py` exists solely to walk it 82 -> 87) and was stale again at 95. G1 would have **FAILED a correct baseline** and blamed it on corruption. `tests/unit/test_preflight_run17.py::test_schema_gate_87_ok` held a **third** copy and would have defended the stale gate. **(c) `METHODS.md` -- the scientific description of this model -- claimed "a total of 64 tabular features"** against a contract of 95, with a group table that summed to **62**: three numbers, no two agreeing. It also listed **HGMD Professional as data source 12** ("Disease mutation flag, report count") -- a source whose licence was never obtained and whose two columns were constant zero throughout. **A wrong number in a methods document is the one that ends up in a paper.** `docs/CHANGELOG.md:3159` had ALREADY recorded the lesson -- *"feature count is hardcoded in >=4 places; centralize into one EXPECTED_TABULAR_FEATURE_COUNT"* -- and it happened again anyway, **because the lesson lived in a changelog instead of a gate.** | **CLOSED 2026-07-13.** The count now lives in **exactly one place** (`EXPECTED_TABULAR_FEATURE_COUNT`), and the baseline, the G1 gate, `METHODS.md` and the preflight tests all **re-derive** it on every run. `build_schema_baseline.py --from-contract` DERIVES the baseline by running the **real feature builder** over the harness fixture and asserting the column set **and order** equal `TABULAR_FEATURES` (order is in the hash, and LightGBM maps columns positionally -- 6.6b). `preflight_run17.py` now **imports** the count and aborts loudly if it cannot, rather than waving a paid run through on a guess. **A BUG WAS CAUGHT IN THE FIX ITSELF, BEFORE IT SHIPPED:** the first `--from-contract` captured the RAW dtypes `engineer_features()` emits -- `int64` for ~40 columns -- but the schema gate validates the **PERSISTED, STANDARDISED** matrix, which is `float64` for all 78 of 78 columns (measured). `SchemaDriftAgent._dtype_family` is IDENTITY for numeric dtypes, so that baseline would have reported **~40 phantom DTYPE CHANGES** and failed Run 17 for drift that did not exist -- replacing a stale-columns bug with a wrong-dtypes bug in a gate whose only value is being trustworthy. **The tell was on screen the whole time and was read past:** the OLD baseline recorded `hgmd_is_disease_mutation` as `float64` even though the builder `.astype(int)`s it -- one line that said "this came from a processed matrix", quoted twice before anyone saw what it meant. Fixed and **PROVEN, not asserted**: new `--verify-against <matrix>` refuses to write the baseline unless every dtype it can check matches a real persisted matrix. Run against Run-15's `X_train`: **76 columns in common, 0 dtype mismatches, 19 reported as UNVERIFIED** (post-Run-15 features) rather than silently accepted. Baseline hash `a759084188...` (contaminated) -> `bbaba75d1f...` (correct). 6 tests (`test_schema_baseline_matches_contract.py`: set, ORDER, count, all-float64 dtypes, HGMD absence, hash self-consistency) + 3 (`test_methods_feature_count.py`: the stated count, the group table's SUM, and no HGMD source row). Full suite **1926 passed / 7 skipped / 0 failed**, G1 schema gate `OK: n_columns=95`. |
| 6.23 | **NEW -- THE README WAS THE LARGEST SURVIVING INSTANCE OF ROOT PATTERN (a).** Audited claim-by-claim 2026-07-14; full diagnosis in `docs/audits/README_AUDIT_2026-07-14.md`. It stated the **FEATURE COUNT in NINE places with FOUR different values** (80 x6, 78, 79) against a true contract of 95; the **TEST COUNT in THREE places with THREE values** (862 in a badge, 501/501 twice) against a true 1,926 passing; and that the message-bus suite passed on **Python 3.14.3** in one paragraph and **3.12.10** in another (the project runs 3.11/3.12; 3.14 is the version under which `requirements.txt` was mis-compiled, silently dropping the entire torch stack -- 6.18). It listed **HGMD Professional as an integrated data source in three places** -- never licensed, never wired, constant zero for the life of the project. Its training quickstart used **`--parquet`, A FLAG THAT HAS NEVER EXISTED** (the script takes `--clinvar`): anyone who copied the documented way to train this model got an argparse error. It pointed at four files that do not exist (`scripts/benchmark.py`, `models/phase2_pipeline.joblib`, `models/registry.json`, `models/drift_reference.pkl`). **TWO CLAIMS WERE NOT MERELY STALE BUT SCIENTIFICALLY FALSE:** (i) *"Connector silent-zero hardening -- regression tests assert that connector fallbacks fail loud, not silently return 0.0"* is the **exact opposite of the truth** and names the precise safeguard whose absence produced 6.21 (36 of 78 features constant zero in Run 15); (ii) *"empirically calibrated probability thresholds"* -- `models/classification_thresholds.json` **does not exist**, so every ACMG tier boundary the API serves is a hard-coded default, and `_load_thresholds()` swallows a malformed calibration file behind a bare **`except Exception: pass`** (its own docstring says "silently"). **THE BASE-MODEL ROSTER SAID TWELVE; IT IS THIRTEEN** -- the list omitted `svm` and `svm_bagged_rbf` (-2) and counted the Graph Attention Network as a base classifier (+1), which it is not (it yields `gnn_score`, a FEATURE, and no out-of-fold column). **That undercount is a disabled alarm:** 6.6a is the defect in which a 13-model ensemble silently became a 12-model ensemble, and anyone checking a run's twelve models against a README saying twelve would have concluded the ensemble was complete. **A RETRACTION IS RECORDED IN THE AUDIT:** the first draft filed the "five-tier ACMG classification" claim as FALSE on the grounds that training is binary. That conclusion was WRONG and is withdrawn -- the five tiers are a deliberate design (`api/schemas.py:343-353`), training on confident labels and recovering the ordinal scale from a calibrated probability is standard practice, and the auditor reached the wrong conclusion by reading ONE threshold band and generalising. The skim-and-conclude failure, committed inside the audit. | **CLOSED 2026-07-14.** All performance figures **WITHDRAWN** at the owner's instruction -- every one came from a run whose feature space was 46% non-existent, and the per-model comparison table ranked twelve algorithms against each other on it (a cross-algorithm comparison is precisely the artefact a half-empty feature space invalidates: model families degrade differently under missing signal). The figures are **not restated even in the withdrawal notice** -- a number in a warning banner is still a number people screenshot. `tests/unit/test_readme_claims.py` (8 tests) now binds the README to the code: the feature count at every claim site, the group table's SUM, the test count against `tests/EXPECTED_SUITE_SIZE`, the **base-model roster read from a LIVE `VariantEnsemble` instance**, the drift-monitor exit codes (including 4 = NOT CHECKED, and that it still exists in the script), HGMD's absence, every quickstart flag against the script that actually defines it, and a **ban on any performance-shaped figure reappearing**. Two of those tests FAILED on first run and **both failures were the TEST's fault** -- the feature-count sweep matched historical prose ("36 of its 78 features"), and the flag check attributed `--port` (uvicorn) and five `run_drift_monitor.py` flags to `run_phase2_eval.py` because it filtered at code-BLOCK level when the quickstart is one block holding four commands. Both rebuilt. |
| 6.24 | **NEW -- 181 CONSOLE STRINGS IN 41 FILES CARRIED NON-ASCII, AND THE PROJECT HAD ALREADY WRITTEN THE FIX DOWN AND NOT DONE IT.** Found 2026-07-14. `logger.*` / `print()` / `warnings.warn()` / `raise` were passed 73 em-dashes, 31 arrows, 13 box-drawing characters, plus emoji, Greek letters and ellipses. On a Windows cp1252 console these render as **mojibake**. **THIS WAS OBSERVED, RECORDED, AND IGNORED:** `docs/runs/RUN_16_results.md:84` -- *"Cosmetic: Reactome warning string uses an em-dash that mojibakes in PowerShell -- switch to ASCII `--` in reactome.py."* **`reactome.py` still carried the em-dash EIGHT WEEKS LATER.** A finding in a document is a comment. **Reconfiguring stdout to UTF-8 does NOT fix it** -- `scripts/train.py:61` already does exactly that; Python then writes correct UTF-8 *bytes* and a cp1252 console *decodes* them as cp1252. Fixing it that way needs `chcp 65001` on every machine, in every terminal, forever: **a fix whose correctness depends on the terminal's code page is root pattern (d)**, the same shape as the `sed`-patched `imodelsx` that silently dropped the Kolmogorov-Arnold Network from every Continuous Integration run for two months. The project had ALREADY made this decision -- `tests/unit/test_variant_ensemble_ascii.py` asserts ASCII-only log messages -- and enforced it on **exactly ONE file out of the 41 that violated it**. | **CLOSED 2026-07-14.** `scripts/maintenance/fix_console_ascii.py` (Abstract-Syntax-Tree-precise; docstrings, comments and FILE-bound strings such as the HTML report, the chi-squared markdown and the OpenAPI descriptions are deliberately untouched -- those declare their own encoding and no console is involved) + `tests/unit/test_console_strings_are_ascii.py` (2 tests, incl. a **negative test that plants offenders** and asserts the scanner fires on the `logger`/`print`/`raise` lines and NOT on the docstring, the file-bound HTML, or a never-printed constant). **THREE DEFECTS WERE FOUND IN THE REPAIR TOOL ITSELF, and each is the lesson: (i) BYTE-OFFSET SLICING** -- `ast` col_offset is a UTF-8 **byte** offset, not a character offset; the first rewriter sliced a Python `str` with it. An em-dash is one character but three bytes, so every offset after a non-ASCII character on the line shifted by +2 per character. **The bug is invisible on pure-ASCII lines -- i.e. on exactly the lines that do not need fixing** -- and it had already written to 36 files before it was caught. Fixed: encode, slice by byte, decode. **(ii) NO STRUCTURAL PROOF** -- the rewriter checked only that the result still PARSED, and a mis-spliced edit can be perfectly valid Python. Added `_structural_shape()`, which compares both trees with every string value blanked, proving the only thing that changed is string content. **(iii) SPLIT BRAIN** -- the gate was widened to catch INDIRECT strings (`_DIVIDER = "="*60` -> `logger.info(_DIVIDER)`) and the fixer was not, so the gate flagged 21 strings the repair tool could not repair. A repair tool that cannot fix what its own gate reports turns a red test into a manual chore, and manual chores are what rot. **ONE detector, shared by both.** VERIFIED: 181 strings, 41 files, **174 insertions / 174 deletions -- exactly balanced**, so no line was added or lost and only string contents changed. Gate green. Also fixed as a side effect: `evaluator.py:574` mixed ASCII `-` with box-drawing in the SAME table rule, rendering a broken separator even on a perfect UTF-8 terminal. |
| 6.25 | **NEW -- THE LOCKFILE GATE WAS BROKEN IN FIVE WAYS AT ONCE, AND THE SCRIPT IT CALLS WAS NEVER COMMITTED.** Found 2026-07-14/15. (a) `requirements-api.lock` was **compiled on Windows/Python 3.12 and deployed to Linux/3.11**: it carried `colorama==0.4.6` (a Windows-only transitive of `click`) and **lacked `uvloop`** (the Linux performance loop `uvicorn[standard]` pulls in) -- so the file that pins the production inference image had never described the production inference image. Regenerated **on Linux under 3.11**: `colorama` gone, `uvloop==0.22.1` present, `nvidia-nccl-cu12==2.30.7 # via xgboost` **newly appearing** -- and every other version **identical**, because `pip-compile` preserves the pins of an existing output file. (b) The Continuous Integration job's own regeneration step compiled to `/tmp/regenerated.lock` -- **a path that does not exist** -- so `pip-compile` had no pins to preserve and resolved everything to latest; **22 packages drifted within minutes**, and the gate reported drift it had itself created. (c) It installed `pip-tools` **unpinned**, so the gate's own tool was a moving target. (d) It compared **byte-identity**, which platform markers (`sys_platform == "win32"`) make unachievable across runners by construction. (e) **`scripts/check_lock_satisfies.py` -- the replacement gate -- WAS UNTRACKED while `.github/workflows/ci.yml:85` invoked it.** Every local run passed because the file exists in the working tree; the first push would have failed the job on a missing file. **Root pattern (d), and the fourth instance this month: the green was evidence about the environment, not the code.** **AND THE CASCADE:** `test` and `drift` both declared `needs: lockfile-check`, so a red lockfile job **skipped the entire pytest suite** -- 1,936 tests silently not run on Linux, reported as "skipped" rather than "unknown". | **CLOSED 2026-07-15** (`a889684`). `check_lock_satisfies.py` committed and pinned into the workflow; the gate now asserts **SATISFACTION** (every requirement resolvable from the lock at the pinned version), not byte-identity, plus a **platform gate** that fails if `colorama` reappears or `uvloop` vanishes. Cascade broken: `test` no longer `needs: lockfile-check`; `docker-build` needs both. **KNOWN GAP, recorded rather than hidden:** the checker compares `Requirement.name` + `.specifier` and **ignores `.extras`** -- `uvicorn` and `uvicorn[standard]` compare equal. Extras are an environment MARKER, not part of the name; closing it needs marker evaluation, and it is 6.18's territory. |
| 6.26 | **NEW -- THE README SAID 13 AGENTS. THERE ARE 22. AND `run_agents.py` HELD A SECOND COPY OF THE ORCHESTRATOR.** Found 2026-07-15. **(a) THE AGENT COUNT.** A 41% undercount of the supervisory layer the document calls the system's defining feature, stated in **SIX** places and wrong in all six. **EIGHT of the thirteen names it gave were wrong** -- it listed `SchemaDriftAgent`, `ConceptDriftAgent`, `LabelShiftAgent`, `CalibrationDriftAgent`, `InfrastructureDriftAgent`, `FairnessSubgroupAgent`, `AdversarialSubmissionAgent`, `AnnotationPolicyAgent`; every real class is a `...MonitorAgent`. **NINE agents were entirely undocumented:** `AdaptationAgent`, `AgentOpsMonitorAgent`, `DataReadinessAgent`, `DatabaseFreshnessMonitorAgent`, `FeatureCoverageSentinelMonitorAgent`, `FinOpsAdvisorAgent`, `ModelInsightsAgent`, `ProvisioningAgent`, `ReclassificationSentinelMonitorAgent`. `SchemaDriftAgent` was the most dangerous wrong name because **a class by that name EXISTS** -- it is the schema-drift DETECTOR, does not descend from `BaseAgent`, and is not in the registry: anyone auditing the agent layer against that table would have found the wrong object and stopped. **THE 2026-07-14 AUDIT FOUND THIS AND FILED IT "UNRESOLVED"** (its §4.2: *"Whether all thirteen are scheduled and running is UNRESOLVED"*) -- while `scripts/check_agents_active.py`, **named in a comment inside the registry being read** (`orchestrator.py:160`), answers it in one command: *"22 agents (registered=22, scheduled=22) ... 0 dormant"*. A finding in a document is a comment. **(b) `run_agents.py` IMPORTED THE ORCHESTRATOR TWICE, TWO LINES APART** -- once bare (behind `sys.path.insert(0, agent_layer/)`) and once by full package path. Python keys `sys.modules` by import path, so those are **two distinct module objects**: two `Orchestrator` CLASSES (`isinstance` then depends on which path built the object), two `PIPELINE_DEFINITIONS`, two copies of every `_Lazy` cache. The `sys.path.insert` also shadowed the standard library process-wide. The 2026-07-01 remediation was written to eliminate exactly this, reported *"4 repointed"*, and **missed this one** -- while its own post-condition check still sits in `Install_unit2_reactivate_message_bus.ps1:111`. The check existed, was correct, and was never run against this file. **THREE FAILURES IN BUILDING THE FIX, each recorded because each is the lesson: (i)** the first Abstract-Syntax-Tree scan returned **13** -- exactly the README's wrong number -- because it filtered bases on `"Agent" in base_name` and therefore missed every agent inheriting `DriftMonitorBase`. A malformed search returned the wrong answer **and made it look confirmed**. **(ii)** the second scan returned **18**, by regexing quoted keys at module level and sweeping up `full`, `drift`, `ts`, `status`, `duration_ms`, `error`. The registry is an INSTANCE attribute built inside a method. **(iii)** the tests failed on `ImportError: cannot import name 'AgentOrchestrator'`. **The class is `Orchestrator`.** The name was assumed, never checked, then propagated into **NINE places including TWO README lines** -- briefly documenting a class that does not exist. Twenty other files import it correctly; not one was read. | **CLOSED 2026-07-15** (`184f2c6`, `f078953`). Bare import and `sys.path` hack deleted (`import sys` retained -- `sys.exit()` is called twelve times, and removing it with the hack would have turned every command-line exit into a `NameError`; blast radius measured, not assumed). README corrected at all six sites with the real roster; `tests/unit/test_readme_claims.py` reads **both** rosters from a LIVE instance (`Orchestrator.__new__` + `_register_agents()`, no `__init__` side effects) and enumerates every claim site. **STILL OPEN:** every agent reports `age=25.06d` -- last orchestrator run **2026-06-20** -- while `check_agents_active.py` calls them ACTIVE with 0 dormant. A liveness checker that calls a 25-day silence ACTIVE is measuring a proxy; its threshold has not been read. |
| 6.27 | **NEW -- `genomiclm_llr` HAS BEEN IDENTICALLY 0.0 FOR ALL 4,420,180 COHORT ROWS SINCE THE CONNECTOR WAS WRITTEN.** Found 2026-07-15. `genomic_lm._masked_centre_logratio` located the variant's centre token with `off = tok(win, return_offsets_mapping=True).get("offset_mapping")`. **Nucleotide Transformer's tokeniser is `EsmTokenizer`, and `is_fast` is False** -- a pure-Python ("slow") tokeniser. HuggingFace raises `NotImplementedError` for that argument on **every** slow tokeniser; only `PreTrainedTokenizerFast` supports it. It raised on **every window**. A bare `except Exception` two frames up swallowed it into `logger.debug` -- **below the default level, so it printed NOTHING** -- and carried `# pragma: no cover`, so the coverage tool was told not to look either. Measured on the owner's box: `is_fast: False | class: EsmTokenizer` / `OFFSET RAISES -> NotImplementedError`. **THREE INDEPENDENT BLIND SPOTS KEPT IT GREEN.** (i) `genomiclm_delta_norm` never touches offset mapping, so the sibling feature stayed **ALIVE** and the connector looked healthy from outside. (ii) `build_reference_slice` **FEEDS** `genomiclm_llr` a synthetic `rng.uniform(-12, 4)`, and `engineer_features` reads it as a plain `df.get` passthrough -- so the stage-5 zero-audit graded the **FIXTURE** and never invoked the connector (**root pattern (c)**). (iii) The harness comment asserted all six new columns were *"live connectors -- Run-17 real-data smoke shows them populated"*; **the smoke audit it CITED records `genomiclm_llr` DEAD IN ALL SPLITS.** The comment contradicted its own evidence, and the 2026-07-11 handoff had warned in bold: *"if any SHOULD populate on the fixture and doesn't, that specific one is a real regression, not an allowlist gap."* It was. **AND A HYPOTHESIS WAS REFUTED BEFORE IT WAS SHIPPED:** the first diagnosis was that `mask_token_id is None` returned zeros. Measured: `mask_token: <mask> | mask_token_id: 2`. **That branch never fires.** Had it been "fixed", the real cause would have been papered over and declared closed. | **CLOSED 2026-07-15** (`8d7df86`). `centre_token_index()` reconstructs offsets from the **token strings the tokeniser actually emitted** -- assuming nothing about k-mer size, remainder handling, or special-token count, and **ASSERTING** the round-trip rather than trusting it (a silent mislocation would score the WRONG BASE for every variant: worse than zeros, because zeros are visibly dead). Handler and pragma deleted; an all-unlocatable result **RAISES**. 12 tests, no network, including `ast`-level tripwires banning `return_offsets_mapping` from any CALL (while permitting the docstrings that explain this incident), banning exception handlers in the LLR path, and banning `pragma: no cover`. **Left alone, this would have been caught -- by hard-failing Run 17 at `_assert_no_dead_features`, after full data preparation, on the rented graphics-processing-unit, on paid compute.** A true gate firing in the most expensive place available. |
| 6.28 | **NEW -- 21,814 COHORT ROWS CARRY FABRICATED SEQUENCE, AND FOUR SEPARATE DETECTORS WERE BLIND TO ALL OF THEM.** Found 2026-07-15. The live artifact's manifest is unambiguous: `n_rows_built 4420180 / n_ok 4398366 / n_poly 21814`, with `poly_reason_breakdown` = `empty_allele 19988, non_acgt_allele 1771, ref_mismatch 53, fetch_failed 2` (sums exactly). **91.6% are alleleless variants** -- a rebuild target with an existing workstream, not noise. **TWO FABRICATION PATHS EXIST WITH DIFFERENT FILL CHARACTERS:** `delta_window_builder.POLY = "N"` is written INTO the parquet and flagged `ok=False`; `seq_window_join`'s own fallback used `"A"`. **Every consumer detected only the second.** `train.py:485` (`_POLY_WIN`), `genomic_lm:201/250` (`self._poly` + `_mapped_mask`), the join's own fallback, and `rekey_seq_windows_v2:146` -- **the last of which gates a WRITE** (`return 6`, refuses to publish). Four detectors, one blind spot, 21,814 rows through it. They reached `cnn_1d` as an **ALL-ZERO** tensor with `ref == alt` (delta channels identically zero) and reached Nucleotide Transformer as `||alt_emb - ref_emb|| == exactly 0.0` -- the value `genomic_lm`'s own docstring defines as *"window unavailable / model unavailable"*. **The EXIT_NOT_CHECKED bug again (6.20): "I could not look" rendered identically to "I looked and found zero".** ok-fraction 99.507% clears `MIN_OK_FRACTION` (0.95), so no gate fired. **WIDENING THE CHECKS TO ALSO MATCH "N"*101 WOULD HAVE BEEN PATCHWORK ON AN ERROR OF PRINCIPLE: a window reading "A"*101 MAY BE REAL** -- poly-A tracts are real biology, and content can never separate *"the reference genuinely says A"* from *"we gave up and typed A"*. Only PROVENANCE can, and `build_seq_windows.py:154` had been writing an `ok` column into the artifact **the entire time**. The ground truth was in the artifact; the join never read it. **A FIFTH FABRICATOR SAT INSIDE A MODEL:** `CNN1DClassifier._encode_batch` opened with `win = "A" * self.window` and `.fillna(win)`d every input -- the only site that MANUFACTURED windows rather than merely mis-detecting them. Its `fasta_seq` branch read a column **measured at 100% NULL across all 4,420,180 rows** and set `ref = alt`, making 4 of 13 channels dead and 8 duplicated. **AND THE `X_seq: pd.Series` ANNOTATION ON `fit`/`predict_proba`/`evaluate` WAS FALSE** -- `scripts/train.py` has always passed a DataFrame (`_att_train.windows`); every test passed a Series **because the signature said so**. The tolerant adapter let the two disagree, so the suite was green for years on a code path **the run never executes**, for the sequence model's only input. | **CLOSED 2026-07-15** (`9157133` + follow-on). `attach_delta_windows` returns a `WindowAttachment` with an explicit `usable` mask read from the builder's `ok` column; `_mapped_mask`, `self._poly` and `_POLY_WIN` deleted. Tier 1's hardcoded `return out, 0` fixed -- it reported ZERO unmapped while `.fillna` fabricated a window for every null row, and **a test asserted exactly that**, its own comment naming the fabrication approvingly (*"# NaN -> poly-A fill"*). `_encode_batch` is strict by default and **RAISES** on a Series, a bare column, a null, or `fasta_seq`; an opt-in `single_sequence_mode=True` gives an HONEST 5-channel reference-only model (verified: `_in_channels()` 13 vs 5, shape `(4, 5, 101)`, flag in `get_params()`) rather than 13 channels of which 9 are redundant -- the mode was never the problem, **choosing it by accident was**. `train.py`'s `has_sequences` now reads the **TRAIN** split (it read TEST, while `cnn_1d` is FITTED on train) and warns when train/test coverage diverges >5%. `run_phase2_eval`'s 0.5% abort gate now counts placeholders. **`tests/unit/test_no_content_based_poly_detection.py`** bans the whole class repo-wide. **THE BAN HAD THE DEFECT IT WAS BUILT TO PREVENT:** its first version matched only `Constant * X`, so `sw.PAD_CHAR * sw.WINDOW` walked through -- and its `_ALLOWED` entry **described that blind spot as a reassurance**. Fixed to resolve names. On its first clean run it found **six byte-order-mark source files** and an invalid `'\M'` escape that had been reported against `<unknown>` until `filename=` was passed to `ast.parse`. **THREE GATES WERE SILENTLY DISARMED BY THE FIX ITSELF AND CAUGHT ONLY BY TESTS:** rekey's write gate, `train.py`'s `has_sequences`, and `_encode_batch` -- each a content check that, when the filler changed, did not start FAILING but started **PASSING UNCONDITIONALLY**. Content checks do not fail loudly when they rot; they go quiet. **STILL OPEN:** every Run 17 launcher points `--seq-windows` at `clinvar_grch38_clean_seq.parquet`, which has **no `ok` column** -- so this entire mechanism is INERT on the configured path. See 6.29. |
| 6.29 | **CORRECTED 2026-07-15, SAME DAY, BEFORE ANY ACTION WAS TAKEN ON IT.** **The original entry said "every Run 17 launcher points at the STALE artifact." That framing is UNSUPPORTED and probably BACKWARDS.** It was written without reading `memory.md`, which records the standing rule: **"Clean cohort is required: `clinvar_grch38.parquet` raw must go through `clean_cohort.py --apply` before use; null ref/alt rows cause gate failures and are scientifically incorrect"** -- and the Stage-1 capstone smoke failure whose root cause was *"`clinvar_grch38.parquet` retains **21,091** structural/CNV rows with null/empty ref/alt"*. **4,420,180 − 4,399,089 = 21,091, exactly.** So `clinvar_grch38_clean.parquet` / `clean_seq.parquet` are the **CLEANED** cohort and the launchers naming them are following the standing rule; `seq_windows/seq_windows.parquet` was built from **`pathfix`, which still contains those rows** -- and its own `poly_reason_breakdown` reports **`empty_allele: 19,988`**, which are those very rows failing to yield a window. **The artifact carrying provenance may be the one built on the WRONG cohort.** THIS IS A HYPOTHESIS. It is recorded as one because the rule it violates -- *"label causal claims as hypotheses until verified via filesystem/grep/git"* -- is the rule that produced the error. **VERIFY BEFORE ACTING:** does `clean_seq.parquet`'s 4,399,089 == `clean.parquet`'s 4,399,089 == pathfix minus the alleleless rows? Does `clean_cohort.py --apply` remove exactly 21,091? | **THE SUBSTANTIVE FINDINGS BELOW STAND; THE FRAMING DOES NOT.** What remains measured and true regardless: |
| 6.29a | **`--seq-windows` MEANS TWO DIFFERENT THINGS IN TWO SCRIPTS.** Found 2026-07-15. `train.py:102-107` declares `--seq-windows` a **DIRECTORY** (default `data/processed/seq_windows`) and appends `/seq_windows.parquet`. `run_phase2_eval.py:49-52` declares it a **FILE** (*"Parquet with fasta_seq_ref/fasta_seq_alt delta windows"*) and passes it straight to `attach_delta_windows`. One flag, one repository, opposite contracts. `launch_run17_baseline.sh:304` invokes `run_phase2_eval.py`, so the launcher is **self-consistent** -- but it names `clinvar_grch38_clean_seq.parquet` (4,399,089 rows, 6/12/2026, **19 columns, NO `ok`**) rather than `seq_windows/seq_windows.parquet` (4,420,180 rows, 7/10/2026, **8 columns WITH `ok` + `reason`**). **The two artifacts are 21,091 rows and one month apart, and only one carries provenance.** So `attach_delta_windows` warns *"no 'ok' column: builder-placeholder rows CANNOT be identified"*, `usable` degrades to `notna()`, and 6.28's entire mechanism is blind on the path Run 17 will actually run. `clinvar_grch38_clean.parquet` (the `--clinvar` target) carries **no** `fasta_seq_ref`/`fasta_seq_alt`, so tier 1 cannot rescue it. **AND TWO GATES CANNOT SEE ANY OF THIS:** `preflight_gate.py:91` fails only when `--seq-windows` is **EMPTY** -- a non-empty path to a stale artifact sails through, in a check named *"silent CNN degradation"*; and `test_launch_run17.py::test_required_flags_present[--seq-windows ]` is parametrised on the **flag string** and asserts only that the substring appears. **Root pattern (c), twice, guarding the same branch.** | **OPEN.** Fix converges on: one meaning for `--seq-windows` (the directory semantics, since `verify_seq_windows` needs the manifest beside the parquet), all four launchers + `smoke_all_models.py` + `preflight_gate.py` repointed at `data/processed/seq_windows`, and `test_launch_run17.py` rewritten to assert the value **RESOLVES to an artifact carrying `ok`** -- not that the flag exists. **BLOCKED ON ONE UNKNOWN:** whether `run_phase2_eval.py` can consume the new artifact's 8-column schema. **NOT TESTED.** Also unresolved: how many of the 21,814 placeholder rows (keyed to the 4,420,180-row `pathfix` cohort) even EXIST in `clean_seq`'s 4,399,089 -- so the 0.4935%-vs-0.5% razor in `run_phase2_eval`'s abort gate may not transfer. **An inference resting on an inference; do not build on it.** Full evidence separation: `docs/status/STATUS_2026-07-15_evidence-separated.md`. |
| 6.7 | `±inf` input raises a raw pandas `IntCastingNaNError`. Fail-loud is right; a pandas internal as the message is not. | OPEN |
| 6.8 | 62 orphan scripts + 7 doc-only, unclassified -- they **blocked the G1 gate** on 2026-07-12. | **CLOSED** (`0b93d30`, `5924092`). 68 archived to `scripts/forensics/` with a README; provenance trail repaired. |
| 6.9 | Guards (`sys.path`, `data/` pollution, G1 §13c) have **no permanent self-test**. All three were negative-tested by hand on 2026-07-12 and all three fire correctly -- but a guard nobody re-tests can die silently, which is exactly how seven AlphaFold tests stayed dead for weeks. | OPEN |
| 6.10 | Repo authority: `monzia-moodie` and `monzia-moodie-repo-projects` resolve to the **same** repository. A GitHub-side transfer, not a git command. Open **pull request #1** (`run9a-prep`). | OPEN |
| 6.11 | **JEPA** (Joint-Embedding Predictive Architecture) -- tracked item, not started. | OPEN |
| 6.12 | **NEW.** Disk: 8.73 GB reclaimed by deleting `data/raw/cache/alphafold` (36,073 files) -- possible for the first time, because the seven tests that silently keyed off it now use a committed 101 KB fixture. Free space 5.4 GB → 14.74 GB. Still below the 20 GB G1 recommends. | PARTLY CLOSED |

## 6A. CURRENT STATE SNAPSHOT (2026-07-15) -- SUPERSEDES §5 (2026-07-12)

**Everything here is MEASURED. The command is named. Re-run it; do not believe it.**
Full evidence separation, including what is inference and what is unknown:
`docs/status/STATUS_2026-07-15_evidence-separated.md`.

### Contract, roster, suite

| thing | value | how it was measured |
|---|---|---|
| Tabular features | **95** | `EXPECTED_TABULAR_FEATURE_COUNT`, enforced against `TABULAR_FEATURES` at import |
| Base models | **13** | live `VariantEnsemble.base_estimators` |
| Agents | **22** | `check_agents_active.py`: *"22 agents (registered=22, scheduled=22) ... 0 dormant"*; corroborated by AST inheritance scan and a live `_register_agents()` |
| Tests | **1966 collected** | `pytest tests/ --collect-only -q` -> "1966 tests collected in 16.76s", exit 0 |
| Suite status | **9 FAILING** | 7 from the `_encode_batch` contract change (all pass a `pd.Series`, per the false annotation); 2 from the poly ban reporting real findings |
| Cohort | **4,420,180 rows / 28,350 genes** | `clinvar_grch38_pathfix.parquet`; `clean`/`clean_seq` are **4,399,089** -- 21,091 fewer |
| Labels | uncertain 2,718,963 (61.5%) / likely_benign 1,083,576 / benign 276,240 / pathogenic 229,602 / likely_pathogenic 111,799 | `value_counts` on `pathogenicity`; sums to 4,420,180 |
| Trainable (excl. uncertain) | **1,701,217** -- 341,401 pos / 1,359,816 neg (1:3.98) | derived from the above |
| Seq windows | 4,398,366 ok / **21,814 poly** (0.494%) | `seq_windows.manifest.json`; reasons: empty_allele 19,988, non_acgt 1,771, ref_mismatch 53, fetch_failed 2 |
| Disk | **6.62 GB free of 935.59 GB**, ONE physical disk | `Get-Disk` -> Number 0 only; `Get-Volume` |
| Last agent run | **2026-06-20 (`age=25.06d`)** | reported by every agent, while the checker calls them ACTIVE |
| Last sealed run | **Run 15** (`032a2ab`, 2026-06-09) | Run 16 was an expensive failure; Run 17 not launched |

### The four architectures -- HONEST STATUS

**None of the four is complete. Run 17 is gated behind all of them.**

| architecture | state | evidence |
|---|---|---|
| **JEPA** | **NOT STARTED** | `jepa` appears in **two files, both prose** (this roadmap, one remediation doc). Zero code. No self-supervised or pretraining code exists anywhere: `pretrain`, `self_supervised`, `ssl`, `embedding_dim` all return zero hits in `src/`. |
| **Conformal prediction** | **PHASE 1 DONE, PHASE 2 PARTIAL** | `conformal/` holds SIX modules: `scores`, `split`, `calibrate`, `coverage`, `grouped`, `mondrian`. 28 tests. LAC matches MAPIE **element-wise exactly**; APS confirmed a valid tighter variant. **MISSING** against the design (§12 of the conformal specification): RAPS, ordinal conformal risk control, multilabel, gene-ranking CRC, `risk_control`, `artifacts` (provenance/fail-closed), `config`, `subgroup` policy, `evaluation`, `monitoring`. That is **6 of 14 modules**. |
| **Expanded metric stack** | **PARTIAL** | `evaluation/metrics.py` (30 tests) has AUROC, AUPRC + lift-over-floor, Brier, Expected Calibration Error, calibration slope/intercept, bootstrap confidence intervals, stratified evaluation. The required panel (§16 of the conformal specification) additionally demands: validity (per-class + worst-group coverage, coverage gap), efficiency (set-size distribution, singleton/doubleton/empty rates), clinical behaviour (pathogenic-exclusion rate, severe-error rate, deferral burden, PPV/NPV among singletons), selective prediction (risk-at-coverage, AURC), multimodal robustness, and shift robustness. |
| **RNA** | **EXPRESSION ONLY -- NO RNA FOUNDATION MODEL** | `data/rnaseq.py` (5 features) and `data/gtex.py` (6) supply **expression**. `pipelines/rna_pipeline.py` supplies splice mechanics. **There is no RNA sequence foundation model and no transcriptomic foundation model in this codebase** -- no RNA-FM, RiNALMo, ERNIE-RNA, RNAErnie, Uni-RNA, UTR-LM, RNABERT; no scGPT, Geneformer, scFoundation, BMFM-RNA. |

### Foundation-model layer -- what is actually present

| biological layer | model | state |
|---|---|---|
| DNA sequence | Nucleotide Transformer (`genomic_lm.py`) | **WIRED.** `genomiclm_delta_norm` alive; `genomiclm_llr` was dead for the connector's entire life -- **fixed 2026-07-15 (6.27)**, not yet verified on real data. |
| Protein sequence | ESM-2 (`esm2.py`) | **WIRED.** `esm2_delta_norm` + `esm2_llr`. |
| Protein structure | AlphaFold (`alphafold.py`) | **WIRED.** 4 features. |
| Networks | Graph Attention Network + hetero-KG (`gnn.py`, `hetero_gnn.py`) | **WIRED.** `gnn_score`, `hetero_gnn_score`. |
| **RNA sequence** | — | **ABSENT** |
| **RNA structure** | — | **ABSENT** |
| **Transcriptomics** | — | **ABSENT** (expression matrices are consumed as scalar features, not as a foundation model) |
| DNA long-context | — | **ABSENT** (HyenaDNA / Caduceus not evaluated) |

**ALSO FOUND 2026-07-15, UNDOCUMENTED:** `data/primateai3d.py` EXISTS as a connector and **no PrimateAI-3D feature appears in `TABULAR_FEATURES`.** A dormant connector nobody has recorded -- the same shape as HGMD before 6.21a. Disposition it: wire it, or record why not.

---

## 6B. FORWARD PLAN -- what must land before Run 17, and in what order

**Run 17 is NOT the next step, and no smoke test is due.** Monzia's requirement, 2026-07-15:
JEPA, conformal prediction, the full expanded metric stack, and the RNA architecture must be
**fully incorporated, and rigorously checked, evaluated, tested, validated and verified**,
with evidence, before a smoke test is even discussed.

This section is a **plan, not a promise**. It is expected to change. Items may move earlier
or later. What must NOT change is that each one lands with evidence rather than assertion.

### P0 -- integrity work that blocks everything downstream

1. **6.29 -- `--seq-windows` means two things, and every launcher points at the artifact
   without provenance.** Until this is closed, every window-provenance guard built on
   2026-07-15 is INERT on the configured path. **Blocked on one untested unknown:** can
   `run_phase2_eval.py` consume `seq_windows.parquet`'s 8-column schema?
2. **The 9 failing tests.** 7 are the `_encode_batch` contract change (the fixtures pass a
   `Series`, exactly as the false `X_seq: pd.Series` annotation instructed); 2 are the poly
   ban reporting six byte-order-mark files and an invalid escape sequence.
3. **`maxentscan_delta` may be the next `genomiclm_llr`.** `rna_pipeline.py:340` does
   `ref_col = alt_col = df["fasta_seq"].fillna("")` -- `ref == alt`, reading a column
   **measured at 100% NULL across all 4,420,180 rows** -- and the harness fixture feeds it.
   That is 6.27's exact shape. **NOT VERIFIED.** One command against a real engineered
   matrix settles it. **Highest-value open lead in the project.**
4. **`primateai3d.py`** -- dormant connector, undocumented (above).
5. **Disk.** 6.62 GB free. ~78 GB reclaimable locally (Docker WSL disk 72.46 GB; two
   backups still holding the AlphaFold blob excised from git history). ~196 GB of gnomAD
   VCFs in `C:\Users\monzi\data\` are the Google Drive archive target -- **`chr17` appears
   TWICE**, so that starts with de-duplication, not upload. Google Drive is **archive
   only**: never the hot training path (CLAUDE.md §4.1).

### P1 -- RNA architecture

Per the 2026-07-15 specification, RNA needs **both** categories; they solve different
problems and are complementary:

* **RNA sequence foundation model.** Priority **RiNALMo** (state of the art across
  downstream tasks, strong generalisation to unseen RNA families -- which is the property
  this project needs, since the splits are gene-disjoint by construction) with **RNA-FM**
  as the mature generalist baseline. **ERNIE-RNA** is the structural complement and is
  especially relevant here, because this project's stated emphasis includes variant effects
  on RNA structure and splicing.
* **Transcriptomic foundation model.** **Geneformer** or **scGPT**, depending on whether
  the emphasis is gene regulation or single-cell expression. These learn from expression
  matrices, not nucleotides -- so they are additive to, not a replacement for, `rnaseq_*`
  and `gtex_*`.
* **Specialised, if the biology demands it:** UTR-LM (5' untranslated region biology,
  translation efficiency, ribosome loading).

**The engineering constraint learned from ESM-2 and Nucleotide Transformer, and it is not
optional:** every foundation model added so far has been **collapsed to two scalars** (an
L2 delta norm and a log-likelihood ratio) before it touches the feature contract, and one
of those scalars was **silently zero for the connector's entire life** (6.27). A third and
fourth foundation model wired the same way will fail the same way. Each RNA model must land
with: a fail-loud connector (no bare `except`), a real test of the connector rather than of
a fixture, and provenance recorded in the run artifacts.

### P2 -- conformal prediction, completed

Phase 1 (binary label-conditional / Mondrian) is **done and cross-checked against MAPIE**.
Remaining, in the specification's order:

* **Phase 2** -- RAPS; ordinal contiguous prediction sets over the five ACMG tiers;
  ordinal conformal risk control with a distance-weighted loss (an error from Pathogenic to
  Likely Pathogenic is not an error from Pathogenic to Benign); severe-error risk;
  VUS-deferral analysis. **61.5% of the cohort is `uncertain`** -- the deferral analysis is
  not a nicety, it is the majority case.
* **Phase 3** -- modality-signature-aware calibration and missing-modality strata.
* **Phase 4** -- adaptive top-K gene candidate sets with retrieval-failure risk control.
* **Phase 5** -- subgroup and temporal validation.
* **Phase 6** -- deployment monitoring of nonconformity/set-size/coverage drift.
* **Cross-cutting** -- `artifacts.py` provenance with **fail-closed** inference (model hash,
  class order, feature schema, calibration cohort size), and `config.py` for alpha levels,
  subgroup policy, and severe-error weights.

### P3 -- the expanded metric panel

Build the panel from §16 of the specification, for **every task and every alpha**: validity
(marginal, per-class, worst-group, coverage gap, bootstrap confidence intervals),
efficiency (mean/median/p90/p95 set size, singleton/doubleton/empty rates), clinical
behaviour (pathogenic-exclusion, severe opposite-class error, deferral burden, PPV among
singleton-pathogenic calls), selective prediction (risk-at-coverage, AURC), and robustness
under missing modality and under shift.

### P4 -- JEPA

**Both design documents agree on the sequencing, and it is not the obvious one:**

> *"Do not start with JEPA until this supervised fusion branch is reproducible, calibrated,
> attribution-validated, and benchmarked."*

The order is: **expose embeddings → masked fusion v1 → validate attribution → benchmark
against the stacker → THEN add JEPA pretraining** (Stage 0 before Stage 1).

**Step 1 is the blocker, and it is a build, not a wiring job.** Every foundation-model
embedding is currently **destroyed on creation** -- `genomic_lm.py:284` computes a 512-dim
vector and reduces it to `np.linalg.norm(...)` on the next line; ESM-2, the Graph Attention
Network and the tabular network do the same. Fusion v1 needs those vectors, so an embedding
cache with version-keyed provenance (`source_model_name`, `checkpoint_hash`,
`embedding_dim`, `preprocessing_version`) is the first artifact.

**Arithmetic that constrains the design:** at the 4,420,180-row cohort, float16, ref+alt
only, Nucleotide Transformer + ESM-2 only, pooled only, the cache is **≈14.7 GB against
6.62 GB free**. Token-level retention -- which the design explicitly requires ("do not cache
only one pooled vector") -- is ~154 GB. **Scoping to the 1,701,217 trainable rows brings the
pooled cache to ≈5.7 GB.** This is a real constraint on a real disk and it decides the
design; it is not a detail to resolve later.

**Run 17's shape, if all of the above lands:** three arms in ONE run -- the 13-model
stacker, Fusion v1 (Stage 0, no JEPA objective), and JEPA-pretrained Fusion v1 (Stage 1) --
on identical folds and an identical gene-disjoint test set, so the comparison is **paired
and internally valid**. That satisfies both Monzia's requirement that JEPA be in Run 17 and
the documents' requirement that Stage 0 precede Stage 1: Stage 0 exists as a **trained arm**
rather than as a prior run.

### What could NOT be established on 2026-07-15, and why

* **The July chat sessions could not be read.** `list_sessions` returns exactly one session
  ("GitHub landing page updates"), which is not this project's history. That record is not
  reachable from the working environment.
* **`docs/sessions/` STOPS AT 2026-07-06.** Of 67 session files, only two are from July.
  **Nine days -- 07-07 through 07-15 -- have no session log**, and they cover the git
  history rewrite, both Continuous Integration failure classes, the drift revival, the
  Run-15 silent-zero discovery, the README audit, and everything since. Partial cover
  survives in `docs/status/REMEDIATION_2026-07-13*` and the 2026-07-11 handoff. **The
  discipline that exists precisely to prevent stale state broke down exactly when the state
  was changing fastest.** This entry is the record that it did.

### On "does the project execute flawlessly end-to-end?"

**No, and there is no evidence that it does.** Stating it plainly rather than producing a
reassuring document:

* **9 tests are failing.** A green suite is a precondition for any end-to-end claim.
* **6.29 is open** -- the launchers point at an artifact without provenance.
* **The end-to-end path has not been executed** in any form during this work. No smoke run,
  no `run_phase2_eval` invocation, no artifact produced.
* **Nine commits are unpushed**, so Continuous Integration has not seen any of it.
* **The last end-to-end evidence is Run 15** (2026-06-09) -- and 6.21 established that its
  feature space was **46% constant zero**, so it is evidence of execution, not of
  correctness.

An end-to-end verification is a **deliverable to be built**, not a status to be reported.
It requires, at minimum: a green suite, 6.29 closed, `maxentscan_delta` verified, a local
smoke run producing artifacts, and a feature census printed beside them proving every
declared feature carried information. **None of that has been done.**

---

---

## 6C. CURRENT STATE SNAPSHOT (2026-07-18) -- SUPERSEDES 6A (2026-07-15)

**Everything here is MEASURED, the command is named, and the TREE it was measured on is named.**
Session record: `docs/sessions/SESSION_2026-07-18_phase3b-window-builder-retirement.md`.
Commits: `e57835e` (Phase 3b), `9362f2c` (line-ending governance), `d1c2c4e` (session record), `8975b1a` (README test badge), `433d2e8` (this section).

### What 6A now states falsely

| 6A row | 6A said | MEASURED 2026-07-18 |
|---|---|---|
| Tests | 1966 collected | **1968 collected**, and the ratchet gate `--assert-suite-size` PASSES |
| Suite status | **9 FAILING** | **0 FAILING** -- 1,961 passed, 7 skipped, 630.21s |

The suite-status row is the one that mattered: it would have told a reader the tree was red when it is
green. Both rows were true when written. Neither was re-derived. That is section 7's root pattern (a),
and 6A is left intact above so that it says so.

Unchanged and re-confirmed from 6A: 95 tabular features, 13 base models, 22 agents.

### Phase 3b -- the superseded window builder is retired

Two window builders existed and both were live. `delta_window_builder.py` (surviving) writes
`POLY = "N" * 101` and records `ok=False` with a reason. `seq_windows.py` (retired) used
`PAD_CHAR = "A"` and wrote **no provenance column at all**.

`"A"` is a member of `encode_sequence`'s `BASES`, so every placeholder position one-hot-encoded to a
**confident adenine** -- a positive assertion about the genome that nobody made. `"N"` is absent from
`BASES` and encodes to an honest all-zero vector. And with no `ok` column, `attach_delta_windows` took
its `has_ok=False` branch and declared every row usable behind a logger warning nothing was reading.

**RETIRED** (after the replacement was built and tested, not before -- a retirement audit refused this
same deletion earlier the same day, correctly, because `scripts/populate_fasta_seq.py` was then the sole
repository-resident producer of a 534 MB artifact that eighteen files consume):

```
src/genomic_variant_classifier/data/seq_windows.py          197 lines, 7 poly-ban offenders
src/genomic_variant_classifier/data/populate_fasta_seq.py   221 lines
scripts/populate_fasta_seq.py                                85 lines
tests/unit/test_seq_windows.py                              155 lines, 16 tests
tests/unit/test_populate_fasta_seq.py                       154 lines,  5 tests
```

**REPLACED**: `scripts/build_clean_seq_from_windows.py` (252 lines, join-based producer, +9 tests) and
`tests/test_build_seq_windows.py` (+10 tests porting the surviving builder's coverage).

**FOUR BLIND DETECTORS MOVED FROM CONTENT TO PROVENANCE.** None was fixed by changing `"A"` to `"N"`:

| file | was | is |
|---|---|---|
| `scripts/preflight_run16_inputs.py` | **a LAUNCH GATE** reporting 100% real on a cohort with placeholders; sampled only the first 4,000 rows against a 50% threshold | reads the whole `ok` column; says "cannot tell" when provenance is absent |
| `scripts/probe_cohort_seq_density.py` | reported `dummy=0` when it could not know | distinguishes zero from cannot-tell |
| `correctness_harness.py` stage 3a | **a CORRECTNESS GATE, triply dead** -- wrong column (100% null), obsolete `"A"*101` constant, synthetic random-ACGT fixture; structurally incapable of firing | three outcomes: fail >50% unusable, warn when `ok` absent, quiet when all usable |
| `scripts/run9_ablations.py` | fed fabricated adenine to `cnn_1d` | centralised placeholder; `cnn_1d` excluded from ablations, since constant input cannot be learned from |

Poly-ban offenders **12 -> 0**. Placeholder construction is centralised in
`delta_window_builder.placeholder_window()`, so the literal exists in exactly one place. The left-edge
padding at lines 91/128/132 (`"N" * n` with `ok=True`) is legitimate and was deliberately left alone.

**A TEST WAS PASSING FOR THE WRONG REASON, AND PYTEST SAID NOTHING.**
`test_cohort_all_dummy_fails` wrote all-`"A"*101` windows and asserted the gate refused. It did refuse --
because the `ok` column was missing, not because the windows were placeholders. The name described a
behaviour the body never exercised. Replaced by `test_cohort_mostly_placeholder_fails`, which supplies
provenance so the gate is tested on the axis its name claims.

### The cohort, and the reconciliation that closes 6.29a's open unknown

**6.29a's PREMISE HAS CHANGED, AND IT IS THE BLOCKER THAT MOVED.** Section 6.29a records
`clinvar_grch38_clean_seq.parquet` as *"19 columns, NO `ok`"*, and declared 6.28's entire
provenance mechanism INERT on the path Run 17 would actually run. **MEASURED 2026-07-18: the
artifact now carries an `ok` column** -- 21 columns, 4,399,089 rows, 4,398,366 usable, 723 placeholder
(0.0164%) -- reasons `fetch_failed` 2, `non_acgt_allele` 668, `ref_mismatch` 53. The artifact was regenerated during this session. `attach_delta_windows` therefore
takes its provenance branch rather than degrading `usable` to `notna()`, and the mechanism is
LIVE on the configured path.

**THE 0.5% RAZOR IN 6.29a DISSOLVES, AND THE UNKNOWN IT RESTED ON IS ANSWERED.**
6.29a left this explicitly open: *"how many of the 21,814 placeholder rows ... even EXIST in
`clean_seq`'s 4,399,089 ... An inference resting on an inference; do not build on it."*

Answer, measured four independent ways through four code paths on 2026-07-18: **723 exist.**
The reconciliation is exact and is recorded here so that it is never re-derived:

| artifact | rows | usable | placeholder | fraction |
|---|---:|---:|---:|---:|
| `seq_windows.manifest.json` (pathfix) | 4,420,180 | 4,398,366 | 21,814 | 0.4935% |
| `clinvar_grch38_clean_seq.parquet` | 4,399,089 | 4,398,366 | 723 | 0.0164% |
| difference | 21,091 | **0** | 21,091 | |

**Usable is IDENTICAL in both: 4,398,366.** Rows dropped by cleaning (21,091) equals
placeholders dropped (21,091), exactly. Cleaning removed *only* rows that were already
placeholders -- all 19,988 `empty_allele` plus 1,103 of the 1,771 `non_acgt`, leaving 668.
The two figures are the same underlying truth seen through two cohorts, not a contradiction.

Consequence for `run_phase2_eval`'s abort gate: the fraction is **0.0164%**, not 0.4935%.
The old figure sat 0.0065 percentage points under the 0.5% threshold -- a razor that would have
flipped on any small change in the cohort. It is now under by a factor of thirty. The gate is
no longer load-bearing on a coin-toss.


### Suite-size ratchet -- corrected with the tree named

1962 -> **1968**. The 1962 entry existed because a count of 1966 taken on 2026-07-15 had included four
tests from a file that had never been committed: *"The measurement was real; the thing it measured was
not the thing being asserted."* That entry set a condition for moving the number -- commit the file --
and this session met it.

The number was copied from `pytest tests/ --collect-only -q` on the staged tree at `87b670e` with 44
paths staged. It was not computed; the ratchet's history records four consecutive hand-computed values
that were wrong (1882, 1891, 1932, 1944), each caught by the ratchet. **The gate was then run**, which is
what separates a number that is enforced from a number that is written down.

Also closed: `EXPECTED_SUITE_SIZE` had no file extension, so only `.gitattributes`'s `* text=auto`
default matched it -- meaning carriage-return line-feed in a Windows working tree, while every other
governed text file carries an explicit `eol=lf`. The single source of truth for suite size was the one
governed text file without a line-ending rule. Fixed in `9362f2c`; verified with `git check-attr`
(`text: set`, `eol: lf`, 0 carriage-return bytes) rather than by reading the file.

### Still open after this session

- **Part 3 of the hybrid, now UNBLOCKED**: make `X_seq` OPTIONAL in `VariantEnsemble.fit`/`evaluate`/
  `predict_proba`, failing loudly when `cnn_1d` is active without sequence. This removes the CLASS of
  defect rather than the instance. It was gated on committing the 2026-07-15 work, which landed in
  `e57835e`.
- **Run 9 ablation coverage gap**: six external annotation families have NO ablation mask -- ribonucleic
  acid sequencing (5 features), COSMIC (2), KEGG (2), GenomicLM / Nucleotide Transformer (2), Reactome
  (1), heterogeneous graph neural network (1; the existing `no_gnn` covers `gnn_score`, not
  `hetero_gnn_score`). 32 of 95 contract features match no prefix, most legitimately core descriptors.
  This bears on the project's PRIMARY goal: ablation is the instrument by which feature-class
  contribution is quantified, and six unmeasurable families is six unanswerable questions.
- **Session-document gap**: `docs/sessions/` runs to 2026-07-06 and then to 2026-07-18. The work of
  07-13, 07-14 and 07-15 -- roughly four sessions, documented in the ratchet history and in
  `docs/status/` -- has no session record. Recorded rather than backfilled: reconstructing it from the
  ratchet alone would produce a plausible narrative rather than a measured one.
- **`data/primateai3d.py`** remains a connector with no feature in `TABULAR_FEATURES` (6A, 2026-07-15).
  Still undispositioned.
- **Minor**: `test_no_content_based_poly_detection.py:238,251` cites `populate_fasta_seq.py:59`, a file
  that no longer exists. `scripts/download_finngen_R10_DEPRECATED.py` self-declares deprecated for a
  FinnGen release superseded in `77c66f5`. `*.bak_*` is duplicated at `.gitignore` lines 155 and 158.


## 7. THE PATTERN, stated once so it is not re-learned

Every defect above is one of **four** shapes. (Two were identified on 2026-07-12; the last two
were paid for on 2026-07-13, and they are the expensive ones.)

**(a) A number written down once and never re-derived becomes a lie on a schedule.**
`KNOWN_ZERO_DEFAULT` commented as 27 while the literal held 25. `variant_ensemble.py` saying
"65 features" against a 97-feature contract. A G1 pytest floor of 1485 against a suite passing
1815. `RUN_17_PLAN` asserting 91. Each was a guard that had quietly stopped guarding.
**Fix: derive it at gate time. Do not store it.**

> **2026-07-13 addendum, and it is damning.** The G1 pytest floor rotted **four more times in
> two days** -- 1485 -> 1805 -> 1842 -> 1850 -- *every single time* beneath an emphatic
> all-capitals comment demanding that it be raised, written by the person who then failed to
> raise it. **The comment does not enforce itself, and no volume of emphasis will make it.**
> This is a DESIGN defect, not a discipline defect. See 6.14: the fix is a ratchet -- the
> suite itself fails until the constant is bumped, exactly as `EXPECTED_TABULAR_FEATURE_COUNT`
> guards `TABULAR_FEATURES`. **If a rule can be forgotten, it will be. Make forgetting fail.**

**(b) A library that hard-codes a working-directory-relative writable path makes the test
suite a function of the developer's disk.** The AlphaMissense fallback (12 tests red on a
populated box, green on a clean one). `ESM2Connector`'s default cache. `ProteinStructurePipeline`
downloading a structure **into the checkout**. `FinnGenConnector` with no injection point at all.
Every one was invisible locally and visible only on a cold clone.

**(c) A gate that checks a PROXY instead of the thing it claims to protect is not a gate.**
G1 section 13c checked `RUN_17_PLAN.md` for *completeness* -- unfilled `<DECISION>` markers --
and never for *truth*, so it green-lit a paid run against a document that misstated the very
feature contract under test. `vm_bootstrap_run.sh` section E checked that `KANClassifier`
**imports**; the bug was in `fit()`; it imports perfectly; **Run 17 would have sailed through
green and published a twelve-model comparison with a headline model missing** (6.16). The
five-stage correctness harness imported `engineer_features` while the training pipeline ran a
second, drifted copy -- so the gate validated a code path the run never executed.
**Fix: gates must assert the thing they protect. A model that imports is not a model that
trains. A document that is complete is not a document that is correct.**

**(d) A green result from a mutated environment is evidence about the environment, not about
the code.** The developer's `.venv312` held a `sed`-patched `imodelsx` from 2026-05 until
2026-07-13. Locally the Kolmogorov-Arnold Network trained, every test passed, and that green
was used -- in writing, in a remediation document -- to conclude that *"no historical run has
ever lost a model."* On Linux, where the science actually runs, KAN had been raising
`NameError`, being silently swallowed, and vanishing from the ensemble **in every Continuous
Integration run for two months**. The same shape produced the `sed` itself: a fix applied on
some machines and not others, so that "works here" and "works" were quietly different claims.
**Fix: the developer's machine must run what the runner runs. Never patch `site-packages`;
repair in-process, in code that ships. And when a local suite is green, ask what the green is
evidence OF.**

---

**The meta-rule underneath all four:** *a finding in a log is a comment; a finding in a
document is a comment; a finding that fails a test is a gate.* `INCIDENT_2026-06-14` had
already recorded the `data/` pollution -- nothing happened for four weeks, because nothing
failed. The 41 warnings printed in every run for weeks and were scrolled past; **every single
one of them turned out to be a real defect or the visible edge of one.** Nothing here was
discovered by being clever. It was discovered by making things fail loudly and then reading
the output.
**Fix: never hard-code a writable path without an override, and let a guard fail loudly if
anything writes into `data/`.**

And the meta-lesson, which cost the most: **a finding recorded in a document is a comment; a
finding that fails a test is a gate.** `INCIDENT_2026-06-14` had already written down that
`test_lovd_annotation_reaches_training_matrix.py` "writes to the REAL data/". Nothing happened
for four weeks, because nothing ever failed.

**Next:** 6.2 (branch; expect red — that red is information), then 6.5 (the harness's own
sanity model does not converge, which is a scientific problem, not a hygiene one).
## ROADMAP delta -- 2026-07-20 (post-3bba87e measurements)

Everything in this section was MEASURED on 2026-07-20 **after** the day's three commits
(`fb23543`, `106d107`, `3bba87e`) had landed. Each figure names the command that produced it.
Re-run them; do not believe them.

### 1. 6.29a -- the headline claim is CLOSED; the flag ambiguity is NOT

6.29a records `data/processed/clinvar_grch38_clean_seq.parquet` as
*"4,399,089 rows, 6/12/2026, **19 columns, NO `ok`**"* and concludes that every Run 17 launcher
points at an artifact carrying no provenance, leaving 6.28's entire mechanism inert on the
configured path.

**Measured 2026-07-20** (`pyarrow.parquet.ParquetFile(...).schema_arrow.names`):

    rows 4399089   cols 21
    variant_id source_db chrom pos gene_symbol transcript_id pathogenicity allele_freq
    clinical_sig protein_change fasta_seq source_id metadata ref alt consequence
    ReviewStatus fasta_seq_ref fasta_seq_alt ok reason

The row count matches 6.29a exactly. **The column count and the provenance claim do not.**
The artifact gained `ok` and `reason` after 6.29a was written on 2026-07-15.

`probe_window_provenance_2026-07-20.py` then ran `attach_delta_windows` against BOTH artifacts,
keyed off the same 50,000 rows, reading the thresholds live from `EnsembleConfig`:

| artifact | provenance | verified | usable | fraction | unmapped | placeholder | gate |
|---|---|---|---|---|---|---|---|
| `clinvar_grch38_clean_seq.parquet` (21 col) | `parquet+ok` | **True** | 49,970/50,000 | 0.999400 | 0 | 30 | **ACCEPT** |
| `seq_windows/seq_windows.parquet` (8 col) | `parquet+ok` | **True** | 49,970/50,000 | 0.999400 | 0 | 30 | **ACCEPT** |

Identical because `seq_windows.parquet` is a **superset**: 4,420,180 against 4,399,089, and the
extra **21,091** rows are exactly the alleleless variants absent from the clean cohort
(4,420,180 - 4,399,089 = 21,091, the same figure 6.29 reconciles against `clean_cohort.py`).
Keying off `clean_seq` finds the same variants in both files.

Provenance resolved to `parquet+ok` rather than `rows+ok` because the probe passed only
`chrom/pos/ref/alt`, forcing the parquet tier. With `--clinvar` loading all 21 columns, the
cohort frame carries the sequence columns itself and tier 1 `rows+ok` fires instead.
**Both tiers are now measured as verified.**

**A PREDICTION WAS MADE AND WAS WRONG.** Reading 6.29a, it was asserted that
`seq_require_verified_provenance=True` would cause the gate shipped in `106d107` to REFUSE the
launcher path. It does not; it accepts. The evidence was already in this session's own
`fb23543` work, which measured that exact file resolving through tier-1 `rows+ok` -- a branch
that cannot fire without an `ok` column. **A document was quoted over a measurement taken the
same day.** That is the view-first rule, broken in the direction it exists to prevent.

**STILL OPEN, unchanged:** `--seq-windows` means a **DIRECTORY** in `train.py:102` (default
`data/processed/seq_windows`, appends `/seq_windows.parquet`) and a **FILE** in
`run_phase2_eval.py:49`. One flag, one repository, opposite contracts. `run_phase2_eval.py`
reads its value through `attach_delta_windows` at lines 436-438, and the 8-column schema is
`chrom/pos/ref/alt/fasta_seq_ref/fasta_seq_alt/ok/reason` -- so the "NOT TESTED" blocker
6.29a records is now answerable, and the reader is identified.

### 2. THE MONITORING LAYER RUNS, SUCCEEDS, AND CANNOT REPORT A FINDING

Three separate defects, none a blocker, all real. Established via the GitHub Actions
representational-state-transfer application programming interface (public repository, no
authentication -- the `gh auth login` token remains dead) and by reading the workflow files.

**(a) `scripts/run_data_freshness.py` returns 0 unconditionally.** 28 lines; `main()` ends
`return 0` regardless of findings. Run locally 2026-07-20: `sources=24 changes=0
report=reports\data_freshness\FRESHNESS_2026-07-20.md`, exit code 0. `changes=24` would exit 0
identically. Line 22 also defaults twice -- `(results or {}).get(agent, {}) or {}` -- so a
pipeline that never ran prints `sources=None changes=None report=None` and still exits 0.
The workflow compounds it with `if-no-files-found: warn`, so an absent report is green too.
**The monitor can only go red by crashing**, which is what the 2026-06-22 and 2026-06-29
scheduled failures were. Neither has an incident record.

**(b) The report's own content carries two detector defects.** The 2026-07-20 report says
`alphafold [missing] absent on disk`. It is not: `data/external/alphafold/` holds
`alphafold_cohort.parquet` (110,220,322 bytes) and `alphafold_coverage.json` (2,094,560 bytes),
both written 2026-07-03 -- 107.1 MB, against the 1.8 MB the same monitor reported present on
2026-06-14. The registry entry (`monitoring/registry.py:137-140`) names
`data/raw/cache/alphafold`, the `.cif` cache it was written for; the July build wrote elsewhere.
**Stale registry path, not lost data.** A first reading of `[missing]` as data loss was wrong
and is retracted here.
Further, `database_freshness_detector.py:109` scans the PARENT directory for cruft, so sources
sharing a directory inherit each other's clutter -- `.OOMbak` files belonging to AlphaMissense
were reported against `hgmd`, `omim` and `clingen`. And line 107 sums a whole directory tree,
so `gtex` and `esm2` -- which BOTH declare `local_path = "data/raw/cache"`
(registry.py:123 and :142) -- reported byte-identical sizes on two separate dates
(1348.4 MB on 06-14, 21071.3 MB on 07-20). Neither figure describes either source.

**(c) The only scheduled automation is dry-run, so agent liveness ages permanently.**
`orchestrator.py:261` guards `_record_run_telemetry` with `if not self._dry_run:`.
`data_freshness.yml:30` invokes the script with no `--no-dry-run`. So five successful
scheduled runs (06-15, 06-22, 06-29, 07-06, 07-13) wrote no `agent_runs` telemetry, and all 22
agents still report `last=2026-06-20T02:30`. On 2026-07-20 that reached `age=30.27d`, crossing
the `--max-age-days` default of 30.0 -- **the whole fleet turned STALE this morning.**
`check_agents_active.py` exiting 0 is CORRECT: its docstring (lines 25, 37) defines STALE as a
warning and the hard failures as ERRORED / DRY_RUN_ONLY / SECTION_ONLY / NEVER_RUN /
UNSCHEDULED / MISSING_IMPL. `--strict` exits 1, verified. **An earlier claim that the checker
was "not doing its job" was wrong and is retracted.**

**The Monthly Drift Monitor's current bytes have never executed.** `drift_monitor.yml` is
34,876 bytes, last modified 2026-07-14; its last run was 2026-07-01; the next is 2026-08-01.
Its own comment (lines 60-66) records that the version before the repair ran *without its drift
libraries and without the code it was supposed to run* -- and one of those three runs reported
SUCCESS. The repair itself needed three commits (`4528414`, `68d8321`, `69b9f01`, the last
because a patcher missed carriage-return line-feed anchors). **A fix for a silent failure,
never executed, is not yet a fix.** All three `workflow_dispatch` inputs are inert by default
(`release_name="manual"`, `auto_retrain=false`, `schema_matrix=""` -> NOT CHECKED, exit 3) and
the job is concurrency-guarded, so a manual dispatch costs one hosted runner and nothing else.

**Scheduling itself is reliable but late.** Both crons fire at the top of the hour, the most
contended slot. Delays measured from the application programming interface
(timestamps are Coordinated Universal Time, cross-checked against
Continuous Integration #546 at 08:18 UTC = 04:18 local): Data Freshness 3h17m to 6h00m across
five consecutive Mondays; Drift 2h05m to 5h15m across three consecutive months. **No run has
ever been dropped.** An earlier claim of "four of five Mondays" was read off a truncated list
and is corrected: it is five of five.

### 3. JEPA V1 READINESS -- measured, not assumed

`audit_jepa_readiness_2026-07-20.py` (read-only, proven by abstract-syntax-tree walk) parsed
every modality module. **JEPA has zero code in this repository** -- three mentions, all in
documentation (`HANDOFF_2026-07-15:119`, `REMEDIATION_2026-07-13:577`, `ROADMAP.md:814`).

| modality | array-returning functions | what Milestone 1 costs |
|---|---|---|
| DNA `genomic_lm.py` | **0 of 15** | Nucleotide Transformer vectors never leave the function. **Restructure.** |
| Protein `esm2.py` | 5 of 36, incl. `_embed_sequence() -> Optional[np.ndarray]` (415) and `_cache_get_embedding()` (160) | Embeddings already computed **and cached**, then discarded. **Closest to ready.** |
| Graph `gnn.py` | 4 of 25, but `predict_proba`/`score_all_nodes` return SCORES | No hidden-state accessor. **Restructure.** |
| RNA `rna_pipeline.py` + `rnaseq.py` | **0 of 8** | No embedding surface at all. |

This confirms the roadmap's existing statement that every foundation model added so far has been
**collapsed to two scalars**, and that step 1 is a build rather than a wiring job.

**DISK IS A HARD BLOCKER.** `Get-PSDrive C` on 2026-07-20: **10.91 GB free**, 924.68 GB used.
The roadmap's measured minimum embedding cache is **~14.7 GB** -- and that is Nucleotide
Transformer plus ESM-2 only, **pooled only**, while the JEPA design explicitly requires
token-level retention. Shortfall at minimum **3.79 GB**, and free space has FALLEN from the
14.74 GB recorded at 6.12. **JEPA V1 cannot cache embeddings locally.** The embedding store must
target Drive or cloud from the outset, not as a later migration.

### 4. CONFORMAL -- EXTEND the existing package; do NOT create `uncertainty/`

The conformal specification proposes `src/genomic_variant_classifier/uncertainty/` with fifteen
modules. **`src/genomic_variant_classifier/conformal/` already exists** with six, and covers
roughly half the specification including the harder half.

| specification module | reality |
|---|---|
| `scores` | **EXISTS** -- `scores.py`: LAC, APS, RAPS (`*_scores_true` and `*_scores_all` for each) |
| `binary` | **EXISTS** -- `mondrian.py::MondrianConformalClassifier(group_mode="class")` is label-conditional binary |
| `multiclass` | **EXISTS** -- APS/RAPS via `split.py::SplitConformalClassifier(score=...)` |
| `subgroup` | **EXISTS** -- `coverage.py::per_stratum_coverage`, `group_coverage_disjoint` |
| `evaluation` | **EXISTS** -- `coverage.py`: marginal, per-class, set-size summary, abstention rates |
| `splits` | **EXISTS** -- `split.py::conformal_quantile`; `calibrate.py::_gene_disjoint_mask` |
| `config` | **PARTIAL** -- `calibrate.py::CalibrationConfig` |
| `artifacts`, `ordinal`, `multilabel`, `gene_ranking`, `risk_control`, `monitoring` | **GENUINELY ABSENT** |

`grouped.py::GroupedConformalClassifier(group_agg="max")` is precisely the specification's
**gene-cluster conformal mode**. Both levels of guarantee -- variant-level and gene-cluster --
already exist in code.

**Six modules to add, not thirteen.** The absent six are the clinically-motivated half:
conformal risk control, ordinal contiguous sets, multilabel, gene ranking, provenance/fail-closed
artifacts, and drift monitoring.

**A CORRECTION IS OWED HERE.** The audit's own section 4 reported `binary`, `multiclass`,
`subgroup` and `evaluation` as ABSENT. It matched **file names**, not capability. That is the
same failure this section documents elsewhere, committed inside the audit built to avoid it.

The specification's four-partition requirement (train 60 / tune 15 / conformal 10 / test 15,
all gene-disjoint) maps onto existing structure: the v2 gene-disjoint `tune` partition already
feeds `X_tab_cal_ext` and does the probability-calibration job. **What is missing is
specifically the conformal partition**, not the concept. The specification's own instruction --
`split_protocol_v1` and `v2` in parallel until equivalence is proven, never a one-step
overwrite -- is the right discipline and is adopted.

### 5. METRIC SURFACE -- eleven files, one unused canonical module

Parsed 2026-07-20. Eleven files call metrics independently; `evaluation/metrics.py` exists and
is NOT called by `variant_ensemble.evaluate()`, which computes its own five inline
(`roc_auc_score`, `average_precision_score`, `f1_score`, `matthews_corrcoef`,
`brier_score_loss`). Never called anywhere: `balanced_accuracy_score`, `cohen_kappa`,
`ndcg_score`.

**The audit UNDERSTATED what exists**, because it matched scikit-learn function names only.
`evaluation/metrics.py` additionally implements Expected Calibration Error, calibration slope
and intercept, bootstrap confidence intervals, lift-over-floor and stratified evaluation --
none of which is a scikit-learn call -- and carries 30 tests. The **Expanded metric stack**
row in the measured-state table already records this as PARTIAL against section 16 of the
conformal specification.

**The work is therefore not "add metrics".** It is: one canonical module every caller routes
through, and a living glossary GENERATED from that module rather than maintained beside it --
so a metric cannot be defined in one place and computed differently in another. Same principle
as the suite-size ratchet and the README badge, and the only form that does not go stale.

### 6. METHODOLOGY -- twenty-one instances of one failure in a single session

**A checker that string-matches or name-matches fires on something that merely resembles its
target.** Instances on 2026-07-19/20 include: a docstring-staleness regex matching digits
inside a date; an import check matching `torch.utils.data` because it ends in `.data`; three
checks matching prose that DESCRIBES the rule (a refusal message, a Protocol docstring, and --
inside the fix for the previous one -- a replacement docstring quoting the old assertion);
`DOC.count("| 5 |")` matching cells in a reconciliation table; `\d{3}\.\d{2}` truncating
`1026.25` to `026.25`; a destructive-call detector flagging `str.replace` by method name; and
the conformal gap analysis above, which compared module names to a specification instead of
comparing capabilities.

**The durable lesson is not "remember to parse."** It is that **outcome-asserting checks catch
what careful reading does not** -- most of these were found by a machine check written to
assert a RESULT rather than to confirm an ACTION.

**And the failure that is not in that class, recorded separately because it is worse:** on
2026-07-20 a stale roadmap entry (6.29a) was quoted over a measurement taken the same day by
this session's own commit. No checker was involved. That is root pattern (a) consumed rather
than produced, and the defence against it is the one this project already states: read the
artifact, not the description of the artifact.

### 7. WHAT IS OPEN AFTER TODAY

- **`--seq-windows` dual meaning** (6.29a's surviving half). Directory in `train.py:102`,
  file in `run_phase2_eval.py:49`.
- **ESM-2 coverage on the 4,399,089-row cohort is UNMEASURED.** The parser exists, is wired and
  is tested; the coverage number is what decides whether `esm2_delta_norm` and `esm2_llr`
  contribute signal, and whether JEPA V1's protein modality is real.
- **Monitoring remediation, three separable fixes:** the registry's stale AlphaFold path plus
  the detector's parent-directory and directory-total defects; an exit code that can express a
  finding; and a scheduled run that can write telemetry. The third needs a decision, because
  `--no-dry-run` enables human-in-the-loop prompts a hosted runner cannot answer.
- **Monthly Drift Monitor never dispatched** since its 2026-07-14 repair.
- **Expanded metric stack:** route eleven callers through one module; generate the glossary.
- **Conformal:** add the six genuinely-absent modules; add the fourth partition behind
  `split_protocol_v2`.
- **JEPA:** blocked on storage (3.79 GB short at the absolute minimum) and on the ESM-2
  coverage measurement.
- **Continuous Integration #540-#546 all green**, twelve consecutive, confirmed by screenshot
  and by the GitHub Actions application programming interface. The three red runs (#535-#537) are the 2026-07-19 ratchet-split failures.


## ROADMAP delta -- 2026-07-20 (part two): the metric kernel

The delta above was written before the metric work and does not mention it. Commit `5615cd0`,
Continuous Integration #548 GREEN, ratchet 2017 -> 2055.

**Six of seven audited defects in `evaluation/metrics.py` repaired.** Every one was verified by
reading the file before any code was written. Full narrative in
`docs/sessions/SESSION_2026-07-20_metric-kernel-fail-closed.md`.

| defect | what it was | state |
|---|---|---|
| A | `evaluate()` cleaned `score` and `prob` on TWO SEPARATE MASKS -- equal lengths, different rows, calibration paired with wrong labels, silently | **FIXED** -- one joint mask, exposed as `CleanArrays.mask` |
| B | `_clean` ended `astype(int)`; `[0,1,3]` makes `(1-y).sum()` negative and AUROC SIGNED; `[0,1,2]` fires `_degenerate` spuriously | **FIXED** -- validated, never coerced |
| C | legacy `compute_classification_metrics` / `ModelEvaluator` unsafe: `confusion_matrix(...).ravel()` raises on single class, specificity returns 0 where undefined | **DEFERRED** -- head byte-identical; needs a call-site census |
| D | `is_probability([])` returned True; the old test ASSERTED it (6.21a's shape) | **FIXED** |
| E | the calibration solver could not report nonconvergence | **FIXED** -- `CalibrationFit`, iterable for compatibility |
| F | `stratified_evaluate` dropped rows with a missing group label; strata did not partition the cohort | **FIXED** -- `__MISSING__` stratum + partition assertion |
| G | subgroup sufficiency tested total `n` only | **FIXED** -- class-support floors with a `status` |

**THE BOOTSTRAP IGNORED GENE CLUSTERING** -- found independently, confirmed by the audit.
**RECONCILED 2026-07-26 (Option C commit 2).** One canonical engine; gene-cluster resampling is
REQUIRED for a certified interval, variant-level resampling is explicit-only and never
certification-eligible, and there is no silent fallback between them. Every interval now carries
its own status, resampling unit, stratification, cluster provenance and replicate accounting.
Evaluation report schema version 2; schema-version-1 artifacts stay readable and are NEVER read
as certified. Two further defects were found in the dispatcher while landing this: its docstring
failed the existing StrEnum floor guard, and its replicate accounting modelled a sampling scheme
bootstrap_ci does not use (0.506 of replicates reported degenerate against a theoretical 0.500),
which was invisible because the status it fed was a hard-coded constant.
Resampling variants as independent understates variance, so **every confidence interval this
project has published is anti-conservative**. `cluster_bootstrap_ci` resamples whole genes and
REPORTS THE DESIGN EFFECT so historical intervals can be re-read. Measured where six of thirty
genes have inverted discrimination: naive `[0.7548, 0.8439]`, clustered `[0.6611, 0.9228]`,
**design effect 2.935x**. A control pins it near 1.0 when gene assignment is arbitrary.

**Expanded metric stack: still PARTIAL.** The kernel is now fail-closed and gained `log_loss`
and `auprc_gain`, but the registry, the typed report schema and the clinical panels are not
built. Against the project metric specification's sixteen panels, what exists covers Panel B
(binary discrimination) and most of Panel D (calibration). Panels A, C, E, F, G, H, I, J, K, L,
M, N, O and P are absent, and Priority 1 -- one canonical evaluation registry -- is the next
commit.

**NEW OPEN ITEM -- METHODS.md section 3.1 is stale in three ways.** It says "Four tabular base
models were trained on the 64-feature matrix" against a roster of thirteen and a contract of
95; nine models are absent (catboost, cnn_1d, deep_ensemble, kan, logistic_regression,
mc_dropout, svm, svm_bagged_rbf, tabular_nn). Line 152 says the sequence convolutional network
is "excluded from the inference pipeline", written before its 2026-07-05 Tier-1
re-architecture. Line 164 says STRING "combined score >= 500" while the registry caches
`string_graph_700.pkl` -- UNVERIFIED, flagged for measurement rather than asserted.
`test_methods_feature_count.py` passes throughout because it checks the count sentence, the
group-table sum and HGMD's absence, and never the roster --  while
`test_readme_claims.py:375` has read the roster from a live ensemble since 2026-07-14. The fix
is to GENERATE section 3.1 from `VariantEnsemble.base_estimators` and widen the gate.

**A CORRECTION TO ROADMAP 6.23.** It records all performance figures as WITHDRAWN 2026-07-14
and states "the figures are not restated even in the withdrawal notice". On 2026-07-15 the
performance-figure ban in `test_readme_claims.py` was DELETED DELIBERATELY, and the test file
records the decision and its reasoning at line 698. README lines 319-332 carry the Run 15
AUROC under "Early results" with the caveats the metric specification's Finding 5 asks for.
**6.23 is now the stale entry** -- one day younger than the line 41 defect corrected above,
and the same shape.

## ROADMAP delta -- 2026-07-20 part three: the calibration surface

Full record: `docs/sessions/SESSION_2026-07-20_calibration-defect-repair.md`.
Commit `44511fa`, Continuous Integration #550 green, suite 2055 -> 2060.

**The census that reframed the metric programme.** All 813 Python files parsed by abstract
syntax tree. **No module under `src/` imports `evaluation/metrics.py`** -- the kernel hardened
in `5615cd0` has no production caller. The legacy interface has zero external callers, so its
removal is a live scope decision rather than a risk. Ten independent implementations of
expected calibration error were found, plus three bootstrap, two rank-AUROC, three coverage.

**Two defects, six files, both repaired.** The open top bin `(p >= lo) & (p < hi)` with
`hi == 1.0` silently excludes every prediction of exactly 1.0 -- 86.7% under-report on a
20%-pure-leaf fixture, in three files including `calibrate_thresholds.py`, which selects
operating thresholds. And a previously undocumented misalignment where `calibration_curve`'s
non-empty bins are zipped against `np.histogram`'s full bin counts -- correct whenever every
bin is occupied, and **64x under-reported on sparse saturated data**. Both reached the same
wrong number by different routes, so unanimity read as correctness.

**Retraction.** `evaluator.py:305` does NOT carry the open-top defect; it was repaired
2026-07-10. The claim came from `metrics.py`'s docstring, stale since that date -- the seventh
recorded instance of a fact stated twice where only one copy was maintained. Corrected with
its dates.

**Open.** `calibration_drift_agent.py:45` is the tenth implementation and is UNEVALUATED --
its constructor requires `classes`, `baseline_ece`, `output_dir`. It monitors calibration
drift in production and is not assumed clean. Also open: `benchmark.py:125` dead
`bin_midpoints`; three unreconciled bootstrap implementations against the 2.935x design
effect; METHODS.md section 3.1.

## ROADMAP delta -- 2026-07-25: clean-cohort Phase 1b-E complete, 1b-C authorized-open

Decision record: docs/measurements/DECISION_2026-07-25_cohort-v2-authorization-and-phase-split.md
Evidence commit: docs(measurements): preserve clean-cohort adjudication and ontology evidence

Phase 1b-E (clean-cohort evidence + design) -- COMPLETE. The clean_cohort builder's
duplicate-group representative selection was measured to be input-order dependent (stable
sort + positional .iloc[0]): 1,610 legacy order-sensitive selections, 0 under the
deterministic P6 policy (order-invariance verified). A group-level evidence adjudicator
(P6) over a lossless multi-axis parse of clinical_sig was designed and measurement-validated:
14 binary-vs-explicit-conflict labels withheld, 189 binary-vs-uncertain labels recovered,
net +175 trainable (reconciles exactly), 22 added irreducible-conflict quarantines. The
clinical_sig ontology census is complete: 102/102 distinct values parse, 0 unconsumed
tokens (fail-closed gate satisfied); ontology repair expands positives by +208, separable
from the P6 order-correction gain.

Phase 1b-C (certified cohort v2 construction) -- AUTHORIZED_NOT_IMPLEMENTED. Gates
C1-C10 in the decision record. Step 1b is NOT complete until 1b-C completes.

Dependency gate. CLINICAL_SIGNIFICANCE_ONTOLOGY_VERSION = 1.0 is a validated
specification version, not a shipped module. Metric-stack IMPLEMENTATION may proceed on
cohort-agnostic infrastructure; production metric BACKFILL/certification is
BLOCKED_BY_COHORT_V2. v1 cohort retained for lineage, not deterministically derived, not
to be overwritten.
## ROADMAP delta -- 2026-07-25: transient drift-monitor network failure (CI run 30178381817, dfa5dbe)

The docs-only commit dfa5dbe ("docs(architecture): evaluation-layer wiring audit
for metric-stack seam") produced a RED Continuous Integration run, 30178381817.
Root cause is recorded here because a transient failure on main must not vanish
unexplained.

WHAT FAILED. Only the two `drift monitor (isolated env)` matrix legs (Python 3.11
and 3.12). Both `pytest` legs (3.11 and 3.12), `lockfile drift check`, and
`Docker build smoke test` all passed on the first attempt; `Push image to GHCR`
was correctly skipped (release-only). The code was never implicated -- a docs-only
commit cannot change what the isolated drift job installs or runs.

ROOT CAUSE, from the attempt-1 log. During the step "Install the ISOLATED drift
environment", pip was downloading nannyml-0.13.1-py3-none-any.whl (23.0 MB) and
the connection dropped mid-stream after 1,048,576 of ~23,000,000 bytes:

    pip._vendor.urllib3.exceptions.ProtocolError:
      ('Connection broken: IncompleteRead(1048576 bytes read, 21974671 more expected)')
    ##[error]Process completed with exit code 2.

This is a transient network/CDN interruption during a large binary download from
the Python Package Index, NOT a dependency conflict (no ResolutionImpossible / No
matching distribution appeared -- resolution had completed and the resolved wheel
was downloading), NOT a `pip check` conflict (that step was never reached), and
NOT a `tests_drift/` failure (also never reached).

RESOLUTION. A bare re-run of only the failed jobs (`gh run rerun 30178381817
--failed`, attempt 2 started 2026-07-25T23:14:56Z) passed both drift legs with
zero change to the commit or to requirements-drift.txt. A failure that clears on
an unchanged re-run is transient by definition. Every job on run 30178381817 is
now green.

STRUCTURAL NOTE (not fixed here). The drift job installs a large dependency tree
(nannyml, gcsfs, grpcio, google-cloud-storage, evidently, litestar and their
transitive dependencies) fresh from the Python Package Index on every run, with
no download retry or timeout. No `pip install` step in ci.yml currently passes
`--retries` or `--timeout` -- the main pytest job (requirements-api.lock,
requirements.txt, requirements-dev.txt) shares the same exposure. A single dropped
connection anywhere in that tree fails the whole job. Hardening this with pip's
built-in retry/timeout is tracked as a separate, deliberate commit and is not
bundled with unrelated work.

## ROADMAP delta -- 2026-07-25: P6 audit R2 provenance correction (deferred, bounded) + preemption rule

Two entries recorded at the clean boundary after the CI-hardening work, before the
metric-stack seam begins.

------------------------------------------------------------------------------
WORK ITEM: P6 audit R2 provenance correction
  Status:          REQUIRED_PROVENANCE_CORRECTION
  Blocks:          cohort-v2 certification (the certified v2 build must consume the
                   corrected R2 artifact, not the ambiguous original)
  Does NOT block:  the cohort-agnostic metric-stack seam (CanonicalVariantTable) --
                   the seam does not consume P6 adjudication arithmetic
  Underlying data: NOT invalid. Both measurements are correct; only the terminology
                   is ambiguous and one figure is reported under an overloaded name.

  DEFECT. docs/measurements/CLEAN_COHORT_P6_AUDIT_2026-07-25.txt uses the word
  "canonical" for two DIFFERENT estimands:
    * 63  = representative-row label changes vs legacy (row-selection diagnostic:
            label on the P0-selected representative row vs the P6-selected one).
            The POLICY TABLE line 49 and ACCEPTANCE line 87 call this
            "kept-row / canonical-label changes".
    * 203 = final group-adjudicated binary-label changes vs legacy output label
            (the scientifically operative cohort-label change: 14 binary-vs-
            explicit-conflict + 189 binary-vs-uncertain = 203, lines 66-67).
            Line 57 calls this "GROUP-ADJUDICATED ... STRICTER, different basis ...
            NOT comparable" to the 63 line.
  The committed file is internally inconsistent (line 87 names 63 "canonical",
  while lines 65-67 describe the 203-count as changes vs the "P6 canonical label").
  An incomplete local edit (now reverted) had merged the two into a single row of
  203 and deleted the distinction -- that was worse (self-contradictory), so it was
  restored, not committed.

  R2 CORRECTION PLAN (one focused change, when the seam is done; REGENERATE, do not
  hand-edit):
    1. Extend scripts/probe_clean_cohort_p6_2026-07-25.py with a typed per-variant
       PolicyDelta (representative_row_changed, representative_row_label_changed,
       final_adjudicated_label_changed, trainability_changed, quarantine_changed;
       plus legacy/p6 representative and output labels). No generic label_changed.
    2. Generate from ONE per-variant pass: the policy table, acceptance block, a
       2x2 overlap of {representative-row label changed} x {final adjudicated label
       changed} (reconciling n10+n11==63 and n01+n11==203), and exact label /
       trainability / quarantine transition matrices.
    3. Rename unambiguously: "representative-row label changes" (63) and "final
       group-adjudicated binary-label changes" (203). Remove every unqualified
       "canonical-label changes". Use n/a (not 0) for P0-P5 in the group-adjudicated
       row, since those policies produce no independent group-adjudicated label.
    4. Write docs/measurements/CLEAN_COHORT_P6_AUDIT_2026-07-25_R2.txt as a
       SUPERSEDING artifact. Append only a short supersession pointer to the
       original; do NOT rewrite the original evidence (preserve provenance).
    5. Verify the probe inventory ratchet (test_review_status_tier.py) remains
       exact after the probe change; add tests for the reconciliation asserts;
       CI green on 3.11 and 3.12.

------------------------------------------------------------------------------
PROCESS RULE (adopted 2026-07-25): work-item preemption

  PREEMPT the active task ONLY when a new finding:
    1. invalidates an assumption of the active task, OR
    2. can corrupt the active task's output, OR
    3. is an immediate release / safety / data-integrity blocker, OR
    4. cannot be safely isolated and recorded.
  Otherwise: restore safe state, record the exact work item, continue the active
  dependency path, and schedule the finding at the nearest clean boundary.

  Rationale: this session's CI detour and the P6 discovery both showed the failure
  mode where every discovered issue becomes the next immediate build regardless of
  dependency. The transient drift failure was correctly recorded as isolated infra
  drift and not mixed into unrelated work; the same discipline applies to the P6
  correction, which is real but independent of the metric seam.

------------------------------------------------------------------------------
ORDERING -- GROUND TRUTH as of 2026-07-27. Supersedes the 2026-07-25 ordering.

THREE TIERS. Nothing previously planned is dropped. Tier 1 is in flight and
completes FIRST; Tier 2 is queued behind it and must be complete BEFORE Run 17;
Tier 3 is Run 17 itself. A Tier 2 item may not start while a Tier 1 item it
depends on is open.

=====================================================================
TIER 1 -- IN FLIGHT. These complete before anything in Tier 2 begins.
=====================================================================

  1. [done] restore the incomplete P6 edit
  2. [done] record P6 R2 as required-before-cohort-v2-certification
  3. [done] CanonicalVariantTable metric seam, landed as 2e04bd9 (Option C step
           5.1). CERTIFICATION OF THE SEAM REMAINS OPEN -- see carried item (a):
           as_meta() emits gene_id but never gene_symbol, so a seam-produced
           frame yields an EMPTY gene_errors list while its docstring claims to
           be the frame ClinicalEvaluator.evaluate expects.
  4. [done 2026-07-26] P6 R2 probe + superseding audit (bounded provenance
           repair). Closed at 4ca92d7, Continuous Integration run #621 green,
           ratchet 3169.
           WHAT LANDED: an additive --emit-json capture proven to leave the
           evidence artifact byte-identical; a frozen golden reference
           (CLEAN_COHORT_P6_GOLDEN_2026-07-26.json); a layered reconciliation
           (ProbeConfig -> load -> compute -> summarize -> render) built on ONE
           immutable PolicyDelta pass; Table A as a 2x2 overlap and Table B as a
           3x2 JOINT table with every margin derived; a machine-readable sidecar
           emitted from the SAME Reconciliation object as the text; and a
           supersession pointer appended to the original with no number rewritten.
           MEASURED, first time: n11 = 29, n_na1 = 17, and the joint cells
           neither_changed = 0 against legacy_missing_only_changed = 17 -- all
           seventeen Table B label changes fall in the legacy-missing row, a fact
           that independent marginals could not represent.
           THREE PLAN CORRECTIONS FORCED BY THE SOURCE:
             (i)   n01 + n11 == 203 is FALSIFIED. The 63 and 203 counts are summed
                   over different universes. Replaced by n01 + n11 + n_na1 == 203.
             (ii)  "five booleans" became four total booleans plus ONE NULLABLE
                   comparison: representative_row_label_changed is undefined, not
                   False, when P6 selects no representative row.
             (iii) base_quar is NOT a subset of p6_quar; the reverse holds. Legacy
                   withholds a representative from 107 variants, P6 from 85, and
                   P6 NEVER newly quarantines on this cohort -- it un-quarantines 22.
           ALSO FOUND: "explicit conflicts preserved: 112" counts withheld-label
           STATES (85 irreducible + 27 ambiguous), not preserved explicit
           conflicts; and the published 63 counts a comparison against a MISSING
           ROW for 10 of its members.
  5. [done] reconcile duplicate calibration / bootstrap paths. Calibration by the
           2026-07-20 census, pinned by test_calibration_implementations_agree.py.
           Bootstrap on 2026-07-26 (Option C commit 2): three implementations
           became one canonical engine and the resampling unit became an explicit,
           typed part of every confidence interval. Ratchet 2991 -> 3118.
  6. [NEXT] metric registry / orchestrator -- COHORT-AGNOSTIC.
           The integration layer over panels that mostly exist already: binary
           metrics, calibration and bootstrap (evaluation/metrics.py), conformal
           (conformal/), capability and validation states
           (evaluation/capabilities.py), gene-cluster resampling
           (evaluation/cluster_resolution.py), clustering metrics, representation
           geometry. There is NO registry or orchestrator module today.
           ARCHITECTURAL SAFEGUARD, verified intact 2026-07-27: no module under
           src/ imports probe code. The metric stack consumes
           CanonicalVariantRecord; it never reaches into a probe.
  7. certified cohort-v2 implementation under corrected P6 evidence -- gates
           C1-C10 of DECISION_2026-07-25_cohort-v2-authorization-and-phase-split.md.
           UNBLOCKED 2026-07-26: that decision recorded R2 as blocking v2
           certification because "the certified v2 build must consume the
           corrected R2 artifact, not the ambiguous original". The corrected
           artifact now exists and is golden-verified.
  8. production metric backfill, calibration and certification ON v2.

  THE DEPENDENCY RULE, quoted from the decision record so this ordering question
  is settled by evidence and not by whoever reads a summary line next:

      Metric-stack IMPLEMENTATION may proceed on cohort-agnostic infrastructure;
      production metric BACKFILL/certification is BLOCKED_BY_COHORT_V2.
      ... if the next metric-stack task requires real production labels,
      cohort-specific expected values, or the new evidence-summary fields, then
      v2 construction moves ahead of that task. The DEPENDENCY -- not the session
      label -- determines the order.

  On 2026-07-27 I recommended swapping 6 and 7 and was WRONG: I reasoned from the
  roadmap summary line rather than from the decision record, which distinguishes
  implementation from backfill. A registry needs none of the three triggers, so 6
  stays before 7. Recorded because the error is instructive.

=====================================================================
TIER 2 -- QUEUED BEHIND TIER 1. ALL REQUIRED BEFORE RUN 17.
=====================================================================

  9. EXPANDED METRIC STACK -- close the PARTIAL recorded at section line 848.
           Present: AUROC, AUPRC with lift over floor, Brier, Expected Calibration
           Error, calibration slope/intercept, bootstrap intervals, stratified
           evaluation. MISSING, per section 16 of the conformal specification:
           validity (per-class and worst-group coverage, coverage gap); efficiency
           (set-size distribution, singleton/doubleton/empty rates); clinical
           behaviour (pathogenic-exclusion rate, severe-error rate, deferral
           burden, positive and negative predictive value among singletons);
           selective prediction (risk-at-coverage, area under the risk-coverage
           curve); multimodal robustness; shift robustness.

 10. CONFORMAL PREDICTION -- source: The_best_conformal_prediction_implementation.
           Per-label conformal thresholds with multilabel risk control; adaptive
           candidate-gene sets with top-K retrieval risk control;
           modality-signature-aware calibration; missing-modality evaluation.
           OVERLAPS item 9 -- the section-16 validity and efficiency panels ARE
           the conformal evaluation surface. Build them once, not twice.

 11. MOFA+ MULTI-OMICS FACTOR ANALYSIS -- source: MOFA__implementation.
           Interpretable shared and modality-specific biological programs;
           cross-modal support and discordance; gene-program rankings; subgroup
           structure; reconstruction anomalies; missing-view uncertainty.
           HARD CONSTRAINT from the source: integrate through LEAKAGE-SAFE
           PROJECTION, never full-cohort transductive fitting, and only where the
           views share a real, explicitly defined observational axis.
           Doubles as a rigorous linear-probabilistic benchmark for the VAE and
           JEPA representations in items 13 and 15.

 12. RNA FOUNDATION MODELS -- source: RNA_sequence_foundation_models_implementation.
           Two complementary families. RNA SEQUENCE: RiNALMo (priority), RNA-FM,
           ERNIE-RNA (structural priors, relevant to splice variants).
           TRANSCRIPTOMIC: Geneformer or scGPT.
           New connectors plus version-keyed embedding caches. Sequencing note:
           these feed items 14 and 15, so they land before the fusion trunk.

 13. HETEROGENEOUS GNN / VAE / GAN / 3D CNN -- source:
           implementing_GNN_VAE_GAN_3DCNN. Four DIFFERENTIATED EVIDENCE
           INSTRUMENTS, not four more classifiers.
           MANDATORY per the source, and per this project's own history: graph
           splits that defeat the named leakage mechanisms, and NEGATIVE CONTROLS
           for gene ranking. Carried item: test_ablate_gnn currently SKIPS locally
           on torch_scatter / torch_sparse 0xc0000139, so graph coverage is absent
           on the Windows box -- confirm runnable before any GNN claim.

 14. KAN REPOSITIONING -- source: improved_KAN_implementat.
           ADDITION, NOT REPLACEMENT (confirmed 2026-07-27). A KANEncoder branch
           producing embeddings with quality and uncertainty metadata, alongside
           the existing KANClassifier, which REMAINS one of the thirteen permanent
           base models. FastKAN for production, pykan pinned for interpretation,
           symbolic analysis and spline visualisation.
           THE FATE OF THE ORIGINAL KANClassifier IS DECIDED AFTER RUN 17, NOT
           BEFORE. Precedent: item 6.16 -- KAN was silently absent from every
           Continuous Integration run for two months and a twelve-model ensemble
           reported normal metrics.
           Evaluate on scientific criteria as well as predictive: spline
           stability, biological plausibility, agreement with established
           attribution methods, reproducibility across retraining.

 15. FUSION v1, THEN JEPA -- source: JEPA_Implementation.
           THE SOURCE IMPOSES THE ORDER and it is adopted verbatim: expose
           embeddings; build masked supervised fusion v1; validate attribution;
           benchmark against the stacker; THEN add JEPA pretraining. JEPA does not
           start until fusion v1 is reproducible, calibrated, attribution-validated
           and benchmarked.
           Fusion v1 inputs, honestly scoped: Nucleotide Transformer pooled
           embedding, ESM-2 pooled/delta embedding, GNN pre-readout node
           embedding, TabularNN penultimate embedding, presence masks and learned
           absent tokens. RNA, pathology images and further foundation models are
           MASKED FUTURE SLOTS, not pretend inputs.
           HARD RULES: gene contribution is validated, not merely interpreted;
           Head A (pathogenicity) and Head B (disease category) attributions stay
           SEPARATE so disease priors cannot masquerade as discovered biology;
           disease-head evaluation is restricted to labels seen in training;
           missingness is modelled as structured signal, never zero-filled; the
           fusion-versus-stacker comparison is calibrated and paired; every
           embedding cache is version-keyed and reproducible.

 16. MIXTURE OF EXPERTS -- source: moe_integration.
           DUAL-GATE architecture. Allocation tells the system WHERE TO COMPUTE;
           relevance estimates WHICH MECHANISMS ARE SUPPORTED; reliability
           estimates WHETHER THE EVIDENCE IS TRUSTWORTHY. These are distinct,
           trained under distinct constraints, calibrated separately and admitted
           under separate gates. A normalized routing weight is NOT a relevance
           estimate.
           Panel S0 gates INTERPRETATION, not every later panel's execution.
           Requires the fusion trunk from item 15 to route over.
           See docs/PANEL_S0_ROUTING_IDENTIFIABILITY.md.

=====================================================================
TIER 3 -- RUN 17. Cannot launch until Tier 2 is complete.
=====================================================================

 17. RUN 17 -- planned and gated, NOT launched. docs/runs/RUN17_SCOPE.md
           (commit 94bf6ae). Last executed run: Run 16.
           ITS OWN PRE-LAUNCH GATES, all still open:
             * 6.18 stage 2 -- requirements.in / .lock / .txt remain a dual source
               of truth. Add the transformers and pandas ceilings to
               requirements.in, recompile UNDER Python 3.12, verify the new lock
               reproduces the known-good stack exactly, extend lockfile-check to
               the main pair, then make ONE hash-pinned file the single install
               target. Deliberate and separately verified, never a side effect.
             * 6.20 -- the drift monitor has an aggregate reference profile but no
               NEW-release feature matrix on a hosted runner, so it reports
               UNKNOWN (exit 4). Honest, red and loud, but not yet capable.
             * test_ablate_gnn skips locally on torch_scatter / torch_sparse
               0xc0000139 -- confirm runnable before Run 17 activates gnn_score.
             * pandas .fillna downcasting FutureWarning in variant_ensemble.py
               wants an explicit cast.

=====================================================================
CARRIED FORWARD -- recorded rather than preempted. Not gates on any tier.
=====================================================================

    a. as_meta() emits gene_id but never gene_symbol, so a seam-produced frame
       yields an EMPTY gene_errors list while its docstring claims to be the
       frame ClinicalEvaluator.evaluate expects. Pre-existing at 2e04bd9.
       Belongs with item 3's certification.
    b. EVALUATION_WIRING_AUDIT_2026-07-25.md:271 attributes per-gene analysis to
       gene_id; per-gene analysis reads gene_symbol.
    c. ValidationMetrics carries (0.0, 0.0) interval defaults -- fabricated
       evidence -- but is constructed nowhere. Deferred WITH a proof test that
       fails the moment anything constructs it.
    d. default=str remains at prediction_artifacts.py manifest and statistics
       writers; the eval-report writer was in scope and is fixed.
    e. metrics.py still lacks `from __future__ import annotations`.
    f. frozen=True on EvaluationReport: proven safe (one constructor, zero
       mutations) but an orthogonal mutability change.
    g. a generic cluster_id projection in the seam, so ClinicalEvaluator stops
       embedding schema-discovery policy at all.
    h. the Continuous Integration matrix runs 3.11 and 3.12 only and does not
       exercise the declared 3.10 floor.
    i. FIVE MONTE CARLO DROPOUT TESTS RUN NOWHERE. Enumerated 2026-07-26:
       tests/integration/test_mc_dropout_calibration.py holds five unconditional
       @pytest.mark.skip stubs, dormant since 2026-05-27, that skip on BOTH
       Windows and Linux. mc_dropout is one of the thirteen permanent base
       models, and these five are the tests of its epistemic-uncertainty claims,
       including the out-of-distribution claim that epistemic uncertainty is
       higher on held-out gene families. That model's uncertainty quantification
       is currently asserted by nothing, while the README markets uncertainty as
       a clinical product. Unblocking needs the Run 15 cohort, gene-family-
       disjoint splits, and expected-calibration-error infrastructure.
       LARGEST SCIENTIFIC DEBT IN THIS LIST, and it now sits in front of a much
       longer queue than when it was recorded.
    j. FOUR TESTS HAVE NO CONTINUOUS-INTEGRATION COVERAGE. The four in
       tests/unit/test_run17_postflight_paths.py (skipif sys.platform !=
       "win32") execute only on Windows; a regression there is invisible to the
       pipeline. Conversely test_preflight_data_paths.py:45 runs only on Linux
       and never on the development machine. Measured skip counts differ by
       platform -- Windows 7, Linux 9 confirmed -- against a suite size that is
       identical on both, because collection is platform-independent while
       skipping is not.


=====================================================================
ROADMAP delta -- 2026-07-27 (Tier 1 item 6, commit 2a)
=====================================================================

TIER 1 ITEM 6 SEQUENCE, restructured by ruling on 2026-07-27:

  commit 1   d3851a3   MetricResult to the vocabulary layer          LANDED
  commit 2   a6df4ef   the typed immutable metric registry           LANDED
  commit 2b* 974d426   controlled metadata vocabulary                LANDED
  commit 2a  (this)    fail-closed prediction-input contract         LANDED
  commit 2a-1  --      EvaluationPopulation; label selection moves    NEXT
  commit 2b  --       calibration binning + register all point estimates
  commit 3   --       schema-v3 report integration

  (*the earlier "2b" label was the vocabulary commit and is retained as landed
   history; the ruled sequence renumbers the remaining work as above.)

WITHDRAWN: an earlier package this session, "commit 3a", combined the binning
repair with a population repair built the OPPOSITE way -- kernels kept filtering
and merely recorded what they dropped, and one of its tests asserted
certification_eligible is True on a cohort containing non-finite model output. It
was withdrawn BEFORE installation; nothing from it reached the repository. Its
binning work is carried to the binning commit.

CORRECTION ON RECORD: metrics.evaluate was described as "already conforming" to
the fail-closed ruling on the strength of instrumentation showing no non-finite
value reaching any kernel from it. The instrumentation was right, the conclusion
was wrong: it reaches that state by filtering predictions itself and DISCLOSING
the narrowing. Transparency is not validity. It conforms to population-accounting
transparency only, is marked non-certifiable, and is pinned by a test.

DEFECT CLOSED. clean_arrays dropped non-finite rows on one joint mask over
labels, scores and probabilities alike. Labels and predictions no longer share a
mask: labels are SELECTED upstream by a named transitional selector, predictions
are VALIDATED and fail closed.

NEW CARRIED ITEM (l) -- THE TRANSITIONAL LABEL MASK. Label population selection
still lives in metrics.select_finite_reference_labels rather than in
EvaluationPopulation. This is a STAGED decision, not an oversight, and it is
tripwired by two tests that fail if the declaration or the named selector is
removed without the replacement arriving. One precise deletion target for commit
2a-1. NOT A GATE on anything else.

NEW CARRIED ITEM (m) -- metrics.evaluate IS A SURVIVOR-FILTERING PATH. It is
retained unchanged for compatibility and marked non-certifiable in its docstring.
Whether it is frozen permanently as historical compatibility or gains a strict
mode is a deliberate decision for its own commit and must not be made
incidentally. NOT A GATE.


=====================================================================
ROADMAP delta -- 2026-07-27 (Tier 1 item 6, commit 2a-1)
=====================================================================

  commit 1   d3851a3   MetricResult to the vocabulary layer          LANDED
  commit 2   a6df4ef   the typed immutable metric registry           LANDED
  commit 2b* 974d426   controlled metadata vocabulary                LANDED
  commit 2a  b22012a   fail-closed prediction-input contract         LANDED
  commit 2a-1 (this)   the evaluation population contract            LANDED
  commit 2b  --        calibration binning + register all point estimates
  commit 3   --        schema-v3 report integration

CLOSED. metrics.select_finite_reference_labels is RETIRED. Label eligibility is an
explicit, recorded restriction of an EvaluationPopulation. Carried item (l) -- the
transitional label mask -- is DISCHARGED, and its two tripwires were RETIRED
AGAINST THEIR REPLACEMENT rather than deleted: they now assert the selector is
gone and the seam states its contract in the present tense.

NEW CARRIED ITEM (n) -- cohort_version IS A WEAK PROVENANCE IDENTITY.
Deliberately NOT fixed in 2a-1; fixing it there would have combined exact
population identity, provenance-policy strength and certification admissibility in
one commit and forced twenty fixture edits unrelated to the population
abstraction.

    cohort_version call sites audited 2026-07-27:
        generic "v2": 20
        "v2-xyz": 1
        "v2-abc": 1
        "v1": 1

    Current mitigation:
        population_source_id also hashes the ordered variant_id sequence and
        partition, so distinct row sets remain distinguishable.

    Residual ambiguity:
        identical variants in identical order and partition, evaluated under
        different label/adjudication policies but the same generic
        cohort_version, produce the same population_source_id.

This is a DATASET-POLICY PROVENANCE defect, not a row-membership defect. The
dedicated commit should introduce separate identities with separate
responsibilities:

    dataset_identity        the concrete underlying cohort artifact or release
    cohort_policy_version   inclusion, exclusion, deduplication, label mapping,
                            adjudication and cleaning policy
    partition_identity      the exact split assignment and split-protocol version
    population_source_id    hashes those identities plus the ordered variant ids

Certification may then require STRONG, not merely non-blank, values -- expressed
as structured fields and an explicit admissibility check
(provenance.validate_for_certification), never as a string heuristic that rejects
"v2" while accepting "v2-final", which only measures formatting. NOT A GATE on
commit 2b or 3.

CARRIED ITEM (m) UNCHANGED. metrics.evaluate remains a survivor-filtering
compatibility path, marked non-certifiable, its label mask retained in
clean_arrays for that path alone.

PROCESS FINDING, 2026-07-27. An edit anchored to a test function and extended to
END OF FILE destroyed three functions appended after that anchor in the previous
commit, costing eight test cases -- including the parametrised gate test that had
closed a gap found by the previous commit's own sabotage matrix. It was caught by
the MEASURED collection delta (25 -> 19 where 27 was expected) and by nothing
else. Anchored edits must bound both ends; ratchet moves are measured, never
computed.
