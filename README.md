# Genomic Variant Pathogenicity Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Holdout AUROC](https://img.shields.io/badge/Holdout%20AUROC-0.9984-brightgreen.svg)]()
[![Variants](https://img.shields.io/badge/Training%20variants-1.49M-blue.svg)]()
[![Features](https://img.shields.io/badge/Tabular%20features-80-blue.svg)]()
[![Agents](https://img.shields.io/badge/Core%20agents-7-blueviolet.svg)]()
[![Tests](https://img.shields.io/badge/Tests-862%20passing-success.svg)]()

A production-grade, multi-modal machine learning system for the five-tier clinical
classification of human genomic variants -- **Pathogenic, Likely Pathogenic, Uncertain
Significance, Likely Benign, and Benign** -- in accordance with ACMG/AMP guidelines.

The system integrates genomic sequence data, population-stratified allele frequencies,
protein structural annotations, tissue-specific gene expression, and variant
co-classification evidence from a suite of biological databases into a unified
stacking-ensemble architecture, deployed as a production FastAPI REST service and
continuously supervised by an autonomous agent layer of seven core monitoring agents
-- plus a committed drift-detection suite -- over a typed inter-agent message bus.
Whole-slide histopathology imaging (TCGA) is a future multi-modal phase tracked in
`docs/ROADMAP.md`.

**Run 15 (sealed 2026-06-09, commit 032a2ab): Test AUROC 0.9984 / Val 0.9983 / unseen-gene-holdout 0.9988** on gene-stratified expert-reviewed ClinVar variants (Test n=304,711).
The model trains on a ~1.49 M-variant cohort drawn from ~2.49 M ClinVar missense variants, now a
80-feature matrix (Run 15: Vast.ai RTX 4090, ~11.5 h, ~$6). Earlier Run-8: holdout 0.9863 / test 0.9833 on 78 features.

---

## Architecture

The classifier operates as a three-branch fusion model wrapped in an autonomous
supervisory agent layer:

**Tabular Branch** -- A stacking meta-learner trained on out-of-fold predictions from
a roster of up to twelve base classifiers: Random Forest, XGBoost, LightGBM, CatBoost,
Gradient Boosting, Logistic Regression, a Kolmogorov-Arnold Network (KAN), a
PyTorch tabular neural network, a PyTorch 1D-CNN, Monte-Carlo Dropout, Deep Ensemble
Wrapper, and a Graph Attention Network over the STRING protein-protein interaction
graph. Input features span **80 dimensions** drawn from a suite of biological databases (further sources are being wired in the current data-expansion phase).

**Sequence Branch** -- A PyTorch 1D-CNN operating over 101 bp genomic context windows
(one-hot encoded) combined with ESM-2 protein-language-model features (HuggingFace
`transformers` backend). Two signals are derived: the scalar L2 embedding-delta
(`esm2_delta_norm`, secondary) and -- as of Phase 1 -- the primary log-likelihood-ratio
`esm2_llr` (`logit[mut] - logit[wt]` from the ESM-2 650M masked-LM head; WT-marginal by
default, masked-marginal opt-in). `esm2_llr` is SIGNED (negative = more damaging) and
enters the ensemble as a CONTINUOUS feature -- its sign is not a class label (even benign
variants score negative), so the model learns the threshold. ESM-2 silent-zero failure
modes are detected by `tests/unit/test_esm2_activation.py` per `INCIDENT_2026-04-17`.

**Histopathology Branch (planned -- future multi-modal expansion).** A ResNet-50 branch
over TCGA whole-slide tiles is a roadmap ambition for the multi-modal program; it is not
yet implemented. The current system is the tabular variant classifier described above.
Image, RNA, and protein-structure modalities are tracked as future phases in `docs/ROADMAP.md`.

```
ClinVar . gnomAD v4.1 . FinnGen R12 . 1000 Genomes . AlphaMissense . SpliceAI
. EVE . OMIM . ClinGen . dbNSFP v4.7 . GTEx v10 . UniProt . LOVD . AlphaFold . STRING
. CADD v1.7 . PhyloP v1 . HGMD Professional . dbSNP b156
                              |
               14-step Spark ETL annotation pipeline
                              |
            78-feature engineering (engineer_features)
                              |
     +-----------+------------+------------+-----------+
     |                        |                        |
12-model stacking        GNN (GAT)              ESM-2 embeddings
RF . XGBoost . LGBM      STRING DB PPI          protein language
CatBoost . GBM . LR      graph topology         model embeddings
KAN . tabular_nn                                       |
cnn_1d . MCDrop                                 PyTorch 1D-CNN
. DeepEnsemble                                  sequence branch (101 bp)
     |                        |                        |
     +-----------+------------+----------------+-------+
                              |
                    ResNet-50 Histopathology Branch [PLANNED - see ROADMAP]
                    TCGA-BRCA . TCGA-LUAD . TCGA-COAD
                    224x224 tiles . 20x magnification
                              |
                  Stacking meta-learner (LR)
                  + Platt calibration
                  + Conformal prediction intervals
                  + Epistemic/aleatoric uncertainty
                              |
              ClinicalEvaluator (AUROC, AUPRC, ECE,
              calibration, per-consequence breakdown)
                              |
     +------------------------+------------------------+
     |                                                 |
  FastAPI REST API                       Autonomous Agent Layer
  7 endpoints . auth . rate-limit        7 core agents + drift suite
  Docker . GHCR . CI/CD                  typed inter-agent message bus
  Prometheus . Grafana                   shared state + orchestrator
                                         continual learning + EWC
                                         versioned model registry
                                         shadow -> production promotion
```

## Key properties

**Clinically robust** -- Five-tier ACMG/AMP classification (Pathogenic to Benign) with
empirically calibrated probability thresholds, conformal prediction intervals at
configurable coverage levels, and per-variant uncertainty scores (epistemic +
aleatoric, MC-Dropout and Deep-Ensemble) that flag cases requiring human expert
review.

**Temporally aware** -- A dedicated continual learning pipeline runs on every ClinVar
monthly release. It detects three classes of scientific drift -- feature/covariate
drift as gnomAD cohorts expand and functional score models are retrained, label
drift as ClinVar reclassifies variants, and concept drift as new biology changes
what features predict pathogenicity -- and when drift exceeds configurable thresholds,
adaptive retraining is triggered automatically using Elastic Weight Consolidation
(EWC) to prevent catastrophic forgetting of stable biological signal.

**Scientifically current** -- Integrates 18 biological databases spanning population
genetics (gnomAD v4.1, FinnGen R12 with 500,348 Finnish individuals, 1000 Genomes
Phase 3 across 5 continental strata), evolutionary conservation (PhyloP, GERP, EVE),
deep learning functional predictions (AlphaMissense, SpliceAI, ESM-2, CADD v1.7),
gene-disease knowledge bases (OMIM, ClinGen, LOVD, HGMD), protein structure
(AlphaFold pLDDT, UniProt), tissue expression (GTEx v10), splice mechanics
(MaxEntScan), variant identity (dbSNP b156, dbNSFP v4.7), and protein-protein
interaction topology (STRING DB v12).

**Phenotypically grounded (planned).** A future TCGA histopathology branch will link
variant pathogenicity classification to observable tumor-tissue morphology across breast,
lung adenocarcinoma, and colorectal cancer cohorts -- a multi-modal capability on the
roadmap (`docs/ROADMAP.md`), not yet implemented.

**Production deployed** -- FastAPI service on port 8000, multi-stage Dockerfile
(builder / api / trainer targets), image published to GHCR
(`ghcr.io/monzia-moodie/genomic-variant-api`), CI/CD via GitHub Actions with
lockfile checks, full pytest sweep, Docker smoke tests, and monthly scheduled
drift monitoring.

**Autonomously maintained** -- A 13-agent monitoring layer (DataFreshnessAgent,
VersionMonitorAgent, SchemaDriftAgent, ConceptDriftAgent, LabelShiftAgent,
CalibrationDriftAgent, InfrastructureDriftAgent, FairnessSubgroupAgent,
AdversarialSubmissionAgent, AnnotationPolicyAgent, InterpretabilityAgent,
LiteratureScoutAgent, TrainingLifecycleAgent) communicates over a typed
inter-agent message bus (`agent_layer/message_bus.py`, 34/34 tests passing on
Python 3.14.3) to continuously monitor upstream databases, detect distribution
shift, trigger targeted retraining, and produce SHAP-based interpretability audits
without manual intervention.

**Operationally hardened** -- Dual-layer preflight gates (local
`scripts/preflight_check.py` enforces clean git tree, HEAD == origin/main, full
pytest suite, GCS object presence, and connector-importability; on-VM
`scripts/preflight_vm.sh` validates CUDA, data files on container FS, and a
1000-row LightGBM smoke fit BEFORE GPU billing starts). Multi-cloud training
runbooks for GCP (`gcp_run{6,7,8}_startup.sh`), Lambda Labs
(`lambda_run8_startup.sh`), and Vast.ai (`launch_run{9,10}_vm.sh`). An
append-only `docs/CHANGELOG.md` (1,500+ lines, searchable by error string) and
a structured `docs/incidents/` directory record every root cause and fix.
**501/501 unit tests** and integration suite green at HEAD.

## Tabular model roster

| Family | Implementations | Status |
|--------|-----------------|--------|
| Gradient-boosted trees | LightGBM, XGBoost, CatBoost, scikit-learn GBM | Production |
| Bagged trees | Random Forest | Production |
| Linear | Logistic Regression (also stacking meta-learner) | Production |
| Kolmogorov-Arnold | KAN (pykan / efficient-kan; MLP fallback) | Re-enabled 2026-04-20 |
| Neural -- tabular | `TabularNNClassifier` (PyTorch, BatchNorm1d + Dropout) | Migrated TF -> PyTorch (Run 8 final) |
| Neural -- sequence | `CNN1DClassifier` (PyTorch, Conv1d + AdaptiveMaxPool1d) | Migrated TF -> PyTorch (Run 8 final) |
| Bayesian uncertainty | `MCDropoutWrapper`, `DeepEnsembleWrapper` | epistemic + aleatoric decomposition |
| Graph | 3-layer GAT over STRING PPI (gene-level prior) | Production |
| Foundation model | ESM-2 650M: `esm2_llr` LLR (primary) + scalar L2 delta (secondary), HF transformers | Phase 1 done; full-cohort scoring after Run-16 coord-sync |

## Feature set (80 features)

| Group | Count | Key features |
|-------|-------|-------------|
| Allele frequency | 6 | af_raw, af_log10, af_is_absent, af_is_ultra_rare |
| Variant type | 7 | ref_len, alt_len, len_diff, is_snv, is_insertion, is_deletion |
| Consequence | 6 | consequence_severity, is_loss_of_function, is_missense, is_splice |
| Functional scores | 9 | CADD, SIFT, PolyPhen-2, REVEL, PhyloP, GERP, AlphaMissense, SpliceAI, EVE |
| Score flags | 5 | cadd_high, sift_deleterious, polyphen_probably_damaging, n_tools_pathogenic |
| Gene-level | 4 | gene_constraint_oe, n_pathogenic_in_gene, gene_has_known_disease |
| gnomAD constraint | 4 | loeuf, syn_z, mis_z, pli_score (v4.1) |
| Protein (UniProt) | 2 | has_uniprot_annotation, n_known_pathogenic_protein_variants |
| Expression (GTEx) | 6 | gtex_max_tpm, gtex_tissue_specificity, gtex_is_eqtl, gtex_max_abs_effect |
| Gene-disease | 3 | omim_n_diseases, omim_is_autosomal_dominant, clingen_validity_score |
| HGMD | 2 | hgmd_is_disease_mutation, hgmd_n_reports |
| LOVD | 1 | lovd_variant_class (ordinal 0-4) |
| Chromosome | 3 | is_autosome, is_sex_chrom, is_mitochondrial |
| GNN-derived | 1 | gnn_score (GAT over STRING PPI graph) |
| RNA splice | 4 | maxentscan_score, dist_to_splice_site, exon_number, is_canonical_splice |
| Protein structure | 4 | alphafold_plddt, solvent_accessibility, secondary_structure_context, dist_to_active_site |
| 1000 Genomes AF | 5 | af_1kg_afr, af_1kg_eur, af_1kg_eas, af_1kg_sas, af_1kg_amr |
| FinnGen R12 AF | 3 | finngen_af_fin, finngen_af_nfsee, finngen_enrichment |
| Reactome | 1 | reactome_pathway_count |
| ESM-2 (650M) | 2 | esm2_delta_norm (secondary), esm2_llr (primary, signed LLR) |
| Reserved (Deep Ensemble) | 2 | uncertainty_epistemic, uncertainty_aleatoric |

`TABULAR_FEATURES` and `engineer_features()` are kept in sync by a runtime
assertion at the bottom of the engineering function (per CHANGELOG 2026-04-16).

## Drift detection and continual learning

### Statistical detectors

- **PSI (Population Stability Index)** -- per-feature, runs on every data source update
- **Kolmogorov-Smirnov test** -- nonparametric, continuous features
- **Maximum Mean Discrepancy (MMD)** -- kernel-based joint distribution test
- **ADWIN** -- adaptive windowing detector for streaming variant ingestion
- **Szekely-Rizzo energy statistic** -- sensitive to distribution shape changes
- **ClinVar reclassification tracker** -- monitors flip rate in training set monthly

### Adaptive retraining

- **EWC (Elastic Weight Consolidation)** -- protects important weights during retraining
- **Online EWC** -- running Fisher estimate across multiple ClinVar releases
- **LSIF importance weighting** -- density ratio estimation for sample re-weighting
- **Temporal sample decay** -- exponentially downweights older ClinVar submissions
- **TreeEWCProxy** -- gradient-boosted-tree analogue of EWC for non-differentiable bases

### Lifecycle

- **Versioned model registry** (`monitoring/registry.py`) -- staging -> shadow -> production
- **Shadow deployment** -- new models run in parallel before promotion
- **Connector silent-zero hardening** -- regression tests assert that connector fallbacks
  fail loud, not silently return 0.0 (post-`INCIDENT_2026-04-17`, post-`INCIDENT_2026-05-02`)

## Autonomous agent layer (13 specialised agents)

Located under `src/genomic_variant_classifier/agent_layer/`, with each agent
inheriting from `BaseAgent` and communicating over a typed `message_bus`.

| Agent | Concern |
|-------|---------|
| `DataFreshnessAgent` | Polls ClinVar, gnomAD, AlphaMissense, SpliceAI manifests; raises when stale |
| `VersionMonitorAgent` | Tracks upstream dataset version numbers and breaking-change deltas |
| `SchemaDriftAgent` | Detects column/dtype changes in incoming connector parquets |
| `ConceptDriftAgent` | Monitors feature -> label relationship stability via residual analysis |
| `LabelShiftAgent` | Tracks prior class probabilities across ClinVar monthly releases |
| `CalibrationDriftAgent` | Watches ECE / reliability diagrams over time |
| `InfrastructureDriftAgent` | Catches dependency / runtime drift (sklearn / lightgbm / CUDA) |
| `FairnessSubgroupAgent` | Per-ancestry, per-consequence, per-gene-tier performance audit |
| `AdversarialSubmissionAgent` | Flags suspect or out-of-distribution prediction requests |
| `AnnotationPolicyAgent` | Enforces source-priority and provenance rules at ingestion |
| `InterpretabilityAgent` | SHAP-based audit per release; persists explanations for review |
| `LiteratureScoutAgent` | bioRxiv / PubMed feed for new functional-score models and ClinVar policy changes |
| `TrainingLifecycleAgent` | Orchestrates retraining trigger -> EWC -> shadow -> promotion |

`agent_layer/orchestrator.py` schedules agent execution and routes typed messages;
`agent_layer/shared_state.py` provides a JSON-persisted shared blackboard;
`agent_layer/test_message_bus.py` exercises the bus (34/34 passing on Py 3.14.3).

## REST API

```
GET  /health          Liveness + readiness
GET  /info            Model metadata, 80 features, drift status
GET  /metrics         Prometheus metrics
GET  /gene/{symbol}   Gene-level feature lookup
GET  /rsid/{rs_id}    rs-ID resolution + prediction
POST /predict         Single variant -> 5-tier classification + uncertainty
POST /batch           Up to 1,000 variants
```

Auth: X-API-Key header; rate limiting via `slowapi`; structured JSON logging;
Prometheus `/metrics` instrumentation via `prometheus-fastapi-instrumentator`.

## Performance

Evaluated on **349,067 held-out variants** (gene-stratified; no gene appears in both
train and test). Training cohort: 1,197,216 variants (20.3% pathogenic) at the
Run 8 baseline; recent runs (Run 14/15) use the full ~1.49 M-variant cohort.

| Metric | Value |
|--------|-------|
| Holdout AUROC (Run 8 baseline) | **0.9847** |
| Brier score | 0.0584 |
| Sensitivity @ 90% specificity | 0.900 |
| Specificity @ 90% sensitivity | 0.918 |
| Evaluation set | 349,067 variants, gene-stratified |
| Training set | 1,197,216 variants |
| Label source | ClinVar expert-reviewed (tier 2+) |

### Per-model performance (validation set, Run 8 baseline)

| Model | AUROC | AUPRC | F1 (macro) | MCC | Brier |
|-------|-------|-------|-----------|-----|-------|
| **gradient_boosting** | **0.9756** | 0.9190 | 0.8876 | 0.7758 | 0.0497 |
| lightgbm | 0.9751 | 0.9171 | 0.8651 | 0.7522 | 0.0700 |
| logistic_regression | 0.9747 | 0.9133 | 0.8627 | 0.7499 | 0.0696 |
| catboost | 0.9744 | 0.9153 | 0.8657 | 0.7530 | 0.0708 |
| xgboost | 0.9743 | 0.9124 | 0.8471 | 0.7276 | 0.0930 |
| ENSEMBLE_STACKER | 0.9709 | 0.8572 | 0.8791 | 0.7630 | 0.0584 |
| random_forest | 0.9681 | 0.8892 | 0.8725 | 0.7536 | 0.0663 |

### Recent training-run history

| Run | Date | Hardware | Holdout AUROC | Notes |
|-----|------|----------|--------------:|-------|
| Run 14 | 2026-06-03 | Vast.ai RTX 4090 | 0.9975 | commit eb11029; SNV/indel leakage traced entirely to null ref/alt records (no real-allele leakage) |
| **Run 15** | **2026-06-09** | **Vast.ai RTX 4090** | **0.9984** (test) | **commit 032a2ab; Val 0.9983 / unseen-gene-holdout 0.9988; 79 features. ESM-2 650M LLR + 80-feature contract added 2026-06-10 (Phase 1), realized at next regen** |

Per-run details live in `docs/sessions/SESSION_<date>.md` and root-cause records
in `docs/incidents/INCIDENT_<date>_<topic>.md`.

## Operational rigour

- **Dual-layer preflight** -- `scripts/preflight_check.py` (local) gates every
  launch against clean git, HEAD == origin/main, full pytest, GCS object
  presence, and importability of `transformers`/`torch`. `scripts/preflight_vm.sh`
  (on-VM) gates against CUDA, data-file presence on the container FS, and a
  1,000-row LightGBM smoke fit -- catching the sklearn/lightgbm `force_all_finite`
  skew BEFORE GPU billing starts.
- **Multi-cloud training** -- runbooks for GCP (`gcp_run{6,7,8}_startup.sh`,
  `trap EXIT`-based shutdown for guaranteed model upload), Lambda Labs
  (`lambda_run8_startup.sh`), and Vast.ai (`launch_run{9,10}_vm.sh`,
  non-interactive `vastai destroy`, auto-tmux session protection).
- **Append-only CHANGELOG** -- `docs/CHANGELOG.md` is searchable by exact error
  string; every session records *Attempted / Failed / Fixed / Learned*.
- **INCIDENT system** -- ten root-cause records to date covering GPU quota,
  silent-zero connectors (SpliceAI, ESM-2, EVE, LOVD), pickle nested-class
  serialisation, GCP billing deletion, GNN key errors, and split duplicates.
- **Session logs** -- `docs/sessions/SESSION_<date>.md` is the chronological
  record of every working day; each session entry links forward into the
  CHANGELOG and INCIDENTS.
- **Test depth** -- 501/501 unit tests + integration tests, including
  regression tests for every silent-zero failure mode and an inter-agent
  message-bus suite (34/34 on Py 3.14.3).
- **Recovery artifacts** -- `logs/training/run9_master.log.recovery.md`
  captures the last 100 lines of a master training log after a VM destroy
  beat the SCP-back step.

## Repository structure

```
src/genomic_variant_classifier/
  agent_layer/   - 13 specialised agents + typed message_bus + orchestrator + shared_state
  api/           - FastAPI service (7 endpoints), auth, schemas, InferencePipeline
  data/          - 18 database connectors + Spark ETL + DataPrepPipeline + real_data_prep
  evaluation/    - ClinicalEvaluator, benchmark framework, conformal prediction, metrics
  features/      - engineer_features (80-column pipeline, runtime sync assertion)
  models/        - VariantEnsemble, GNN (GAT), KAN, MC-Dropout, CatBoost wrapper
  monitoring/    - DriftDetector, ClinVarTracker, ModelRegistry
  pipelines/     - RNA splice pipeline, protein structure pipeline
  reports/       - HTML report generator
  training/      - ContinualLearner, EWC, OnlineEWC, TreeEWCProxy
  utils/         - helpers, shared utilities
scripts/
  run_phase2_eval.py        - main training entry point
  run9_ablations.py         - LOCO ablation harness (14 ablation targets)
  export_model.py           - InferencePipeline serialisation + smoke test
  run_drift_monitor.py      - monthly drift check CLI (exit 0/1/2/3)
  calibrate_thresholds.py   - empirical ACMG threshold calibration
  validate_external.py      - external cohort validation (LOVD, UK Biobank)
  conformal_prediction.py   - split conformal intervals
  benchmark.py / benchmark_polars.py - algorithm and ETL benchmarks
  preflight_check.py        - local pre-launch gate
  preflight_vm.sh           - on-VM post-SSH gate
  gcp_run{6,7,8}_startup.sh, lambda_run8_startup.sh, launch_run{9,10}_vm.sh
docs/
  CHANGELOG.md              - append-only session ledger (1,500+ lines)
  ROADMAP.md, PHASE_3_ROADMAP.md
  incidents/                - 10 root-cause records
  sessions/                 - chronological session logs
  reviews/, validated/, hypotheses/
models/
  registry.json             - versioned model registry
  phase2_pipeline.joblib    - current production InferencePipeline
  drift_reference.pkl       - DriftDetector reference snapshot
configs/  default.yaml, config.yaml
deploy/   grafana/, prometheus.yml
tests/    unit/, integration/, fixtures/, smoke_test_imports.py
```

## Quickstart

```bash
# Run the API
MODEL_PATH=models/phase2_pipeline.joblib uvicorn genomic_variant_classifier.api.main:app --port 8000

# Classify a variant
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"chrom":"17","pos":43115726,"ref":"AAC","alt":"A",
       "consequence":"frameshift_variant","allele_freq":0.0,
       "alphamissense_score":0.95,"n_pathogenic_in_gene":2800}'

# Run monthly drift check
python scripts/run_drift_monitor.py \
  --reference-splits outputs/phase2_with_gnomad/splits/ \
  --new-clinvar  data/processed/clinvar_grch38_2024_07.parquet \
  --old-clinvar  data/processed/clinvar_grch38_2024_01.parquet \
  --output-dir   outputs/drift_reports/2024_07/ \
  --auto-retrain

# Train (full ensemble, 80 features)
python scripts/run_phase2_eval.py \
  --parquet data/processed/clinvar_grch38.parquet \
  --output  outputs/run10/full \
  --lovd-path data/external/lovd/lovd_all_variants.parquet \
  --dbnsfp-path data/external/dbnsfp/dbnsfp_grch38.parquet

# Local preflight before a paid GPU run
python scripts/preflight_check.py

# Docker
docker compose up api
```

## Roadmap

- **Phase 3 -- Polars ETL evaluation.** Replace pandas bottlenecks in the
  annotation pipeline; benchmark already shows ~3.3x speedup on the
  gnomAD-constraint join (500 K variants). See `scripts/benchmark_polars.py`
  and `docs/PHASE_3_ROADMAP.md`.
- **Phase 4 -- Algorithm expansion and benchmarking.** ESM-2 upgraded to the 650M
  masked-LM with a log-likelihood-ratio feature (`esm2_llr`, Phase 1 -- done); next are
  ESM C 600M and a full-cohort regen after the Run-16 coordinate-index sync. Run KAN
  through the benchmark harness against MLP, integrate Deep Ensemble uncertainty into
  VUS flagging, and fuse GNN gene embeddings with `TABULAR_FEATURES` before stacking.
  Tracked in `docs/ROADMAP.md`.
- **Phase 5 -- Clinical validation and manuscript.** Prospective validation
  on BRCA1/2, TP53, PTEN, ATM panels; comparison against ClinVar star-rating
  on expert-reviewed variants; model card; manuscript draft.
- **Deferred -- Psychiatric GWAS pleiotropy.** Integration of the OpenMed PGC
  dataset (1.14 B rows, 52 PGC meta-analyses, 12 psychiatric conditions) as
  five new locus-level features (`gwas_psych_min_pval`, `_hit_count`,
  `_disorder_breadth`, `_max_neg_log10p`, `_is_lead_snp`). Pre-aggregated
  via Polars (filter to p < 5e-8 -> per-rsID summary). Gated on Phase 3
  Polars evaluation and Run 6+ completion; see `ROADMAP_PSYCH_GWAS_ENTRY.md`.

The roadmap is a living document -- see `docs/ROADMAP.md` for the live
checklist and `docs/CHANGELOG.md` for what has actually shipped.

## Author

**Monzia Moodie** -- [@monzia-moodie](https://github.com/monzia-moodie)

## License

MIT License
