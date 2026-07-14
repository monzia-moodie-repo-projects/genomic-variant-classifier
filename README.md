# Genomic Variant Pathogenicity Classifier

[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Features](https://img.shields.io/badge/Tabular%20features-95-blue.svg)]()
[![Agents](https://img.shields.io/badge/Core%20agents-13-blueviolet.svg)]()
[![Tests](https://img.shields.io/badge/Tests-1926%20passing-success.svg)]()
[![Performance](https://img.shields.io/badge/Headline%20metrics-withdrawn%20pending%20Run%2017-orange.svg)]()

A production-grade, multi-modal machine learning system for the five-tier clinical
classification of human genomic variants -- **Pathogenic, Likely Pathogenic, Uncertain
Significance, Likely Benign, and Benign** -- in accordance with ACMG/AMP guidelines.

The system integrates genomic sequence data, population-stratified allele frequencies,
protein structural annotations, tissue-specific gene expression, and variant
co-classification evidence from a suite of biological databases into a unified
stacking-ensemble architecture, deployed as a production FastAPI REST service and
continuously supervised by an autonomous agent layer of thirteen specialised agents
-- plus a committed drift-detection suite -- over a typed inter-agent message bus.
Whole-slide histopathology imaging (TCGA) is a future multi-modal phase tracked in
`docs/ROADMAP.md`.

> ## ⚠️ ALL PERFORMANCE NUMBERS HAVE BEEN WITHDRAWN (2026-07-14)
>
> **Every AUROC previously quoted in this README came from Run 15, and Run 15 was produced with
> 36 of its 78 features CONSTANT ZERO — 46% of the feature space, across 1,038,974 variants.**
>
> GTEx, 1000 Genomes, FinnGen, AlphaFold/protein structure, MaxEntScan, UniProt, ESM-2, EVE,
> dbSNP, PhyloP, OMIM and ClinGen were all silently stubbed to 0.0 by connectors that return
> zeros instead of raising when their source file is absent. The headline metric was produced by
> the **38 features that were real**. Discovered 2026-07-13; see `docs/ROADMAP.md` §6.21.
>
> The surviving live features are dominated by `cadd_phred`, `revel_score`, `sift_score`,
> `polyphen2_score` and `n_tools_pathogenic` — in-silico predictors **themselves trained on
> ClinVar**. That circularity is now the leading candidate for why so high a score was
> attainable at all (`docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md`).
>
> **The figures themselves are deliberately not restated here, even to disown them.** A number
> in a withdrawal notice is still a number people screenshot and quote. They are recorded in
> `docs/ROADMAP.md` §6.21 and `docs/audits/README_AUDIT_2026-07-14.md`, in context, where the
> caveat cannot be separated from the value.
>
> **No performance figure from this project should be quoted, cited, or relied upon until
> Run 17 completes.** A guard now aborts training before a single model is fitted if any
> declared feature is constant (`feature_census`, roadmap 6.21), so Run 17 cannot repeat this.
> A clean table will be populated here when it lands.

The model trains on a cohort drawn from ClinVar, using a **95-feature** tabular contract
(`EXPECTED_TABULAR_FEATURE_COUNT`, enforced by `tests/unit/test_feature_count_contract.py`).

---

## Architecture

The classifier operates as a three-branch fusion model wrapped in an autonomous
supervisory agent layer:

**Tabular Branch** -- A stacking meta-learner (Logistic Regression) trained on out-of-fold
predictions from a roster of **13 base classifiers**:

| # | base model | family |
|---|---|---|
| 1 | `random_forest` | bagged trees |
| 2 | `xgboost` | gradient-boosted trees |
| 3 | `lightgbm` | gradient-boosted trees |
| 4 | `svm` | Nystroem-approximated support vector machine |
| 5 | `svm_bagged_rbf` | bagged radial-basis-function SVM |
| 6 | `logistic_regression` | linear (also the stacking meta-learner) |
| 7 | `gradient_boosting` | gradient-boosted trees (scikit-learn) |
| 8 | `catboost` | gradient-boosted trees (categorical-aware) |
| 9 | `tabular_nn` | PyTorch tabular neural network |
| 10 | `cnn_1d` | PyTorch 1D convolutional network (consumes the one-hot DNA window) |
| 11 | `kan` | Kolmogorov-Arnold Network |
| 12 | `mc_dropout` | Monte-Carlo Dropout (epistemic uncertainty) |
| 13 | `deep_ensemble` | Deep Ensemble wrapper (epistemic + aleatoric) |

> **This README said "twelve" until 2026-07-14, and that was dangerous, not cosmetic.**
> The old list made two compounding errors that nearly cancelled: it **omitted `svm` and
> `svm_bagged_rbf`** (two real base classifiers), and it **counted the Graph Attention Network
> as a base classifier** — which it is not. The GAT is not in the ensemble roster; it produces
> `gnn_score`, a **feature**, and contributes no out-of-fold column. 13 − 2 + 1 = 12.
>
> Roadmap 6.6a is the defect in which **a 13-model ensemble silently became a 12-model
> ensemble**: KAN's out-of-fold step raised, a bare `except Exception` swallowed it, and the
> model vanished from `trained_models_`, from the blend, and from every comparison artifact —
> while the run reported normal metrics. **Anyone checking a run's twelve models against a
> README that said twelve would have concluded the ensemble was complete.** The document would
> have concealed the exact defect it took weeks to find.
>
> `VariantEnsemble.ensemble_completeness_` now records roster / trained / dropped / complete
> into the run artifacts, `EnsembleConfig.allow_base_model_dropout = False` makes a failed base
> model **raise**, and `tests/unit/test_readme_claims.py` fails the suite if this table and
> `_build_estimators()` ever disagree again.

Input features span **95 dimensions** drawn from a suite of biological databases (further sources are being wired in the current data-expansion phase).

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
ClinVar . gnomAD v4.1 . FinnGen R12/R13 . 1000 Genomes . AlphaMissense . SpliceAI
. EVE . OMIM . ClinGen . dbNSFP . GTEx v11 . UniProt . LOVD . AlphaFold . STRING
. CADD . PhyloP . dbSNP b156 . COSMIC . KEGG . Reactome . RNA-seq
   (HGMD is NOT a source -- never licensed, never wired; see below)
                              |
               14-step Spark ETL annotation pipeline
                              |
            95-feature engineering (engineer_features)
                              |
     +-----------+------------+------------+-----------+
     |                        |                        |
13-model stacking        GNN (GAT)              ESM-2 embeddings
RF . XGBoost . LGBM      STRING DB PPI          protein language
CatBoost . GBM . LR      graph topology         model embeddings
svm . svm_bagged_rbf     -> gnn_score, a               |
KAN . tabular_nn            FEATURE, not a       PyTorch 1D-CNN
cnn_1d . MCDrop             base classifier      sequence branch (101 bp)
. DeepEnsemble
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
  7 endpoints . auth . rate-limit        13 specialised agents
  Docker . GHCR . CI/CD                  typed inter-agent message bus
  Prometheus . Grafana                   shared state + orchestrator
                                         continual learning + EWC
                                         versioned model registry
                                         shadow -> production promotion
```

## Key properties

**Clinically robust** -- Five-tier ACMG/AMP classification (Pathogenic, Likely pathogenic,
Uncertain significance, Likely benign, Benign), conformal prediction intervals at
configurable coverage levels, and per-variant uncertainty scores (epistemic +
aleatoric, MC-Dropout and Deep-Ensemble) that flag cases requiring human expert
review. The model is trained on high-confidence ClinVar labels (Pathogenic/Likely
pathogenic vs Benign/Likely benign; VUS and conflicting records are excluded to avoid label
noise) and the five tiers are recovered from the calibrated probability at inference
(`api/schemas.py::score_to_classification`).

> **The tier boundaries are NOT currently calibrated.** `models/classification_thresholds.json`
> does not exist, so `_load_thresholds()` serves hard-coded defaults — and it does so behind a
> bare `except Exception: pass`, meaning a malformed calibration file would also be silently
> ignored. Run `scripts/calibrate_thresholds.py` against Run 17 and commit its output before
> any clinical claim is made about where the Pathogenic / Likely-pathogenic boundary sits.
> (2026-07-14; see `docs/audits/README_AUDIT_2026-07-14.md` §1.1.)

**Temporally aware** -- A drift-monitoring and continual-learning pipeline is wired to the
ClinVar monthly release cycle. It is designed to detect three classes of scientific drift --
feature/covariate drift as gnomAD cohorts expand and functional score models are retrained,
label drift as ClinVar reclassifies variants, and concept drift as new biology changes what
features predict pathogenicity -- and to trigger adaptive retraining using Elastic Weight
Consolidation (EWC) to prevent catastrophic forgetting of stable biological signal.

> **Status, 2026-07-14 — read this before relying on the above.** Until 2026-07-13 the
> scheduled monitor **had never performed a single check**: it fired monthly, created an empty
> directory, observed the directory was empty, and reported `drift_level=none` with a green
> tick. No drift was ever detected because none was ever measured, and no retraining was ever
> triggered (roadmap 6.20). The lie is fixed — the reference is now a committed aggregate
> profile whose Population Stability Index is bit-identical to the raw matrix, "not checked"
> has its own exit code (4), and the workflow goes **red** when it cannot see data. But the
> monitor still has no new-release feature matrix on a hosted runner, so **it currently reports
> UNKNOWN every month**. That is the honest state, and it is deliberate.

**Scientifically current** -- Integrates a broad suite of biological databases spanning population
genetics (gnomAD v4.1, FinnGen R12 with 500,348 Finnish individuals, 1000 Genomes
Phase 3 across 5 continental strata), evolutionary conservation (PhyloP, GERP, EVE),
deep learning functional predictions (AlphaMissense, SpliceAI, ESM-2, Nucleotide Transformer,
CADD), gene-disease knowledge bases (OMIM, ClinGen, LOVD), protein structure
(AlphaFold pLDDT, UniProt), tissue expression (GTEx v11), splice mechanics
(MaxEntScan), pathway membership (KEGG, Reactome), somatic recurrence (COSMIC),
variant identity (dbSNP b156, dbNSFP), and protein-protein
interaction topology (STRING DB v12). **HGMD is not among them** — see the feature-set section.

**Phenotypically grounded (planned).** A future TCGA histopathology branch will link
variant pathogenicity classification to observable tumor-tissue morphology across breast,
lung adenocarcinoma, and colorectal cancer cohorts -- a multi-modal capability on the
roadmap (`docs/ROADMAP.md`), not yet implemented.

**Containerised** -- FastAPI service on port 8000 with a multi-stage Dockerfile
(builder / api / trainer targets) that builds the `genomic-variant-api` image locally,
plus a scheduled GitHub Actions drift-monitoring workflow. Publishing the image to a
container registry such as GHCR and a full build/test CI pipeline are roadmap items.

**Autonomously maintained** -- A monitoring layer of thirteen specialised agents (DataFreshnessAgent,
VersionMonitorAgent, SchemaDriftAgent, ConceptDriftAgent, LabelShiftAgent,
CalibrationDriftAgent, InfrastructureDriftAgent, FairnessSubgroupAgent,
AdversarialSubmissionAgent, AnnotationPolicyAgent, InterpretabilityAgent,
LiteratureScoutAgent, TrainingLifecycleAgent) communicates over a typed
inter-agent message bus (`agent_layer/message_bus.py`, 35/35 tests passing on
Python 3.12) to monitor upstream databases, detect distribution
shift, trigger targeted retraining, and produce SHAP-based interpretability audits.

> This README previously claimed the bus suite passed on **Python 3.14.3** here, and on
> **3.12.10** eighty lines further down. The project runs **3.11 and 3.12**. Python 3.14 is the
> version under which `requirements.txt` was mis-compiled, silently omitting `torch`,
> `torch-geometric`, `networkx`, `numba`, `pandera`, `pyspark` and `river` because torch has no
> 3.14 wheels (roadmap 6.18). Corrected 2026-07-14.

**Operationally hardened** -- Dual-layer preflight gates (local
`scripts/preflight_check.py` enforces clean git tree, HEAD == origin/main, full
pytest suite, GCS object presence, and connector-importability; on-VM
`scripts/preflight_vm.sh` validates CUDA, data files on container FS, and a
1000-row LightGBM smoke fit BEFORE GPU billing starts). Multi-cloud training
runbooks for GCP (`gcp_run{6,7,8}_startup.sh`), Lambda Labs
(`lambda_run8_startup.sh`), and Vast.ai (`launch_run{9,10}_vm.sh`). An
append-only `docs/CHANGELOG.md` (1,500+ lines, searchable by error string) and
a structured `docs/incidents/` directory record every root cause and fix.
**1,926 tests passing** (1,933 collected, 7 skipped) at HEAD, guarded by a suite-size
ratchet (`tests/EXPECTED_SUITE_SIZE`) that fails the build in BOTH directions — fewer means
tests have silently vanished; more means the ratchet was not bumped.

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

## Feature set (95 features)

**95** is `EXPECTED_TABULAR_FEATURE_COUNT` in
`src/genomic_variant_classifier/models/variant_ensemble.py`, and it is the single source of
truth. This table, `METHODS.md`, the committed schema baseline, the G1 preflight gate and the
inference contract all **re-derive** it — `tests/unit/test_readme_claims.py` fails the suite if
this document and the code ever disagree again. (Before 2026-07-14 this README stated the
feature count in **nine** places with **four** different values: 80, 78, 79 and 80.)

| Group | Count | Key features |
|-------|-------|-------------|
| Allele frequency | 6 | af_raw, af_log10, af_is_absent, af_is_ultra_rare |
| Variant type | 7 | ref_len, alt_len, len_diff, is_snv, is_insertion, is_deletion |
| Consequence | 6 | consequence_severity, is_loss_of_function, is_missense, is_splice |
| Functional scores | 9 | CADD, SIFT, PolyPhen-2, REVEL, PhyloP, GERP, AlphaMissense, SpliceAI, EVE |
| Score flags + meta | 5 | cadd_high, sift_deleterious, polyphen_probably_damaging, n_tools_pathogenic |
| Gene-level | 4 | gene_constraint_oe, n_pathogenic_in_gene, gene_has_known_disease |
| Protein (UniProt) | 2 | has_uniprot_annotation, n_known_pathogenic_protein_variants |
| Expression (GTEx) | 6 | gtex_max_tpm, gtex_tissue_specificity, gtex_is_eqtl, gtex_max_abs_effect |
| Variant coding context | 2 | codon_position, dbsnp_af |
| Gene-disease (OMIM/ClinGen) | 4 | omim_n_diseases, omim_n_diseases_molecular, omim_is_autosomal_dominant, clingen_validity_score |
| LOVD | 1 | lovd_variant_class (ordinal 0-4) |
| Chromosome | 3 | is_autosome, is_sex_chrom, is_mitochondrial |
| Gene network | 2 | gnn_score (GAT over STRING PPI), hetero_gnn_score |
| RNA splice | 5 | maxentscan_score, maxentscan_delta, dist_to_splice_site, exon_number, is_canonical_splice |
| Protein structure | 4 | alphafold_plddt, solvent_accessibility, secondary_structure_context, dist_to_active_site |
| 1000 Genomes AF | 5 | af_1kg_afr, af_1kg_eur, af_1kg_eas, af_1kg_sas, af_1kg_amr |
| FinnGen R12 | 3 | finngen_af_fin, finngen_af_nfsee, finngen_enrichment |
| FinnGen R13 | 3 | finngen_r13_af_fin, finngen_r13_af_nfsee, finngen_r13_enrichment |
| ESM-2 (650M) | 2 | esm2_delta_norm (secondary), esm2_llr (primary, signed LLR) |
| Nucleotide Transformer | 2 | genomiclm_delta_norm, genomiclm_llr |
| COSMIC | 2 | cosmic_recurrence, cosmic_sig_tier |
| KEGG | 2 | kegg_pathway_count, kegg_disease_pathway_flag |
| Reactome | 1 | reactome_pathway_count |
| gnomAD v4.1 constraint | 4 | pli_score, loeuf, syn_z, mis_z |
| RNA-seq expression | 5 | rnaseq_mean_log_tpm, rnaseq_detection_rate, rnaseq_log2_cv, rnaseq_log2fc, rnaseq_de_neglog10p |
| **Total** | **95** | = `EXPECTED_TABULAR_FEATURE_COUNT` |

**HGMD is NOT in the feature set.** `hgmd_is_disease_mutation` and `hgmd_n_reports` were listed
here until 2026-07-13 and were **constant zero for the entire life of the project** — the HGMD
Professional licence was never obtained and the connector was never wired. They were removed
from the contract (97 → 95).

They will not return in that form even if the licence is obtained. HGMD's "DM" (disease
mutation) classification is, at the variant level, a near-copy of the ClinVar-Pathogenic label
this model is trained to predict; using it as a feature would leak the target, and — because a
novel variant of uncertain significance has no HGMD entry — would bias the model toward benign
on exactly the variants it exists to score. If the licence is obtained, HGMD will enter as a
**gene-level, leave-one-out aggregate** (HGMD-DM count in the gene, *excluding* the variant
being scored), mirroring `n_pathogenic_in_gene`.

`uncertainty_epistemic` and `uncertainty_aleatoric` were also listed here as
"Reserved (Deep Ensemble)" and counted toward the total. They live in `PHASE_4_FEATURES`, not
in `TABULAR_FEATURES`, and are not part of the trained contract.

## Drift detection and continual learning

### Statistical detectors

- **PSI (Population Stability Index)** -- per-feature. Computed from a committed
  aggregate reference profile (histogram counts + quantile grids, no variant rows) whose PSI is
  **bit-identical** to the raw reference matrix -- measured worst delta 0.000e+00 across all 78
  features. Until 2026-07-13 this line said "runs on every data source update"; it had never run
  at all (roadmap 6.20).
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
- **Connector silent-zero hardening** -- **this claim was FALSE until 2026-07-13 and is the
  most consequential defect this project has found.** Connectors do NOT fail loud when their
  source file is absent: they return zeros (`omim.py:105` — `if gene_table.empty: result[...] = 0;
  return result`, with no log, no warning and no raise). **Run 15 trained, evaluated and
  published with 36 of its 78 features CONSTANT ZERO.** The launcher's abort gates check that a
  FILE EXISTS, which is a proxy — a present-but-empty file, a schema change or a failed join all
  pass a file check and still deliver a column of zeros.
  **NOW ENFORCED** (roadmap 6.21): `feature_census()` runs on the FULL engineered matrix before
  the training subsample and before a single model is fitted; it names every dead feature, its
  data source and the CLI flag responsible, then **exits 2**. `VariantEnsemble.fit()` carries a
  zero-variance guard as backstop. A source that fails to populate now costs seconds in the
  smoke run, not eleven hours of paid compute followed by a published algorithm comparison.

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
`agent_layer/test_message_bus.py` exercises the bus (34/34 passing on Python 3.12.10).

## REST API

```
GET  /health          Liveness + readiness
GET  /info            Model metadata, 95 features, drift status
GET  /metrics         Prometheus metrics
GET  /gene/{symbol}   Gene-level feature lookup
GET  /rsid/{rs_id}    rs-ID resolution + prediction
POST /predict         Single variant -> 5-tier classification + uncertainty
POST /batch           Up to 1,000 variants
```

Auth: X-API-Key header; rate limiting via `slowapi`; structured JSON logging;
Prometheus `/metrics` instrumentation via `prometheus-fastapi-instrumentator`.

## Performance

### ⚠️ WITHDRAWN — no performance figures are published for this project at present (2026-07-14)

This section previously carried a holdout AUROC, a Brier score, sensitivity/specificity
operating points, a seven-row per-model comparison table, and a training-run history.
**All of it has been removed, and none of it should be quoted from git history either.**

**The reason:** on 2026-07-13 a feature census of the Run-15 training matrix found that **36 of
its 78 features were CONSTANT ZERO** — 46% of the feature space, across 1,038,974 variants.
Whole data sources were silently stubbed to 0.0 because their connectors return zeros rather
than raising when a source file is absent: GTEx (6 features), 1000 Genomes (5), FinnGen (3),
AlphaFold/protein structure (4), splice/MaxEntScan (4), UniProt (2), OMIM (2), HGMD (2), plus
ESM-2, EVE, dbSNP, PhyloP, ClinGen, `codon_position` and gene constraint.

Every number that stood here was therefore produced by **38 real features**, not the 78 or 80
the document claimed — and the per-model comparison table ranked twelve algorithms against one
another on a feature space that was half imaginary. **A cross-algorithm comparison is exactly
the artefact that a half-empty feature space invalidates**, because different model families
degrade differently under missing signal.

Compounding it: the features that *were* live are dominated by `cadd_phred`, `revel_score`,
`sift_score`, `polyphen2_score` and `n_tools_pathogenic` — in-silico predictors **themselves
trained on ClinVar**. With HGMD/LOVD/ClinGen refuted as the leakage explanation
(`docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md`), that circularity is now the leading
candidate, by elimination, for why so high a score was attainable at all on this task.

The withdrawn figures are **not restated in this file, even to disown them** — a number in a
warning banner is still a number that gets quoted out of context. They live in
`docs/ROADMAP.md` §6.21 and `docs/audits/README_AUDIT_2026-07-14.md`, where the caveat travels
with the value. `tests/unit/test_readme_claims.py` fails the suite if any performance-shaped
figure reappears in this document.

**What has changed so this cannot recur:** `feature_census()` now runs on the FULL engineered
matrix before the training subsample and before a single model is fitted. It names every dead
feature, the data source it came from, and the CLI flag responsible — then **exits 2**
(roadmap 6.21). `VariantEnsemble.fit()` carries a zero-variance guard as a second line. Run 17
physically cannot be produced with a silently-empty feature.

**A clean table will be published here when Run 17 completes**, with the feature census printed
alongside it as evidence that every declared feature actually carried information.

Per-run details live in `docs/sessions/SESSION_<date>.md` and root-cause records
in `docs/incidents/INCIDENT_<date>_<topic>.md`. `docs/ROADMAP.md` §6.21 is the full account.

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
- **Test depth** -- 1,926 passing / 1,933 collected, including regression tests for
  every silent-zero failure mode found to date and an inter-agent message-bus suite,
  on Python 3.11 and 3.12.
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
  run_drift_monitor.py      - monthly drift check CLI (exit 0/1/2/3/4; 4 = NOT CHECKED)
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

# Run monthly drift check -- from the COMMITTED aggregate reference profile.
# No credentials, no cloud fetch, no 23.8 MB cohort matrix. The Population Stability Index
# computed from the profile is BIT-IDENTICAL to the raw matrix (measured worst delta 0.000e+00).
# Exit 4 = NOT CHECKED -- distinct from 0 (no drift) and from 3 (urgent retrain).
python scripts/run_drift_monitor.py \
  --reference-profile data/reference/drift/run15_reference_profile.json \
  --new-clinvar  data/processed/clinvar_grch38_latest.parquet \
  --old-clinvar  data/processed/clinvar_grch38_previous.parquet \
  --output-dir   outputs/drift_reports/latest/

# Full-fidelity drift check (adds the joint Maximum Mean Discrepancy and Szekely-Rizzo energy
# tests, which need real reference samples). Run where the cohort matrix lives.
# --auto-retrain REFUSES to run from the isolated drift environment: it would unpickle a
# LightGBM 4.6.0 booster into the 4.5.0 runtime that nannyml pins.
python scripts/run_drift_monitor.py \
  --reference-splits outputs/run17_report/full/splits/ \
  --new-clinvar  data/processed/clinvar_grch38_latest.parquet \
  --old-clinvar  data/processed/clinvar_grch38_previous.parquet \
  --output-dir   outputs/drift_reports/latest/

# Train (full ensemble, 95 features)
#
# NOTE: the flag is --clinvar, NOT --parquet. This README said --parquet until 2026-07-14;
# that flag has never existed and the command failed with an argparse error for anyone who
# copied it. The authoritative full command -- every source flag, every abort gate -- is
# scripts/launch_run17_baseline.sh; do not hand-assemble it.
python scripts/run_phase2_eval.py \
  --clinvar   data/processed/clinvar_grch38_clean.parquet \
  --output    outputs/run17/full \
  --lovd-path data/external/lovd/lovd_all_variants.parquet \
  --dbnsfp-path data/external/dbnsfp/dbnsfp_clinvar_index.parquet
  # ... plus --gnomad, --spliceai, --alphamissense, --gtex-path, --omim-genemap2-path,
  #     --phylop-path, --alphafold-path, --clingen-path, --eve-path, --finngen-path, ...
  # A missing source no longer trains silently on zeros: the feature census aborts (exit 2)
  # and names the flag responsible. See roadmap 6.21.

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
