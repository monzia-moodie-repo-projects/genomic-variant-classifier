# Genomic Variant Pathogenicity Classifier

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tabular features](https://img.shields.io/badge/tabular%20features-95-blue.svg)]()
[![Base models](https://img.shields.io/badge/base%20models-13-blue.svg)]()
[![Agents](https://img.shields.io/badge/autonomous%20agents-22-blueviolet.svg)]()
[![Tests](https://img.shields.io/badge/tests-6136-success.svg)]()
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()

A multi-modal machine learning system for the five-tier clinical classification of human
genomic variants — **Pathogenic, Likely Pathogenic, Uncertain Significance, Likely Benign,
and Benign** — in accordance with ACMG/AMP guidelines.

It integrates genomic sequence, population-stratified allele frequencies, protein structure
and language-model representations, gene-network topology, tissue-specific expression, and
curated gene–disease evidence into a **95-feature** matrix, consumed by a **13-model**
stacking ensemble. It is served as a FastAPI REST service and supervised by an
autonomous layer of **22 specialised agents** communicating over a typed message bus.

Training draws on a cohort of over four million ClinVar variants across more than 28,000
genes, annotated from some twenty biological databases.


---

## Purpose

A **whole-genome** variant classifier, and not a benchmarking exercise. Four goals that
reinforce one another:

- **Clinical** — return not a bare score but a calibrated, interpretable, honestly-bounded
  assessment a clinician can act on and interrogate.
- **Biological** — establish which genes, which variants, which combinations, and by what
  mechanism contribute to disease. Prediction is the instrument; understanding is the object.
- **Methodological** — measure the tools themselves. Which algorithms genuinely work on this
  problem, where they fail, and why. The models are objects of study, not just instruments.
- **Translational** — new methodology that makes personalised medicine more rigorous.

Three constraints follow and are treated as first-class engineering concerns: every base
model sees **identical folds, features and splits**, so differences are attributable to the
algorithms; splits are **gene-disjoint**, so scoring well only on catalogued variants in
catalogued genes does not count as solving the problem; and *"I do not know"* is a
**calibrated, auditable output** rather than a probability loitering near 0.5.

---

## Architecture

A multi-branch fusion model wrapped in an autonomous supervisory layer.

```
   Population genetics . Conservation . Functional predictors . Gene-disease
   knowledge bases . Protein structure . Expression . Splice mechanics .
   Protein-protein interaction topology
                              |
                    Annotation pipeline -> Feature engineering
                              |
     +------------+-----------+-----------+------------+
     |            |                       |            |
  Tabular      Sequence                 Graph      Histopathology
  ensemble     1D-CNN over              GAT over    [PLANNED]
  (multi-      ref/alt windows          STRING PPI
  family)      + ESM-2 / DNA-LM         + hetero-KG
     |            |                       |            |
     +------------+-----------+-----------+------------+
                              |
                  Stacking meta-learner
                  + calibration + conformal sets + uncertainty
                              |
                    Clinical evaluation
                              |
     +------------------------+------------------------+
     |                                                 |
  FastAPI REST service                   Autonomous agent layer
  auth . rate limiting                   typed message bus
  Prometheus metrics . Docker            continual learning . model registry
```

| Branch | What it contributes | Status |
|---|---|---|
| Tabular | 13 base classifiers over the feature matrix, stacked and calibrated | live |
| Sequence | 1D convolution over ref/alt context windows; ESM-2 and Nucleotide Transformer variant-effect signals | live |
| Graph | Graph Attention Network over STRING, plus a heterogeneous knowledge graph; both enter the matrix as *features*, not as classifiers | live |
| Histopathology | Whole-slide imaging over TCGA cohorts, linking prediction to tissue morphology | planned |

---

## Base models

Thirteen classifiers spanning six algorithm families. The roster is deliberately diverse:
the point is to compare families, not to find one winner and discard the evidence.

| # | Key | Model | Family |
|---:|---|---|---|
| 1 | `random_forest` | Random Forest | Bagged trees |
| 2 | `xgboost` | XGBoost | Gradient-boosted trees |
| 3 | `lightgbm` | LightGBM | Gradient-boosted trees |
| 4 | `catboost` | CatBoost | Gradient-boosted trees |
| 5 | `gradient_boosting` | scikit-learn Gradient Boosting | Gradient-boosted trees |
| 6 | `logistic_regression` | Logistic Regression — also the stacking meta-learner | Linear |
| 7 | `svm` | Support Vector Machine | Kernel |
| 8 | `svm_bagged_rbf` | Bagged radial-basis-function Support Vector Machine | Kernel |
| 9 | `kan` | Kolmogorov-Arnold Network | Kolmogorov-Arnold |
| 10 | `tabular_nn` | Tabular neural network | Neural |
| 11 | `cnn_1d` | 1D sequence convolutional network | Neural |
| 12 | `mc_dropout` | Monte-Carlo Dropout | Bayesian uncertainty |
| 13 | `deep_ensemble` | Deep Ensemble | Bayesian uncertainty |

The ensemble's composition is written into the run artifacts, so *which models actually
trained* is a recorded fact: a base model whose out-of-fold step fails raises rather than
quietly leaving the roster. The Graph Attention Network is deliberately absent from this
table — it contributes the `gnn_score` feature and produces no out-of-fold column.

---

## Feature set (95 tabular features)

| Group | Count | Representative features |
|---|---:|---|
| Allele frequency | 6 | `af_raw`, `af_log10`, `af_is_absent`, `af_is_ultra_rare` |
| Variant type | 7 | `ref_len`, `alt_len`, `is_snv`, `is_insertion`, `is_deletion` |
| Consequence | 6 | `consequence_severity`, `is_loss_of_function`, `is_missense`, `is_splice` |
| Functional predictors | 9 | CADD, SIFT, PolyPhen-2, REVEL, PhyloP, GERP, AlphaMissense, SpliceAI, EVE |
| Predictor flags + meta-score | 5 | `cadd_high`, `sift_deleterious`, `n_tools_pathogenic` |
| Gene-level | 4 | `gene_constraint_oe`, `n_pathogenic_in_gene`, `gene_has_known_disease` |
| gnomAD constraint | 4 | `pli_score`, `loeuf`, `syn_z`, `mis_z` |
| Protein annotation (UniProt) | 2 | `has_uniprot_annotation`, `n_known_pathogenic_protein_variants` |
| Protein structure (AlphaFold) | 4 | `alphafold_plddt`, `solvent_accessibility`, `dist_to_active_site` |
| Expression (GTEx) | 6 | `gtex_max_tpm`, `gtex_tissue_specificity`, `gtex_is_eqtl` |
| RNA-seq expression | 5 | `rnaseq_mean_log_tpm`, `rnaseq_log2fc`, `rnaseq_de_neglog10p` |
| RNA splice context | 5 | `maxentscan_score`, `maxentscan_delta`, `dist_to_splice_site` |
| Gene–disease annotation | 4 | `omim_n_diseases`, `omim_is_autosomal_dominant`, `clingen_validity_score` |
| LOVD | 1 | `lovd_variant_class` |
| 1000 Genomes population AF | 5 | `af_1kg_afr`, `af_1kg_eur`, `af_1kg_eas`, `af_1kg_sas`, `af_1kg_amr` |
| FinnGen R12 | 3 | `finngen_af_fin`, `finngen_af_nfsee`, `finngen_enrichment` |
| FinnGen R13 | 3 | `finngen_r13_af_fin`, `finngen_r13_af_nfsee`, `finngen_r13_enrichment` |
| ESM-2 protein language model | 2 | `esm2_delta_norm`, `esm2_llr` (signed log-likelihood ratio) |
| Nucleotide Transformer DNA-LM | 2 | `genomiclm_delta_norm`, `genomiclm_llr` |
| COSMIC | 2 | `cosmic_recurrence`, `cosmic_sig_tier` |
| KEGG | 2 | `kegg_pathway_count`, `kegg_disease_pathway_flag` |
| Reactome | 1 | `reactome_pathway_count` |
| Graph-derived | 2 | `gnn_score` (STRING), `hetero_gnn_score` (knowledge graph) |
| Chromosome context | 3 | `is_autosome`, `is_sex_chrom`, `is_mitochondrial` |
| Coding context | 2 | `codon_position`, `dbsnp_af` |
| **Total** | **95** | |

The count lives in exactly one place — `EXPECTED_TABULAR_FEATURE_COUNT` — and is enforced
against the feature list at import time. A source that fails to populate fails loudly
rather than contributing a column of zeros.

## Autonomous agent layer (22 agents)

Under `src/genomic_variant_classifier/agent_layer/`, twenty-two agents inherit from a common
`BaseAgent` and communicate over a typed message bus, with a JSON-persisted shared
blackboard and an orchestrator that schedules execution and routes messages.

| Agent | Concern |
|---|---|
| `DataFreshnessAgent` | Polls upstream source manifests; raises when data goes stale |
| `DatabaseFreshnessMonitorAgent` | Tracks database release cadence and staleness budgets |
| `DataReadinessAgent` | Gates whether the inputs a run needs are actually present and sane |
| `VersionMonitorAgent` | Watches upstream dataset versions and breaking-change deltas |
| `SchemaDriftMonitorAgent` | Detects column and dtype changes in incoming connector data |
| `ConceptDriftMonitorAgent` | Monitors feature→label relationship stability |
| `LabelShiftMonitorAgent` | Tracks class priors across ClinVar releases |
| `CalibrationDriftMonitorAgent` | Watches calibration error and reliability over time |
| `InfrastructureDriftMonitorAgent` | Catches dependency and runtime drift |
| `FeatureCoverageSentinelMonitorAgent` | Guards against features quietly losing coverage |
| `ReclassificationSentinelMonitorAgent` | Monitors ClinVar reclassification flip rate |
| `FairnessSubgroupMonitorAgent` | Per-ancestry, per-consequence, per-gene-tier auditing |
| `AdversarialSubmissionMonitorAgent` | Flags out-of-distribution or suspect requests |
| `AnnotationPolicyMonitorAgent` | Enforces source-priority and provenance rules |
| `InterpretabilityAgent` | SHAP-based attribution audits, persisted per release |
| `ModelInsightsAgent` | Surfaces model-behaviour findings for review |
| `LiteratureScoutAgent` | Monitors preprints and publications for new methods |
| `TrainingLifecycleAgent` | Orchestrates retrain → consolidate → shadow → promote |
| `AdaptationAgent` | Evaluates and records candidate adaptations |
| `AgentOpsMonitorAgent` | Monitors the agent layer itself — heartbeats, backlogs, errors |
| `FinOpsAdvisorAgent` | Cost advisory for paid compute |
| `ProvisioningAgent` | Provisioning for training infrastructure |

Messages carrying consequence require explicit human approval before the receiving agent
acts. `AgentOpsMonitorAgent` exists so that a silently dead agent is a detectable condition
rather than an absence nobody notices.

---

## Evaluation, uncertainty, and drift

A number in a clinical report is a claim, so every reported quantity carries what it
measured, over which rows, and whether it can be trusted.

- **One computation path.** Metrics are computed once by a typed registry; the flat report
  fields are derived views, not independent calculations.
- **A refusal is a result.** An uncomputable metric returns a typed status and reason, never
  zero and never an exception. Silence and zero are both lies.
- **Populations are named or admitted to be unnamed.** Unattributed populations carry no
  membership fingerprint, and comparison returns `UNKNOWN` rather than a false equality.
- **Thresholds and selection policies are declared.** A chosen operating point records the
  objective, the tie-break, and how many candidates were feasible.
- **Conformal prediction sets.** `{pathogenic}`, `{benign}`, both (defer), or empty (out of
  domain), with label-conditional calibration so coverage holds for the rare class.
- **Epistemic and aleatoric uncertainty** are separated, flagging cases for expert review.
- **Drift is monitored in three classes** — covariate, label, and concept — with retraining
  under Elastic Weight Consolidation and a shadow-to-production model registry. The monitor
  reports `0/1/2/3/4`, where **4 is NOT CHECKED**: *"I looked and found nothing"* and *"I
  could not look"* are different statements.

Full rationale, incidents, and the defect register: `docs/ROADMAP.md`.

---

## REST API

```
GET  /health          Liveness + readiness
GET  /info            Model metadata and drift status
GET  /metrics         Prometheus metrics
GET  /gene/{symbol}   Gene-level feature lookup
GET  /rsid/{rs_id}    rs-ID resolution + prediction
POST /predict         Single variant -> five-tier classification + uncertainty
POST /batch           Batch classification
```

Authentication via `X-API-Key`, rate limiting via `slowapi`, structured JSON logging, and
Prometheus instrumentation, from a multi-stage Dockerfile with builder, api and trainer
targets.

---

## Early results

**Preliminary, and an early waypoint rather than a result.** The Run 15 baseline (sealed
2026-06-09, commit `032a2ab`) reported a test AUROC of **0.9984** on gene-stratified,
expert-reviewed ClinVar variants, with a comparable unseen-gene-holdout figure.

That configuration was narrower than the one now in the repository: the feature space, the
model roster, the split protocol and the data-integrity gates have all changed since. A
like-for-like per-model table, with the evaluation protocol stated alongside, will be
published from the next full training run.

---

## Quickstart

```bash
# Run the API
MODEL_PATH=models/phase2_pipeline.joblib \
  uvicorn genomic_variant_classifier.api.main:app --port 8000

# Classify a variant
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"chrom":"17","pos":43115726,"ref":"AAC","alt":"A",
       "consequence":"frameshift_variant","allele_freq":0.0,
       "alphamissense_score":0.95,"n_pathogenic_in_gene":2800}'

# Train
python scripts/run_phase2_eval.py \
  --clinvar data/processed/clinvar_grch38_clean.parquet \
  --output  outputs/run/full

# Drift check
python scripts/run_drift_monitor.py \
  --reference-profile data/reference/drift/reference_profile.json \
  --new-data          data/processed/<new_release>.parquet \
  --output-dir        outputs/drift_reports/
# exit 0/1/2/3/4 -- 0 no drift, 1 monitor, 2 retrain, 3 urgent_retrain,
#                   4 NOT CHECKED (no data reached the monitor)
#
# --new-data (or --new-clinvar) is REQUIRED to get a verdict. Without one the
# monitor runs, compares nothing, and returns 4 -- which is honest, and useless.

# Docker
docker compose up api
```

Run `--help` on any script for its full argument set.

---

## Repository structure

```
src/genomic_variant_classifier/
  agent_layer/   - agents, typed message bus, orchestrator, shared state
  api/           - FastAPI service, auth, schemas, inference
  data/          - connectors, ETL, split protocol, data preparation
  evaluation/    - clinical evaluator, metric registry, thresholds, artifacts
  models/        - variant ensemble, graph networks, uncertainty wrappers, sequence CNN
  monitoring/    - drift detection, reference profiles, model registry
  pipelines/     - RNA splice, protein structure
  reports/       - HTML report generation
  training/      - continual learner, Elastic Weight Consolidation
scripts/         - training entry points, preflight gates, drift monitoring, runbooks
tests/           - unit, integration and conformal suites
docs/            - ROADMAP.md, CHANGELOG.md, incidents/, sessions/, measurements/
```

---

## Status and next steps

Current phase: the evaluation subsystem. Metrics, population identity, threshold semantics
and operating-point selection are typed, tested and recorded.

Next, in order: **data expansion** with every source gated to fail loudly; **algorithm
expansion and benchmarking** across a common harness on identical folds; a **self-supervised
joint-embedding representation** benchmarked against the current stacker rather than
replacing it; **conformal uncertainty** extended to ordinal five-class and multi-label
prediction sets; **multi-modal expansion** into RNA and whole-slide histopathology; and
**clinical validation** with a model card and manuscript.

`docs/ROADMAP.md` carries the live checklist, the open follow-up register, and the full
dated history.

---

## Author

**Monzia Moodie** — [@monzia-moodie](https://github.com/monzia-moodie)

## License

MIT License
