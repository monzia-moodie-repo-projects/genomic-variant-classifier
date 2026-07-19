# Genomic Variant Pathogenicity Classifier

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tabular features](https://img.shields.io/badge/tabular%20features-95-blue.svg)]()
[![Base models](https://img.shields.io/badge/base%20models-13-blue.svg)]()
[![Agents](https://img.shields.io/badge/autonomous%20agents-22-blueviolet.svg)]()
[![Tests](https://img.shields.io/badge/tests-1967-success.svg)]()
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()

A multi-modal machine learning system for the five-tier clinical classification of human
genomic variants — **Pathogenic, Likely Pathogenic, Uncertain Significance, Likely Benign,
and Benign** — in accordance with ACMG/AMP guidelines.

The system integrates genomic sequence, population-stratified allele frequencies, protein
structure and language-model representations, gene-network topology, tissue-specific
expression, and curated gene–disease evidence into a **95-feature** matrix, consumed by a
**13-model** stacking ensemble. It is served as a FastAPI REST service and supervised by an
autonomous layer of **22 specialised agents** communicating over a typed inter-agent
message bus.

Training draws on a cohort of over four million ClinVar variants across more than 28,000
genes, annotated from some twenty biological databases through a multi-stage pipeline.

> Figures throughout this document describe a snapshot and are refreshed after each full
> training run. `docs/ROADMAP.md` is the authoritative, continuously maintained record.

---

## Purpose

This is a **whole-genome** variant classifier, and it is not a benchmarking exercise. It is
a multi-goal research system, and the goals reinforce one another.

**Clinical.** To make diagnosis more accurate, more detailed, more personalised, and
clearer — returning not a bare score but a calibrated, interpretable, honestly-bounded
assessment that a clinician can act on and interrogate.

**Biological.** To bring greater scientific understanding and insight into the contribution
of genes to the development and progression of human disease — which genes, which variants,
which combinations, and by what mechanism. Prediction is the instrument; understanding is
the object.

**Methodological.** To measure the performance of, and better understand, the very tools
used to analyse genomic data. Which machine-learning models genuinely work on this problem,
where they fail, why, and how they can be improved. Here the models are objects of study in
their own right, not merely instruments pointed at one.

**Translational.** To create new methodology that makes personalised medicine more
scientifically rigorous, more precise, and more reliable.

### How this shapes the system

**Attribution over prediction.** The per-variant probability is the visible output; the
underlying aim is to establish which genes and which evidence actually carry the signal,
and how confidently that can be claimed.

**Fair algorithm comparison.** Every base model is trained on identical folds, identical
features, and identical splits, so the differences between them are attributable to the
algorithms rather than to their inputs. Preserving that invariant is a first-class
engineering concern, not a side effect.

**Modality integration, not concatenation.** DNA, RNA, protein, gene networks — and, as a
future phase, clinical imaging — each describe the same biology from a different angle. The
long-term aim is a representation that learns the relationships between those views, rather
than a wider feature vector.

**Uncertainty as a clinical product.** A variant of uncertain significance is the case that
matters most. The system is built so that "I do not know" is a calibrated, auditable,
first-class output rather than a probability loitering near 0.5.

**Generalisation to unseen genes.** Splits are gene-disjoint by construction. A model that
scores well only on catalogued variants in catalogued genes has not solved the problem this
project exists to solve.

---

## Architecture

The classifier is a multi-branch fusion model wrapped in an autonomous supervisory layer.

### Tabular branch

A stacking meta-learner trained on out-of-fold predictions from **thirteen base
classifiers** spanning six algorithm families. The roster is deliberately diverse: the
point is to compare families, not to find one winner and discard the evidence.

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

Base-model predictions are combined by the logistic-regression meta-learner and calibrated
on a gene-disjoint partition. The ensemble's composition is written into the run artifacts,
so *which models actually trained* is a recorded fact rather than an assumption — a base
model whose out-of-fold step fails raises, rather than quietly leaving the roster.

The Graph Attention Network is deliberately **not** in this table. It contributes
`gnn_score`, a feature, and produces no out-of-fold column; it is not a base classifier.

### Sequence branch

A 1D convolutional network over genomic context windows centred on the variant, encoding
the reference and alternate alleles together with their difference — so the model sees the
*change*, not merely the surrounding sequence. Protein-language-model representations
(ESM-2) and DNA-language-model representations (Nucleotide Transformer) contribute
variant-effect signals derived from masked-language-model likelihoods and embedding
geometry.

### Graph branch

A Graph Attention Network over the STRING protein–protein interaction network supplies
gene-level network context, with a heterogeneous knowledge-graph variant incorporating
pathway membership. These yield gene-level priors that enter the tabular matrix as
features rather than as independent classifiers.

### Histopathology branch (planned)

A whole-slide imaging branch over TCGA cohorts is a tracked future phase, linking
variant-level prediction to observable tissue morphology. It is not yet implemented; see
`docs/ROADMAP.md`.

```
   Population genetics . Conservation . Functional predictors . Gene-disease
   knowledge bases . Protein structure . Expression . Splice mechanics .
   Protein-protein interaction topology
                              |
                    Annotation pipeline
                              |
                    Feature engineering
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
                  + probability calibration
                  + conformal prediction sets
                  + epistemic / aleatoric uncertainty
                              |
                    Clinical evaluation
                              |
     +------------------------+------------------------+
     |                                                 |
  FastAPI REST service                   Autonomous agent layer
  auth . rate limiting                   typed message bus
  Prometheus metrics                     shared state + orchestrator
  Docker                                 continual learning + EWC
                                         versioned model registry
                                         shadow -> production promotion
```

---

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

The feature count lives in exactly one place — `EXPECTED_TABULAR_FEATURE_COUNT` — and is
enforced against the feature list at import time. A source that fails to populate causes a
loud failure rather than a silent column of zeros.

---

## Uncertainty and conformal prediction

Point probabilities are insufficient for clinical use, so the system carries an explicit
uncertainty layer.

**Calibration.** Post-hoc probability calibration is fitted on a partition of genes the
base models never trained on, so the calibrated probabilities are honest about
generalisation to new genes rather than to new variants in familiar genes.

**Conformal prediction sets.** Rather than forcing every variant into a class, the
conformal layer emits a *set* — `{pathogenic}`, `{benign}`, `{pathogenic, benign}` (defer),
or empty (out of domain) — with finite-sample coverage guarantees under exchangeability.
Label-conditional (Mondrian) calibration is used so that coverage holds for the rare
pathogenic class rather than being satisfied on average by the majority class.

**Epistemic and aleatoric decomposition.** Monte-Carlo Dropout and Deep Ensemble wrappers
separate uncertainty the model could reduce with more data from uncertainty inherent to the
variant, flagging cases that warrant human expert review.

---

## Drift detection and continual learning

Biological reference data is not static. ClinVar reclassifies variants, gnomAD cohorts
grow, and functional-score models are retrained upstream. A classifier that ignores this
is accurate on the day it ships and quietly wrong thereafter.

**Statistical detectors.** Population Stability Index, Kolmogorov–Smirnov, Maximum Mean
Discrepancy, the Székely–Rizzo energy statistic, and adaptive windowing for streaming
ingestion. A reference profile is committed to the repository so drift can be measured
against a fixed baseline without moving cohort data.

**"Not checked" is its own answer.** The monitor reports `0` no drift, `1` monitor,
`2` retrain, `3` urgent retrain — and `4` **NOT CHECKED**. That fifth code exists because
*"I looked and found nothing"* and *"I could not look"* are different statements, and
reporting the second as the first is how a monitoring system lies. Where a test cannot be
computed from the committed profile, it reports itself as not computed rather than as
passing.

**Three classes of drift are tracked separately** — covariate drift as upstream data
expands, label drift as ClinVar reclassifies, and concept drift as new biology changes
what the features mean. They have different remedies and are not conflated.

**Adaptive retraining.** When drift exceeds configured thresholds, retraining is triggered
using Elastic Weight Consolidation to preserve stable biological signal while
incorporating new evidence, with importance weighting and temporal decay across releases.

**Lifecycle.** A versioned model registry moves candidates through staging, shadow
deployment, and production promotion. New models run in parallel with the incumbent before
they replace it.

---

## Autonomous agent layer (22 agents)

Under `src/genomic_variant_classifier/agent_layer/`, twenty-two specialised agents inherit
from a common `BaseAgent` and communicate over a typed message bus, with a JSON-persisted
shared blackboard and an orchestrator that schedules execution and routes messages.

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

Messages whose subjects carry consequence require explicit human approval before the
receiving agent acts on them. The agent layer monitors itself: `AgentOpsMonitorAgent`
exists so that a silently dead agent is a detectable condition rather than an absence
nobody notices.

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

Authentication via `X-API-Key`; rate limiting via `slowapi`; structured JSON logging;
Prometheus instrumentation. Served from a multi-stage Dockerfile with builder, api, and
trainer targets.

---

## Early results

**These are preliminary and will be refined as the project develops.**

The Run 15 baseline (sealed 2026-06-09, commit `032a2ab`) reported a test AUROC of
**0.9984** on gene-stratified, expert-reviewed ClinVar variants, with a comparable
unseen-gene-holdout figure. Earlier baselines reported lower values on narrower feature
sets.

These numbers describe an earlier and narrower configuration of the system than the one
now in the repository. The feature space, the model roster, the split protocol, and the
data-integrity gates have all changed since. Treat them as an early waypoint rather than a
result: a like-for-like table — per-model, per-metric, with the evaluation protocol stated
alongside — will be published from the next full training run.

---

## Repository structure

```
src/genomic_variant_classifier/
  agent_layer/   - specialised agents, typed message bus, orchestrator, shared state
  api/           - FastAPI service, auth, schemas, inference pipeline
  data/          - database connectors, ETL, split protocol, data preparation
  evaluation/    - clinical evaluator, benchmark framework, metrics, artifacts
  models/        - variant ensemble, graph networks, Kolmogorov-Arnold Network,
                   uncertainty wrappers, sequence CNN
  monitoring/    - drift detection, reference profiles, performance estimation,
                   ClinVar tracking, model registry
  pipelines/     - RNA splice pipeline, protein structure pipeline
  reports/       - HTML report generation
  training/      - continual learner, Elastic Weight Consolidation
  utils/         - shared helpers
scripts/         - training entry points, preflight gates, drift monitoring,
                   data preparation, launch runbooks, forensics
tests/           - unit, integration, and conformal test suites
docs/
  ROADMAP.md     - the living record: every change, dated, with its evidence
  CHANGELOG.md   - append-only session ledger, searchable by error string
  incidents/     - root-cause records
  sessions/      - chronological working logs
  status/        - dated status and remediation reports
configs/         - configuration
deploy/          - Grafana dashboards, Prometheus configuration
```

`docs/ROADMAP.md` is the authoritative history. It preserves what was found, what was
wrong, and what was done about it, in order.

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
  --output-dir        outputs/drift_reports/
# exit 0/1/2/3/4 -- 0 no drift, 1 monitor, 2 retrain, 3 urgent_retrain,
#                   4 NOT CHECKED (no data reached the monitor)

# Docker
docker compose up api
```

Run `--help` on any script for its full argument set.

---

## Roadmap

- **Data expansion.** Continue wiring biological sources into the feature matrix, with
  each source gated so that a source which fails to populate fails loudly rather than
  contributing a column of zeros.
- **Algorithm expansion and benchmarking.** Extend the base roster and run every member
  through a common benchmark harness on identical folds, so cross-algorithm comparisons
  are attributable to the algorithms.
- **Joint-Embedding Predictive Architecture.** A self-supervised representation layer over
  multi-modal foundation-model embeddings, benchmarked against the current stacker rather
  than replacing it.
- **Conformal uncertainty as a scientific instrument.** Extend the conformal layer to
  ordinal five-class prediction sets, multi-label disease categories, and calibrated
  gene-candidate sets; analyse where uncertainty concentrates biologically.
- **Multi-modal expansion.** RNA and whole-slide histopathology branches.
- **Clinical validation and manuscript.** Prospective validation on curated gene panels,
  comparison against expert review status, model card, and manuscript.

See `docs/ROADMAP.md` for the live checklist and the full history.

---

## Author

**Monzia Moodie** — [@monzia-moodie](https://github.com/monzia-moodie)

## License

MIT License
