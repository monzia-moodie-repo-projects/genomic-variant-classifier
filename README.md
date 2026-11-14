# Genomic Variant Pathogenicity Classifier

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tabular features](https://img.shields.io/badge/tabular%20features-91-blue.svg)]()
[![Base models](https://img.shields.io/badge/base%20models-13-blue.svg)]()
[![Agents](https://img.shields.io/badge/autonomous%20agents-22-blueviolet.svg)]()
[![Tests](https://img.shields.io/badge/tests-9282-success.svg)]()
[![Status](https://img.shields.io/badge/status-active%20development-orange.svg)]()

A multi-modal machine learning system built toward the five-tier clinical classification of
human genomic variants — **Pathogenic, Likely Pathogenic, Uncertain Significance, Likely
Benign, and Benign** — in accordance with ACMG/AMP guidelines. It serves a pathogenicity
score today; the five tiers return only with a qualified, bound decision policy.

It integrates genomic sequence, population-stratified allele frequencies, protein and DNA
language-model representations, gene-network topology, tissue-specific expression, and
curated gene–disease evidence into a **91-feature** matrix, consumed by a **13-model**
ensemble whose single authoritative score is the stored convex blend of its members'
probabilities, computed exactly (the logistic stacker is kept as a research candidate,
never served). Protein-structure features are currently **quarantined**: excluded
from the contract, refused at fitting, loading and serving (a model trained with them is
never served), until repaired (see *Feature set*). It is served as a FastAPI REST service and supervised by an
autonomous layer of **22 specialised agents** communicating over a typed message bus.

Training draws on ClinVar on the Genome Reference Consortium human build 38 (GRCh38): 4,399,089 rows (one per variant) after
cleaning, of which 1,490,324 carry a binary label under the current review-status policy and form the training cohort (both
measured 2026-07-24, `docs/measurements/`), annotated from some twenty biological databases.


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
algorithms; splits are meant to be **gene-disjoint**, so scoring well only on catalogued variants in
catalogued genes does not count as solving the problem; and *"I do not know"* is a
**calibrated, auditable output** rather than a probability loitering near 0.5.

The splits are not yet gene-disjoint in fact: they group rows by gene-symbol STRING, and a
string naming several genes is a different group from a string naming one of them. MEASURED
2026-11-08 on Run 15: 19.60% of its test rows and 15.66% of its validation rows share a gene
with a training string (`docs/claims/`). Grouping by gene is Package 1D work
(GENE-IDENTITY-SPLIT-1, open).

---

## Architecture

A multi-branch fusion model wrapped in an autonomous supervisory layer.

```
   Population genetics . Conservation . Functional predictors . Gene-disease
   knowledge bases . Protein structure [QUARANTINED] . Expression . Splice mechanics .
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
                  Stored convex blend (one authoritative score)
                  + predictive entropy; conformal sets (research)
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
| Tabular | 13 base classifiers over the feature matrix, combined by the stored convex blend | live |
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
| 6 | `logistic_regression` | Logistic Regression — also the research stacker (never served) | Linear |
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

A run is **complete evidence only when its `completion_manifest.json` verifies** (package 1, set 1b): the manifest is
written last, binds every output by SHA-256, names the selected predictor and the prediction blocks it produced, and is
re-measured by every consumer (`prediction_artifacts.verify_completed_run`). Out-of-fold predictions carry their original
training-split row, fold, label and unit identities, composed through any subsample and verified against the labels the
ensemble was fitted on; predictions, calibration points and the blend's member-score attributions are all reduced from one
bound capture of each split.

---

## Feature set (91 tabular features)

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
| Protein structure (AlphaFold) — **quarantined** | 0 | the four structural features are excluded from the contract until repaired: `quarantine_policy.py`, `docs/CONTAINMENT_2026-07-24.md` §4 |
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
| **Total** | **91** | |

The count lives in exactly one place — `EXPECTED_TABULAR_FEATURE_COUNT` — and is checked
against the feature list by the test suite, while the feature builder's output is checked
against the list, names and order, before any model trains. A source that fails to populate fails loudly
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
- **Disagreement between stochastic draws is decomposed exactly**, each identity in its own
  unit: the entropy of the mean (nats) is the mean draw entropy plus the mean divergence of
  the draws from their mean; the Bernoulli variance of the mean (probability squared) is the
  mean draw variance plus the spread between draws. A model without stochastic draws reports
  the decomposition as unavailable, never as zero.
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
GET  /rsid/{rs_id}    rs-ID resolution; a prediction only when the served model is ready
POST /predict         Single variant -> prediction record (schema 3.0.0)
POST /batch           Batch of prediction records under one serving identity
```

A prediction record (schema 3.0.0) carries the stored convex blend's score at full binary64
precision, never rounded or clipped; a descriptive binary call under one tie rule (positive
when the score is at least 0.5); the predictive entropy of that score in nats; a descriptive,
uniformly weighted view of how the fitted members disagree (labelled as not the blend); and the
identities that bind
it to its input rows, its combiner and the registered fitted predictor. Calibration, a
decision policy, feature attribution and a coverage interval are each reported as
unavailable until one is qualified and bound -- the five-tier labels are no longer served.
`/predict` and `/batch` answer 503 before any model is called unless the served artifact is
the registered production record with an exact roster and the same fitted predictor
(`predictor_contract_sha256`).

Authentication via `X-API-Key`, rate limiting via `slowapi`, structured JSON logging, and
Prometheus instrumentation, from a multi-stage Dockerfile with builder, api and trainer
targets.

---

## Early results

**No valid performance result yet.** The Run 15 baseline (sealed 2026-06-09, commit
`032a2ab`) reported a test AUROC of 0.9984 on expert-reviewed ClinVar variants grouped by
gene-symbol string, with a comparable unseen-gene-holdout figure. Both evaluations are
**INVALIDATED_EVALUATION** (claims `run15.test.ensemble_evaluation`,
`run15.validation.ensemble_evaluation` and `run15.c3.unseen_gene_generalization`, owner ruling
2026-11-08): MEASURED on the run's own files, every evaluated row's gene-count feature equals the
pathogenic labels of the evaluated rows of its gene, its own label included (0 mismatches over
304,711 test, 146,329 validation and 213,436 holdout rows), the strings shared genes with
training, and the holdout's graph-network scores were trained on its labels. The numbers are
kept as a record of what the run printed; they are not evidence, and how much they overstate
is not estimated. The records, with every input bound by SHA-256: `docs/claims/`.

A trivial baseline sets the scale: on the same test split, the pathogenic rate of the variant's
consequence category alone (19 categories, rates from training) comes within three hundredths
of the reported figure (measured 2026-11-08; `docs/CHANGELOG.md`). A result from the next full
run will be reported as an increment over such baselines, within variant classes, on a
gene-disjoint evaluation.

---

## Quickstart

```bash
# Environments: two locked dependency profiles, each installed EXACTLY from its hashed lock into a NEW folder and then proven to
# BE that lock (src/genomic_variant_classifier/deps/profiles.py; targets linux-3.11, linux-3.12, windows-3.12)
python scripts/install_locked_profile.py --profile train-test --target linux-3.12 --venv ../env-train-test   # training and tests
python scripts/install_locked_profile.py --profile drift      --target linux-3.12 --venv ../env-drift        # drift monitor:
#   nannyml requires lightgbm<4.6 while the ensemble trains on 4.6.0, so the drift monitor has its own environment

# Tests, in the train-test environment (.github/workflows/ci.yml adds --tb=short and coverage, and starts pytest without the
# workflow-command variables). The repository root's conftest.py guards BOTH suites: it runs joblib's thread backend
# sequentially for the session and fails a test that leaves the warning filters different in any way (an entry added included);
# imports the libraries that register filters when first imported at session start, so no test's filters depend on which tests
# ran with it, and stops the session if collection registers one; contains temporary files under pytest's base temporary folder;
# sets HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE to 1 and refuses every attempt to reach the network inside
# the call -- a proxy on the loopback included, from any *_proxy variable or the system's own settings -- failing the test that
# tried (since 2026-11-05); and fails the session if the working tree changed. tests/conftest.py
# adds --assert-suite-size, which requires the collected count to equal tests/EXPECTED_SUITE_SIZE. Git 2.45 or later is required
# (the release-approval checks read Git objects with --no-lazy-fetch, and refuse an older Git by name). Three tests check
# git-ignored LOCAL data by design (tests/local_data.py: the UniProt index, the EVE corpus, the Run-15 matrix): they run only
# where that data is present, and are skipped with one reason in a clone; `-m local_data` runs exactly them
python -m pytest tests/ -q -rs --assert-suite-size
python -m pytest tests/ -q -rs -m local_data          # only the local-data tests
python -m pytest tests_drift/ -v -rs                  # in the drift environment, with the drift job's options

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
src/genomic_variant_classifier/        (every subpackage and top-level module, listed 2026-11-04: 22 packages, and monitoring/
                                        and training/, which have no __init__.py)
  agent_layer/               - agents, typed message bus, orchestrator, shared state
  api/                       - FastAPI service, auth, schemas, inference
  conformal/                 - conformal prediction, from scratch in numpy
  data/                      - connectors, extract-transform-load (ETL), split protocol, data preparation
  deps/                      - dependency governance: the locked profiles, their installation and observation
  environment_qualification/ - qualification of the R environment for the scientific stage
  evaluation/                - clinical evaluator, metric registry, thresholds, artifacts
  features/                  - feature engineering
  inference/                 - exact statistical inference for the scientific stage (the DANDELION method's ranking, confirmation)
  models/                    - variant ensemble, graph networks, uncertainty wrappers, sequence convolutional neural network
  monitoring/                - drift detection, reference profiles, model registry
  operations/                - operation classification, evidence and admission verification
  paths/                     - where the project keeps things: one authority, four roots
  pipelines/                 - ribonucleic acid (RNA) splice, protein structure
  provenance/                - scientific evidence provenance
  reports/                   - HyperText Markup Language (HTML) report generation
  repository_hygiene/        - what belongs in the repository and what is residue (the working-tree snapshot)
  repository_measurement/    - the semantics of repository inspections
  repository_records/        - durable machine evidence (verification and migration records)
  source_monitor/            - qualified monitoring of upstream data sources
  state/                     - mutable operational state: atomic, identified, fail-closed
  training/                  - continual learner, Elastic Weight Consolidation
  transactions/              - rollback state, transactional and outside the tree
  utils/                     - shared helpers
  containment.py, model_admission.py, quarantine_policy.py - the containment rules, the one model-admission route, the
                              quarantined features
scripts/                     - training entry points, preflight gates, drift monitoring, installers, runbooks
tests/                       - unit, integration and conformal suites (the train-test environment)
tests_drift/                 - the drift-monitor suite (the drift environment)
conftest.py                  - the session guards for both suites
docs/                        - ROADMAP.md, CHANGELOG.md, incidents/, sessions/, measurements/
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
