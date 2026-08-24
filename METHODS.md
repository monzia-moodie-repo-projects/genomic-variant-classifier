# Methods

## Genomic Variant Pathogenicity Classifier — Technical Description

**Version:** Phase 2 (v2.0.0)
**Held-out performance:** see section 5.1. The figure previously stated here
is not attributable from this repository.

---

## 1. Data

### 1.1 Training labels

Variants were obtained from ClinVar (GRCh38, quarterly release) and filtered to
high-confidence clinical classifications using ClinVar review status tier ≤ 3
(criteria provided by at least one submitter, no conflicting interpretations).
Pathogenic and Likely pathogenic variants were assigned label 1; Benign and
Likely benign variants were assigned label 0. Variants of Uncertain Significance
(VUS) and those with conflicting interpretations were excluded from training.

Resulting label distribution: ~15% pathogenic, ~85% benign (~1.2 M variants
after quality filtering).

### 1.2 Feature annotation

Variant annotations were added from the following sources, in pipeline order:

| Step | Source | Features |
|------|--------|----------|
| 1 | dbNSFP v4.6 | SIFT, PolyPhen-2 HDIV, REVEL, CADD raw, PhyloP 100-way, GERP++ |
| 2 | PhyloP v1 | phylop_score (multi-alignment conservation override) |
| 3 | CADD v1.7 | cadd_phred (REST API or pre-scored file; optional) |
| 4 | SpliceAI v1.3 | splice_ai_score (max delta score across 4 splice signals) |
| 5 | AlphaMissense | alphamissense_score (per-amino-acid pathogenicity) |
| 6 | GTEx v8 | max TPM, tissue expression breadth, eQTL flag, effect size |
| 7 | VEP v110 | Consequence, codon position, exon/intron annotation |
| 8 | OMIM | Disease count, inheritance mode |
| 9 | ClinGen | Gene validity score (curated gene–disease relationships) |
| 10 | dbSNP build 156 | Supplemental allele frequency |
| 11 | EVE | Evolutionary model variant effect score |
| 12 | MaxEntScan (Phase 6.1) | Splice-site strength score, distance to canonical splice site, exon number, canonical GT-AG flag |
| 13 | AlphaFold / UniProt (Phase 6.2) | Per-residue pLDDT, relative solvent accessibility, secondary structure class, distance to active site |
| 14 | gnomAD v4.1 constraint | pLI, LOEUF, synonymous Z, missense Z |
| 15 | 1000 Genomes | Per-superpopulation allele frequencies (AFR, EUR, EAS, SAS, AMR) |
| 16 | FinnGen R12 / R13 | Finnish and non-Finnish-Swedish-Estonian allele frequency, enrichment |
| 17 | ESM-2 (Evolutionary Scale Modeling 2) | Protein language-model log-likelihood ratio, normalised delta |
| 18 | Nucleotide Transformer | DNA language-model log-likelihood ratio, normalised delta |
| 19 | COSMIC Cancer Mutation Census | Recurrence count, significance tier |
| 20 | KEGG | Pathway count, disease-pathway flag |
| 21 | Reactome | Pathway membership count |
| 22 | RNA-seq (Phase D) | Mean log TPM, detection rate, log2 coefficient of variation, log2 fold-change, differential-expression significance |
| 23 | STRING-DB | Gene-network pathogenicity score; heterogeneous knowledge-graph score |

**HGMD (Human Gene Mutation Database) Professional is NOT a source.** An earlier version of
this table listed it as source 12, supplying a "disease mutation flag" and a "report count".
That was never true: the licence was never obtained (see `docs/ROADMAP.md` — "HGMD | hgmd_* (2)
| PAID, blocked"), the connector was never wired, and both columns were CONSTANT ZERO for the
entire life of the project — contributing nothing while the methods document credited them.
They were removed from the feature contract on 2026-07-13.

They will not be reinstated in their previous form even if the licence is obtained. HGMD's
"DM" (disease mutation) classification is, at the variant level, a near-copy of the
ClinVar-Pathogenic label this model is trained to predict; using it as a feature would leak the
target, and — because a novel variant of uncertain significance has no HGMD entry — would bias
the model toward "benign" on precisely the variants it exists to classify. Should the licence
be obtained, HGMD will enter as a **gene-level, leave-one-out aggregate** (count of HGMD-DM
variants in the gene, excluding the variant being scored), mirroring the existing
`n_pathogenic_in_gene` feature.

Allele frequencies were sourced primarily from gnomAD v4.1 exomes; variants
absent from gnomAD were supplemented with 1000 Genomes Phase 3 allele
frequencies.

### 1.3 Train / validation / test split

Variants were split gene-stratified (no gene appears in more than one split)
using GroupShuffleSplit:

- **Train**: 70% of genes
- **Validation**: 10% of genes (used for hyperparameter selection and Platt
  scaling)
- **Test / holdout**: 20% of genes (single evaluation at end of training)

This strategy prevents gene-level label leakage, which inflates AUROC estimates
when the same gene appears in both train and test sets.

---

## 2. Feature Engineering

A total of **95 tabular features** are derived from raw annotations. This figure is
`EXPECTED_TABULAR_FEATURE_COUNT` in `src/genomic_variant_classifier/models/variant_ensemble.py`,
and it is the single source of truth: `tests/unit/test_methods_feature_count.py` fails the test
suite if the number stated here and the number in the code ever disagree.

> **Correction, 2026-07-13.** This section previously claimed **64** features, and the group
> table below summed to 62. Both were long stale — the contract had grown to 95 while the
> document stood still, and nothing re-derived it. Restating the number by hand would only
> reset the clock on the same defect, so the agreement is now enforced by a test.

| Group | Count | Description |
|-------|-------|-------------|
| Allele frequency | 6 | Raw AF, log₁₀ AF, binary rarity indicators |
| Variant type | 7 | SNV/indel, insertion/deletion, ref/alt length |
| Consequence | 6 | Missense, loss-of-function, splice, coding, severity score |
| Functional scores | 9 | CADD, SIFT, PolyPhen-2, REVEL, PhyloP, GERP++, AlphaMissense, SpliceAI, EVE |
| Binary flags + meta-score | 5 | Thresholded predictor flags and the count of tools calling pathogenic |
| Gene-level | 4 | n_pathogenic_in_gene, gene constraint observed/expected, gene has known disease |
| Protein features | 2 | UniProt annotation present, count of known pathogenic protein variants |
| GTEx expression | 6 | Max TPM, tissue breadth, specificity, eQTL flag, p-value, effect size |
| Variant coding context | 2 | Codon position, dbSNP allele frequency |
| Gene–disease annotation | 4 | OMIM disease count, OMIM molecular-basis count, OMIM dominant inheritance, ClinGen validity |
| LOVD | 1 | LOVD variant class |
| Chromosome context | 3 | Autosome, sex chromosome, mitochondrial |
| Gene network | 2 | STRING-DB graph-neural-network score; heterogeneous knowledge-graph score |
| RNA splice context | 5 | MaxEntScan score, distance to splice site, exon number, canonical GT-AG flag, splice indicator |
| Protein structure | 4 | AlphaFold pLDDT, relative solvent accessibility, secondary structure, distance to active site |
| 1000 Genomes population AF | 5 | AFR, EUR, EAS, SAS, AMR allele frequencies |
| FinnGen R12 | 3 | Finnish AF, non-Finnish-Swedish-Estonian AF, enrichment |
| FinnGen R13 | 3 | Finnish AF, non-Finnish-Swedish-Estonian AF, enrichment |
| ESM-2 (protein language model) | 2 | Log-likelihood ratio, normalised delta |
| Nucleotide Transformer (DNA language model) | 2 | Log-likelihood ratio, normalised delta |
| COSMIC Cancer Mutation Census | 2 | Recurrence count, significance tier |
| KEGG | 2 | Pathway count, disease-pathway flag |
| Reactome | 1 | Pathway membership count |
| gnomAD v4.1 constraint | 4 | pLI, LOEUF, synonymous Z, missense Z |
| RNA-seq expression | 5 | Mean log TPM, detection rate, log2 coefficient of variation, log2 fold-change, differential-expression significance |
| **Total** | **95** | = `EXPECTED_TABULAR_FEATURE_COUNT` |

Missing values were imputed with biologically neutral defaults (e.g., AF = 0 for
absent from gnomAD, SIFT = 0.5 for uncovered positions).

---

## 3. Model Architecture

### 3.1 Base estimators

Four tabular base models were trained on the 64-feature matrix:

| Model | Library | Key hyperparameters |
|-------|---------|---------------------|
| LightGBM | lightgbm 4.x | num_leaves=63, learning_rate=0.05, n_estimators=500 |
| XGBoost | xgboost 2.x | max_depth=6, learning_rate=0.05, n_estimators=500 |
| Gradient Boosting | scikit-learn | max_depth=5, learning_rate=0.05, n_estimators=300 |
| Random Forest | scikit-learn | n_estimators=300, max_features=0.4 |

Hyperparameters were optimised using Optuna (TPE sampler, 100 trials) on the
validation split. Final hyperparameters are stored in
`models/best_lgbm_params.json` and `models/best_xgboost_params.json`.

A 1D-CNN sequence model was trained on 101-bp FASTA context windows but is
excluded from the inference pipeline (requires sequence context unavailable
at API inference time).

### 3.2 Stacking meta-learner

Out-of-fold predictions from 5-fold cross-validation were used as inputs to a
Logistic Regression meta-learner (C = 1.0, solver = lbfgs). The stacking
ensemble averages model strengths and reduces variance.

### 3.3 Graph neural network (gene-level prior)

A variant graph was constructed from the STRING protein interaction database
(combined score ≥ 500) with genes as nodes and interaction confidence as edge
weights. A 3-layer Graph Attention Network (GAT) with 64-dimensional hidden
layers was trained to predict gene-level pathogenicity from the mean variant
features per gene. At inference time, gene-level GNN scores are pre-computed
and stored as a lookup table (`GNNScorer`), enabling O(1) retrieval.

---

## 4. Calibration

Probability calibration was performed using Platt scaling (logistic regression
fit on the validation split predictions). Classification thresholds were set by
anchoring to ≥ 90% positive predictive value (PPV) for the Pathogenic tier on
the validation set, then sweeping down through the ACMG five-tier scale.
Calibrated thresholds are stored in `models/classification_thresholds.json`.

Conformal prediction intervals (split conformal, Papadopoulos 2002) provide
guaranteed marginal coverage at α ∈ {0.01, 0.05, 0.10, 0.20} calibrated on
the validation split.

---

## 5. Evaluation

### 5.1 Held-out performance

> **Correction, 2026-08-24.** This section previously stated
> `AUROC (gene-stratified holdout) 0.9847` and `AUPRC 0.8936`, and the header
> block above repeated the AUROC as *"0.9847 (gene-stratified, 154,404
> variants)"*. **Neither figure is attributable from this repository.**
>
> `docs/measurements/MEASUREMENT_2026-08-08_baseline1-provenance-census.md`
> established that the earliest appearance of `0.9847` is a commit SUBJECT
> LINE. No committed artefact establishes it, and `git ls-files outputs`
> returns nothing from Phase 2 and nothing from Run 8.
>
> The cohort quoted beside it IS attributable, and it convicts the claim:
> `154,404` is `n_val` in `outputs/run14/full/metrics.json` -- Run 14's
> **validation** split -- whose measured AUROC is `0.9974`, recorded four lines
> away in the same file. The number and its denominator have different origins.
>
> `docs/audits/README_AUDIT_2026-07-14.md` asked whether Run 8's holdout AUROC
> was `0.9847` or `0.9863` and recorded the question as UNRESOLVED. It cannot
> be resolved here: the Run 8 artefacts were never committed.
>
> **No corrected figure is substituted, because there is none to substitute.**
> Restating a number by hand is what produced this defect; the 2026-07-13
> correction in section 2 records the same lesson about the same document.

Held-out performance is reported PER RUN, from committed artefacts, and is not
restated here:

| Run | Artefact | What it carries |
|---|---|---|
| Run 14 | `outputs/run14/full/metrics.json` | test and validation AUROC, AUPRC, F1, MCC, Brier, and all three split sizes |
| Run 14 | `outputs/run14/reproducibility_manifest.json` | per-model metrics, artefact digests, pinned environment |
| Run 10b | `outputs/run10b_final/full/metrics_partial.json` | declares `"status": "partial"`; three outputs recorded as lost |
| calibration | `outputs/calibration/calibration_metrics.json` | expected calibration error, 15 bins |

A figure read from one of these carries its origin. `docs/METRICS.md` reports
test AUROC and out-of-fold blend in separate named columns, and
`SealedEvaluation` (`src/genomic_variant_classifier/evaluation/sealed_evaluation.py`)
makes that distinction a **field** rather than a key-name convention, so a
computed figure and one scraped from a training log can no longer be read as
one quantity.

### 5.2 External validation

The pipeline can be evaluated against external cohorts using
`scripts/validate_external.py`, which produces AUROC, AUPRC, ECE, MCE, and
per-threshold sensitivity/specificity/PPV/NPV breakdowns.

---

## 6. Software and Reproducibility

| Component | Version |
|-----------|---------|
| Python | 3.11–3.12 |
| LightGBM | ≥ 4.3 |
| XGBoost | ≥ 2.0 |
| scikit-learn | ≥ 1.4 |
| FastAPI | ≥ 0.111 |
| Optuna | ≥ 3.5 |

Training is fully reproducible from `scripts/run_phase2_eval.py` given the
same input data and random seed (`--random-state 42`).  All splits and
hyperparameter search results are written to `data/splits/` and
`models/` respectively.

---

## 7. Ethical Considerations

This classifier is intended as a research and clinical decision-support tool.
Outputs should be interpreted by trained clinical geneticists in the context of
the full clinical picture.  The model does not account for compound
heterozygosity, polygenic risk, or phenotype-specific penetrance.  Users are
responsible for compliance with institutional guidelines for variant
interpretation.

---

## References

1. Landrum MJ et al. ClinVar: improving access to variant interpretations and
   supporting evidence. *Nucleic Acids Res.* 2018;46:D1062-D1067.
2. Karczewski KJ et al. The mutational constraint spectrum quantified from
   variation in 141,456 humans. *Nature.* 2020;581:434-443.
3. Cheng J et al. Accurate proteome-wide missense variant effect prediction
   with AlphaMissense. *Science.* 2023;381:eadg7492.
4. Jaganathan K et al. Predicting Splicing from Primary Sequence with Deep
   Learning. *Cell.* 2019;176:535-548.
5. Yeo G, Burge CB. Maximum Entropy Modeling of Short Sequence Motifs with
   Applications to RNA Splicing Signals. *J Comput Biol.* 2004;11:377-394.
6. Jumper J et al. Highly accurate protein structure prediction with
   AlphaFold. *Nature.* 2021;596:583-589.
7. Szklarczyk D et al. The STRING database in 2021. *Nucleic Acids Res.*
   2021;49:D605-D612.
8. Papadopoulos H et al. Inductive confidence machines for regression.
   *ECML.* 2002;345-356.
