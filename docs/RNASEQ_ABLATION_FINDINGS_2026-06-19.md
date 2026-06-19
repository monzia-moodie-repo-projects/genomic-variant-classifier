# RNA-seq feature-importance ablation — findings (2026-06-19)

Author: Monzia Moodie

## Why this ran
Across recent runs the five `rnaseq_*` features sat at the TOP of the gradient/permutation
feature-importance ranking (e.g. `rnaseq_de_neglog10p`, `rnaseq_mean_log_tpm`, `rnaseq_log2_cv`,
`rnaseq_log2fc` above `consequence_severity` and `splice_ai_score`). A GTEx Brain-Cortex-vs-Whole-Blood
TISSUE contrast ranking as a top pathogenicity predictor is biologically implausible as *direct* signal,
so it had to be tested rather than trusted. These features are GENE-LEVEL (constant within a gene), so the
risk is gene-identity / constraint memorisation dressed up as RNA biology — the same class of concern as
`n_pathogenic_in_gene`.

DE leakage was already guarded (`leakage_guard = normal_tissue_reference_not_variant_label`): the contrast
is normal-tissue reference data, independent of the variant labels. So the question is NOT label leakage but
whether the importance reflects real discriminative contribution.

## Design
Four perturbed copies of the canonical parquet (`scripts/make_rnaseq_ablation_parquets.py`, seed 20260618)
plus a no-parquet control, each fed to the same Phase-2 harness
(`--max-train 5000 --skip-nn --skip-kan --skip-cnn`, 10 base models, gene-disjoint splits):

| config      | what changed                                              |
|-------------|-----------------------------------------------------------|
| full        | canonical real parquet (all 5 features)                   |
| drop_de     | `rnaseq_log2fc` & `rnaseq_de_neglog10p` zeroed (DE block) |
| drop_all    | all 5 `rnaseq_*` zeroed                                    |
| shuffle     | 5 features permuted across genes (breaks gene→profile map)|
| no_rnaseq   | `--rnaseq-path` omitted entirely (control for drop_all)   |

Context caveat: these runs carry only ClinVar-derived + SpliceAI(cache) + a constant pLDDT=50 structural
STUB + `rnaseq_*` + `n_pathogenic_in_gene`. dbNSFP / AlphaMissense / GTEx / constraint / STRING-GNN / kg are
NOT present. So this isolates `rnaseq_*` in a MINIMAL feature context; under the full Run-17 source set the
marginal value of `rnaseq_*` can only shrink (more competing real features). The conclusion is therefore
conservative.

## Results (held-out AUROC)

| config     | test AUROC | val AUROC | Δtest vs full | Δval vs full |
|------------|-----------:|----------:|--------------:|-------------:|
| full       |     0.9360 |    0.9461 |             — |            — |
| drop_de    |     0.9346 |    0.9461 |       −0.0014 |       0.0000 |
| shuffle    |     0.9354 |    0.9383 |       −0.0006 |      −0.0078 |
| drop_all   |     0.9304 |    0.9370 |       −0.0056 |      −0.0091 |
| no_rnaseq  |     0.9304 |    0.9370 |       −0.0056 |      −0.0091 |

Sanity: `no_rnaseq == drop_all` exactly (omitting the parquet ≡ zeroing the features) → harness wiring correct.

Derived quantities:
- **Total marginal value of ALL rnaseq** (full − drop_all): **+0.0056 test / +0.0091 val** (~0.6–0.9 AUROC pt).
- **DE-block marginal value** (full − drop_de): **+0.0014 test / +0.0000 val** — the tissue-contrast DE pair
  (the most "implausible as pathogenicity" features) contributes essentially nothing to held-out AUROC.
- **Gene-shuffle retention** (shuffle − drop_all): **+0.0050 test / +0.0013 val**. On test the benefit largely
  survives random gene→profile reassignment (≈89% retained → non-gene-specific); on val it mostly collapses
  toward drop_all (≈14% retained → gene-specific). The two splits DISAGREE, and all magnitudes are ≤0.009,
  i.e. within the noise of a single-seed `--max-train 5000` run. The shuffle test is therefore inconclusive
  at this scale and must NOT be cited as proof of either interpretation.

## Within-gene AUROC (the firmer control)
`scripts/compute_within_gene_auroc.py` on the full-rnaseq run, restricted to genes with ≥2 of each class:

| split | genes | weighted within-gene AUROC | unweighted |
|-------|------:|---------------------------:|-----------:|
| test  |   780 |                     0.9512 |     0.9261 |
| val   |   344 |                     0.9479 |     0.9240 |

WITHIN a single gene every gene-level feature is constant — `rnaseq_*` AND `n_pathogenic_in_gene` cannot vary
— so within-gene discrimination is driven purely by VARIANT-level features (consequence, SpliceAI, length,
etc.). Weighted within-gene AUROC ≈ 0.95 means the model discriminates pathogenic vs benign strongly even
where the gene priors give it nothing. The low-AUROC tail (PAK2 n=4 → 0.00, ANKRD26 → 0.17, …) is entirely
small-n genes; the median within-gene AUROC is ~0.976.

## Conclusion (honest)
1. The `rnaseq_*` feature-importance dominance is **NOT** matched by discriminative contribution: removing all
   five costs only ~0.6–0.9 AUROC pt, and the tissue-contrast DE pair costs ~0 on val. High importance + low
   marginal value is the classic signature of **tree split-bias toward high-cardinality continuous features**,
   not genuine reliance on tissue biology or gene identity.
2. Strong within-gene AUROC (~0.95) shows the model's discrimination rests on variant-level features,
   independent of BOTH `rnaseq_*` and `n_pathogenic_in_gene`. This is the most reassuring single number.
3. The gene-shuffle is inconclusive at this scale (test/val disagree, tiny magnitudes). It should be re-run at
   Run-17 scale (full feature set, larger `--max-train`, ≥3 seeds) before any claim of shuffle-invariance.
4. Reporting guidance: cite `rnaseq_*` as a **redundant gene-prior the ensemble leans on when present**, NOT
   as biological signal; always pair the importance ranking with this ablation + the within-gene AUROC.

## Metrics glossary additions
- **within-gene AUROC** — AUROC computed separately within each gene (≥2 of each class), then summarised
  weighted (by n) and unweighted. Controls for ALL gene-level features (which are constant within a gene);
  isolates variant-level discriminative power. Range [0,1]; ~0.95 here.
- **drop-block ablation** — ΔAUROC from zeroing a named feature block (e.g. DE pair, all rnaseq). Measures the
  block's MARGINAL contribution, which (unlike feature importance) is robust to split-bias.
- **gene-shuffle permutation** — permute gene-level features across genes (break the gene→value map) and re-fit.
  If a block encodes real per-gene signal, AUROC should fall toward the drop-block value; invariance implies a
  non-gene-specific (distributional / split-bias) contribution. Inconclusive here at small scale.

## Provenance / status
- Ablation parquets: `scripts/make_rnaseq_ablation_parquets.py` (committed 0febfb8).
- Export + within-gene: `scripts/export_phase2_predictions_from_saved_models.py`,
  `scripts/compute_within_gene_auroc.py`, `scripts/diagnose_phase2_prediction_reconstruction.py` (committed f2141c0).
- INFERENCE CONTRACT discovered en route: the saved base models in `models/ensemble_models/*.joblib` consume
  the RAW (unscaled) feature matrix. Applying `scaler.joblib` before `predict_proba` double-scales and collapses
  AUROC (tree models → ~0.45–0.50). Any standalone inference path MUST feed raw X. Confirmed by exact AUROC
  reconstruction (0.9360 raw vs 0.6083 double-scaled).
