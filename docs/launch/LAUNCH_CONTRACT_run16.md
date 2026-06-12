# Run 16 Launch Contract (v2)

Provenance: 2026-06-12 Run-16b smoke-gate session.
Status: TABULAR RUN 16 VALIDATED + LAUNCH-READY. 1KGP + GNN -> committed Run-17 scope
(RUN17_SCOPE.md), NOT deferrals. Author: Monzia Moodie.

---

## 0. What "validated" means

`--fast` all-models smoke models/smoke_run16b ran end-to-end (962 s, no OOM, no crash):
13 base models, 81 features, ENSEMBLE_STACKER test AUROC 0.9994. Feature-matrix
population confirmed (splits/X_*.parquet): gnomAD af_log10, dbNSFP cadd/sift/revel/
n_tools_pathogenic live; LOVD all-default (smoke-size). Schema re-sealed 78->81
(run16b-smoke), green vs all 3 splits. Smoke is necessary, NOT sufficient -- the on-box
gates in Sec. 4 are authoritative at full scale.

---

## 1. Validated flag set (full-cohort run)

vs the smoke: full cohort, production ESM-2 (650M), GPU, full estimators (NO --fast).

```
python scripts/train.py \
  --clinvar           data/processed/clinvar_grch38_clean_seq.parquet \
  --alphamissense     data/external/alphamissense/AlphaMissense_hg38.tsv.gz \
  --gnomad            data/processed/gnomad_v4_exomes.parquet \
  --gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
  --dbnsfp-path       data/external/dbnsfp/dbnsfp_clinvar_index.parquet \
  --lovd-path         data/external/lovd/lovd_all_variants.parquet \
  --esm2-model        esm2_t33_650M_UR50D \
  --esm2-uniprot-index data/external/uniprot/uniprot_human_reviewed.parquet \
  --esm2-device       cuda \
  --out-dir           outputs/run16
```

OMITTED (verified reasons):
- --uniprot     : on-disk parquet is the ESM-2 index (gene_symbol/uniprot_id/sequence);
                  _join_uniprot needs source_id + pathogenicity -> would KeyError/dead.
- --finngen-path: no FinnGen file on disk.
- --kg-path     : flag does NOT exist in train.py; 1KGP not staged. -> Run 17 (RUN17_SCOPE A).
- (GNN)         : no --string-db flag; gnn_score is an unset placeholder in data-prep.
                  -> Run 17 (RUN17_SCOPE B).

dbNSFP: --dbnsfp-path may point anywhere in data/external/dbnsfp/. The connector
hard-codes its cache name to dbnsfp_clinvar_index.parquet (the docstring saying
dbnsfp_full_index.parquet is WRONG -- fixed by patch_dbnsfp_docstring.py). It reads the
2.69M-row ClinVar index; the 895 MB full index is never read.

---

## 2. Resolved scope (1KGP + GNN -> Run 17, committed)

These were investigated to source, not deferred by hand-wave. Both are tracked with hard
acceptance criteria in RUN17_SCOPE.md.

### 2.1  1000 Genomes -- RESOLVED: Run 17 (Track A)
- ThousandGenomesConnector fills only the COMBINED allele_freq; it does NOT populate
  af_1kg_afr/eur/eas/sas/amr (those have no source wired -> permanent stubs until B6/A6).
- No kg parquet staged; build_1kg_parquet.py (referenced by the connector docstring)
  does not exist. Requires VCF acquisition + build script + a --kg-path flag.
- Run-16 impact: af_log10/af_* live from gnomAD (66.6% AF coverage); af_1kg_* dormant.

### 2.2  STRING-DB GNN -- RESOLVED: Run 17 (Track B)
- gnn.py is complete (StringDBGraph, VariantGAT, GNNTrainer, GNNScorer.score_dataframe)
  but is NOT wired into train.py; gnn_score = df.get("gnn_score", 0.5) -> dormant.
- Making gnn_score live REQUIRES cross-fitting to the gene-disjoint splits (train GNN on
  train-fold genes only, score test genes by propagation). Without it, gnn_score leaks
  test labels into the ensemble -- the 0.9994 inflation hazard, amplified.
- Run-16 impact: gnn_score dormant-by-design (scaled 0.0). Honest placeholder.

---

## 3. Staging manifest (Vast.ai)

SHIP:
| File | Size | Notes |
|---|---|---|
| data/processed/clinvar_grch38_clean_seq.parquet | 523 MB | 19 cols + ReviewStatus |
| data/external/alphamissense/AlphaMissense_hg38.tsv.gz | 613 MB | TSV, parsed on box |
| data/external/alphamissense/alphamissense_protein_index.parquet | 17.8 MB | ESM-2 coords |
| data/processed/gnomad_v4_exomes.parquet | 37.6 MB | variant_id, allele_freq |
| data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv | 91 MB | + .constraint_index.parquet |
| data/external/dbnsfp/dbnsfp_clinvar_index.parquet | 34.6 MB | the connector cache (2.69M) |
| data/external/lovd/lovd_all_variants.parquet | 0.25 MB | lovd_variant_class |
| data/external/uniprot/uniprot_human_reviewed.parquet | 11 MB | ESM-2 index |
| data/raw/cache/spliceai_scores_snv.parquet | 431 MB | SpliceAI cache (no OOM in smoke) |
| data/reference/schema/schema_baseline.json | small | 81-col, run16b-smoke |

DO NOT SHIP:
- data/external/dbnsfp/dbnsfp_full_index.parquet (895 MB) -- never read by the connector.
- the 34.6 MB promoted dbnsfp_full_index.parquet copy -- redundant; delete locally too.
- data/raw/cache/alphamissense_scores_hg38.parquet.OOMbak (740 MB) -- 16 GiB OOM.
- models/smoke_run16b/*, clinvar_enriched.parquet -- smoke artifacts.

ESM-2 650M: download on box (facebook/esm2_t33_650M_UR50D) or SCP the HF cache.

---

## 4. On-box BLOCKING gates (before training; abort if any fails)

After full data-prep produces outputs/run16/splits/, BEFORE model training:

1. SCHEMA DRIFT (authoritative):
   `python scripts/run_schema_drift_check.py --matrix outputs/run16/splits/X_train.parquet`
   PASS = green (exit 0). A red dtype change = a column flipped int->float at scale;
   re-seal from the full matrix, do not proceed.

2. FEATURE POPULATION:
   `python scripts/audit_smoke_feature_population.py outputs/run16/splits`
   PASS = all FAIL-severity sources POPULATED. lovd_variant_class SHOULD now be > 0 at
   full scale -- if still 0 across 1.49M it is a join-key bug, not coverage: STOP.

3. STANDING RUN GATES: full suite green; zero open BUG incidents; all prior anomalies
   closed; all <DECISION> resolved; checkpoint/budget/trap verified.

---

## 5. Launch + monitoring

- ONE preflight script fills instance/SSH host+port/key (id_lambda_run8)/paths, validated.
- vastai: lowest $/hr 4090 (~$0.38-0.76; dlperf>=80 pcie_bw>=12). Symlink
  /workspace/{data,outputs} -> repo.
- Checkpoint each base estimator + OOF right after its AUROC log; verify < 30 min else ABORT.
- Irreversible commands (vastai destroy, rm -rf) in a SEPARATE paste block after manual verify.

---

## 6. Watch-items (monitor at full scale)

- cnn_1d OOF 0.5132 / test 0.4782 -- at/below random across 2 smokes. Full scale is the
  discriminator: still ~0.5 at 1.49M => real architecture/scaling defect.
- LOVD 0/1681 -- see gate 4.2. Expect > 0 at full scale.
- CIRCULARITY: cadd_phred is the #1 feature; CADD/REVEL/SIFT/AlphaMissense are themselves
  ClinVar-trained. Run a no_meta_predictors ablation to quantify independent signal;
  document in the metrics glossary. Matters more than the headline AUROC.
- kan: AUROC 0.9859 / Brier 0.2223 (poor calibration); ranks fine for the stacker.
- n_pathogenic_in_gene #2 -- re-confirm gene-disjoint splits + no cross-fold count leakage.
- DORMANT-by-design (NOT bugs): gnn_score, af_1kg_*, and uniprot features = 0. Will
  activate in Run 17 (gnn) / a 1KGP build. The schema baseline already seals them.
- real_data_prep.py:501 FutureWarning (gnomAD fillna downcast) -- tech debt.

---

## 7. Post-run verification + teardown

- SCP outputs back; `echo y | vastai destroy` immediately.
- Re-run feature-population audit + schema drift-check on the returned full splits.
- Document: per-model algorithm comparison, METRICS glossary deltas, watch-item
  resolutions; CHANGELOG + ROADMAP + SESSION doc; commit + push.
