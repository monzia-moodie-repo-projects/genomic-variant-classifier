# RUN 17 — GPU provisioning runbook

**Author:** Monzia Moodie
**Pinned HEAD:** `3a0c988` (clean hygiene commit, 2026-07-06; update to the actual commit you SCP up)
**Compute:** Vast.ai RTX 4090 (~$0.38–0.76/hr). Local machine is CPU-only — never run the ensemble locally.

> **REVISION 2026-07-06 (97-feature update).** Since the last revision the matrix grew 87 → **97**
> features via three connectors activated this session: Nucleotide Transformer (`genomiclm_*`, via the
> already-present `--seq-windows`), COSMIC CMC (`cosmic_*`, `--cosmic-path`), and KEGG (`kegg_*`,
> `--kegg-path`). The step-1 data list and the step-3 smoke command below were updated accordingly.
> The launcher (`launch_run17_baseline.sh`) already hard-fails if the CMC TSV or KEGG parquet is absent.
> ESM-2/EVE are NO LONGER stubbed (HGVSp parser delivered) — the old "expected-zero" caveat is retired.

This is the on-box sequence from instance boot to teardown. Steps marked **[live ops]** are user-executed
(SSH / SCP / vastai / spend) and cannot be dry-run in the sandbox; everything they invoke is validated.

---

## 0. Pre-SCP checklist (on the laptop)

- `git log --oneline -1 origin/main` → note the SHA; this is `EXPECTED_HEAD`.
- The kg parquet committed in-repo is the single tracked file under the otherwise-ignored `data/external`;
  all other data is local-only and must be SCP'd explicitly.
- **STRING artifacts — send BOTH** (this is the one cross-gate inconsistency to pre-empt):
  - `launch_run17_baseline.sh` step-1 **hard-fails** without `data/external/string/9606.protein.links.detailed.v12.0.txt.gz` + `…info…txt.gz`.
  - `preflight_run17.py` is satisfied by the cached `data/raw/cache/string_graph_700.pkl`.
  - SCP **all three** so neither gate trips.

## 1. SCP up [live ops]

- Repo at `EXPECTED_HEAD` → `/workspace/genomic-variant-classifier`.
- Data tree under `/workspace/genomic-variant-classifier/data/` including: clinvar/gnomad/spliceai/
  alphamissense/gnomad-constraint/dbnsfp/gtex/reactome (+ `ReactomePathways.gmt`)/rnaseq/kg, the STRING
  trio above, and (optional) lovd.
- **NEW (2026-07-06) — REQUIRED for the 97-feature run; the launcher hard-fails without them:**
  - `data/external/cosmic/CancerMutationCensus_AllData_v104_GRCh37.tsv.gz` (~301 MiB; COSMIC CMC — carries the GRCh38 position column)
  - `data/external/kegg_gene_pathways.parquet` (KEGG gene→pathway mapping)
  Both are on Drive: `rclone copy genvarcla:genomic-variant-classifier/data/external/cosmic/CancerMutationCensus_AllData_v104_GRCh37.tsv.gz data/external/cosmic/ -P` and
  `rclone copy genvarcla:genomic-variant-classifier/data/external/kegg_gene_pathways.parquet data/external/ -P` (run ON the VM).
- Symlink `/workspace/{data,outputs}` → repo if the box layout needs it; `rm -rf` the target before `ln -s`.

## 2. VM environment gate — `Run_Preflight_VM.sh` [live ops]

```bash
cd /workspace/genomic-variant-classifier
bash scripts/Run_Preflight_VM.sh <EXPECTED_HEAD>
```
Hard gate (exit 0 required): GPU+CUDA, VRAM ≥ 20 GB, `torch_geometric`+`networkx`, `imodelsx`+`KANClassifier`,
disk ≥ 150 GB, RAM ≥ 50 GB, and HEAD == `EXPECTED_HEAD`. **Any FAIL → do not launch.** (This file was
retargeted from the Run-15 launcher to `launch_run17_baseline.sh` in commit 87a04e2.)

## 3. All-models smoke gate — `smoke_all_models.py` [live ops]

```bash
python scripts/smoke_all_models.py \
  --clinvar data/processed/clinvar_grch38_clean.parquet \
  --gnomad data/processed/gnomad_v4_exomes.parquet \
  --spliceai data/external/spliceai/spliceai_index.parquet \
  --alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz \
  --seq-windows data/processed/clinvar_grch38_clean_seq.parquet \
  --gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
  --dbnsfp-path data/external/dbnsfp/dbnsfp_clinvar_index.parquet \
  --lovd-path data/external/lovd/lovd_all_variants.parquet \
  --cosmic-path data/external/cosmic/CancerMutationCensus_AllData_v104_GRCh37.tsv.gz \
  --kegg-path data/external/kegg_gene_pathways.parquet
```
**(2026-07-06) The last two flags are REQUIRED** — without them the VM smoke silent-zeros `cosmic_*`
and `kegg_*` and does NOT exercise the 97-feature matrix. Expect `Features: 97`, all 13 models, and
the three coverage lines (16c NT / 16d COSMIC / 16e KEGG).
`--max-train 3000`, `--string-db auto`, **no `--skip` flags** (full roster exercised cheaply; SVM/KAN run at
n<100k). Exit 0 required: full roster with finite AUROC, no `OOF failed`/`Traceback`/`skipping`/`DEGENERATE`,
`gnn_score` non-degenerate. **This also applies the imodelsx KAN patch to the box** (so does the launcher now,
idempotently). Any non-zero → blocked.

## 3.5 RNA-seq gene-prior ablation (settle the gene-shuffle question) — `launch_run17_rnaseq_ablation.sh` [live ops]

```bash
GNN=1 MAX_TRAIN=50000 SEEDS="11 23 37" bash scripts/launch_run17_rnaseq_ablation.sh
```
Runs `full`×1 + `drop_all`×1 + `gene_shuffle`×3 at the **full feature set** (all sources + GNN → all 97
features live), tree-focused model skips, then `aggregate_rnaseq_ablation.py` → `outputs/run17_ablation/ablation_summary.csv`
+ a printed retention verdict (non-gene-specific / gene-specific / inconclusive-if-splits-disagree).

**Cost reality (verified):** with `GNN=1`, each of the 5 runs trains the full hetero-GNN. This is **necessary,
not redundant** — `node_feat_cols` (run_phase2_eval.py:431) includes `rnaseq_*`, so `gene_shuffle` perturbs the
GNN node features and the GNN output is genuinely config-dependent; GNN=1 therefore ablates rnaseq through both
the tabular and graph paths. `--max-train` caps only the *train* base-model fits; val/test are full and the GNN
trains on the full graph regardless, so the GNN dominates cost (~5× full-GNN training). Rough order: ~1.5–2.5 h
≈ \$1–2. To trade rigor for cost: `GNN=0` answers the narrower tabular-only question for ~1/5 the cost.

SCP `ablation_summary.csv` back (tiny) and review the verdict before relying on the "redundant gene-prior"
interpretation from the small-scale run.

## 4. Full Run 17 — `launch_run17_baseline.sh` [live ops]

```bash
bash scripts/launch_run17_baseline.sh
```
Own 6-step preflight (incl. `[2b]` imodelsx KAN patch), then `run_phase2_eval.py` with the full input set,
`--skip-svm` only, `--unseen-gene-holdout`, `--hetero-gnn`, `--kg`, `--rnaseq-path`. T+45min checkpoint
sentinel; post-run artifact verify. **Checkpoint discipline:** verify a base estimator + OOF appears within
~30 min of training start; if not, ABORT and investigate rather than burning hours.

## 5. SCP back + teardown [live ops]

- SCP `outputs/run17_baseline/full/` back; verify `metrics.json`, `per_model_metrics{,_val}.csv`,
  `oof_predictions.parquet`, `feature_importance.csv`, `models/ensemble.joblib` + manifest.
- **`echo y | vastai destroy <id>`** immediately after transfer is verified (CLI ≥1.0.12 added interactive
  confirmation). Put any irreversible command in its own paste block after manual verification.

---

## Known feature-health caveats (so post-run health is read correctly)

- **ESM-2 / EVE remain stubbed** (identically 0) pending the HGVSp parser (roadmap; INCIDENT_2026-04-17).
  [RETIRED 2026-07-06: ESM-2/EVE are now active via the HGVSp parser; they are no longer expected-zero.]
- `n_pathogenic_in_gene` is recomputed train-only per fold; the standalone leakage ablation remains a separate
  scientifically-critical diagnostic (C3 permutation confirmed genuine signal, but gene-prevalence memorization
  is still worth isolating).
- Inference contract: saved base models consume **raw (unscaled) X**; any standalone inference must feed raw X
  (pre-scaling collapses the trees — blended AUROC 0.6083).

## Hand-off

Paste the `Run_Preflight_VM.sh` output here and I'll gate it (exit-0 + each PASS line) before you spend on the
smoke/ablation.
