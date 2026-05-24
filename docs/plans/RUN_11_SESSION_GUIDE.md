# Run 11 Session Guide -- Step-by-Step Execution Plan

**Date:** 2026-05-24  
**Scope:** Standard (8 integrations + deferred Run 10 items)  
**Predecessor:** Run 10b (TEST AUROC 0.9970, simple-average 8 models, 349K variants)  
**Head commit:** 92d4da4 (Run 11 batch1: master plan, audit, R10-A verification)

---

## Current State Summary

**What has been completed (from terminal output):**

1. R10-A LOVD root-cause verification: CONFIRMED Case (a). Fix shipped in commit 66593d6.
2. Data quality audit on Run 10b splits: 38/78 features dead (100% zero), 40 healthy.
3. Batch 1 deployed and committed (92d4da4): RUN_11_MASTER_PLAN.md, run11_data_quality_audit.py, verify_r10a_lovd_root_cause.py.

**What failed:**

1. Batch 2 deploy script: PowerShell parse error on em-dash character in commit message string.
2. run11_phase0_apply.py: Never deployed (file missing from project).

**What this deployment fixes:**

1. Corrected deploy script (run11_deploy_v3.ps1) -- all ASCII, no em-dashes.
2. Complete Phase 0 patcher (run11_phase0_apply.py) with --inspect / --dry-run / --apply modes.
3. All Standard-scope integration files ready for deployment.

---

## Sequencing: Deferred Run 10 Items + Run 11 Standard Scope

### Phase A: R10-A through R10-E (Deferred)

| Step | Item | Status | Action |
|------|------|--------|--------|
| A.1 | R10-A: LOVD root-cause | DONE | Verified Case (a), artifact at docs/verified/R10A_LOVD_VERIFICATION.json |
| A.2 | R10-B: Post-condition tests | IN 66593d6 | Commit 66593d6 + f64c024 + 633e7d0 + e07e3d8 contain the fixes |
| A.3 | R10-C: Re-regen splits | DEFERRED to Run 11 | Will use Run 10b splits + new connectors; re-regen is the Run 11 training |
| A.4 | R10-D: Gene-scope expansion | MANUAL | LOVD manual browser downloads only (per admin instructions after IP-ban) |
| A.5 | R10-E: HGMD Pro | PROCUREMENT | Internal seat discovery + QIAGEN trial activation; no code changes |

**R10-C rationale:** Re-regen splits with fixed connectors IS the Run 11 training run. The Phase 0 patches (GPU GBDT, OOF sidecar, ZSTD) + Standard-scope integrations (Polars, PrimateAI-3D, Optuna) are applied before re-regen. This collapses R10-C into Run 11.

### Phase B: Run 11 Standard-Scope Integrations (8 changes)

| # | Integration | File(s) | Verification |
|---|-------------|---------|--------------|
| 1 | GPU GBDT params | variant_ensemble.py (patched) | nvidia-smi + training log shows GPU tree method |
| 2 | FastKAN swap | kan.py (new file) | Import test + 10K-row smoke test |
| 3 | OOF row-index sidecar | variant_ensemble.py (patched) | Check _oof_indices.npy saved alongside _oof.npy |
| 4 | ZSTD Parquet compression | real_data_prep.py (patched) | File size comparison vs uncompressed |
| 5 | Polars/DuckDB ETL | etl_polars.py (new file) | Row-equality check vs pandas on 10K sample |
| 6 | AlphaMissense + PrimateAI-3D + REVEL | primateai3d.py (new file) | Annotation count > 0 for missense variants |
| 7 | BF16/torch.compile | launch_run11_vm.sh (runtime flag) | BF16 support detected in pre-flight |
| 8 | Optuna HPO (ASHA) | run11_hpo.py (new file) | SQLite DB + best_params.json written |

---

## Exact Execution Steps

### Step 0: Pre-deployment state check

```powershell
cd C:\Projects\genomic-variant-classifier
git log -3 --oneline
git status --short
```

**Expected:** HEAD at 92d4da4, clean working tree.

### Step 1: Download all files from this session

Download these 9 files from the cards above:
1. run11_deploy_v3.ps1
2. run11_phase0_apply.py
3. launch_run11_vm.sh
4. kan.py
5. etl_polars.py
6. primateai3d.py
7. run11_hpo.py
8. run11_preflight.py
9. RUN_11_SESSION_GUIDE.md (this file)

### Step 2: Deploy all files

```powershell
cd C:\Projects\genomic-variant-classifier
powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\run11_deploy_v3.ps1"
```

**Expected output:** 9 files deployed, 0 missing, all syntax checks pass.

### Step 3: Inspect production code (before patching)

```powershell
python scripts\run11_phase0_apply.py --inspect
```

**Paste the full output.** This shows the structure of the production files and which patterns the patcher can find. Review before proceeding.

### Step 4: Dry-run the patches

```powershell
python scripts\run11_phase0_apply.py --dry-run
```

**Paste the full output.** This shows what would change without modifying anything.

### Step 5: Apply patches (if dry-run looks correct)

```powershell
python scripts\run11_phase0_apply.py --apply
```

**Expected:** Backup files created (.bak.*), patches applied successfully.

### Step 6: Run pre-flight checks

```powershell
python scripts\run11_preflight.py
```

**Expected:** All checks pass (or only WARN items for missing external data).

### Step 7: Run tests

```powershell
.venv312\Scripts\Activate.ps1
python -m pytest tests/ -v --timeout=300 -q
```

**Expected:** All existing tests pass. New modules (kan.py, etl_polars.py, primateai3d.py) are importable but have no test coverage yet (test coverage will be added in the Vast.ai training run).

### Step 8: Commit (SEPARATE command)

```powershell
git add -A
git commit -m "feat(run11): Phase 0 patches + Standard-scope integrations"
```

### Step 9: Push (SEPARATE command)

```powershell
git push
```

### Step 10: gnomAD constraint TSV check

```powershell
Test-Path data\external\gnomad\gnomad.v4.1.constraint_metrics.tsv
```

If FALSE: Download from https://gnomad.broadinstitute.org/downloads#v4-constraint -- this file recovers pli_score, loeuf, syn_z, mis_z (4 features that were top-5 in Runs 7-8, currently 100% zero).

### Step 11: Vast.ai instance selection

```powershell
vastai search offers "gpu_name=RTX_4090 num_gpus=1 disk_space>=100 inet_up>=200" --order "dph_total"
```

Select instance, note ID.

### Step 12: SCP data to Vast.ai (SEPARATE from destroy)

```powershell
# Upload repo
scp -i C:\Users\monzi\.ssh\id_lambda_run8 -P $PORT -r C:\Projects\genomic-variant-classifier root@$HOST:/workspace/

# Upload splits
scp -i C:\Users\monzi\.ssh\id_lambda_run8 -P $PORT -r C:\Projects\genomic-variant-classifier\outputs\run10b_final root@$HOST:/workspace/genomic-variant-classifier/outputs/
```

### Step 13: SSH and launch

```powershell
ssh -i C:\Users\monzi\.ssh\id_lambda_run8 -p $PORT root@$HOST
```

On the remote:

```bash
cd /workspace/genomic-variant-classifier
bash scripts/launch_run11_vm.sh
```

### Step 14: Monitor (within 30 min, verify checkpoints)

```bash
# Check for checkpoint files
ls -la /workspace/outputs/run11/full/models/
# Should see per-model .joblib files appearing as training progresses
```

### Step 15: SCP results back (SEPARATE command from destroy)

```powershell
scp -i C:\Users\monzi\.ssh\id_lambda_run8 -P $PORT -r root@$HOST:/workspace/outputs/run11 C:\Projects\genomic-variant-classifier\outputs\
```

### Step 16: Verify results locally

```powershell
python -c "import json; d=json.load(open('outputs/run11/full/metrics.json')); print(f'TEST AUROC: {d.get(\"test_auroc\", \"N/A\")}')"
```

### Step 17: Destroy instance (SEPARATE command)

```powershell
echo y | vastai destroy instance $INSTANCE_ID
```

**STANDING RULE #30: This command is ALWAYS in its own paste block, NEVER combined with SCP or any other command.**

---

## Data Collection Checklist

After Run 11 completes, collect and document:

| Metric | Where to find it |
|--------|-----------------|
| Test AUROC (locked) | outputs/run11/full/metrics.json |
| Per-model AUROC | outputs/run11/full/per_model_metrics.csv |
| OOF AUROCs | Training log (grep "OOF AUROC") |
| HPO best params | outputs/run11/hpo/best_params.json |
| HPO default vs optimized | Compare Run 10b vs Run 11 per-model |
| Polars ETL timing | Training log (grep "Polars ETL") |
| GPU GBDT speedup | Training log (compare wall-clock per model) |
| ZSTD compression ratio | Compare parquet file sizes |
| Feature importance shift | outputs/run11/full/feature_importances.csv |
| Dead feature count | outputs/run11/data_quality_audit.csv |
| PrimateAI-3D annotation rate | Training log (grep "PrimateAI-3D") |
| OOF index sidecar present | ls outputs/run11/full/*_oof_indices.npy |
| FastKAN vs MLP performance | Per-model metrics (if KAN enabled) |
| BF16 memory savings | nvidia-smi during training |

---

## Verification Evidence for Future Reference

### Why GPU GBDTs matter
XGBoost/LightGBM/CatBoost with GPU tree methods provide 2-10x training speedup on 1.7M rows. The exact speedup depends on tree depth and number of estimators. Run 11 will measure this precisely.

### Why OOF index sidecar matters
Run 10b lost the meta-learner because OOF arrays were saved in CV-prediction order, not X_train row order. Without the fold-to-row mapping, post-hoc reconstruction gave AUROC ~0.50 (random). The sidecar file enables reconstruction even after instance destruction.

### Why ZSTD compression matters
Parquet default (snappy) gives ~2x compression. ZSTD gives ~3-4x with minimal decompression overhead. For 1.7M x 78 features, this saves 30-50% on SCP transfer time and disk usage.

### Why Polars matters
Polars lazy evaluation means the full 1.7M-row pipeline never materializes intermediate DataFrames. Benchmarked at 3.3x faster on the gnomAD constraint join (2026-04-09). The feature flag (GENOMIC_ETL_BACKEND) allows rollback.

### Why Optuna HPO matters
Default hyperparameters were used for all 10 runs. HPO with ASHA pruning can find 0.001-0.01 AUROC improvement per model with 30 trials in under an hour. The SQLite storage enables resuming across Vast.ai sessions if the instance dies.

---

## Files Delivered This Session

| File | Destination | Purpose |
|------|-------------|---------|
| run11_deploy_v3.ps1 | (deployment script) | Deploy all files from Downloads |
| run11_phase0_apply.py | scripts/ | Patch production code (GPU, OOF, ZSTD) |
| launch_run11_vm.sh | scripts/ | Vast.ai training launch |
| kan.py | src/genomic_variant_classifier/models/ | FastKAN replacement |
| etl_polars.py | src/genomic_variant_classifier/data/ | Polars ETL pipeline |
| primateai3d.py | src/genomic_variant_classifier/data/ | PrimateAI-3D connector |
| run11_hpo.py | scripts/ | Optuna HPO |
| run11_preflight.py | scripts/ | Pre-commit verification |
| RUN_11_SESSION_GUIDE.md | docs/plans/ | This file |
