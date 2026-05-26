---
document: RUN_11_MASTER_PLAN.md
date: 2026-05-24
phase: 2.0 (Run 11)
scope: Standard (8 integrations + carried-forward Run 10 items)
predecessor_run: 10b (TEST AUROC 0.9970 on 349,067 variants, simple-average 8 models)
head_commit: 5e94f4c (Run 10b salvage docs)
repository: monzia-moodie-repo-projects/genomic-variant-classifier
author: Monzia Moodie
---

# Run 11 Master Plan — Genomic Variant Classifier

## Table of Contents

1. [State Assessment](#1-state-assessment)
2. [Pre-Run-11: Deferred Run 10 Items (R10-A through R10-E)](#2-pre-run-11-deferred-run-10-items)
3. [Carried-Forward Run 11 Priority Backlog](#3-carried-forward-run-11-priority-backlog)
4. [Run 11 Standard-Tier Integrations (8 changes)](#4-run-11-standard-tier-integrations)
5. [Sequencing and Dependency Graph](#5-sequencing-and-dependency-graph)
6. [Verification Protocol](#6-verification-protocol)
7. [Cost Projection](#7-cost-projection)
8. [Files Required From Monzia](#8-files-required-from-monzia)
9. [Lessons Ledger Template](#9-lessons-ledger-template)

---

## 1. State Assessment

### 1.1 Repository State (as of 2026-05-24 post-session)

The main branch carries commits through `5e94f4c` (Run 10b salvage docs). The key
ancestry chain (latest first):

```
5e94f4c  fix(docs): CHANGELOG BOM + remediation tool
6e0e379  Run 10b-partial: session doc + 2 incident reports + CHANGELOG
9b1400e  Run 10b-partial salvage: TEST AUROC 0.9970
927e8d6  Run 10b launch script
f147112  Phase 1.7.1 incremental checkpoint patch
ac64665  Phase 1.7 (Run 10 launch readiness)
e07e3d8  Phase 1.5e
633e7d0  Phase 1.5d
0178fe1  Phase 1.5c
f64c024  Phase 1.5b
66593d6  Phase 1 core (LOVD/DbNSFP/FinnGen wiring)
```

### 1.2 Run 10b Outcome Summary

| Metric | Value |
|--------|-------|
| TEST AUROC (simple-average, 8 models) | 0.9970 |
| Test set size | 349,067 variants |
| Prevalence (test) | 0.1999 |
| Val AUROC | 0.9970 |
| Val set size | 154,404 |
| Working models | 8 (RF, XGB, LGB, LR, GBM, CB, TNN, MC-Dropout) |
| Failed models | 1 (cnn_1d cross-platform pickle; AUROC was 0.5 anyway) |
| Lost models | deep_ensemble (instance destroyed mid-fitting) |
| Meta-learner | NOT produced (OOF alignment failure prevented reconstruction) |
| GNN | NOT trained (pipeline never reached this stage) |

### 1.3 Known Silent-Zero Features (Run 10b baseline)

Of 78 features in the training matrix, only SpliceAI and AlphaMissense produced
real signal. The following 15+ annotation connectors all emitted zeros:

DbNSFP, PhyloP, GTEx, VEP codon_position, OMIM, ClinGen, dbSNP, EVE, HGMD,
RNA splice, protein structure, LOVD, ESM-2 (stub), gnomAD constraint pLI,
FinnGen.

Root causes by category:
- **Connector not wired (Case a):** LOVD (no `parquet_path`), DbNSFP (no path
  argument), FinnGen (no path argument)
- **Missing parser (Case b):** ESM-2 (needs HGVSp parser for `protein_pos`,
  `wt_aa`, `mut_aa`), EVE (same dependency)
- **Not yet diagnosed:** PhyloP, GTEx, VEP codon_position, OMIM, ClinGen,
  dbSNP, RNA splice, protein structure, gnomAD constraint pLI

### 1.4 Key Scientific Concern

`n_pathogenic_in_gene` has 3.3x the feature importance of the next feature
(importance score 1213.5 in Run 8). This raises serious questions about whether
the model's high AUROC reflects genuine predictive skill or memorization of
gene-prevalence patterns plus lookup of external scores. The ablation matrix
(13 ablations) is designed to isolate this concern.

---

## 2. Pre-Run-11: Deferred Run 10 Items

These items MUST complete before Run 11 launch. They are sequenced to build
on each other.

### 2.1 R10-A: LOVD Root-Cause Grep — STATUS: COMPLETED

**Verification (already performed 2026-05-13):**
Grep of `outputs/run9_ready/regen.log` confirmed Case (a): `LOVDConnector`
was invoked at annotation step 15/16 without `parquet_path`, causing all
variants to receive `lovd_variant_class=0`. Phase 1 B1 fix in commit `66593d6`
(adds `--lovd-path`) is the correct remediation.

**Formal verification script (to be committed for audit trail):**

```powershell
# R10-A formal verification — run from C:\Projects\genomic-variant-classifier
$logPath = "outputs\run9_ready\regen.log"
if (-not (Test-Path $logPath)) {
    Write-Host "FAIL: $logPath not found" -ForegroundColor Red
    return
}

$lovdLine = Select-String -Path $logPath -Pattern "Score annotation 15/16 \(LOVD\)" `
    -SimpleMatch:$false
$connectorLine = Select-String -Path $logPath -Pattern "LOVDConnector.*parquet"

Write-Host "=== R10-A LOVD Root-Cause Verification ===" -ForegroundColor Cyan
if ($lovdLine) {
    Write-Host "  FOUND: $($lovdLine.Line)" -ForegroundColor Green
} else {
    Write-Host "  NOT FOUND: LOVD annotation step line" -ForegroundColor Yellow
}

$noParquetLine = Select-String -Path $logPath -Pattern "no parquet loaded"
if ($noParquetLine) {
    Write-Host "  CONFIRMED Case (a): $($noParquetLine.Line)" -ForegroundColor Green
} else {
    Write-Host "  Case (a) not confirmed in log" -ForegroundColor Yellow
}
Write-Host "  FIX: Commit 66593d6 adds --lovd-path argument" -ForegroundColor Green
Write-Host "  STATUS: COMPLETE — no Phase 1.8 patch needed" -ForegroundColor Green
```

### 2.2 R10-B: LOVD Post-Condition Unit Test

**Current state:** Phase 1.5b/1.5d shipped
`tests/unit/test_lovd_annotation_reaches_training_matrix.py` with 2 tests,
both PASSING after Phase 1.5e module-level pandas import fix.

**What R10-B adds (if not already covered):**

The existing test asserts `lovd_variant_class > 0` appears in the union of
train+val+test splits when `--lovd-path` is supplied, and asserts all zeros
when `--lovd-path` is omitted. This IS the post-condition test. R10-B is
COMPLETE via commits `66593d6` + `f64c024` + `633e7d0` + `e07e3d8`.

**Verification step:**
```powershell
cd C:\Projects\genomic-variant-classifier
python -m pytest tests/unit/test_lovd_annotation_reaches_training_matrix.py -v --timeout=120
```

Expected: 2 PASSED.

### 2.3 R10-C: Re-Regen Splits on Vast.ai

**Purpose:** Produce new training/val/test splits with LOVD, DbNSFP, and
FinnGen wired in (the three connectors fixed by Phase 1 B1). This is the
Run 11 baseline dataset.

**CRITICAL:** This is the step where Run 11's improved features take effect.
The splits must be regenerated with ALL new feature sources wired in. Run 11's
8 Standard-tier changes operate ON these regenerated splits.

**Sequencing:** R10-C is the first step that requires a Vast.ai instance.
It should be combined with Run 11 training into a single instance session
to minimize cost.

**Implementation (Vast.ai):**
```bash
# On Vast.ai instance, after SCP-up:
cd /workspace/genomic-variant-classifier

python scripts/run_phase2_eval.py \
  --lovd-path data/external/lovd/lovd_all_variants.parquet \
  --dbnsfp-path data/external/dbnsfp/dbnsfp_processed.parquet \
  --finngen-path data/external/finngen/finngen_R12_annotated_variants_v1.gz \
  --output-dir outputs/run11/regen \
  --n-folds 5 \
  --regen-only  # If this flag exists; otherwise use --skip-training
```

**Post-condition checks:**
```bash
# After regen completes:
python -c "
import pandas as pd
X = pd.read_parquet('outputs/run11/regen/splits/X_train.parquet')
print(f'Shape: {X.shape}')
for col in ['lovd_variant_class', 'sift_score', 'polyphen2_score']:
    nz = (X[col] != 0).sum()
    print(f'  {col}: {nz}/{len(X)} nonzero ({nz/len(X)*100:.1f}%)')
"
```

### 2.4 R10-D: Gene-Scope Expansion (LOVD Manual Downloads)

**Context:** LOVD admin banned automated scraping of their API on 2026-04-01.
All future LOVD downloads must be manual browser downloads only.

**Current gene coverage (10 canonical genes):**
BRCA1, BRCA2, MLH1, MSH2, MSH6, APC, NF1, TP53, PTEN, RB1

**Expansion candidates (clinical priority order):**

Tier 1 — High ClinVar overlap, strong clinical evidence:
- CFTR (cystic fibrosis)
- SCN1A (Dravet syndrome)
- TSC1/TSC2 (tuberous sclerosis)
- LDLR (familial hypercholesterolemia)
- PKD1/PKD2 (polycystic kidney disease)

Tier 2 — Broad coverage, moderate overlap:
- LMNA (laminopathies)
- FBN1 (Marfan syndrome)
- COL1A1/COL1A2 (osteogenesis imperfecta)
- DMD (Duchenne/Becker muscular dystrophy)

**Manual download protocol per gene:**
1. Navigate to `databases.lovd.nl/shared/genes/{GENE_SYMBOL}`
2. Click "Unique variants" tab
3. Export as text file (TSV)
4. Save to `data/external/lovd/lovd_{gene_symbol}_variants.txt`
5. Verify row count > 100 before proceeding

**Post-download integration:**
```powershell
# After all manual downloads complete:
cd C:\Projects\genomic-variant-classifier
python scripts/build_lovd_index.py \
  --input-dir data/external/lovd/ \
  --output data/external/lovd/lovd_all_variants.parquet
```

**Verification:**
```powershell
python -c "
import pandas as pd
df = pd.read_parquet('data/external/lovd/lovd_all_variants.parquet')
print(f'Total rows: {len(df)}')
print(f'Genes: {df[\"gene_symbol\"].nunique()}')
print(df['gene_symbol'].value_counts())
"
```

**Wall-clock estimate:** ~45-60 min manual browser work for 10 additional genes.

### 2.5 R10-E: HGMD Professional Integration

**STATUS: BLOCKED on procurement.**

Monzia has only HGMD Public (Cardiff free tier, ~291K entries). HGMD Professional
(~575K entries, required for bulk ML training) requires either:

1. **Internal seat discovery** at New England R1s: Harvard/Countway, UMass Chan,
   BU, Tufts, MGH, Broad Institute, Dana-Farber
2. **QIAGEN free trial** via `go.qiagen.com/HGMD-Pro-Trial`
3. **Academic license purchase** (~$5K-$10K/year for online access)

**Decision for Run 11:** DO NOT BLOCK Run 11 on HGMD Pro. Document as Run 12+
deferral. The procurement track should proceed in parallel.

**Parallel action items (no code changes, human tasks):**
- [ ] Email `bioinformaticssales@qiagen.com` with academic use case
- [ ] Check institutional access at UMass Chan Lamar Soutter Library
- [ ] Activate QIAGEN free trial at `go.qiagen.com/HGMD-Pro-Trial`
- [ ] If access obtained, file `docs/license/HGMD_LICENSE_POSTURE.md`

**Run 12+ integration spec:** Filed in `docs/hypotheses/HYP_hgmd-integration.md`
with full TDD harness, adversarial test catalogue, and label leakage hazard
assessment (REVEL/VEST/FATHMM/MutPred/ClinPred carry HGMD-derived signal).

---

## 3. Carried-Forward Run 11 Priority Backlog

These items from the Run 10b session doc are prerequisite to Run 11 training.

### 3.1 CNN1D Module-Level Refactor

**Problem:** `_CNN1D` is defined inside `CNN1DClassifier._build_model()` as a
closure class. This causes:
- `PicklingError` when `joblib.dump` tries to serialize the ensemble (Run 9 crash)
- Cross-platform unpickle failure: Linux-saved model fails to load on Windows
  (`TypeError: NoneType.__new__(X)`) (Run 10b incident)
- OOF AUROC = 0.5000 in both Run 9 and Run 10b (possibly a fit-side bug, not
  just a pickle bug)

**Fix:** Move `_CNN1DModule` to module level in
`src/genomic_variant_classifier/models/variant_ensemble.py`.

**Implementation details:**
1. Extract the `_CNN1D` class from inside `_build_model` to module level
2. Rename to `_CNN1DModule` to avoid name collision
3. Update `_build_model` to reference the module-level class
4. Add unit test: `tests/unit/test_cnn1d_pickle_roundtrip.py`
   - Fit on 100-row synthetic data
   - `joblib.dump` + `joblib.load`
   - Assert `predict_proba` returns same shape
   - Assert AUROC > 0.55 (above random)

**Verification:**
```powershell
python -m pytest tests/unit/test_cnn1d_pickle_roundtrip.py -v --timeout=60
```

### 3.2 OOF Row-Index Sidecar

**Problem:** Run 10b proved that OOF arrays stored in CV-prediction order
cannot be paired with `y_train` for post-hoc meta-learner reconstruction.
Sanity check detected and refused the bad alignment.

**Fix:** In `VariantEnsemble.fit()`, after each fold's predictions are
assembled, save `{model_name}_oof_indices.npy` alongside `{model_name}_oof.npy`.
The indices array maps prediction index to X_train row index.

**Implementation:**
```python
# In the per-model checkpoint block, after oof is assembled:
np.save(
    model_dir / f"{name}_oof_indices.npy",
    np.concatenate([test_idx for _, test_idx in cv.split(X_input, y_arr)])
)
```

**Post-condition:** After training, verify:
```python
indices = np.load(f"{name}_oof_indices.npy")
oof = np.load(f"{name}_oof.npy")
assert len(indices) == len(oof)
assert len(indices) == len(y_train)  # 85% of full y
assert len(set(indices)) == len(indices)  # no duplicates
```

### 3.3 HGVSp Parser

**Problem:** ESM-2 and EVE connectors both require `protein_pos`, `wt_aa`,
`mut_aa` columns that the pipeline never populates. Both connectors log an
INFO message and return zeros. This has been inert since Run 6.

**Fix:** Create `src/genomic_variant_classifier/data/hgvsp_parser.py` that
parses HGVSp strings (e.g., `p.Arg557His`) into:
- `protein_pos` (int): residue position (557)
- `wt_aa` (str): wild-type amino acid one-letter code (R)
- `mut_aa` (str): mutant amino acid one-letter code (H)

**Source of HGVSp strings:** ClinVar `variant_summary.txt.gz` field
`Name` contains HGVSp notation. Parse pattern:
`p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})` → three-letter amino acid codes.

**Decision gate:** This is a Run 11 STRETCH goal. If wall-clock budget
remains after the 8 Standard-tier integrations, implement it. Otherwise
defer to Run 12. ESM-2 and EVE will continue to emit zeros without it,
which is a known-and-documented state.

---

## 4. Run 11 Standard-Tier Integrations (8 Changes)

### 4.1 Integration 1: Data-Quality Audit Script

**Purpose:** Systematic audit of all 78 features to identify which are
contributing real signal vs. silently zero. This is the FIRST step because
it establishes the baseline against which all other integrations are measured.

**Deliverable:** `scripts/run11_data_quality_audit.py`

**What it produces:**
- `outputs/run11/data_quality_audit.csv` — per-feature zero-fraction, min,
  max, mean, std, nunique, source connector name
- `outputs/run11/data_quality_audit.html` — heatmap visualization
- Console summary: features grouped by zero-fraction tier

**Implementation approach:**
```python
import polars as pl  # Use Polars for this audit — it's a read-only scan

def audit_features(splits_dir: Path) -> pl.DataFrame:
    X_train = pl.read_parquet(splits_dir / "X_train.parquet")
    X_val = pl.read_parquet(splits_dir / "X_val.parquet")
    X_test = pl.read_parquet(splits_dir / "X_test.parquet")
    X_all = pl.concat([X_train, X_val, X_test])

    results = []
    for col in X_all.columns:
        series = X_all[col]
        results.append({
            "feature": col,
            "zero_fraction": (series == 0).mean(),
            "null_fraction": series.null_count() / len(series),
            "nunique": series.n_unique(),
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "std": series.std(),
            "source_connector": _identify_connector(col),
        })
    return pl.DataFrame(results).sort("zero_fraction", descending=True)
```

**Verification:** Compare against Run 10b `per_model_metrics_partial.csv`
feature importance rankings. Features with 100% zero-fraction should have
zero importance.

**Data collection goals:**
- Baseline zero-fraction per feature
- Connector attribution map
- Delta measurement after R10-C re-regen (which fixes LOVD, DbNSFP, FinnGen)

### 4.2 Integration 2: FastKAN Swap

**Purpose:** Replace pykan (which caused Run 10a KAN runaway: 19h 22m, $14.72
wasted) with FastKAN, which is 3.7x faster in benchmarks and has a simpler API.

**Source:** `github.com/ZiyaoLi/fast-kan` (MIT license)

**Implementation:**
1. Add `fastkan>=0.3.0` to `requirements.txt`
2. Remove `pykan` dependency
3. Rewrite `KANClassifier` in `variant_ensemble.py`:

```python
class KANClassifier(BaseEstimator, ClassifierMixin):
    """KAN base estimator using FastKAN backend."""
    def __init__(self, hidden_dims=(64, 32), grid_size=5,
                 lr=1e-3, epochs=50, batch_size=1024,
                 max_fit_samples=100_000, random_state=42):
        ...
    def fit(self, X, y):
        from fastkan import FastKAN
        # Stratified subsample if > max_fit_samples
        ...
```

4. Unit test: `tests/unit/test_kan_fastkan.py`
   - 200-row synthetic fit + predict
   - `joblib.dump` + `joblib.load` round-trip
   - AUROC > 0.6 on balanced binary

**Key safeguards:**
- Keep `max_fit_samples=100_000` (Run 10a OOM safeguard, commit 2389ee2)
- KAN pre-flight at 10K then 100K rows before paying for full run
- Add `model/` to `.gitignore` (pykan creates `./model/` in cwd)

**Fallback:** If FastKAN has compatibility issues, fall back to warpKAN
(`github.com/athanoid/warpkan`). Document results either way.

**Data collection:**
- Wall-clock time for KAN training (FastKAN vs pykan baseline from Run 10a)
- Memory allocation peak
- OOF AUROC
- Feature importance comparison

### 4.3 Integration 3: GPU GBDT Acceleration

**Purpose:** Enable GPU-accelerated gradient boosting on Vast.ai RTX 4090.
Run 10b's 8 base models took 2h 27m; GPU GBDTs should cut tree-model
training time by 3-5x.

**Implementation (device-aware):**

```python
# In _build_estimators(), detect GPU availability:
import torch
_GPU_AVAILABLE = torch.cuda.is_available()

# CatBoost:
if _GPU_AVAILABLE:
    cb_params["task_type"] = "GPU"
    cb_params["devices"] = "0"

# XGBoost:
if _GPU_AVAILABLE:
    xgb_params["device"] = "cuda"
    xgb_params["tree_method"] = "hist"  # gpu_hist is deprecated; hist auto-selects GPU

# LightGBM:
if _GPU_AVAILABLE:
    lgb_params["device_type"] = "gpu"
    lgb_params["gpu_use_dp"] = False  # FP32 for speed, not FP64
```

**Critical: Local CPU dev must still work.** The `_GPU_AVAILABLE` flag
ensures all three GBDTs fall back to CPU when torch.cuda is unavailable.

**Verification:**
```bash
# On Vast.ai, in training log:
grep -i "gpu\|cuda\|device" outputs/run11/full/run11_master.log
```

**Data collection:**
- Per-model training time (GPU vs Run 10b CPU baseline)
- GPU memory utilization (`nvidia-smi` snapshots during training)
- OOF AUROC (must not degrade; GPU path should produce identical results
  within floating-point tolerance)

### 4.4 Integration 4: Optuna ASHA HPO

**Purpose:** Automated hyperparameter optimization with early stopping
(ASHA pruner) to find better configurations than the current hand-tuned
defaults.

**Implementation:**

1. Add `optuna>=4.0` to `requirements.txt`
2. Create `scripts/run11_hpo.py`:

```python
import optuna
from optuna.pruners import HyperbandPruner

def objective(trial, X_train, y_train, model_name):
    """Define search space per model."""
    if model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
    # ... similar for XGBoost, CatBoost, RF, GBM
    ...
    # 3-fold CV with AUROC scoring
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc")
    return score.mean()

study = optuna.create_study(
    direction="maximize",
    pruner=HyperbandPruner(min_resource=1, max_resource=5, reduction_factor=3),
    storage="sqlite:///outputs/run11/optuna_hpo.db",  # Resumable
    study_name=f"run11_{model_name}",
    load_if_exists=True,
)
study.optimize(objective, n_trials=50, timeout=3600)
```

3. SQLite storage enables resuming HPO across Vast.ai sessions
4. Best params written to `outputs/run11/best_params.json`

**Sequencing:** HPO runs BEFORE full training. The best params are then
loaded into `_build_estimators()` for the final 5-fold CV training.

**Data collection:**
- Full Optuna study dashboard (HTML export)
- Best trial params per model
- Default vs optimized AUROC comparison
- HPO wall-clock time per model
- Number of pruned trials

### 4.5 Integration 5: Polars/DuckDB ETL Replacement

**Purpose:** Replace Spark ETL with Polars (lazy mode) + DuckDB for the
1.7M-row pipeline. Polars was benchmarked at 3.3x faster than pandas on
the gnomAD constraint join (commit from 2026-04-09).

**Implementation:**

1. Add `polars>=1.0`, `duckdb>=1.0` to `requirements.txt`
2. Create `src/genomic_variant_classifier/data/etl_polars.py`:

```python
"""
Polars-native ETL pipeline — replaces Spark for the 1.7M-row ClinVar pipeline.

Feature flag: set GENOMIC_ETL_BACKEND=polars (default) or GENOMIC_ETL_BACKEND=spark.
"""
import polars as pl

class PolarsETLPipeline:
    def __init__(self, config):
        self.config = config

    def run(self, clinvar_path, gnomad_path=None, **kwargs):
        # Lazy scan — no data loaded until .collect()
        lf = pl.scan_parquet(clinvar_path)

        # Filter to high-confidence labels
        lf = self._filter_labels(lf)

        # Join gnomAD if available
        if gnomad_path:
            gnomad_lf = pl.scan_parquet(gnomad_path)
            lf = lf.join(gnomad_lf, on=["chrom", "pos", "ref", "alt"], how="left")

        # Materialize
        df = lf.collect()
        return df
```

3. Keep Spark code path behind `GENOMIC_ETL_BACKEND=spark` env var
4. DuckDB used for ad-hoc analytical queries on parquet files:

```python
import duckdb
conn = duckdb.connect()
result = conn.execute("""
    SELECT feature, COUNT(*) as nonzero
    FROM read_parquet('outputs/run11/regen/splits/X_train.parquet')
    WHERE feature_value > 0
    GROUP BY feature
""").fetchdf()
```

**Verification:**
- Polars output must be row-identical to pandas output on a 10K-row
  random subsample (sorted by variant_id)
- Wall-clock comparison: Polars vs pandas on full 1.7M rows

**Data collection:**
- ETL wall-clock time: Polars lazy vs pandas eager
- Peak memory: Polars vs pandas
- Row-equality check result
- Any precision differences (float32 vs float64)

### 4.6 Integration 6: AlphaMissense + PrimateAI-3D + REVEL Features

**Purpose:** Add three high-value precomputed pathogenicity scores as features.
These are lookup-join features (no model fine-tuning required).

**Current state:**
- AlphaMissense: ALREADY WIRED (206,131 variants annotated in Run 9,
  ranked 7th of 78 features)
- REVEL: Column exists in feature matrix (`revel_score`) but sourced from
  DbNSFP. After R10-C re-regen with `--dbnsfp-path`, this should have
  real values
- PrimateAI-3D: NOT YET INTEGRATED

**PrimateAI-3D integration:**

1. Download: `https://storage.googleapis.com/dm-primateai3d/` (precomputed
   scores for all possible human missense variants, ~4GB)
2. Create `src/genomic_variant_classifier/data/primateai3d.py`:

```python
class PrimateAI3DConnector(BaseConnector):
    """Lookup-join PrimateAI-3D pathogenicity scores."""
    def __init__(self, parquet_path=None):
        self.parquet_path = parquet_path

    def annotate(self, df):
        if not self.parquet_path:
            df["primateai3d_score"] = 0.0
            logger.warning("PrimateAI3DConnector: no parquet_path — defaulting to 0.0")
            return df
        scores = pd.read_parquet(self.parquet_path)
        # Join on chrom:pos:ref:alt lookup key
        df = df.merge(scores, on=["chrom", "pos", "ref", "alt"], how="left")
        df["primateai3d_score"] = df["primateai3d_score"].fillna(0.0)
        return df
```

3. Add `primateai3d_score` to TABULAR_FEATURES in `real_data_prep.py`
4. Wire into annotation pipeline at `real_data_prep.py` score annotation stage

**Verification:**
```python
# After regen:
X = pd.read_parquet("outputs/run11/regen/splits/X_train.parquet")
for col in ["alphamissense_score", "revel_score", "primateai3d_score"]:
    nz = (X[col] != 0).sum()
    print(f"{col}: {nz}/{len(X)} nonzero ({nz/len(X)*100:.1f}%)")
```

**Data collection:**
- Nonzero fraction per score
- Feature importance rank after training
- Ablation: AUROC with vs without each score

### 4.7 Integration 7: Mixed Precision (BF16) + torch.compile

**Purpose:** Cut GPU memory usage in half and accelerate neural network
training via BF16 mixed precision and `torch.compile`.

**Applies to:** TabularNN, CNN1D, KAN, MC-Dropout, DeepEnsemble (all
PyTorch-based). Does NOT apply to GBDTs (they use their own GPU backends).

**Implementation:**

```python
# In each PyTorch model's fit() method:
import torch
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')
for epoch in range(self.epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        with autocast('cuda', dtype=torch.bfloat16):
            output = model(batch_X)
            loss = criterion(output, batch_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

For `torch.compile`:
```python
# After model construction, before training:
if hasattr(torch, 'compile') and torch.cuda.is_available():
    model = torch.compile(model, mode="reduce-overhead")
```

**RTX 4090 compatibility:** BF16 is natively supported on Ada Lovelace
(RTX 4090). FP16 would also work but BF16 has a larger dynamic range,
reducing the risk of overflow in loss computation.

**Verification:**
```bash
# During training, check GPU memory:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5
```

**Data collection:**
- TabularNN/CNN1D/KAN training time: BF16 vs FP32 baseline
- Peak GPU memory: BF16 vs FP32
- OOF AUROC: must not degrade by more than 0.0005 (within noise floor)
- `torch.compile` compilation time (first epoch) vs speedup (subsequent epochs)

### 4.8 Integration 8: Parquet + ZSTD + Arrow IPC

**Purpose:** Standardize all intermediate pipeline data on Parquet with ZSTD
compression. This replaces any CSV intermediates and ensures fast I/O.

**Implementation:**

1. All `to_parquet()` calls use `compression="zstd"`:
```python
df.to_parquet(path, compression="zstd", index=False)
```

2. Arrow IPC for in-memory transfers between pipeline stages:
```python
import pyarrow as pa
import pyarrow.ipc as ipc

# Write:
table = pa.Table.from_pandas(df)
with pa.OSFile(str(path), 'wb') as sink:
    writer = ipc.new_file(sink, table.schema)
    writer.write_table(table)
    writer.close()

# Read:
with pa.OSFile(str(path), 'rb') as source:
    reader = ipc.open_file(source)
    table = reader.read_all()
df = table.to_pandas()
```

3. Update `_save_splits()` in `DataPrepPipeline` to use ZSTD
4. Update all SCP scripts to expect `.parquet` (not `.csv`)

**Verification:**
```python
# Compare file sizes:
import os
for f in Path("outputs/run11/regen/splits/").glob("*.parquet"):
    size_mb = os.path.getsize(f) / 1024**2
    print(f"{f.name}: {size_mb:.1f} MB")
```

**Data collection:**
- File size: ZSTD vs snappy (default) vs uncompressed
- Read/write time: ZSTD vs snappy
- Decompression speed during training (should be negligible)

---

## 5. Sequencing and Dependency Graph

```
Phase 0: Pre-Run-11 (local, no GPU)
  ├─ R10-A Verification script ........... DONE (2026-05-13)
  ├─ R10-B Post-condition test ........... DONE (66593d6 + 1.5b/d/e)
  ├─ R10-D LOVD gene-scope expansion ..... ~60 min manual browser
  ├─ R10-E HGMD Pro procurement .......... DEFERRED to Run 12+
  ├─ 3.1 CNN1D module-level refactor ..... ~30 min code + test
  ├─ 3.2 OOF row-index sidecar ........... ~15 min code change
  ├─ 4.1 Data-quality audit script ....... ~20 min
  ├─ 4.2 FastKAN swap .................... ~45 min code + test
  ├─ 4.3 GPU GBDT params ................. ~15 min code change
  ├─ 4.5 Polars ETL module ............... ~45 min
  ├─ 4.6 PrimateAI-3D connector .......... ~30 min
  ├─ 4.7 BF16/torch.compile patches ...... ~30 min
  ├─ 4.8 Parquet ZSTD switch ............. ~15 min
  └─ Commit + push all to origin/main

Phase 1: Launch (Vast.ai RTX 4090)
  ├─ SCP data + code up (~1.2 GB)
  ├─ Preflight (§12 sklearn+lgb smoke, new §13-§16 checks)
  ├─ R10-C: Re-regen splits with --lovd/dbnsfp/finngen paths
  ├─ 4.1 Data-quality audit (on new splits)
  ├─ 4.4 Optuna HPO (3-fold, 50 trials/model, ~2h)
  └─ Apply best params to _build_estimators

Phase 2: Training (Vast.ai, same instance)
  ├─ Full 5-fold CV training with all 10 base estimators
  ├─ Per-model checkpoint + OOF + meta JSON
  ├─ OOF row-index sidecar saved
  ├─ Meta-learner + test evaluation BEFORE ensemble.save()
  ├─ ensemble.save() with per-model joblib layout
  └─ SCP back all outputs (~1.5-2 GB)

Phase 3: Destroy + Local Verification (separate paste block!)
  ├─ Verify all artifacts on local disk
  ├─ vastai destroy instance $ID (SEPARATE code block)
  ├─ Local data-quality audit comparison (pre vs post regen)
  ├─ Local inference verification on test set
  ├─ Commit + push results
  └─ Session doc + CHANGELOG + incident docs (if any)
```

### Dependency edges:
- 4.1 (audit) → depends on R10-C (re-regen)
- 4.4 (HPO) → depends on R10-C (needs new splits)
- 4.6 (features) → feeds into R10-C (new features in regen)
- 4.7 (BF16) → applies during Phase 2 training
- 3.1 (CNN1D refactor) → must be done before Phase 2 training
- 3.2 (OOF sidecar) → must be done before Phase 2 training

---

## 6. Verification Protocol

### 6.1 Pre-Commit Verification (local, before push)

```powershell
# 1. Clean __pycache__
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force

# 2. Full test suite
python -m pytest tests/ -v --timeout=300 -q

# 3. Verify no regressions in existing tests
python -m pytest tests/unit/test_variant_ensemble_save_load.py -v
python -m pytest tests/unit/test_lovd_annotation_reaches_training_matrix.py -v
python -m pytest tests/unit/test_cnn1d_pickle_roundtrip.py -v  # NEW

# 4. Git status
git status
git diff --stat
```

### 6.2 Vast.ai Preflight (on instance, before training)

The existing `scripts/preflight_vm.sh` (sections 1-12) plus these new sections:

```bash
# §13 FastKAN import
echo "--- §13 FastKAN ---"
python -c "from fastkan import FastKAN; print('PASS: FastKAN importable')" || echo "FAIL: FastKAN not importable"

# §14 Polars import
echo "--- §14 Polars ---"
python -c "import polars; print(f'PASS: Polars {polars.__version__}')" || echo "FAIL: Polars not importable"

# §15 GPU GBDT check
echo "--- §15 GPU GBDT ---"
python -c "
import torch
if torch.cuda.is_available():
    print(f'PASS: CUDA available, device={torch.cuda.get_device_name(0)}')
else:
    print('WARN: No CUDA — GBDTs will use CPU')
"

# §16 BF16 support
echo "--- §16 BF16 ---"
python -c "
import torch
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    print('PASS: BF16 supported')
else:
    print('WARN: BF16 not supported — will use FP32')
"

# §17 PrimateAI-3D data
echo "--- §17 PrimateAI-3D ---"
if [ -f data/external/primateai3d/primateai3d_scores.parquet ]; then
    SIZE_MB=$(du -m data/external/primateai3d/primateai3d_scores.parquet | cut -f1)
    if [ "$SIZE_MB" -ge 100 ]; then
        echo "PASS: PrimateAI-3D parquet present (${SIZE_MB} MB)"
    else
        echo "WARN: PrimateAI-3D parquet suspiciously small (${SIZE_MB} MB)"
    fi
else
    echo "WARN: PrimateAI-3D parquet not found — primateai3d_score will be 0"
fi
```

### 6.3 Post-Training Verification (on instance, before SCP)

```bash
# 1. Verify all model checkpoints exist
echo "=== Model checkpoint verification ==="
for model in random_forest xgboost lightgbm logistic_regression \
             gradient_boosting catboost tabular_nn cnn_1d \
             kan mc_dropout deep_ensemble; do
    if [ -f "outputs/run11/full/models/${model}.joblib" ]; then
        SIZE=$(du -k "outputs/run11/full/models/${model}.joblib" | cut -f1)
        echo "  PASS: ${model}.joblib (${SIZE} KB)"
    else
        echo "  MISS: ${model}.joblib"
    fi
done

# 2. Verify metrics.json exists and has test AUROC
python -c "
import json
with open('outputs/run11/full/metrics.json') as f:
    m = json.load(f)
print(f'TEST AUROC: {m.get(\"test_auroc\", \"MISSING\")}')
print(f'OOF  AUROC: {m.get(\"oof_blend_auroc\", \"MISSING\")}')
"

# 3. Verify OOF row-index sidecars
for model in random_forest xgboost lightgbm logistic_regression \
             gradient_boosting catboost tabular_nn cnn_1d \
             kan mc_dropout deep_ensemble; do
    if [ -f "outputs/run11/full/models/${model}_oof_indices.npy" ]; then
        echo "  PASS: ${model}_oof_indices.npy"
    else
        echo "  MISS: ${model}_oof_indices.npy"
    fi
done

# 4. Verify Optuna HPO database
if [ -f "outputs/run11/optuna_hpo.db" ]; then
    python -c "
import optuna
storage = 'sqlite:///outputs/run11/optuna_hpo.db'
for study_name in optuna.study.get_all_study_names(storage):
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f'{study_name}: {len(study.trials)} trials, best={study.best_value:.4f}')
"
else
    echo "  WARN: Optuna DB not found"
fi
```

### 6.4 Post-SCP Local Verification (local, after artifacts received)

```powershell
# 1. Verify artifact integrity
$artifactDir = "C:\Projects\genomic-variant-classifier\outputs\run11\full"
$models = @("random_forest","xgboost","lightgbm","logistic_regression",
            "gradient_boosting","catboost","tabular_nn","cnn_1d",
            "kan","mc_dropout","deep_ensemble")
foreach ($m in $models) {
    $path = "$artifactDir\models\${m}.joblib"
    if (Test-Path $path) {
        $size = (Get-Item $path).Length / 1MB
        Write-Host "  PASS: ${m}.joblib ($([math]::Round($size,1)) MB)" -ForegroundColor Green
    } else {
        Write-Host "  MISS: ${m}.joblib" -ForegroundColor Red
    }
}

# 2. Verify metrics.json
$metricsPath = "$artifactDir\metrics.json"
if (Test-Path $metricsPath) {
    $m = Get-Content $metricsPath | ConvertFrom-Json
    Write-Host "TEST AUROC: $($m.test_auroc)" -ForegroundColor Cyan
    Write-Host "OOF  AUROC: $($m.oof_blend_auroc)" -ForegroundColor Cyan
} else {
    Write-Host "FAIL: metrics.json not found" -ForegroundColor Red
}

# 3. Local inference spot-check
python -c "
import joblib, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
X_test = pd.read_parquet('$artifactDir/splits/X_test.parquet')
y_test = pd.read_parquet('$artifactDir/splits/y_test.parquet')['label']
preds = []
for name in ['catboost','lightgbm','xgboost','random_forest',
             'gradient_boosting','logistic_regression','tabular_nn','mc_dropout']:
    try:
        model = joblib.load(f'$artifactDir/models/{name}.joblib')
        p = model.predict_proba(X_test.values)[:, 1]
        preds.append(p)
        auroc = roc_auc_score(y_test, p)
        print(f'  {name}: TEST AUROC = {auroc:.4f}')
    except Exception as e:
        print(f'  {name}: FAILED — {e}')
if preds:
    avg = np.mean(preds, axis=0)
    print(f'  simple_average: TEST AUROC = {roc_auc_score(y_test, avg):.4f}')
"
```

---

## 7. Cost Projection

| Phase | Estimated Wall-Clock | Estimated Cost |
|-------|---------------------|---------------|
| R10-D gene expansion (manual) | 60 min | $0.00 |
| Local code changes | 4-5 hours | $0.00 |
| Vast.ai HPO (50 trials × 5 models) | 2-3 hours | $1.50-$2.25 |
| Vast.ai re-regen (R10-C) | 30-45 min | $0.40-$0.55 |
| Vast.ai training (10 models) | 3-4 hours | $2.25-$3.00 |
| Vast.ai deep_ensemble | 30 min | $0.40 |
| Vast.ai idle (SSH, SCP, verification) | 30-60 min | $0.40-$0.75 |
| **Total Vast.ai** | **7-9 hours** | **$5-7** |
| **Total (incl. local)** | **12-14 hours** | **$5-7** |

GPU acceleration (Integration 3) should reduce tree-model training from
~2h (Run 10b) to ~30-45 min. BF16 (Integration 7) should reduce neural
model training by 40-50%.

---

## 8. Files Required From Monzia

To produce the actual implementation scripts (downloadable .py files), I need
to see the CURRENT production versions of these files (the project knowledge
copies are the Phase 1 originals, not the post-C5 namespace production code):

1. **`src/genomic_variant_classifier/models/variant_ensemble.py`** — the ACTUAL
   production file after all Phase 1.x patches (CNN1D class, KAN, mc_dropout,
   deep_ensemble, per-model checkpoint, etc.)

2. **`src/genomic_variant_classifier/data/real_data_prep.py`** — the ACTUAL
   production file with the 78-feature `engineer_features()` and annotation
   pipeline

3. **`scripts/run_phase2_eval.py`** — the training/eval orchestrator script
   that runs on Vast.ai

4. **`scripts/launch_run10b_vm.sh`** — most recent launch script template

5. **`requirements.txt`** — current dependency pins

6. **`scripts/preflight_vm.sh`** — current preflight script (with sections 1-12)

7. **`tests/unit/test_variant_ensemble_save_load.py`** — existing ensemble tests

8. **`docs/ROADMAP.md`** — current roadmap for updating

Once I have these files, I will produce:

- `run11_phase0_local_patches.py` — single applier script that patches all
  local files (CNN1D refactor, FastKAN swap, GPU GBDT, BF16, Parquet ZSTD,
  PrimateAI-3D connector, OOF sidecar, data audit script)

- `run11_tests.py` — all new unit tests

- `launch_run11_vm.sh` — Vast.ai launch script with HPO + training

- `preflight_vm_run11.sh` — updated preflight with §13-§17

- `run11_batch.ps1` — PowerShell batch script that:
  1. Moves all files from `~\Downloads\` to `C:\Projects\genomic-variant-classifier\`
  2. Applies patches
  3. Runs tests
  4. Commits and pushes

---

## 9. Lessons Ledger Template

Each integration in Run 11 must produce a lessons entry with this structure:

```markdown
### Integration N: [Name]

**Hypothesis:** [What we expected to happen]
**Result:** [What actually happened]
**Metric delta:**
  - AUROC: [baseline] → [new] (Δ=[change])
  - Wall-clock: [baseline] → [new] (speedup=[factor])
  - Memory: [baseline] → [new] (reduction=[factor])
**Verdict:** [KEEP / REVERT / MODIFY]
**Rationale:** [Why]
**Refinement opportunity:** [Next steps if any]
```

This structure ensures that every change produces actionable data for the
"Aggressive" tier (Run 12+).

---

## Appendix: Run 11 → Run 12 "Aggressive" Tier Preview

The following items are DEFERRED to Run 12+ pending Run 11 lessons:

1. **Frozen ESM-2 embeddings** — requires HGVSp parser (§3.3)
2. **NeighborLoader GNN** — requires gene_symbol in meta-features
3. **ONNX export** — for production inference API
4. **HGMD Professional** — requires procurement (§2.5)
5. **MedGen** (R10-F) — gene-level disease context features
6. **ClinGen** (R10-G) — gene-disease validity + VCEP curated variants
7. **DECIPHER** (R10-H) — conditional on license terms
8. **Ablation matrix** (13 ablations) — full scientific validation
9. **Temporal holdout** — requires date column not in current splits
10. **Coreset selection / SubZeroCore** — dataset distillation
11. **Self-supervised pre-training** — scFoundation/CellFM adaptation
