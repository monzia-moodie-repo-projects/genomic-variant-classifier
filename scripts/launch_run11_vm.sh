#!/usr/bin/env bash
# launch_run11_vm.sh — Run 11: Standard-tier integrations
#
# Changes from launch_run10b_skip_kan_v2.sh:
#   - Added --gnomad-constraint (FINDING F1: recovers 4 top-5 features)
#   - Added --finngen-path (FINDING F2: recovers 3 features)
#   - Added --kg (FINDING F3: recovers 5 1KGP population AF features)
#   - --skip-kan REMOVED: KAN always on; imodelsx/efficient-kan primary backend (Integration 2)
#   - GPU GBDT auto-detected by torch.cuda (Integration 3)
#   - BF16 auto-detected by torch.cuda.is_bf16_supported (Integration 7)
#   - Parquet ZSTD compression (Integration 8)
#   - Extended preflight (§13-§17)
#   - Optuna HPO phase before full training (Integration 4)
#   - OOF row-index sidecar saved (Carried-forward 3.2)
#   - PrimateAI-3D path wired (Integration 6, if data present)

set -euo pipefail

# PATH fix for vastai/pytorch mini image
if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
    export PATH="/venv/main/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || echo /venv/main/bin/python)"
echo "==> Python interpreter: $PY ($($PY --version 2>&1))"

LOG=/workspace/run11_master.log
OUTDIR=/workspace/outputs/run11/full
REPO=/workspace/genomic-variant-classifier
DATA=/workspace/data

cleanup() {
    local rc=$?
    echo "============================================================" | tee -a "$LOG"
    if [ $rc -eq 0 ]; then
        echo "==> run_phase2_eval.py exit 0 (success)" | tee -a "$LOG"
    else
        echo "==> run_phase2_eval.py exit $rc" | tee -a "$LOG"
        echo "==> ABORT at $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"
    fi
    echo "==> Checkpoints under $OUTDIR/models/:" | tee -a "$LOG"
    ls -la "$OUTDIR/models/" 2>&1 | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
}
trap cleanup EXIT

echo "==> Run 11 launch @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee "$LOG"

# ── 1. Data file preflight ───────────────────────────────────────────────
echo "==> [1/7] Data file preflight" | tee -a "$LOG"
FAIL=0
for f in \
    "$DATA/processed/clinvar_grch38.parquet" \
    "$DATA/processed/gnomad_v4_exomes.parquet" \
    "$DATA/external/spliceai/spliceai_index.parquet" \
    "$DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz" \
    "$DATA/external/lovd/lovd_all_variants.parquet" \
    "$DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet" \
; do
    if [ ! -f "$f" ]; then
        echo "==> MISSING: $f" | tee -a "$LOG"
        FAIL=1
    else
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "==> OK: $f ($(( SZ / 1048576 )) MB)" | tee -a "$LOG"
    fi
done

# Warn-only files (don't block launch if absent)
for f in \
    "$DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv" \
    "$DATA/external/finngen/finnge_R12_annotated_variants_v1.gz" \
    "$DATA/external/1kg/1kg_phase3_af.parquet" \
    "$DATA/external/primateai3d/primateai3d_scores.parquet" \
; do
    if [ -f "$f" ]; then
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "==> OK (optional): $f ($(( SZ / 1048576 )) MB)" | tee -a "$LOG"
    else
        echo "==> WARN (optional): $f not found" | tee -a "$LOG"
    fi
done

if [ $FAIL -eq 1 ]; then
    echo "==> ABORT (exit 2): missing required data files" | tee -a "$LOG"
    exit 2
fi

# ── 2. Patch sanity check ────────────────────────────────────────────────
cd "$REPO"
echo "==> [2/7] Patch verification" | tee -a "$LOG"
HAS_JOBLIB=$(grep -c "^import joblib$" src/genomic_variant_classifier/models/variant_ensemble.py || echo 0)
HAS_CHECKPOINT=$(grep -c "joblib.dump(self.trained_models_" src/genomic_variant_classifier/models/variant_ensemble.py || echo 0)
HAS_CNN1D_MODULE=$(grep -c "_CNN1DModule" src/genomic_variant_classifier/models/variant_ensemble.py || echo 0)
if [ "$HAS_JOBLIB" -lt 1 ] || [ "$HAS_CHECKPOINT" -lt 1 ] || [ "$HAS_CNN1D_MODULE" -lt 2 ]; then
    echo "==> ABORT (exit 3): Required patches missing" | tee -a "$LOG"
    echo "==>   import joblib:    $HAS_JOBLIB (expected >= 1)" | tee -a "$LOG"
    echo "==>   checkpoint code:  $HAS_CHECKPOINT (expected >= 1)" | tee -a "$LOG"
    echo "==>   CNN1D module-level: $HAS_CNN1D_MODULE (expected >= 2)" | tee -a "$LOG"
    exit 3
fi
echo "==> Patches verified: joblib=$HAS_JOBLIB, ckpt=$HAS_CHECKPOINT, cnn1d=$HAS_CNN1D_MODULE" | tee -a "$LOG"
echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# ── 3. Stale .pyc cache cleanup ──────────────────────────────────────────
echo "==> [3/7] Clearing stale .pyc cache" | tee -a "$LOG"
find "$REPO/src" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── 4. Smoke import test ─────────────────────────────────────────────────
echo "==> [4/7] Smoke import test" | tee -a "$LOG"
if ! python -c "
from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble
from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline
print('OK: VariantEnsemble + DataPrepPipeline importable')
" 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 4): import failed" | tee -a "$LOG"
    exit 4
fi

# ── 5. GPU + BF16 + dependency check ────────────────────────────────────────
echo "==> [5/7] GPU / BF16 / dependency check" | tee -a "$LOG"
python -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
else:
    print('WARNING: No CUDA — GBDTs will use CPU, no BF16')
try:
    import polars
    print(f'Polars {polars.__version__}')
except ImportError:
    print('WARNING: Polars not installed — audit will use pandas')
try:
    import optuna
    print(f'Optuna {optuna.__version__}')
except ImportError:
    print('WARNING: Optuna not installed — HPO will be skipped')
import sklearn, lightgbm
print(f'sklearn {sklearn.__version__}  lightgbm {lightgbm.__version__}')
# 1000-row LGBM smoke fit (Phase 1.5b lesson)
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
clf = LGBMClassifier(n_estimators=10, verbose=-1)
clf.fit(X, y)
_ = clf.predict_proba(X[:5])
print('LGBM smoke fit: OK')
" 2>&1 | tee -a "$LOG"

# ── 6. STRING DB + optional data symlinks ─────────────────────────────────
echo "==> [6/7] STRING DB + symlinks" | tee -a "$LOG"
STRING_LINKS="$DATA/external/string/9606.protein.links.detailed.v12.0.txt.gz"
STRING_INFO="$DATA/external/string/9606.protein.info.v12.0.txt.gz"
STRING_ARG=""
if [ -f "$STRING_LINKS" ] && [ -f "$STRING_INFO" ]; then
    mkdir -p data/external/string
    ln -sf "$STRING_LINKS" data/external/string/
    ln -sf "$STRING_INFO" data/external/string/
    STRING_ARG="--string-db auto"
    echo "==> STRING DB wired (GNN enabled)" | tee -a "$LOG"
else
    echo "==> STRING DB not found (GNN will skip)" | tee -a "$LOG"
fi

mkdir -p "$OUTDIR"

# ── 7. Build CLI args dynamically based on available data ─────────────────
echo "==> [7/7] Building CLI arguments" | tee -a "$LOG"

# Required args
ARGS="--clinvar $DATA/processed/clinvar_grch38.parquet"
ARGS="$ARGS --gnomad $DATA/processed/gnomad_v4_exomes.parquet"
ARGS="$ARGS --spliceai $DATA/external/spliceai/spliceai_index.parquet"
ARGS="$ARGS --alphamissense $DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz"
ARGS="$ARGS --lovd-path $DATA/external/lovd/lovd_all_variants.parquet"
ARGS="$ARGS --dbnsfp-path $DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet"
ARGS="$ARGS --gtex-genes BRCA1 BRCA2 TP53 PTEN ATM"
ARGS="$ARGS $STRING_ARG"
ARGS="$ARGS --output $OUTDIR"

# FINDING F1: gnomAD constraint (recovers loeuf, syn_z, mis_z, pli_score)
GNOMAD_CONSTRAINT="$DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv"
if [ -f "$GNOMAD_CONSTRAINT" ]; then
    ARGS="$ARGS --gnomad-constraint $GNOMAD_CONSTRAINT"
    echo "==> gnomAD constraint wired (recovers 4 features)"
ARGS="$ARGS --skip-cnn"
echo "==> CNN_1D skipped (no fasta_seq data available)" | tee -a "$LOG"

# --- imodelsx v1.0.13 bug fix (bare-name references in KANClassifier.fit) ---
IMODELSX_KAN=$(python -c "import imodelsx.kan.kan_sklearn as m; print(m.__file__)" 2>/dev/null)
if [ -n "$IMODELSX_KAN" ] && grep -q "test_size=test_size" "$IMODELSX_KAN"; then
    sed -i 's/test_size=test_size/test_size=self.test_size/g' "$IMODELSX_KAN"
    sed -i 's/random_state=random_state/random_state=self.random_state/g' "$IMODELSX_KAN"
    sed -i 's/shuffle=shuffle/shuffle=self.shuffle/g' "$IMODELSX_KAN"
    echo "==> imodelsx_patch: fixed 3 bare-name refs in $IMODELSX_KAN" | tee -a "$LOG"
else
    echo "==> imodelsx_patch: already patched or not installed" | tee -a "$LOG"
fi  # A3 fix 2026-05-27: removed redundant outer tee
else
    echo "==> WARN: gnomAD constraint TSV not found — pli/loeuf/syn_z/mis_z will be 0" | tee -a "$LOG"
fi

# PM11b (2026-05-27): C3 hypothesis falsifier (b) - unseen-gene-holdout ablation flag
ARGS="$ARGS --unseen-gene-holdout"

# FINDING F2: FinnGen (recovers finngen_af_fin, finngen_af_nfsee, finngen_enrichment)
FINNGEN="$DATA/external/finngen/finnge_R12_annotated_variants_v1.gz"
if [ -f "$FINNGEN" ]; then
    ARGS="$ARGS --finngen-path $FINNGEN"
    echo "==> FinnGen wired (recovers 3 features)" | tee -a "$LOG"
else
    echo "==> WARN: FinnGen TSV not found — FinnGen features will be 0" | tee -a "$LOG"
fi

# FINDING F3: 1000 Genomes (recovers af_1kg_afr/eur/eas/sas/amr)
KG="$DATA/external/1kg/1kg_phase3_af.parquet"
if [ -f "$KG" ]; then
    ARGS="$ARGS --kg $KG"
    echo "==> 1KGP wired (recovers 5 features)" | tee -a "$LOG"
else
    echo "==> WARN: 1KGP parquet not found — 1KGP AF features will be 0" | tee -a "$LOG"
fi

echo "==> Full CLI args:" | tee -a "$LOG"
echo "==> $ARGS" | tee -a "$LOG"

# ── Launch ────────────────────────────────────────────────────────────────
echo "==> ALL PREFLIGHT PASSED. Launching run_phase2_eval.py @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"

python scripts/run_phase2_eval.py $ARGS 2>&1 | tee -a "$LOG"

echo "==> run_phase2_eval.py finished @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"

# ── Post-training verification ────────────────────────────────────────────
echo "==> Post-training verification" | tee -a "$LOG"

for f in "$OUTDIR/metrics.json" "$OUTDIR/per_model_metrics.csv"; do
    if [ -f "$f" ]; then
        echo "==> VERIFIED: $f" | tee -a "$LOG"
    else
        echo "==> WARNING: expected output missing: $f" | tee -a "$LOG"
    fi
done

# Model checkpoint count
N_MODELS=$(ls "$OUTDIR/models/"*.joblib 2>/dev/null | wc -l)
echo "==> Model checkpoints: $N_MODELS .joblib files" | tee -a "$LOG"

# OOF sidecar count
N_OOF_IDX=$(ls "$OUTDIR/models/"*_oof_indices.npy 2>/dev/null | wc -l)
echo "==> OOF row-index sidecars: $N_OOF_IDX files" | tee -a "$LOG"

if [ -f "$OUTDIR/metrics.json" ]; then
    python -c "
import json
m = json.load(open('$OUTDIR/metrics.json'))
print(f'==> TEST AUROC: {m.get(\"auroc\", \"N/A\")}')
print(f'==> VAL  AUROC: {m.get(\"val_auroc\", \"N/A\")}')
print(f'==> Run 10b baseline: 0.9970 (compare delta)')
n_features = m.get('n_features', 'N/A')
print(f'==> Features: {n_features}')
" 2>&1 | tee -a "$LOG"
fi

# Data quality audit on fresh splits
echo "==> Running post-training data quality audit" | tee -a "$LOG"
if [ -f "scripts/run11_data_quality_audit.py" ]; then
    python scripts/run11_data_quality_audit.py \
        --splits-dir "$OUTDIR/splits" \
        --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG"
fi

echo "==> Run 11 complete @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"
echo "==> NEXT: SCP artifacts back, then SEPARATELY destroy instance" | tee -a "$LOG"
echo "==> STANDING RULE #30: vastai destroy in SEPARATE paste block!" | tee -a "$LOG"
