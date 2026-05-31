#!/usr/bin/env bash
# launch_run15_baseline.sh -- Run 15 honest gene-disjoint baseline.
#
# Purpose: the de-leaked baseline that compares an honest AUROC against the
# leakage-inflated ~0.9974, validates the clean splits + GNN path, and emits the
# full per-model checkpoint set for the post-baseline battery.
#
# This is a NEW script, NOT a reuse of launch_run11_vm.sh. Differences (each one
# fixes a defect or Run-11-ism in that script):
#   - clinvar_grch38_clean.parquet (de-leaked), NOT the dirty clinvar_grch38.parquet
#   - --skip-cnn is UNCONDITIONAL. launch_run11_vm.sh nested --skip-cnn (and the
#     imodelsx patch) inside the `if gnomAD-constraint exists` then-branch, so it
#     only applied when that TSV was present -- a latent logic bug.
#   - no LOVD / dbNSFP required: the honest-baseline input set only.
#   - no --unseen-gene-holdout: that is the C3 ablation (a second full retrain),
#     not the baseline.
#   - no brittle source-grep patch checks (launch_run11's `_CNN1DModule>=2` and
#     `joblib.dump(self.trained_models_)` greps would falsely abort on HEAD).
#     The run10a checkpoint mechanism is verified in CI by
#     tests/unit/test_ensemble_persistence.py; a runtime sentinel replaces the grep.
#   - OUTDIR pinned to outputs/run15_baseline/full to match
#     Run15_Postflight.ps1 -RemoteOutputs.

set -euo pipefail

# PATH fix for the vast.ai pytorch mini image
if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
    export PATH="/venv/main/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || echo /venv/main/bin/python)"

REPO=/workspace/genomic-variant-classifier
DATA="$REPO/data"
OUTDIR="$REPO/outputs/run15_baseline/full"
LOG=/workspace/run15_baseline_master.log

cleanup() {
    rc=$?
    echo "============================================================" | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "==> run_phase2_eval.py exit 0 (success) @ $(date -u +'%F %T') UTC" | tee -a "$LOG"
    else
        echo "==> run_phase2_eval.py exit $rc -- ABORT @ $(date -u +'%F %T') UTC" | tee -a "$LOG"
    fi
    echo "==> Checkpoints under $OUTDIR/models/:" | tee -a "$LOG"
    ls -la "$OUTDIR/models/" 2>&1 | tee -a "$LOG" || true
    echo "============================================================" | tee -a "$LOG"
}
trap cleanup EXIT

echo "==> Run 15 baseline launch @ $(date -u +'%F %T') UTC" | tee "$LOG"
echo "==> Python: $PY ($($PY --version 2>&1))" | tee -a "$LOG"

# -- 1. Required data preflight (honest-baseline input set only) ---------------
echo "==> [1/6] Data preflight" | tee -a "$LOG"
FAIL=0
for f in \
    "$DATA/processed/clinvar_grch38_clean.parquet" \
    "$DATA/processed/gnomad_v4_exomes.parquet" \
    "$DATA/external/spliceai/spliceai_index.parquet" \
    "$DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz" \
; do
    if [ ! -f "$f" ]; then
        echo "==> MISSING (required): $f" | tee -a "$LOG"; FAIL=1
    else
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "==> OK: $f ($(( SZ / 1048576 )) MB)" | tee -a "$LOG"
    fi
done

# STRING is required for the baseline's GNN path (validates gnn_score std > 0)
STRING_LINKS="$DATA/external/string/9606.protein.links.detailed.v12.0.txt.gz"
STRING_INFO="$DATA/external/string/9606.protein.info.v12.0.txt.gz"
for f in "$STRING_LINKS" "$STRING_INFO"; do
    if [ ! -f "$f" ]; then
        echo "==> MISSING (required for GNN): $f" | tee -a "$LOG"; FAIL=1
    else
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "==> OK: $f ($(( SZ / 1048576 )) MB)" | tee -a "$LOG"
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "==> ABORT (exit 2): missing required inputs" | tee -a "$LOG"; exit 2
fi

cd "$REPO"

# -- 2. Smoke import (no brittle source-greps) --------------------------------
echo "==> [2/6] Smoke import" | tee -a "$LOG"
if ! python -c "from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble; from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline; print('import OK')" 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 3): import failed" | tee -a "$LOG"; exit 3
fi
echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# -- 3. Stale .pyc cleanup ----------------------------------------------------
echo "==> [3/6] Clear stale .pyc" | tee -a "$LOG"
find "$REPO/src" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# -- 4. GPU / dependency check ------------------------------------------------
echo "==> [4/6] GPU / dependency check" | tee -a "$LOG"
python -c "
import torch, sklearn, lightgbm
print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU', torch.cuda.get_device_name(0))
else:
    print('WARNING: no CUDA -- GBDT/NN on CPU (much slower)')
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)
print('lightgbm smoke fit OK')
" 2>&1 | tee -a "$LOG"

# -- 5. STRING wiring (connector resolves data/external/string under CWD=REPO) -
echo "==> [5/6] STRING wiring" | tee -a "$LOG"
mkdir -p "$OUTDIR"
echo "==> STRING links $(stat -c%s "$STRING_LINKS") bytes; info $(stat -c%s "$STRING_INFO") bytes" | tee -a "$LOG"

# -- 6. Build CLI + launch ----------------------------------------------------
echo "==> [6/6] Launch" | tee -a "$LOG"
ARGS="--clinvar $DATA/processed/clinvar_grch38_clean.parquet"
ARGS="$ARGS --gnomad $DATA/processed/gnomad_v4_exomes.parquet"
ARGS="$ARGS --spliceai $DATA/external/spliceai/spliceai_index.parquet"
ARGS="$ARGS --alphamissense $DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz"
ARGS="$ARGS --string-db auto"
ARGS="$ARGS --min-review-tier 3 --n-folds 5 --skip-cnn"
ARGS="$ARGS --output $OUTDIR"
echo "==> ARGS: $ARGS" | tee -a "$LOG"

# Checkpoint sentinel: WARN (never auto-kill) if no per-model checkpoint by T+45min.
# Mechanism is CI-verified; a healthy first checkpoint should land within ~1 base estimator.
( sleep 2700
  if ls "$OUTDIR"/models/*.joblib >/dev/null 2>&1; then
      echo "==> CHECKPOINT SENTINEL @ T+45min: checkpoints present -- OK." | tee -a "$LOG"
  else
      echo "==> CHECKPOINT SENTINEL @ T+45min: NO $OUTDIR/models/*.joblib yet -- investigate." | tee -a "$LOG"
  fi
) &
SENTINEL_PID=$!

echo "==> ALL PREFLIGHT PASSED. Launching @ $(date -u +'%F %T') UTC" | tee -a "$LOG"
set +e
python scripts/run_phase2_eval.py $ARGS 2>&1 | tee -a "$LOG"
RUN_RC=${PIPESTATUS[0]}
set -e
kill "$SENTINEL_PID" 2>/dev/null || true

echo "==> run_phase2_eval.py rc=$RUN_RC @ $(date -u +'%F %T') UTC" | tee -a "$LOG"

# -- Post-run artifact verification (matches the verified output contract) -----
echo "==> Post-run artifact check" | tee -a "$LOG"
for f in \
    "$OUTDIR/metrics.json" \
    "$OUTDIR/per_model_metrics.csv" \
    "$OUTDIR/per_model_metrics_val.csv" \
    "$OUTDIR/oof_predictions.parquet" \
    "$OUTDIR/feature_importance.csv" \
    "$OUTDIR/models/ensemble.joblib" \
    "$OUTDIR/models/ensemble.manifest.json" \
; do
    if [ -f "$f" ]; then echo "==> VERIFIED: $f" | tee -a "$LOG"; else echo "==> MISSING: $f" | tee -a "$LOG"; fi
done
N_MODELS=$(ls "$OUTDIR/models/"*.joblib 2>/dev/null | wc -l || echo 0)
echo "==> model .joblib count: $N_MODELS" | tee -a "$LOG"
if [ -f "$OUTDIR/metrics.json" ]; then
    python -c "import json; m=json.load(open('$OUTDIR/metrics.json')); print('==> TEST AUROC:', m.get('auroc','N/A'), '| VAL AUROC:', m.get('val_auroc','N/A'))" 2>&1 | tee -a "$LOG"
fi
echo "==> NEXT (laptop, SEPARATE paste blocks): Run15_Postflight.ps1 -> Vastai_Destroy_Confirmed.ps1" | tee -a "$LOG"
exit "$RUN_RC"
