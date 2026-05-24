#!/usr/bin/env bash
# launch_run10b_skip_kan_v2.sh — Run 10b: --skip-kan + Phase 1.7.1 checkpoint patch verified
#
# v2 changes (2026-05-24):
#   - patch detection by CODE SIGNATURE not comment text (more robust)
#   - data file preflight restored
#   - .pyc cache invalidation before import
#   - smoke import test before launching 2-3h training
#   - all checks fail-fast with explicit exit codes (2/3/4/5)

set -euo pipefail

# PATH fix for vastai/pytorch mini image
if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
    export PATH="/venv/main/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || echo /venv/main/bin/python)"
echo "==> Python interpreter: $PY ($($PY --version 2>&1))"

LOG=/workspace/run10b_master.log
OUTDIR=/workspace/outputs/run10a/full
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

echo "==> Run 10b launch @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee "$LOG"

# ── 1. Data file preflight ───────────────────────────────────────────────
echo "==> [1/5] Data file preflight" | tee -a "$LOG"
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
if [ $FAIL -eq 1 ]; then
    echo "==> ABORT (exit 2): missing data files" | tee -a "$LOG"
    exit 2
fi

# ── 2. Patch sanity check (CODE SIGNATURE, not comment text) ─────────────
cd "$REPO"
echo "==> [2/5] Patch verification (code-signature based)" | tee -a "$LOG"
HAS_JOBLIB=$(grep -c "^import joblib$" src/genomic_variant_classifier/models/variant_ensemble.py || echo 0)
HAS_CHECKPOINT=$(grep -c "joblib.dump(self.trained_models_" src/genomic_variant_classifier/models/variant_ensemble.py || echo 0)
if [ "$HAS_JOBLIB" -lt 1 ] || [ "$HAS_CHECKPOINT" -lt 1 ]; then
    echo "==> ABORT (exit 3): Phase 1.7.1 checkpoint patch missing" | tee -a "$LOG"
    echo "==>   import joblib lines:                 $HAS_JOBLIB (expected >= 1)" | tee -a "$LOG"
    echo "==>   joblib.dump(self.trained_models_ x:  $HAS_CHECKPOINT (expected >= 1)" | tee -a "$LOG"
    echo "==> Run on cloud: cd $REPO && git pull origin main" | tee -a "$LOG"
    echo "==> Need commit f147112 or later." | tee -a "$LOG"
    exit 3
fi
echo "==> Patch verified: import=$HAS_JOBLIB, checkpoint=$HAS_CHECKPOINT" | tee -a "$LOG"
echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# ── 3. Stale .pyc cache cleanup ──────────────────────────────────────────
echo "==> [3/5] Clearing stale .pyc cache under $REPO/src" | tee -a "$LOG"
find "$REPO/src" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "==> .pyc cleared" | tee -a "$LOG"

# ── 4. Smoke import test ─────────────────────────────────────────────────
echo "==> [4/5] Smoke import test (should print 'OK' in <5s)" | tee -a "$LOG"
if ! python -c "from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble; print('OK')" 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 4): import of patched code failed" | tee -a "$LOG"
    exit 4
fi

# ── 5. STRING DB symlinks (idempotent) ───────────────────────────────────
echo "==> [5/5] STRING DB setup" | tee -a "$LOG"
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

# ── Launch ────────────────────────────────────────────────────────────────
echo "==> ALL PREFLIGHT PASSED. Starting run_phase2_eval.py with --skip-kan @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"

python scripts/run_phase2_eval.py \
    --clinvar       "$DATA/processed/clinvar_grch38.parquet" \
    --gnomad        "$DATA/processed/gnomad_v4_exomes.parquet" \
    --spliceai      "$DATA/external/spliceai/spliceai_index.parquet" \
    --alphamissense "$DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz" \
    --lovd-path     "$DATA/external/lovd/lovd_all_variants.parquet" \
    --dbnsfp-path   "$DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet" \
    --gtex-genes    BRCA1 BRCA2 TP53 PTEN ATM \
    $STRING_ARG \
    --skip-kan \
    --output        "$OUTDIR" \
    2>&1 | tee -a "$LOG"

echo "==> run_phase2_eval.py finished @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"

for f in "$OUTDIR/metrics.json" "$OUTDIR/per_model_metrics.csv"; do
    if [ -f "$f" ]; then
        echo "==> VERIFIED: $f" | tee -a "$LOG"
    else
        echo "==> WARNING: expected output missing: $f" | tee -a "$LOG"
    fi
done

if [ -f "$OUTDIR/metrics.json" ]; then
    python -c "
import json
m = json.load(open('$OUTDIR/metrics.json'))
print(f\"==> TEST AUROC: {m.get('auroc', 'N/A')}\")
print(f\"==> VAL  AUROC: {m.get('val_auroc', 'N/A')}\")
print(f\"==> Run 10 baseline: 0.98163 (compare delta AUROC)\")
" 2>&1 | tee -a "$LOG"
fi
