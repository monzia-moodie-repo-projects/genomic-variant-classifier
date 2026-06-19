#!/usr/bin/env bash
# launch_run17_baseline.sh -- Run 17 full multi-source GPU run (forked from launch_run15_baseline.sh).
#
# Activates, vs Run 15, FOUR present-but-previously-constant signal groups WITHOUT a schema change:
#   - af_1kg_{afr,eur,eas,sas,amr}  via --kg  (1000G 30x GRCh38 per-superpop AF parquet)
#   - rnaseq_{mean_log_tpm,detection_rate,log2_cv,log2fc,de_neglog10p}  via --rnaseq-path
#   - hetero_gnn_score              via --hetero-gnn + --kg-edges reactome:<gmt>
#   - gtex_* / reactome_pathway_count (carried from Run 15, kept REQUIRED here)
#
# CONTRACT vs launch_run15_baseline.sh (every difference fixes a Run-17 silent-zero risk):
#   - --rnaseq-path is REQUIRED + wired. The preflight_run17 --emit-kg template OMITS it, so a
#     hand-copied launch would zero the now-real rnaseq_* features. This script hard-fails if absent.
#   - --kg is REQUIRED + wired (af_1kg_*). Hard-fail if the parquet is missing (no silent constant).
#   - --hetero-gnn + --kg-edges reactome:<gmt> wired (hetero_gnn_score; else stays 0.5 default).
#   - --skip-svm ON (run_phase2_eval --help: required >100k; n_train ~1.2M, RBF is O(n^2)).
#   - gtex/reactome/kg/rnaseq ELEVATED to REQUIRED (hard-fail), matching the no-zero-feature directive
#     (Run 15 treated gtex/reactome as if-present). LOVD stays if-present (low-coverage, optional).
#   - --esm2-uniprot-index intentionally NOT wired: ESM-2/EVE stay stubbed until the HGVSp parser
#     lands (roadmap; INCIDENT_2026-04-17). Wiring it now would add a KNOWN-zero feature, not signal.
#   - OUTDIR pinned to outputs/run17_baseline/full.

set -euo pipefail

if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
    export PATH="/venv/main/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || echo /venv/main/bin/python)"

REPO=/workspace/genomic-variant-classifier
DATA="$REPO/data"
OUTDIR="$REPO/outputs/run17_baseline/full"
LOG=/workspace/run17_baseline_master.log

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

echo "==> Run 17 baseline launch @ $(date -u +'%F %T') UTC" | tee "$LOG"
echo "==> Python: $PY ($($PY --version 2>&1))" | tee -a "$LOG"

# -- 1. Required data preflight (Run-17 full input set; hard-fail = no silent zero) -----------
echo "==> [1/6] Data preflight" | tee -a "$LOG"
FAIL=0
KG_PARQUET="$DATA/external/1kgp/kg_grch38_af.parquet"
RNASEQ_PARQUET="$DATA/external/rnaseq_gene_expression.parquet"
GTEX_PARQUET="$DATA/external/gtex_gene_expression.parquet"
REACTOME_PARQUET="$DATA/external/reactome_gene_pathways.parquet"
REACTOME_GMT="$DATA/external/reactome/ReactomePathways.gmt"
for f in \
    "$DATA/processed/clinvar_grch38_clean.parquet" \
    "$DATA/processed/clinvar_grch38_clean_seq.parquet" \
    "$DATA/processed/gnomad_v4_exomes.parquet" \
    "$DATA/external/spliceai/spliceai_index.parquet" \
    "$DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz" \
    "$DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv" \
    "$DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet" \
    "$GTEX_PARQUET" \
    "$REACTOME_PARQUET" \
    "$REACTOME_GMT" \
    "$KG_PARQUET" \
    "$RNASEQ_PARQUET" \
; do
    if [ ! -f "$f" ]; then
        echo "==> MISSING (required): $f" | tee -a "$LOG"; FAIL=1
    else
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "==> OK: $f ($(( SZ / 1048576 )) MB)" | tee -a "$LOG"
    fi
done

# STRING required for the GNN path (gnn_score std > 0)
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

# -- 2. Smoke import ----------------------------------------------------------
echo "==> [2/6] Smoke import" | tee -a "$LOG"
if ! python -c "from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble; from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline; from genomic_variant_classifier.data.rnaseq import annotate_rnaseq_from_parquet; from genomic_variant_classifier.data.thousandgenomes import ThousandGenomesConnector; print('import OK')" 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 3): import failed" | tee -a "$LOG"; exit 3
fi
echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# -- 2b. imodelsx KAN package patch (idempotent) ------------------------------
# patch_imodelsx_kan.py docstring: "the smoke gate invokes it first, and the launcher should too."
# Unpatched, imodelsx KANClassifier.fit raises NameError -> KAN drops from BOTH ensemble fits, an
# invalid model-comparison run. The full launcher has no other KAN guard (only the smoke gate does).
echo "==> [2b/6] imodelsx KAN patch" | tee -a "$LOG"
if ! python scripts/patch_imodelsx_kan.py 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 3): imodelsx KAN patch failed (KAN would drop from the ensemble)" | tee -a "$LOG"; exit 3
fi

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

# -- 5. KG + rnaseq + STRING wiring sanity (read-only column probes) -----------
echo "==> [5/6] KG + rnaseq column probes" | tee -a "$LOG"
mkdir -p "$OUTDIR"
python -c "
import pyarrow.parquet as pq, sys
kg = pq.ParquetFile(r'$KG_PARQUET').schema_arrow.names
need = ['AFR_AF','EUR_AF','EAS_AF','SAS_AF','AMR_AF']
miss = [c for c in need if c not in kg]
print('kg cols:', kg)
if miss:
    print('ABORT: kg parquet missing per-superpop AF cols', miss); sys.exit(4)
rs = pq.ParquetFile(r'$RNASEQ_PARQUET').schema_arrow.names
need_rs = ['gene_symbol','rnaseq_mean_log_tpm','rnaseq_detection_rate','rnaseq_log2_cv','rnaseq_log2fc','rnaseq_de_neglog10p']
miss_rs = [c for c in need_rs if c not in rs]
print('rnaseq cols:', rs)
if miss_rs:
    print('ABORT: rnaseq parquet missing cols', miss_rs); sys.exit(5)
print('kg + rnaseq column contracts OK')
" 2>&1 | tee -a "$LOG"
echo "==> STRING links $(stat -c%s "$STRING_LINKS") bytes; info $(stat -c%s "$STRING_INFO") bytes" | tee -a "$LOG"

# -- 6. Build CLI + launch ----------------------------------------------------
echo "==> [6/6] Launch" | tee -a "$LOG"
ARGS="--clinvar $DATA/processed/clinvar_grch38_clean.parquet"
ARGS="$ARGS --seq-windows $DATA/processed/clinvar_grch38_clean_seq.parquet"
ARGS="$ARGS --gnomad $DATA/processed/gnomad_v4_exomes.parquet"
ARGS="$ARGS --spliceai $DATA/external/spliceai/spliceai_index.parquet"
ARGS="$ARGS --alphamissense $DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz"
ARGS="$ARGS --gnomad-constraint $DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv"
ARGS="$ARGS --dbnsfp-path $DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet"
ARGS="$ARGS --gtex-path $GTEX_PARQUET"
ARGS="$ARGS --reactome-path $REACTOME_PARQUET"
ARGS="$ARGS --rnaseq-path $RNASEQ_PARQUET"
ARGS="$ARGS --kg $KG_PARQUET"
ARGS="$ARGS --string-db auto"
ARGS="$ARGS --hetero-gnn --kg-edges reactome:$REACTOME_GMT"
ARGS="$ARGS --min-review-tier 3 --n-folds 5"
ARGS="$ARGS --skip-svm"
# LOVD is ON-if-present (B9): absence must never silently zero it, but it is optional (low coverage).
LOVD_PARQUET="$DATA/external/lovd/lovd_all_variants.parquet"
if [ -f "$LOVD_PARQUET" ]; then
    ARGS="$ARGS --lovd-path $LOVD_PARQUET"
    echo "==> LOVD wired: $LOVD_PARQUET" | tee -a "$LOG"
else
    echo "==> LOVD absent ($LOVD_PARQUET); proceeding without it (B9 if-present)" | tee -a "$LOG"
fi
ARGS="$ARGS --unseen-gene-holdout"
ARGS="$ARGS --output $OUTDIR"
echo "==> rnaseq wired: $RNASEQ_PARQUET" | tee -a "$LOG"
echo "==> kg wired: $KG_PARQUET" | tee -a "$LOG"
echo "==> hetero-GNN edges: reactome:$REACTOME_GMT" | tee -a "$LOG"
echo "==> NOTE: --esm2-uniprot-index intentionally absent (ESM-2/EVE stubbed pending HGVSp parser)" | tee -a "$LOG"
echo "==> ARGS: $ARGS" | tee -a "$LOG"

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
exit $RUN_RC
