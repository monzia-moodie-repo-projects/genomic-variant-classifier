#!/usr/bin/env bash
# launch_run17_rnaseq_ablation.sh -- Run-17-SCALE RNA-seq gene-prior ablation (settles the small-scale
# test-vs-val gene-shuffle disagreement). Runs as the FIRST job on the provisioned 4090, before the full
# Run 17. Unlike the small reduced-context probe, this wires the FULL Run-17 input set (dbnsfp,
# alphamissense, gtex, constraint, STRING/GNN, kg, rnaseq) so all 87 features are live, at larger
# --max-train and >=3 seeds.
#
# Configs:  full (rnaseq intact, x1) | drop_all (rnaseq off == floor, x1) | gene_shuffle (per seed)
# Cost knobs (env):  MAX_TRAIN (default 50000), SEEDS (default "11 23 37"), GNN (1=on/full features,
#                    0=drop GNN for a cheaper tree-only pass), SKIP (model skips; default skips NN/CNN/KAN/SVM).
#
# Usage (on the VM, repo root, AFTER Run_Preflight_VM.sh is green):
#   bash scripts/launch_run17_rnaseq_ablation.sh
set -euo pipefail

if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
    export PATH="/venv/main/bin:$PATH"
fi

REPO=/workspace/genomic-variant-classifier
DATA="$REPO/data"
OUTROOT="$REPO/outputs/run17_ablation"
ABLDIR="$DATA/external/rnaseq/ablations"
LOG=/workspace/run17_ablation_master.log
RNASEQ="$DATA/external/rnaseq_gene_expression.parquet"

MAX_TRAIN="${MAX_TRAIN:-50000}"
SEEDS="${SEEDS:-11 23 37}"
GNN="${GNN:-1}"
SKIP="${SKIP:---skip-nn --skip-cnn --skip-kan --skip-svm}"

mkdir -p "$OUTROOT" "$ABLDIR"
cd "$REPO"
echo "==> RNA-seq Run-17-scale ablation @ $(date -u +'%F %T') UTC | MAX_TRAIN=$MAX_TRAIN SEEDS='$SEEDS' GNN=$GNN" | tee "$LOG"
echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# imodelsx KAN patch is harmless here (KAN skipped), but keep parity with the launcher's contract.
python scripts/patch_imodelsx_kan.py 2>&1 | tee -a "$LOG" || true

# -- Common Run-17 feature args (FULL input set) ------------------------------
COMMON="--clinvar $DATA/processed/clinvar_grch38_clean.parquet"
COMMON="$COMMON --seq-windows $DATA/processed/clinvar_grch38_clean_seq.parquet"
COMMON="$COMMON --gnomad $DATA/processed/gnomad_v4_exomes.parquet"
COMMON="$COMMON --spliceai $DATA/external/spliceai/spliceai_index.parquet"
COMMON="$COMMON --alphamissense $DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz"
COMMON="$COMMON --gnomad-constraint $DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv"
COMMON="$COMMON --dbnsfp-path $DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet"
COMMON="$COMMON --gtex-path $DATA/external/gtex_gene_expression.parquet"
COMMON="$COMMON --reactome-path $DATA/external/reactome_gene_pathways.parquet"
COMMON="$COMMON --kg $DATA/external/1kgp/kg_grch38_af.parquet"
COMMON="$COMMON --min-review-tier 3 --n-folds 5 $SKIP --max-train $MAX_TRAIN"
if [ -f "$DATA/external/lovd/lovd_all_variants.parquet" ]; then
    COMMON="$COMMON --lovd-path $DATA/external/lovd/lovd_all_variants.parquet"
fi
if [ "$GNN" = "1" ]; then
    COMMON="$COMMON --string-db auto --hetero-gnn --kg-edges reactome:$DATA/external/reactome/ReactomePathways.gmt"
    echo "==> GNN ON (full feature set incl gnn_score/hetero_gnn_score)" | tee -a "$LOG"
else
    echo "==> GNN OFF (cheaper tree-only pass; gnn_score will be its default constant)" | tee -a "$LOG"
fi

run_one() {
    local cfg="$1" seed="$2" rnaseq="$3"
    local name="$cfg"; [ -n "$seed" ] && name="${cfg}_seed${seed}"
    local out="$OUTROOT/$name"
    echo "============================================================" | tee -a "$LOG"
    echo "==> [$name] --rnaseq-path $rnaseq -> $out @ $(date -u +'%T') UTC" | tee -a "$LOG"
    set +e
    python scripts/run_phase2_eval.py $COMMON --rnaseq-path "$rnaseq" --output "$out" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ] || [ ! -f "$out/metrics.json" ]; then
        echo "==> [$name] FAILED rc=$rc (metrics.json present: $([ -f "$out/metrics.json" ] && echo yes || echo no))" | tee -a "$LOG"
        return 1
    fi
    echo "==> [$name] OK" | tee -a "$LOG"
}

# full (rnaseq intact) -- use the real parquet directly
run_one full "" "$RNASEQ"

# floor (rnaseq fully off) -- build once
python scripts/make_rnaseq_ablation_parquet.py --src "$RNASEQ" --out "$ABLDIR/drop_all.parquet" --mode drop_all 2>&1 | tee -a "$LOG"
run_one drop_all "" "$ABLDIR/drop_all.parquet"

# gene_shuffle -- per seed (the only seed-dependent config)
for s in $SEEDS; do
    python scripts/make_rnaseq_ablation_parquet.py --src "$RNASEQ" --out "$ABLDIR/gene_shuffle_seed${s}.parquet" --mode gene_shuffle --seed "$s" 2>&1 | tee -a "$LOG"
    run_one gene_shuffle "$s" "$ABLDIR/gene_shuffle_seed${s}.parquet"
done

echo "============================================================" | tee -a "$LOG"
echo "==> Aggregating" | tee -a "$LOG"
python scripts/aggregate_rnaseq_ablation.py --runs-root "$OUTROOT" --out "$OUTROOT/ablation_summary.csv" 2>&1 | tee -a "$LOG"
echo "==> DONE @ $(date -u +'%F %T') UTC. Summary: $OUTROOT/ablation_summary.csv" | tee -a "$LOG"
