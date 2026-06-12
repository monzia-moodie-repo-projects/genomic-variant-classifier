#!/usr/bin/env bash
# =============================================================================
# launch_run16_vm.sh -- Run 16 on-box launcher.
#
# Modeled on scripts/launch_run11_vm.sh (Runs 11-15), with the lessons from this
# session folded in (improvement, not a copy):
#   * ENV GATE that ABORTS before launch if imports or CUDA fail (launch_run11
#     had this as [4/7]; an earlier slim port of mine dropped it -> train.py was
#     launched into ModuleNotFoundError. Restored AND made self-healing: it pip-
#     installs requirements.txt if deps are missing, because this box's deps were
#     never installed -- up's phantom-pgrep guard exited before its pip step).
#   * comm=python guard + orphan-probe sweep (fixes L20; launch_run11 lacked it).
#   * Checks the deps train.py ACTUALLY imports -- NOT the GNN deps that
#     Run_Preflight_VM.sh gates on (run16 train.py does not use the GNN path).
#   * Writes a real master log the monitor greps; runs the VERBATIM
#     LAUNCH_CONTRACT_run16 train.py command (entrypoint is train.py, not
#     run_phase2_eval.py).
#
# Author:  Monzia Moodie
#
#   nohup bash scripts/launch_run16_vm.sh >/dev/null 2>&1 &
#   .\scripts\Run16_Monitor.ps1 -SshHost ssh8.vast.ai -SshPort 18494 -Mode Quick
# =============================================================================
set -u

LOG=/workspace/run16_master.log
REPO=/workspace/genomic-variant-classifier
cd "$REPO" || { echo "==> ABORT: repo not found at $REPO"; exit 1; }

# src-layout package: make genomic_variant_classifier importable without a build step.
# requirements.txt installs deps only, NOT the project itself (L22).
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
    export PATH="/venv/main/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || echo /opt/conda/bin/python)"

echo "==> Run 16 launch @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee "$LOG"
echo "==> HEAD: $(git rev-parse --short HEAD 2>/dev/null)" | tee -a "$LOG"
echo "==> Python: $PY ($($PY --version 2>&1))" | tee -a "$LOG"

# --- sweep orphaned status probes (L20): bash, cmdline contains 'echo PROBE_OK' ---
pkill -f 'echo PROBE_OK' 2>/dev/null || true

# --- double-launch guard: accept ONLY a real python train.py (comm=python) ---
RUNNING=""
for p in $(pgrep -f "scripts/train\.py" 2>/dev/null); do
    c=$(cat "/proc/$p/comm" 2>/dev/null)
    case "$c" in python*) RUNNING=$p; break;; esac
done
if [ -n "$RUNNING" ]; then
    echo "==> ABORT: a python train.py is already running (PID $RUNNING). Not relaunching." | tee -a "$LOG"
    exit 0
fi

# --- ENV GATE (the dropped [4/7] gate, restored + self-healing) -------------------
# Exactly the deps train.py imports (NOT torch_geometric -- run16 does not use GNN).
ENV_CHECK='import pandas,numpy,sklearn,catboost,lightgbm,xgboost,imodelsx,transformers,torch,genomic_variant_classifier; assert torch.cuda.is_available()'
if ! $PY -c "$ENV_CHECK" 2>/dev/null; then
    echo "==> deps missing/incomplete -> pip install -r requirements.txt (one-time, ~3-5 min)" | tee -a "$LOG"
    $PY -m pip install -r requirements.txt --break-system-packages 2>&1 | tail -8 | tee -a "$LOG"
    $PY -m pip install -e . --no-deps 2>&1 | tail -4 | tee -a "$LOG"   # editable install of src/ package
fi
if $PY -c "$ENV_CHECK; import torch; print('==> ENV_OK torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG" | grep -q '==> ENV_OK'; then
    :
else
    echo "==> ABORT (exit 4): environment not ready -- imports or CUDA failed. NOT launching train.py." | tee -a "$LOG"
    echo "==>   debug: $PY -c \"$ENV_CHECK\"" | tee -a "$LOG"
    exit 4
fi

# --- imodelsx v1.0.13 KAN bug fix (AFTER install, so imodelsx exists) ---
IMODELSX_KAN=$($PY -c "import imodelsx.kan.kan_sklearn as m; print(m.__file__)" 2>/dev/null || true)
if [ -n "$IMODELSX_KAN" ] && grep -q "test_size=test_size" "$IMODELSX_KAN"; then
    sed -i 's/test_size=test_size/test_size=self.test_size/g' "$IMODELSX_KAN"
    sed -i 's/random_state=random_state/random_state=self.random_state/g' "$IMODELSX_KAN"
    sed -i 's/shuffle=shuffle/shuffle=self.shuffle/g' "$IMODELSX_KAN"
    echo "==> imodelsx_patch: fixed 3 bare-name refs in $IMODELSX_KAN" | tee -a "$LOG"
else
    echo "==> imodelsx_patch: already patched or marker absent" | tee -a "$LOG"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | tee -a "$LOG"

# --- LAUNCH: verbatim LAUNCH_CONTRACT_run16 command, unbuffered, to the master log ---
echo "==> Launching scripts/train.py (run16 contract) @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee -a "$LOG"
nohup $PY -u scripts/train.py \
  --clinvar            data/processed/clinvar_grch38_clean_seq.parquet \
  --alphamissense      data/external/alphamissense/AlphaMissense_hg38.tsv.gz \
  --gnomad             data/processed/gnomad_v4_exomes.parquet \
  --gnomad-constraint  data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
  --dbnsfp-path        data/external/dbnsfp/dbnsfp_clinvar_index.parquet \
  --lovd-path          data/external/lovd/lovd_all_variants.parquet \
  --esm2-model         esm2_t33_650M_UR50D \
  --esm2-uniprot-index data/external/uniprot/uniprot_human_reviewed.parquet \
  --esm2-device        cuda \
  --out-dir            outputs/run16 \
  >> "$LOG" 2>&1 &

TRAIN_PID=$!
echo "==> TRAIN_PID=$TRAIN_PID" | tee -a "$LOG"
# verify it is still alive 3s later (not an instant import crash like before)
sleep 3
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "==> train.py alive 3s after launch (PID $TRAIN_PID). monitor: Run16_Monitor.ps1 -Mode Tail" | tee -a "$LOG"
else
    echo "==> WARN: train.py exited within 3s -- check the tail above for a traceback." | tee -a "$LOG"
fi
