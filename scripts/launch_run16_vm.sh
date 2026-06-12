#!/usr/bin/env bash
# =============================================================================
# launch_run16_vm.sh -- Run 16 on-box launcher.
#
# Modeled on the established scripts/launch_run11_vm.sh (Runs 11-15): writes a
# real master log the monitor greps, frames steps with "==>", applies the
# imodelsx KAN patch, and refuses to double-launch. Differs from run11 only
# where Run 16 differs: entrypoint is scripts/train.py (NOT run_phase2_eval.py)
# and the flag set is the verbatim LAUNCH_CONTRACT_run16.md command.
#
# Author:  Monzia Moodie
#
# Run it (from the box, repo cloned, data staged):
#   nohup bash scripts/launch_run16_vm.sh >/dev/null 2>&1 &
# Then monitor from the laptop:
#   .\scripts\Run16_Monitor.ps1 -SshHost ssh8.vast.ai -SshPort 18494 -Mode Quick
# =============================================================================
set -u

LOG=/workspace/run16_master.log
REPO=/workspace/genomic-variant-classifier
cd "$REPO" || { echo "==> ABORT: repo not found at $REPO"; exit 1; }

if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
    export PATH="/venv/main/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || echo /venv/main/bin/python)"

echo "==> Run 16 launch @ $(date -u +'%Y-%m-%d %H:%M:%S') UTC" | tee "$LOG"
echo "==> HEAD: $(git rev-parse --short HEAD 2>/dev/null)" | tee -a "$LOG"
echo "==> Python: $PY ($($PY --version 2>&1))" | tee -a "$LOG"

# --- sweep orphaned status probes (L20): their cmdline contains 'scripts/train.py'
#     and 'echo PROBE_OK'; they are bash, not python, and can confuse pgrep guards.
pkill -f 'echo PROBE_OK' 2>/dev/null || true

# --- double-launch guard: accept ONLY a real python train.py (comm=python), never
#     a bash probe whose cmdline merely contains the string 'scripts/train.py'.
RUNNING=""
for p in $(pgrep -f "scripts/train\.py" 2>/dev/null); do
    c=$(cat "/proc/$p/comm" 2>/dev/null)
    case "$c" in python*) RUNNING=$p; break;; esac
done
if [ -n "$RUNNING" ]; then
    echo "==> ABORT: a python train.py is already running (PID $RUNNING). Not relaunching." | tee -a "$LOG"
    exit 0
fi

# --- imodelsx v1.0.13 KAN bug fix (verbatim from launch_run11_vm.sh) ---
IMODELSX_KAN=$($PY -c "import imodelsx.kan.kan_sklearn as m; print(m.__file__)" 2>/dev/null || true)
if [ -n "$IMODELSX_KAN" ] && grep -q "test_size=test_size" "$IMODELSX_KAN"; then
    sed -i 's/test_size=test_size/test_size=self.test_size/g' "$IMODELSX_KAN"
    sed -i 's/random_state=random_state/random_state=self.random_state/g' "$IMODELSX_KAN"
    sed -i 's/shuffle=shuffle/shuffle=self.shuffle/g' "$IMODELSX_KAN"
    echo "==> imodelsx_patch: fixed 3 bare-name refs in $IMODELSX_KAN" | tee -a "$LOG"
else
    echo "==> imodelsx_patch: already patched or not installed" | tee -a "$LOG"
fi

# --- env + GPU smoke check ---
$PY -c "import catboost,lightgbm,xgboost,torch; print('==> ENV_OK cuda=', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | tee -a "$LOG"

# --- LAUNCH: verbatim LAUNCH_CONTRACT_run16.md command, unbuffered (-u) so the
#     master log updates live, redirected to the master log the monitor reads.
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
echo "==> launched. monitor: scripts/Run16_Monitor.ps1 -Mode Quick  (log: $LOG)" | tee -a "$LOG"
