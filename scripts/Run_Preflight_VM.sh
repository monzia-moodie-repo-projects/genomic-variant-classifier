#!/usr/bin/env bash
# scripts/Run_Preflight_VM.sh
# =========================================================================
# G2 - On-VM ENVIRONMENT/HARDWARE pre-flight for Run 15 (Charter v1.1 gate G2).
#
# Runs ON the Vast.ai instance AFTER SCP-up of the repo, BEFORE
# scripts/launch_run11_vm.sh. This is the hardware+environment COMPLEMENT to
# launch_run11_vm.sh's built-in data/code preflight, with NO overlap:
#   - launch_run11_vm.sh owns: data-file presence (under $DATA), patch verify,
#     import smoke (VariantEnsemble/DataPrepPipeline), 1000-row LGBM smoke.
#   - THIS script owns: GPU/CUDA hard gate, GNN deps (torch_geometric,
#     networkx), KAN deps (imodelsx, KANClassifier), VRAM/disk/RAM floors,
#     and repo-commit verification.
# Data-file checks are intentionally NOT duplicated here (launch does them at
# the correct $DATA paths). See also scripts/preflight_vm.sh (DEPRECATED for
# Run 15 path/contract; retained only as an optional deep data audit).
#
# Usage (on the instance, from /workspace/genomic-variant-classifier):
#   bash scripts/Run_Preflight_VM.sh [EXPECTED_HEAD]
#   # exit 0 = environment green; proceed to launch_run11_vm.sh
#   # exit 1 = abort; do NOT start the run
#
# Env overrides: REPO, MIN_VRAM_MIB, MIN_DISK_GB, MIN_RAM_GB, EXPECTED_HEAD
# =========================================================================

set -uo pipefail

REPO="${REPO:-/workspace/genomic-variant-classifier}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-20000}"
MIN_DISK_GB="${MIN_DISK_GB:-150}"
MIN_RAM_GB="${MIN_RAM_GB:-50}"
EXPECTED_HEAD="${EXPECTED_HEAD:-${1:-}}"

# PATH fix for vastai/pytorch images (mirror launch_run11_vm.sh) so G2 checks the
# SAME interpreter the training run will use, not a stray system python.
if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
  export PATH="/venv/main/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || echo /venv/main/bin/python)"
echo "=== interpreter: $PY ==="

C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YLW=$'\033[33m'; C_RST=$'\033[0m'
PASS_COUNT=0; FAIL_COUNT=0; FAILURES=()
pass() { printf "%s[PASS]%s %s\n" "$C_GRN" "$C_RST" "$1"; PASS_COUNT=$((PASS_COUNT+1)); }
fail() { printf "%s[FAIL]%s %s\n" "$C_RED" "$C_RST" "$1"; FAIL_COUNT=$((FAIL_COUNT+1)); FAILURES+=("$1"); }
warn() { printf "%s[WARN]%s %s\n" "$C_YLW" "$C_RST" "$1"; }

# ----- 1. GPU + CUDA hard gate -----
echo "=== [1/6] GPU / CUDA hard gate ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L | grep -qi "GPU "; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    pass "nvidia-smi reports GPU: $GPU_INFO"
    VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ "$VRAM_MIB" =~ ^[0-9]+$ ]] && [[ "$VRAM_MIB" -ge "$MIN_VRAM_MIB" ]]; then
      pass "VRAM ${VRAM_MIB} MiB >= ${MIN_VRAM_MIB} MiB floor"
    else
      fail "VRAM ${VRAM_MIB:-unknown} MiB < ${MIN_VRAM_MIB} MiB floor"
    fi
  else
    fail "nvidia-smi present but no GPU listed"
  fi
else
  fail "nvidia-smi not found on PATH -- no GPU, ABORT"
fi

if "$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null; then
  pass "torch.cuda.is_available() == True"
else
  fail "torch.cuda.is_available() == False -- driver/CUDA mismatch or torch is CPU-only"
fi

# ----- 2. GNN dependencies (torch_geometric, networkx) -----
echo "=== [2/6] GNN dependencies ==="
if "$PY" -c "import torch_geometric, networkx; print('torch_geometric', torch_geometric.__version__, 'networkx', networkx.__version__)" 2>/dev/null; then
  pass "torch_geometric + networkx importable (GNN / run_phase2_eval path)"
else
  fail "torch_geometric/networkx import failed -- GNN cannot run in run_phase2_eval mode"
fi

# ----- 3. KAN backend dependencies (imodelsx + project KANClassifier) -----
echo "=== [3/6] KAN dependencies ==="
if "$PY" -c "import imodelsx.kan.kan_sklearn" 2>/dev/null; then
  pass "imodelsx.kan.kan_sklearn importable (KAN primary backend)"
else
  fail "imodelsx.kan.kan_sklearn import failed -- KAN primary backend unavailable"
fi
if "$PY" -c "from genomic_variant_classifier.models.kan import KANClassifier" 2>/dev/null; then
  pass "genomic_variant_classifier.models.kan.KANClassifier importable"
else
  fail "KANClassifier import failed -- check namespace/install"
fi

# ----- 4. Free disk on the workspace volume -----
echo "=== [4/6] Disk space ==="
DISK_TARGET="$REPO"
[[ -d "$DISK_TARGET" ]] || DISK_TARGET="$(dirname "$REPO")"
[[ -d "$DISK_TARGET" ]] || DISK_TARGET="/"
DISK_AVAIL_GB=$(df -BG --output=avail "$DISK_TARGET" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$DISK_AVAIL_GB" ]] && [[ "$DISK_AVAIL_GB" -ge "$MIN_DISK_GB" ]]; then
  pass "disk free ${DISK_AVAIL_GB} GB on $DISK_TARGET >= ${MIN_DISK_GB} GB floor"
else
  fail "disk free ${DISK_AVAIL_GB:-unknown} GB on $DISK_TARGET < ${MIN_DISK_GB} GB floor"
fi

# ----- 5. RAM (total provisioned) -----
echo "=== [5/6] RAM ==="
RAM_TOTAL_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null)
if [[ -n "$RAM_TOTAL_GB" ]] && [[ "$RAM_TOTAL_GB" -ge "$MIN_RAM_GB" ]]; then
  pass "RAM total ${RAM_TOTAL_GB} GB >= ${MIN_RAM_GB} GB floor"
else
  fail "RAM total ${RAM_TOTAL_GB:-unknown} GB < ${MIN_RAM_GB} GB floor"
fi

# ----- 6. Repo commit verification -----
echo "=== [6/6] Repo commit ==="
if git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
  HEAD_SHA=$(git -C "$REPO" rev-parse --short HEAD)
  HEAD_FULL=$(git -C "$REPO" rev-parse HEAD)
  pass "git HEAD on VM: $HEAD_SHA"
  if [[ -n "$EXPECTED_HEAD" ]]; then
    if [[ "$HEAD_FULL" == "$EXPECTED_HEAD"* ]] || [[ "$HEAD_SHA" == "$EXPECTED_HEAD"* ]] || [[ "$EXPECTED_HEAD" == "$HEAD_SHA"* ]]; then
      pass "HEAD matches EXPECTED_HEAD ($EXPECTED_HEAD)"
    else
      fail "HEAD $HEAD_SHA != EXPECTED_HEAD $EXPECTED_HEAD -- wrong code SCP'd up"
    fi
  else
    warn "no EXPECTED_HEAD provided -- HEAD not verified against a pinned commit"
  fi
  if git -C "$REPO" diff --quiet 2>/dev/null && git -C "$REPO" diff --cached --quiet 2>/dev/null; then
    pass "repo working tree clean"
  else
    warn "repo working tree has uncommitted changes on the VM"
  fi
else
  fail "$REPO is not a git repo -- cannot verify code state"
fi

# ----- Summary -----
echo ""
echo "========================================================================"
echo "G2 VM env preflight: $PASS_COUNT pass, $FAIL_COUNT fail"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  echo "FAILURES:"
  for f in "${FAILURES[@]}"; do echo "  - $f"; done
  echo ""
  echo "DO NOT launch. Fix the above, then re-run G2 before launch_run11_vm.sh."
  exit 1
fi
echo "G2 GREEN -- environment ready. Proceed to: bash scripts/launch_run11_vm.sh"
exit 0