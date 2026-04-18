#!/usr/bin/env bash
# scripts/preflight_vm.sh
# =========================================================================
# On-VM preflight that runs AFTER SSH into the Vast.ai instance, BEFORE
# training starts. Catches path mismatches and silent-fallback bugs that
# the local preflight cannot see, because they only exist on the container
# filesystem.
#
# Usage (on the Vast.ai instance, in /workspace):
#   bash scripts/preflight_vm.sh
#   # exit code 0 = safe to train; non-zero = abort
#
# This is the on-VM complement to scripts/preflight_check.py. Both must
# pass. The local one prevents misconfiguration before GPU billing starts;
# this one prevents misconfiguration after the container is built.
# =========================================================================

set -euo pipefail

# Colors (best-effort; some terminals may not honor them)
C_RED=$'\033[31m'
C_GRN=$'\033[32m'
C_YLW=$'\033[33m'
C_RST=$'\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
FAILURES=()

pass() {
  printf "%s[PASS]%s %s\n" "$C_GRN" "$C_RST" "$1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  printf "%s[FAIL]%s %s\n" "$C_RED" "$C_RST" "$1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  FAILURES+=("$1")
}

warn() {
  printf "%s[WARN]%s %s\n" "$C_YLW" "$C_RST" "$1"
}

# -------------------------------------------------------------------------
# 1. GPU present and reachable
# -------------------------------------------------------------------------
echo "=== GPU check ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L | grep -qi "GPU "; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    pass "nvidia-smi reports GPU: $GPU_INFO"
  else
    fail "nvidia-smi present but no GPU listed"
  fi
else
  fail "nvidia-smi not found on PATH -- no GPU, ABORT"
fi

# Confirm PyTorch sees the GPU — this catches container/driver mismatches
# that nvidia-smi alone won't detect (e.g. CUDA-too-old-for-torch).
if python -c "import torch; assert torch.cuda.is_available(); print('torch cuda version:', torch.version.cuda)" 2>/dev/null; then
  pass "torch.cuda.is_available() == True"
else
  fail "torch.cuda.is_available() == False -- driver/CUDA mismatch"
fi

# -------------------------------------------------------------------------
# 2. SpliceAI parquet
# -------------------------------------------------------------------------
echo "=== SpliceAI parquet check ==="
SPLICEAI_PATH="${SPLICEAI_PATH:-data/external/spliceai/spliceai_index.parquet}"
if [[ -f "$SPLICEAI_PATH" ]]; then
  SZ_MB=$(du -m "$SPLICEAI_PATH" | cut -f1)
  if [[ "$SZ_MB" -ge 300 ]]; then
    pass "SpliceAI parquet: $SPLICEAI_PATH (${SZ_MB} MB)"
  else
    fail "SpliceAI parquet at $SPLICEAI_PATH is only ${SZ_MB} MB (expected >= 300)"
  fi
else
  fail "SpliceAI parquet MISSING at $SPLICEAI_PATH -- GNN will silently return zeros"
fi

# -------------------------------------------------------------------------
# 3. STRING DB files
# -------------------------------------------------------------------------
echo "=== STRING DB check ==="
STRING_LINKS="${STRING_LINKS:-data/external/string/9606.protein.links.detailed.v12.0.txt.gz}"
STRING_INFO="${STRING_INFO:-data/external/string/9606.protein.info.v12.0.txt.gz}"

if [[ -f "$STRING_LINKS" ]]; then
  SZ=$(du -h "$STRING_LINKS" | cut -f1)
  pass "STRING DB links: $STRING_LINKS ($SZ)"
else
  fail "STRING DB links MISSING at $STRING_LINKS -- GNN cannot build graph"
fi

if [[ -f "$STRING_INFO" ]]; then
  SZ=$(du -h "$STRING_INFO" | cut -f1)
  pass "STRING DB info: $STRING_INFO ($SZ)"
else
  fail "STRING DB info MISSING at $STRING_INFO -- GNN cannot map symbols"
fi

# -------------------------------------------------------------------------
# 4. AlphaMissense
# -------------------------------------------------------------------------
echo "=== AlphaMissense check ==="
AM_PATH="${ALPHAMISSENSE_PATH:-data/external/alphamissense/AlphaMissense_hg38.tsv.gz}"
if [[ -f "$AM_PATH" ]]; then
  SZ=$(du -h "$AM_PATH" | cut -f1)
  pass "AlphaMissense: $AM_PATH ($SZ)"
else
  # Check for the parquet index variant
  AM_PQ="data/external/alphamissense/alphamissense_index.parquet"
  if [[ -f "$AM_PQ" ]]; then
    SZ=$(du -h "$AM_PQ" | cut -f1)
    pass "AlphaMissense (parquet): $AM_PQ ($SZ)"
  else
    fail "AlphaMissense MISSING at $AM_PATH or $AM_PQ"
  fi
fi

# -------------------------------------------------------------------------
# 5. ClinVar
# -------------------------------------------------------------------------
echo "=== ClinVar check ==="
CLINVAR_PATH="${CLINVAR_PATH:-data/raw/clinvar/clinvar_GRCh38.vcf.gz}"
if [[ -f "$CLINVAR_PATH" ]]; then
  SZ=$(du -h "$CLINVAR_PATH" | cut -f1)
  pass "ClinVar: $CLINVAR_PATH ($SZ)"
else
  fail "ClinVar MISSING at $CLINVAR_PATH -- no training labels"
fi

# -------------------------------------------------------------------------
# 6. Transformers installed (for ESM-2 activation)
# -------------------------------------------------------------------------
echo "=== Transformers check ==="
if python -c "import transformers; assert transformers.__version__ >= '4.40', f'got {transformers.__version__}'" 2>/dev/null; then
  TV=$(python -c "import transformers; print(transformers.__version__)")
  pass "transformers installed: $TV"
else
  fail "transformers not installed or version < 4.40 -- ESM-2 will stay in stub mode"
fi

# -------------------------------------------------------------------------
# 7. Git HEAD
# -------------------------------------------------------------------------
echo "=== Git HEAD check ==="
if git rev-parse HEAD >/dev/null 2>&1; then
  HEAD_SHA=$(git rev-parse --short HEAD)
  pass "Git HEAD on VM: $HEAD_SHA"
  # Sanity: warn if HEAD is old
  DAYS_OLD=$(git log -1 --format=%ct HEAD | xargs -I{} python3 -c "import time; print(int((time.time() - {}) / 86400))")
  if [[ "$DAYS_OLD" -gt 2 ]]; then
    warn "HEAD is $DAYS_OLD days old -- consider `git pull` before training"
  fi
else
  fail "not in a git repo -- cannot verify code state"
fi

# -------------------------------------------------------------------------
# 8. Critical Python imports
# -------------------------------------------------------------------------
echo "=== Python import check ==="
python - <<'EOF' 2>&1 | while read ln; do echo "    $ln"; done
import importlib, sys
required = [
    "pandas", "numpy", "torch", "sklearn", "xgboost", "lightgbm",
    "catboost", "torch_geometric", "transformers", "pyarrow",
    "networkx", "scipy",
]
failed = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception as e:
        failed.append((mod, str(e)[:80]))
if failed:
    for m, err in failed:
        print(f"MISSING: {m} ({err})")
    sys.exit(1)
print("ALL_IMPORTS_OK")
EOF

if python - <<'EOF' 2>/dev/null
import importlib
for mod in ["pandas", "torch", "xgboost", "lightgbm", "catboost",
            "torch_geometric", "transformers"]:
    importlib.import_module(mod)
EOF
then
  pass "all required Python modules importable"
else
  fail "at least one required Python module fails to import"
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "VM preflight: $PASS_COUNT pass, $FAIL_COUNT fail"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  echo "FAILURES:"
  for f in "${FAILURES[@]}"; do
    echo "  - $f"
  done
  echo ""
  echo "DO NOT start training. Fix failures first or the run will produce"
  echo "misleading results with silent-zero feature contributions."
  exit 1
fi
echo "All on-VM checks passed -- safe to start training."
exit 0