#!/usr/bin/env bash
# =====================================================================
# vm_bootstrap_run.sh  --  GenAssoc fresh-box environment bootstrap + gate
# Version 1.0 (2026-07-06).  Run FIRST on any fresh Vast.ai / cloud GPU box,
# BEFORE the data pull, the preflight, or the run.
# ---------------------------------------------------------------------
# PURPOSE
#   Every GenAssoc run (1-17) has failed at START on an environment problem that
#   was predictable and preventable: missing system tools, a base-image Python
#   stack that silently drifts from requirements.lock, or an unpinned model
#   revision that breaks on any cache but the developer's. This script makes that
#   whole class of failure structurally impossible: it bootstraps the box in a
#   fixed, ordered, IDEMPOTENT, FAIL-LOUD sequence and REFUSES to hand off to the
#   run until every failure mode observed in prior runs is verified clean.
#
# DOCTRINE (why each phase exists -- keyed to the run that taught it)
#   A. System tools      -- fresh images lack unzip/curl/git-lfs; rclone install
#                           itself failed for want of unzip (Run 17 smoke).
#   B. rclone + remote    -- binary absent on base image; conf must be present and
#                           the remote must actually LIST before trusting it (R17).
#   C. Python stack pin   -- imodelsx/etc. drag pandas 3.0, transformers 5.13 OVER
#                           the pinned stack; pandas 2->3 and the NT/transformers
#                           break are silent until deep in prep (R17). Pin to LOCK.
#   D. Model-load proof   -- ESM-2 + Nucleotide Transformer must LOAD on THIS box's
#                           fresh HF cache before the run, with a PINNED revision,
#                           or NT fetches head-of-main remote code that imports a
#                           symbol transformers removed (R17, the smoke-killer).
#   E. Import + GPU gate   -- torch.cuda, torch_geometric, imodelsx, KANClassifier,
#                           and the project package must all import (R17 preflight).
#   F. Handoff            -- only if A-E are GREEN does it print PROCEED. Anything
#                           red => nonzero exit, explicit remediation, NO run.
#
# DESIGN RULES (the standing project disciplines, enforced in code)
#   * Verify FIRST, from ground truth: versions are read from requirements.lock at
#     runtime, never hardcoded here, so this script cannot itself go stale.
#   * Nothing fails silently: every check prints PASS/FAIL; any FAIL aborts.
#   * Idempotent: safe to re-run; installs are pinned + skip-if-satisfied.
#   * No patchwork: the NT fix is a real revision pin passed to the loader via env,
#     not a monkeypatch of transformers.
#
# USAGE (on the VM, from the repo root)
#   cd /workspace/genomic-variant-classifier
#   bash scripts/vm_bootstrap_run.sh 2>&1 | tee /workspace/vm_bootstrap.txt
#   # exit 0 => PROCEED to vm_pull_run17_data.sh, then Run_Preflight_VM.sh, then launch
#
# ENV OVERRIDES
#   REPO           (default: the current dir)
#   LOCK           (default: $REPO/requirements.lock)
#   NT_REVISION    NT model commit to pin (STRONGLY set this -- see Phase D)
#   PY             python interpreter (default: autodetect conda/venv)
#   SKIP_MODEL_PROBE=1  skip Phase D model loads (NOT recommended)
# =====================================================================

set -uo pipefail

REPO="${REPO:-$(pwd)}"
LOCK="${LOCK:-$REPO/requirements.lock}"
PY="${PY:-$(command -v python || command -v python3)}"
# The NT revision that is (a) compatible with the pinned transformers and (b) what
# the developer's working local cache uses. MUST be filled in -- see companion doc
# RUN_BOOTSTRAP_DOCTRINE.md section "Pinning the NT revision". Left blank => Phase D
# will FAIL LOUD rather than silently pull head-of-main (the Run-17 bug).
NT_REVISION="${NT_REVISION:-}"

C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YLW=$'\033[33m'; C_RST=$'\033[0m'
PASS=0; FAIL=0; FAILED_ITEMS=()
ok()   { printf "%s[PASS]%s %s\n" "$C_GRN" "$C_RST" "$1"; PASS=$((PASS+1)); }
bad()  { printf "%s[FAIL]%s %s\n" "$C_RED" "$C_RST" "$1"; FAIL=$((FAIL+1)); FAILED_ITEMS+=("$1"); }
warn() { printf "%s[WARN]%s %s\n" "$C_YLW" "$C_RST" "$1"; }
hdr()  { printf "\n=== %s ===\n" "$1"; }

echo "############ GenAssoc VM BOOTSTRAP ($(date -u +%FT%TZ)) ############"
echo "REPO=$REPO  LOCK=$LOCK  PY=$PY"
[ -f "$LOCK" ] || { echo "${C_RED}FATAL: requirements.lock not found at $LOCK -- is the repo extracted?${C_RST}"; exit 2; }
[ -n "$PY" ]   || { echo "${C_RED}FATAL: no python interpreter found on PATH.${C_RST}"; exit 2; }

# helper: read an exact pin from requirements.lock (handles 'pkg==x.y.z \' lines and extras like pandas[parquet])
lockver() {  # $1 = distribution name (lowercase)
  grep -iE "^\s*$1(\[[^]]*\])?==" "$LOCK" | head -n1 | sed -E 's/^[^=]*==//; s/[ \\]+$//'
}

# =====================================================================
hdr "A. SYSTEM TOOLS (fresh images lack these; rclone install needs unzip)"
# apt is present on the pytorch/ubuntu base images. Install quietly, idempotent.
NEED_APT=(unzip curl ca-certificates git)
MISSING_APT=()
for t in "${NEED_APT[@]}"; do command -v "$t" >/dev/null 2>&1 || MISSING_APT+=("$t"); done
if [ "${#MISSING_APT[@]}" -gt 0 ]; then
  echo "installing: ${MISSING_APT[*]}"
  apt-get update -qq && apt-get install -y -qq "${MISSING_APT[@]}" >/dev/null 2>&1 || warn "apt install had issues -- continuing to verify"
fi
for t in unzip curl git; do
  if command -v "$t" >/dev/null 2>&1; then ok "system tool present: $t"; else bad "system tool MISSING: $t (apt-get install -y $t)"; fi
done

# =====================================================================
hdr "B. RCLONE + DRIVE REMOTE (binary + working genvarcla: listing)"
if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found -- installing (python-unzip path, no dependency on system unzip)"
  cd /tmp
  curl -fsSL -O https://downloads.rclone.org/rclone-current-linux-amd64.zip || warn "rclone download failed"
  "$PY" -c "import zipfile; zipfile.ZipFile('rclone-current-linux-amd64.zip').extractall('.')" 2>/dev/null \
    || (command -v unzip >/dev/null 2>&1 && unzip -oq rclone-current-linux-amd64.zip)
  cp rclone-*-linux-amd64/rclone /usr/local/bin/ && chmod +x /usr/local/bin/rclone
  cd "$REPO"
fi
if command -v rclone >/dev/null 2>&1; then ok "rclone installed: $(rclone version 2>/dev/null | head -1)"; else bad "rclone STILL missing -- install manually"; fi
# remote config present?
if [ -f /root/.config/rclone/rclone.conf ]; then ok "rclone.conf present"; else bad "rclone.conf MISSING at /root/.config/rclone/ (scp it up)"; fi
# remote actually LISTS (proves the OAuth token works on THIS box) -- cheap, decisive
if command -v rclone >/dev/null 2>&1 && rclone lsf genvarcla:genomic-variant-classifier/data/external/ >/dev/null 2>&1; then
  ok "genvarcla: remote lists data/external/ (Drive reachable + token valid)"
else
  bad "genvarcla: remote does NOT list -- token/config problem; fix before data pull"
fi

# =====================================================================
hdr "C. PYTHON STACK PINNED TO requirements.lock (kill silent drift)"
# The exact libs that (a) other installs love to drift and (b) break the run when they do.
# Read each pin from the lock; install --no-deps at the pinned version if mismatched.
STACK=(numpy pandas scikit-learn scipy pyarrow xgboost lightgbm catboost transformers tokenizers huggingface-hub)
declare -A INSTALL_AT
for pkg in "${STACK[@]}"; do
  want="$(lockver "$pkg")"
  if [ -z "$want" ]; then warn "no lock pin found for $pkg -- skipping (verify manually)"; continue; fi
  have="$("$PY" -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null || echo "MISSING")"
  if [ "$have" = "$want" ]; then
    ok "$pkg == $want (matches lock)"
  else
    warn "$pkg is $have, lock wants $want -- will pin"
    INSTALL_AT["$pkg"]="$want"
  fi
done
if [ "${#INSTALL_AT[@]}" -gt 0 ]; then
  SPECS=(); for pkg in "${!INSTALL_AT[@]}"; do SPECS+=("$pkg==${INSTALL_AT[$pkg]}"); done
  echo "pinning to lock (--no-deps, surgical -- won't re-drift the rest): ${SPECS[*]}"
  "$PY" -m pip install --no-deps -q "${SPECS[@]}" || bad "pip pin install failed -- see output"
  # re-verify after install
  for pkg in "${!INSTALL_AT[@]}"; do
    have="$("$PY" -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null || echo MISSING)"
    if [ "$have" = "${INSTALL_AT[$pkg]}" ]; then ok "$pkg pinned -> $have"; else bad "$pkg still $have (wanted ${INSTALL_AT[$pkg]})"; fi
  done
fi
# ensure GNN/KAN deps + project package are installed (present-or-install)
"$PY" -c "import torch_geometric, networkx" 2>/dev/null && ok "torch_geometric + networkx import" || {
  warn "installing torch_geometric + networkx (match torch $("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null))"
  "$PY" -m pip install -q networkx torch_geometric || bad "torch_geometric/networkx install failed"
}
"$PY" -c "import imodelsx.kan.kan_sklearn" 2>/dev/null && ok "imodelsx import" || {
  warn "installing imodelsx (NOTE: pulls deps that drift the stack -- re-pin runs next)"
  "$PY" -m pip install -q imodelsx || bad "imodelsx install failed"
  # imodelsx notoriously re-drifts pandas/transformers -- re-pin the critical few immediately
  RP=(); for pkg in pandas transformers tokenizers huggingface-hub scikit-learn; do v="$(lockver "$pkg")"; [ -n "$v" ] && RP+=("$pkg==$v"); done
  echo "re-pinning post-imodelsx (the exact drift that killed Run 17): ${RP[*]}"
  "$PY" -m pip install --no-deps -q "${RP[@]}" || bad "post-imodelsx re-pin failed"
}
# project package importable?
"$PY" -c "import genomic_variant_classifier" 2>/dev/null && ok "genomic_variant_classifier importable" || {
  warn "installing project package editable (pip install -e . --no-deps)"
  ( cd "$REPO" && "$PY" -m pip install -e . --no-deps -q ) || bad "project package install failed"
}

# =====================================================================
hdr "D. MODEL-LOAD PROOF on THIS box (ESM-2 + NT, pinned revision) -- the Run-17 killer"
if [ "${SKIP_MODEL_PROBE:-0}" = "1" ]; then
  warn "SKIP_MODEL_PROBE=1 -- skipping model loads (NOT recommended; this is the exact gate Run 17 needed)"
else
  # ESM-2: uses transformers' BUILT-IN class -> loads on any recent transformers; quick sanity.
  if "$PY" - <<'PYEOF' 2>/tmp/esm2_probe.err
from transformers import AutoTokenizer, AutoModelForMaskedLM
m="facebook/esm2_t6_8M_UR50D"
AutoTokenizer.from_pretrained(m); AutoModelForMaskedLM.from_pretrained(m)
print("ESM2_OK")
PYEOF
  then ok "ESM-2 loads on this box"; else bad "ESM-2 FAILED to load -- see /tmp/esm2_probe.err"; fi

  # Nucleotide Transformer: uses REMOTE code (trust_remote_code). Head-of-main pulls a
  # modeling_esm.py that imports find_pruneable_heads_and_indices, REMOVED from the pinned
  # transformers -> ImportError. THE FIX is to pin revision=. Refuse to proceed unpinned.
  if [ -z "$NT_REVISION" ]; then
    bad "NT_REVISION is NOT set -- NT would pull head-of-main remote code and break (the Run-17 bug). Set NT_REVISION to a commit compatible with transformers $(lockver transformers). See RUN_BOOTSTRAP_DOCTRINE.md."
  else
    if "$PY" - "$NT_REVISION" <<'PYEOF' 2>/tmp/nt_probe.err
import sys
from transformers import AutoModelForMaskedLM, AutoTokenizer
rev=sys.argv[1]; m="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
AutoTokenizer.from_pretrained(m, trust_remote_code=True, revision=rev)
AutoModelForMaskedLM.from_pretrained(m, trust_remote_code=True, revision=rev)
print("NT_OK")
PYEOF
    then ok "NT loads on this box at pinned revision $NT_REVISION"
    else bad "NT FAILED at revision $NT_REVISION -- try a different compatible commit; see /tmp/nt_probe.err"; fi
    # remind that the code should ALSO pin this, not just this bootstrap
    warn "PERMANENT FIX: pin revision='$NT_REVISION' in genomic_lm.py from_pretrained() calls so the code is reproducible without this env var."
  fi
fi

# =====================================================================
hdr "E. IMPORT + GPU GATE (fast fail before any run)"
"$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && ok "torch.cuda.is_available()" || bad "CUDA not available to torch"
"$PY" -c "from genomic_variant_classifier.models.kan import KANClassifier" 2>/dev/null && ok "KANClassifier import" || bad "KANClassifier import failed"

# =====================================================================
hdr "F. HANDOFF"
echo "------------------------------------------------------------------------"
echo "bootstrap: $PASS pass, $FAIL fail"
if [ "$FAIL" -gt 0 ]; then
  echo "${C_RED}BLOCKED -- fix these before proceeding:${C_RST}"
  for f in "${FAILED_ITEMS[@]}"; do echo "  - $f"; done
  exit 1
fi
echo "${C_GRN}BOOTSTRAP GREEN.${C_RST} Environment matches lock, models load, GPU ready."
echo "PROCEED:  bash /workspace/vm_pull_run17_data.sh   # data staging (verified, fail-loud)"
echo "THEN:     MIN_DISK_GB=80 bash scripts/Run_Preflight_VM.sh"
echo "THEN:     bash scripts/launch_run17_baseline.sh   # the real run"
