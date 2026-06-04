#!/usr/bin/env python3
"""Wire verify_gnn_score.py into launch_run15_baseline.sh's post-run block (VM-side,
pre-destroy). Makes the script's existing 'validates gnn_score std > 0' contract real:
runs the non-degeneracy check on $OUTDIR/splits, logs a loud banner and writes an
$OUTDIR/GNN_VERIFY_FAILED sentinel on failure (postflight can Test-Path it). Does NOT
alter RUN_RC. Inserts before the final 'NEXT (laptop ...)' echo. Count-guarded,
idempotent, line-ending agnostic (preserves LF for the .sh)."""
from __future__ import annotations
import shutil, sys
from pathlib import Path

MARKER = "verify_gnn_score.py"
ANCHOR = 'echo "==> NEXT (laptop, SEPARATE paste blocks): Run15_Postflight.ps1 -> Vastai_Destroy_Confirmed.ps1" | tee -a "$LOG"\n'
BLOCK = (
    "# -- GNN-score non-degeneracy gate (makes the 'validates gnn_score std > 0'\n"
    "# -- contract real; catches a GNN swallowed by run_phase2_eval's except\n"
    "# -- BEFORE the instance is destroyed) ----------------------------------------\n"
    'echo "==> [post] GNN-score non-degeneracy gate" | tee -a "$LOG"\n'
    "set +e\n"
    'python scripts/verify_gnn_score.py "$OUTDIR/splits" 2>&1 | tee -a "$LOG"\n'
    "GNN_VERIFY_RC=${PIPESTATUS[0]}\n"
    "set -e\n"
    'if [ "$GNN_VERIFY_RC" -eq 0 ]; then\n'
    '    echo "==> gnn_score: OK (non-degenerate)" | tee -a "$LOG"\n'
    '    rm -f "$OUTDIR/GNN_VERIFY_FAILED" 2>/dev/null || true\n'
    "else\n"
    '    echo "==> ############################################################" | tee -a "$LOG"\n'
    '    echo "==> ## gnn_score DEGENERATE (verify rc=$GNN_VERIFY_RC): the GNN was" | tee -a "$LOG"\n'
    '    echo "==> ## swallowed; do NOT trust this run GNN contribution. Inspect" | tee -a "$LOG"\n'
    '    echo "==> ## [GNN-TRACE] lines above BEFORE destroying the instance." | tee -a "$LOG"\n'
    '    echo "==> ############################################################" | tee -a "$LOG"\n'
    "    echo \"gnn_score degenerate; verify rc=$GNN_VERIFY_RC; $(date -u +'%F %T') UTC\" > \"$OUTDIR/GNN_VERIFY_FAILED\"\n"
    "fi\n"
)

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already wires verify_gnn_score (idempotent)"); return 0
    if data.count(ANCHOR) != 1:
        print(f"ABORT: anchor count={data.count(ANCHOR)} (want 1); no change"); return 2
    out = data.replace(ANCHOR, BLOCK + ANCHOR, 1)
    final = out.replace("\n", nl) if nl == "\r\n" else out
    shutil.copy2(path, path.with_suffix(path.suffix + ".gnnverify.bak"))
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path}; endings={'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scripts/launch_run15_baseline.sh"))
