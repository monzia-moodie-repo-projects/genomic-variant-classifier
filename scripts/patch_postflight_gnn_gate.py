#!/usr/bin/env python3
"""Wire a VM-side gnn_score non-degeneracy gate into Run15_Postflight.ps1.
Postflight does NOT SCP the split parquets back, so we verify on the VM via SSH
(splits live there, instance still alive) and feed the verdict into $gateResults,
which drives .gate_exit_code -> Vastai_Destroy_Confirmed.ps1 (refuses on != 0).
A degenerate gnn_score (GNN swallowed by run_phase2_eval's except) thus BLOCKS
destroy. Two insertions, count-guarded, idempotent, line-ending agnostic."""
from __future__ import annotations
import shutil, sys
from pathlib import Path

MARKER = "gnn_score_nondegenerate"

ANCHOR1 = "$gateResults = [ordered]@{}\n"
BLOCK1 = (
    "# GNN-score non-degeneracy gate (Run-14 silent-GNN guard). The split parquets\n"
    "# carrying the injected gnn_score live on the VM; verify there via SSH rather\n"
    "# than SCP ~GB of parquets back. A degenerate gnn_score fails this gate and\n"
    "# BLOCKS Vastai_Destroy_Confirmed.ps1 (which refuses on gate exit != 0).\n"
    'Write-Host "  GNN-score non-degeneracy (VM-side verify)..." -ForegroundColor Yellow\n'
    '$gnnCmd = "cd /workspace/genomic-variant-classifier && python3 scripts/verify_gnn_score.py $RemoteOutputs/splits; echo VGS_EXIT:`$?"\n'
    "$gnnVerifyOut = Invoke-Ssh $gnnCmd\n"
    'Write-Host ($gnnVerifyOut | Out-String) -ForegroundColor Gray\n'
    'if (($gnnVerifyOut -join "`n") -match \'VGS_EXIT:(\\d+)\') {\n'
    "    $gnnVerifyOk = ([int]$Matches[1] -eq 0)\n"
    "} else {\n"
    "    $gnnVerifyOk = $false   # no exit marker => SSH/verify did not complete => FAIL\n"
    "}\n"
    "$gateResults = [ordered]@{}\n"
)

ANCHOR2 = '$gateResults[\'blend_weights\']         = Test-ArtifactPresent -Root $LocalReport -Filename "blend_weights.json" -MinBytes 50\n'
BLOCK2 = (
    ANCHOR2
    + "$gateResults['gnn_score_nondegenerate'] = $gnnVerifyOk\n"
)

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already has the gnn gate (idempotent)"); return 0
    for label, anc in [("anchor1", ANCHOR1), ("anchor2", ANCHOR2)]:
        if data.count(anc) != 1:
            print(f"ABORT: {label} count={data.count(anc)} (want 1); no change"); return 2
    out = data.replace(ANCHOR1, BLOCK1, 1).replace(ANCHOR2, BLOCK2, 1)
    final = out.replace("\n", nl) if nl == "\r\n" else out
    shutil.copy2(path, path.with_suffix(path.suffix + ".gnngate.bak"))
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path}; endings={'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scripts/Run15_Postflight.ps1"))
