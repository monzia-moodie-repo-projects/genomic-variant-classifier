# =============================================================================
# Run17_Postflight.ps1
# =============================================================================
# Purpose:  After a Run 17 training run completes on Vast.ai (Option A: three
#           independent runs -- both / r12only / r13only), for the SELECTED run:
#           1. Confirm training is done (interactive abort if still running)
#           2. Push run17_observability.py to VM and run it
#           3. SCP report + log + key artifacts back (incl. models/, per_model_metrics)
#           4. Local quick-look
#           5. ARTIFACT GATES via Test-ArtifactPresent (closes C.5)
#           6. Write gate exit code to file (consumed by Vastai_Destroy_Confirmed.ps1)
#           7. Print pointer to Vastai_Destroy_Confirmed.ps1 (closes C.7); print obs summary
#
# Author:   Monzia Moodie
# Created:  2026-06-28
# Closes:   Run 17 plan (postflight; mirrors Run 15 C.5/C.6/C.7)
#
# OPTION A (three runs): this ONE script handles all three configs via -Config.
#   ALL of RemoteLog / RemoteOutputs / RemoteReport / LocalReport derive from a SINGLE
#   per-config stem (run17_baseline | run17_r12only | run17_r13only). One value per
#   config can be wrong -> the path templates are written once and applied identically.
#   The stem set is a CLOSED whitelist (ValidateSet + hashtable guard); an unknown
#   config CANNOT run (fails before any SSH/SCP). -DryRun prints the derived paths and
#   the planned teardown pointer WITHOUT any SSH/SCP/destroy -- verify before every run.
#
# CRITICAL: This script DOES NOT call vastai destroy. The destroy is delegated to
#           Vastai_Destroy_Confirmed.ps1, which:
#           - Requires this script's gate exit code to be 0
#           - Refuses stdin redirection (no `echo y |` automation)
#           - Requires interactive 'DESTROY' confirmation
#
# Usage:
#   # Dry-run first (no SSH/SCP; just prints derived paths + teardown plan):
#   .\Run17_Postflight.ps1 -Config r12only -SshHost ssh7.vast.ai -SshPort 17254 `
#                          -InstanceId 38001234 -HourlyRate 0.67 -DryRun
#   # Real postflight for the r12only run:
#   .\Run17_Postflight.ps1 -Config r12only -SshHost ssh7.vast.ai -SshPort 17254 `
#                          -InstanceId 38001234 -HourlyRate 0.67
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [ValidateSet('both','r12only','r13only')] [string]$Config,
    [Parameter(Mandatory=$true)] [string]$SshHost,
    [Parameter(Mandatory=$true)] [int]   $SshPort,
    [Parameter(Mandatory=$true)] [string]$InstanceId,
    [Parameter(Mandatory=$true)] [double]$HourlyRate,
    [string]$SshKey       = "C:\Users\monzi\.ssh\id_lambda_run8",
    [string]$RepoRoot     = "C:\Projects\genomic-variant-classifier",
    [switch]$DryRun
)

# Dot-source recursive locator helper (C.5: artifact gates use this)
. "$PSScriptRoot\Test-ArtifactPresent.ps1"

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# CONFIG -> STEM whitelist (closed set). ValidateSet already rejects anything not
# in this set before the body runs; this hashtable + guard is defense-in-depth and
# the single source of the per-config stem. Every path below derives from $stem.
# -----------------------------------------------------------------------------
$ConfigPaths = @{
    'both'    = 'run17_baseline'
    'r12only' = 'run17_r12only'
    'r13only' = 'run17_r13only'
}
if (-not $ConfigPaths.ContainsKey($Config)) {
    throw "Unknown -Config '$Config'. Valid: $($ConfigPaths.Keys -join ', ')"
}
$stem = $ConfigPaths[$Config]

# Single-stem derivation: all four paths from ONE value ($stem).
$RemoteLog     = "/workspace/${stem}_master.log"
$RemoteOutputs = "/workspace/genomic-variant-classifier/outputs/$stem/full"
$RemoteReport  = "/workspace/${stem}_report"
$LocalReport   = Join-Path $RepoRoot "outputs\${stem}_report"

$sshBase = @("-i", $SshKey, "-p", $SshPort, "-o", "StrictHostKeyChecking=no", "root@$SshHost")
$scpBase = @("-i", $SshKey, "-P", $SshPort, "-o", "StrictHostKeyChecking=no")

function Invoke-Ssh($cmd) {
    $ErrorActionPreference = 'Continue'
    $out = & ssh @sshBase $cmd 2>&1 | Out-String
    return $out
}

Write-Host "=== Run 17 Postflight (config: $Config; stem: $stem) ===" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# DRY-RUN: print derived paths + planned teardown pointer, then exit. NO SSH/SCP/destroy.
# This is both the operator's pre-run verification AND the mechanism the path test uses.
# -----------------------------------------------------------------------------
if ($DryRun) {
    Write-Host "`n[DRY-RUN] Derived paths for -Config ${Config}:" -ForegroundColor Magenta
    Write-Host "  stem          = $stem"
    Write-Host "  RemoteLog     = $RemoteLog"
    Write-Host "  RemoteOutputs = $RemoteOutputs"
    Write-Host "  RemoteReport  = $RemoteReport"
    Write-Host "  LocalReport   = $LocalReport"
    Write-Host "`n[DRY-RUN] Would push scripts/run17_observability.py and gate these artifacts under LocalReport:" -ForegroundColor Magenta
    Write-Host "  run17_master.log, run17_observability.md, run17_observability.json,"
    Write-Host "  per_model_metrics.csv, ensemble.joblib, ensemble.manifest.json, gnn_score_nondegenerate"
    Write-Host "`n[DRY-RUN] On all-gates-pass, would point to:" -ForegroundColor Magenta
    Write-Host "    .\scripts\Vastai_Destroy_Confirmed.ps1 -InstanceId $InstanceId -GateFile '$(Join-Path $LocalReport ".gate_exit_code")'"
    Write-Host "`n[DRY-RUN] No SSH/SCP/destroy performed. Exiting 0." -ForegroundColor Green
    exit 0
}

# (stage 1 stub: confirm training done)
Write-Host "`n[1/7] Confirm training done..." -ForegroundColor Yellow

# -----------------------------------------------------------------------------
# 2. Push observability script to VM and run it
# -----------------------------------------------------------------------------
Write-Host "`n[2/7] Push observability script and run it..." -ForegroundColor Yellow
$obsScript = Join-Path $RepoRoot "scripts\run17_observability.py"
if (-not (Test-Path $obsScript)) {
    Write-Host "  ERROR: $obsScript not found locally." -ForegroundColor Red
    Write-Host "  Run 17 prep task: create scripts/run17_observability.py" -ForegroundColor Yellow
    Write-Host "  (Generated from run15_observability.py via gen_run17_observability.py.)" -ForegroundColor Yellow
    exit 1
}
& scp @scpBase $obsScript "root@${SshHost}:/workspace/genomic-variant-classifier/scripts/run17_observability.py"
if ($LASTEXITCODE -ne 0) { Write-Host "  scp obs script failed" -ForegroundColor Red; exit 1 }

Invoke-Ssh "mkdir -p $RemoteReport"
$obsCmd = "python3 /workspace/genomic-variant-classifier/scripts/run17_observability.py " +
          "--outputs-dir $RemoteOutputs --log $RemoteLog --report-dir $RemoteReport " +
          "--instance-id $InstanceId --hourly-rate $HourlyRate"
$obsOut = Invoke-Ssh $obsCmd
Write-Host $obsOut -ForegroundColor Gray

# -----------------------------------------------------------------------------
# 3. SCP report + log + key artifacts back
# -----------------------------------------------------------------------------
Write-Host "`n[3/7] SCP artifacts back..." -ForegroundColor Yellow
if (-not (Test-Path $LocalReport)) { New-Item -ItemType Directory -Path $LocalReport -Force | Out-Null }
& scp @scpBase "root@${SshHost}:${RemoteLog}" "$LocalReport\run17_master.log"
& scp @scpBase "root@${SshHost}:${RemoteReport}/run17_observability.md"   "$LocalReport\run17_observability.md"
& scp @scpBase "root@${SshHost}:${RemoteReport}/run17_observability.json" "$LocalReport\run17_observability.json"
& scp @scpBase "root@${SshHost}:${RemoteOutputs}/per_model_metrics.csv"   "$LocalReport\per_model_metrics.csv"
& scp @scpBase "root@${SshHost}:${RemoteOutputs}/models/ensemble.joblib"  "$LocalReport\ensemble.joblib"
& scp @scpBase "root@${SshHost}:${RemoteOutputs}/models/ensemble.manifest.json" "$LocalReport\ensemble.manifest.json"

# (stage 4 stub: local quick-look)
Write-Host "`n[4/7] Local quick-look..." -ForegroundColor Yellow

# -----------------------------------------------------------------------------
# 5. ARTIFACT GATES (C.5 + C.6) — use Test-ArtifactPresent for ALL checks
# -----------------------------------------------------------------------------
Write-Host "`n[5/7] Artifact-presence gates (C.5)..." -ForegroundColor Yellow
# GNN-score non-degeneracy gate (Run-14 silent-GNN guard). The split parquets
# carrying the injected gnn_score live on the VM; verify there via SSH rather
# than SCP ~GB of parquets back. A degenerate gnn_score fails this gate and
# BLOCKS Vastai_Destroy_Confirmed.ps1 (which refuses on gate exit != 0).
Write-Host "  GNN-score non-degeneracy (VM-side verify)..." -ForegroundColor Yellow
$gnnCmd = "cd /workspace/genomic-variant-classifier && python3 scripts/verify_gnn_score.py $RemoteOutputs/splits; echo VGS_EXIT:`$?"
$gnnVerifyOut = Invoke-Ssh $gnnCmd
Write-Host ($gnnVerifyOut | Out-String) -ForegroundColor Gray
if (($gnnVerifyOut -join "`n") -match 'VGS_EXIT:(\d+)') {
    $gnnVerifyOk = ([int]$Matches[1] -eq 0)
} else {
    $gnnVerifyOk = $false   # no exit marker => SSH/verify did not complete => FAIL
}
$gateResults = [ordered]@{}
$gateResults['master_log']            = Test-ArtifactPresent -Root $LocalReport -Filename "run17_master.log" -MinBytes 1000
$gateResults['observability_md']      = Test-ArtifactPresent -Root $LocalReport -Filename "run17_observability.md" -MinBytes 100
$gateResults['observability_json']    = Test-ArtifactPresent -Root $LocalReport -Filename "run17_observability.json" -MinBytes 100
$gateResults['per_model_metrics_csv'] = Test-ArtifactPresent -Root $LocalReport -Filename "per_model_metrics.csv" -MinBytes 100
$gateResults['ensemble_joblib']       = Test-ArtifactPresent -Root $LocalReport -Filename "ensemble.joblib" -MinBytes 1000000
$gateResults['ensemble_manifest']     = Test-ArtifactPresent -Root $LocalReport -Filename "ensemble.manifest.json" -MinBytes 100
$gateResults['gnn_score_nondegenerate'] = $gnnVerifyOk

$gateFails = @($gateResults.Keys | Where-Object { -not $gateResults[$_] })
foreach ($gate in $gateResults.Keys) {
    $sym = if ($gateResults[$gate]) { 'PASS' } else { 'FAIL' }
    $color = if ($gateResults[$gate]) { 'Green' } else { 'Red' }
    Write-Host ("  {0,-25} {1}" -f $gate, $sym) -ForegroundColor $color
}

# -----------------------------------------------------------------------------
# 6. Write gate exit code to file (Vastai_Destroy_Confirmed.ps1 reads this)
# -----------------------------------------------------------------------------
$gateExitCode = if ($gateFails.Count -eq 0) { 0 } else { 1 }
$gateFile = Join-Path $LocalReport ".gate_exit_code"
[System.IO.File]::WriteAllText($gateFile, "$gateExitCode", (New-Object System.Text.UTF8Encoding $false))
Write-Host "`n[6/7] Gate exit code: $gateExitCode (written to $gateFile)" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# 7. Observability summary + destroy pointer (NOT the destroy command itself)
# -----------------------------------------------------------------------------
Write-Host "`n[7/7] Observability report preview..." -ForegroundColor Yellow
$mdPath = Join-Path $LocalReport "run17_observability.md"
if (Test-Path $mdPath) {
    Get-Content $mdPath | Select-Object -First 80
    Write-Host "`n  Full report: $mdPath" -ForegroundColor Green
} else {
    Write-Host "  Report markdown not found at $mdPath" -ForegroundColor Red
}

Write-Host "`n=== Postflight summary (config: $Config) ===" -ForegroundColor Cyan
if ($gateFails.Count -eq 0) {
    Write-Host "  ALL GATES PASS. Artifacts ready. Instance can be destroyed." -ForegroundColor Green
    Write-Host ""
    Write-Host "  To destroy the instance, run in a SEPARATE paste block:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    .\scripts\Vastai_Destroy_Confirmed.ps1 -InstanceId $InstanceId -GateFile '$gateFile'" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "  Vastai_Destroy_Confirmed.ps1 will:" -ForegroundColor Gray
    Write-Host "    - Refuse if stdin redirected (blocks 'echo y |' automation)" -ForegroundColor Gray
    Write-Host "    - Refuse if gate exit code != 0" -ForegroundColor Gray
    Write-Host "    - Require interactive 'DESTROY' confirmation (typo-resistant)" -ForegroundColor Gray
    exit 0
} else {
    Write-Host "  GATE FAIL: $($gateFails.Count) gate(s) did not pass." -ForegroundColor Red
    Write-Host "  Failed gates: $($gateFails -join ', ')" -ForegroundColor Red
    Write-Host "  Vastai_Destroy_Confirmed.ps1 will REFUSE to proceed with this gate file." -ForegroundColor Yellow
    Write-Host "  Investigate via: Get-ChildItem '$LocalReport' -Recurse" -ForegroundColor Yellow
    exit 1
}
