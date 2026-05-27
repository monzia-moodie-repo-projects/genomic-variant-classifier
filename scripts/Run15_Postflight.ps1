# =============================================================================
# Run15_Postflight.ps1
# =============================================================================
# Purpose:  After Run 15 training completes on Vast.ai:
#           1. Confirm training is done (interactive abort if still running)
#           2. Push run15_observability.py to VM and run it
#           3. SCP report + log + key artifacts back (incl. models/, per_model_metrics)
#           4. Local quick-look
#           5. ARTIFACT GATES via Test-ArtifactPresent (closes C.5)
#           6. Write gate exit code to file (consumed by Vastai_Destroy_Confirmed.ps1)
#           7. Print pointer to Vastai_Destroy_Confirmed.ps1 (closes C.7); print obs summary
#
# Author:   Monzia Moodie
# Created:  2026-05-27
# Closes:   Run 15 plan C.5, C.6, C.7
#
# CRITICAL: This script DOES NOT call vastai destroy. The destroy is delegated to
#           Vastai_Destroy_Confirmed.ps1, which:
#           - Requires this script's gate exit code to be 0
#           - Refuses stdin redirection (no `echo y |` automation)
#           - Requires interactive 'DESTROY' confirmation
#
# Usage:
#   .\Run15_Postflight.ps1 -SshHost ssh7.vast.ai -SshPort 17254 `
#                          -InstanceId 38001234 -HourlyRate 0.67
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string]$SshHost,
    [Parameter(Mandatory=$true)] [int]   $SshPort,
    [Parameter(Mandatory=$true)] [string]$InstanceId,
    [Parameter(Mandatory=$true)] [double]$HourlyRate,
    [string]$SshKey       = "C:\Users\monzi\.ssh\id_lambda_run8",
    [string]$RepoRoot     = "C:\Projects\genomic-variant-classifier",
    [string]$RemoteLog    = "/workspace/run15_master.log",
    [string]$RemoteOutputs= "/workspace/genomic-variant-classifier/outputs/run15_fresh",
    [string]$RemoteReport = "/workspace/run15_report",
    [string]$LocalReport  = ""
)

# Dot-source recursive locator helper (C.5: artifact gates use this)
. "$PSScriptRoot\Test-ArtifactPresent.ps1"

$ErrorActionPreference = "Stop"
if (-not $LocalReport) { $LocalReport = Join-Path $RepoRoot "outputs\run15_report" }

$sshBase = @("-i", $SshKey, "-p", $SshPort, "-o", "StrictHostKeyChecking=no", "root@$SshHost")
$scpBase = @("-i", $SshKey, "-P", $SshPort, "-o", "StrictHostKeyChecking=no")

function Invoke-Ssh($cmd) { & ssh @sshBase $cmd 2>&1 }

Write-Host "=== Run 15 Postflight ===" -ForegroundColor Cyan
Write-Host "Target: ${SshHost}:${SshPort}, Instance $InstanceId, ${HourlyRate}/hr" -ForegroundColor Gray
Write-Host ""

# -----------------------------------------------------------------------------
# 1. Confirm training actually finished (not a mid-run pull)
# -----------------------------------------------------------------------------
Write-Host "[1/7] Confirm training completion..." -ForegroundColor Yellow
$pythonRunning = Invoke-Ssh "pgrep -fa 'train.py|run_phase2_eval.py' || echo NONE"
if ($pythonRunning -notmatch "NONE") {
    Write-Host "  WARNING: Training process is still running:" -ForegroundColor Yellow
    Write-Host "  $pythonRunning" -ForegroundColor Yellow
    $confirm = Read-Host "  Continue anyway? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Aborted. Wait for training to finish before re-running postflight." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  No python training process running. OK to proceed." -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# 2. Push observability script to VM and run it
# -----------------------------------------------------------------------------
Write-Host "`n[2/7] Push observability script and run it..." -ForegroundColor Yellow
$obsScript = Join-Path $RepoRoot "scripts\run15_observability.py"
if (-not (Test-Path $obsScript)) {
    Write-Host "  ERROR: $obsScript not found locally." -ForegroundColor Red
    Write-Host "  Run 15 prep task: create scripts/run15_observability.py" -ForegroundColor Yellow
    Write-Host "  (Copy from run14_observability.py and adapt paths/run id.)" -ForegroundColor Yellow
    exit 1
}
& scp @scpBase $obsScript "root@${SshHost}:/workspace/genomic-variant-classifier/scripts/run15_observability.py"
if ($LASTEXITCODE -ne 0) { Write-Host "  scp obs script failed" -ForegroundColor Red; exit 1 }

Invoke-Ssh "mkdir -p $RemoteReport"
$obsCmd = "python3 /workspace/genomic-variant-classifier/scripts/run15_observability.py " +
          "--outputs-dir $RemoteOutputs --log $RemoteLog --report-dir $RemoteReport " +
          "--instance-id $InstanceId --hourly-rate $HourlyRate"
$obsOut = Invoke-Ssh $obsCmd
Write-Host $obsOut -ForegroundColor Gray

# -----------------------------------------------------------------------------
# 3. SCP report + log + key artifacts back
# -----------------------------------------------------------------------------
Write-Host "`n[3/7] SCP artifacts back to $LocalReport ..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $LocalReport | Out-Null

# 3a. Observability report
& scp @scpBase -r "root@${SshHost}:${RemoteReport}/*" "$LocalReport\"
if ($LASTEXITCODE -ne 0) { Write-Host "  scp report failed" -ForegroundColor Red; exit 1 }

# 3b. Master log
& scp @scpBase "root@${SshHost}:${RemoteLog}" "$LocalReport\run15_master.log"

# 3c. Per-model OOF predictions
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/oof" "$LocalReport\oof" 2>$null

# 3d. Test-set predictions + metrics
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/test_eval" "$LocalReport\test_eval" 2>$null

# 3e. Feature importance / blend weights
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/blend_weights.json" "$LocalReport\" 2>$null
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/feature_importance.csv" "$LocalReport\" 2>$null

# 3f. NEW (Run 14 oversight fix): SCP models/ directory — contains ensemble.joblib
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/models" "$LocalReport\models" 2>$null

# 3g. NEW (A7 fix support): SCP per_model_metrics CSVs added by Run 14 observability rewrite
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/per_model_metrics.csv" "$LocalReport\" 2>$null
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/per_model_metrics_val.csv" "$LocalReport\" 2>$null

Write-Host "  Done." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 4. Local quick-look
# -----------------------------------------------------------------------------
Write-Host "`n[4/7] Local artifact summary..." -ForegroundColor Yellow
Get-ChildItem $LocalReport -Recurse -File |
    Select FullName, @{N='SizeMB';E={[math]::Round($_.Length/1MB, 2)}} |
    Sort-Object SizeMB -Descending | Format-Table -AutoSize

# -----------------------------------------------------------------------------
# 5. ARTIFACT GATES (C.5 + C.6) — use Test-ArtifactPresent for ALL checks
# -----------------------------------------------------------------------------
Write-Host "`n[5/7] Artifact-presence gates (C.5)..." -ForegroundColor Yellow
$gateResults = [ordered]@{}
$gateResults['master_log']            = Test-ArtifactPresent -Root $LocalReport -Filename "run15_master.log" -MinBytes 1000
$gateResults['observability_md']      = Test-ArtifactPresent -Root $LocalReport -Filename "run15_observability.md" -MinBytes 100
$gateResults['observability_json']    = Test-ArtifactPresent -Root $LocalReport -Filename "run15_observability.json" -MinBytes 100
$gateResults['per_model_metrics_csv'] = Test-ArtifactPresent -Root $LocalReport -Filename "per_model_metrics.csv" -MinBytes 100
$gateResults['ensemble_joblib']       = Test-ArtifactPresent -Root $LocalReport -Filename "ensemble.joblib" -MinBytes 1000000
$gateResults['ensemble_manifest']     = Test-ArtifactPresent -Root $LocalReport -Filename "ensemble.manifest.json" -MinBytes 100
$gateResults['blend_weights']         = Test-ArtifactPresent -Root $LocalReport -Filename "blend_weights.json" -MinBytes 50

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
$mdPath = Join-Path $LocalReport "run15_observability.md"
if (Test-Path $mdPath) {
    Get-Content $mdPath | Select-Object -First 80
    Write-Host "`n  Full report: $mdPath" -ForegroundColor Green
} else {
    Write-Host "  Report markdown not found at $mdPath" -ForegroundColor Red
}

Write-Host "`n=== Postflight summary ===" -ForegroundColor Cyan
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