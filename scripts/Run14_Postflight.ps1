# =============================================================================
# Run14_Postflight.ps1
# =============================================================================
# Purpose:  After Run 14 training completes on Vast.ai, this script:
#           1. Triggers run14_observability.py on the VM
#           2. SCPs the report + run artifacts back
#           3. Pre-stages everything for local commit
#           4. Prints the EXACT vastai destroy command (run it MANUALLY after
#              verifying artifacts are local)
#
# Author:   Claude, for Monzia Moodie
# Created:  2026-05-26
#
# CRITICAL: This script DOES NOT call vastai destroy. The destroy command is
#           emitted as the last line of output. You re-paste it in a SEPARATE
#           code block after verifying artifacts are local. Standing rule:
#           irreversible cloud commands NEVER share paste block with setup.
#
# Usage:
#   .\Run14_Postflight.ps1 -SshHost ssh7.vast.ai -SshPort 17254 -InstanceId 37999999 -HourlyRate 0.74
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string]$SshHost,
    [Parameter(Mandatory=$true)] [int]   $SshPort,
    [Parameter(Mandatory=$true)] [string]$InstanceId,
    [Parameter(Mandatory=$true)] [double]$HourlyRate,
    [string]$SshKey       = "C:\Users\monzi\.ssh\id_lambda_run8",
    [string]$RepoRoot     = "C:\Projects\genomic-variant-classifier",
    [string]$RemoteLog    = "/workspace/run11_master.log",
    [string]$RemoteOutputs= "/workspace/genomic-variant-classifier/outputs/run9_fresh",
    [string]$RemoteReport = "/workspace/run14_report",
    [string]$LocalReport  = ""  # default: $RepoRoot\outputs\run14_report
)

$ErrorActionPreference = "Stop"
if (-not $LocalReport) { $LocalReport = Join-Path $RepoRoot "outputs\run14_report" }

$sshBase = @("-i", $SshKey, "-p", $SshPort, "-o", "StrictHostKeyChecking=no", "root@$SshHost")
$scpBase = @("-i", $SshKey, "-P", $SshPort, "-o", "StrictHostKeyChecking=no")

function Invoke-Ssh($cmd) { & ssh @sshBase $cmd 2>&1 }

Write-Host "=== Run 14 Postflight ===" -ForegroundColor Cyan
Write-Host "Target: ${SshHost}:${SshPort}, Instance $InstanceId, ${HourlyRate}/hr" -ForegroundColor Gray
Write-Host ""

# -----------------------------------------------------------------------------
# 1. Confirm training actually finished (not a mid-run pull)
# -----------------------------------------------------------------------------
Write-Host "[1/6] Confirm training completion..." -ForegroundColor Yellow
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
Write-Host "`n[2/6] Push observability script and run it..." -ForegroundColor Yellow
$obsScript = Join-Path $RepoRoot "scripts\run14_observability.py"
if (-not (Test-Path $obsScript)) {
    Write-Host "  ERROR: $obsScript not found locally. Copy from Downloads first." -ForegroundColor Red
    exit 1
}
& scp @scpBase $obsScript "root@${SshHost}:/workspace/genomic-variant-classifier/scripts/run14_observability.py"
if ($LASTEXITCODE -ne 0) { Write-Host "  scp failed" -ForegroundColor Red; exit 1 }

Invoke-Ssh "mkdir -p $RemoteReport"
$obsCmd = "python3 /workspace/genomic-variant-classifier/scripts/run14_observability.py " +
          "--outputs-dir $RemoteOutputs --log $RemoteLog --report-dir $RemoteReport " +
          "--instance-id $InstanceId --hourly-rate $HourlyRate"
$obsOut = Invoke-Ssh $obsCmd
Write-Host $obsOut -ForegroundColor Gray

# -----------------------------------------------------------------------------
# 3. SCP report + log + key artifacts back
# -----------------------------------------------------------------------------
Write-Host "`n[3/6] SCP artifacts back to $LocalReport ..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $LocalReport | Out-Null

# 3a. Observability report (small, always)
& scp @scpBase -r "root@${SshHost}:${RemoteReport}/*" "$LocalReport\"
if ($LASTEXITCODE -ne 0) { Write-Host "  scp report failed" -ForegroundColor Red; exit 1 }

# 3b. Master log
& scp @scpBase "root@${SshHost}:${RemoteLog}" "$LocalReport\run11_master.log"

# 3c. Per-model OOF predictions (if present)
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/oof" "$LocalReport\oof" 2>$null

# 3d. Test-set predictions + metrics (if present)
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/test_eval" "$LocalReport\test_eval" 2>$null

# 3e. Feature importance / blend weights
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/blend_weights.json" "$LocalReport\" 2>$null
& scp @scpBase -r "root@${SshHost}:${RemoteOutputs}/feature_importance.csv" "$LocalReport\" 2>$null

Write-Host "  Done." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 4. Local quick-look
# -----------------------------------------------------------------------------
Write-Host "`n[4/6] Local artifact summary..." -ForegroundColor Yellow
Get-ChildItem $LocalReport -Recurse -File |
    Select FullName, @{N='SizeMB';E={[math]::Round($_.Length/1MB, 2)}} |
    Sort-Object SizeMB -Descending | Format-Table -AutoSize

# -----------------------------------------------------------------------------
# 5. Print observability summary
# -----------------------------------------------------------------------------
Write-Host "`n[5/6] Observability report tail..." -ForegroundColor Yellow
$mdPath = Join-Path $LocalReport "run14_observability.md"
if (Test-Path $mdPath) {
    Get-Content $mdPath | Select-Object -First 80
    Write-Host "`n  Full report: $mdPath" -ForegroundColor Green
} else {
    Write-Host "  Report markdown not found at $mdPath" -ForegroundColor Red
}

# -----------------------------------------------------------------------------
# 6. Emit destroy command — DO NOT execute here
# -----------------------------------------------------------------------------
Write-Host "`n[6/6] Destroy command (paste SEPARATELY after verifying local artifacts):" -ForegroundColor Cyan
Write-Host ""
Write-Host "  echo y | vastai destroy instance $InstanceId" -ForegroundColor Magenta
Write-Host ""
Write-Host "Standing rule: irreversible commands NEVER share a paste block with setup/copy commands." -ForegroundColor Yellow
