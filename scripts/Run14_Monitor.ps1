# =============================================================================
# Run14_Monitor.ps1
# =============================================================================
# Purpose:  During-training observability queries for Run 14. Runs locally,
#           SSHes into the Vast.ai instance, and returns a structured
#           snapshot of training progress without disturbing the run.
#
# Author:   Claude, for Monzia Moodie
# Created:  2026-05-26
#
# Usage:
#   .\Run14_Monitor.ps1 -SshHost ssh7.vast.ai -SshPort 17254
#   .\Run14_Monitor.ps1 -SshHost ssh7.vast.ai -SshPort 17254 -Mode Quick
#   .\Run14_Monitor.ps1 -SshHost ssh7.vast.ai -SshPort 17254 -Mode Full
#   .\Run14_Monitor.ps1 -SshHost ssh7.vast.ai -SshPort 17254 -Mode KAN
#
# Modes:
#   Quick - AUROC lines from master log + nvidia-smi one-shot
#   Full  - Quick + disk usage + python process status + recent errors
#   KAN   - Targeted KAN status: backend used, fit success/fail, OOF AUROC
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string]$SshHost,
    [Parameter(Mandatory=$true)] [int]   $SshPort,
    [string]$SshKey = "C:\Users\monzi\.ssh\id_lambda_run8",
    [string]$LogPath = "/workspace/run11_master.log",
    [ValidateSet('Quick','Full','KAN','Errors','Tail')]
    [string]$Mode = 'Quick',
    [int]$TailLines = 50
)

if (-not (Test-Path $SshKey)) {
    Write-Error "SSH key not found at $SshKey"
    exit 1
}

$sshBase = @("-i", $SshKey, "-p", $SshPort, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", "root@$SshHost")

function Invoke-Ssh($cmd) {
    & ssh @sshBase $cmd 2>&1
}

Write-Host "=== Run 14 Monitor [$Mode] @ $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
Write-Host "Target: ${SshHost}:${SshPort}" -ForegroundColor Gray
Write-Host ""

switch ($Mode) {
    'Quick' {
        Write-Host "[AUROC lines from master log]" -ForegroundColor Yellow
        Invoke-Ssh "grep -iE 'auroc|kan|lightgbm|cnn_1d|^==>' $LogPath 2>/dev/null | tail -25"

        Write-Host "`n[GPU + memory snapshot]" -ForegroundColor Yellow
        Invoke-Ssh "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null"
    }

    'Full' {
        Write-Host "[AUROC lines from master log]" -ForegroundColor Yellow
        Invoke-Ssh "grep -iE 'auroc|kan|lightgbm|cnn_1d|^==>' $LogPath 2>/dev/null | tail -40"

        Write-Host "`n[GPU + memory snapshot]" -ForegroundColor Yellow
        Invoke-Ssh "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null"

        Write-Host "`n[Disk usage on /workspace]" -ForegroundColor Yellow
        Invoke-Ssh "df -h /workspace 2>/dev/null && du -sh /workspace/genomic-variant-classifier/outputs/run9_fresh 2>/dev/null"

        Write-Host "`n[Python training process]" -ForegroundColor Yellow
        Invoke-Ssh "ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | grep -E 'train.py|python' | grep -v grep | head -5"

        Write-Host "`n[Last 5 errors / warnings]" -ForegroundColor Yellow
        Invoke-Ssh "grep -iE 'error|warn|fail|exception|traceback' $LogPath 2>/dev/null | grep -v 'UserWarning\|FutureWarning' | tail -10"
    }

    'KAN' {
        Write-Host "[KAN package patch confirmation]" -ForegroundColor Yellow
        Invoke-Ssh "grep -iE 'imodelsx_patch' $LogPath 2>/dev/null"

        Write-Host "`n[KAN training status]" -ForegroundColor Yellow
        Invoke-Ssh "grep -iE 'kan|imodelsx|pykan|efficient_kan' $LogPath 2>/dev/null | grep -v 'imodelsx_patch' | tail -20"

        Write-Host "`n[KAN OOF AUROC]" -ForegroundColor Yellow
        Invoke-Ssh "grep -iE 'kan.*auroc|kan.*OOF|auroc.*kan' $LogPath 2>/dev/null | tail -10"
    }

    'Errors' {
        Write-Host "[All errors / warnings / failures]" -ForegroundColor Yellow
        Invoke-Ssh "grep -niE 'error|fail|exception|traceback|nameerror|attributeerror|importerror' $LogPath 2>/dev/null | grep -v 'UserWarning\|FutureWarning\|DeprecationWarning' | tail -40"
    }

    'Tail' {
        Write-Host "[Last $TailLines lines of master log]" -ForegroundColor Yellow
        Invoke-Ssh "tail -n $TailLines $LogPath 2>/dev/null"
    }
}

Write-Host "`n=== End of $Mode snapshot @ $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
