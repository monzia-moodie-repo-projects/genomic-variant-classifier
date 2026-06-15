# =============================================================================
# Run17_Monitor.ps1
# =============================================================================
# Purpose:  During-training observability for Run 17. Runs locally, SSHes into
#           the Vast.ai instance, and returns a structured snapshot of progress
#           without disturbing the run. Adapted from Run16_Monitor.ps1 with a
#           dedicated GNN mode (Run 17 activates gnn_score, so the [GNN-TRACE]
#           lines and the STRING/Best-val-AUC signals are first-class here).
#
# Author:   Monzia Moodie
# Created:  2026-06-14
#
# Usage:
#   .\Run17_Monitor.ps1 -SshHost ssh8.vast.ai -SshPort 18494
#   .\Run17_Monitor.ps1 -SshHost ssh8.vast.ai -SshPort 18494 -Mode GNN
#   .\Run17_Monitor.ps1 -SshHost ssh8.vast.ai -SshPort 18494 -Mode Full
#   .\Run17_Monitor.ps1 -SshHost ssh8.vast.ai -SshPort 18494 -Mode KAN
#   .\Run17_Monitor.ps1 -SshHost ssh8.vast.ai -SshPort 18494 -Mode Errors
#   .\Run17_Monitor.ps1 -SshHost ssh8.vast.ai -SshPort 18494 -Mode Tail -TailLines 80
#
# Modes:
#   Quick  - AUROC lines + base-models-done count + nvidia-smi one-shot
#   GNN    - gnn_score lifecycle: [GNN-TRACE], STRING source, epochs, Best val AUC,
#            and the non-degeneracy signal (the Run 17 deliverable)
#   Full   - Quick + GNN + disk usage + python process status + recent errors
#   KAN    - KAN backend / fit success / OOF AUROC
#   Errors - All errors / warnings / failures from the master log
#   Tail   - Last N lines of the master log (-TailLines, default 50)
#
# Assumes the run was launched with output redirected to -LogPath, e.g. on the box:
#   PYTHONUNBUFFERED=1 python scripts/run_phase2_eval.py ... 2>&1 | tee /workspace/run17_master.log
#
# Note:  Invoke-Ssh suppresses the vast.ai proxy banner and flattens stderr so
#        PowerShell never renders it as a red error block.
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string]$SshHost,
    [Parameter(Mandatory=$true)] [int]   $SshPort,
    [string]$SshKey = "C:\Users\monzi\.ssh\id_lambda_run8",
    [string]$LogPath = "/workspace/run17_master.log",
    [ValidateSet('Quick','GNN','Full','KAN','Errors','Tail')]
    [string]$Mode = 'Quick',
    [int]$TailLines = 50
)

if (-not (Test-Path $SshKey)) {
    Write-Error "SSH key not found at $SshKey"
    exit 1
}

$sshBase = @("-i", $SshKey, "-p", $SshPort, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", "root@$SshHost")

function Invoke-Ssh($cmd) {
    # Merge stderr (2>&1); flatten every item to a plain string so the vast.ai
    # banner (an ErrorRecord on stderr) is never rendered as a red block; then
    # drop the banner lines. If nothing remains, say so plainly.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & ssh @sshBase $cmd 2>&1 | ForEach-Object { "$_" } | Where-Object {
            ($_ -notmatch 'Welcome to vast\.ai') -and
            ($_ -notmatch 'If authentication fails') -and
            ($_ -notmatch 'Have fun')
        }
        if ($out) { $out } else { '  (no matching lines yet)' }
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

Write-Host "=== Run 17 Monitor [$Mode] @ $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
Write-Host "Target: ${SshHost}:${SshPort}   Log: $LogPath" -ForegroundColor Gray
Write-Host ""

function Show-Quick {
    Write-Host "[AUROC / per-model lines]" -ForegroundColor Yellow
    Invoke-Ssh "grep -iE 'auroc|kan|lightgbm|cnn_1d|^==>' $LogPath 2>/dev/null | tail -25"
    Write-Host "`n[base models with OOF AUROC logged (of 13)]" -ForegroundColor Yellow
    Invoke-Ssh "grep -c 'OOF AUROC' $LogPath 2>/dev/null"
    Write-Host "`n[GPU + memory snapshot]" -ForegroundColor Yellow
    Invoke-Ssh "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null"
}

function Show-Gnn {
    Write-Host "[STRING source resolution (local vs download)]" -ForegroundColor Yellow
    Invoke-Ssh "grep -iE 'GNN-TRACE.*local_links|GNN training: STRING threshold|Downloading .*stringdb' $LogPath 2>/dev/null | tail -10"
    Write-Host "`n[GNN lifecycle trace]" -ForegroundColor Yellow
    Invoke-Ssh "grep -E '\[GNN-TRACE\]' $LogPath 2>/dev/null | tail -25"
    Write-Host "`n[GNN epochs / best val AUC]" -ForegroundColor Yellow
    Invoke-Ssh "grep -iE 'epoch|best val auc|GNN training complete' $LogPath 2>/dev/null | tail -15"
    Write-Host "`n[gnn_score non-degeneracy signal]" -ForegroundColor Yellow
    Invoke-Ssh "grep -iE 'gnn_score.*(std|constant|degenerate|non-degenerate)|verify_gnn_score' $LogPath 2>/dev/null | tail -10"
}

switch ($Mode) {
    'Quick' { Show-Quick }
    'GNN'   { Show-Gnn }
    'Full'  {
        Show-Quick
        Write-Host ""
        Show-Gnn
        Write-Host "`n[Disk usage on /workspace + run17 outputs]" -ForegroundColor Yellow
        Invoke-Ssh "df -h /workspace 2>/dev/null && du -sh /workspace/genomic-variant-classifier/outputs/run17 2>/dev/null"
        Write-Host "`n[Python training process]" -ForegroundColor Yellow
        Invoke-Ssh "ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | grep -E 'run_phase2_eval|python' | grep -v grep | head -5"
        Write-Host "`n[Last 10 errors / warnings]" -ForegroundColor Yellow
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
