# =============================================================================
# Vastai_Destroy_Confirmed.ps1
# =============================================================================
# Purpose:  Safely destroy a Vast.ai instance after a postflight gate has
#           confirmed all required artifacts are present locally.
#
# Author:   Monzia Moodie
# Created:  2026-05-27
# Closes:   Run 15 plan C.7 (Standing Rule from SESSION_2026-05-26.md L143)
#
# Design (defense in depth, 4 independent refusal layers):
#   1. Refuse if stdin is redirected ([Console]::IsInputRedirected).
#      This blocks the failure modes from INCIDENT_2026-05-12 and
#      INCIDENT_2026-05-24, where `echo y | ...` was used to skip the
#      Vast.ai CLI prompt and inadvertently auto-confirmed destroys.
#   2. Refuse if -GateFile path does not exist.
#   3. Refuse if the contents of -GateFile is not exactly "0".
#   4. Require interactive Read-Host returning exactly "DESTROY"
#      (uppercase, no quotes, no whitespace). Typo-resistant.
#
# Exit codes:
#   0  success (destroyed, or user aborted at confirmation prompt)
#   2  stdin is redirected (refused)
#   3  gate file path does not exist (refused)
#   4  gate exit code is not "0" (refused)
#   5  vastai destroy CLI returned non-zero
#
# Usage (interactively, NOT through pipe):
#   .\scripts\Vastai_Destroy_Confirmed.ps1 -InstanceId 38001234 `
#                                          -GateFile 'C:\Projects\genomic-variant-classifier\outputs\run15_report\.gate_exit_code'
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string]$InstanceId,
    [Parameter(Mandatory=$true)] [string]$GateFile
)

$ErrorActionPreference = "Stop"

Write-Host "=== Vastai_Destroy_Confirmed ===" -ForegroundColor Cyan
Write-Host "  InstanceId: $InstanceId" -ForegroundColor Gray
Write-Host "  GateFile:   $GateFile" -ForegroundColor Gray
Write-Host ""

# -----------------------------------------------------------------------------
# Layer 1: Refuse if stdin is redirected (blocks `echo y | ...` automation)
# -----------------------------------------------------------------------------
if ([Console]::IsInputRedirected) {
    Write-Host "REFUSE: stdin is redirected." -ForegroundColor Red
    Write-Host "  This script requires interactive input. Possible cause:" -ForegroundColor Yellow
    Write-Host "    - 'echo y | .\Vastai_Destroy_Confirmed.ps1 ...'" -ForegroundColor Yellow
    Write-Host "    - Bash pipe through PowerShell" -ForegroundColor Yellow
    Write-Host "    - Background invocation without -NoProfile -NoExit" -ForegroundColor Yellow
    Write-Host "  Standing rule (SR #38 / memory 30c): irreversible cloud commands" -ForegroundColor Yellow
    Write-Host "  require interactive confirmation and may not be auto-piped." -ForegroundColor Yellow
    exit 2
}
Write-Host "  Layer 1 (stdin not redirected): PASS" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Layer 2: Refuse if gate file path does not exist
# -----------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $GateFile)) {
    Write-Host "REFUSE: gate file not found at $GateFile" -ForegroundColor Red
    Write-Host "  This file is written by Run15_Postflight.ps1 with the gate exit code." -ForegroundColor Yellow
    Write-Host "  Run postflight first; verify the file appears with content '0'." -ForegroundColor Yellow
    exit 3
}
Write-Host "  Layer 2 (gate file exists): PASS" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Layer 3: Refuse if gate exit code is not exactly "0"
# -----------------------------------------------------------------------------
$gateContent = (Get-Content -LiteralPath $GateFile -Raw).Trim()
if ($gateContent -ne "0") {
    Write-Host "REFUSE: gate exit code is '$gateContent' (expected '0')." -ForegroundColor Red
    Write-Host "  Postflight FAILED. Fix artifact issues before destroying." -ForegroundColor Yellow
    Write-Host "  Re-run scripts\Run15_Postflight.ps1; ensure all gates PASS." -ForegroundColor Yellow
    exit 4
}
Write-Host "  Layer 3 (gate exit code = 0): PASS" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Layer 4: Interactive confirmation (type exact "DESTROY", uppercase)
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "READY to destroy Vast.ai instance: $InstanceId" -ForegroundColor Cyan
Write-Host "  This is IRREVERSIBLE. The instance and all on-instance data are lost." -ForegroundColor Yellow
Write-Host "  Artifacts should already be SCPed locally; verify before continuing." -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "  Type 'DESTROY' (exact, case-sensitive) to proceed, anything else to abort"
if ($confirm -cne "DESTROY") {
    Write-Host "  Aborted (confirmation did not match exactly 'DESTROY')." -ForegroundColor Yellow
    exit 0
}
Write-Host "  Layer 4 (interactive DESTROY confirmation): PASS" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Execute: pipe 'y' to vastai destroy to handle CLI >=1.0.12 interactive prompt
# (per INCIDENT_2026-05-12_vastai-destroy-interactive.md)
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "Destroying instance $InstanceId ..." -ForegroundColor Red
$destroyResult = "y" | & vastai destroy instance $InstanceId 2>&1
Write-Host $destroyResult -ForegroundColor Gray
if ($LASTEXITCODE -ne 0) {
    Write-Host "vastai destroy CLI exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "  Investigate via Vast.ai web console; instance may still be running." -ForegroundColor Yellow
    exit 5
}
Write-Host ""
Write-Host "  Instance $InstanceId destroyed." -ForegroundColor Green
exit 0