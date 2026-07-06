# =====================================================================
# Provision_Vast_Run17.ps1  (2026-07-06)  -- DRY-RUN BY DEFAULT
# ---------------------------------------------------------------------
# Creates the Run-17 GPU instance with the correct vastai 1.2.0 syntax. Following the
# project convention (scripts/maintenance/sync_data_to_gdrive.py), it is DRY-RUN by
# default: it PRINTS the exact create command and does nothing. Pass -Execute to actually
# create the instance (this SPENDS money -- storage bills from creation, GPU from running).
#
# OfferId is a MANDATORY parameter (not a <placeholder> in the command line -- literal
# angle brackets are a PowerShell ParserError, and this project has been bitten by that).
#
#   cd C:\Projects\genomic-variant-classifier
#   .\Provision_Vast_Run17.ps1 -OfferId 12345678                 # dry-run: prints command
#   .\Provision_Vast_Run17.ps1 -OfferId 12345678 -Execute        # actually create (SPENDS)
#   .\Provision_Vast_Run17.ps1 -OfferId 12345678 -DiskGB 250 -Execute
# =====================================================================

param(
    [Parameter(Mandatory=$true)] [long] $OfferId,
    [string] $Image   = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    [int]    $DiskGB  = 200,                 # >= Run-17 150 GB floor, headroom for the data tree
    [string] $OnStart = "nvidia-smi",
    [switch] $Execute
)

$ErrorActionPreference = "Stop"

# preflight: CLI present + version
$ver = (& vastai --version 2>&1)
if ($ver -notmatch '^1\.[2-9]|^[2-9]') {
    throw "vastai $ver -- expected 1.2.0+. Older CLIs use different create syntax. Upgrade: pip install -U vastai"
}

# Build the create command as an argument array (exact 1.2.0 flags).
$vastArgs = @(
    'create','instance', "$OfferId",
    '--image', $Image,
    '--disk',  "$DiskGB",
    '--ssh','--direct',
    '--onstart-cmd', $OnStart
)

Write-Host "== Provision Run-17 VM (vastai $ver) ==" -ForegroundColor Cyan
Write-Host "  offer id : $OfferId"
Write-Host "  image    : $Image"
Write-Host "  disk     : $DiskGB GB  (Run-17 floor 150)"
Write-Host "  onstart  : $OnStart"
Write-Host ""
Write-Host "command:" -ForegroundColor DarkGray
Write-Host "  vastai $($vastArgs -join ' ')" -ForegroundColor Gray

if (-not $Execute) {
    Write-Host ""
    Write-Host "DRY-RUN (default) -- nothing created, no spend." -ForegroundColor Yellow
    Write-Host "Re-run with -Execute to create the instance (this SPENDS money)." -ForegroundColor Yellow
    return
}

Write-Host ""
Write-Host "EXECUTING create instance -- storage billing starts now; GPU billing starts at 'running'." -ForegroundColor Red
$out = (& vastai @vastArgs 2>&1)
$out | Write-Host
$out | Out-File "$HOME\Downloads\vast_create_result.txt"

# Try to surface the new instance id (new_contract) for the next steps.
$idLine = ($out | Select-String -Pattern 'new_contract').ToString()
Write-Host ""
if ($idLine) {
    Write-Host "Instance created. $idLine" -ForegroundColor Green
    Write-Host "NEXT:" -ForegroundColor Cyan
    Write-Host "  vastai show instance <INSTANCE_ID>     # poll until status == running"
    Write-Host "  vastai ssh-url <INSTANCE_ID>           # get ssh://root@HOST:PORT"
    Write-Host "  # then: SCP repo@659610f + data, rclone the cosmic/kegg files, run Run_Preflight_VM.sh 659610f"
    Write-Host "  # teardown when done: .\Vastai_Destroy_Confirmed.ps1 -InstanceId <INSTANCE_ID>"
} else {
    Write-Host "Create returned, but no new_contract line parsed -- check vast_create_result.txt and 'vastai show instances'." -ForegroundColor Yellow
}
