# =====================================================================
# Search_Vast_Offers.ps1  (2026-07-06)  -- READ-ONLY, NO SPEND
# ---------------------------------------------------------------------
# Correct vastai 1.2.0 offer search for the Run-17 GPU box. Encodes the units fix
# (cpu_ram is GB, not MB -- cpu_ram>=64000 asked for 64 TB and returned ZERO rows on
# 2026-07-06) and the < $0.80/hr cap. Cheapest-first. Tees results to Downloads.
#
# If the primary (fully-filtered) search returns zero rows, it AUTO-RELAXES constraints
# in the documented order (disk -> direct_port -> reliability -> verified) and reports
# which relaxation produced results -- so you see the real supply, not an empty header.
#
# Nothing here spends money: `search offers` is read-only. Pick an OFFER_ID from the
# output, then use Provision_Vast_Run17.ps1 to create the instance.
#
#   cd C:\Projects\genomic-variant-classifier
#   .\Search_Vast_Offers.ps1                 # default: RTX 4090+5090+3090, <$0.80/hr
#   .\Search_Vast_Offers.ps1 -Gpus 'RTX_4090' -MaxPrice 0.60
# =====================================================================

param(
    [string[]] $Gpus     = @('RTX_4090','RTX_5090','RTX_3090'),
    [double]   $MaxPrice = 0.80,
    [int]      $MinRamGB = 64,     # >= Run-17 floor 50, with headroom
    [int]      $MinDiskGB= 200,    # >= Run-17 floor 150, with headroom
    [double]   $MinRel   = 0.98,
    [string]   $OutFile  = "$HOME\Downloads\vast_offers.txt"
)

$ErrorActionPreference = "Stop"

# vastai present + version note
$ver = (& vastai --version 2>&1)
Write-Host "vastai version: $ver" -ForegroundColor DarkGray
if ($ver -notmatch '^1\.[2-9]|^[2-9]') {
    Write-Host "WARN: expected 1.2.0+. Older CLIs use different units/subcommands (see VASTAI_PROVISIONING_2026.md)." -ForegroundColor Yellow
}

# GPU clause: single name -> 'gpu_name=X'; multiple -> 'gpu_name in [A,B,C]'
if ($Gpus.Count -eq 1) { $gpuClause = "gpu_name=$($Gpus[0])" }
else { $gpuClause = "gpu_name in [$([string]::Join(',', $Gpus))]" }

# Progressive filter sets: [0] = fullest, then relax one constraint per step.
$priceClause = "dph_total<$([string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0.##}',$MaxPrice))"
$base = "$gpuClause num_gpus=1 rentable=true $priceClause"
$filterSets = @(
    "$base verified=true direct_port_count>=1 disk_space>=$MinDiskGB cpu_ram>=$MinRamGB reliability>=$MinRel",
    "$base verified=true direct_port_count>=1 cpu_ram>=$MinRamGB reliability>=$MinRel",
    "$base verified=true direct_port_count>=1 cpu_ram>=$MinRamGB",
    "$base verified=true cpu_ram>=$MinRamGB",
    "$base"
)
$labels = @(
    "full (disk>=$MinDiskGB, direct, ram>=$MinRamGB, rel>=$MinRel, verified)",
    "relaxed: dropped disk floor",
    "relaxed: dropped reliability floor",
    "relaxed: dropped direct_port floor",
    "minimal: gpu + num_gpus + rentable + price only"
)

"" | Out-File $OutFile   # fresh file
$found = $false
for ($i = 0; $i -lt $filterSets.Count; $i++) {
    $q = $filterSets[$i]
    Write-Host "`n=== search [$($labels[$i])] ===" -ForegroundColor Cyan
    Write-Host "  query: $q" -ForegroundColor DarkGray
    "=== $($labels[$i]) ===" | Out-File $OutFile -Append
    "query: $q" | Out-File $OutFile -Append

    $out = (& vastai search offers "$q" -o 'dph_total' 2>&1)
    $out | Out-File $OutFile -Append
    $out | Write-Host

    # rows = non-empty lines that are NOT the header (header lines start with '#')
    $rows = @($out | Where-Object { $_ -and ($_ -notmatch '^\s*#') -and ($_ -match '\S') })
    if ($rows.Count -gt 0) {
        Write-Host "-> $($rows.Count) offer(s) at this filter level." -ForegroundColor Green
        $found = $true
        break
    } else {
        Write-Host "-> zero rows; relaxing..." -ForegroundColor Yellow
    }
}

Write-Host ""
if ($found) {
    Write-Host "Offers written to $OutFile. Pick an OFFER_ID (first column), then:" -ForegroundColor Green
    Write-Host "  .\Provision_Vast_Run17.ps1 -OfferId <ID>        # dry-run (prints the create command)" -ForegroundColor DarkGray
    Write-Host "  .\Provision_Vast_Run17.ps1 -OfferId <ID> -Execute   # actually create (SPENDS)" -ForegroundColor DarkGray
} else {
    Write-Host "No offers even at the minimal filter. This is genuine supply scarcity, not syntax." -ForegroundColor Red
    Write-Host "Try: widen -Gpus (add RTX_A5000, A100_PCIE), raise -MaxPrice, or retry later." -ForegroundColor Red
}
