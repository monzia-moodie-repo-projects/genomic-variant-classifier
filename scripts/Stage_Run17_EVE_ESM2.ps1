# Stage_Run17_EVE_ESM2.ps1
# Stage the SIX Run-17 annotation sources from Drive (canonical) to local C:, ready
# for SCP to the Vast.ai VM. EVE (14,933 small CSVs) is packaged as ONE tarball
# (per-file SCP overhead would be brutal); the rest are single/few files.
#
#   EVE dir + UniProt index + OMIM + PhyloP + dbSNP + ClinGen
#
# Run from repo root. READ-from-Drive + WRITE-local-staging only; does NOT SCP (that
# happens in the preflight with the run's SSH host/port). No Drive writes.
#
# Per memory: NEVER bulk-write large files through G: (DriveFS caches onto C:). Uses
# `rclone copy` (Drive API direct) to a real local dir, NOT a G:\ copy.

$ErrorActionPreference = "Stop"
$repo = "C:\Projects\genomic-variant-classifier"
Set-Location $repo
$DRIVE = "genvarcla:genomic-variant-classifier/data/external"
$EXT = "$repo\data\external"
$STAGE = "$repo\data\_vm_stage"
New-Item -ItemType Directory -Force -Path $STAGE | Out-Null

function Pull-Source($name, $expectMin) {
    $driveCount = (rclone lsf "$DRIVE/$name" --files-only -R 2>$null | Measure-Object).Count
    Write-Host ("== {0}: Drive files = {1}" -f $name, $driveCount)
    if ($driveCount -lt $expectMin) { throw "$name Drive count $driveCount < expected $expectMin - investigate before staging." }
    rclone copy "$DRIVE/$name" "$EXT\$name" --transfers 16 --progress
    $localCount = (Get-ChildItem "$EXT\$name" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host ("   local files = {0}" -f $localCount)
    if ($localCount -lt $driveCount) { throw "$name local $localCount < Drive $driveCount - copy incomplete." }
    return $localCount
}

Write-Host "== [1/6] EVE variant_files (expect ~3211 score CSVs; full 63GB bundle stays on Drive+local) ==" -ForegroundColor Cyan
# Option B: stage ONLY the score CSVs the VM needs. MSAs/VCFs/plots remain preserved
# locally (data\external\eve\EVE_all_data) and on Drive for later phases.
$eveN = Pull-Source "eve/EVE_all_data/variant_files" 3000

Write-Host "== [2/6] UniProt index ==" -ForegroundColor Cyan
Pull-Source "uniprot" 1 | Out-Null
$UNI = "$EXT\uniprot\uniprot_human_reviewed.parquet"
if (-not (Test-Path $UNI)) { throw "UniProt index missing after copy: $UNI" }

Write-Host "== [3/6] OMIM ==" -ForegroundColor Cyan
Pull-Source "omim" 1 | Out-Null
Write-Host "== [4/6] PhyloP ==" -ForegroundColor Cyan
Pull-Source "phylop" 1 | Out-Null
Write-Host "== [5/6] dbSNP ==" -ForegroundColor Cyan
Pull-Source "dbsnp" 1 | Out-Null
Write-Host "== [5b] ClinGen ==" -ForegroundColor Cyan
Pull-Source "clingen" 1 | Out-Null

# Show EXACTLY which file each format-aware launch glob will pick, so a wrong pick is
# visible NOW (before the VM run), not silently at annotation time.
Write-Host "`n== Launch-glob resolution preview (what the launch script will wire) ==" -ForegroundColor Yellow
$omim = (Get-ChildItem "$EXT\omim\*mim2gene*" -ErrorAction SilentlyContinue | Select -First 1)
if (-not $omim) { $omim = (Get-ChildItem "$EXT\omim\*" -File | Where-Object { $_.Name -notmatch '(?i)readme|checksum|md5' } | Select -First 1) }
$dbsnp = (Get-ChildItem "$EXT\dbsnp\*.parquet" -ErrorAction SilentlyContinue | Select -First 1)
$clingen = (Get-ChildItem "$EXT\clingen\*.csv" -ErrorAction SilentlyContinue | Select -First 1)
$phylop = (Get-ChildItem "$EXT\phylop\*" -File | Where-Object { $_.Name -notmatch '(?i)readme|checksum|md5' } | Select -First 1)
"  OMIM    -> $($omim.FullName)"
"  dbSNP   -> $($dbsnp.FullName)"
"  ClinGen -> $($clingen.FullName)"
"  PhyloP  -> $($phylop.FullName)"
if (-not $omim)    { Write-Warning "OMIM: no non-readme file resolved - launch will ABORT (exit 8). Acquire the mim2gene file." }
if (-not $dbsnp)   { Write-Warning "dbSNP: no .parquet resolved - DbSNPConnector needs a parquet. Launch will ABORT (exit 8)." }
if (-not $clingen) { Write-Warning "ClinGen: no .csv resolved - launch will ABORT (exit 8). Acquire the Validity CSV." }
if (-not $phylop)  { Write-Warning "PhyloP: no file resolved - launch will ABORT (exit 8)." }

Write-Host "`n== [6/6] Package EVE variant_files as a single tarball (~10GB; fast SCP; extract on VM) ==" -ForegroundColor Cyan
$EVE_TAR = "$STAGE\eve_variant_files.tar.gz"
if (Test-Path $EVE_TAR) { Remove-Item $EVE_TAR -Force }
# B4 layout: tar from $EXT\eve so the archive root is EVE_all_data/variant_files.
# VM extracts with `-C data/external/eve`, yielding
#   data/external/eve/EVE_all_data/variant_files  == the local path (identical).
# bsdtar-safe (tar -C <dir> <relative-subdir>); no --transform; no 10GB copy.
$EVE_VF = "$EXT\eve\EVE_all_data\variant_files"
$vfCount = (Get-ChildItem $EVE_VF -Filter *.csv -File -ErrorAction SilentlyContinue | Measure-Object).Count
if ($vfCount -lt 3000) { throw "EVE variant_files has $vfCount CSVs (expected ~3211) at $EVE_VF -- pull incomplete." }
Push-Location "$EXT\eve"
tar -czf $EVE_TAR EVE_all_data/variant_files
Pop-Location
if (-not (Test-Path $EVE_TAR)) { throw "EVE tarball not created." }
# Post-check: archive lists the expected leaf path + CSV count.
$tarList = (tar -tzf $EVE_TAR)
$tarCsv = ($tarList | Where-Object { $_ -match "EVE_all_data/variant_files/.+\.csv$" } | Measure-Object).Count
if ($tarCsv -lt 3000) { throw "EVE tarball lists only $tarCsv CSVs (expected ~3211) -- bad tar layout." }
"EVE tarball: $EVE_TAR ($("{0:N1}" -f ((Get-Item $EVE_TAR).Length/1MB)) MB; $tarCsv CSVs under EVE_all_data/variant_files)"

Write-Host "`n== Staging manifest (SCP these in the Run 17 preflight) ==" -ForegroundColor Green
"  1. $EVE_TAR  -> /workspace/genomic-variant-classifier/data/external/eve/  then: cd .../external/eve && tar -xzf eve_variant_files.tar.gz && ls EVE_all_data/variant_files | wc -l   # expect $eveN (~3211)"
"     launch --eve-path resolves to data/external/eve/EVE_all_data/variant_files (== local path)."
"  2. $UNI  -> .../data/external/uniprot/"
"  3. data\external\{omim,phylop,dbsnp,clingen}\  -> .../data/external/{omim,phylop,dbsnp,clingen}/  (small; SCP directly)"
"Staging prep complete."
