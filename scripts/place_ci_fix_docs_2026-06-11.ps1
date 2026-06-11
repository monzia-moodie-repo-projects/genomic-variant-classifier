# place_ci_fix_docs_2026-06-11.ps1 -- file the CI-fix incident + CHANGELOG entry into the docs/
# canonicals. Idempotent, no-BOM UTF8, atomic-add-safe. Author: Monzia Moodie.
$ErrorActionPreference = "Stop"
$root = "C:\Projects\genomic-variant-classifier"
Set-Location $root
$dl = Join-Path $HOME "Downloads"
$utf8 = New-Object System.Text.UTF8Encoding($false)

$incSrc = Join-Path $dl  "INCIDENT_2026-06-11_ci-optional-deps.md"
$incDst = Join-Path $root "docs\incidents\INCIDENT_2026-06-11_ci-optional-deps.md"
$chPath = Join-Path $root "docs\CHANGELOG.md"
$chEntrySrc = Join-Path $dl "CHANGELOG_entry_2026-06-11.md"

if (-not (Test-Path $chPath))     { throw "MISSING canonical: $chPath" }
if (-not (Test-Path $incSrc))     { throw "MISSING incident source: $incSrc" }
if (-not (Test-Path $chEntrySrc)) { throw "MISSING changelog entry source: $chEntrySrc" }

New-Item -ItemType Directory -Force -Path (Split-Path $incDst) | Out-Null
Copy-Item $incSrc $incDst -Force
Write-Host "placed incident: $incDst"

$chEntry = (Get-Content $chEntrySrc -Raw).TrimEnd()
if (-not (Select-String -Path $chPath -SimpleMatch "CI fix: agent-layer optional deps (pandera, river)" -Quiet)) {
    [System.IO.File]::AppendAllText($chPath, "`r`n" + $chEntry + "`r`n", $utf8)
    Write-Host "appended CHANGELOG entry"
} else { Write-Host "CHANGELOG entry already present; skipped" }

git add -- $incDst
git add -- $chPath
Write-Host "`n=== git status (review before commit) ==="
git status --short
