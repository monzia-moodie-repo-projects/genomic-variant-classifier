# Install_hygiene_gitignore_and_stray_test.ps1
# Purpose: (1) delete the UNTRACKED stray root test_training_lifecycle_drift_removal.py
#          (the tracked canonical copy lives at tests/unit/), and
#          (2) append two .gitignore rules (*.factory_state_* and data/_pandas3/).
# Idempotent, anchored, abort-on-mismatch. Does NOT commit; leaves a clean staged set for review.
# NOTE: docs/roadmap is a TRACKED DIRECTORY (docs/roadmap/RUN17_SCOPE.md) -- NOT touched here.

$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\genomic-variant-classifier"

Write-Host "===== STEP 0: preconditions =====" -ForegroundColor Cyan

# 0a. stray root test must be UNTRACKED (git ls-files empty) but PRESENT on disk
$rootTracked = [bool](git ls-files test_training_lifecycle_drift_removal.py)
$rootPresent = Test-Path .\test_training_lifecycle_drift_removal.py
$unitTracked = [bool](git ls-files tests/unit/test_training_lifecycle_drift_removal.py)
if ($rootTracked) { throw "ABORT: root test is TRACKED; not a stray. Manual review required." }
if (-not $unitTracked) { throw "ABORT: canonical tests/unit copy is NOT tracked; do not delete the root copy." }
if (-not $rootPresent) { Write-Host "  note: root stray already absent (idempotent no-op for deletion)" -ForegroundColor Yellow }
Write-Host "  OK: root stray untracked=$([bool]$rootPresent) present; canonical tests/unit copy tracked" -ForegroundColor Green

# 0b. .gitignore must exist
if (-not (Test-Path .\.gitignore)) { throw "ABORT: .gitignore not found at repo root." }

Write-Host "`n===== STEP 1: delete UNTRACKED stray root test =====" -ForegroundColor Cyan
if ($rootPresent) {
    Remove-Item .\test_training_lifecycle_drift_removal.py -Force
    if (Test-Path .\test_training_lifecycle_drift_removal.py) { throw "ABORT: deletion failed." }
    Write-Host "  OK: removed root stray test_training_lifecycle_drift_removal.py" -ForegroundColor Green
} else {
    Write-Host "  SKIP: already absent" -ForegroundColor Yellow
}

Write-Host "`n===== STEP 2: append .gitignore rules (idempotent) =====" -ForegroundColor Cyan
$utf8 = New-Object System.Text.UTF8Encoding($false)   # BOM-free
$gi   = [System.IO.File]::ReadAllText("$pwd\.gitignore")
$rules = @(
  '*.factory_state_*',
  'data/_pandas3/'
)
$header = '# --- session hygiene 2026-07-01: agent factory-state backups + pandas3 scratch ---'
$toAdd = @()
if ($gi -notmatch [regex]::Escape('*.factory_state_*')) { $toAdd += '*.factory_state_*' }
if ($gi -notmatch [regex]::Escape('data/_pandas3/'))   { $toAdd += 'data/_pandas3/' }

if ($toAdd.Count -eq 0) {
    Write-Host "  SKIP: both rules already present (idempotent)" -ForegroundColor Yellow
} else {
    $block = "`n" + $header + "`n" + ($toAdd -join "`n") + "`n"
    if (-not $gi.EndsWith("`n")) { $block = "`n" + $block }
    [System.IO.File]::WriteAllText("$pwd\.gitignore", $gi + $block, $utf8)
    Write-Host "  OK: appended $($toAdd.Count) rule(s): $($toAdd -join ', ')" -ForegroundColor Green
}

# post-condition: both rules now present
$gi2 = [System.IO.File]::ReadAllText("$pwd\.gitignore")
if ($gi2 -notmatch [regex]::Escape('*.factory_state_*')) { throw "POSTCOND FAIL: factory_state rule missing." }
if ($gi2 -notmatch [regex]::Escape('data/_pandas3/'))   { throw "POSTCOND FAIL: _pandas3 rule missing." }
# guard: no BOM, no CRLF injected into the block
if ($gi2 -match "`r`n`r`n`r`n") { Write-Host "  warn: multiple blank lines near append (cosmetic)" -ForegroundColor Yellow }
Write-Host "  POSTCOND OK: both rules present, BOM-free" -ForegroundColor Green

Write-Host "`n===== STEP 3: stage EXACTLY the intended set, GATE on mismatch =====" -ForegroundColor Cyan
git reset | Out-Null
git add .gitignore
# stage the deletion of the stray (only if it was tracked -- it is NOT, so nothing to stage for it;
# the untracked-file deletion produces no git change. Confirm that below.)
$staged = @(git diff --cached --name-only)
$intended = @(".gitignore")
if ($null -eq $staged -or (Compare-Object $intended ($staged | Sort-Object))) {
    Write-Host "  STAGED SET MISMATCH:" -ForegroundColor Red
    git diff --cached --name-only
    throw "staged set != intended (.gitignore only). Review before commit."
}
Write-Host "  STAGED SET OK: .gitignore only" -ForegroundColor Green
git diff --cached --name-status

Write-Host "`n===== HYGIENE PREP COMPLETE =====" -ForegroundColor Cyan
Write-Host "Staged: .gitignore (2 rules). Root stray test deleted (was untracked -> no git change)." -ForegroundColor Green
Write-Host "Review 'git diff --cached', then commit with the block provided in chat." -ForegroundColor Green
