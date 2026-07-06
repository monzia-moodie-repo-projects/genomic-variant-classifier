# Install_pin_data_freshness_deps.ps1
# Purpose: pin the Data Freshness Monitor workflow's deps to the versions in
#          requirements.txt, so the weekly cron stops floating to pandas 3.0.x
#          (numpy 2.5 / scipy 1.18) and instead runs the stack the project ships.
# Root cause: data_freshness.yml installs UNPINNED `numpy pandas pyarrow scipy`,
#   which resolved to pandas 3.0.4 in run 28370058512 -- a version the project
#   deliberately rolled back from (requirements.txt pins pandas==2.3.3).
# Design: KEEP the cron minimal (the freshness agent needs no heavy ML deps);
#   do NOT switch to `-r requirements.txt` (55+ pkgs incl catboost/xgboost/transformers).
#   Just pin the four packages already listed. Anchored, idempotent, abort-on-mismatch.

$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\genomic-variant-classifier"

$wf = ".github\workflows\data_freshness.yml"

Write-Host "===== STEP 0: preconditions =====" -ForegroundColor Cyan
if (-not (Test-Path $wf)) { throw "ABORT: $wf not found." }
$utf8 = New-Object System.Text.UTF8Encoding($false)   # BOM-free
$raw  = [System.IO.File]::ReadAllText("$pwd\$wf")

# The exact unpinned line we expect to replace (anchor). Must appear verbatim, exactly once.
$oldLine = "          pip install numpy pandas pyarrow scipy"
$already = "          # pinned to requirements.txt versions"

if ($raw.Contains($already)) {
    Write-Host "  SKIP: workflow already pinned (idempotent no-op)." -ForegroundColor Yellow
    Write-Host "  Nothing to change; exiting cleanly." -ForegroundColor Yellow
    return
}

$n = ([regex]::Matches($raw, [regex]::Escape($oldLine))).Count
if ($n -ne 1) { throw "ABORT: expected the unpinned pip line exactly once; found $n. Manual review required." }
Write-Host "  OK: found the unpinned pip line exactly once." -ForegroundColor Green

Write-Host "`n===== STEP 1: replace unpinned line with pinned versions (from requirements.txt) =====" -ForegroundColor Cyan
$newLine = @(
  "          # pinned to requirements.txt versions (project rolled back from pandas 3.0.x;",
  "          # see requirements.txt + docs re: pandas 3.0.4 date_range segfault). Keep this",
  "          # cron minimal -- the freshness agent needs no heavy ML deps.",
  "          pip install numpy==2.4.4 pandas==2.3.3 pyarrow==23.0.1 scipy==1.17.1"
) -join "`n"

$patched = $raw.Replace($oldLine, $newLine)
if ($patched -eq $raw) { throw "ABORT: replacement produced no change." }
[System.IO.File]::WriteAllText("$pwd\$wf", $patched, $utf8)
Write-Host "  OK: pinned numpy==2.4.4 pandas==2.3.3 pyarrow==23.0.1 scipy==1.17.1" -ForegroundColor Green

Write-Host "`n===== STEP 2: post-conditions =====" -ForegroundColor Cyan
$after = [System.IO.File]::ReadAllText("$pwd\$wf")
if ($after -notmatch 'pandas==2\.3\.3') { throw "POSTCOND FAIL: pandas pin missing." }
if ($after -match 'pip install numpy pandas pyarrow scipy(\s|$)') { throw "POSTCOND FAIL: unpinned line still present." }
if ($after -match "`t") { throw "POSTCOND FAIL: TAB introduced." }
# YAML sanity: the pip line must still be indented under the run: block (6 leading spaces)
if ($after -notmatch '(?m)^          pip install numpy==2\.4\.4 pandas==2\.3\.3 pyarrow==23\.0\.1 scipy==1\.17\.1$') {
    throw "POSTCOND FAIL: pinned line indentation/exact-form check failed."
}
Write-Host "  POSTCOND OK: pinned line present, unpinned gone, no TAB, indentation intact." -ForegroundColor Green

Write-Host "`n===== STEP 3: stage EXACTLY the workflow file, GATE (empty-safe) =====" -ForegroundColor Cyan
git reset | Out-Null
git add $wf
$staged   = @(git diff --cached --name-only)
$intended = @(".github/workflows/data_freshness.yml")
if ($staged.Count -eq 0) {
    Write-Host "  NOTHING STAGED -- file already matches committed state (prior run)." -ForegroundColor Yellow
} elseif (Compare-Object $intended ($staged | Sort-Object)) {
    Write-Host "  STAGED SET MISMATCH:" -ForegroundColor Red
    git diff --cached --name-only
    throw "staged set != intended (data_freshness.yml only)."
} else {
    Write-Host "  STAGED SET OK: data_freshness.yml only" -ForegroundColor Green
    git diff --cached --name-status
}

Write-Host "`n===== PREP COMPLETE -- review 'git diff --cached', then commit with the block in chat. =====" -ForegroundColor Cyan
