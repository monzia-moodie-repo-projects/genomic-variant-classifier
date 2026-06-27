# Install_SessionDocs_2026-06-26.ps1
# Places the session doc + prepends the CHANGELOG entry, BOM-free UTF-8.
# Run from repo root: C:\Projects\genomic-variant-classifier
#
# WHY BOM-free + [System.IO.File]: the existing CHANGELOG.md is BOM-free UTF-8 (verified: first bytes
# 35,35,32 = "## "), and Set-Content -Encoding UTF8 / Add-Content / >> would either inject a BOM or
# corrupt non-ASCII (the visible mojibake in the current tail). Our docs are ASCII-only, but we still
# write via [System.IO.File] with UTF8Encoding($false) to match the file and avoid any BOM/CRLF surprise.

$ErrorActionPreference = "Stop"
$repo = (Get-Location).Path
if (-not (Test-Path (Join-Path $repo "docs\CHANGELOG.md"))) {
    throw "Not at repo root (docs\CHANGELOG.md not found). cd to C:\Projects\genomic-variant-classifier first."
}

$dl = Join-Path $HOME "Downloads"
$sessionSrc   = Get-ChildItem $dl -Filter "SESSION_2026-06-26.md" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$changelogSrc = Get-ChildItem $dl -Filter "CHANGELOG_entry_2026-06-26.md" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $sessionSrc)   { throw "SESSION_2026-06-26.md not in Downloads -- download it from chat first." }
if (-not $changelogSrc) { throw "CHANGELOG_entry_2026-06-26.md not in Downloads -- download it from chat first." }

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# --- 1. Session doc: place fresh (must NOT already exist) ---
$sessionDst = Join-Path $repo "docs\sessions\SESSION_2026-06-26.md"
if (Test-Path $sessionDst) { throw "docs\sessions\SESSION_2026-06-26.md already exists -- refusing to clobber. Inspect/merge manually." }
$sessionBody = [System.IO.File]::ReadAllText($sessionSrc.FullName, $utf8NoBom)
[System.IO.File]::WriteAllText($sessionDst, $sessionBody, $utf8NoBom)
Write-Host "OK: wrote $sessionDst ($($sessionBody.Length) chars)" -ForegroundColor Green

# --- 2. CHANGELOG: PREPEND the new entry at the very top (newest-on-top), idempotent ---
$clPath = Join-Path $repo "docs\CHANGELOG.md"
$clText = [System.IO.File]::ReadAllText($clPath, $utf8NoBom)
if ($clText -match "## 2026-06-26 -- OMIM 88-bug fix") {
    Write-Host "IDEMPOTENT: CHANGELOG already has the 2026-06-26 entry; skipping prepend." -ForegroundColor Yellow
} else {
    $entry = [System.IO.File]::ReadAllText($changelogSrc.FullName, $utf8NoBom)
    # Backup first (BOM-free, preserve bytes)
    $bak = "$clPath.pre_2026-06-26.bak"
    if (-not (Test-Path $bak)) { [System.IO.File]::WriteAllText($bak, $clText, $utf8NoBom); Write-Host "OK: backup -> $bak" }
    $newText = $entry.TrimEnd() + "`n`n" + $clText
    [System.IO.File]::WriteAllText($clPath, $newText, $utf8NoBom)
    Write-Host "OK: prepended 2026-06-26 entry to CHANGELOG.md" -ForegroundColor Green
}

# --- 3. POST-CHECKS ---
Write-Host "`n=== POST-CHECKS ===" -ForegroundColor Cyan
# 3a. No BOM introduced
$b = [System.IO.File]::ReadAllBytes($clPath)[0..2]
if ($b[0] -eq 239 -and $b[1] -eq 187 -and $b[2] -eq 191) { Write-Host "  FAIL: BOM present in CHANGELOG" -ForegroundColor Red }
else { Write-Host "  OK: CHANGELOG BOM-free (first bytes $($b -join ','))" -ForegroundColor Green }
# 3b. Entry present exactly once
$cnt = ([regex]::Matches($clText, "## 2026-06-26 -- OMIM 88-bug fix")).Count
$cntNew = ([regex]::Matches([System.IO.File]::ReadAllText($clPath,$utf8NoBom), "## 2026-06-26 -- OMIM 88-bug fix")).Count
Write-Host "  CHANGELOG 2026-06-26 header count: $cntNew (expect 1)"
# 3c. Session doc readable + has the contract line
$sd = [System.IO.File]::ReadAllText($sessionDst, $utf8NoBom)
if ($sd -match "3,155,973" -and $sd -match "0.9999") { Write-Host "  OK: session doc has key facts (88->3.16M, 0.9999 collinearity)" -ForegroundColor Green }
else { Write-Host "  WARN: session doc missing expected facts" -ForegroundColor Yellow }
Write-Host "`nDone. Review: git diff --stat ; then git add docs/ ; git commit" -ForegroundColor Cyan
