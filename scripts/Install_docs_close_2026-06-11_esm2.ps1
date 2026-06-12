# Install_docs_close_2026-06-11_esm2.ps1
# Idempotent docs-close for the ESM-2 650M LLR fix + train.py wiring session.
# Run from the repo root. The three .md source files must sit alongside this script.
# Author: Monzia Moodie
$ErrorActionPreference = "Stop"
$repo = (Get-Location).Path
$src  = $PSScriptRoot
$marker = "docs-close: ecd0474 esm2-llr+train-wiring"

function Read-Text($p)  { return [System.IO.File]::ReadAllText($p) }
function Write-NoBom($p, $t) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($p, $t, $enc)
}
function Append-NoBom($p, $t) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($p, $t, $enc)
}

# -- 1. PLACE session note (create if absent) --
$sessSrc = Join-Path $src  "SESSION_2026-06-11_esm2-llr-and-train-wiring.md"
$sessDst = Join-Path $repo "docs\sessions\SESSION_2026-06-11_esm2-llr-and-train-wiring.md"
if (Test-Path $sessDst) {
    Write-Host "SKIP  session note (exists): $sessDst"
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path $sessDst) | Out-Null
    Write-NoBom $sessDst (Read-Text $sessSrc)
    Write-Host "PLACED  $sessDst"
}

# -- 2. APPEND CHANGELOG (marker-guarded) --
$clDst = Join-Path $repo "docs\CHANGELOG.md"
if ((Test-Path $clDst) -and (Read-Text $clDst).Contains($marker)) {
    Write-Host "SKIP  docs\CHANGELOG.md (marker present)"
} else {
    Append-NoBom $clDst ("`r`n" + (Read-Text (Join-Path $src "CHANGELOG_append_2026-06-11_esm2.md")))
    Write-Host "APPENDED  docs\CHANGELOG.md"
}

# -- 3. APPEND ROADMAP (marker-guarded) --
$rmDst = Join-Path $repo "docs\ROADMAP.md"
if ((Test-Path $rmDst) -and (Read-Text $rmDst).Contains($marker)) {
    Write-Host "SKIP  docs\ROADMAP.md (marker present)"
} else {
    Append-NoBom $rmDst ("`r`n" + (Read-Text (Join-Path $src "ROADMAP_append_2026-06-11_esm2.md")))
    Write-Host "APPENDED  docs\ROADMAP.md"
}

# -- 4. Regenerate ROADMAP.docx (non-fatal) --
try {
    python scripts\make_roadmap_docx.py
    Write-Host "REGENERATED  docs\ROADMAP.docx"
} catch {
    Write-Warning "ROADMAP.docx regen failed (commit the .md anyway): $_"
}

Write-Host "docs-close complete."
