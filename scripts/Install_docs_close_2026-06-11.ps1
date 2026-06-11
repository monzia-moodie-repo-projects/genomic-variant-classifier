# Install_docs_close_2026-06-11.ps1
# Places the 2026-06-11 session doc and appends the CHANGELOG/ROADMAP blocks into the
# docs/ canonicals, then regenerates docs/ROADMAP.docx. Idempotent, no-BOM, absolute paths.
# Does NOT commit -- prints the exact git sequence for manual review. Author: Monzia Moodie.

$ErrorActionPreference = "Stop"
$repo = "C:\Projects\genomic-variant-classifier"
$dl   = "$HOME\Downloads"
$enc  = New-Object System.Text.UTF8Encoding($false)   # no BOM

function Append-IfAbsent($target, $sourceName, $marker) {
    $src = Join-Path $dl $sourceName
    if (-not (Test-Path $src)) { throw "MISSING source: $src" }
    $existing = if (Test-Path $target) { [System.IO.File]::ReadAllText($target) } else { "" }
    if ($existing.Contains($marker)) {
        Write-Host "SKIP (marker present): $target"
        return
    }
    $block = [System.IO.File]::ReadAllText($src)
    [System.IO.File]::AppendAllText($target, $block, $enc)
    Write-Host "APPENDED: $target"
}

# 1. Session file -> docs/sessions/ (skip if already present; do not clobber)
$sessSrc = Join-Path $dl  "SESSION_2026-06-11_ci-and-schema-gate.md"
$sessDst = Join-Path $repo "docs\sessions\SESSION_2026-06-11_ci-and-schema-gate.md"
if (-not (Test-Path $sessSrc)) { throw "MISSING source: $sessSrc" }
if (Test-Path $sessDst) {
    Write-Host "SKIP (exists): $sessDst"
} else {
    Copy-Item $sessSrc $sessDst
    Write-Host "PLACED: $sessDst"
}

# 2. CHANGELOG append (marker = the new section heading)
Append-IfAbsent (Join-Path $repo "docs\CHANGELOG.md") `
    "CHANGELOG_append_2026-06-11.md" `
    "## 2026-06-11 -- Schema-drift activation + preflight gate"

# 3. ROADMAP append (marker = the new section heading)
Append-IfAbsent (Join-Path $repo "docs\ROADMAP.md") `
    "ROADMAP_append_2026-06-11.md" `
    "## Backlog additions -- 2026-06-11 (drift wiring + schema gate)"

# 4. Regenerate ROADMAP.docx from the updated ROADMAP.md (non-fatal if the generator hiccups)
Push-Location $repo
try {
    $before = if (Test-Path "docs\ROADMAP.docx") { (Get-Item "docs\ROADMAP.docx").Length } else { 0 }
    python scripts\make_roadmap_docx.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN: make_roadmap_docx.py exit $LASTEXITCODE -- regenerate manually before committing the docx."
    } else {
        $after = (Get-Item "docs\ROADMAP.docx").Length
        Write-Host "REGENERATED: docs\ROADMAP.docx ($before -> $after bytes)"
    }
} catch {
    Write-Host "WARN: docx regen failed: $($_.Exception.Message) -- regenerate manually."
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Pre-commit check (review before staging):"
Push-Location $repo
git status --short
Pop-Location
