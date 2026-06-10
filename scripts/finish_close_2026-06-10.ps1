# finish_close_2026-06-10.ps1
# Corrected session-close for the 2026-06-10 agent-layer repair.
# Canonical docs live under docs/ (confirmed via git ls-files). This script:
#   - removes the stray ROOT ROADMAP.md / CHANGELOG.md created by an earlier Add-Content
#   - places the session doc in docs/sessions/ under a date+suffix name (no clobber)
#   - idempotently appends the CHANGELOG + ROADMAP entries to the docs/ canonicals
#     using no-BOM UTF8 + absolute paths (avoids the known BOM/append hazards)
#   - regenerates docs/ROADMAP.docx defensively (never aborts the close)
#   - gitignores split_health.csv + *.bak
#   - stages each path individually and prints status; does NOT commit (separate block)
# Idempotent: safe to re-run.

$ErrorActionPreference = "Stop"
$root = "C:\Projects\genomic-variant-classifier"
Set-Location $root

$dl       = Join-Path $HOME "Downloads"
$chPath   = Join-Path $root "docs\CHANGELOG.md"
$rmPath   = Join-Path $root "docs\ROADMAP.md"
$docxPath = Join-Path $root "docs\ROADMAP.docx"
$sessDst  = Join-Path $root "docs\sessions\SESSION_2026-06-10_agent-layer.md"
$utf8     = New-Object System.Text.UTF8Encoding($false)

function Assert-File($p) { if (-not (Test-Path $p)) { throw "MISSING canonical file: $p" } }
Assert-File $chPath
Assert-File $rmPath

# 1) Remove the stray ROOT files (canonical CHANGELOG/ROADMAP are under docs/).
foreach ($stray in @("ROADMAP.md","CHANGELOG.md")) {
  $sp = Join-Path $root $stray
  if (Test-Path $sp) { Remove-Item $sp -Force; Write-Host "removed stray: $sp" }
}

# 2) Place the session doc under a suffixed name.
#    docs/sessions/SESSION_2026-06-10.md ALREADY EXISTS (tracked) -- do NOT overwrite it.
$sessRootStray = Join-Path $root "SESSION_2026-06-10.md"
$sessSrc = if (Test-Path $sessRootStray) { $sessRootStray } else { Join-Path $dl "SESSION_2026-06-10.md" }
if (-not (Test-Path $sessSrc)) { throw "session doc source not found (root or Downloads)" }
Move-Item $sessSrc $sessDst -Force
Write-Host "placed session doc: $sessDst"

# 3) Idempotently append canonical entries (no-BOM UTF8, absolute paths).
$chEntry = (Get-Content (Join-Path $dl "CHANGELOG_entry_2026-06-10.md") -Raw).TrimEnd()
$rmEntry = (Get-Content (Join-Path $dl "ROADMAP_entries_2026-06-10.md") -Raw).TrimEnd()

if (-not (Select-String -Path $chPath -SimpleMatch "Agent layer re-wiring (4 -> 13 operational)" -Quiet)) {
  [System.IO.File]::AppendAllText($chPath, "`r`n" + $chEntry + "`r`n", $utf8)
  Write-Host "appended CHANGELOG entry"
} else { Write-Host "CHANGELOG entry already present; skipped" }

if (-not (Select-String -Path $rmPath -SimpleMatch "Backlog additions -- 2026-06-10 (agent layer)" -Quiet)) {
  [System.IO.File]::AppendAllText($rmPath, "`r`n" + $rmEntry + "`r`n", $utf8)
  Write-Host "appended ROADMAP entry"
} else { Write-Host "ROADMAP entry already present; skipped" }

# 4) Regenerate docs/ROADMAP.docx defensively (mtime-verified; never aborts the close).
$docxBefore = if (Test-Path $docxPath) { (Get-Item $docxPath).LastWriteTimeUtc } else { [datetime]::MinValue }
$docxOk = $false
try {
  python -c "import docx" 2>$null
  if ($LASTEXITCODE -ne 0) { Write-Host "installing python-docx..."; pip install python-docx --quiet }
  python scripts\make_roadmap_docx.py
  if ((Test-Path $docxPath) -and ((Get-Item $docxPath).LastWriteTimeUtc -gt $docxBefore)) {
    $docxOk = $true; Write-Host "regenerated docs/ROADMAP.docx"
  } else {
    Write-Host "WARN: make_roadmap_docx.py ran but docs/ROADMAP.docx not updated; committing .md only"
  }
} catch {
  Write-Host "WARN: docx regen failed ($($_.Exception.Message)); committing .md only"
}

# 5) Keep generated artifacts out of git (idempotent).
$gi = Join-Path $root ".gitignore"
$giLines = if (Test-Path $gi) { @(Get-Content $gi) } else { @() }
foreach ($pat in @("split_health.csv","*.bak")) {
  if ($giLines -notcontains $pat) { Add-Content $gi $pat; Write-Host "gitignore += $pat" }
}

# 6) Stage each path individually (one miss cannot atomically abort the rest).
$toAdd = @($chPath, $rmPath, $sessDst, $gi)
if ($docxOk) { $toAdd += $docxPath }
foreach ($f in $toAdd) {
  if (Test-Path $f) { git add -- $f } else { Write-Host "SKIP add (missing): $f" }
}

Write-Host "`n=== git status (REVIEW before committing) ==="
git status --short
