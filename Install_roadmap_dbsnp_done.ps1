# Install_roadmap_dbsnp_done.ps1
# Update ROADMAP.md to record dbSNP as done+verified (audit 2026-07-01: 37.45% end-to-end
# cohort coverage; parquet dbsnp157_cohort.parquet built 2026-06-26, 3.75M rows, 46% AF>0).
# Two anchored edits:
#   L63  table row: "stub-mode step; activation = data + config" -> done+verified
#   L88  remove dbSNP from the "remaining Phase-D connectors" list (AlphaFold/COSMIC/TCGA/KEGG remain)
# L207 (dbSNP INDEX parquet, a separate artifact from the AF cohort parquet) is left unchanged.
# Anchored, idempotent, abort-on-mismatch. Does NOT commit.

$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\genomic-variant-classifier"
$f = "docs\ROADMAP.md"
if (-not (Test-Path $f)) { throw "ABORT: $f not found." }
$utf8 = New-Object System.Text.UTF8Encoding($false)
$raw  = [System.IO.File]::ReadAllText("$pwd\$f")

# idempotency: if already updated, no-op
if ($raw.Contains("dbsnp_af | free | DONE+VERIFIED")) {
    Write-Host "  SKIP: ROADMAP already records dbSNP done+verified (idempotent)." -ForegroundColor Yellow
    return
}

# --- Edit 1: the L63 table row (anchor on the exact current cell text) ---
$old63 = "| dbSNP/RefSNP | dbsnp_af | free | stub-mode step; activation = data + config |"
$new63 = "| dbSNP/RefSNP | dbsnp_af | free | DONE+VERIFIED 2026-06-26 (build_dbsnp_parquet.py; dbsnp157_cohort.parquet 3.75M rows, 46% AF>0). End-to-end audit 2026-07-01: 37.45% cohort coverage, dbsnp_af>0 confirmed through DbSNPConnector. Wired: --dbsnp-path -> AnnotationConfig -> real_data_prep step 10. |"
$n63 = ([regex]::Matches($raw, [regex]::Escape($old63))).Count
if ($n63 -ne 1) { throw "ABORT: L63 anchor found $n63 times (expected 1). Manual review." }

# --- Edit 2: the L88 "remaining" line (drop dbSNP; keep the rest) ---
$old88 = "- Remaining Phase-D connectors: activate dbSNP + AlphaFold-structure stub steps (data + config), then build COSMIC / TCGA / KEGG."
$new88 = "- Remaining Phase-D connectors: AlphaFold-structure stub steps (data + config), then build COSMIC / TCGA / KEGG. (dbSNP DONE+VERIFIED 2026-06-26; see the dbSNP row above and the 2026-07-01 end-to-end coverage audit.)"
$n88 = ([regex]::Matches($raw, [regex]::Escape($old88))).Count
if ($n88 -ne 1) { throw "ABORT: L88 anchor found $n88 times (expected 1). Manual review." }

Write-Host "===== STEP 1: apply both anchored edits =====" -ForegroundColor Cyan
$patched = $raw.Replace($old63, $new63).Replace($old88, $new88)
if ($patched -eq $raw) { throw "ABORT: no change produced." }
[System.IO.File]::WriteAllText("$pwd\$f", $patched, $utf8)
Write-Host "  OK: L63 + L88 updated." -ForegroundColor Green

Write-Host "`n===== STEP 2: post-conditions =====" -ForegroundColor Cyan
$after = [System.IO.File]::ReadAllText("$pwd\$f")
if ($after -notmatch 'dbsnp_af \| free \| DONE\+VERIFIED') { throw "POSTCOND FAIL: L63 update missing." }
if ($after -match 'activate dbSNP \+ AlphaFold') { throw "POSTCOND FAIL: L88 still lists dbSNP as to-activate." }
if ($after -notmatch 'AlphaFold-structure stub steps \(data \+ config\), then build COSMIC') { throw "POSTCOND FAIL: L88 rewrite missing." }
# L207 must be untouched
if (($after -split "`n" | Select-String 'dbSNP index parquet').Count -ne 1) { throw "POSTCOND FAIL: L207 index row disturbed." }
Write-Host "  POSTCOND OK: L63 done+verified, L88 dbSNP removed, L207 index row intact." -ForegroundColor Green

Write-Host "`n===== STEP 3: stage EXACTLY ROADMAP.md, GATE (empty-safe) =====" -ForegroundColor Cyan
git reset | Out-Null
git add $f
$staged   = @(git diff --cached --name-only)
$intended = @("docs/ROADMAP.md")
if ($staged.Count -eq 0) {
    Write-Host "  NOTHING STAGED -- already matches committed state." -ForegroundColor Yellow
} elseif (Compare-Object $intended ($staged | Sort-Object)) {
    Write-Host "  STAGED SET MISMATCH:" -ForegroundColor Red
    git diff --cached --name-only
    throw "staged set != intended (ROADMAP.md only)."
} else {
    Write-Host "  STAGED SET OK: ROADMAP.md only" -ForegroundColor Green
    git diff --cached --stat
}
Write-Host "`n===== PREP COMPLETE =====" -ForegroundColor Cyan
Write-Host "NOTE: ROADMAP.docx also exists; per the living-doc rule it should be regenerated" -ForegroundColor Yellow
Write-Host "to match. Flagging -- not auto-editing the .docx here." -ForegroundColor Yellow
