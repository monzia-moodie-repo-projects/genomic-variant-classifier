# Commit_pandas3_investigation.ps1 -- close the pandas-3 arc: clean this-arc .bak files, place docs +
# pin-comment update, stage EXACTLY the intended set, gate the staged set, commit BOM-free. Then YOU push.
# Run from repo root. pandas must be 2.3.3 (post-rollback).
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier
$dl = "$env:USERPROFILE\Downloads"

# --- 0. preconditions ---
$pv = (python -c "import pandas; print(pandas.__version__)").Trim()
"pandas: $pv (must be 2.3.3 post-rollback)"
if ($pv -ne "2.3.3") { throw "ABORT: pandas is $pv, expected 2.3.3. Run the rollback first." }

# --- 1. remove ONLY this arc's two .bak files (NOT the ~48 pre-existing) ---
$arcBaks = @(
  "src\genomic_variant_classifier\data\real_data_prep.py.bak",
  "tests\unit\test_annotation_policy_baseline.py.bak"
)
foreach ($b in $arcBaks) {
  if (Test-Path $b) { Remove-Item $b -Force; "removed $b" } else { "absent (ok): $b" }
}

# --- 2. place the two docs from Downloads (content-marker verified) ---
$cl = Get-ChildItem $dl -Filter "CHANGELOG_entry*.md" |
      Where-Object { Select-String -Path $_.FullName -Pattern 'pandas 3.0.4 upgrade attempted' -Quiet } |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $cl) { throw "ABORT: CHANGELOG_entry.md not in $dl (marker missing)." }
$sess = Get-ChildItem $dl -Filter "SESSION_2026-06-29_pandas3-investigation*.md" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $sess) { throw "ABORT: SESSION doc not in $dl." }

# Prepend the CHANGELOG entry (BOM-free) above the existing content.
$existing = [System.IO.File]::ReadAllText("$pwd\docs\CHANGELOG.md")
$entry    = [System.IO.File]::ReadAllText($cl.FullName)
$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText("$pwd\docs\CHANGELOG.md", $entry + "`n" + $existing, $enc)
"CHANGELOG.md prepended (BOM-free)"

New-Item -ItemType Directory -Force -Path "docs\sessions" | Out-Null
Copy-Item $sess.FullName "docs\sessions\SESSION_2026-06-29_pandas3-investigation.md" -Force
"SESSION doc placed"

# --- 3. update ONLY the line-1 pin comment in requirements.txt (pins themselves UNCHANGED) ---
$req = [System.IO.File]::ReadAllText("$pwd\requirements.txt")
$oldComment = "#MANUAL EDIT 2026-04-29 (Monzia): pandas pinned to 2.3.3 to avoid pandas 3.0 string-dtype break."
$newComment = "#MANUAL EDIT 2026-04-29 (Monzia): pandas pinned to 2.3.3 to avoid pandas 3.0 string-dtype break." + "`n" +
              "# 2026-06-29: pandas 3.0.4 upgrade ATTEMPTED + rolled back. Data-prep proven byte-equivalent on" + "`n" +
              "# 3.0.4 (string-dtype break does NOT occur), but pandas 3.0.4's Windows cp312 wheel segfaults" + "`n" +
              "# pd.date_range (0xC0000005) across all numpy 2.0.2-2.3.3. BLOCKED; retry when pandas > 3.0.4."
if ($req.Contains("2026-06-29: pandas 3.0.4 upgrade ATTEMPTED")) {
  "pin comment already updated (idempotent)"
} elseif ($req.Contains($oldComment)) {
  $enc2 = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText("$pwd\requirements.txt", $req.Replace($oldComment, $newComment), $enc2)
  "requirements.txt line-1 comment updated (pins unchanged)"
} else {
  throw "ABORT: expected 2026-04-29 pin comment not found in requirements.txt."
}
# Sanity: the pin line MUST still be pandas==2.3.3
if (-not (Select-String -Path requirements.txt -Pattern '^pandas==2\.3\.3$' -Quiet)) {
  throw "ABORT: requirements.txt no longer pins pandas==2.3.3 -- pin was altered unexpectedly."
}

# --- 4. place harness + patchers into scripts/ from Downloads (if not already there) ---
foreach ($s in @("pandas3_equivalence_harness.py","patch_allele_freq_numeric.py",
                 "patch_suppress_downcast_version_aware.py","patch_river_test_fixture.py")) {
  if (-not (Test-Path "scripts\$s")) {
    $src = Join-Path $dl $s
    if (Test-Path $src) { Copy-Item $src "scripts\$s" -Force; "placed scripts\$s" }
    else { throw "ABORT: scripts\$s missing and not in Downloads." }
  } else { "already present: scripts\$s" }
}

# --- 5. stage EXACTLY the intended set ---
$stage = @(
  "src/genomic_variant_classifier/data/real_data_prep.py",
  "tests/unit/test_annotation_policy_baseline.py",
  "scripts/pandas3_equivalence_harness.py",
  "scripts/patch_allele_freq_numeric.py",
  "scripts/patch_suppress_downcast_version_aware.py",
  "scripts/patch_river_test_fixture.py",
  "docs/CHANGELOG.md",
  "docs/sessions/SESSION_2026-06-29_pandas3-investigation.md",
  "requirements.txt"
)
git reset | Out-Null
foreach ($f in $stage) { git add -- $f }

# --- 6. PRE-COMMIT GATE: staged set must EQUAL the intended set exactly ---
$staged = (git diff --cached --name-only) -split "`n" | Where-Object { $_ -ne "" } | Sort-Object
$want   = $stage | Sort-Object
"=== staged files ==="; $staged | ForEach-Object { "  $_" }
$diff = Compare-Object $staged $want
if ($diff) {
  "=== MISMATCH (staged vs intended) ==="; $diff | Format-Table -AutoSize | Out-String | Write-Host
  throw "ABORT: staged set != intended set. Nothing committed."
}
"=== gate PASS: staged set == intended set ($($staged.Count) files) ==="

# --- 7. commit BOM-free via -F ---
$msg = "pandas-3 upgrade attempted + rolled back; kept 3 fixes proven equiv on 2.3.3 (date_range wheel segfault)" + "`n`n" +
       "pandas 3.0.4 data-prep proven byte-equivalent (hash 49e98393, all 7 merges 709, string-dtype break" + "`n" +
       "does NOT occur) but its Windows cp312 wheel segfaults pd.date_range (0xC0000005) across numpy" + "`n" +
       "2.0.2-2.3.3 (not an ABI mismatch -- disproven). Rolled back to 2.3.3. Kept: allele_freq numeric" + "`n" +
       "fix, version-aware _suppress_fillna_downcast (no Pandas4Warning on pandas>=3), river test fixture" + "`n" +
       "off date_range. Pins unchanged at 2.3.3; line-1 comment records the attempt+block. See CHANGELOG +" + "`n" +
       "docs/sessions/SESSION_2026-06-29_pandas3-investigation.md."
$cmFile = "$pwd\.git\CM_pandas3.txt"
$enc3 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($cmFile, $msg, $enc3)
git commit -F $cmFile
$rc = $LASTEXITCODE
Remove-Item $cmFile -Force
if ($rc -ne 0) { throw "ABORT: git commit failed ($rc)." }

"=== committed. HEAD: ==="
git rev-parse HEAD
git log -1 --stat | Select-Object -First 25
"=== NOT pushed yet -- review, then: git push origin main ==="
