<#
Install_EVE_EntryName_Resolution.ps1
====================================
Applies the EVE entry-name -> HGNC resolution fix (Run 17) end to end.

WHAT THIS FIXES
  EVE per-protein files are named by UniProt ENTRY NAME (1433G_HUMAN.csv), not by
  HGNC symbol. The connector keyed its lookup on the entry-name prefix ("1433G");
  the cohort keys on HGNC ("YWHAG"), so the join silently missed and eve_score
  stayed 0.5 for the whole 1.49M-variant run. Empirically 0/2 matched; 2/2 after.

WHAT IT TOUCHES (all anchor-based, idempotent, LF-safe, each with a .bak backup)
  1. scripts/build_uniprot_index.py        + entry_name column (UniProt 'id' field)
  2. src/.../data/eve.py                    + entry-name resolver + fail-loud guard
  3. src/.../data/real_data_prep.py         + eve_entry_map_path field + ctor thread
  4. scripts/run_phase2_eval.py             + --eve-entry-map flag + thread
     scripts/regen_splits_local.py          + --eve-path + --eve-entry-map (option A mirror)
  5. scripts/launch_run17_baseline.sh       + --eve-entry-map $UNIPROT_INDEX + comment fix
  6. tests/unit/test_eve_entry_name_resolution.py  (new)

PREREQUISITE AFTER THIS SCRIPT: rebuild the UniProt index (it now needs the
entry_name column) -- this script PROMPTS you to run it; it does NOT auto-download.

USAGE (from repo root):
  .\Install_EVE_EntryName_Resolution.ps1            # apply
  .\Install_EVE_EntryName_Resolution.ps1 -Check     # dry-run, report only
#>
[CmdletBinding()]
param([switch]$Check)

$ErrorActionPreference = "Stop"
$repo = (Get-Location).Path
Write-Host "== EVE entry-name resolution installer ==" -ForegroundColor Cyan
Write-Host "repo: $repo`n"

# ---- 0. Pre-checks: repo root + target files present -----------------------
$required = @(
  "scripts\build_uniprot_index.py",
  "src\genomic_variant_classifier\data\eve.py",
  "src\genomic_variant_classifier\data\real_data_prep.py",
  "scripts\run_phase2_eval.py",
  "scripts\regen_splits_local.py",
  "scripts\launch_run17_baseline.sh"
)
$missing = $required | Where-Object { -not (Test-Path (Join-Path $repo $_)) }
if ($missing) {
  Write-Host "ABORT: not at repo root or files missing:" -ForegroundColor Red
  $missing | ForEach-Object { Write-Host "  - $_" }
  exit 1
}
Write-Host "[0] pre-check: all 6 target files present" -ForegroundColor Green

# ---- 1. Stage patchers + test into the repo --------------------------------
$patchDir = Join-Path $repo "scripts"
$srcThis  = $PSScriptRoot
$patchers = @(
  "patch_build_uniprot_index_entry_name.py",
  "patch_eve_entry_name_resolution.py",
  "patch_real_data_prep_eve_entry_map.py",
  "patch_eval_scripts_eve_entry_map.py",
  "patch_launch_run17_eve_entry_map.py"
)
foreach ($p in $patchers) {
  Copy-Item (Join-Path $srcThis $p) (Join-Path $patchDir $p) -Force
  if (Get-Command Unblock-File -ErrorAction SilentlyContinue) { Unblock-File (Join-Path $patchDir $p) }
}
$testDir = Join-Path $repo "tests\unit"
if (-not (Test-Path $testDir)) { New-Item -ItemType Directory -Path $testDir -Force | Out-Null }
Copy-Item (Join-Path $srcThis "test_eve_entry_name_resolution.py") (Join-Path $testDir "test_eve_entry_name_resolution.py") -Force
Write-Host "[1] staged 5 patchers -> scripts\, 1 test -> tests\unit\" -ForegroundColor Green

$flag = ""
if ($Check) { $flag = "--check" }

function Invoke-Patch([string]$name) {
  Write-Host "`n--- $name $flag ---" -ForegroundColor Yellow
  & python (Join-Path $patchDir $name) $flag
  if ($LASTEXITCODE -ne 0) { Write-Host "PATCH FAILED ($name) exit $LASTEXITCODE" -ForegroundColor Red; exit $LASTEXITCODE }
}

# ---- 2. Apply in dependency order ------------------------------------------
Invoke-Patch "patch_build_uniprot_index_entry_name.py"
Invoke-Patch "patch_eve_entry_name_resolution.py"
Invoke-Patch "patch_real_data_prep_eve_entry_map.py"
Invoke-Patch "patch_eval_scripts_eve_entry_map.py"
Invoke-Patch "patch_launch_run17_eve_entry_map.py"

if ($Check) {
  Write-Host "`n[CHECK COMPLETE] no files modified. Re-run without -Check to apply." -ForegroundColor Cyan
  exit 0
}

# ---- 3. Post-checks: py-compile changed modules + bash -n launch -----------
Write-Host "`n[3] post-checks" -ForegroundColor Yellow
$env:PYTHONPATH = "src"
& python -c "import ast; [ast.parse(open(f,encoding='utf-8').read()) for f in [r'scripts\build_uniprot_index.py', r'src\genomic_variant_classifier\data\eve.py', r'src\genomic_variant_classifier\data\real_data_prep.py', r'scripts\run_phase2_eval.py', r'scripts\regen_splits_local.py']]; print('  py-compile OK (5 modules)')"
if ($LASTEXITCODE -ne 0) { Write-Host "POST-CHECK py-compile FAILED" -ForegroundColor Red; exit 1 }

# CRLF audit on every touched file (Python tolerates it; bash does NOT)
$touched = @(
  "scripts\build_uniprot_index.py","src\genomic_variant_classifier\data\eve.py",
  "src\genomic_variant_classifier\data\real_data_prep.py","scripts\run_phase2_eval.py",
  "scripts\regen_splits_local.py","scripts\launch_run17_baseline.sh"
)
$crlf = @()
foreach ($f in $touched) {
  $bytes = [System.IO.File]::ReadAllBytes((Join-Path $repo $f))
  for ($i=1; $i -lt $bytes.Length; $i++) { if ($bytes[$i] -eq 10 -and $bytes[$i-1] -eq 13) { $crlf += $f; break } }
}
if ($crlf) {
  Write-Host "  POST-CHECK FAIL: CRLF found in:" -ForegroundColor Red
  $crlf | ForEach-Object { Write-Host "    - $_" }
  exit 1
}
Write-Host "  CRLF audit OK (0 of $($touched.Count) files have CRLF)" -ForegroundColor Green

# bash -n the launch script if bash is available
$bash = Get-Command bash -ErrorAction SilentlyContinue
if ($bash) {
  & bash -n (Join-Path $repo "scripts\launch_run17_baseline.sh")
  if ($LASTEXITCODE -ne 0) { Write-Host "  bash -n FAILED" -ForegroundColor Red; exit 1 }
  Write-Host "  bash -n launch_run17_baseline.sh OK" -ForegroundColor Green
} else {
  Write-Host "  (bash not on PATH; skip bash -n -- verify on the VM)" -ForegroundColor DarkYellow
}

# ---- 4. Run the test suite (data-gated tests skip until the index is rebuilt)
Write-Host "`n[4] pytest (entry-name suite + existing eve tests)" -ForegroundColor Yellow
& python -m pytest tests\unit\test_eve_entry_name_resolution.py tests\unit\test_eve.py tests\unit\test_eve_gene_resolution.py -q
if ($LASTEXITCODE -ne 0) { Write-Host "TESTS FAILED" -ForegroundColor Red; exit 1 }

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host " INSTALL COMPLETE. NEXT (REQUIRED) -- rebuild the UniProt index:" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host @"
  # 1. Rebuild the index WITH the entry_name column (~30 MB download, 1-2 min):
  python scripts\build_uniprot_index.py

  # 2. Confirm it carries entry_name + resolves the real corpus (these tests now run):
  python -m pytest tests\unit\test_eve_entry_name_resolution.py -v -k "entry_name or corpus"

  # 3. Re-run the empirical probe -> expect 'HGNC-key matches: 2 / 2' (was 0/2).
  #    (see RUNBOOK_EVE_EntryName.md, step 3)

  # 4. Re-upload the rebuilt parquet to Drive, then re-stage to the VM.
  # 5. THEN unblock the launch EVE path + variant_files-only staging (RUNBOOK steps 6-8).
"@ -ForegroundColor White
