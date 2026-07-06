#requires -Version 5.1
<#
  Install_unit2_reactivate_message_bus.ps1
  ------------------------------------------------------------------
  Unit 2 -- reactivate test_message_bus.py as a collected pytest module.

  Correctness of the GATES themselves matters as much as the fix. Pass/fail is
  determined by pytest's EXIT CODE plus explicit count extraction from the
  SUMMARY line -- never by substring-matching the word "error"/"failed" against
  the whole output (which would false-green or false-red). Every pytest run has
  stdin redirected from an EMPTY file, so input() hits EOF instantly and can
  NEVER block on a prompt.

  HARD GATES (any failure => auto-rollback; nothing committed):
    STEP 0  preconditions + backup
    STEP 1  record CURRENT (pre-fix) standalone result (informational only)
    STEP 2  apply edit-script in place + independent structural post-conditions
    STEP 3  git mv into tests/unit/ (dissolves quarantine; testpaths untouched)
    STEP 4  collected pytest of moved file: exit 0 AND exactly 35 passed, 0 failed/error
    STEP 5  tests/unit --collect-only: exit 0 AND 0 collection errors (scipy-victim guard)
    STEP 6  FULL suite: exit 0 AND 0 failed AND 0 error
    STEP 7  summary + separate commit block
#>

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$OLD    = "src\genomic_variant_classifier\agent_layer\test_message_bus.py"
$NEW    = "tests\unit\test_message_bus.py"
$EDIT   = "apply_unit2_reactivate_message_bus.py"
$BACKUP = "$env:TEMP\tmb_unit2_backup.py"

function Section($n){ Write-Host "`n===== $n =====" -ForegroundColor Cyan }
function OK($m){ Write-Host "  OK: $m" -ForegroundColor Green }
function FailMsg($m){ Write-Host "  FAIL: $m" -ForegroundColor Red }

# Run pytest with stdin closed; return an object with ExitCode + parsed summary counts.
# We DO NOT infer pass/fail from arbitrary substrings: exit code is authoritative,
# and counts are parsed from the summary line for a precise, logged assertion.
function Invoke-Pytest([string]$pytestArgs) {
  # Non-interactive by construction: stdin is redirected from an EMPTY file, so any
  # input() call hits EOF immediately (raises EOFError) and can NEVER block the run.
  $tmp     = New-TemporaryFile
  $errf    = New-TemporaryFile
  $emptyIn = New-TemporaryFile
  $argList = @("-u", "-m", "pytest") + ($pytestArgs -split ' ' | Where-Object { $_ -ne '' }) + @("-p", "no:cacheprovider")
  $proc = Start-Process -FilePath "python" -ArgumentList $argList -NoNewWindow -Wait -PassThru `
            -RedirectStandardInput $emptyIn.FullName -RedirectStandardOutput $tmp.FullName -RedirectStandardError $errf.FullName
  $code = $proc.ExitCode
  $out  = (Get-Content $tmp.FullName -Raw) + "`n" + (Get-Content $errf.FullName -Raw)
  Remove-Item $tmp.FullName, $errf.FullName, $emptyIn.FullName -Force -ErrorAction SilentlyContinue
  # pytest summary line examples:
  #   "35 passed in 1.2s" | "1552 passed, 7 skipped in 300s" | "3 failed, 10 passed in 2s"
  #   "2 errors in 0.5s"  | "1 error in 0.3s"
  function _num($pat){ $m=[regex]::Match($out, $pat); if($m.Success){[int]$m.Groups[1].Value}else{0} }
  [pscustomobject]@{
    ExitCode = $code
    Passed   = _num '(\d+) passed'
    Failed   = _num '(\d+) failed'
    # Errors parsed ONLY from the final summary line region to avoid matching test ids
  # like 'test_...[ERROR ... 1 error in 0.50s ...]' that echo in -q collect output.
  Errors   = $(
    $sumLine = ($out -split "`n" | Where-Object { $_ -match '(passed|failed|error|skipped|no tests ran)' -and $_ -match ' in \d' } | Select-Object -Last 1)
    if ($sumLine) { $m=[regex]::Match($sumLine, '(\d+) error(?:s)?\b'); if($m.Success){[int]$m.Groups[1].Value}else{0} } else { 0 }
  )
    Skipped  = _num '(\d+) skipped'
    Tail     = ($out -split "`n" | Select-Object -Last 20) -join "`n"
    Raw      = $out
  }
}

# ---------------------------------------------------------------- STEP 0
Section "STEP 0: preconditions + backup"
if (-not (Test-Path ".git")) { throw "not at repo root (no .git)" }
if (-not (Test-Path $OLD))   { throw "OLD path not found: $OLD (already moved?)" }
if (Test-Path $NEW)          { throw "NEW path already exists: $NEW (resolve partial run first)" }
if (-not (Test-Path $EDIT))  { throw "edit-script not in repo root: $EDIT" }
$dirty = git status --porcelain -- $OLD $NEW
if ($dirty) { throw "target paths not clean in git:`n$dirty" }
Copy-Item $OLD $BACKUP -Force
OK "clean preconditions; backed up to $BACKUP"

# ---------------------------------------------------------------- STEP 1
Section "STEP 1: record CURRENT pre-fix standalone result (informational; stdin closed)"
$emptyIn = New-TemporaryFile
$s1out = New-TemporaryFile
try {
  $p = Start-Process -FilePath "python" -ArgumentList @("-u", $OLD) -NoNewWindow -Wait -PassThru `
        -RedirectStandardInput $emptyIn.FullName -RedirectStandardOutput $s1out.FullName -RedirectStandardError "$($s1out.FullName).err"
  (Get-Content $s1out.FullName -ErrorAction SilentlyContinue | Select-Object -Last 3)
} catch {
  Write-Host "  (pre-fix standalone capture skipped: $($_.Exception.Message))"
} finally {
  Remove-Item $emptyIn.FullName, $s1out.FullName, "$($s1out.FullName).err" -Force -ErrorAction SilentlyContinue
}
Write-Host "  (informational only -- not asserted; pre-fix expected ~30/35 with _check_drift errors)"

# ---------------------------------------------------------------- STEP 2
Section "STEP 2: apply edit-script in place (+ independent post-conditions)"
python $EDIT $OLD --in-place
if ($LASTEXITCODE -ne 0) { throw "edit-script aborted (anchor drift/post-condition) -- tree unchanged" }
$verify = @'
import re, sys, ast
s = open(sys.argv[1], encoding="utf-8").read()
ast.parse(s)
errs = []
if re.search(r"^def _test_", s, flags=re.M): errs.append("a def _test_ remains")
n = len(re.findall(r"^def test_", s, flags=re.M))
if n != 35: errs.append(f"expected 35 def test_, found {n}")
if '"_check_drift"' in s: errs.append("_check_drift reference remains")
if "from orchestrator import Orchestrator" in s: errs.append("bare orchestrator import remains")
if "def _isolated_optional_deps(" not in s: errs.append("isolation fixture missing")
if "def _no_interactive_input(" not in s: errs.append("input-guardrail fixture missing")
if "TESTS = [" in s: errs.append("TESTS list remains")
if "def main(" in s: errs.append("main() remains")
if "def _run(" in s: errs.append("_run harness remains")
if any(l.startswith('for _mod in ("ewc_utils"') for l in s.splitlines()):
    errs.append("module-level stub loop remains at col 0")
print(("POSTCOND_FAIL: " + "; ".join(errs)) if errs else "POSTCOND_OK 35 tests")
sys.exit(1 if errs else 0)
'@
$verify | python - $OLD
if ($LASTEXITCODE -ne 0) { throw "post-condition verification failed after edit-script" }
OK "applied; 35 tests; fixtures in; harness/orphans gone; parses"

# ---------------------------------------------------------------- STEP 3
Section "STEP 3: git mv into tests/unit/"
git mv $OLD $NEW
if ($LASTEXITCODE -ne 0) { throw "git mv failed" }
OK "moved -> $NEW"

function Rollback {
  Write-Host "`n!! ROLLBACK: restoring original file + unstaging move" -ForegroundColor Yellow
  git reset -q -- $OLD $NEW 2>$null
  if (Test-Path $NEW) { Remove-Item $NEW -Force }
  Copy-Item $BACKUP $OLD -Force
  git reset -q -- $OLD 2>$null
  Write-Host "   restored $OLD; tree returned to pre-run state" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- STEP 4
Section "STEP 4: collected pytest of moved file (exit 0 AND 35 passed, 0 failed/error; stdin closed)"
$r4 = Invoke-Pytest "`"$NEW`" -q"
Write-Host $r4.Tail
Write-Host ("  parsed -> exit={0} passed={1} failed={2} errors={3} skipped={4}" -f $r4.ExitCode,$r4.Passed,$r4.Failed,$r4.Errors,$r4.Skipped)
if ($r4.ExitCode -ne 0 -or $r4.Passed -ne 35 -or $r4.Failed -ne 0 -or $r4.Errors -ne 0) {
  FailMsg "moved file did not yield exit0 + exactly 35 passed + 0 failed/error"
  Rollback; throw "STEP 4 failed (rolled back)"
}
OK "35 passed, 0 failed, 0 error, exit 0 -- non-interactive"

# ---------------------------------------------------------------- STEP 5
Section "STEP 5: tests/unit --collect-only (EXIT CODE is authoritative; scipy-victim guard)"
$r5 = Invoke-Pytest "tests\unit -q --collect-only"
Write-Host $r5.Tail
# Collection success is decided by the EXIT CODE, not by counting the word "error"
# (test ids such as test_parse_pytest_output[ERROR ...1 error in 0.50s...] contain the
# literal token and would false-trip a substring counter). As a secondary guard we look
# for pytest's REAL collection-error banner, anchored to line starts.
$realCollectErr = [regex]::IsMatch($r5.Raw, '(?m)^(ERROR collecting |=+ ERROR|!+ Interrupted|\d+ errors? during collection)')
$collectedN = 0
$mCol = [regex]::Match($r5.Raw, '(\d+) tests? collected')
if ($mCol.Success) { $collectedN = [int]$mCol.Groups[1].Value }
Write-Host ("  parsed -> exit={0} collected={1} realCollectionError={2}" -f $r5.ExitCode,$collectedN,$realCollectErr)
if ($r5.ExitCode -ne 0 -or $realCollectErr -or $collectedN -lt 1500) {
  FailMsg "collection of tests/unit not clean (exit!=0, real ERROR-collecting banner, or too few collected)"
  Rollback; throw "STEP 5 failed (rolled back)"
}
OK ("tests/unit collects clean -- {0} tests collected, no pollution regression" -f $collectedN)

# ---------------------------------------------------------------- STEP 6
Section "STEP 6: FULL suite (exit 0 AND 0 failed AND 0 error; the long gate; stdin closed)"
$r6 = Invoke-Pytest "tests -q"
Write-Host $r6.Tail
Write-Host ("  parsed -> exit={0} passed={1} failed={2} errors={3} skipped={4}" -f $r6.ExitCode,$r6.Passed,$r6.Failed,$r6.Errors,$r6.Skipped)
# Exit code is authoritative for the full run; Failed/Passed come from the summary line,
# which pytest prints as e.g. "1552 passed, 7 skipped in 300s" (no stray 'N error' from
# test ids because the summary line is matched, not the whole capture). We still require
# exit 0 and a healthy passed count.
if ($r6.ExitCode -ne 0 -or $r6.Failed -ne 0 -or $r6.Passed -lt 1545) {
  FailMsg "full suite not green (need exit 0, 0 failed, >=1545 passed)"
  Rollback; throw "STEP 6 failed (rolled back)"
}
OK ("full suite green: {0} passed, {1} skipped" -f $r6.Passed,$r6.Skipped)

Remove-Item $BACKUP -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- STEP 7
Section "STEP 7: ALL GATES GREEN"
Write-Host @"

==================================================================
 Unit 2 verified end-to-end on REAL code, NON-INTERACTIVELY:
   - exactly 35 tests collected + passing at tests/unit/ (exit 0)
   - quarantine dissolved (collected normally; testpaths untouched)
   - tests/unit collect-only clean (no scipy/torch pollution regression)
   - full suite green ($($r6.Passed) passed, $($r6.Skipped) skipped)
   - STDIN CLOSED on every pytest run -- no path could block on input()
   - pass/fail judged by EXIT CODE + parsed summary counts (no substring guessing)

 $EDIT and run_tmb.log are TOOLS/scratch -- do NOT commit (leave untracked).
 Proceed to the SEPARATE commit block below.
==================================================================
"@ -ForegroundColor Green
