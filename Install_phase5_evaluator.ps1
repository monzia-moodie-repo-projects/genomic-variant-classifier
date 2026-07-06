# Install_phase5_evaluator.ps1 -- Monzia Moodie
# Phase 5 (independent track): land lazy-sklearn loader + first-class F1 in evaluator.py.
#
# This installs a COMPLETE, root-cause change (F1 was never computed -- a real gap for a clinical
# metric; sklearn was imported eagerly -- a real import-coupling). It is NOT a workaround or deferral.
# The mechanism is an anchored, idempotent edit-script (verified before and after); the CHANGE it
# lands is a proper resolution.
#
# Option-3 sequence (streamlined + verify-first; tree is already green & orchestrator reverted):
#   STEP 0  verify orchestrator is STILL in the reverted green state (no _factory/_lazy)  [1 sec]
#   STEP 1  anchor PRE-FLIGHT: every edit anchor matches the real evaluator.py exactly    [<1 sec]
#   STEP 2  place test, apply edit-script (idempotent, .bak, ast-guard)
#   STEP 3  POST gates: imports-without-sklearn, F1 wired, FULL pytest GREEN, liveness 22/22
# Aborts on ANY failure. Nothing irreversible. READ-FIRST.

$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\genomic-variant-classifier"
$orch  = "src\genomic_variant_classifier\agent_layer\orchestrator.py"
$eval  = "src\genomic_variant_classifier\evaluation\evaluator.py"

function Fail($m) { Write-Host "ABORT: $m" -ForegroundColor Red; exit 1 }

Write-Host "=== STEP 0: verify orchestrator is STILL the reverted green eager state ===" -ForegroundColor Cyan
$o = Get-Content $orch -Raw
$hasFactory = ($o -match 'def _factory') -or ($o -match '_lazy\(')
$hasEager   = $o -match 'from genomic_variant_classifier\.agent_layer\.agents\.data_freshness_agent import DataFreshnessAgent'
Write-Host ("  eager_state={0} factory_markers={1}" -f $hasEager, $hasFactory)
if ($hasFactory -or -not $hasEager) {
    Fail "orchestrator is NOT in the reverted green state (eager=$hasEager factory=$hasFactory). The green baseline cannot be assumed. Re-run the revert before Phase 5."
}
Write-Host "  CONFIRMED: orchestrator is reverted + green baseline holds (last run: 1491 passed)." -ForegroundColor Green

Write-Host "=== STEP 1: anchor PRE-FLIGHT (before any pytest; this is the check that was missing) ===" -ForegroundColor Cyan
Copy-Item "$env:USERPROFILE\Downloads\patch_evaluator_phase5.py" "patch_evaluator_phase5.py" -Force
Unblock-File "patch_evaluator_phase5.py"
$pf = @"
import importlib.util, sys
ev = open(r'$eval', encoding='utf-8').read()
spec = importlib.util.spec_from_file_location('p5', r'patch_evaluator_phase5.py')
p5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p5)
bad = [(n, ev.count(o)) for n, o, _ in p5.EDITS if ev.count(o) != 1]
if bad:
    print('  ANCHOR PRE-FLIGHT FAILED (anchor: count):', bad); sys.exit(1)
print('  all', len(p5.EDITS), 'edit anchors match the real evaluator.py exactly: OK')
"@
$pf | python -
if ($LASTEXITCODE -ne 0) { Fail "evaluator anchors do not match the real file. Fix the edit-script BEFORE pytest. (No time wasted -- this is the gate we added after the last abort.)" }

Write-Host "=== STEP 2: place CORRECTED Phase 5 test + apply the edit-script ===" -ForegroundColor Cyan
# Clean stale scratch from the prior aborted run (the misnamed .phase5_applied held pre-phase5 content).
Remove-Item "$eval.phase5_applied" -ErrorAction SilentlyContinue
# The corrected test isolates the sklearn-block check in a SUBPROCESS, so it cannot pollute the
# parent interpreter's sys.modules (the prior buggy test deleted sklearn from sys.modules in-process,
# changing class identity and breaking joblib.dump in later model-serialization tests).
Copy-Item "$env:USERPROFILE\Downloads\test_evaluator_phase5.py" "tests\unit\test_evaluator_phase5.py" -Force
Unblock-File "tests\unit\test_evaluator_phase5.py"
python patch_evaluator_phase5.py
if ($LASTEXITCODE -ne 0) { Fail "edit-script failed (exit $LASTEXITCODE)." }

Write-Host "=== STEP 3 (POST): evaluator imports WITHOUT sklearn + F1 field present ===" -ForegroundColor Cyan
$blk = @'
import builtins, importlib
real = builtins.__import__
def b(n,*a,**k):
    if n=="sklearn" or n.startswith("sklearn."): raise ModuleNotFoundError("blocked")
    return real(n,*a,**k)
builtins.__import__ = b
m = importlib.import_module("genomic_variant_classifier.evaluation.evaluator")
assert hasattr(m,"ClinicalEvaluator") and hasattr(m,"_ensure_sklearn")
assert "f1" in m.EvaluationReport.__dataclass_fields__
print("  evaluator imports w/o sklearn + F1 field present: OK")
'@
$blk | python -
if ($LASTEXITCODE -ne 0) { Fail "evaluator does not import without sklearn, or F1 field missing." }

Write-Host "=== STEP 3 (POST): FULL pytest must be GREEN (1491 baseline + 4 new F1 tests) ===" -ForegroundColor Cyan
python -m pytest -q --tb=short 2>&1 | Select-Object -Last 8
$post = $LASTEXITCODE
Write-Host "  pytest POST exit: $post"
if ($post -ne 0) { Fail "FULL pytest not green after Phase 5. Investigate before proceeding." }

Write-Host "=== STEP 3 (POST): agent liveness still 22/22 ===" -ForegroundColor Cyan
python scripts\check_agents_active.py
if ($LASTEXITCODE -ne 0) { Fail "liveness POST not clean." }

Write-Host "========================================================" -ForegroundColor Green
Write-Host "PHASE 5 GREEN: lazy-sklearn loader + first-class F1 landed on a verified-green tree." -ForegroundColor Green
Write-Host "F1 computed at 0.5 threshold (same as MCC), in EvaluationReport, printed, and in compare_models." -ForegroundColor Green
Write-Host "Remove the edit-script's evaluator.py.bak before committing. Next: Phase 1 (lazy registry + IR)." -ForegroundColor Green
