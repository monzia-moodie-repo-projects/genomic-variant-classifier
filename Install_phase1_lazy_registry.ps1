<#
  Install_phase1_lazy_registry.ps1 -- Monzia Moodie

  Phase 1 of the orchestrator redesign: lazy agent registry + widened per-agent guard.

  What it installs:
    - src/genomic_variant_classifier/agent_layer/_lazy_agent.py        (NEW module: _Lazy descriptor)
    - tests/unit/test_lazy_agent.py                                    (NEW)
    - tests/unit/test_orchestrator_lazy_registry.py                    (NEW: CI guarantee, subprocess-isolated)
    - tests/unit/test_orchestrator_agent_isolation.py                 (NEW: per-agent isolation)
    - transforms orchestrator.py via patch_orchestrator_phase1.py:
        * _register_agents -> lazy { "Name": _Lazy("module:Class") } (NO agent imported at construction)
        * run_pipeline per-agent guard widened to wrap CONSTRUCTION (graceful failure, criterion #4)

  Gate sequence (fail-closed -- any gate failure aborts BEFORE the next step):
    STEP 0  verify orchestrator is the eager+green baseline (NOT already lazy, no factory markers)
    STEP 1  anchor PRE-FLIGHT (both edit regions match exactly once) -- <1s, BEFORE any pytest
    STEP 2  place _lazy_agent.py + the 3 tests; apply the edit-script (idempotent, .bak, ast-guard)
    STEP 3  POST: orchestrator constructs under sklearn+torch BLOCK (the lazy guarantee)
    STEP 4  POST: CI simulation -- database_monitor runs sklearn-blocked (the red->green proof)
    STEP 5  POST: FULL pytest GREEN (1495 baseline + new Phase 1 tests)
    STEP 6  POST: agent liveness still 22/22
    STEP 7  POST: the 7 existing wiring tests pass unchanged

  Prereqs: download _lazy_agent.py, patch_orchestrator_phase1.py, preflight_phase1_anchors.py,
  test_lazy_agent.py, test_orchestrator_lazy_registry.py, test_orchestrator_agent_isolation.py to Downloads.
#>

$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\genomic-variant-classifier"

function Fail($msg) { Write-Host "ABORT: $msg" -ForegroundColor Red; exit 1 }

$orch  = "src\genomic_variant_classifier\agent_layer\orchestrator.py"
$lazy  = "src\genomic_variant_classifier\agent_layer\_lazy_agent.py"
$dl    = "$env:USERPROFILE\Downloads"

# Bring the tools into the repo root + tests into place.
foreach ($f in @("patch_orchestrator_phase1.py","preflight_phase1_anchors.py")) {
    Copy-Item "$dl\$f" "." -Force; Unblock-File ".\$f"
}

Write-Host "=== STEP 0: verify orchestrator is the eager+green baseline ===" -ForegroundColor Cyan
$o = Get-Content $orch -Raw
$eager     = $o -match 'from genomic_variant_classifier\.agent_layer\.agents\.data_freshness_agent import DataFreshnessAgent'
$hasLazy   = $o -match 'PHASE1_LAZY_REGISTRY'
$hasGuard  = $o -match 'PHASE1_GUARDED_CONSTRUCTION'
$hasFactory= ($o -match 'def _factory') -or ($o -match '_lazy\(')
Write-Host "  eager_imports=$eager  lazy_sentinel=$hasLazy  guard_sentinel=$hasGuard  factory_markers=$hasFactory"
if (-not $eager)  { Fail "orchestrator is NOT the eager baseline (eager imports missing)." }
if ($hasLazy -or $hasGuard) { Fail "orchestrator already has Phase 1 sentinels -- nothing to do (idempotent)." }
if ($hasFactory) { Fail "orchestrator has unexpected factory markers -- not the clean eager baseline." }
Write-Host "  CONFIRMED: clean eager baseline." -ForegroundColor Green

Write-Host "=== STEP 1: anchor PRE-FLIGHT (both edit regions, before any pytest) ===" -ForegroundColor Cyan
python preflight_phase1_anchors.py
if ($LASTEXITCODE -ne 0) { Fail "anchor pre-flight failed -- edit-script anchors do not match the real file." }

Write-Host "=== STEP 2: place _lazy_agent.py + the 3 tests; apply the edit-script ===" -ForegroundColor Cyan
Copy-Item "$dl\_lazy_agent.py" $lazy -Force; Unblock-File $lazy
foreach ($t in @("test_lazy_agent.py","test_orchestrator_lazy_registry.py","test_orchestrator_agent_isolation.py")) {
    Copy-Item "$dl\$t" "tests\unit\$t" -Force; Unblock-File "tests\unit\$t"
}
python patch_orchestrator_phase1.py
if ($LASTEXITCODE -ne 0) { Fail "edit-script failed (exit $LASTEXITCODE)." }
# Confirm both sentinels landed + the file still parses.
$o2 = Get-Content $orch -Raw
if ($o2 -notmatch 'PHASE1_LAZY_REGISTRY')      { Fail "lazy-registry sentinel missing after edit-script." }
if ($o2 -notmatch 'PHASE1_GUARDED_CONSTRUCTION'){ Fail "guarded-construction sentinel missing after edit-script." }
python -c "import ast,sys; ast.parse(open(r'$orch',encoding='utf-8').read()); print('  orchestrator parses clean')"
if ($LASTEXITCODE -ne 0) { Fail "orchestrator does not parse after edit-script." }

Write-Host "=== STEP 3 (POST): orchestrator constructs under sklearn+torch BLOCK ===" -ForegroundColor Cyan
$blk = @'
import builtins
real = builtins.__import__
_B = ("sklearn","torch","xgboost","lightgbm","catboost","shap","transformers")
def b(n,*a,**k):
    if any(n==x or n.startswith(x+".") for x in _B):
        raise ModuleNotFoundError("blocked: "+n)
    return real(n,*a,**k)
builtins.__import__ = b
from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
from genomic_variant_classifier.agent_layer.shared_state import SharedState
o = Orchestrator(SharedState(), dry_run=True)
from genomic_variant_classifier.agent_layer._lazy_agent import _Lazy
assert all(isinstance(v,_Lazy) for v in o._agent_registry.values()), "registry values must be _Lazy"
print("  constructed under heavy-dep block: OK ("+str(len(o._agent_registry))+" lazy agents)")
'@
$blk | python -
if ($LASTEXITCODE -ne 0) { Fail "orchestrator does NOT construct under sklearn+torch block (lazy registry broken)." }

Write-Host "=== STEP 4 (POST): CI simulation -- database_monitor runs sklearn-blocked ===" -ForegroundColor Cyan
$ci = @'
import builtins
real = builtins.__import__
def b(n,*a,**k):
    if n=="sklearn" or n.startswith("sklearn."):
        raise ModuleNotFoundError("blocked: "+n)
    return real(n,*a,**k)
builtins.__import__ = b
from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
from genomic_variant_classifier.agent_layer.shared_state import SharedState
o = Orchestrator(SharedState(), dry_run=True)
r = o.run_pipeline("database_monitor")
assert "DatabaseFreshnessMonitorAgent" in r, r
assert r["DatabaseFreshnessMonitorAgent"].get("action") != "error", r
print("  database_monitor ran clean under sklearn-block: OK -> "+str(list(r.keys())))
'@
$ci | python -
if ($LASTEXITCODE -ne 0) { Fail "CI simulation failed -- database_monitor does not run under sklearn-block." }
Write-Host "  >>> This is the red->green CI proof: the Data Freshness workflow path now runs sklearn-free." -ForegroundColor Green

Write-Host "=== STEP 5 (POST): FULL pytest must be GREEN (1495 baseline + new Phase 1 tests) ===" -ForegroundColor Cyan
python -m pytest -q --tb=short 2>&1 | Select-Object -Last 8
$post = $LASTEXITCODE
Write-Host "  pytest POST exit: $post"
if ($post -ne 0) { Fail "FULL pytest not green after Phase 1. Investigate before proceeding." }

Write-Host "=== STEP 6 (POST): agent liveness still 22/22 ===" -ForegroundColor Cyan
python scripts\check_agents_active.py
if ($LASTEXITCODE -ne 0) { Fail "liveness POST not clean (a registered agent went dormant)." }

Write-Host "=== STEP 7 (POST): the existing wiring tests pass unchanged ===" -ForegroundColor Cyan
python -m pytest tests\unit\ -k "wiring" -q 2>&1 | Select-Object -Last 6
if ($LASTEXITCODE -ne 0) { Fail "existing wiring tests regressed under the lazy registry." }

Write-Host "========================================================" -ForegroundColor Green
Write-Host "PHASE 1 GREEN: lazy agent registry + widened guard landed on a verified tree." -ForegroundColor Green
Write-Host "  - orchestrator imports ZERO agent modules at construction (sklearn/torch deferred)" -ForegroundColor Green
Write-Host "  - database_monitor runs sklearn-free (the Data Freshness CI fix)" -ForegroundColor Green
Write-Host "  - one agent's missing optional dep is isolated; the pipeline survives" -ForegroundColor Green
Write-Host "  - drift-agent from_default_baseline routing preserved; 22/22 agents active" -ForegroundColor Green
Write-Host "Next: re-run the CI1 simulation to confirm red->green, then commit + push Phase 5 + Phase 1 together." -ForegroundColor Green
Write-Host "Remove the edit-script's orchestrator.py.bak before committing." -ForegroundColor Yellow
