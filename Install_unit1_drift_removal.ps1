# ============================================================================
# Install_unit1_drift_removal.ps1
# Unit 1: remove the vestigial _check_drift phantom-drift path from
#         TrainingLifecycleAgent, and add a COLLECTED regression test.
#
# Run from repo root:  .\Install_unit1_drift_removal.ps1
#
# Gate battery (aborts on first failure; NOTHING is committed by this script):
#   STEP 0  verify repo root + target + downloaded artifacts
#   STEP 1  place the new collected regression test under tests/unit/
#   STEP 2  API PREFLIGHT on real (pre-fix) code  -> must construct+run cleanly
#   STEP 3  ANTI-VACUITY: run new test on pre-fix code -> MUST FAIL (rc=1)
#   STEP 4  apply the edit-script -> agent edited; py_compile + token post-check
#   STEP 5  run new test on post-fix code -> MUST PASS (rc=0)
#   STEP 6  targeted collected tests (agent registry/liveness) -> green
#   STEP 7  22/22 agent liveness
#   STEP 8  FULL pytest suite -> green
# On success: prints the gated commit block to run separately.
# ============================================================================
$ErrorActionPreference = "Stop"
$DL   = Join-Path $env:USERPROFILE "Downloads"
$AGENT = "src\genomic_variant_classifier\agent_layer\agents\training_lifecycle_agent.py"
$TESTDST = "tests\unit\test_training_lifecycle_drift_removal.py"
$EDIT = "apply_unit1_drift_removal.py"

function Fail($m) { Write-Host "ABORT: $m" -ForegroundColor Red; exit 1 }
function OK($m)   { Write-Host "  OK: $m" -ForegroundColor Green }

Write-Host "===== STEP 0: preconditions =====" -ForegroundColor Cyan
if (-not (Test-Path "pyproject.toml")) { Fail "run from repo root (pyproject.toml not found)" }
if (-not (Test-Path $AGENT)) { Fail "target agent file not found: $AGENT" }
foreach ($f in @($EDIT, "test_training_lifecycle_drift_removal.py")) {
    if (-not (Test-Path (Join-Path $DL $f))) { Fail "missing download: $DL\$f" }
}
# anchors must be present (file at expected pre-fix state)
$agentText = [System.IO.File]::ReadAllText((Join-Path $PWD $AGENT))
if ($agentText -notmatch '_check_drift') { Fail "agent file does not contain _check_drift; unexpected state (already fixed?)" }
OK "repo root, target, and downloaded artifacts present; agent at expected pre-fix state"

Write-Host "===== STEP 1: place collected regression test =====" -ForegroundColor Cyan
Copy-Item (Join-Path $DL "test_training_lifecycle_drift_removal.py") $TESTDST -Force
Copy-Item (Join-Path $DL $EDIT) $EDIT -Force
if (-not (Test-Path $TESTDST)) { Fail "failed to place test at $TESTDST" }
OK "placed $TESTDST and $EDIT"

Write-Host "===== STEP 2: API preflight on real (pre-fix) code =====" -ForegroundColor Cyan
$pre = @'
import sys, tempfile, os
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.message_bus import MessageBus, DATA_UPDATED
from genomic_variant_classifier.agent_layer.agents.training_lifecycle_agent import TrainingLifecycleAgent
d = tempfile.mkdtemp()
state = SharedState(state_file=os.path.join(d, "state.json"))
a = TrainingLifecycleAgent(state)
assert hasattr(a, "_bus"), "agent has no _bus attribute -- test design assumption wrong"
mid = a._bus.send("DataFreshnessAgent","TrainingLifecycleAgent",DATA_UPDATED,
                  {"source":"gnomAD","ingest_approved":True,"change_type":"release"},
                  requires_approval=True)
assert mid, "bus.send did not return a message id"
a._bus.approve(mid)
import unittest.mock as m
with m.patch.object(a, "_require_approval", return_value=False):
    r = a.run(dry_run=False)
assert isinstance(r, dict) and "retrain_triggered" in r, "run() did not return expected dict"
print("PREFLIGHT_OK", r.get("retrain_triggered"), r.get("trigger_reason"))
'@
$pre | python -
if ($LASTEXITCODE -ne 0) { Fail "API preflight failed -- the regression test's construction assumptions do not hold on the real code. Investigate before proceeding." }
OK "real-code API matches the test's construction (agent._bus.send/approve, run() dict)"

Write-Host "===== STEP 3: ANTI-VACUITY -- new test MUST FAIL on pre-fix code =====" -ForegroundColor Cyan
python -m pytest $TESTDST -q 2>&1 | Tee-Object -Variable preOut | Out-Host
$preRc = $LASTEXITCODE
if ($preRc -eq 0) { Fail "regression test PASSED against pre-fix code -> it is VACUOUS. Fix the test before proceeding." }
if ($preRc -eq 5) { Fail "regression test was NOT COLLECTED (rc=5). Check placement/naming." }
if ($preRc -ne 1) { Fail "regression test ERRORED (rc=$preRc) rather than failing on assertions. Investigate." }
OK "test fails on pre-fix code (rc=1) -- non-vacuous"

Write-Host "===== STEP 4: apply edit-script =====" -ForegroundColor Cyan
python $EDIT
if ($LASTEXITCODE -ne 0) { Fail "edit-script aborted (rc=$LASTEXITCODE)" }
python -c "import py_compile,sys; py_compile.compile(r'$AGENT', doraise=True); print('py_compile OK')"
if ($LASTEXITCODE -ne 0) { Fail "agent file fails to compile after edit" }
$post = [System.IO.File]::ReadAllText((Join-Path $PWD $AGENT))
foreach ($tok in @("_check_drift","drift_detected","detect_drift","import subprocess")) {
    if ($post.Contains($tok)) { Fail "forbidden token still present after edit: $tok" }
}
OK "agent edited, compiles, no vestigial tokens remain"

Write-Host "===== STEP 5: new test MUST PASS on post-fix code =====" -ForegroundColor Cyan
python -m pytest $TESTDST -q 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { Fail "regression test failed on post-fix code (rc=$LASTEXITCODE)" }
OK "regression test passes on post-fix code"

Write-Host "===== STEP 6: targeted collected tests (registry/liveness/orchestrator) =====" -ForegroundColor Cyan
python -m pytest tests/unit/test_registry.py tests/unit/test_lazy_agent.py `
  tests/unit/test_orchestrator_lazy_registry.py tests/unit/test_orchestrator_agent_isolation.py `
  tests/unit/test_check_agents_active.py tests/unit/test_interpretability_agent.py -q 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { Fail "targeted collected tests failed" }
OK "targeted agent/registry/orchestrator tests green"

Write-Host "===== STEP 7: 22/22 agent liveness =====" -ForegroundColor Cyan
python scripts\check_agents_active.py 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { Fail "agent liveness check FAILED (an agent went dormant)" }
OK "all agents active"

Write-Host "===== STEP 8: FULL pytest suite =====" -ForegroundColor Cyan
Write-Host "  (this is the long gate; baseline was 1514 collected, expect 1517 with the 3 new tests)"
python -m pytest -q 2>&1 | Tee-Object -Variable fullOut | Out-Host
if ($LASTEXITCODE -ne 0) { Fail "FULL pytest suite not green (rc=$LASTEXITCODE)" }
OK "full suite green"

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host " ALL GATES GREEN. Unit 1 fix verified end-to-end on real code." -ForegroundColor Green
Write-Host " Edit-script $EDIT is a tool -- do NOT commit it (delete or leave untracked)." -ForegroundColor Yellow
Write-Host " Proceed to the SEPARATE commit block (stages exactly 2 files)." -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
