# Install_lazy_registration.ps1 -- apply lazy agent registration to orchestrator.py, then PROVE
# Monzia's condition (no diminished agents) with hard gates. Run from repo root. pandas 2.3.3.
$ErrorActionPreference = "Stop"
Set-Location C:\Projects\genomic-variant-classifier
$dl = "$env:USERPROFILE\Downloads"

# -- 0. place patcher (content-marker verified) --
$pat = Get-ChildItem $dl -Filter "patch_orchestrator_lazy_registration.py" |
       Where-Object { Select-String -Path $_.FullName -Pattern 'LAZY agent registration' -Quiet } |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $pat) { throw "ABORT: patch_orchestrator_lazy_registration.py not in $dl (marker missing)." }
Copy-Item $pat.FullName "scripts\patch_orchestrator_lazy_registration.py" -Force
Unblock-File "scripts\patch_orchestrator_lazy_registration.py"

# -- 1. PRE-CHECK: capture agent count BEFORE (must equal AFTER) --
"=== PRE: check_agents_active.py (baseline 22/22 expected) ==="
python scripts\check_agents_active.py
$preExit = $LASTEXITCODE
"check_agents_active PRE exit: $preExit (0 = all active)"
if ($preExit -ne 0) { throw "ABORT: agents not all-active BEFORE patch (exit $preExit). Fix that first." }

# -- 2. apply the patcher --
"=== applying lazy-registration patcher ==="
python scripts\patch_orchestrator_lazy_registration.py
if ($LASTEXITCODE -ne 0) { throw "ABORT: patcher failed (exit $LASTEXITCODE)." }

# -- 3. POST-CHECK A: orchestrator imports + constructs, registry still 22 keys --
"=== POST-A: orchestrator constructs + 22 keys (run_pipeline path intact) ==="
$py = @'
import sys
sys.path.insert(0, "src")
from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator, PIPELINE_DEFINITIONS
from genomic_variant_classifier.agent_layer.shared_state import SharedState
o = Orchestrator(SharedState(), dry_run=True)
keys = list(o._agent_registry.keys())
print("registry key count:", len(keys))
assert len(keys) == 22, f"EXPECTED 22 keys, got {len(keys)}"
# every value is now a zero-arg factory; resolving one must yield a class
f = o._agent_registry["DatabaseFreshnessMonitorAgent"]
cls = f()
print("factory resolves to class:", cls.__name__)
assert cls.__name__ == "DatabaseFreshnessMonitorAgent"
print("POST-A PASS: 22 keys, factory resolves correctly")
'@
$py | python -
if ($LASTEXITCODE -ne 0) { throw "ABORT: POST-A failed." }

# -- 4. POST-CHECK B: check_agents_active still 22/22, 0 dormant (Monzia's literal condition) --
"=== POST-B: check_agents_active.py (MUST still be 22/22, 0 dormant) ==="
python scripts\check_agents_active.py
$postExit = $LASTEXITCODE
"check_agents_active POST exit: $postExit"
if ($postExit -ne 0) { throw "ABORT: agents not all-active AFTER patch (exit $postExit). Reverting." }

# -- 5. POST-CHECK C: the actual CI command runs (database_monitor dry-run) --
"=== POST-C: the exact CI command (PYTHONPATH=src python scripts/run_data_freshness.py) ==="
$env:PYTHONPATH = "src"
python scripts\run_data_freshness.py
if ($LASTEXITCODE -ne 0) { throw "ABORT: run_data_freshness.py failed (exit $LASTEXITCODE)." }
Remove-Item Env:\PYTHONPATH

# -- 6. POST-CHECK D: simulate the CI minimal env -- orchestrator construct must NOT import sklearn --
"=== POST-D: construct orchestrator with sklearn BLOCKED (simulates minimal CI) ==="
$py2 = @'
import sys, builtins
sys.path.insert(0, "src")
_real_import = builtins.__import__
def _block(name, *a, **k):
    if name == "sklearn" or name.startswith("sklearn."):
        raise ModuleNotFoundError("No module named 'sklearn' (simulated minimal CI)")
    return _real_import(name, *a, **k)
builtins.__import__ = _block
from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
from genomic_variant_classifier.agent_layer.shared_state import SharedState
o = Orchestrator(SharedState(), dry_run=True)   # must NOT import sklearn
print("constructed with sklearn blocked; keys:", len(o._agent_registry))
r = o.run_pipeline("database_monitor")           # must NOT need sklearn
print("database_monitor ran under sklearn-block:", bool(r))
builtins.__import__ = _real_import
print("POST-D PASS: orchestrator + freshness pipeline work without sklearn")
'@
$py2 | python -
if ($LASTEXITCODE -ne 0) { throw "ABORT: POST-D failed -- orchestrator still needs sklearn at construct/freshness time." }

"========================================================"
"GREEN: lazy registration applied. 22/22 agents active, freshness pipeline runs without sklearn."
"Next: full pytest (separate), then commit + push, then watch CI."
