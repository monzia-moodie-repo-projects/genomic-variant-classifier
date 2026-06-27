#!/usr/bin/env python3
r"""patch_data_freshness_scipy.py

Add scipy to the Data Freshness Monitor workflow's install step.

The scheduled 'Data Freshness Monitor' (.github/workflows/data_freshness.yml) installs only
numpy/pandas/pyarrow + `pip install -e .`, but the agent layer it imports needs scipy
(declared in requirements.txt as scipy==1.17.1; imported at
src/.../agent_layer/agents/label_shift_agent.py:11 `from scipy import stats`). So the
dry-run import chain orchestrator -> LabelShiftMonitorAgent -> LabelShiftAgent -> scipy
raises ModuleNotFoundError and the workflow fails on every schedule.

This mirrors drift_monitor.yml:44 which already does `pip install -r requirements-api.txt scipy`.
Minimal fix: append scipy to the existing numpy/pandas/pyarrow line.

ANCHOR-BASED, IDEMPOTENT, LF.
"""
from __future__ import annotations
import argparse
from pathlib import Path

TARGET = Path(".github/workflows/data_freshness.yml")
OLD = "          pip install numpy pandas pyarrow\n"
NEW = "          pip install numpy pandas pyarrow scipy\n"
MARKER = "pip install numpy pandas pyarrow scipy"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): scipy already in data_freshness install step."); return 0
    c = src.count(OLD)
    if c != 1:
        print(f"FAIL: anchor 'pip install numpy pandas pyarrow' occurs {c}x (need 1)."); return 3
    if ns.check:
        print("CHECK: anchor found once."); print("RESULT: PASS (check)"); return 0

    backup = TARGET.with_suffix(".yml.pre_scipy.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
    new = src.replace(OLD, NEW, 1)
    TARGET.write_text(new, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    checks = {
        "scipy added to install line": MARKER in after,
        "no double-add": after.count("scipy") == 1,
        "rest of file intact (run_data_freshness present)": "run_data_freshness.py" in after,
    }
    for k, v in checks.items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
