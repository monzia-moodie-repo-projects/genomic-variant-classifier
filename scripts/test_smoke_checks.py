#!/usr/bin/env python3
"""
test_smoke_checks.py — offline unit test for smoke_all_models.check_outputs.

Pure pandas; no torch / no data / no run_phase2_eval. Validates the gate logic
that decides GREEN vs BLOCKED from a per_model_metrics.csv + a captured log:
  * full roster present + finite AUROC + stacker present -> pass
  * a missing model (errored -> dropped from ensemble -> absent) -> FAIL
  * a NaN AUROC (ran but degenerate) -> FAIL
  * a 'Traceback' / 'OOF failed' / 'skipping' line in the log -> FAIL

Run:  python test_smoke_checks.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_all_models import check_outputs  # noqa: E402

EXPECTED = {
    "random_forest", "xgboost", "lightgbm", "logistic_regression",
    "gradient_boosting", "tabular_nn", "cnn_1d", "svm",
    "catboost", "kan", "mc_dropout", "deep_ensemble",
}


def _write_pmm(tmp: Path, rows: dict[str, float]) -> Path:
    df = pd.DataFrame({"auroc": rows})
    p = tmp / "per_model_metrics.csv"
    df.to_csv(p)
    return p


def main() -> int:
    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="smoke_checks_"))

    full = {m: 0.9 for m in EXPECTED}
    full["ENSEMBLE_STACKER"] = 0.92

    # case 1: all good
    _write_pmm(tmp, full)
    ok, msgs = check_outputs(tmp, "training complete\nall folds done\n", EXPECTED)
    if not ok:
        failures.append(f"PASS case wrongly BLOCKED: {msgs}")

    # case 2: KAN missing (errored -> dropped)
    missing = dict(full)
    del missing["kan"]
    _write_pmm(tmp, missing)
    ok, msgs = check_outputs(tmp, "clean log\n", EXPECTED)
    if ok or not any("kan" in m for m in msgs):
        failures.append(f"MISSING-model case not caught: ok={ok} msgs={msgs}")

    # case 3: NaN AUROC for svm
    nanv = dict(full)
    nanv["svm"] = float("nan")
    _write_pmm(tmp, nanv)
    ok, msgs = check_outputs(tmp, "clean log\n", EXPECTED)
    if ok or not any("svm" in m and "NaN" in m for m in msgs):
        failures.append(f"NaN-AUROC case not caught: ok={ok} msgs={msgs}")

    # case 4: Traceback in log
    _write_pmm(tmp, full)
    ok, msgs = check_outputs(tmp, "Traceback (most recent call last):\n", EXPECTED)
    if ok or not any("Traceback" in m for m in msgs):
        failures.append(f"Traceback case not caught: ok={ok} msgs={msgs}")

    # case 5: 'OOF failed' / 'skipping' in log
    _write_pmm(tmp, full)
    ok, msgs = check_outputs(tmp, "kan OOF failed: name 'test_size' is not defined -- skipping\n", EXPECTED)
    if ok:
        failures.append(f"OOF-failed/skipping case not caught: msgs={msgs}")

    # case 6: stacker missing
    nostack = {m: 0.9 for m in EXPECTED}
    _write_pmm(tmp, nostack)
    ok, msgs = check_outputs(tmp, "clean\n", EXPECTED)
    if ok or not any("ENSEMBLE_STACKER" in m for m in msgs):
        failures.append(f"missing-stacker case not caught: ok={ok} msgs={msgs}")

    print("test_smoke_checks:")
    if failures:
        for f in failures:
            print("  [FAIL] " + f)
        return 1
    print("  [ok] all 6 gate-logic cases behave correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
