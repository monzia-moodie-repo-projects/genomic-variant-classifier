#!/usr/bin/env python3
"""Verify the drift-monitoring stack -- WHICH LIBRARY IS WHERE, AND WHAT ACTUALLY WORKS.

Rewritten 2026-07-13 (roadmap 6.19).

WHAT THIS SCRIPT USED TO BE
---------------------------
Fourteen lines. Six bare module-level imports, then six `print(lib.__version__)` calls:

    import evidently
    import nannyml
    import alibi_detect          # <-- DROPPED from the project. Not installed. LINE 3.
    import river
    import great_expectations
    import fairlearn

It could not run. `alibi_detect` is explicitly recorded as DROPPED in requirements.in
("alibi-detect dropped -- 0.13.0 (latest, Dec 2025) still pins numpy<2.0 and is effectively
unmaintained for numpy 2.x"). The import is at module level, so the script died on line 3,
every time, and printed nothing.

**The script whose entire job was to verify the drift libraries was itself broken by a
library the project had removed.** Nobody noticed, because nobody ran it -- and nothing ran
it for them.

WHAT IT DOES NOW
----------------
Reports the TRUE state of the drift stack, across BOTH environments, and distinguishes the
three things the old script conflated:

    NOT INSTALLED  -- the package is absent from this environment (which may be CORRECT:
                      nannyml and evidently belong ONLY in the drift environment)
    BROKEN         -- installed, but cannot be imported (say WHY, in full)
    OK             -- installed and importable

It EXITS NON-ZERO if anything expected in THIS environment is missing or broken, so it can be
used as a gate rather than as a thing someone might read.

THE TWO ENVIRONMENTS
--------------------
    TRAINING  (requirements.txt, .venv312)      lightgbm 4.6.0, plotly 6.6.0, torch, pyspark
    DRIFT     (requirements-drift.txt, .venv-drift)  lightgbm 4.5.0, plotly 5.24.1, NO pyspark

They are separate because nannyml requires lightgbm<4.6 while the ensemble TRAINS on 4.6.0 --
a base model. Downgrading the model to suit the monitoring tool is refused. See roadmap 6.19.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import sys

# module name -> (which environment it belongs to, what dies without it)
EXPECTED: dict[str, tuple[str, str]] = {
    # --- the TRAINING environment (requirements.txt) ---
    "river": (
        "training",
        "online drift detection (agent_layer/agents/annotation_policy_agent.py)",
    ),
    "pandera": (
        "training",
        "schema-drift detection (agent_layer/agents/schema_drift_agent.py)",
    ),
    "scipy": (
        "training",
        "the ACTUAL feature-drift engine: PSI / Kolmogorov-Smirnov / Maximum Mean "
        "Discrepancy (monitoring/drift_detector.py)",
    ),
    # --- the ISOLATED DRIFT environment (requirements-drift.txt) ---
    "nannyml": (
        "drift",
        "Confidence-Based Performance Estimation -- estimating ROC AUC on the UNLABELLED new "
        "ClinVar release (monitoring/performance_estimator.py)",
    ),
    "evidently": (
        "drift",
        "the tabular distribution-drift HTML report (scripts/run_drift_monitor.py "
        "--evidently)",
    ),
}

# Explicitly recorded as REMOVED. If one of these turns up installed, say so -- a resurrected
# dependency is drift too.
REMOVED: dict[str, str] = {
    "alibi_detect": (
        "DROPPED (requirements.in): 0.13.0 still pins numpy<2.0 and is effectively "
        "unmaintained for numpy 2.x. Replaced by scipy.stats + river. The OLD version of this "
        "script imported it at module level and therefore could not run AT ALL."
    ),
    "great_expectations": (
        "declared in requirements.in but imported by NOTHING except the old version of this "
        "script. Never installed outside the developer's laptop."
    ),
    "fairlearn": (
        "declared in requirements.in but imported by NOTHING except the old version of this "
        "script. The fairness agent (agent_layer/agents/fairness_subgroup_agent.py) does not "
        "use it."
    ),
}


def _probe(mod: str) -> tuple[str, str]:
    """Return (status, detail). Never raises. Distinguishes ABSENT from BROKEN."""
    try:
        version = md.version(mod.replace("_", "-"))
    except md.PackageNotFoundError:
        return "NOT INSTALLED", ""
    try:
        importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001 -- we WANT to report every failure mode
        return "BROKEN", f"{type(exc).__name__}: {exc}"
    return "OK", version


def main() -> int:
    which = "DRIFT" if _probe("nannyml")[0] == "OK" else "TRAINING"
    print(f"Drift-stack verification -- this looks like the {which} environment.")
    print(f"  python: {sys.version.split()[0]}   {sys.executable}")
    print()

    problems = 0

    print("--- expected libraries ---")
    for mod, (env, purpose) in sorted(EXPECTED.items()):
        status, detail = _probe(mod)
        belongs_here = (env == "drift") == (which == "DRIFT")

        if status == "OK":
            print(f"  OK             {mod:<20} {detail}")
        elif status == "NOT INSTALLED" and not belongs_here:
            print(f"  absent (OK)    {mod:<20} belongs to the {env} environment, not this one")
        elif status == "NOT INSTALLED":
            print(f"  *** MISSING    {mod:<20} SILENTLY DISABLES: {purpose}")
            problems += 1
        else:  # BROKEN
            print(f"  *** BROKEN     {mod:<20} {detail}")
            print(f"                 {' ' * 20} SILENTLY DISABLES: {purpose}")
            problems += 1

    print()
    print("--- libraries recorded as REMOVED (a resurrected dependency is drift too) ---")
    for mod, why in sorted(REMOVED.items()):
        status, detail = _probe(mod)
        if status == "NOT INSTALLED":
            print(f"  absent (OK)    {mod:<20} {why[:60]}...")
        else:
            print(f"  *** PRESENT    {mod:<20} but it is recorded as REMOVED: {why}")
            problems += 1

    print()
    if problems:
        print(f"{problems} problem(s). Exiting 1.")
        print()
        print("If nannyml/evidently are missing and you need them, you are in the TRAINING")
        print("environment. They live in the ISOLATED DRIFT environment, because nannyml")
        print("requires lightgbm<4.6 while the ensemble TRAINS on lightgbm 4.6.0:")
        print()
        print("  python -m venv .venv-drift")
        print("  .venv-drift/Scripts/pip install -r requirements-drift.txt")
        print("  .venv-drift/Scripts/pip install -e . --no-deps")
        return 1

    print("All expected drift libraries present and importable in this environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
