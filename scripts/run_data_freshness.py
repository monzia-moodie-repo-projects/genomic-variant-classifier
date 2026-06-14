#!/usr/bin/env python3
"""run_data_freshness.py -- Monzia Moodie
Run the registry-driven DatabaseFreshnessMonitorAgent and print a one-line summary + the report path. Used by
.github/workflows/data_freshness.yml (scheduled) and runnable locally. Dry-run by default (no HITL prompts).
"""
from __future__ import annotations

import argparse
import sys

from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
from genomic_variant_classifier.agent_layer.shared_state import SharedState


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Registry-driven data-freshness monitor")
    ap.add_argument("--no-dry-run", action="store_true",
                    help="allow HITL re-acquisition approval prompts (default: dry-run)")
    args = ap.parse_args(argv)
    orch = Orchestrator(SharedState(), dry_run=not args.no_dry_run)
    results = orch.run_pipeline("database_monitor")
    res = (results or {}).get("DatabaseFreshnessMonitorAgent", {}) or {}
    print(f"sources={res.get('sources')} changes={res.get('changes_detected')} report={res.get('report')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
