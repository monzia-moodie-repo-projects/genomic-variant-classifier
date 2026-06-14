"""scripts/run_adaptation.py  --  Monzia Moodie

Run the adaptation pipeline: VersionMonitorAgent (refresh the version_monitor
section) -> AdaptationAgent (consume it; plan or evaluate candidates).

Plan-only by default (safe, fast): records candidates to the append-only ledger
and adds a review-item alert. Set ADAPTATION_EVALUATE=1 to additionally build a
throwaway venv per candidate, install the project + candidate version, run the
test suite in isolation, and record the verdict. --dry-run plans without writing
the ledger or alerting (internal state section still updated).

Env knobs (all optional): ADAPTATION_EVALUATE, ADAPTATION_LEDGER,
ADAPTATION_MAX_CANDIDATES, ADAPTATION_PROJECT_ROOT, ADAPTATION_KEEP_VENV.
"""
from __future__ import annotations

import argparse
import json

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the adaptation pipeline.")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan without writing the ledger or alerting")
    args = ap.parse_args()

    orch = Orchestrator(SharedState(), dry_run=args.dry_run)
    results = orch.run_pipeline("adaptation")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
