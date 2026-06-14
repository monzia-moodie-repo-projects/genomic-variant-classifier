#!/usr/bin/env python3
"""build_infrastructure_baseline.py  --  Monzia Moodie

Capture the infrastructure reference baseline (pinned_packages + expected_dag_hash +
golden_set) for InfrastructureDriftMonitorAgent. All MODEL-FREE: pinned_packages from the
live env (importlib.metadata), expected_dag_hash = sha256 of the supplied DAG spec, and
golden_set = the feature pipeline's output on a fixed set of variants (--golden-features
parquet, must include a variant_id column). Mirrors build_schema_baseline.py.
Output: data/reference/infrastructure/infrastructure_baseline.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.agent_layer.agents.infrastructure_drift_agent import (
    InfrastructureDriftAgent,
)

DEFAULT_OUT = Path("data/reference/infrastructure/infrastructure_baseline.json")


def default_monitored_packages() -> list:
    return sorted(InfrastructureDriftAgent.__dataclass_fields__["monitored_packages"].default)


def build_baseline(golden_set: pd.DataFrame, dag_spec: str, packages=None) -> dict:
    if "variant_id" not in golden_set.columns:
        raise ValueError("golden_set must include a 'variant_id' column.")
    pkgs = list(packages) if packages else default_monitored_packages()
    pinned = InfrastructureDriftAgent.current_package_versions(pkgs)
    dag_hash = hashlib.sha256(dag_spec.encode("utf-8")).hexdigest()
    return {
        "pinned_packages": pinned,
        "expected_dag_hash": dag_hash,
        "golden_set": golden_set.to_dict(orient="records"),
        "n_golden": int(len(golden_set)),
        "dag_spec_chars": len(dag_spec),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the infrastructure reference baseline (model-free).")
    ap.add_argument("--golden-features", type=Path, required=True,
                    help="parquet of the feature pipeline output on a fixed variant set (incl. variant_id)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dag-spec", help="DAG spec string to hash")
    grp.add_argument("--dag-spec-file", type=Path, help="file whose contents are the DAG spec")
    ap.add_argument("--packages", nargs="*", default=None,
                    help="packages to pin (default: InfrastructureDriftAgent.monitored_packages)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    golden = pd.read_parquet(args.golden_features)
    dag_spec = args.dag_spec if args.dag_spec is not None else args.dag_spec_file.read_text(encoding="utf-8")
    payload = build_baseline(golden, dag_spec, args.packages)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  pinned {len(payload['pinned_packages'])} packages; "
          f"dag_hash={payload['expected_dag_hash'][:16]}...; golden_set rows={payload['n_golden']}")


if __name__ == "__main__":
    main()
