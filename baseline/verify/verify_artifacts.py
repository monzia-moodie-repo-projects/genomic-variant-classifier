"""Verify every artifact the incident record claims, and report any it omits.

An incident document that points at artifacts is only as good as those paths.
Checks existence, size and digest for each, and refuses to report success if
any is missing. Read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

CLAIMED = [
    "outputs/review_status_audit_001/summary.json",
    "outputs/review_status_audit_001/review_source_disagreements.csv",
    "outputs/review_status_audit_001/join_failure_diagnosis.json",
    "outputs/review_status_audit_001/failure_mode_crosstab.json",
    "outputs/cohort_corrected_review_v1/cohort_manifest.json",
    "outputs/cohort_corrected_review_v2/cohort_manifest.json",
    "outputs/cohort_corrected_review_v2/cohort_corrected.parquet",
    "outputs/cohort_corrected_review_v2/decision_table.parquet",
    "outputs/split_registry_v1/split_registry_manifest.json",
    "outputs/split_registry_v1/gene_partition_registry.parquet",
    "outputs/split_registry_v1/membership_and_partition.parquet",
    "outputs/split_registry_canonical_v1/split_registry_manifest.json",
    "outputs/repair_experiment_v1/repair_experiment_report.json",
    "outputs/repair_experiment_v1/evaluation_predictions.parquet",
    "outputs/repair_experiment_v1/uncertainty.json",
    "outputs/repair_experiment_v1/stability_and_coverage.json",
    "outputs/repair_experiment_v1/interaction_ci.json",
    "outputs/exposure_ledger_v1/exposure_ledger.parquet",
    "outputs/exposure_ledger_v1/exposure_ledger_summary.json",
    "data/processed/clinvar_grch38_canonical_review.parquet",
    "data/processed/clinvar_grch38_canonical_review.parquet.derivation.json",
]


def digest(path, limit=64 * 1024 * 1024):
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
            read += len(block)
            if read >= limit:
                return h.hexdigest()[:16] + " (first 64MB)"
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    root = Path(args.root)

    present, missing = [], []
    for rel in CLAIMED:
        path = root / rel
        if path.exists():
            present.append({"path": rel, "bytes": path.stat().st_size,
                            "sha256_prefix": digest(path)})
        else:
            missing.append(rel)

    print(f"=== artifact verification against {root} ===")
    for r in present:
        print(f"  OK      {r['path']}  ({r['bytes']:,} bytes, {r['sha256_prefix']})")
    for rel in missing:
        print(f"  MISSING {rel}")
    print()
    print(f"present: {len(present)}/{len(CLAIMED)} | missing: {len(missing)}")

    report = {"root": str(root), "n_claimed": len(CLAIMED),
              "n_present": len(present), "n_missing": len(missing),
              "present": present, "missing": missing,
              "all_present": not missing}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    if missing:
        raise SystemExit(f"{len(missing)} claimed artifact(s) missing -- the record is "
                         f"inaccurate and must be corrected before it is relied on")


if __name__ == "__main__":
    main()

