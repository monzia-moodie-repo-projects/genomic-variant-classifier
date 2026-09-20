"""Replacement for augment_reviewstatus.py's VCF-join derivation.

The defect it fixes (INCIDENT 2026-09-20): the top-level ReviewStatus column was
reconstructed by joining ClinVar's VCF on chrom:pos:ref:alt, and unmatched keys
were written as "". Because variant_summary and the VCF use different position
conventions for indels, the join failed for 97.33% of deletions, and the empty
string resolved to the missing-evidence tier. A RETRIEVAL failure was recorded
as an ABSENCE OF EVIDENCE, removing 130,224 label-eligible variants.

This module derives the top-level field from the canonical record already stored
at ingestion (metadata.review_status) instead of re-joining a second source. It
never fabricates a value: an unresolvable status raises.

Non-destructive: writes a NEW cohort file and refuses to overwrite.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

POLICY_ID = "canonical-review-derivation-1"


class ReconciliationFailure(ValueError):
    """A required review status could not be retrieved. Never silently filled."""


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_resolver(repo):
    path = Path(repo) / "src/genomic_variant_classifier/data/review_status.py"
    name = "_derive_review_" + sha256_file(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, sha256_file(path)


def canonical_review_status(metadata):
    """Extract review_status from the canonical ingestion record.

    Distinguishes, and never collapses:
      - metadata absent entirely          -> ReconciliationFailure
      - metadata present, key absent      -> ReconciliationFailure
      - metadata present, value present   -> the value (including missing tokens,
                                             which the resolver classifies)
    """
    if metadata is None or (not isinstance(metadata, (Mapping, str)) and pd.isna(metadata)):
        raise ReconciliationFailure("metadata absent; canonical review record unavailable")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, Mapping):
        raise ReconciliationFailure(
            f"metadata must be an object or JSON string; got {type(metadata).__name__}")
    if "review_status" not in metadata:
        raise ReconciliationFailure("canonical record has no review_status key")
    return metadata["review_status"]


def join_derived_status(key, vcf_map):
    """Any join-based derivation MUST refuse on an unmatched key.

    Retained deliberately as a guarded reference: the original defect was
    precisely that this path returned "" instead of raising. Kept so the
    acceptance test can assert the refusal directly.
    """
    if key not in vcf_map:
        raise ReconciliationFailure(
            f"no source record matched key {key!r}; a failed join is a retrieval "
            f"failure, not an absence of review evidence")
    return vcf_map[key]


def derive(cohort_path, repo, output_path, *, overwrite_existing_column=False):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists -- refusing to overwrite")

    resolver, resolver_sha = load_resolver(repo)
    policy_digest = hashlib.sha256(json.dumps(
        {"policy_id": POLICY_ID, "source": "metadata.review_status",
         "resolver_sha256": resolver_sha}, sort_keys=True).encode()).hexdigest()

    df = pd.read_parquet(cohort_path)
    n = len(df)

    failures = {}
    values = []
    for idx, meta in enumerate(df["metadata"]):
        try:
            values.append(canonical_review_status(meta))
        except ReconciliationFailure as e:
            failures[str(e)] = failures.get(str(e), 0) + 1
            values.append(None)
    if failures:
        raise ReconciliationFailure(
            f"Canonical review record unavailable for some rows: {failures}. "
            f"Repair ingestion; do not fill.")

    derived = pd.Series(values, index=df.index)

    keys = derived.map(resolver.normalise)
    unknown = {}
    for k in keys.unique():
        try:
            resolver.resolve(k)
        except resolver.UnmatchedReviewStatusError:
            unknown[k] = int((keys == k).sum())
    if unknown:
        raise resolver.UnmatchedReviewStatusError(
            f"Unrecognised review vocabulary, with row counts: {unknown}. "
            f"Add each to REVIEW_STATUS_TIER before rerunning; no fallback tier "
            f"is fabricated.")

    report = {
        "derived_utc": datetime.now(timezone.utc).isoformat(),
        "policy_id": POLICY_ID,
        "policy_digest": policy_digest,
        "canonical_source": "metadata.review_status",
        "retired_source": "VCF chrom:pos:ref:alt join (augment_reviewstatus.py)",
        "resolver_sha256": resolver_sha,
        "source_cohort_path": str(Path(cohort_path).resolve()),
        "source_cohort_sha256": sha256_file(cohort_path),
        "n_rows": n,
        "status_distribution": {str(k): int(v) for k, v in keys.value_counts().items()},
    }

    if "ReviewStatus" in df.columns:
        existing = df["ReviewStatus"].map(resolver.normalise)
        agree = int((existing == keys).sum())
        disagree = n - agree
        report["existing_column_present"] = True
        report["existing_agrees_rows"] = agree
        report["existing_disagrees_rows"] = disagree
        if disagree and not overwrite_existing_column:
            raise ReconciliationFailure(
                f"An existing ReviewStatus column disagrees with the canonical record on "
                f"{disagree:,} of {n:,} rows. It is stale or incorrectly derived. Re-run "
                f"with overwrite_existing_column=True to replace it deliberately.")
    else:
        report["existing_column_present"] = False

    out = df.copy()
    out["ReviewStatus"] = derived.values
    out.to_parquet(output_path, index=False)
    report["output_path"] = str(output_path.resolve())
    report["output_sha256"] = sha256_file(output_path)
    Path(str(output_path) + ".derivation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--overwrite-existing-column", action="store_true")
    args = p.parse_args()
    r = derive(args.cohort, args.repo, args.output,
               overwrite_existing_column=args.overwrite_existing_column)
    for k in ("n_rows", "existing_column_present", "existing_agrees_rows",
              "existing_disagrees_rows", "output_sha256"):
        if k in r:
            print(f"{k}: {r[k]}")


if __name__ == "__main__":
    main()
