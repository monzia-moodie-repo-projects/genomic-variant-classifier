"""Build a NEW, immutable corrected cohort version with row-level inclusion
reasons. Never overwrites an existing cohort or output directory.

Canonical source decision (authenticated, not assumed):
  metadata['review_status'] is ClinVar variant_summary's own ReviewStatus,
  stored at ingestion. The top-level ReviewStatus column was independently
  reconstructed via a VCF chrom:pos:ref:alt join whose unmatched keys became
  "". That join was MEASURED to fail for 97.33% of deletions on a confirmed
  coordinate-convention mismatch (99.92% of sampled failures recover at
  pos-1), while SNV failures reflect genuine absence from the VCF. Among
  label-eligible rows there is ZERO genuine missing nested evidence. The
  nested value is therefore the authenticated source.
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

PATHOGENIC_TERMS = frozenset({"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"})
BENIGN_TERMS = frozenset({"Benign", "Likely benign", "Benign/Likely benign"})


class ReviewEvidenceError(ValueError):
    """Raised when review evidence cannot be resolved for an otherwise eligible row."""


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_resolver(repo):
    path = Path(repo) / "src/genomic_variant_classifier/data/review_status.py"
    name = "_canonical_review_" + sha256_file(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, sha256_file(path)


def nested_review_status(value):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get("review_status")
    if isinstance(value, str):
        return json.loads(value).get("review_status")
    if pd.isna(value):
        return None
    raise ValueError(f"metadata must be an object, JSON string, or missing; got {type(value).__name__}")


def allele_state(ref, alt):
    if not isinstance(ref, str) or not isinstance(alt, str):
        return "non_string_allele"
    if not ref or not alt:
        return "empty_allele"
    if set(ref + alt) - set("ACGT"):
        return "non_acgt_allele"
    if ref == alt:
        return "ref_equals_alt"
    return "resolved"


def build(cohort_path, repo, output_dir, *, max_review_tier=3, exclude_conflicting=True,
          apply_allele_filter=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists -- refusing to overwrite a prior cohort version")

    resolver, resolver_sha = load_resolver(repo)
    cohort_sha = sha256_file(cohort_path)

    label_policy_sha = sha256_text(json.dumps({
        "pathogenic_terms": sorted(PATHOGENIC_TERMS),
        "benign_terms": sorted(BENIGN_TERMS),
        "exclude_conflicting": exclude_conflicting,
        "match": "exact_case_sensitive_after_strip",
    }, sort_keys=True))
    review_policy_sha = sha256_text(json.dumps({
        "canonical_source": "metadata.review_status",
        "resolver_sha256": resolver_sha,
        "max_review_tier": max_review_tier,
        "predicate": "tier <= max_review_tier",
    }, sort_keys=True))

    df = pd.read_parquet(cohort_path)
    n_input = len(df)

    ids = df["variant_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("variant_id must be present and unique across the input universe")

    sig = df["clinical_sig"].fillna("").str.strip()
    is_binary = sig.isin(PATHOGENIC_TERMS | BENIGN_TERMS)
    is_conflicting = sig.str.contains("onflict", regex=False, na=False)
    label_eligible = is_binary & (~is_conflicting if exclude_conflicting else True)
    labels = sig.map({**{t: 1 for t in PATHOGENIC_TERMS}, **{t: 0 for t in BENIGN_TERMS}}).astype("Int64")

    allele = pd.Series([allele_state(r, a) for r, a in zip(df["ref"], df["alt"])], index=df.index)
    allele_eligible = allele.eq("resolved")

    raw_review = df["metadata"].map(nested_review_status)
    keys = raw_review.map(resolver.normalise)

    tiers, paths = {}, {}
    for key in keys.unique():
        try:
            result = resolver.resolve(key)
            tiers[key], paths[key] = result.tier, result.path.value
        except resolver.UnmatchedReviewStatusError:
            tiers[key], paths[key] = None, "UNKNOWN_VOCABULARY"

    review_tier = keys.map(tiers).astype("Int64")
    review_path = keys.map(paths)

    unknown_and_eligible = review_tier.isna() & label_eligible & allele_eligible
    if unknown_and_eligible.any():
        offenders = keys[unknown_and_eligible].value_counts().to_dict()
        raise ReviewEvidenceError(
            f"Unknown review vocabulary among otherwise-eligible rows: {offenders}. "
            f"Add each to REVIEW_STATUS_TIER before rebuilding; no fallback tier is fabricated."
        )

    review_eligible = review_tier.le(max_review_tier).fillna(False)
    # Ruling section 8: the first migration experiment changes ONE thing -- the
    # review-status source. Allele resolvability is RECORDED for every row but
    # does not gate inclusion unless explicitly requested.
    if apply_allele_filter:
        included = (label_eligible & allele_eligible & review_eligible).astype(bool)
    else:
        included = (label_eligible & review_eligible).astype(bool)

    reasons = []
    for lab, alle, rev, allele_name, path_name in zip(
        label_eligible, allele_eligible, review_eligible, allele, review_path
    ):
        r = []
        if not lab:
            r.append("label_not_binary_or_conflicting")
        if apply_allele_filter and not alle:
            r.append(f"allele_{allele_name}")
        if not rev:
            r.append("review_tier_above_threshold" if path_name != "UNKNOWN_VOCABULARY"
                     else "review_vocabulary_unknown")
        reasons.append("|".join(r))

    decisions = pd.DataFrame({
        "variant_identity": ids,
        "label": labels,
        "label_eligible": label_eligible,
        "review_eligible": review_eligible,
        "allele_eligible": allele_eligible,
        "identity_resolved": True,
        "included": included,
        "exclusion_reasons": reasons,
        "review_status_resolved": keys,
        "review_tier": review_tier,
        "review_path": review_path,
        "allele_state": allele,
    })

    corrected = df[included].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    decisions_path = output_dir / "decision_table.parquet"
    cohort_out = output_dir / "cohort_corrected.parquet"
    decisions.to_parquet(decisions_path, index=False)
    corrected.to_parquet(cohort_out, index=False)

    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_version": "corrected-review-v1",
        "canonical_review_source": "metadata.review_status",
        "superseded_representation": "top-level ReviewStatus (VCF-join derived)",
        "source_cohort_path": str(Path(cohort_path).resolve()),
        "source_snapshot_sha256": cohort_sha,
        "label_policy_sha256": label_policy_sha,
        "review_policy_sha256": review_policy_sha,
        "resolver_sha256": resolver_sha,
        "max_review_tier": max_review_tier,
        "exclude_conflicting": exclude_conflicting,
        "apply_allele_filter": apply_allele_filter,
        "n_input_universe": n_input,
        "n_included": int(included.sum()),
        "n_excluded": int((~included).sum()),
        "exclusion_reason_counts": {str(k): int(v) for k, v in
                                     decisions.loc[~included, "exclusion_reasons"].value_counts().items()},
        "label_distribution": {str(k): int(v) for k, v in
                               decisions.loc[included, "label"].value_counts().items()},
        "n_included_with_unresolved_allele": int((included & ~allele_eligible).sum()),
        "included_allele_state_counts": {str(k): int(v) for k, v in
                                          allele[included].value_counts().items()},
        "decision_table_sha256": sha256_file(decisions_path),
        "corrected_cohort_sha256": sha256_file(cohort_out),
    }
    (output_dir / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    if manifest["n_included"] + manifest["n_excluded"] != n_input:
        raise ValueError("Decision accounting does not close")

    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-review-tier", type=int, default=3)
    p.add_argument("--apply-allele-filter", action="store_true",
                   help="Also gate inclusion on allele resolvability. OFF by default.")
    args = p.parse_args()
    manifest = build(args.cohort, args.repo, args.output_dir,
                     max_review_tier=args.max_review_tier,
                     apply_allele_filter=args.apply_allele_filter)
    for k in ("n_input_universe", "n_included", "n_excluded", "label_distribution",
              "apply_allele_filter", "n_included_with_unresolved_allele",
              "included_allele_state_counts"):
        print(f"{k}: {manifest[k]}")
    print(f"corrected_cohort_sha256: {manifest['corrected_cohort_sha256']}")


if __name__ == "__main__":
    main()

