"""Freeze one gene-to-partition registry shared by BOTH arms of the repair
experiment, then report membership cells and per-partition support.

A single frozen, hash-based, label-free registry applied to BOTH arms removes
the confound the ruling names ("the seed is not the split"): train/validation/
test GENE membership is identical across arms, so the only thing that differs
is which ROWS of those genes are eligible. Read-only.
"""
import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from migration import SplitRegistry, cohort_cells

PATHOGENIC_TERMS = frozenset({"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"})
BENIGN_TERMS = frozenset({"Benign", "Likely benign", "Benign/Likely benign"})

POLICY_ID = "review-repair-split-policy-1"
SALT = "freeze-before-evaluation"
WEIGHTS = (7, 1, 2)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nested_review_status(value):
    if isinstance(value, Mapping):
        return value.get("review_status")
    if isinstance(value, str):
        return json.loads(value).get("review_status")
    return None


def load_resolver(repo):
    import importlib.util, sys
    path = Path(repo) / "src/genomic_variant_classifier/data/review_status.py"
    name = "_reg_review_" + sha256_file(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def memberships(cohort_path, repo, max_review_tier):
    resolver = load_resolver(repo)
    df = pd.read_parquet(cohort_path, columns=["variant_id", "gene_symbol", "clinical_sig",
                                                "ReviewStatus", "metadata"])
    sig = df["clinical_sig"].fillna("").str.strip()
    label_eligible = sig.isin(PATHOGENIC_TERMS | BENIGN_TERMS) & ~sig.str.contains("onflict", regex=False, na=False)
    df["label"] = sig.map({**{t: 1 for t in PATHOGENIC_TERMS}, **{t: 0 for t in BENIGN_TERMS}}).astype("Int64")

    def tier_of_series(raw):
        keys = raw.map(resolver.normalise)
        cache = {}
        for k in keys.unique():
            try:
                cache[k] = resolver.resolve(k).tier
            except resolver.UnmatchedReviewStatusError:
                cache[k] = None
        return keys.map(cache).astype("Int64")

    top_tier = tier_of_series(df["ReviewStatus"])
    nested_tier = tier_of_series(df["metadata"].map(nested_review_status))

    df["legacy_member"] = (label_eligible & top_tier.le(max_review_tier).fillna(False)).astype(bool)
    df["corrected_member"] = (label_eligible & nested_tier.le(max_review_tier).fillna(False)).astype(bool)
    return df


def run(cohort_path, repo, output_dir, max_review_tier=3):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists -- refusing to overwrite")

    df = memberships(cohort_path, repo, max_review_tier)
    print(f"universe rows: {len(df):,}")
    print(f"  legacy members:    {int(df['legacy_member'].sum()):,}")
    print(f"  corrected members: {int(df['corrected_member'].sum()):,}")

    cells = cohort_cells(
        legacy_members=df.loc[df["legacy_member"], "variant_id"].tolist(),
        corrected_members=df.loc[df["corrected_member"], "variant_id"].tolist(),
        universe=df["variant_id"].tolist(),
    )
    cell_counts = {k: len(v) for k, v in cells.items()}
    print()
    print("membership cells:", cell_counts)
    total = sum(cell_counts.values())
    if total != len(df):
        raise ValueError(f"Cell accounting does not close: {total} != {len(df)}")
    print(f"  accounting closes: {total:,} == {len(df):,}")

    member_genes = sorted(set(df.loc[df["legacy_member"] | df["corrected_member"], "gene_symbol"].dropna()))
    registry = SplitRegistry(POLICY_ID, SALT, WEIGHTS, {}).extend(member_genes)
    print()
    print(f"registry: {len(registry.assignments):,} genes | sha256 {registry.sha256}")

    assign = pd.Series(registry.assignments)
    df["partition"] = df["gene_symbol"].map(assign)

    print()
    print("=== support by partition and arm ===")
    support = {}
    for part in ("train", "validation", "test"):
        sel = df["partition"].eq(part)
        row = {}
        for arm, col in (("legacy", "legacy_member"), ("corrected", "corrected_member")):
            m = sel & df[col]
            n = int(m.sum())
            pos = int((df.loc[m, "label"] == 1).sum())
            row[arm] = {"rows": n, "genes": int(df.loc[m, "gene_symbol"].nunique()),
                        "positives": pos, "negatives": n - pos,
                        "prevalence": round(pos / n, 6) if n else None}
        support[part] = row
        print(f"  {part}:")
        for arm in ("legacy", "corrected"):
            r = row[arm]
            print(f"    {arm}: rows={r['rows']:,} genes={r['genes']:,} "
                  f"pos={r['positives']:,} prevalence={r['prevalence']}")

    test_sel = df["partition"].eq("test")
    e_common = int((test_sel & df["legacy_member"] & df["corrected_member"]).sum())
    e_added = int((test_sel & ~df["legacy_member"] & df["corrected_member"]).sum())
    print()
    print(f"evaluation cells on test partition: common={e_common:,} added={e_added:,}")

    output_dir.mkdir(parents=True, exist_ok=False)
    assignment_frame = pd.DataFrame({"gene_symbol": list(registry.assignments.keys()),
                                      "partition": list(registry.assignments.values())})
    assignment_frame.to_parquet(output_dir / "gene_partition_registry.parquet", index=False)
    df[["variant_id", "gene_symbol", "label", "legacy_member", "corrected_member", "partition"]].to_parquet(
        output_dir / "membership_and_partition.parquet", index=False)

    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "policy_id": POLICY_ID, "salt": SALT, "weights": list(WEIGHTS),
        "weights_note": "train/validation/test; declared in the governing ruling. Differs from "
                        "the earlier ad-hoc 70/15/15 GroupShuffleSplit, which is superseded "
                        "because a seeded split re-run on changed membership would confound "
                        "the repair with population reassignment.",
        "registry_sha256": registry.sha256,
        "source_cohort_path": str(Path(cohort_path).resolve()),
        "source_cohort_sha256": sha256_file(cohort_path),
        "max_review_tier": max_review_tier,
        "n_universe": len(df),
        "n_legacy_members": int(df["legacy_member"].sum()),
        "n_corrected_members": int(df["corrected_member"].sum()),
        "membership_cells": cell_counts,
        "n_genes_in_registry": len(registry.assignments),
        "support": support,
        "evaluation_cells_test_partition": {"common": e_common, "added": e_added},
    }
    (output_dir / "split_registry_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Wrote {output_dir}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-review-tier", type=int, default=3)
    args = p.parse_args()
    run(args.cohort, args.repo, args.output_dir, args.max_review_tier)


if __name__ == "__main__":
    main()
