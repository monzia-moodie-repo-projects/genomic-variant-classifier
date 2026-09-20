"""Build the exposure ledger required before ANY population can be called
independent confirmation.

Records, per the declared schema: canonical variant identity, validated group
identity, use, execution identity, population identity, time.

Exposure is recorded ONLY from verifiable artifacts. Where an exposure is known
to have occurred but its artifacts no longer exist, it is recorded as an
explicit gap rather than omitted -- an incomplete ledger cannot prove
independence, and pretending completeness is the failure mode this guards against.

Outcome-free schema inspection is NOT exposure. Reading aggregate label counts
and then choosing a cohort or feature IS exposure, and is recorded as such.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from migration import EXPOSURES, confirmation_screen


def build(membership_path, predictions_path, output_dir, execution_id):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists -- refusing to overwrite")

    mem = pd.read_parquet(membership_path)
    preds = pd.read_parquet(predictions_path)
    now = datetime.now(timezone.utc).isoformat()

    entries = []

    train = mem[mem["partition"].eq("train") &
                (mem["legacy_member"] | mem["corrected_member"])]
    for vid, gene in zip(train["variant_id"], train["gene_symbol"]):
        entries.append({"variant_id": vid, "group_id": gene, "use": "training",
                        "execution_id": execution_id,
                        "population_id": "repair_experiment_v1/train", "time": now})

    for vid, gene in zip(preds["variant_id"], preds["gene_symbol"]):
        entries.append({"variant_id": vid, "group_id": gene, "use": "test_feedback",
                        "execution_id": execution_id,
                        "population_id": "repair_experiment_v1/test", "time": now})

    for e in entries:
        if e["use"] not in EXPOSURES:
            raise ValueError(f"Undeclared exposure type: {e['use']}")

    ledger = pd.DataFrame(entries)
    exposed_variants = set(ledger["variant_id"])
    exposed_groups = set(ledger["group_id"])
    universe_variants = set(mem["variant_id"])
    universe_groups = set(mem["gene_symbol"].dropna())

    summary = {
        "built_utc": now,
        "execution_id": execution_id,
        "schema": ["variant_id", "group_id", "use", "execution_id", "population_id", "time"],
        "n_entries": len(ledger),
        "by_use": {str(k): int(v) for k, v in ledger["use"].value_counts().items()},
        "n_exposed_variants": len(exposed_variants),
        "n_exposed_groups": len(exposed_groups),
        "n_universe_variants": len(universe_variants),
        "n_universe_groups": len(universe_groups),
        "fraction_groups_exposed": len(exposed_groups) / len(universe_groups),
        "known_gaps": [
            "baseline_run1 (earlier this session): trained and evaluated on the legacy "
            "cohort under a 70/15/15 GroupShuffleSplit. Output artifacts were deleted "
            "before this ledger existed. The split is deterministic and reconstructible "
            "from the legacy cohort with split_cohort.py (random_state=42), but has NOT "
            "been reconstructed here. Genes exposed there are not in this ledger.",
            "Cohort-wide audits read clinical_sig and review status across all 4,399,089 "
            "rows, and cohort/feature decisions were made from those aggregate label "
            "distributions. That is aggregate rather than per-row exposure and is not "
            "enumerated here, but it means no subset of THIS snapshot is fully naive.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    ledger.to_parquet(output_dir / "exposure_ledger.parquet", index=False)
    (output_dir / "exposure_ledger_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"ledger entries: {len(ledger):,}")
    for k, v in summary["by_use"].items():
        print(f"  {k}: {v:,}")
    print(f"exposed variants: {len(exposed_variants):,} / {len(universe_variants):,}")
    print(f"exposed groups:   {len(exposed_groups):,} / {len(universe_groups):,} "
          f"({100*summary['fraction_groups_exposed']:.2f}%)")

    print()
    print("=== confirmation screen: candidates drawn from THIS snapshot ===")
    unexposed_groups = sorted(universe_groups - exposed_groups)
    print(f"groups with no recorded exposure: {len(unexposed_groups):,}")
    if unexposed_groups:
        sample_gene = unexposed_groups[0]
        cand_rows = mem[mem["gene_symbol"].eq(sample_gene)].head(3)
        candidates = [{"variant_id": v, "group_id": g}
                      for v, g in zip(cand_rows["variant_id"], cand_rows["gene_symbol"])]
        ledger_records = ledger[["variant_id", "group_id", "use"]].to_dict("records")
        screened = confirmation_screen(candidates, ledger_records)
        for s in screened:
            print(f"  {s['variant_id']} (gene {s['group_id']}): "
                  f"passes={s['passes_recorded_exposure_screen']} blockers={s['blockers']}")

    print()
    print("Read this result with the known gaps above. A screen result of passes")
    print("means no overlap in THIS ledger -- not proof of independence.")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--membership", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--execution-id", default="repair_experiment_v1")
    args = p.parse_args()
    build(args.membership, args.predictions, args.output_dir, args.execution_id)


if __name__ == "__main__":
    main()
