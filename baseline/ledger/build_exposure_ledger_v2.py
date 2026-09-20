"""Exposure ledger covering MULTIPLE executions.

v1 recorded one execution (repair_experiment_v1) and concluded that the
validation partition was the only usable unexposed population in this snapshot.
That conclusion expired the moment validation was used for development: the
constraint re-measurement and the representation-arms experiment both fitted
models and read results on it. A ledger that still calls validation unexposed
would be wrong, and wrong in the direction that matters -- it would license an
independence claim that is not true.

Executions are declared in a JSON config so adding one is a data change, not a
code change. Each execution contributes:
  - training exposure for the train-partition rows it fitted on
  - a use-typed exposure for the population whose predictions it produced

Read-only with respect to every existing artifact.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from migration import EXPOSURES, confirmation_screen


def build(membership_path, config_path, output_dir, root):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists -- refusing to overwrite")

    root = Path(root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    executions = config["executions"]
    if not executions:
        raise ValueError("config declares no executions")

    mem_all = pd.read_parquet(membership_path)
    mem = mem_all[mem_all["legacy_member"] | mem_all["corrected_member"]]
    train = mem[mem["partition"].eq("train")]
    now = datetime.now(timezone.utc).isoformat()

    entries = []
    per_execution = {}
    for ex in executions:
        eid = ex["execution_id"]
        if eid in per_execution:
            raise ValueError(f"duplicate execution_id: {eid}")
        n_before = len(entries)

        if ex.get("training_from_membership"):
            for vid, gene in zip(train["variant_id"], train["gene_symbol"]):
                entries.append({"variant_id": vid, "group_id": gene, "use": "training",
                                "execution_id": eid,
                                "population_id": f"{eid}/train", "time": now})

        pred_path = root / ex["predictions"]
        if not pred_path.is_file():
            raise FileNotFoundError(f"{eid}: predictions not found at {pred_path}")
        use = ex["predictions_use"]
        if use not in EXPOSURES:
            raise ValueError(f"{eid}: undeclared exposure type {use!r}")
        preds = pd.read_parquet(pred_path, columns=["variant_id", "gene_symbol"])
        for vid, gene in zip(preds["variant_id"], preds["gene_symbol"]):
            entries.append({"variant_id": vid, "group_id": gene, "use": use,
                            "execution_id": eid,
                            "population_id": ex["population_id"], "time": now})

        per_execution[eid] = {"entries": len(entries) - n_before,
                              "predictions_rows": len(preds),
                              "predictions_use": use,
                              "population_id": ex["population_id"]}
        print(f"{eid}: +{len(entries)-n_before:,} entries "
              f"({use} on {len(preds):,} rows)")

    ledger = pd.DataFrame(entries)
    exposed_groups = set(ledger["group_id"])
    exposed_variants = set(ledger["variant_id"])
    universe_groups = set(mem_all["gene_symbol"].dropna())
    eligible_groups = set(mem["gene_symbol"].dropna())

    unexposed_all = universe_groups - exposed_groups
    unexposed_eligible = eligible_groups - exposed_groups

    print()
    print(f"ledger entries: {len(ledger):,}")
    print(ledger["use"].value_counts().to_string())
    print()
    print(f"exposed variants: {len(exposed_variants):,}")
    print(f"exposed groups:   {len(exposed_groups):,} / {len(universe_groups):,} universe")
    print(f"unexposed groups (any):              {len(unexposed_all):,}")
    print(f"unexposed groups WITH usable labels: {len(unexposed_eligible):,}")
    if not unexposed_eligible:
        print("  -> NO label-eligible gene in this snapshot is unexposed.")
        print("     Independent confirmation requires a population outside it.")

    summary = {
        "built_utc": now,
        "schema": ["variant_id", "group_id", "use", "execution_id", "population_id", "time"],
        "executions": per_execution,
        "n_entries": len(ledger),
        "by_use": {str(k): int(v) for k, v in ledger["use"].value_counts().items()},
        "n_exposed_variants": len(exposed_variants),
        "n_exposed_groups": len(exposed_groups),
        "n_universe_groups": len(universe_groups),
        "n_eligible_groups": len(eligible_groups),
        "n_unexposed_groups_any": len(unexposed_all),
        "n_unexposed_groups_with_usable_labels": len(unexposed_eligible),
        "known_gaps": config.get("known_gaps", []),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    ledger.to_parquet(output_dir / "exposure_ledger.parquet", index=False)
    (output_dir / "exposure_ledger_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("=== screen: a candidate drawn from the validation partition ===")
    val_gene = mem[mem["partition"].eq("validation")]["gene_symbol"].iloc[0]
    cand = mem[mem["gene_symbol"].eq(val_gene)].head(2)
    screened = confirmation_screen(
        [{"variant_id": v, "group_id": g} for v, g in zip(cand["variant_id"], cand["gene_symbol"])],
        ledger[["variant_id", "group_id", "use"]].to_dict("records"))
    for s in screened:
        print(f"  {s['variant_id']} (gene {s['group_id']}): "
              f"passes={s['passes_recorded_exposure_screen']} blockers={s['blockers']}")

    print()
    print(f"Wrote {output_dir}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--membership", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    build(args.membership, args.config, args.output_dir, args.root)


if __name__ == "__main__":
    main()
