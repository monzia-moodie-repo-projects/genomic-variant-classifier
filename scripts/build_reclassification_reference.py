#!/usr/bin/env python3
"""build_reclassification_reference.py -- Monzia Moodie

Reference for ReclassificationSentinelMonitorAgent: the committed split membership as a compact
(variant_id, split) parquet. Extracts variant_id from meta_{split}.parquet for each split present
(the column is 'variant_id' and the file is meta_{split}.parquet -- confirmed from
run_drift_monitor.run_label_drift). Splits whose meta file is absent or lacks the id column are
SKIPPED with a printed note -- never silently mislabeled (cf. the legacy run_drift_monitor bug that
assigns meta_test ids to 'training'). Per-split files can be overridden explicitly.

RUN AT Run-15/17 (needs the committed splits).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path("data/reference/reclassification/reclassification_reference.parquet")
SPLITS = ("train", "val", "test")


def build_reference(splits_dir=None, variant_id_col: str = "variant_id", *,
                    per_split_paths: dict | None = None):
    """Return (reference_df[variant_id, split], skipped[list[(split, reason)]]).

    per_split_paths overrides the default meta_{split}.parquet lookup for any split.
    """
    frames, skipped = [], []
    for split in SPLITS:
        if per_split_paths and split in per_split_paths:
            mp = Path(per_split_paths[split])
        elif splits_dir is not None:
            mp = Path(splits_dir) / f"meta_{split}.parquet"
        else:
            skipped.append((split, "no splits_dir and no per_split path")); continue
        if not mp.exists():
            skipped.append((split, f"{mp.name} not found")); continue
        df = pd.read_parquet(mp)
        if variant_id_col not in df.columns:
            skipped.append((split, f"'{variant_id_col}' absent (cols: {list(df.columns)[:8]})")); continue
        ids = df[variant_id_col].astype(str).drop_duplicates()
        frames.append(pd.DataFrame({"variant_id": ids.to_numpy(), "split": split}))
    if not frames:
        raise SystemExit(f"ABORT: no split ids extracted. Skipped: {skipped}")
    return pd.concat(frames, ignore_index=True), skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the reclassification split-membership reference.")
    ap.add_argument("--splits-dir", type=Path,
                    default=Path("outputs/run15_rerun_report/full/splits"),
                    help="Directory containing meta_{train,val,test}.parquet.")
    ap.add_argument("--variant-id-col", default="variant_id")
    ap.add_argument("--train-meta", type=Path, default=None, help="Override path for the train meta parquet.")
    ap.add_argument("--val-meta", type=Path, default=None)
    ap.add_argument("--test-meta", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    overrides = {k: v for k, v in (("train", a.train_meta), ("val", a.val_meta), ("test", a.test_meta)) if v}
    ref, skipped = build_reference(a.splits_dir, a.variant_id_col, per_split_paths=overrides or None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(a.out, index=False)
    counts = ref.groupby("split").size().to_dict()
    print(f"wrote {a.out}: {counts} ({len(ref)} ids total)")
    for split, reason in skipped:
        print(f"  [skip] {split}: {reason}")
    if "train" not in counts:
        print("  WARN: no 'train' ids -> flip_rate_training will be 0; point --train-meta at the training meta parquet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
